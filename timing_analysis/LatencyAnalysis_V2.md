# Single-Node Latency Analysis: Direct `env.step()` vs FilesystemProtocol IPC

## Setup

| Parameter | Value |
|-----------|-------|
| Platform | Linux (GPFS cluster, della-gpu) |
| Chrome | Headless, 1800x900, CDP fast screenshot |
| Warmup rounds | 2 (discarded) |
| Timed rounds | 10 |
| Settle time | 2s between steps (direct test only, outside timing) |
| Action type | JSON state change (position delta + rotation delta + zoom) |

**Test A (Direct):** Single process calls `env.step(action)` directly.
Measures the baseline: Selenium URL navigation + JS state read + CDP screenshot.

**Test B (IPC):** Two processes connected via `FilesystemProtocol`.
Server process owns Chrome and runs `NGLServer.process_actions()`.
Client process runs `NGLClient.send_actions()` (write action file, poll for observation file).
Measures the full round-trip: file write + busy-poll + env.step() + pickle serialization + file read.

## Results

### Direct `env.step()`

| Metric | Value |
|--------|-------|
| Mean | 132.07 ms |
| Std | 11.98 ms |
| Median | 137.40 ms |
| Min | 103.60 ms |
| Max | 144.50 ms |

### Communication IPC

| Metric | Value |
|--------|-------|
| Mean | 150.62 ms |
| Std | 11.02 ms |
| Median | 152.67 ms |
| Min | 136.90 ms |
| Max | 171.10 ms |

### Overhead

| Metric | Value |
|--------|-------|
| IPC overhead (mean) | +18.54 ms |
| Ratio | 1.14x |

## Analysis

### Latency Breakdown

Each direct `env.step()` call involves three Selenium operations:

1. `driver.get(new_url)` -- navigate Chrome to the new JSON-encoded Neuroglancer URL
2. `driver.execute_script(...)` -- read `viewer.state` from the page
3. `driver.execute_cdp_cmd("Page.captureScreenshot", ...)` -- capture a JPEG screenshot via CDP, then decode into a numpy array

These account for the ~132 ms baseline. The IPC path adds ~19 ms on top, which breaks down into:

| Source | Estimated Cost |
|--------|---------------|
| `msgpack.packb` (action serialization) + file write + `os.rename` | ~1-2 ms |
| Server-side busy-poll latency (`read_actions`) | ~1-3 ms |
| `pickle.dump` (observation serialization, numpy array) | ~3-5 ms |
| Client-side busy-poll latency (`read_observations`) | ~1-3 ms |
| `pickle.load` (observation deserialization) | ~3-5 ms |
| GPFS filesystem overhead (4 file operations per round-trip) | ~3-5 ms |

The observation payload is dominated by the screenshot (H×W×3 RGB numpy array from the ~1800×900 frame), making pickle serialization/deserialization the largest component of the IPC cost. The raw contiguous buffer of a numpy array pickles efficiently.

### Variance

Both tests show moderate variance (std ~11-12 ms). Chrome's rendering pipeline (`--headless=new` with Vulkan/ANGLE) has a somewhat non-deterministic first-paint time per navigation, which dominates the jitter. IPC variance (11.02 ms) is slightly *lower* than direct (11.98 ms) -- the filesystem round-trip is deterministic, so the IPC test inherits its jitter almost entirely from the Chrome step underneath it.

### Conclusion

The `FilesystemProtocol` introduces a **~19 ms (1.14x) overhead** per step relative to direct in-process calls. Since `env.step()` is dominated by Chrome rendering (~132 ms), the IPC layer accounts for only ~12% of total loop time -- confirming the multi-process architecture is a cheap way to separate the Chrome environment from the RL controller, a requirement for scaling to multiple agents or running the model on a different node.

# Cross-Node Latency Analysis

The same-node tests above run both processes on a single machine with local GPFS access. In a real training setup, the RL model (client) runs on a GPU compute node while the Neuroglancer environment (server) runs on a separate node. To measure the additional cost of cross-node GPFS I/O, we run two more settings where the client is submitted to a compute node via `sbatch` and the server remains on the login node (della-gpu).

## Setup

| Setting | Protocol | Processes | Topology |
|---------|----------|-----------|----------|
| Naive file IPC | Direct `pickle` + `os.remove`, no atomic rename | 2 | Cross-node |
| Communication IPC | `FilesystemProtocol` (msgpack/pickle + `os.rename`) | 2 | Cross-node |

- **Naive**: Client and server exchange data via raw `pickle.dump`/`pickle.load` on shared files, polling with `os.path.exists` and deleting after read. No atomicity guarantees -- read/write conflicts are caught by `try/except` and retried with a 1 ms sleep.
- **Communication**: Uses the full `FilesystemProtocol` with atomic `os.rename` for safe handoff. Client-side polling augmented with `time.sleep(0.001)` to handle cross-node GPFS visibility delays (the built-in busy-poll has no sleep and exhausts retries before the server can respond).

## Results

### Naive File IPC (cross-node)

Client on della-h12n8.

| Metric | Value |
|--------|-------|
| Mean | 161.41 ms |
| Std | 12.97 ms |
| Median | 161.10 ms |
| Min | 132.56 ms |
| Max | 178.72 ms |
| Read/write conflicts | 0 |

### Communication IPC (cross-node)

Client on della-h17n13.

| Metric | Value |
|--------|-------|
| Mean | 159.17 ms |
| Std | 10.18 ms |
| Median | 160.51 ms |
| Min | 140.01 ms |
| Max | 172.89 ms |

## Full Comparison

| # | Setting | Mean | Std | Overhead vs Direct |
|---|---------|------|-----|--------------------|
| 1 | Direct (1 process, same node) | 132.07 ms | 11.98 ms | -- (baseline) |
| 2 | Communication IPC (2 processes, same node) | 150.62 ms | 11.02 ms | +18.54 ms (1.14x) |
| 3 | Naive file IPC (2 processes, cross-node) | 161.41 ms | 12.97 ms | +29.34 ms (1.22x) |
| 4 | Communication IPC (2 processes, cross-node) | 159.17 ms | 10.18 ms | +27.10 ms (1.21x) |

## Analysis

**Cross-node GPFS cost (setting 2 vs 4):** Moving the Communication IPC from same-node to cross-node adds ~9 ms. This is the pure network + GPFS metadata propagation overhead -- the time for a file written on one node to become visible and readable on another.

**Communication vs Naive on cross-node (setting 3 vs 4):** Communication is effectively the same speed as Naive (159.17 ms vs 161.41 ms, a 2 ms gap that is within measurement noise, and in the opposite direction from what you might expect given the protocol's extra work). **The Communication protocol no longer adds meaningful latency over the raw-pickle approach, even on GPFS.**

This is a notable change from earlier runs (when observations were PIL Images and Communication was ~12 ms slower than Naive), and the reason is almost entirely the screenshot format change. The observation payload is dominated by the Neuroglancer frame; with the upstream refactor it is now a raw `numpy.ndarray` instead of a `PIL.Image`, and that shifts the per-step cost structure:

| Cost component | Old (PIL) | New (numpy) | Why it changed |
|---|---|---|---|
| `pickle.dump` (observation) | ~5-10 ms | ~3-5 ms | PIL's `__reduce__` goes through `tobytes()` + re-encodes mode/palette/info; numpy dumps the raw contiguous buffer plus a small header. |
| `pickle.load` (observation) | ~5-10 ms | ~3-5 ms | Same reason in reverse -- PIL rebuilds an `Image` object; numpy restores a buffer view. |
| `msgpack.packb` (action, 17 floats) | ~0.5 ms | ~0.5 ms | Unchanged. |
| `os.rename` vs `os.remove` (metadata op) | ~3-8 ms delta on GPFS | ~3-8 ms delta on GPFS | Unchanged. |

Previously the serialization itself was ~15-20 ms per round-trip, so the extra rename on top of that was clearly visible. Now serialization has shrunk to ~6-10 ms, which leaves the per-round-trip budget dominated by the Chrome step (~132 ms) and cross-node GPFS visibility (~10-15 ms). The `os.rename` vs `os.remove` delta gets drowned out by the variance of those larger terms -- under measurement noise, Communication and Naive are indistinguishable in mean latency.

A secondary factor: `NGLServer.process_actions()` was simplified in the upstream sync (fewer `print` calls per step, and `start_session` now waits for `viewer.isReady()` at startup so per-step calls don't pay for warm-up rendering). Some of the old Communication overhead was log I/O, not protocol work.

**Atomicity comes essentially for free now.** The Naive protocol only "wins" by skipping atomic rename and never retrying on torn reads. In our first Naive run we observed 1 read/write conflict in 10 rounds, adding ~30 ms of jitter to that round; the second run saw 0. Since Communication is now no slower in the mean, there is no performance reason to prefer Naive -- its lack of atomicity is pure risk with no payoff.

**Naive variance (12.97 ms) is now slightly higher than Communication (10.18 ms):** the opposite of the earlier result. This is consistent with the conflict behavior -- even without a caught conflict in run 2, the lack of atomic handoff leaves more room for GPFS cache-visibility jitter than the rename-based swap.

**Total overhead budget:** In the realistic cross-node Communication setting (#4), the ~27 ms overhead (1.21x) decomposes roughly as:

| Component | Estimated Cost |
|-----------|---------------|
| Serialization (msgpack + pickle + unpickle, numpy observation) | ~6-10 ms |
| GPFS cross-node file visibility (2 round-trips) | ~10-15 ms |
| Polling + `os.rename` metadata operations | ~3-5 ms |

The `env.step()` Chrome rendering (~132 ms) remains the dominant cost. The total per-step latency of ~159 ms supports ~6.3 steps/second, which is sufficient for RL training loops where model inference adds further per-step time.