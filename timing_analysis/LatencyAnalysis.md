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
| Mean | 75.05 ms |
| Std | 6.32 ms |
| Median | 72.52 ms |
| Min | 69.55 ms |
| Max | 91.89 ms |

### Communication IPC

| Metric | Value |
|--------|-------|
| Mean | 104.81 ms |
| Std | 9.42 ms |
| Median | 104.71 ms |
| Min | 89.59 ms |
| Max | 122.16 ms |

### Overhead

| Metric | Value |
|--------|-------|
| IPC overhead (mean) | +29.76 ms |
| Ratio | 1.40x |

## Analysis

### Latency Breakdown

Each direct `env.step()` call involves three Selenium operations:

1. `driver.get(new_url)` -- navigate Chrome to the new JSON-encoded Neuroglancer URL
2. `driver.execute_script(...)` -- read `viewer.state` from the page
3. `driver.execute_cdp_cmd("Page.captureScreenshot", ...)` -- capture a JPEG screenshot via CDP

These account for the ~75 ms baseline. The IPC path adds ~30 ms on top, which breaks down into:

| Source | Estimated Cost |
|--------|---------------|
| `msgpack.packb` (action serialization) + file write + `os.rename` | ~1-2 ms |
| Server-side busy-poll latency (`read_actions`) | ~1-5 ms |
| `pickle.dump` (observation serialization, includes PIL Image) | ~5-10 ms |
| Client-side busy-poll latency (`read_observations`) | ~1-5 ms |
| `pickle.load` (observation deserialization) | ~5-10 ms |
| GPFS filesystem overhead (4 file operations per round-trip) | ~5-10 ms |

The observation payload is significantly larger than the action payload because it contains a full PIL Image (~1800x900 JPEG decoded into memory), making pickle serialization/deserialization the dominant IPC cost.

### Variance

Both tests show low variance (std ~6-9 ms), confirming that the measurements are stable. The slightly higher variance in the IPC test is expected due to GPFS I/O jitter and busy-poll scheduling non-determinism.

### Conclusion

The `FilesystemProtocol` introduces a **~30 ms (1.4x) overhead** per step relative to direct in-process calls. Given that a single `env.step()` is dominated by Chrome rendering (~75 ms), the IPC cost is a minor fraction of the total loop time. This overhead is acceptable for the multi-process architecture, which enables separating the Chrome environment from the RL controller -- a requirement for scaling to multiple agents or running the model on a different node.

# Cross-Node Latency Analysis

The same-node tests above run both processes on a single machine with local GPFS access. In a real training setup, the RL model (client) runs on a GPU compute node while the Neuroglancer environment (server) runs on a separate node. To measure the additional cost of cross-node GPFS I/O, we run two more settings where the client is submitted to a compute node via `sbatch` and the server remains on the login node.

## Setup

| Setting | Protocol | Processes | Topology |
|---------|----------|-----------|----------|
| Naive file IPC | Direct `pickle` + `os.remove`, no atomic rename | 2 | Cross-node |
| Communication IPC | `FilesystemProtocol` (msgpack/pickle + `os.rename`) | 2 | Cross-node |

- **Naive**: Client and server exchange data via raw `pickle.dump`/`pickle.load` on shared files, polling with `os.path.exists` and deleting after read. No atomicity guarantees -- read/write conflicts are caught by `try/except` and retried with a 1 ms sleep.
- **Communication**: Uses the full `FilesystemProtocol` with atomic `os.rename` for safe handoff. Client-side polling augmented with `time.sleep(0.001)` to handle cross-node GPFS visibility delays (the built-in busy-poll has no sleep and exhausts retries before the server can respond).

## Results

### Naive File IPC (cross-node)

| Metric | Value |
|--------|-------|
| Mean | 118.79 ms |
| Std | 5.49 ms |
| Median | 119.04 ms |
| Min | 107.29 ms |
| Max | 127.04 ms |

### Communication IPC (cross-node)

| Metric | Value |
|--------|-------|
| Mean | 131.18 ms |
| Std | 9.42 ms |
| Median | 133.76 ms |
| Min | 110.98 ms |
| Max | 143.15 ms |

## Full Comparison

| # | Setting | Mean | Std | Overhead vs Direct |
|---|---------|------|-----|--------------------|
| 1 | Direct (1 process, same node) | 75.05 ms | 6.32 ms | -- (baseline) |
| 2 | Naive file IPC (2 processes, cross-node) | 118.79 ms | 5.49 ms | +43.74 ms (1.58x) |
| 3 | Communication IPC (2 processes, same node) | 104.81 ms | 9.42 ms | +29.76 ms (1.40x) |
| 4 | Communication IPC (2 processes, cross-node) | 131.18 ms | 9.42 ms | +56.13 ms (1.75x) |

## Analysis

**Cross-node GPFS cost (setting 3 vs 4):** Moving the Communication IPC from same-node to cross-node adds ~26 ms. This is the pure network + GPFS metadata propagation overhead -- the time for a file written on one node to become visible and readable on another.

**Communication protocol cost (setting 2 vs 4):** Naive is ~12 ms faster than Communication on the same cross-node topology. The difference comes from the protocol's extra work per round-trip: msgpack encoding for actions, and two `os.rename` operations (atomic swap pattern) instead of a single `os.remove`. On GPFS, rename is a heavier metadata operation than remove.

**Naive variance:** Naive shows the lowest std (5.49 ms) despite lacking atomicity, because in a sequential two-party exchange there is no true concurrent access -- the client only reads after the server finishes writing, and vice versa. Read/write conflicts would appear under higher concurrency or GPFS cache inconsistency, but were not observed in this 10-round test.

**Total overhead budget:** In the realistic cross-node Communication setting (#4), the ~56 ms overhead (1.75x) decomposes roughly as:

| Component | Estimated Cost |
|-----------|---------------|
| Serialization (msgpack + pickle + depickle) | ~15-20 ms |
| GPFS cross-node file visibility (2 round-trips) | ~25-30 ms |
| Polling + `os.rename` metadata operations | ~5-10 ms |

The env.step() Chrome rendering (~75 ms) remains the dominant cost. The total per-step latency of ~131 ms supports ~7.6 steps/second, which is sufficient for RL training loops where model inference adds further per-step time.
