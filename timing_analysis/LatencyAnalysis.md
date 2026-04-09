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

TODO