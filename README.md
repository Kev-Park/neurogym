# Neurogym

## 1. Chrome & ChromeDriver Installation

Install Chrome for Testing and the matching ChromeDriver.

```bash
mkdir -p chrome-install && cd chrome-install

# Check latest stable version
curl -s https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions.json | python3 -m json.tool | head -10

# Download Chrome + ChromeDriver
VERSION="147.0.7727.50"
curl -O "https://storage.googleapis.com/chrome-for-testing-public/${VERSION}/linux64/chrome-linux64.zip"
curl -O "https://storage.googleapis.com/chrome-for-testing-public/${VERSION}/linux64/chromedriver-linux64.zip"


unzip chrome-linux64.zip && unzip chromedriver-linux64.zip
chmod +x chrome-linux64/chrome chromedriver-linux64/chromedriver
```

Then update `config.json` with the correct paths:

```json
{
    "driver_path_linux": "/path/to/chrome-install/chromedriver-linux64/chromedriver",
    "chrome_binary_path_linux": "/path/to/chrome-install/chrome-linux64/chrome"
}
```

## 2. Screenshot Rendering

At first, the screenshots come out blank because of rendering issues. I configure Chrome to use SwiftShader to provide complete WebGL support without a physical GPU display:

```
--headless=new
--use-gl=angle
--use-angle=vulkan
--enable-features=Vulkan
--enable-unsafe-swiftshader
```

These flags are set in `environment.py` → `initialize_chrome_options()` when `headless=True`.

## 3. IPC (Inter-Process Communication)

When running multiple agents or separating the environment process (Chrome) from the controller process (RL model), simultaneous file reads/writes cause data corruption:


Added `ngllib/ipc.py` with an `IPCChannel` class that provides:

- **Atomic writes**: write to a temp file, then `os.rename()` to the target path (rename is atomic on the same filesystem), so readers never see partial data
- **File locks**: shared locks (`fcntl.flock`) on reads to prevent conflicts
- **Signal files**: empty `.flag` files for "ready" notifications between processes

### Architecture

```
Process A (Environment)                Process B (Controller)
───────────────────────                ───────────────────────
Start Chrome + Neuroglancer
Write initial state -> files
Signal: state_ready
                                       Wait for state_ready
                                       Read state + image
                                       Choose action
                                       Write action -> file
                                       Signal: action_ready
Wait for action_ready
Read action
Execute step(), write new state
Signal: state_ready
                                       Read new state
                                       Choose next action
                                       ...
Exit after N steps                     Exit after N steps
```

### Files modified

- `ngllib/ipc.py` — new IPC module
- `ngllib/__init__.py` — added `IPCChannel` export
- `ngllib/environment.py` — added `ipc_dir` parameter to `__init__()`, auto-write state in `step()`, new methods `step_ipc()` and `run_ipc_loop()`

### Usage

See `demo.py` and `demo_ipc.py`
