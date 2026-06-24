# ngllib

`ngllib` is a [Gymnasium](https://gymnasium.farama.org/)-compliant RL environment
that drives [Neuroglancer](https://github.com/google/neuroglancer) (a web-based
3D connectomics viewer) through [Playwright](https://playwright.dev/python/)
browser automation. Every `step` dispatches a mouse / keyboard / viewer-state
action to a live (optionally headless) Neuroglancer session and returns the
resulting viewer state plus a screenshot as a structured Dict observation.

Same `Environment` class works in two modes:

- **Direct** (single process): standard `gym.Env`; the browser lives in your
  Python process.
- **Distributed** (two processes / two nodes): the browser-owning `Environment`
  runs behind `ngllib.distributed.serve.serve(env, transport)` on a renderer
  node; the learner / policy uses `ngllib.RemoteEnv(transport)` and writes
  identical RL code — only the constructor differs.

> **Migrating from 0.1?** See [`MIGRATING.md`](MIGRATING.md) — 0.2 is a hard
> breaking release that conforms the public API to Gymnasium.

## Installation

```bash
pip install ngllib
playwright install chromium
```

The second step is **required** — Playwright downloads its own Chromium (~450 MB)
to `~/.cache/ms-playwright/`; `ngllib` will not launch a browser until it has
been run. To install from source (editable):

```bash
git clone https://github.com/Kev-Park/neurogym.git
cd neurogym
pip install -e .
playwright install chromium
```

On PEP 668 / externally-managed clusters add `--break-system-packages` to the
`pip` call. To redirect the Chromium download to scratch storage:

```bash
PLAYWRIGHT_BROWSERS_PATH=/path/to/scratch playwright install chromium
```

## Quickstart — direct mode

```python
import numpy as np
from ngllib import Environment

env = Environment(headless=True, orientation="euler")
obs, info = env.reset(seed=0)

# Edit-state action: pan +10 in X, rotate 0.2 rad around X, zoom out.
action = {
    "action_type":      3,                                          # 0=left, 1=right, 2=double, 3=edit_state
    "mouse_xy":         np.array([100, 100], dtype=np.float32),
    "modifiers":        np.array([0, 0, 0], dtype=np.int8),
    "delta_pos":        np.array([10, 0, 0], dtype=np.float32),
    "delta_xs_scale":   np.array([0], dtype=np.float32),
    "delta_orient":     np.array([0.2, 0, 0], dtype=np.float32),
    "delta_proj_scale": np.array([2000], dtype=np.float32),
}
for _ in range(5):
    obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

Or via `gymnasium.make` (auto-wraps with `TimeLimit`):

```python
import gymnasium as gym
env = gym.make("Neuroglancer-v0", headless=True, max_episode_steps=300)
```

See [`main.py`](main.py) for a fuller example that saves screenshots.

## Distributed mode (separate renderer + learner nodes)

**Server (renderer node)** — owns the browser, listens for actions:

```bash
python -m ngllib.distributed.serve socket --host 0.0.0.0 --port 5555 --orientation euler
```

or programmatically:

```python
from ngllib import Environment
from ngllib.distributed.serve import serve
from ngllib.distributed.transports import SocketTransport

env = Environment(headless=True, orientation="euler")
serve(env, SocketTransport.server(host="0.0.0.0", port=5555))
```

**Client (learner / agent node)** — speaks the same Gym API:

```python
from ngllib import RemoteEnv
from ngllib.distributed.transports import SocketTransport

env = RemoteEnv(SocketTransport.client(host="renderhost.example.com", port=5555))
obs, info = env.reset(seed=0)
# ...identical step loop as direct mode...
env.close()
```

`FilesystemTransport.{server,client}(action_dir=..., obs_dir=...)` is the
file-swap alternative for environments where a TCP socket is awkward (firewalls,
batch schedulers); see the notebooks in [`demos/`](demos/) for runnable
client/server pairs over both transports.

## Configuration

`Environment` reads a tiny JSON config (default URL, credentials slots) that's
packaged inside the wheel. Most settings are explicit constructor kwargs, not
config-file fields — `headless`, `renderer`, `window_size`, `image_size`,
`orientation`, `left_pane` / `right_pane`, `retry_on_reset`,
`browser_restart_every`, the task hooks (`reset_state_provider`,
`reward_factory`, `termination_factory`), etc. See the `Environment.__init__`
signature for the full list.

To override the deployment defaults (URL, credentials) supply your own file:

```python
env = Environment(headless=True, config_path="my_config.json")
```

The `google_email_address` / `google_password` fields are blank in the shipped
default and only used when you actually need a logged-in Neuroglancer session.
**Do not commit real credentials** — keep them in your own local `config.json`.

## Headless rendering

When `headless=True`, `Environment._build_launch_args()` selects a GL backend
per OS (ANGLE + D3D11 on Windows, Metal on macOS, Vulkan on Linux), falling
back to SwiftShader software rendering when constructed with `renderer="cpu"`.
This is what keeps headless screenshots from coming out blank without a
physical GPU display.

## Errors

Typed hierarchy under `ngllib.errors`:

- `NgllibError` — base class for everything `ngllib` raises.
- `BrowserError` — Chromium hung / crashed / launch failed.
- `ProviderError` — `reset_state_provider`, `reward_factory`, or
  `termination_factory` raised.
- `TransportError` (with subclasses `ConnectionLost`, `HandshakeFailed`) —
  transport-layer failures.

In distributed mode the same exception class is re-raised on the client —
`try / except BrowserError` catches the same conditions whether the env is
local or remote.

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
