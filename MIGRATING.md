# Migrating from 0.1 → 0.2

`ngllib` 0.2 is a **hard breaking release** — the public surface was rewritten
end-to-end to conform to the Gymnasium API and enable distributed RL via a new
transport / `RemoteEnv` / `serve` stack. No deprecation shim is provided.

This guide is a side-by-side cheat sheet of what changed and the smallest
edit needed for each piece.

## TL;DR

| Concept | 0.1 | 0.2 |
|---|---|---|
| Class | `ngllib.Environment` (custom shape) | `ngllib.Environment(gym.Env)` |
| Construction | mixed `__init__` + `start_session(**options)` | **all** kwargs in `__init__`; no `**options` |
| Lifecycle | `start_session()` / `end_session()` | standard Gym `reset()` (lazy launches browser) / `close()` |
| `reset()` return | `None` | `(obs, info)` |
| `step()` return | `(state, reward, done, json_state)` (4-tuple) | `(obs, reward, terminated, truncated, info)` (5-tuple) |
| Observation | `(pos_state_list, numpy_image)` tuple | `spaces.Dict({"position", "xs_scale", "orientation", "proj_scale", "image"})` |
| Action | 17-element flat list of floats | `spaces.Dict({"action_type", "mouse_xy", "modifiers", "delta_*"})` with `Discrete(4)` action_type |
| Reward | `reward_function(state, action, prev_state) -> (reward, done)` | `reward_factory(task_info) -> (obs, action, prev_obs, terminated) -> reward` |
| Termination | bundled in reward | separate `termination_factory(task_info) -> (obs, action, prev_obs) -> bool` |
| Start-state sampling | hardcoded URL | `reset_state_provider: StateProvider` constructor hook (state schema = `NglState` TypedDict) |
| Truncation | n/a (consumer-managed) | use Gymnasium `TimeLimit` wrapper or `gymnasium.make(..., max_episode_steps=...)` |
| IPC | `ngllib.utils.Communication.{SocketProtocol, NGLClient, NGLServer}` (removed) | `ngllib.distributed.transports.{SocketTransport, FilesystemTransport}` + `ngllib.RemoteEnv` + `ngllib.distributed.serve.serve` |
| Gym registration | none | `gymnasium.make("Neuroglancer-v0", ...)` auto-wraps `TimeLimit` |
| Errors | bare `Exception` / `ConnectionError` | typed hierarchy under `ngllib.errors` (re-raised across the wire) |

## Constructor + lifecycle

```python
# 0.1
env = Environment(headless=True, config_path="config.json", verbose=False,
                  reward_function=my_reward)
env.start_session(euler_angles=True, fast=True, left_pane=False, right_pane=True)
# ...
env.end_session()
```

```python
# 0.2
env = Environment(
    headless=True,
    orientation="euler",                 # was: start_session(euler_angles=True)
    screenshot_format="jpeg",            # was: start_session(fast=True)
    left_pane=False, right_pane=True,    # was: start_session(left_pane=..., right_pane=...)
    reward_factory=make_reward,          # see "Reward & termination" below
    # config_path defaults to the packaged config.json — no need to pass one
)
obs, info = env.reset(seed=0)            # lazy-launches the browser here
# ...
env.close()                              # was: env.end_session()
```

There is no `start_session()` / `end_session()` anymore. The browser launches
lazily on the first `reset()`; `close()` shuts it down.

## `step()` return shape

```python
# 0.1
state, reward, done, json_state = env.step(action_vec)
pos_state, image = state
```

```python
# 0.2
obs, reward, terminated, truncated, info = env.step(action_dict)
position    = obs["position"]      # np.float32 (3,)
image       = obs["image"]         # np.uint8 (H, W, 3)
xs_scale    = obs["xs_scale"]      # np.float32 (1,)
orientation = obs["orientation"]   # np.float32 (3,) or (4,) per the `orientation` kwarg
proj_scale  = obs["proj_scale"]    # np.float32 (1,)
# json_state moved into info:
json_state = info["json_state"]
```

`terminated` and `truncated` are the standard Gymnasium split:
- `terminated` — the env decided the episode is done (success / failure).
- `truncated` — the episode was cut short by something external (typically the
  `TimeLimit` wrapper hitting `max_episode_steps`).

The base `Environment.step()` always returns `truncated=False` — let `TimeLimit`
handle truncation (`gymnasium.make("Neuroglancer-v0", max_episode_steps=300)`
auto-wraps it).

## Action format

The 17-element flat list is replaced by a `Dict` with a single discrete
`action_type` head and parameter fields that are consumed depending on the
chosen type. Mutually-exclusive verbs (clicks vs. state edit) are encoded
directly in the action space — no `if/elif` priority.

```python
# 0.1: positional 17-vec
action = [
    0, 0, 0,      # left, right, double click booleans
    100, 100,     # x, y mouse position
    0, 0, 0,      # shift, ctrl, alt
    1,            # json_change flag
    10, 0, 0,    # delta position x, y, z
    0,            # delta crossSectionScale
    0.2, 0, 0,    # delta projection orientation (euler)
    2000,         # delta projection scale
]
```

```python
# 0.2: Dict
import numpy as np
action = {
    "action_type":      3,                                          # 0=left, 1=right, 2=double, 3=edit_state
    "mouse_xy":         np.array([100, 100], dtype=np.float32),     # consumed when action_type in {0,1,2}
    "modifiers":        np.array([0, 0, 0], dtype=np.int8),         # [shift, ctrl, alt]
    "delta_pos":        np.array([10, 0, 0], dtype=np.float32),     # consumed when action_type == 3
    "delta_xs_scale":   np.array([0], dtype=np.float32),
    "delta_orient":     np.array([0.2, 0, 0], dtype=np.float32),    # length 3 (euler) or 4 (quaternion)
    "delta_proj_scale": np.array([2000], dtype=np.float32),
}
```

The `Discrete(4)` `action_type` makes the four verbs mutually exclusive at the
space level — no more "what happens if both left_click=1 and json_change=1?"
ambiguity. The relevant parameter fields per type are documented in
`Environment._apply_actions`.

If your existing code emits the 17-vec flat list, write a small `ActionWrapper`
that decodes the flat list into the Dict; see `neurogym-agent/envs/action_translator.py`
for a reference example.

## Reward & termination

Both move from per-step callables to **factories** that bind once at `reset`
via the per-episode `task_info` dict. This lets heavy per-episode setup happen
once (KD-tree, target coords, tolerances, …) and keeps `task_info` out of the
per-step signature.

```python
# 0.1: single per-step callable
def my_reward(state, action, prev_state):
    z, z_prev = state[0][0][2], prev_state[0][0][2]
    reward = 0.001 * (z - z_prev)
    done = abs(z - TARGET_Z) <= TOLERANCE
    return reward, done

env = Environment(reward_function=my_reward, ...)
```

```python
# 0.2: factories — reward and termination separated
def make_termination(task_info):
    target_z = task_info["target_z"]
    tol      = task_info["tolerance"]
    def terminated(obs, action, prev_obs):
        return abs(obs["position"][2] - target_z) <= tol
    return terminated

def make_reward(task_info):
    target_z = task_info["target_z"]
    def reward(obs, action, prev_obs, terminated):
        if terminated:
            return 1.0
        return 0.001 * (obs["position"][2] - prev_obs["position"][2])
    return reward

env = Environment(
    reset_state_provider=MyProvider(...),  # populates task_info per episode
    reward_factory=make_reward,
    termination_factory=make_termination,
    ...,
)
```

`task_info` is whatever your `reset_state_provider` returned from its `__call__`
on this episode. The library treats it as opaque — its shape is a contract
between your provider and your factories. The recommended pattern is to keep
`task_info` small (`{"segment_id": ..., "target_z": ...}`) and have the factories
close over a shared data table (e.g. DuckDB over a parquet) loaded once.

If you don't need any of this, omit both factories — the defaults give
`reward = 0.0` and `terminated = False` forever (you'll want a `TimeLimit`
wrapper to actually end episodes).

## Reset state + provider hook

```python
# 0.1: reset takes only a URL
env.reset(url="https://neuroglancer-demo.appspot.com/#!...")
```

```python
# 0.2: three forms
env.reset(seed=42)                                          # provider drives normally
env.reset(options={"state": ng_state_dict})                 # state override, provider auto-derives task_info
env.reset(options={"state": ng_state_dict, "task_info": …}) # full override, provider not consulted
```

A `StateProvider` implements two methods (it's a `typing.Protocol` — duck-typed,
no inheritance needed):

```python
class MyProvider:
    def __call__(self, rng, options):
        seg_id = self._sample(rng)
        start  = self._build_state(seg_id)
        return start, {"segment_id": seg_id, "target_z": self._target(seg_id)}

    def task_info_from_state(self, state):
        seg_id = state["segments"][0]
        return {"segment_id": seg_id, "target_z": self._target(seg_id)}
```

## IPC (distributed RL)

The 0.1 `ngllib.utils.Communication` module is **removed**. The 0.2 stack:

- `ngllib.distributed.transports.{SocketTransport, FilesystemTransport}` —
  dumb byte pipes (3 methods: `send` / `recv` / `close`). New mediums slot in
  as duck-typed `Transport` implementations.
- `ngllib.distributed.serve.serve(env, transport)` — server runner (also
  exposed as `python -m ngllib.distributed.serve socket|filesystem ...`).
- `ngllib.RemoteEnv(transport)` — client-side `gym.Env` that proxies all calls
  over the transport. Indistinguishable from a local `Environment` to the agent
  (same `observation_space` / `action_space`, same Gym API). Re-exported at the
  top level for convenience; also importable as
  `ngllib.distributed.remote.RemoteEnv`.

```python
# 0.1
from ngllib.utils.Communication import SocketProtocol, NGLClient, NGLServer
proto = SocketProtocol(host=..., port=..., is_server=False, timeout=30)
client = NGLClient(protocol=proto)
state, reward, done, info = client.send_actions(action_vec)
```

```python
# 0.2: server side
from ngllib import Environment
from ngllib.distributed.serve import serve
from ngllib.distributed.transports import SocketTransport
serve(Environment(headless=True, ...), SocketTransport.server(host="0.0.0.0", port=5555))

# 0.2: client side
from ngllib import RemoteEnv
from ngllib.distributed.transports import SocketTransport
env = RemoteEnv(SocketTransport.client(host="renderhost", port=5555))
obs, reward, terminated, truncated, info = env.step(action_dict)
```

Switching to filesystem is one constructor argument:
`FilesystemTransport.server(action_dir=..., obs_dir=...)`
and `FilesystemTransport.client(...)` on the other end.

## Errors

```python
# 0.1: bare exception types
try:
    env.step(...)
except ConnectionError:
    ...
```

```python
# 0.2: typed hierarchy under ngllib.errors
from ngllib import BrowserError, ProviderError, ConnectionLost, NgllibError
try:
    env.step(...)
except BrowserError:
    ...  # Chromium hung / crashed / launch failed
except NgllibError:
    ...  # base class catches anything our lib raises
```

When using `RemoteEnv`, the same exception class raised by the server is
re-raised on the client — `except BrowserError` works whether the env is
local or remote.

## Gym registration

```python
# 0.2: standard Gymnasium make()
import gymnasium as gym
env = gym.make("Neuroglancer-v0", headless=True)   # auto-wraps with TimeLimit(max_episode_steps=300)
```

All `Environment` constructor kwargs can be passed through `gym.make(...)`.
