"""Gymnasium environment over a Neuroglancer viewer, with a pluggable renderer.

`Environment` owns the whole env contract -- spaces, task hooks, episode
bookkeeping, the viewer state and its transitions, reset/step -- and delegates
pixels (and, for Chrome, the live copy of the state) to a `Renderer`. The two
shipped renderers are `ngllib.chrome.ChromeRenderer` (Playwright + Chromium,
the deployment target and therefore normative for state and action semantics)
and `ngllib.simulator.SimulatorRenderer` (CloudVolume + moderngl/EGL).

    env = Environment()                                   # Chrome, today's defaults
    env = Environment(backend=SimulatorRenderer(), orientation="euler")
    env = gym.make("Neuroglancer-v0", backend=SimulatorRenderer())
"""

from __future__ import annotations

import copy
import logging
import os
import time
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image

from . import state as S
from .errors import ProviderError, RendererError
from .events import EventLog
from .providers import NglState, StateProvider
from .renderer import Renderer

logger = logging.getLogger(__name__)

# Public type aliases for the factory signatures.
RewardFactory = Callable[[dict[str, Any]], Callable[..., float]]
TerminationFactory = Callable[[dict[str, Any]], Callable[..., bool]]


def _noop_reward_factory(task_info: dict[str, Any]):
    return lambda obs, action, prev_obs, terminated: 0.0


def _noop_termination_factory(task_info: dict[str, Any]):
    return lambda obs, action, prev_obs: False


def mask_ui_enabled() -> bool:
    """NGL_MASK_UI=0 turns the UI mask off for both backends at once."""
    return os.environ.get("NGL_MASK_UI", "1") != "0"


class Environment(gym.Env):
    """RL environment driving a Neuroglancer viewer through a `Renderer`.

    The renderer is opened lazily on the first `reset()`; construction only
    builds spaces.

    `reset_ahead` hides the reset tail by preparing the NEXT episode during the
    current one: the environment draws the next state from the provider and
    hands it to `renderer.warm()`, an advisory hint the renderer may act on
    (Chrome pre-navigates a second browser context; the simulator prefetches
    the mesh and tiles). Only provider-driven resets take part, so an
    explicit-state reset (eval) never consumes provider draws.
    `reset_ahead_after_steps` says how far into an episode to warm; `None`
    defers to the renderer's own preference.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        backend: Renderer | None = None,
        orientation: S.Orientation = "quaternion",
        reset_state_provider: StateProvider | None = None,
        reward_factory: RewardFactory | None = None,
        termination_factory: TerminationFactory | None = None,
        reset_ahead: bool = False,
        reset_ahead_after_steps: int | None = None,
        mask_ui: bool | None = None,
        verbose: bool = False,
    ):
        super().__init__()
        if orientation not in ("quaternion", "euler"):
            raise ValueError(
                f"`orientation` must be 'quaternion' or 'euler'; got {orientation!r}")
        if backend is None:
            from .chrome import ChromeRenderer

            backend = ChromeRenderer()
        if not isinstance(backend, Renderer):
            raise TypeError(
                f"`backend` must implement ngllib.renderer.Renderer; got {type(backend).__name__}")

        self._renderer: Renderer = backend
        self.orientation = orientation
        self.verbose = verbose
        self.reset_ahead = reset_ahead
        self._warm_after = (backend.warm_after_steps if reset_ahead_after_steps is None
                            else int(reset_ahead_after_steps))
        # Blank the regions where Chrome draws UI (toolbar, scale bar, pane
        # buttons) in BOTH backends' frames, before any resize, so a policy
        # cannot tell the two apart by chrome rather than by data. Measured
        # only on the calibrated two-pane capture, so applied only there.
        self.mask_ui = mask_ui_enabled() if mask_ui is None else bool(mask_ui)

        self._reset_state_provider = reset_state_provider
        self._reward_factory = reward_factory or _noop_reward_factory
        self._termination_factory = termination_factory or _noop_termination_factory

        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()

        self._events = EventLog()
        self._renderer.events = self._events
        self._opened = False

        # Episode state
        self._rng: np.random.Generator = np.random.default_rng()
        self._episode_count = 0
        self._state: dict[str, Any] | None = None
        self._prev_obs: dict[str, Any] | None = None
        self._task_info: dict[str, Any] = {}
        self._reward_fn: Callable | None = None
        self._terminated_fn: Callable | None = None
        self._next: tuple[dict[str, Any], dict[str, Any]] | None = None
        self._provider_episode = False
        self._ep_steps = 0
        self._ep_terminated = False
        self._last_step_glitched = False
        self._ep_step_ms_sum = 0.0
        self._ep_step_ms_max = 0.0
        self._ep_slow_steps = 0

    # =========================================================================
    # Gymnasium public API
    # =========================================================================

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}
        # Ending-episode stats (for reset-reason attribution) BEFORE the bump.
        prev_steps, prev_term = self._ep_steps, self._ep_terminated
        prev_glitched, first = self._last_step_glitched, self._episode_count == 0
        self._episode_count += 1
        self._events.episode = self._episode_count
        t0 = time.monotonic()

        if not self._opened:
            self._renderer.open()
            self._opened = True

        # A seed reseeded the stream the pre-drawn state came from, and
        # options override the reset (eval): both fall back to the inline draw.
        provider_driven = seed is None and not options
        if provider_driven and self._next is not None:
            start_state, task_info = self._next
            warm_used = True
        else:
            start_state, task_info = self._resolve_reset_state(options)
            warm_used = False
        self._next = None
        self._provider_episode = provider_driven and self._reset_state_provider is not None

        try:
            self._reward_fn = self._reward_factory(task_info)
            self._terminated_fn = self._termination_factory(task_info)
        except Exception as e:
            raise ProviderError(f"factory raised at reset: {e}") from e
        self._task_info = task_info

        if start_state is None:
            start_state = self._renderer.default_state()
        if isinstance(start_state, dict):
            start_state = S.coerce(start_state)
        elif not isinstance(start_state, str):
            raise ProviderError(
                "start_state must be None, a URL str, or an NglState dict; "
                f"got {type(start_state).__name__}")
        if self.verbose:
            logger.info("reset -> %s", start_state if isinstance(start_state, str)
                        else {k: start_state[k] for k in ("position", "projectionScale")})

        tn = time.monotonic()
        self._renderer.reset_to(start_state)
        navigate_ms = (time.monotonic() - tn) * 1000.0
        tg = time.monotonic()
        st, image = self._renderer.observe()
        gather_ms = (time.monotonic() - tg) * 1000.0
        obs = self._make_obs(st, image)
        self._state = st
        self._prev_obs = obs

        self._events.emit(
            "reset", total_ms=(time.monotonic() - t0) * 1000.0,
            navigate_ms=navigate_ms, gather_ms=gather_ms, first=first, warm=warm_used,
            prev_steps=prev_steps, prev_terminated=prev_term, prev_glitched=prev_glitched,
            prev_step_ms_mean=(self._ep_step_ms_sum / prev_steps if prev_steps else 0.0),
            prev_step_ms_max=self._ep_step_ms_max, prev_slow_steps=self._ep_slow_steps,
            segment=(task_info.get("segment_id") if isinstance(task_info, dict) else None),
        )
        self._ep_steps = 0
        self._ep_terminated = False
        self._last_step_glitched = False
        self._ep_step_ms_sum = 0.0
        self._ep_step_ms_max = 0.0
        self._ep_slow_steps = 0

        if self._warm_after <= 0:
            self._maybe_warm_next()
        return obs, {"task_info": task_info, "json_state": copy.deepcopy(st), "step": 0}

    def step(self, action):
        t_step = time.monotonic()
        action_type = int(action["action_type"])
        try:
            if action_type in S.CLICK_KINDS:
                x, y = (float(v) for v in action["mouse_xy"])
                mods = S.modifiers_to_str(action["modifiers"])
                if self.verbose:
                    logger.info("%s at (%.1f, %.1f) modifiers=%r",
                                S.CLICK_KINDS[action_type], x, y, mods)
                self._renderer.click(S.CLICK_KINDS[action_type], x, y, mods)
            elif action_type == 3:
                new_state = S.apply_state_edit(self._state, action, self.orientation)
                if self.verbose:
                    logger.info("state edit -> pos=%s", new_state["position"])
                self._renderer.set_state(new_state)
            else:
                raise ValueError(f"action_type must be 0, 1, 2, or 3; got {action_type}")
            st, image = self._renderer.observe()
        except RendererError:
            # The renderer has already done its own recovery bookkeeping; the
            # episode is truncated by whoever wraps us (ResilientStepWrapper).
            self._last_step_glitched = True
            raise
        obs = self._make_obs(st, image)

        # Termination runs first so the reward fn can read `terminated` for terminal bonuses.
        try:
            terminated = bool(self._terminated_fn(obs, action, self._prev_obs))
        except Exception as e:
            raise ProviderError(f"termination_function raised: {e}") from e
        try:
            reward = float(self._reward_fn(obs, action, self._prev_obs, terminated))
        except Exception as e:
            raise ProviderError(f"reward_function raised: {e}") from e

        truncated = False  # TimeLimit wrapper handles step-count truncation.
        info = {"task_info": self._task_info, "json_state": copy.deepcopy(st)}

        self._state = st
        self._prev_obs = obs
        self._ep_steps += 1
        self._ep_terminated = terminated
        if self._ep_steps >= self._warm_after:
            self._maybe_warm_next()
        dur_ms = (time.monotonic() - t_step) * 1000.0
        self._ep_step_ms_sum += dur_ms
        self._ep_step_ms_max = max(self._ep_step_ms_max, dur_ms)
        if dur_ms > 2000.0:
            self._ep_slow_steps += 1
            if dur_ms > 5000.0:
                self._events.emit("slow_step", ms=dur_ms, step_idx=self._ep_steps)
        return obs, reward, terminated, truncated, info

    def close(self):
        """Tear down the renderer. Idempotent."""
        self._next = None
        self._renderer.close()
        self._opened = False

    # =========================================================================
    # Internal: spaces
    # =========================================================================

    @property
    def renderer(self) -> Renderer:
        return self._renderer

    def _build_observation_space(self) -> spaces.Dict:
        orient_dim = 3 if self.orientation == "euler" else 4
        return spaces.Dict(
            {
                "position": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "xs_scale": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                "orientation": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(orient_dim,), dtype=np.float32),
                "proj_scale": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                "image": spaces.Box(
                    low=0, high=255, shape=self._renderer.layout.image_shape, dtype=np.uint8),
            }
        )

    def _build_action_space(self) -> spaces.Dict:
        W, H = self._renderer.layout.window_size
        orient_dim = 3 if self.orientation == "euler" else 4
        return spaces.Dict(
            {
                "action_type": spaces.Discrete(4),  # 0=left, 1=right, 2=double, 3=edit_state
                "mouse_xy": spaces.Box(
                    low=np.array([0, 0], dtype=np.float32),
                    high=np.array([W, H], dtype=np.float32),
                    dtype=np.float32,
                ),
                "modifiers": spaces.MultiBinary(3),  # [shift, ctrl, alt]
                "delta_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "delta_xs_scale": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "delta_orient": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(orient_dim,), dtype=np.float32),
                "delta_proj_scale": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            }
        )

    # =========================================================================
    # Internal: reset state resolution + reset-ahead
    # =========================================================================

    def _resolve_reset_state(
        self, options: dict[str, Any]
    ) -> tuple[NglState | str | None, dict[str, Any]]:
        """Implement the three reset-override forms."""
        if "state" in options:
            start_state = options["state"]
            if "task_info" in options:
                task_info = options["task_info"]
            elif self._reset_state_provider is not None:
                try:
                    task_info = self._reset_state_provider.task_info_from_state(start_state)
                except Exception as e:
                    raise ProviderError(f"provider.task_info_from_state raised: {e}") from e
            else:
                task_info = {}
            return start_state, task_info

        if self._reset_state_provider is not None:
            try:
                start_state, task_info = self._reset_state_provider(self._rng, options)
            except Exception as e:
                raise ProviderError(f"reset_state_provider raised: {e}") from e
            return start_state, task_info

        # No provider, no override: the renderer's default start state.
        return None, {}

    def _maybe_warm_next(self) -> None:
        """Draw the next episode's state and hint the renderer -- once per
        provider-driven episode. The draw comes from the same rng stream at
        the point it would be taken inline, so sampling is unchanged."""
        if not (self.reset_ahead and self._provider_episode) or self._next is not None:
            return
        try:
            state, task_info = self._reset_state_provider(self._rng, {})
        except Exception as e:
            logger.warning("reset-ahead pre-sample failed (%s)", e)
            return
        self._next = (state, task_info)
        if isinstance(state, dict):
            try:
                self._renderer.warm(S.coerce(state))
            except Exception as e:  # advisory: a failed warm just costs the cold path
                logger.warning("reset-ahead warm failed (%s)", e)

    # =========================================================================
    # Internal: observation
    # =========================================================================

    def _make_obs(self, st: dict[str, Any], image: np.ndarray) -> dict[str, Any]:
        if self.mask_ui and image.ndim == 3 and image.shape[:2] == (450, 900):
            from .simulator.pane2d import mask_ui

            image = mask_ui(image)
        image_size = self._renderer.layout.image_size
        if image_size is not None and image.shape[:2] != (image_size[1], image_size[0]):
            image = np.asarray(Image.fromarray(image).resize(image_size))
        return {
            "position": np.asarray(st["position"], dtype=np.float32),
            "xs_scale": np.asarray([st["crossSectionScale"]], dtype=np.float32),
            "orientation": S.orientation_obs(st, self.orientation),
            "proj_scale": np.asarray([st["projectionScale"]], dtype=np.float32),
            "image": image,
        }
