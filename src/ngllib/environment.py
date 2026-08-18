"""Gymnasium-compliant RL environment driving Neuroglancer via Playwright."""

from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import logging
import os
import platform
import sys
import threading
import time
import urllib.parse
from typing import Any, Callable, Literal

import gymnasium as gym
import numpy as np
import psutil
from gymnasium import spaces
from PIL import Image
from playwright.sync_api import sync_playwright

from .errors import BrowserError, ProviderError
from .providers import NglState, StateProvider
from .utils.geom import euler_to_quaternion, quaternion_to_euler
from .utils.MouseActionHandler import MouseActionHandler

logger = logging.getLogger(__name__)

# Public type aliases for the factory signatures.
RewardFactory = Callable[[dict[str, Any]], Callable[..., float]]
TerminationFactory = Callable[[dict[str, Any]], Callable[..., bool]]


class _BrowserWatchdog:
    """Kills the browser process if a Playwright call hangs past `timeout_s`.

    Playwright's sync API is thread-affine, so a hung call can't be cancelled
    in-thread; killing Chrome from a timer thread makes the blocked call raise
    immediately (legacy-proven). No-op when `timeout_s` is falsy.
    """

    def __init__(self, timeout_s: float | None, kill_fn: Callable[[], None]):
        self.fired = False
        self._timer: threading.Timer | None = None
        if timeout_s:
            def _fire():
                self.fired = True
                kill_fn()

            self._timer = threading.Timer(timeout_s, _fire)
            self._timer.daemon = True
            self._timer.start()

    def cancel(self) -> None:
        if self._timer is not None:
            self._timer.cancel()


def _noop_reward_factory(task_info: dict[str, Any]):
    return lambda obs, action, prev_obs, terminated: 0.0


def _noop_termination_factory(task_info: dict[str, Any]):
    return lambda obs, action, prev_obs: False


class Environment(gym.Env):
    """RL environment that drives a headless Neuroglancer viewer.

    Browser launches lazily on the first `reset()`; `__init__` only builds
    spaces and loads config.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        # --- Browser & rendering -------------------------------------------------
        headless: bool = True,
        renderer: Literal["gpu", "cpu"] = "gpu",
        window_size: tuple[int, int] = (1800, 900),
        image_size: tuple[int, int] | None = None,
        screenshot_format: Literal["jpeg", "png"] = "jpeg",
        draw_mouse: bool = False,
        left_pane: bool = False,
        right_pane: bool = True,
        # --- Obs/action representation (locks spaces at construction) ------------
        orientation: Literal["quaternion", "euler"] = "quaternion",
        # --- Task hooks ----------------------------------------------------------
        reset_state_provider: StateProvider | None = None,
        reward_factory: RewardFactory | None = None,
        termination_factory: TerminationFactory | None = None,
        # --- Self-healing (sensible-on defaults; None / 0 disables) --------------
        retry_on_reset: int = 3,
        browser_restart_every: int | None = 90,
        fresh_context_every_reset: bool = True,
        step_timeout_s: float | None = 30.0,
        reset_timeout_s: float | None = 240.0,
        state_ready_timeout_s: float = 2.0,
        restart_after_consecutive_failures: int = 3,
        nav_timeout_ms: int = 90_000,
        recovery_mode: Literal["escalate", "in_place"] = "escalate",
        reset_ahead: bool = False,
        reset_ahead_after_steps: int = 270,
        capture_scale: float = 1.0,
        clear_cache_on_recycle: bool = True,
        extra_launch_args: list[str] | None = None,
        # --- Deployment defaults -------------------------------------------------
        config_path: str | None = None,
        verbose: bool = False,
    ):
        super().__init__()

        if not (left_pane or right_pane):
            raise ValueError("At least one of `left_pane` or `right_pane` must be True.")
        if orientation not in ("quaternion", "euler"):
            raise ValueError(
                f"`orientation` must be 'quaternion' or 'euler'; got {orientation!r}"
            )
        if renderer not in ("gpu", "cpu"):
            raise ValueError(f"`renderer` must be 'gpu' or 'cpu'; got {renderer!r}")
        if screenshot_format not in ("jpeg", "png"):
            raise ValueError(
                f"`screenshot_format` must be 'jpeg' or 'png'; got {screenshot_format!r}"
            )
        if recovery_mode not in ("escalate", "in_place"):
            raise ValueError(
                f"`recovery_mode` must be 'escalate' or 'in_place'; got {recovery_mode!r}"
            )
        if not (0.0 < capture_scale <= 1.0):
            raise ValueError(f"`capture_scale` must be in (0, 1]; got {capture_scale!r}")

        # Store config
        self.headless = headless
        self.renderer = renderer
        self.window_size = window_size
        self.image_size = image_size
        self.screenshot_format = screenshot_format
        self.draw_mouse = draw_mouse
        self.left_pane = left_pane
        self.right_pane = right_pane
        self.orientation = orientation
        self.verbose = verbose
        self.retry_on_reset = retry_on_reset
        self.browser_restart_every = browser_restart_every
        self.fresh_context_every_reset = fresh_context_every_reset
        self.step_timeout_s = step_timeout_s
        self.reset_timeout_s = reset_timeout_s
        self.state_ready_timeout_s = state_ready_timeout_s
        self.restart_after_consecutive_failures = restart_after_consecutive_failures
        self.nav_timeout_ms = nav_timeout_ms
        self.recovery_mode = recovery_mode
        # M5 reset-ahead: pre-navigate the NEXT episode in a warm BrowserContext
        # while the current episode is still stepping, so the reset swaps pages
        # instead of paying navigation+settle (~3s median, 8-70s tail) on the
        # critical path. All prep runs on the env's own thread (Playwright sync
        # objects are thread-bound) as small ticks piggybacked on step().
        self.reset_ahead = reset_ahead
        self.reset_ahead_after_steps = reset_ahead_after_steps
        # Capture the frame downscaled IN THE BROWSER (compositor does it on the
        # GPU) instead of shipping full-res pixels for Python to decode+resize —
        # the decode/resize is GIL-held per-step work, the true aggregate cost.
        # Aspect ratio is preserved (window 2:1 -> capture 2:1, just smaller).
        self.capture_scale = capture_scale
        # Per-episode HTTP-cache clear forces re-downloading the Neuroglancer app
        # + mesh chunks every navigation. The in-page memory the recycle exists
        # to free (JS heap/WebGL) is separate; the HTTP cache is disk-backed,
        # LRU-bounded by Chrome, and wiped by the periodic browser restart.
        self.clear_cache_on_recycle = clear_cache_on_recycle
        self.extra_launch_args = list(extra_launch_args or [])
        self._warm: dict[str, Any] | None = None
        self._warm_skip_ep = -1  # episode whose prep failed; don't retry within it
        self._chrome_pid: int | None = None
        self._consecutive_step_failures = 0

        self.config = self._load_config(config_path)
        self._image_shape = self._compute_image_shape(
            window_size, left_pane, right_pane, image_size, capture_scale
        )

        # Task hooks (default no-ops if unspecified)
        self._reset_state_provider = reset_state_provider
        self._reward_factory = reward_factory or _noop_reward_factory
        self._termination_factory = termination_factory or _noop_termination_factory

        # Spaces — locked at construction by orientation + image_shape
        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()

        # Browser state (lazy — launched on first reset)
        self._playwright = None
        self.browser = None
        self.page = None
        self._action_handler: MouseActionHandler | None = None

        # Episode state
        self._rng: np.random.Generator = np.random.default_rng()
        self._episode_count = 0
        self._needs_browser_restart = False  # set on step-time observation failure
        self._prev_obs: dict[str, Any] | None = None
        self._prev_json: dict[str, Any] | None = None
        self._task_info: dict[str, Any] = {}
        self._reward_fn: Callable | None = None
        self._terminated_fn: Callable | None = None

        # --- Optional event instrumentation (storm-precursor analysis) --------
        # Zero behavioral effect; JSONL emitted only when NGLLIB_EVENT_LOG is set
        # (path template may contain '{pid}'). Captures per-reset + per-glitch
        # context (reason, phase timings, browser age, settle polls) so we can
        # reconstruct what triggers the FIRST reset before a storm.
        self._event_log_path = os.environ.get("NGLLIB_EVENT_LOG")
        self._evt_fh = None
        self._evt_host: str | None = None
        self._ep_steps = 0             # steps completed in the current episode
        self._ep_terminated = False    # did the last step self-terminate (success)?
        self._last_step_glitched = False
        self._last_settle_polls = 0    # state-settle poll iterations in last gather
        self._last_nav_attempts = 1    # navigate attempts used in last reset
        # Per-episode step-time accumulators (F4 attribution: stragglers with no
        # reset/glitch events = steps themselves ran slow; this makes that
        # visible). Summarized into the reset event; individual steps >5s also
        # emit an immediate slow_step event.
        self._ep_step_ms_sum = 0.0
        self._ep_step_ms_max = 0.0
        self._ep_slow_steps = 0        # steps >2s in the current episode

    def _emit(self, evt: str, **fields) -> None:
        """Append one JSONL event when NGLLIB_EVENT_LOG is set; else no-op."""
        if not self._event_log_path:
            return
        try:
            if self._evt_fh is None:
                import socket
                self._evt_host = socket.gethostname()
                self._evt_fh = open(
                    self._event_log_path.format(pid=os.getpid(), host=self._evt_host),
                    "a", buffering=1,
                )
            rec = {
                "evt": evt, "ts": time.time(), "mono": time.monotonic(),
                "pid": os.getpid(), "host": self._evt_host,
                "episode": self._episode_count, **fields,
            }
            self._evt_fh.write(json.dumps(rec) + "\n")
        except Exception:
            pass  # instrumentation must never break the env

    # =========================================================================
    # Gymnasium public API
    # =========================================================================

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # Ending-episode stats (for reset-reason attribution) BEFORE the bump.
        _prev_steps, _prev_term = self._ep_steps, self._ep_terminated
        _prev_glitched, _first = self._last_step_glitched, self._episode_count == 0
        self._episode_count += 1
        options = options or {}
        _t0 = time.monotonic()
        _restarted = False

        if self._needs_browser_restart or (
            self.browser_restart_every is not None
            and self._episode_count > 1
            and (self._episode_count - 1) % self.browser_restart_every == 0
        ):
            self._needs_browser_restart = False
            self._restart_browser()
            _restarted = True

        # M5: adopt the warm pre-navigated context if prep finished. Skip when a
        # seed reseeds the RNG (pre-sampled state used the old stream) or options
        # override the reset (eval paths) — those fall back to the inline path.
        _warm_used = False
        if (
            self.reset_ahead
            and isinstance(self._warm, dict)
            and self._warm.get("ready")
            and not _restarted
            and seed is None
            and not options
        ):
            try:
                start_state, task_info = self._adopt_warm()
                _warm_used = True
            except Exception as e:
                logger.warning("reset-ahead adopt failed (%s); inline reset", e)
                self._discard_warm()
        if not _warm_used:
            self._discard_warm()  # stale/unready prep can't be reused
            self._ensure_browser_launched()
            if self.fresh_context_every_reset:
                self._recycle_context()
            start_state, task_info = self._resolve_reset_state(options)

        try:
            self._reward_fn = self._reward_factory(task_info)
            self._terminated_fn = self._termination_factory(task_info)
        except Exception as e:
            raise ProviderError(f"factory raised at reset: {e}") from e
        self._task_info = task_info

        _tn = time.monotonic()
        if not _warm_used:
            self._navigate_with_retry(start_state)
        _navigate_ms = (time.monotonic() - _tn) * 1000.0

        _tg = time.monotonic()
        obs, json_state = self._gather_observation()
        _gather_ms = (time.monotonic() - _tg) * 1000.0
        self._prev_obs = obs
        self._prev_json = json_state

        self._emit(
            "reset", total_ms=(time.monotonic() - _t0) * 1000.0,
            navigate_ms=_navigate_ms, gather_ms=_gather_ms,
            nav_attempts=self._last_nav_attempts, settle_polls=self._last_settle_polls,
            restarted=_restarted, first=_first, warm=_warm_used,
            prev_steps=_prev_steps, prev_terminated=_prev_term, prev_glitched=_prev_glitched,
            prev_step_ms_mean=(self._ep_step_ms_sum / _prev_steps if _prev_steps else 0.0),
            prev_step_ms_max=self._ep_step_ms_max,
            prev_slow_steps=self._ep_slow_steps,
            segment=(task_info.get("segment_id") if isinstance(task_info, dict) else None),
        )
        # Start counters for the new episode.
        self._ep_steps = 0
        self._ep_terminated = False
        self._last_step_glitched = False
        self._ep_step_ms_sum = 0.0
        self._ep_step_ms_max = 0.0
        self._ep_slow_steps = 0

        info = {
            "task_info": task_info,
            "json_state": json_state,
            "step": 0,
        }
        return obs, info

    def step(self, action):
        _t_step = time.monotonic()
        wd = self._watchdog(self.step_timeout_s)
        try:
            self._apply_actions(action)
            obs, json_state = self._gather_observation()
        except BrowserError as e:
            # Transient viewer-state races are common under GPU contention (a
            # slow render not settled when we read state). A SINGLE failure just
            # truncates the episode; only a browser that fails repeatedly is
            # genuinely sick, so escalate to a full restart only after N
            # consecutive failures. (Escalating on every miss caused a
            # restart-storm that tanked throughput at high M — 2026-07-09.)
            self._consecutive_step_failures += 1
            self._last_step_glitched = True
            self._emit("glitch", phase="step", step_idx=self._ep_steps,
                       consecutive=self._consecutive_step_failures,
                       settle_polls=self._last_settle_polls, signature=str(e)[:140])
            if (
                self.recovery_mode == "escalate"
                and self._consecutive_step_failures
                >= self.restart_after_consecutive_failures
            ):
                # 'in_place' skips the full-browser-restart escalation: the browser
                # is still alive (a watchdog KILL takes the wd.fired path below and
                # forces a restart regardless), so the cheap per-reset context
                # recycle (fresh_context_every_reset) recovers without a cold
                # relaunch storm.
                self._needs_browser_restart = True
            raise
        except Exception as e:
            if wd.fired:
                self._consecutive_step_failures += 1
                self._last_step_glitched = True
                self._emit("glitch", phase="step", step_idx=self._ep_steps,
                           consecutive=self._consecutive_step_failures,
                           signature=f"watchdog hang >{self.step_timeout_s}s: {str(e)[:100]}")
                if self._consecutive_step_failures >= self.restart_after_consecutive_failures:
                    self._needs_browser_restart = True
                raise BrowserError(
                    f"step hung >{self.step_timeout_s}s; browser killed by watchdog: {e}"
                ) from e
            raise
        finally:
            wd.cancel()
        self._consecutive_step_failures = 0  # a clean step ends the streak

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

        info = {
            "task_info": self._task_info,
            "json_state": json_state,
        }

        self._prev_obs = obs
        self._prev_json = json_state
        self._ep_steps += 1
        self._ep_terminated = terminated
        if self.reset_ahead:
            self._reset_ahead_tick()
        _dur_ms = (time.monotonic() - _t_step) * 1000.0
        self._ep_step_ms_sum += _dur_ms
        if _dur_ms > self._ep_step_ms_max:
            self._ep_step_ms_max = _dur_ms
        if _dur_ms > 2000.0:
            self._ep_slow_steps += 1
            if _dur_ms > 5000.0:
                self._emit("slow_step", ms=_dur_ms, step_idx=self._ep_steps,
                           settle_polls=self._last_settle_polls)
        return obs, reward, terminated, truncated, info

    def close(self):
        """Tear down the browser. Idempotent."""
        self._discard_warm()
        try:
            if self.page is not None:
                self.page.close()
        except Exception:
            pass
        try:
            if self.browser is not None:
                self.browser.close()
        except Exception:
            pass
        try:
            if self._playwright is not None:
                self._playwright.stop()
        except Exception:
            pass
        self.page = None
        self.browser = None
        self._playwright = None
        self._action_handler = None
        self._chrome_pid = None
        self._driver_pid = None

    # =========================================================================
    # Internal: config & space construction
    # =========================================================================

    @staticmethod
    def _load_config(config_path: str | None) -> dict[str, Any]:
        if config_path is None:
            from importlib.resources import files
            return json.loads(files("ngllib").joinpath("config.json").read_text())
        with open(config_path) as f:
            return json.load(f)

    @staticmethod
    def _compute_image_shape(
        window_size: tuple[int, int],
        left_pane: bool,
        right_pane: bool,
        image_size: tuple[int, int] | None,
        capture_scale: float = 1.0,
    ) -> tuple[int, int, int]:
        # numpy shape: (H, W, C). image_size kwarg is (W, H) for user-friendliness.
        if image_size is not None:
            iw, ih = image_size
            return (ih, iw, 3)
        W, H = round(window_size[0] * capture_scale), round(window_size[1] * capture_scale)
        if left_pane and right_pane:
            return (H, W, 3)
        return (H, W // 2, 3)  # single-pane crop

    def _build_observation_space(self) -> spaces.Dict:
        orient_dim = 3 if self.orientation == "euler" else 4
        return spaces.Dict(
            {
                "position": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
                "xs_scale": spaces.Box(
                    low=0.0, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "orientation": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(orient_dim,), dtype=np.float32
                ),
                "proj_scale": spaces.Box(
                    low=0.0, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "image": spaces.Box(
                    low=0, high=255, shape=self._image_shape, dtype=np.uint8
                ),
            }
        )

    def _build_action_space(self) -> spaces.Dict:
        W, H = self.window_size
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
                "delta_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
                "delta_xs_scale": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "delta_orient": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(orient_dim,), dtype=np.float32
                ),
                "delta_proj_scale": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
            }
        )

    # =========================================================================
    # Internal: browser lifecycle
    # =========================================================================

    def _build_launch_args(self) -> list[str]:
        args = [
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
        ]
        if self.headless:
            # Compositor throttling dominates step latency: after an action the
            # browser waits for a vsync/frame-rate-limited frame before the
            # screenshot can capture the new state. Disabling both removes that
            # wait (offscreen render has no use for vsync/fps caps). NOT using
            # --run-all-compositor-stages-before-draw: it sped things up too but
            # intermittently DEADLOCKS page.screenshot (30s timeout) in headless.
            args += [
                "--disable-gpu-vsync",
                "--disable-frame-rate-limit",
            ]
            if self.renderer == "cpu":
                args += ["--use-gl=swiftshader", "--enable-unsafe-swiftshader"]
            else:  # "gpu" — auto-select per OS
                os_name = platform.system()
                if os_name == "Darwin":
                    args += ["--use-gl=angle", "--use-angle=metal"]
                elif os_name == "Windows":
                    args += ["--use-gl=angle", "--use-angle=d3d11"]
                else:  # Linux
                    args += [
                        "--use-gl=angle",
                        "--use-angle=vulkan",
                        "--enable-features=Vulkan",
                        "--enable-unsafe-swiftshader",
                    ]
        args += self.extra_launch_args
        return args

    def _ensure_browser_launched(self) -> None:
        if self.browser is not None:
            return
        # Playwright subprocess-spawns Chromium via asyncio, which on Windows
        # requires WindowsProactorEventLoopPolicy. IPython 7+ swaps in the
        # Selector policy globally; restore Proactor here when that's happened.
        if sys.platform == "win32":
            current = asyncio.get_event_loop_policy()
            if not isinstance(current, asyncio.WindowsProactorEventLoopPolicy):
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        # Guard the LAUNCH itself: driver spawn / chromium launch can hang with
        # no pid yet known to kill (observed 2026-08-17: post-tree-kill relaunch
        # froze here, silent forever). Fallback kills any node/chrome children
        # spawned since pre_launch — ours with near-certainty; a rare sibling
        # casualty just self-heals through its own recovery path.
        pre_launch = time.time()

        def _kill_new_children():
            try:
                for c in psutil.Process(os.getpid()).children(recursive=True):
                    try:
                        if c.create_time() >= pre_launch - 1.0 and (
                            "node" in c.name().lower() or "chrom" in c.name().lower()
                        ):
                            c.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                logger.warning("launch watchdog swept children spawned since launch")
            except Exception:
                pass

        _launch_wd = _BrowserWatchdog(120.0, _kill_new_children)
        try:
            _kids_before = {c.pid for c in psutil.Process(os.getpid()).children()}
            self._playwright = sync_playwright().start()
            # The driver (a direct node child of this process) mediates every
            # sync call; killing it unblocks ANY hung Playwright call — the
            # watchdog fallback for phases where no Chrome pid exists (launch/
            # relaunch), which previously hung unguarded (2026-08-17).
            try:
                self._driver_pid = next(
                    (c.pid for c in psutil.Process(os.getpid()).children()
                     if c.pid not in _kids_before), None)
            except Exception:
                self._driver_pid = None
            W, H = self.window_size
            self.browser = self._playwright.chromium.launch(
                headless=self.headless,
                args=self._build_launch_args(),
            )
            self.page = self.browser.new_page(viewport={"width": W, "height": H})
            self._action_handler = MouseActionHandler(self.page)
            # Playwright doesn't expose the browser process; find it for the
            # hang watchdog (kill target). THIS env's Chrome is a child of THIS
            # env's driver — an unambiguous lookup. The old process-wide
            # first-match could pick a SIBLING env's browser in the 16-env
            # process, making the watchdog kill the wrong browser: the sibling
            # recovered (looked like a routine glitch) while the truly hung env
            # stayed blocked forever (the silent-runner saga, 2026-08-17).
            self._chrome_pid = None
            if self._driver_pid is not None:
                for _ in range(20):
                    try:
                        kids = psutil.Process(self._driver_pid).children(recursive=True)
                        self._chrome_pid = next(
                            (c.pid for c in kids if "chrom" in c.name().lower()), None)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        break
                    if self._chrome_pid is not None:
                        break
                    time.sleep(0.1)
            if self._chrome_pid is None:
                self._chrome_pid = self._find_chrome_pid(pre_launch)  # legacy fallback
        except Exception as e:
            detail = str(e) or repr(e)
            raise BrowserError(
                f"failed to launch Chromium: {type(e).__name__}: {detail}. "
                "Ensure 'playwright install chromium' has been run in this venv."
            ) from e
        finally:
            _launch_wd.cancel()

    def _restart_browser(self) -> None:
        logger.info("periodic browser restart at episode %d", self._episode_count)
        self.close()
        self._ensure_browser_launched()

    @staticmethod
    def _find_chrome_pid(pre_launch: float) -> int | None:
        try:
            for child in psutil.Process(os.getpid()).children(recursive=True):
                try:
                    name = child.name().lower()
                    if ("chrome" in name or "chromium" in name) and (
                        child.create_time() >= pre_launch - 1.0
                    ):
                        return child.pid
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception:
            pass
        return None

    def _kill_chrome(self) -> None:
        """Watchdog target: kill the browser so a blocked Playwright call raises.

        Kills the ENTIRE Chrome process tree of THIS env, first ascending from
        the recorded pid to the topmost Chrome ancestor. Killing only the
        recorded pid can hit a child (gpu/renderer process) while the
        socket-owning main browser survives — the blocked call then never
        raises and the env thread hangs silently, stalling the whole runner
        (observed 2026-08-17: hour-long iteration, no raise, no log). Scoped
        to this env's tree so sibling browsers in the same process are safe.
        """
        self._needs_browser_restart = True
        pid = self._chrome_pid
        if pid is None:
            self._kill_driver()
            return
        try:
            proc = psutil.Process(pid)
            # Ascend to the topmost chrome process (the main browser).
            for _ in range(6):
                par = proc.parent()
                if par is not None and "chrom" in (par.name() or "").lower():
                    proc = par
                else:
                    break
            killed = []
            for c in proc.children(recursive=True):
                try:
                    c.kill()
                    killed.append(c.pid)
                except psutil.NoSuchProcess:
                    pass
            proc.kill()
            killed.append(proc.pid)
            logger.warning("watchdog killed hung Chrome tree (%d procs, root %d)",
                           len(killed), proc.pid)
        except psutil.NoSuchProcess:
            self._kill_driver()

    def _kill_driver(self) -> None:
        """Last-resort unblocker: kill this env's Playwright node driver."""
        dpid = getattr(self, "_driver_pid", None)
        if dpid is None:
            return
        try:
            proc = psutil.Process(dpid)
            for c in proc.children(recursive=True):
                try:
                    c.kill()
                except psutil.NoSuchProcess:
                    pass
            proc.kill()
            logger.warning("watchdog killed Playwright driver (pid %d)", dpid)
        except psutil.NoSuchProcess:
            pass
        self._driver_pid = None

    def _watchdog(self, timeout_s: float | None) -> _BrowserWatchdog:
        armed = self._chrome_pid is not None or getattr(self, "_driver_pid", None) is not None
        return _BrowserWatchdog(timeout_s if armed else None, self._kill_chrome)

    def _recycle_context(self) -> None:
        """Fresh BrowserContext + Page (and HTTP-cache clear) for the episode.

        Page-level state (JS heap, WebGL contexts, caches) accumulates across
        episodes and steadily degrades step throughput on long runs; recycling
        the context every reset keeps it flat. The periodic full browser
        restart still handles browser-process-level leaks.
        """
        W, H = self.window_size
        old = self.page.context if self.page is not None else None
        context = self.browser.new_context(viewport={"width": W, "height": H})
        page = context.new_page()
        if self.clear_cache_on_recycle:
            try:  # chromium-only CDP call; harmless to skip on failure
                cdp = context.new_cdp_session(page)
                cdp.send("Network.clearBrowserCache")
                cdp.detach()
            except Exception:
                pass
        self.page = page
        self._action_handler = MouseActionHandler(page)
        if old is not None:
            try:
                old.close()
            except Exception:
                pass

    # =========================================================================
    # Internal: M5 reset-ahead (warm-context prep off the critical path)
    # =========================================================================

    def _discard_warm(self) -> None:
        w = self._warm
        self._warm = None
        if isinstance(w, dict):
            try:
                w["context"].close()
            except Exception:
                pass

    def _reset_ahead_tick(self) -> None:
        """One small prep step, piggybacked on step(); runs on the env's own
        thread (Playwright sync objects are thread-bound). First eligible tick
        creates a fresh context and kicks navigation (browser loads it in the
        background); later ticks cheaply poll readiness. Any failure discards
        the prep — reset() then just takes the normal inline path."""
        if self._ep_steps < self.reset_ahead_after_steps:
            return
        if self._warm_skip_ep == self._episode_count and self._warm is None:
            return
        w = self._warm
        if w is None:
            wd = self._watchdog(15.0)
            ctx = None
            try:
                start_state, task_info = self._resolve_reset_state({})
                url = self._start_state_to_url(start_state)
                W, H = self.window_size
                ctx = self.browser.new_context(viewport={"width": W, "height": H})
                page = ctx.new_page()
                if self.clear_cache_on_recycle:
                    try:  # mirror _recycle_context's cache clear (chromium-only)
                        cdp = ctx.new_cdp_session(page)
                        cdp.send("Network.clearBrowserCache")
                        cdp.detach()
                    except Exception:
                        pass
                # wait_until="commit" returns once navigation starts; the
                # browser keeps loading/rendering in the background while the
                # active episode continues stepping.
                page.goto(url, timeout=15_000, wait_until="commit")
                self._warm = {
                    "context": ctx, "page": page, "start_state": start_state,
                    "task_info": task_info, "ready": False,
                    "t0": time.monotonic(),
                }
            except Exception as e:
                if ctx is not None:
                    try:
                        ctx.close()
                    except Exception:
                        pass
                self._warm = None
                self._warm_skip_ep = self._episode_count
                self._emit("warm_prep_failed", signature=str(e)[:120])
            finally:
                wd.cancel()
        elif not w.get("ready"):
            wd = self._watchdog(10.0)
            try:
                page = w["page"]
                raw = page.evaluate(
                    "() => (window.viewer && window.viewer.state) ? "
                    "JSON.stringify(window.viewer.state) : null"
                )
                if raw is not None and page.evaluate(
                    "() => !!(window.viewer && window.viewer.isReady && "
                    "window.viewer.isReady())"
                ):
                    w["ready"] = True
                    self._emit(
                        "warm_ready",
                        prep_ms=(time.monotonic() - w["t0"]) * 1000.0,
                    )
            except Exception:
                pass  # still loading (or context died — adopt/discard handles it)
            finally:
                wd.cancel()

    def _adopt_warm(self):
        """Swap the ready warm context in as the active page. Returns the
        pre-sampled (start_state, task_info)."""
        w = self._warm
        self._warm = None
        old = self.page.context if self.page is not None else None
        self.page = w["page"]
        self._action_handler = MouseActionHandler(self.page)
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
        return w["start_state"], w["task_info"]

    # =========================================================================
    # Internal: reset / state navigation
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
                    raise ProviderError(
                        f"provider.task_info_from_state raised: {e}"
                    ) from e
            else:
                task_info = {}
            return start_state, task_info

        if self._reset_state_provider is not None:
            try:
                start_state, task_info = self._reset_state_provider(self._rng, options)
            except Exception as e:
                raise ProviderError(f"reset_state_provider raised: {e}") from e
            return start_state, task_info

        # No provider, no override: use the default URL from config.
        return None, {}

    def _navigate_with_retry(self, start_state) -> None:
        """Navigate with retry per self-healing config. Restarts the browser between attempts."""
        last_err: Exception | None = None
        for attempt in range(max(1, self.retry_on_reset + 1)):
            try:
                self._navigate(start_state)
                self._last_nav_attempts = attempt + 1
                return
            except BrowserError:
                # Launch failures aren't retryable through restart.
                raise
            except Exception as e:
                last_err = e
                self._last_nav_attempts = attempt + 1
                self._emit("glitch", phase="reset", attempt=attempt + 1,
                           settle_polls=self._last_settle_polls, signature=str(e)[:140])
                logger.warning("reset attempt %d failed: %s", attempt + 1, e)
                if attempt < self.retry_on_reset:
                    try:
                        if self.recovery_mode == "in_place":
                            # Cheap recovery at the source (legacy-style): a fresh
                            # BrowserContext on the SAME Chrome, not a full browser
                            # relaunch. A cold Chromium restart under many-browser
                            # load is what turns one glitch into a restart-storm
                            # straggler; recycling the context settles in ~seconds.
                            self._recycle_context()
                        else:
                            self._restart_browser()
                    except Exception:
                        # If the cheap recycle can't run (e.g. the browser process
                        # is actually dead), fall back to a full restart.
                        try:
                            self._restart_browser()
                        except Exception:
                            pass
                    time.sleep(0.5)
        raise BrowserError(
            f"reset failed after {self.retry_on_reset + 1} attempts: {last_err}"
        )

    def _navigate(self, start_state) -> None:
        """Resolve start_state to a URL, navigate, and wait for viewer ready.

        Guarded by the reset watchdog: a hung navigation gets its Chrome
        killed, raising here as a retryable error so `_navigate_with_retry`
        restarts the browser and tries again (NOT BrowserError — that aborts
        the retry loop by design).
        """
        wd = self._watchdog(self.reset_timeout_s)
        try:
            self._navigate_inner(start_state)
        except Exception as e:
            if wd.fired:
                raise RuntimeError(
                    f"navigation hung >{self.reset_timeout_s}s; browser killed by watchdog: {e}"
                ) from e
            raise
        finally:
            wd.cancel()

    def _start_state_to_url(self, start_state) -> str:
        """Resolve the three start_state forms (None / URL str / NglState dict)."""
        if start_state is None:
            url = self.config.get("default_ngl_start_url")
            if not url:
                raise ProviderError(
                    "No start state provided and config has no 'default_ngl_start_url'"
                )
            return url
        if isinstance(start_state, str):
            return start_state
        if isinstance(start_state, dict):
            return self._state_dict_to_url(start_state)
        raise ProviderError(
            "start_state must be None, a URL str, or an NglState dict; "
            f"got {type(start_state).__name__}"
        )

    def _navigate_inner(self, start_state) -> None:
        url = self._start_state_to_url(start_state)
        if self.verbose:
            print(f"Navigating to: {url[:120]}{'...' if len(url) > 120 else ''}")
        # Longer than Playwright's 30s default: a cold start with many browsers
        # sharing a node/GPU can legitimately take >30s to load Neuroglancer
        # (thundering-herd on first reset). The reset watchdog bounds true hangs.
        self.page.goto(url, timeout=self.nav_timeout_ms)

        # Wait for viewer to initialize.
        for _ in range(600):
            if self._get_json_state_raw() is not None:
                break
            time.sleep(0.1)
        # Wait for chunks to load + rendering to complete.
        for _ in range(400):
            try:
                ready = self.page.evaluate(
                    "() => !!(window.viewer && window.viewer.isReady && window.viewer.isReady())"
                )
                if ready:
                    break
            except Exception:
                pass
            time.sleep(0.025)

    def _state_dict_to_url(self, state: NglState) -> str:
        """Overlay NglState onto the default URL's parsed state. Use `state["extra"]`
        for fields not first-class above; pass a URL string to bypass merging."""
        base_url = self.config.get("default_ngl_start_url", "")
        if "#!" in base_url:
            prefix, encoded = base_url.split("#!", 1)
            try:
                base_state = json.loads(urllib.parse.unquote(encoded))
            except Exception:
                base_state = {}
        else:
            prefix = "https://neuroglancer-demo.appspot.com/"
            base_state = {}

        merged = copy.deepcopy(base_state)
        if "extra" in state and isinstance(state["extra"], dict):
            merged.update(state["extra"])
        if "position" in state:
            merged["position"] = list(state["position"])
        if "crossSectionScale" in state:
            merged["crossSectionScale"] = float(state["crossSectionScale"])
        if "projectionOrientation" in state:
            merged["projectionOrientation"] = list(state["projectionOrientation"])
        if "projectionScale" in state:
            merged["projectionScale"] = float(state["projectionScale"])
        if "segments" in state and "layers" in merged:
            for layer in merged["layers"]:
                if layer.get("type") == "segmentation":
                    layer["segments"] = list(state["segments"])
                    break

        return prefix + "#!" + urllib.parse.quote(json.dumps(merged))

    # =========================================================================
    # Internal: observation construction
    # =========================================================================

    _REQUIRED_STATE_FIELDS = ("position", "crossSectionScale", "projectionScale")

    def _gather_observation(self) -> tuple[dict[str, Any], dict[str, Any]]:
        json_state = self._get_json_state()
        # The viewer JSON transiently omits fields mid-update (the render hasn't
        # settled when we read state). This scales with GPU contention (more
        # browsers/GPU = slower renders = more misses), so poll up to
        # state_ready_timeout_s for the fields to appear before giving up —
        # resolving the race in-place is far cheaper than truncating the episode.
        poll = 0.05
        polls = 0
        deadline = time.monotonic() + self.state_ready_timeout_s
        while not all(k in json_state for k in self._REQUIRED_STATE_FIELDS):
            if time.monotonic() >= deadline:
                self._last_settle_polls = polls
                missing = sorted(set(self._REQUIRED_STATE_FIELDS) - set(json_state))
                raise BrowserError(f"viewer state missing fields after retries: {missing}")
            time.sleep(poll)
            polls += 1
            json_state = self._get_json_state()
        self._last_settle_polls = polls
        image = self._get_screenshot()
        image = self._crop_panes(image)
        if self.image_size is not None:
            image = self._resize_image(image, self.image_size)

        position = np.asarray(json_state["position"], dtype=np.float32)
        xs_scale = np.asarray([json_state["crossSectionScale"]], dtype=np.float32)
        orient_raw = json_state.get("projectionOrientation", [0.0, 0.0, 0.0, 1.0])
        if self.orientation == "euler":
            orient = np.asarray(quaternion_to_euler(orient_raw), dtype=np.float32)
        else:
            orient = np.asarray(orient_raw, dtype=np.float32)
        proj_scale = np.asarray([json_state["projectionScale"]], dtype=np.float32)

        obs = {
            "position": position,
            "xs_scale": xs_scale,
            "orientation": orient,
            "proj_scale": proj_scale,
            "image": image,
        }
        return obs, json_state

    def _crop_panes(self, image: np.ndarray) -> np.ndarray:
        if self.left_pane and self.right_pane:
            return image
        mid = image.shape[1] // 2
        return image[:, mid:] if self.right_pane else image[:, :mid]

    @staticmethod
    def _resize_image(image: np.ndarray, target_wh: tuple[int, int]) -> np.ndarray:
        pil = Image.fromarray(image)
        pil = pil.resize(target_wh)
        return np.asarray(pil)

    def _get_json_state_raw(self) -> str | None:
        try:
            result = self.page.evaluate(
                "() => (window.viewer && window.viewer.state) ? "
                "JSON.stringify(window.viewer.state) : null"
            )
            self._last_state_read_error = None
            return result
        except Exception as e:
            # Preserve the underlying cause instead of swallowing it: after a
            # watchdog kill the connection error surfaces HERE, and converting
            # it silently to "could not read" masked the true blocked call
            # (2026-08-17 hang forensics).
            self._last_state_read_error = f"{type(e).__name__}: {str(e)[:140]}"
            return None

    def _get_json_state(self) -> dict[str, Any]:
        raw = self._get_json_state_raw()
        if raw is None:
            cause = getattr(self, "_last_state_read_error", None)
            raise BrowserError(
                "could not read Neuroglancer viewer state from page"
                + (f" [cause: {cause}]" if cause else "")
            )
        state = json.loads(raw)
        if "projectionOrientation" not in state:
            state["projectionOrientation"] = [0.0, 0.0, 0.0, 1.0]
        return state

    def _capture_cdp_session(self):
        """CDP session cached per page (page changes on context recycle/adopt)."""
        if getattr(self, "_cap_cdp_page", None) is not self.page:
            self._cap_cdp = self.page.context.new_cdp_session(self.page)
            self._cap_cdp_page = self.page
        return self._cap_cdp

    def _get_screenshot(self) -> np.ndarray:
        if self.capture_scale != 1.0:
            # Browser-side downscale: the compositor scales on the GPU and ships
            # capture_scale^2 x fewer pixels — cutting the GIL-held Python
            # decode/resize per step, which is the aggregate throughput cost.
            W, H = self.window_size
            fmt = "jpeg" if self.screenshot_format == "jpeg" else "png"
            params: dict[str, Any] = {
                "format": fmt,
                "clip": {"x": 0, "y": 0, "width": W, "height": H,
                         "scale": self.capture_scale},
            }
            if fmt == "jpeg":
                params["quality"] = 85
            res = self._capture_cdp_session().send("Page.captureScreenshot", params)
            data = base64.b64decode(res["data"])
            pil = Image.open(io.BytesIO(data)).convert("RGB")
            # CDP's scaled clip can be off-by-one vs round(); normalize so the
            # observation shape is exact.
            expW, expH = round(W * self.capture_scale), round(H * self.capture_scale)
            if pil.size != (expW, expH):
                pil = pil.resize((expW, expH))
            return np.asarray(pil)
        if self.screenshot_format == "jpeg":
            data = self.page.screenshot(type="jpeg", quality=85)
        else:
            data = self.page.screenshot()
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        return np.asarray(pil)

    # =========================================================================
    # Internal: action application
    # =========================================================================

    def _apply_actions(self, action: dict[str, Any]) -> None:
        action_type = int(action["action_type"])
        if action_type in (0, 1, 2):
            self._apply_click(action_type, action)
        elif action_type == 3:
            self._apply_state_edit(action)
        else:
            raise ValueError(
                f"action_type must be 0, 1, 2, or 3; got {action_type}"
            )

    def _apply_click(self, action_type: int, action: dict[str, Any]) -> None:
        x, y = (float(v) for v in action["mouse_xy"])
        key_pressed = self._modifiers_to_str(action["modifiers"])
        kind = {0: "left_click", 1: "right_click", 2: "double_click"}[action_type]
        if self.verbose:
            print(f"Click ({kind}) at ({x:.1f}, {y:.1f}) modifiers={key_pressed!r}")
        self._action_handler.execute_click(x, y, kind, key_pressed)

    def _apply_state_edit(self, action: dict[str, Any]) -> None:
        if self._prev_json is None:
            return  # haven't gathered initial state yet (shouldn't happen post-reset)
        new_state = copy.deepcopy(self._prev_json)

        dpos = action["delta_pos"]
        new_state["position"][0] += float(dpos[0])
        new_state["position"][1] += float(dpos[1])
        new_state["position"][2] += float(dpos[2])
        new_state["crossSectionScale"] += float(action["delta_xs_scale"][0])

        d = action["delta_orient"]
        if self.orientation == "euler":
            old_euler = quaternion_to_euler(new_state["projectionOrientation"])
            new_euler = [
                old_euler[0] + float(d[0]),
                old_euler[1] + float(d[1]),
                old_euler[2] + float(d[2]),
            ]
            new_state["projectionOrientation"] = euler_to_quaternion(new_euler)
        else:
            for i in range(4):
                new_state["projectionOrientation"][i] += float(d[i])

        new_state["projectionScale"] = min(
            500_000,
            new_state["projectionScale"] + float(action["delta_proj_scale"][0]),
        )

        self._change_json_state_url(new_state)

        if self.verbose:
            print(f"State edit applied: pos={new_state['position']}")

    @staticmethod
    def _modifiers_to_str(modifiers) -> str:
        parts = []
        if int(modifiers[0]):
            parts.append("Shift")
        if int(modifiers[1]):
            parts.append("Ctrl")
        if int(modifiers[2]):
            parts.append("Alt")
        return ", ".join(parts)

    def _change_json_state_url(self, new_state: dict[str, Any]) -> None:
        serialized = json.dumps(new_state)
        encoded = urllib.parse.quote(serialized)
        url = f"https://neuroglancer-demo.appspot.com/#!{encoded}"
        self.page.goto(url)
