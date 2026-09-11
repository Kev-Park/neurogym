"""Chrome renderer: a headless Chromium running Neuroglancer, driven by Playwright.

This is the deployment target, so it is normative for state and action
semantics: the environment hands it states and input events, Neuroglancer
does whatever it does, and `observe()` reads the result back out of the page.
Everything about keeping a browser alive -- watchdogs, context recycling,
periodic restarts, navigation retries -- lives here and nowhere else.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import platform
import sys
import threading
import time
from typing import Any, Callable, Literal

import numpy as np
import psutil
from PIL import Image

from .dataset import (
    DatasetSpec,
    default_start_url,
    merge_state,
    ngl_state_from_viewer,
    split_state_url,
    state_to_url,
)
from .errors import BrowserError
from .events import EventLog
from .renderer import PaneLayout
from .state import REQUIRED_FIELDS
from .utils.MouseActionHandler import MouseActionHandler

logger = logging.getLogger(__name__)


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


class ChromeRenderer:
    """Playwright + Chromium backend for `ngllib.Environment`.

    Every parameter keeps the value the Chrome environment shipped with, so
    `Environment()` builds the same env it always did.

    `renderer` selects Chrome's GL stack: `cpu` is SwiftShader; `gpu` is ANGLE
    over Vulkan (Linux) / Metal / D3D11 -- on a node without the NVIDIA Vulkan
    ICD it silently falls back to SwiftShader, so pin nodes when HW rendering
    matters. `start_url` (default: `config.json`'s) is both the default start
    state and the base every NglState is overlaid onto.
    """

    # Two live browser contexts are expensive, so the environment is asked to
    # warm the next episode late in the current one (the reset tail it hides
    # is ~3s median, 8-70s tail).
    warm_after_steps = 270

    def __init__(
        self,
        *,
        headless: bool = True,
        renderer: Literal["gpu", "cpu"] = "gpu",
        window_size: tuple[int, int] = (1800, 900),
        capture_scale: float = 1.0,
        image_size: tuple[int, int] | None = None,
        left_pane: bool = False,
        right_pane: bool = True,
        screenshot_format: Literal["jpeg", "png"] = "jpeg",
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
        clear_cache_on_recycle: bool = True,
        extra_launch_args: list[str] | None = None,
        # --- Deployment identity -------------------------------------------------
        start_url: str | None = None,
        config_path: str | None = None,
    ):
        if renderer not in ("gpu", "cpu"):
            raise ValueError(f"`renderer` must be 'gpu' or 'cpu'; got {renderer!r}")
        if screenshot_format not in ("jpeg", "png"):
            raise ValueError(
                f"`screenshot_format` must be 'jpeg' or 'png'; got {screenshot_format!r}")
        if recovery_mode not in ("escalate", "in_place"):
            raise ValueError(
                f"`recovery_mode` must be 'escalate' or 'in_place'; got {recovery_mode!r}")

        self.layout = PaneLayout(
            window_size=window_size, capture_scale=capture_scale, image_size=image_size,
            left_pane=left_pane, right_pane=right_pane)
        self.events = EventLog(path_template="")  # replaced by the environment
        self.headless = headless
        self.renderer = renderer
        self.screenshot_format = screenshot_format
        self.retry_on_reset = retry_on_reset
        self.browser_restart_every = browser_restart_every
        self.fresh_context_every_reset = fresh_context_every_reset
        self.step_timeout_s = step_timeout_s
        self.reset_timeout_s = reset_timeout_s
        self.state_ready_timeout_s = state_ready_timeout_s
        self.restart_after_consecutive_failures = restart_after_consecutive_failures
        self.nav_timeout_ms = nav_timeout_ms
        self.recovery_mode = recovery_mode
        # Per-episode HTTP-cache clear forces re-downloading the Neuroglancer app
        # + mesh chunks every navigation. The in-page memory the recycle exists
        # to free (JS heap/WebGL) is separate; the HTTP cache is disk-backed,
        # LRU-bounded by Chrome, and wiped by the periodic browser restart.
        self.clear_cache_on_recycle = clear_cache_on_recycle
        self.extra_launch_args = list(extra_launch_args or [])

        self.start_url = start_url or default_start_url(config_path)
        self._url_prefix, self._base_state = split_state_url(self.start_url)
        self.dataset = DatasetSpec.from_state(self._base_state)

        # Browser state (lazy — launched on open())
        self._playwright = None
        self.browser = None
        self.page = None
        self._action_handler: MouseActionHandler | None = None
        self._chrome_pid: int | None = None
        self._driver_pid: int | None = None
        self._warm: dict[str, Any] | None = None
        self._viewer_json: dict[str, Any] | None = None  # last full readback
        self._episode_count = 0
        self._needs_browser_restart = False  # set on step-time observation failure
        self._consecutive_step_failures = 0
        self._last_settle_polls = 0
        self._last_nav_attempts = 1
        self._last_state_read_error: str | None = None

    # =========================================================================
    # Renderer protocol
    # =========================================================================

    def open(self) -> None:
        self._ensure_browser_launched()

    def close(self) -> None:
        """Tear down the browser. Idempotent."""
        self._discard_warm()
        for obj, method in ((self.page, "close"), (self.browser, "close"),
                            (self._playwright, "stop")):
            try:
                if obj is not None:
                    getattr(obj, method)()
            except Exception:
                pass
        self.page = None
        self.browser = None
        self._playwright = None
        self._action_handler = None
        self._chrome_pid = None
        self._driver_pid = None

    def default_state(self) -> dict[str, Any]:
        return ngl_state_from_viewer(self._base_state)

    def warm(self, state: dict[str, Any]) -> None:
        """Pre-navigate `state` in a fresh BrowserContext while the current
        episode keeps stepping; `reset_to` swaps pages instead of paying
        navigate+settle on the critical path. Runs on the env's own thread
        (Playwright sync objects are thread-bound)."""
        self._discard_warm()
        if self.browser is None:
            return
        wd = self._watchdog(15.0)
        ctx = None
        try:
            url = self._state_to_url(state)
            W, H = self.layout.window_size
            ctx = self.browser.new_context(viewport={"width": W, "height": H})
            page = ctx.new_page()
            if self.clear_cache_on_recycle:
                self._clear_cache(ctx, page)
            # wait_until="commit" returns once navigation starts; the browser
            # keeps loading/rendering in the background.
            page.goto(url, timeout=15_000, wait_until="commit")
            self._warm = {"context": ctx, "page": page, "state": state,
                          "t0": time.monotonic()}
        except Exception as e:
            if ctx is not None:
                try:
                    ctx.close()
                except Exception:
                    pass
            self._warm = None
            self.events.emit("warm_prep_failed", signature=str(e)[:120])
        finally:
            wd.cancel()

    def reset_to(self, state: dict[str, Any] | str) -> None:
        self._episode_count += 1
        restarted = False
        if self._needs_browser_restart or (
            self.browser_restart_every is not None
            and self._episode_count > 1
            and (self._episode_count - 1) % self.browser_restart_every == 0
        ):
            self._needs_browser_restart = False
            self._restart_browser()
            restarted = True

        warm_used = False
        w = self._warm
        if (w is not None and not restarted and isinstance(state, dict)
                and w["state"] == state and self._warm_ready()):
            self._adopt_warm()
            warm_used = True
        else:
            self._discard_warm()  # stale, unready or mismatched prep can't be reused
            self._ensure_browser_launched()
            if self.fresh_context_every_reset:
                self._recycle_context()
            self._navigate_with_retry(state)
        self.events.emit("chrome_reset", restarted=restarted, warm=warm_used,
                         nav_attempts=self._last_nav_attempts)

    def set_state(self, state: dict[str, Any]) -> None:
        base = self._viewer_json if self._viewer_json is not None else self._base_state
        url = state_to_url(self._url_prefix, merge_state(base, state))
        self._step_call(lambda: self.page.goto(url))

    def click(self, kind: str, x: float, y: float, modifiers: str) -> None:
        self._step_call(lambda: self._action_handler.execute_click(x, y, kind, modifiers))

    def observe(self) -> tuple[dict[str, Any], np.ndarray]:
        viewer_json, image = self._step_call(self._gather)
        self._consecutive_step_failures = 0  # a clean observation ends the streak
        self._viewer_json = viewer_json
        return ngl_state_from_viewer(viewer_json), image

    # =========================================================================
    # Internal: step-time failure accounting
    # =========================================================================

    def _step_call(self, fn: Callable[[], Any]) -> Any:
        """Run one Playwright interaction under the step watchdog.

        Transient viewer-state races are common under GPU contention (a slow
        render not settled when we read state). A SINGLE failure just truncates
        the episode; only a browser that fails repeatedly is genuinely sick, so
        escalate to a full restart only after N consecutive failures.
        (Escalating on every miss caused a restart-storm that tanked throughput
        at high M — 2026-07-09.)
        """
        wd = self._watchdog(self.step_timeout_s)
        try:
            return fn()
        except BrowserError as e:
            self._note_step_failure(str(e)[:140])
            if self.recovery_mode == "escalate" and (
                    self._consecutive_step_failures >= self.restart_after_consecutive_failures):
                # 'in_place' skips the full-browser-restart escalation: the browser
                # is still alive (a watchdog KILL takes the wd.fired path below and
                # forces a restart regardless), so the cheap per-reset context
                # recycle (fresh_context_every_reset) recovers without a cold
                # relaunch storm.
                self._needs_browser_restart = True
            raise
        except Exception as e:
            if wd.fired:
                self._note_step_failure(
                    f"watchdog hang >{self.step_timeout_s}s: {str(e)[:100]}")
                if self._consecutive_step_failures >= self.restart_after_consecutive_failures:
                    self._needs_browser_restart = True
                raise BrowserError(
                    f"step hung >{self.step_timeout_s}s; browser killed by watchdog: {e}"
                ) from e
            raise
        finally:
            wd.cancel()

    def _note_step_failure(self, signature: str) -> None:
        self._consecutive_step_failures += 1
        self.events.emit("glitch", phase="step", consecutive=self._consecutive_step_failures,
                         settle_polls=self._last_settle_polls, signature=signature)

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
        from playwright.sync_api import sync_playwright

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

        launch_wd = _BrowserWatchdog(120.0, _kill_new_children)
        try:
            kids_before = {c.pid for c in psutil.Process(os.getpid()).children()}
            self._playwright = sync_playwright().start()
            # The driver (a direct node child of this process) mediates every
            # sync call; killing it unblocks ANY hung Playwright call — the
            # watchdog fallback for phases where no Chrome pid exists (launch/
            # relaunch), which previously hung unguarded (2026-08-17).
            try:
                self._driver_pid = next(
                    (c.pid for c in psutil.Process(os.getpid()).children()
                     if c.pid not in kids_before), None)
            except Exception:
                self._driver_pid = None
            W, H = self.layout.window_size
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
            launch_wd.cancel()

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
            pass
        # ALWAYS also kill the driver: the sync client blocks on the DRIVER
        # pipe, and a wedged driver (e.g. hung CDP send) never propagates the
        # browser's death — the blocked call then never raises (observed across
        # v1-v5 silent stalls, incl. after correctly-attributed tree-kills).
        # Driver death breaks the pipe -> guaranteed raise; recovery does a
        # full relaunch either way.
        self._kill_driver()

    def _kill_driver(self) -> None:
        """Last-resort unblocker: kill this env's Playwright node driver."""
        dpid = self._driver_pid
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
        armed = self._chrome_pid is not None or self._driver_pid is not None
        return _BrowserWatchdog(timeout_s if armed else None, self._kill_chrome)

    @staticmethod
    def _clear_cache(context, page) -> None:
        try:  # chromium-only CDP call; harmless to skip on failure
            cdp = context.new_cdp_session(page)
            cdp.send("Network.clearBrowserCache")
            cdp.detach()
        except Exception:
            pass

    def _recycle_context(self) -> None:
        """Fresh BrowserContext + Page (and HTTP-cache clear) for the episode.

        Page-level state (JS heap, WebGL contexts, caches) accumulates across
        episodes and steadily degrades step throughput on long runs; recycling
        the context every reset keeps it flat. The periodic full browser
        restart still handles browser-process-level leaks.
        """
        W, H = self.layout.window_size
        old = self.page.context if self.page is not None else None
        context = self.browser.new_context(viewport={"width": W, "height": H})
        page = context.new_page()
        if self.clear_cache_on_recycle:
            self._clear_cache(context, page)
        self._swap_page(page, old)

    def _swap_page(self, page, old_context) -> None:
        self.page = page
        self._action_handler = MouseActionHandler(page)
        if old_context is not None:
            try:
                old_context.close()
            except Exception:
                pass

    # =========================================================================
    # Internal: warm context (reset-ahead)
    # =========================================================================

    def _discard_warm(self) -> None:
        w = self._warm
        self._warm = None
        if w is not None:
            try:
                w["context"].close()
            except Exception:
                pass

    def _warm_ready(self) -> bool:
        """Has the warm page finished loading? Readiness is monotonic, so one
        check at reset time sees everything a per-step poll would have."""
        w = self._warm
        wd = self._watchdog(10.0)
        try:
            page = w["page"]
            raw = page.evaluate(
                "() => (window.viewer && window.viewer.state) ? "
                "JSON.stringify(window.viewer.state) : null"
            )
            ready = raw is not None and bool(page.evaluate(
                "() => !!(window.viewer && window.viewer.isReady && "
                "window.viewer.isReady())"
            ))
            if ready:
                self.events.emit("warm_ready", prep_ms=(time.monotonic() - w["t0"]) * 1000.0)
            return ready
        except Exception:
            return False  # still loading, or the context died
        finally:
            wd.cancel()

    def _adopt_warm(self) -> None:
        """Swap the ready warm context in as the active page."""
        w = self._warm
        self._warm = None
        old = self.page.context if self.page is not None else None
        self._swap_page(w["page"], old)
        self._last_nav_attempts = 0

    # =========================================================================
    # Internal: navigation
    # =========================================================================

    def _navigate_with_retry(self, state) -> None:
        """Navigate with retry per self-healing config. Restarts the browser between attempts."""
        last_err: Exception | None = None
        for attempt in range(max(1, self.retry_on_reset + 1)):
            try:
                self._navigate(state)
                self._last_nav_attempts = attempt + 1
                return
            except BrowserError:
                # Launch failures aren't retryable through restart.
                raise
            except Exception as e:
                last_err = e
                self._last_nav_attempts = attempt + 1
                self.events.emit("glitch", phase="reset", attempt=attempt + 1,
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

    def _navigate(self, state) -> None:
        """Resolve `state` to a URL, navigate, and wait for viewer ready.

        Guarded by the reset watchdog: a hung navigation gets its Chrome
        killed, raising here as a retryable error so `_navigate_with_retry`
        restarts the browser and tries again (NOT BrowserError — that aborts
        the retry loop by design).
        """
        wd = self._watchdog(self.reset_timeout_s)
        try:
            self._navigate_inner(state)
        except Exception as e:
            if wd.fired:
                raise RuntimeError(
                    f"navigation hung >{self.reset_timeout_s}s; browser killed by watchdog: {e}"
                ) from e
            raise
        finally:
            wd.cancel()

    def _state_to_url(self, state: dict[str, Any] | str) -> str:
        """A URL string is navigated to as-is; an NglState is overlaid onto the
        start URL's parsed state. Use `state["extra"]` for fields that are not
        first-class."""
        if isinstance(state, str):
            return state
        return state_to_url(self._url_prefix, merge_state(self._base_state, state))

    def _navigate_inner(self, state) -> None:
        url = self._state_to_url(state)
        logger.debug("navigating to %s%s", url[:120], "..." if len(url) > 120 else "")
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

    # =========================================================================
    # Internal: observation
    # =========================================================================

    def _gather(self) -> tuple[dict[str, Any], np.ndarray]:
        json_state = self._get_json_state()
        # The viewer JSON transiently omits fields mid-update (the render hasn't
        # settled when we read state). This scales with GPU contention (more
        # browsers/GPU = slower renders = more misses), so poll up to
        # state_ready_timeout_s for the fields to appear before giving up —
        # resolving the race in-place is far cheaper than truncating the episode.
        poll = 0.05
        polls = 0
        deadline = time.monotonic() + self.state_ready_timeout_s
        while not all(k in json_state for k in REQUIRED_FIELDS):
            if time.monotonic() >= deadline:
                self._last_settle_polls = polls
                missing = sorted(set(REQUIRED_FIELDS) - set(json_state))
                raise BrowserError(f"viewer state missing fields after retries: {missing}")
            time.sleep(poll)
            polls += 1
            json_state = self._get_json_state()
        self._last_settle_polls = polls
        image = self._crop_panes(self._get_screenshot())
        return json_state, image

    def _crop_panes(self, image: np.ndarray) -> np.ndarray:
        lay = self.layout
        if lay.left_pane and lay.right_pane:
            return image
        mid = image.shape[1] // 2
        return image[:, mid:] if lay.right_pane else image[:, :mid]

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
            cause = self._last_state_read_error
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
        if self.layout.capture_scale != 1.0:
            # Browser-side downscale: the compositor scales on the GPU and ships
            # capture_scale^2 x fewer pixels — cutting the GIL-held Python
            # decode/resize per step, which is the aggregate throughput cost.
            W, H = self.layout.window_size
            fmt = "jpeg" if self.screenshot_format == "jpeg" else "png"
            params: dict[str, Any] = {
                "format": fmt,
                "clip": {"x": 0, "y": 0, "width": W, "height": H,
                         "scale": self.layout.capture_scale},
            }
            if fmt == "jpeg":
                params["quality"] = 85
            res = self._capture_cdp_session().send("Page.captureScreenshot", params)
            data = base64.b64decode(res["data"])
            pil = Image.open(io.BytesIO(data)).convert("RGB")
            # CDP's scaled clip can be off-by-one vs round(); normalize so the
            # observation shape is exact.
            if pil.size != self.layout.capture_size:
                pil = pil.resize(self.layout.capture_size)
            return np.asarray(pil)
        if self.screenshot_format == "jpeg":
            data = self.page.screenshot(type="jpeg", quality=85)
        else:
            data = self.page.screenshot()
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        return np.asarray(pil)
