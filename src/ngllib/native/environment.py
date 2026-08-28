"""Browser-free drop-in for `ngllib.Environment` (native renderer).

Same Gymnasium contract as the browser env — Dict observation
{position, xs_scale, orientation, proj_scale, image}, Dict action
(action_type 1 = right_click -> move-to-mouse-position, 3 = edit_state),
provider/reward/termination hooks, info {task_info, json_state} — but the
frame comes from CloudVolume + moderngl/EGL instead of Playwright + Chrome.

Parity provenance (2026-08-27/28 campaign, 300 states / 320 input probes vs
a browser baseline that re-renders pixel-identically):
- 3D pane block-SSIM 0.845 median; mesh tol-IoU 0.885.
- 2D pane block-SSIM 0.898 median, registration pixel-exact (jitter sd 0).
- rotate/zoom state edits bit-exact (identical arithmetic, 0.00 error).
- clicks: ~2 screen px lateral; along-view residual traced to NG's
  per-chunk LOD pick mesh (we pick against the full-res mesh + section
  plane).
Calibrated constants below (SCALE_CAL_NM, EM_GAIN, LEFT_SHIFT_PX) come from
that campaign — change them only with a parity-harness re-run.

Geometry is fixed at the calibrated capture: window 1800x900 CSS at
capture_scale 0.5 -> two 450x450 panes, 17 captured px of toolbar.
"""

from __future__ import annotations

import copy
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, Callable, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image

from ..errors import ProviderError
from ..providers import StateProvider
from ..utils.geom import euler_to_quaternion, quaternion_to_euler
from .colors import segment_color
from .em import MeshStore, worker_label_tile, worker_mesh, worker_tile
from .render3d import MeshRenderer

logger = logging.getLogger(__name__)

VOXEL_NM = np.array([4.0, 4.0, 40.0])
# Calibrated: nm per projectionScale unit (grid + golden-section refine on
# browser pairs; ~= the 4nm xy voxel, refined empirically).
SCALE_CAL_NM = 4.07
# Calibrated: browser/native EM intensity ratio on grey 2D-pane pixels
# (median of per-state ratios; NG's image-layer opacity-0.5 default does NOT
# halve on-screen EM).
EM_GAIN = 0.978
# Calibrated: baked 2D-pane fetch-center correction in captured px (dy, dx).
# Raw registration medians were (-6, 0); the correction transfer gain is -2
# (baking c changes the residual by -2c), so the nulling correction is half.
LEFT_SHIFT_PX = (-3.0, 0.0)

# Capture geometry at capture_scale 0.5 (the calibrated configuration).
PANE = 450          # each pane, captured px
TOOLBAR = 17        # NG top toolbar (~33 CSS px) at capture scale 0.5
PANE_H = PANE - TOOLBAR
CSS_PANE = 900.0    # each pane, CSS px
CSS_TOOLBAR = 33.0
CSS_VIEW_H = 867.0  # pane CSS height below the toolbar

_noop_reward_factory = lambda task_info: (  # noqa: E731
    lambda obs, action, prev_obs, terminated: 0.0)
_noop_termination_factory = lambda task_info: (  # noqa: E731
    lambda obs, action, prev_obs: False)


class NativeEnvironment(gym.Env):
    """Native (browser-free) Neuroglancer-equivalent environment.

    Accepts the browser `Environment`'s core kwargs; browser-lifecycle
    options (self-healing timeouts, restart cadence, launch args) have no
    native counterpart and are intentionally not part of this signature —
    `ngllib_agent.env_build` only forwards them to the browser backend.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        window_size: tuple[int, int] = (1800, 900),
        image_size: tuple[int, int] | None = None,
        left_pane: bool = False,
        right_pane: bool = True,
        capture_scale: float = 0.5,
        orientation: Literal["quaternion", "euler"] = "quaternion",
        reset_state_provider: StateProvider | None = None,
        reward_factory: Callable | None = None,
        termination_factory: Callable | None = None,
        cache_dir: str | None = None,
        mesh_budget_bytes: int = 2 << 30,
        reset_ahead: bool = True,
    ):
        super().__init__()
        if tuple(window_size) != (1800, 900) or capture_scale != 0.5:
            raise ValueError(
                "NativeEnvironment is calibrated for window_size=(1800, 900) "
                f"at capture_scale=0.5; got {window_size} @ {capture_scale}")
        if not (left_pane or right_pane):
            raise ValueError("At least one of `left_pane`/`right_pane` must be True.")
        if orientation not in ("quaternion", "euler"):
            raise ValueError(f"orientation must be 'quaternion' or 'euler'; got {orientation!r}")

        self.window_size = tuple(window_size)
        self.image_size = tuple(image_size) if image_size else None
        self.left_pane = left_pane
        self.right_pane = right_pane
        self.capture_scale = capture_scale
        self.orientation = orientation
        # None = no CloudVolume disk cache (default; NFS caches tax cold
        # fetches 4-30x — see em.py). Pass a LOCAL-disk dir to opt in.
        self._cache_dir = cache_dir

        self._reset_state_provider = reset_state_provider
        self._reward_factory = reward_factory or _noop_reward_factory
        self._termination_factory = termination_factory or _noop_termination_factory

        self._image_shape = self._compute_image_shape()
        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()

        # GL/data backends are lazy (first reset) so construction stays cheap
        # and importable off-GPU, mirroring the browser env's lazy launch.
        self._renderer: MeshRenderer | None = None
        self._meshes: MeshStore | None = None
        self._mesh_budget = mesh_budget_bytes

        self._rng: np.random.Generator = np.random.default_rng()
        self._json_state: dict[str, Any] | None = None
        self._prev_obs: dict[str, Any] | None = None
        self._task_info: dict[str, Any] = {}
        self._reward_fn: Callable | None = None
        self._terminated_fn: Callable | None = None
        # Tile pipeline: `_tiles` is the last COMPLETED tile set (memo-keyed);
        # `_pending` is an in-flight fetch group. reset() blocks on it (the
        # first observation is exact); step() adopts it when done and renders
        # with the previous tiles meanwhile — Neuroglancer-equivalent
        # semantics (Chrome renders whatever chunks are loaded; a frame after
        # a move shows the previous slice until streaming catches up). Cold
        # chunk fetches are 3-7s serial from S3; blocking per step on that
        # was the density-probe wall.
        self._tile_key = None
        self._tiles: dict[str, Any] = {}
        self._pending: tuple | None = None  # (key, {name: Future}, ext)
        # Reset-ahead prefetch (native analog of the browser env's M5):
        # measured 38s reset tail = mesh download/decode/normals + cold
        # tiles for the NEXT episode's neuron — all prefetchable during the
        # current episode. The provider draw for the next episode is taken
        # from the SAME rng stream at the same point it would be inline, so
        # sampling is unchanged (the wall-clock curriculum sees it ~one
        # episode early — negligible). seed/options resets discard it.
        self.reset_ahead = reset_ahead
        self._prefetch: dict[str, Any] | None = None

    # ------------------------------------------------------------------ spaces

    def _compute_image_shape(self) -> tuple[int, int, int]:
        if self.image_size is not None:
            iw, ih = self.image_size
            return (ih, iw, 3)
        if self.left_pane and self.right_pane:
            return (PANE, 2 * PANE, 3)
        return (PANE, PANE, 3)

    def _build_observation_space(self) -> spaces.Dict:
        orient_dim = 3 if self.orientation == "euler" else 4
        return spaces.Dict({
            "position": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "xs_scale": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "orientation": spaces.Box(low=-np.inf, high=np.inf, shape=(orient_dim,), dtype=np.float32),
            "proj_scale": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "image": spaces.Box(low=0, high=255, shape=self._image_shape, dtype=np.uint8),
        })

    def _build_action_space(self) -> spaces.Dict:
        W, H = self.window_size
        orient_dim = 3 if self.orientation == "euler" else 4
        return spaces.Dict({
            "action_type": spaces.Discrete(4),
            "mouse_xy": spaces.Box(
                low=np.array([0, 0], dtype=np.float32),
                high=np.array([W, H], dtype=np.float32), dtype=np.float32),
            "modifiers": spaces.MultiBinary(3),
            "delta_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "delta_xs_scale": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "delta_orient": spaces.Box(low=-np.inf, high=np.inf, shape=(orient_dim,), dtype=np.float32),
            "delta_proj_scale": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        })

    # ------------------------------------------------------------------ gym API

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}
        self._ensure_backends()

        pf = None
        if (self.reset_ahead and self._prefetch is not None
                and seed is None and not options):
            pf = self._prefetch  # adopt: state pre-drawn from the same rng stream
        self._prefetch = None
        if pf is not None:
            start_state, task_info = pf["state"], pf["task_info"]
        else:
            start_state, task_info = self._resolve_reset_state(options)
        if start_state is None or not isinstance(start_state, dict):
            raise ValueError(
                "NativeEnvironment requires an explicit NglState dict from the "
                "provider or reset options (no default-URL fallback).")
        try:
            self._reward_fn = self._reward_factory(task_info)
            self._terminated_fn = self._termination_factory(task_info)
        except Exception as e:
            raise ProviderError(f"factory raised at reset: {e}") from e
        self._task_info = task_info

        st = copy.deepcopy(start_state)
        st.setdefault("projectionOrientation", [0.0, 0.0, 0.0, 1.0])
        for k in ("position", "projectionScale", "crossSectionScale", "segments"):
            if k not in st:
                raise ValueError(f"reset state missing required field {k!r}")
        self._json_state = st
        self._tile_key = None

        rid = str(st["segments"][0])
        if not self._renderer.has_mesh(rid):
            v = vn = f = None
            if pf is not None and pf.get("mesh_fut") is not None:
                try:
                    v, vn, f = pf["mesh_fut"].result(timeout=240)
                except Exception as e:
                    logger.warning("prefetched mesh failed (%s); inline fetch", e)
                    v = None
            if v is None:
                v, f = self._meshes.get(rid)
                vn = None
            self._renderer.load_mesh(rid, v, f, normals=vn)
        if pf is not None:
            # Adopt the prefetched tile group; the blocking gather below
            # resolves it (usually already done).
            self._pending = pf["tiles"]

        obs = self._gather_observation(block_tiles=True)
        self._prev_obs = obs
        # Only provider-driven resets prefetch: an explicit-state reset
        # (eval) must not consume provider rng draws for episodes that will
        # never run.
        if self.reset_ahead and seed is None and not options:
            self._schedule_prefetch()
        return obs, {"task_info": task_info, "json_state": copy.deepcopy(st), "step": 0}

    def step(self, action):
        action_type = int(action["action_type"])
        if action_type == 1:
            self._apply_click(action)
        elif action_type == 3:
            self._apply_state_edit(action)
        else:
            raise NotImplementedError(
                f"NativeEnvironment supports right_click (1) and edit_state (3); "
                f"got action_type={action_type}")

        obs = self._gather_observation()
        try:
            terminated = bool(self._terminated_fn(obs, action, self._prev_obs))
        except Exception as e:
            raise ProviderError(f"termination_function raised: {e}") from e
        try:
            reward = float(self._reward_fn(obs, action, self._prev_obs, terminated))
        except Exception as e:
            raise ProviderError(f"reward_function raised: {e}") from e

        info = {"task_info": self._task_info,
                "json_state": copy.deepcopy(self._json_state)}
        self._prev_obs = obs
        return obs, reward, terminated, False, info

    def close(self):
        # The class-level tile pool outlives individual envs on purpose
        # (shared by the process's env fleet; reaped at interpreter exit).
        self._pending = None
        self._prefetch = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._meshes = None

    # ------------------------------------------------------------------ internals

    # One fetch pool per PROCESS, shared by all envs in it (a runner hosts
    # 16 threaded envs): GIL-isolates chunk download/decode and dedupes the
    # workers' chunk LRUs across envs. Spawn context — fork would inherit
    # CUDA/EGL state.
    _TILE_POOL: ProcessPoolExecutor | None = None

    @classmethod
    def _tile_pool(cls) -> ProcessPoolExecutor:
        if cls._TILE_POOL is None:
            # Tune down (NGL_NATIVE_FETCH_WORKERS) when many runner
            # processes share a node — fetch workers multiply per runner.
            cls._TILE_POOL = ProcessPoolExecutor(
                max_workers=int(os.environ.get(
                    "NGL_NATIVE_FETCH_WORKERS", "6")),
                mp_context=multiprocessing.get_context("spawn"))
        return cls._TILE_POOL

    def _ensure_backends(self) -> None:
        if self._renderer is None:
            self._renderer = MeshRenderer(PANE, PANE_H, self._mesh_budget)
            logger.info("native renderer GL: %s",
                        self._renderer.ctx.info["GL_RENDERER"])
        if self._meshes is None:
            self._meshes = MeshStore(self._cache_dir)

    def _resolve_reset_state(self, options: dict[str, Any]):
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
                return self._reset_state_provider(self._rng, options)
            except Exception as e:
                raise ProviderError(f"reset_state_provider raised: {e}") from e
        return None, {}

    # ---- actions

    def _apply_click(self, action: dict[str, Any]) -> None:
        """NG right-click = move-to-mouse-position; background = no-op."""
        x_css, y_css = (float(v) for v in action["mouse_xy"])
        st = self._json_state
        if x_css >= CSS_PANE:
            self._click_3d(x_css, y_css)
        else:
            # 2D xy slice: orthographic — clicked point maps linearly to the
            # z-plane at crossSectionScale canonical (4nm) units per CSS px.
            if y_css < CSS_TOOLBAR:
                return
            xs = float(st["crossSectionScale"])
            cx_css, cy_css = CSS_PANE / 2.0, CSS_TOOLBAR + CSS_VIEW_H / 2.0
            st["position"][0] += (x_css - cx_css) * xs
            st["position"][1] += (y_css - cy_css) * xs

    def _click_3d(self, x_css: float, y_css: float) -> None:
        st = self._json_state
        pos_nm = np.asarray(st["position"], dtype=np.float64) * VOXEL_NM
        quat = st["projectionOrientation"]
        zoom_nm = float(st["projectionScale"]) * SCALE_CAL_NM
        ext = self._pane_extents_nm()
        depth, view, proj = self._renderer.pick_depth(
            str(st["segments"][0]), pos_nm, quat, zoom_nm, plane_extent_nm=ext)
        fx = x_css * self.capture_scale - PANE
        fy = y_css * self.capture_scale - TOOLBAR
        ix, iy = int(round(fx)), int(round(fy))
        if not (0 <= ix < PANE and 0 <= iy < PANE_H):
            return
        d = depth[iy, ix]
        px, py = ix, iy
        if d >= 0.9999:
            # NG issues the pick over a small radius; take the front-most hit
            # in a 3px window (matches the browser-validated harness).
            y0, y1 = max(0, iy - 3), min(PANE_H, iy + 4)
            x0, x1 = max(0, ix - 3), min(PANE, ix + 4)
            win = depth[y0:y1, x0:x1]
            if not (win < 0.9999).any():
                return  # background: no-op
            yy, xx = np.unravel_index(np.argmin(win), win.shape)
            d, px, py = win[yy, xx], x0 + xx, y0 + yy
        ndc = np.array([2 * (px + 0.5) / PANE - 1,
                        1 - 2 * (py + 0.5) / PANE_H,
                        2 * d - 1, 1.0])
        w = np.linalg.inv(proj @ view) @ ndc
        st["position"] = [float(v) for v in (w[:3] / w[3]) / VOXEL_NM]

    def _apply_state_edit(self, action: dict[str, Any]) -> None:
        """Bit-exact port of the browser env's `_apply_state_edit`."""
        new_state = self._json_state
        dpos = action["delta_pos"]
        new_state["position"][0] += float(dpos[0])
        new_state["position"][1] += float(dpos[1])
        new_state["position"][2] += float(dpos[2])
        new_state["crossSectionScale"] += float(action["delta_xs_scale"][0])

        d = action["delta_orient"]
        if self.orientation == "euler":
            old_euler = quaternion_to_euler(new_state["projectionOrientation"])
            new_state["projectionOrientation"] = euler_to_quaternion([
                old_euler[0] + float(d[0]),
                old_euler[1] + float(d[1]),
                old_euler[2] + float(d[2]),
            ])
        else:
            for i in range(4):
                new_state["projectionOrientation"][i] += float(d[i])

        # Floor at 1.0 (deviation from the browser arithmetic ONLY at the
        # degenerate point): repeated zoom-ins can hit projectionScale == 0
        # exactly (14000 - 7x2000), which makes the camera matrix singular
        # here — Chrome just renders garbage and carries on.
        new_state["projectionScale"] = max(1.0, min(
            500_000,
            new_state["projectionScale"] + float(action["delta_proj_scale"][0]),
        ))

    # ---- observation

    def _pane_extents_nm(self) -> tuple[float, float]:
        xs = float(self._json_state["crossSectionScale"])
        return xs * CSS_PANE * 4.0, xs * CSS_VIEW_H * 4.0

    def _tile_key_for(self, pos, xs, rid):
        return (round(pos[0], 2), round(pos[1], 2), round(pos[2], 2),
                round(float(xs), 5), rid, self.left_pane)

    def _tile_state_key(self):
        st = self._json_state
        return self._tile_key_for(st["position"], st["crossSectionScale"],
                                  str(st["segments"][0]))

    def _submit_tile_group(self, pos, xs, rid) -> tuple:
        """(key, futs, ext) for an arbitrary state — used both for the
        current state and for reset-ahead prefetch."""
        pos_nm = np.asarray(pos, dtype=np.float64) * VOXEL_NM
        ext = (float(xs) * CSS_PANE * 4.0, float(xs) * CSS_VIEW_H * 4.0)
        pool = self._tile_pool()
        cd = self._cache_dir
        futs = {"plane": pool.submit(
            worker_tile, cd, pos_nm, ext[0], ext[1], 1024, False)}
        if self.left_pane:
            # Baked registration correction: move the fetch center by the
            # calibrated (dy, dx) captured px, in nm.
            shifted = pos_nm + np.array([
                LEFT_SHIFT_PX[1] * ext[0] / PANE,
                LEFT_SHIFT_PX[0] * ext[1] / PANE_H, 0.0])
            futs["left"] = pool.submit(
                worker_tile, cd, shifted, ext[0], ext[1], 1024, True)
            futs["label"] = pool.submit(
                worker_label_tile, cd, shifted, ext[0], ext[1], rid,
                (PANE, PANE_H))
        return (self._tile_key_for(pos, xs, rid), futs, ext)

    def _submit_tile_fetch(self, key):
        st = self._json_state
        self._pending = self._submit_tile_group(
            st["position"], st["crossSectionScale"], str(st["segments"][0]))

    def _schedule_prefetch(self) -> None:
        """Pre-draw the next episode and start its mesh/tile fetches."""
        if self._reset_state_provider is None or self._prefetch is not None:
            return
        try:
            state, task_info = self._reset_state_provider(self._rng, {})
        except Exception as e:
            logger.warning("reset-ahead pre-sample failed (%s)", e)
            return
        try:
            rid = str(state["segments"][0])
            mesh_fut = (None if self._renderer.has_mesh(rid)
                        else self._tile_pool().submit(
                            worker_mesh, self._cache_dir, rid))
            tiles = self._submit_tile_group(
                state["position"], state["crossSectionScale"], rid)
        except Exception as e:
            logger.warning("reset-ahead prefetch submit failed (%s)", e)
            mesh_fut, tiles = None, None
        if tiles is None:
            return
        self._prefetch = {"state": state, "task_info": task_info,
                          "mesh_fut": mesh_fut, "tiles": tiles}

    def _adopt_pending(self, timeout_s: float = 180.0) -> None:
        key, futs, ext = self._pending
        tiles: dict[str, Any] = {"ext": ext, "plane": None, "left": None,
                                 "label": None}
        for name, fut in futs.items():
            try:
                tiles[name] = fut.result(timeout=timeout_s)
            except FuturesTimeout:
                logger.warning("EM %s tile fetch timed out; skipped", name)
            except Exception as e:
                logger.warning("EM %s tile fetch failed (%s); skipped", name, e)
        self._tile_key, self._tiles = key, tiles
        self._pending = None

    def _fetch_tiles(self, block: bool) -> dict[str, Any]:
        """Tiles for the current state. block=True (reset) waits for exact
        tiles; block=False (step) returns the last completed set while the
        fetch streams in — see the pipeline note in __init__."""
        key = self._tile_state_key()
        if key == self._tile_key:
            return self._tiles
        if self._pending is not None and self._pending[0] != key:
            # Superseded in-flight fetch: adopt it only if it already
            # finished (warms the LRU either way), then refetch.
            if all(f.done() for f in self._pending[1].values()):
                self._adopt_pending()
                if key == self._tile_key:
                    return self._tiles
            elif block:
                self._adopt_pending()  # drain before the exact fetch
            else:
                return self._tiles  # keep rendering stale; let it land
        if self._pending is None:
            self._submit_tile_fetch(key)
        if block or all(f.done() for f in self._pending[1].values()):
            self._adopt_pending()
        return self._tiles

    def _render_left(self, tiles: dict[str, Any]) -> np.ndarray:
        """2D xy EM pane: browser-matched filter chain + tint + crosshair.
        Memoized on the tile set — the pane only changes when tiles do, and
        the PIL chain is the priciest per-step CPU otherwise."""
        if "left_canvas" in tiles:
            return tiles["left_canvas"]
        canvas = np.zeros((PANE, PANE, 3), dtype=np.uint8)
        if tiles["left"] is None:
            tiles["left_canvas"] = canvas
            return canvas
        # Filter chain (calibrated): subpixel-phase tile -> GL-linear resample
        # to the 900x867 CSS pane -> Chrome's area-average capture downscale.
        big = Image.fromarray(tiles["left"]).resize((900, 867), Image.BILINEAR)
        img = np.asarray(big.resize((PANE, PANE_H), Image.BOX)
                         ).astype(np.float32) * EM_GAIN
        rgb = np.repeat(img[..., None], 3, axis=2)
        if tiles["label"] is not None:
            col = np.asarray(
                segment_color(int(self._json_state["segments"][0]))) * 255.0
            # NG segmentation 2D: selectedAlpha 0.5 over the image.
            rgb[tiles["label"]] = 0.5 * col[None, :] + 0.5 * rgb[tiles["label"]]
        # One-sided crosshair (red +x, green +y), alpha 0.5.
        cy, cx = PANE_H // 2, PANE // 2
        length = int(min(900, 867) / 4 / 2)
        row = rgb[cy, cx:cx + length]
        rgb[cy, cx:cx + length] = 0.5 * np.array([255, 0, 0]) + 0.5 * row
        colm = rgb[cy:cy + length, cx]
        rgb[cy:cy + length, cx] = 0.5 * np.array([0, 255, 0]) + 0.5 * colm
        canvas[TOOLBAR:] = np.clip(rgb, 0, 255).astype(np.uint8)
        tiles["left_canvas"] = canvas
        return canvas

    def _render_right(self, tiles: dict[str, Any]) -> np.ndarray:
        st = self._json_state
        rid = str(st["segments"][0])
        pos_nm = np.asarray(st["position"], dtype=np.float64) * VOXEL_NM
        pane = self._renderer.render(
            rid, pos_nm, st["projectionOrientation"],
            float(st["projectionScale"]) * SCALE_CAL_NM,
            segment_color(int(rid)),
            em_tile=tiles["plane"], em_extent_nm=tiles["ext"],
            em_gain=EM_GAIN)
        out = np.zeros((PANE, PANE, 3), dtype=np.uint8)
        out[TOOLBAR:] = pane
        return out

    def _gather_observation(self, block_tiles: bool = False) -> dict[str, Any]:
        st = self._json_state
        tiles = self._fetch_tiles(block=block_tiles)
        panes = []
        if self.left_pane:
            panes.append(self._render_left(tiles))
        if self.right_pane:
            panes.append(self._render_right(tiles))
        image = panes[0] if len(panes) == 1 else np.concatenate(panes, axis=1)
        if self.image_size is not None:
            image = np.asarray(Image.fromarray(image).resize(self.image_size))

        orient_raw = st["projectionOrientation"]
        if self.orientation == "euler":
            orient = np.asarray(quaternion_to_euler(orient_raw), dtype=np.float32)
        else:
            orient = np.asarray(orient_raw, dtype=np.float32)
        return {
            "position": np.asarray(st["position"], dtype=np.float32),
            "xs_scale": np.asarray([st["crossSectionScale"]], dtype=np.float32),
            "orientation": orient,
            "proj_scale": np.asarray([st["projectionScale"]], dtype=np.float32),
            "image": image,
        }
