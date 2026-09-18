"""Simulator renderer: CloudVolume + moderngl/EGL, no browser.

Same panes as Chrome -- a 2D xy EM slice on the left, the 3D projection on
the right -- produced from the data directly instead of through Neuroglancer.
The environment owns the state; this class holds the copy it renders from,
answers what Neuroglancer would do to a click (`pick`, `select`), and streams
tiles and meshes the way Chrome streams chunks.

Parity provenance (2026-08-27..09-10 campaigns, against a Chrome baseline that
reproduces itself exactly on settled frames):
- input parity 72/72 (same state + same click pixel -> same selection, both
  panes); rotate/zoom state edits bit-exact.
- 2D pane block-SSIM 0.970 interior / 0.899 whole; 3D pane 0.836 whole,
  0.40 over content blocks (NG's per-chunk mesh LOD, not implemented here);
  mesh silhouette IoU 0.58, section plane IoU 0.75.
Calibrated constants live in pane2d with the measurement that fixes each.

Geometry is calibrated at window 1800x900 CSS, capture_scale 0.5 -> two
450x450 panes with 17 captured px of toolbar. Other geometries render but warn.

Operational knobs stay environment variables (they are per-process or per-node
budgets set by launchers, not per-env choices): NGL_NATIVE_FETCH_WORKERS,
NGL_NATIVE_MESH_LRU_MB, NGL_NATIVE_FINE_MAX_PX, NGL_NATIVE_COARSE_MAX_PX,
NGL_NATIVE_WARM_FACTOR, NGL_NATIVE_PARALLEL_PARTS, NGL_NATIVE_MESH_LAG_STEPS.
"""

from __future__ import annotations

import copy
import logging
import multiprocessing
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, Literal

import numpy as np

from .. import state as S
from ..dataset import DatasetSpec, default_start_url, ngl_state_from_viewer, split_state_url
from ..events import EventLog
from ..renderer import PaneLayout
from .colors import segment_color
from .em import (
    MeshStore,
    Source,
    unpack_ids,
    worker_em_plane,
    worker_ids,
    worker_mesh,
    worker_pane_parts,
    worker_tile,
    worker_warm,
)
from . import pane2d as pane2d_mod
from .pane2d import (
    CALIBRATED_DATASET,
    CSS_PANE,
    EM_GAIN,
    PANE,
    PANE_H,
    PANEL_CX_CLICK,
    PANEL_CY_CLICK,
    PANEL_TOP_CLICK,
    SCALE_CAL_NM,
    TOOLBAR,
    compose_left_parts,
    pane_extents_nm,
    tint_plane,
)
from .render3d import MeshRenderer

logger = logging.getLogger(__name__)

PaneMode = Literal["atomic", "progressive", "concurrent", "random"]
PANE_MODES = ("atomic", "progressive", "concurrent", "random")


class SimulatorRenderer:
    """Browser-free backend for `ngllib.Environment`.

    `pane_mode` is the 2D-pane fill policy after a move -- the one controlled
    experiment this project has on pane dynamics (plan 7.2, four modes trained
    at pace740's config, k=5 on the 200-pair holdout):
      atomic      one fine fetch, swapped in when it lands; the PREVIOUS
                  location shows meanwhile. Default: 94.0% on Chrome.
      random      atomic plus a per-episode random adopt delay (0-4 steps),
                  i.e. domain randomization over fetch latency. 92.9%.
      concurrent  coarse and fine fetched together, as Neuroglancer does
                  (filterVisibleSources yields every scale). 92.3%.
      progressive coarse preview, then fine, sequentially. 81.5% at k=1:
                  the policy preferred a sharp stale pane to a blurry current
                  one. The mechanism is the DURATION of a degraded pane.
    Fidelity to Chrome's fill behaviour does not predict transfer, so the
    criterion for a mode is measured transfer, not resemblance.

    `cache_dir` opts into a CloudVolume DISK cache. Default None: on bucket
    NFS the cache's write-through metadata taxed every cold fetch 4-30x
    (2026-08-28); the in-RAM chunk LRU covers repeats. Pass a LOCAL-disk dir
    only. `mesh_budget_bytes` sizes the GPU mesh slot pool (None = the
    `NGL_NATIVE_VAO_LRU_MB` node knob, else 2 GB; see `MeshRenderer`).
    """

    # The simulator's warm work is a background prefetch; nothing is gained by
    # delaying it, so the environment warms right after reset.
    warm_after_steps = 0

    # One fetch pool per PROCESS, shared by all renderers in it (a runner hosts
    # many threaded envs): GIL-isolates chunk download/decode. Spawn context --
    # fork would inherit CUDA/EGL state.
    _TILE_POOLS: list | None = None
    _POOL_SEQ: int = 0

    # Coarse level requested first; MeshStore.get walks down to whatever the
    # segment actually has. NG streams meshes the same way.
    #
    # 1, not 2, even though 2 is coarser: the walk-down costs a failed
    # round-trip when a level is absent, and segments vary (measured ranges
    # 0..1 and 0..2). Requesting lod<=2 measured 0.52 s against lod<=1 at
    # 0.44 s -- asking for the coarsest level available anywhere is slower on
    # average than asking for one every segment has.
    MESH_COARSE_LOD = 1

    def __init__(
        self,
        *,
        window_size: tuple[int, int] = (1800, 900),
        capture_scale: float = 0.5,
        image_size: tuple[int, int] | None = None,
        left_pane: bool = False,
        right_pane: bool = True,
        cache_dir: str | None = None,
        mesh_budget_bytes: int | None = None,
        pane_mode: PaneMode = "atomic",
        dataset: DatasetSpec | None = None,
        config_path: str | None = None,
        cuda_ipc: bool = False,
    ):
        if pane_mode not in PANE_MODES:
            raise ValueError(f"`pane_mode` must be one of {PANE_MODES}; got {pane_mode!r}")
        # cuda_ipc (dino-server CUDA-IPC path): the RIGHT (3D GL) pane stays in
        # VRAM and observe() returns a reduce_tensor (rebuild, args) payload
        # instead of a numpy image. Right-pane-only (the 2D EM pane is CPU-composed,
        # so it cannot avoid the CPU bounce); left_pane must be False.
        self.cuda_ipc = bool(cuda_ipc)
        if self.cuda_ipc and left_pane:
            raise ValueError("cuda_ipc requires right-pane-only (left_pane=False)")
        self.layout = PaneLayout(
            window_size=window_size, capture_scale=capture_scale, image_size=image_size,
            left_pane=left_pane, right_pane=right_pane)
        self.layout.warn_if_uncalibrated("SimulatorRenderer")
        self.events = EventLog(path_template="")  # replaced by the environment
        self.pane_mode: str = pane_mode

        # Deployment identity: the same start URL Chrome navigates to. The
        # dataset is what the simulator reads; the URL's viewer state is the
        # default start state, exactly as for Chrome.
        self._url_prefix, self._base_state = split_state_url(default_start_url(config_path))
        self.dataset = dataset or DatasetSpec.from_state(self._base_state)
        if self.dataset != CALIBRATED_DATASET:
            warnings.warn(
                f"SimulatorRenderer: dataset {self.dataset} is not the one the parity "
                f"constants were fitted on ({CALIBRATED_DATASET}); it will render, but "
                "no parity claim holds.", stacklevel=2)
        self.source = Source(self.dataset, cache_dir)
        self._voxel_nm = self.source.voxel_nm
        self._canonical_nm = self.source.canonical_nm

        # GL/data backends are lazy (open()) so construction stays cheap and
        # importable off-GPU.
        self._renderer: MeshRenderer | None = None
        self._meshes: MeshStore | None = None
        self._mesh_budget = mesh_budget_bytes
        self._pick_em = None

        # Stable shard for this renderer, so its fetches keep landing on the
        # same worker and that worker's chunk LRU stays relevant to it.
        cls = type(self)
        self._pool_shard = cls._POOL_SEQ
        cls._POOL_SEQ += 1
        self._parallel_parts = os.environ.get("NGL_NATIVE_PARALLEL_PARTS", "1") != "0"
        # Stage resolutions. Defaults are the shipping values; both are levers
        # for an A/B the pane-mode campaign could not run, because mip
        # selection is DISCRETE and it only ever tested the extremes.
        # Measured against the browser frames (probe_mip_tradeoff):
        #
        #   max_px <=384   block_ssim 0.7394   sharpness 0.47x Chrome
        #   max_px 512-768            0.8730             0.75x
        #   max_px 1024               0.8877             0.86x
        #
        # 512 and 768 resolve to the SAME mip, as do 256 and 384 -- so there
        # are three operating points, not five.
        self._fine_px = int(os.environ.get("NGL_NATIVE_FINE_MAX_PX", "1024"))
        self._coarse_px = int(os.environ.get("NGL_NATIVE_COARSE_MAX_PX", "256"))
        # Background chunk-cache warming: costs bandwidth, changes no pixel.
        self._warm_factor = float(os.environ.get("NGL_NATIVE_WARM_FACTOR", "1.6"))
        # NGL_NATIVE_MESH_LAG_STEPS: hold a landed mesh until this many steps
        # after the selection. Default 0 -- show it as soon as it arrives.
        #
        # Mesh lag CANNOT be matched to Chrome in both units at once. Measured
        # (probe_select_dynamics 883469, probe_mesh_latency): our cold fetch is
        # 0.77 s median against Chrome's ~0.67 s end-to-end, so in WALL TIME
        # the two already agree -- but a simulator step costs 0.007 s against
        # Chrome's 0.031 s, so the same latency spans ~110 of our steps and
        # ~21 of Chrome's. Steps are what the policy experiences. Closing that
        # means either slowing the simulator down, which defeats its purpose,
        # or modelling the lag in step units, which is what this lever does.
        # Off by default: the pane-mode campaign found fidelity to Chrome's
        # streaming does not by itself predict transfer.
        self._mesh_lag = int(os.environ.get("NGL_NATIVE_MESH_LAG_STEPS", "0"))

        self._state: dict[str, Any] | None = None
        self._steps = 0
        self._block_next = False
        # Tile pipeline: `_tiles` is the last COMPLETED tile set (memo-keyed);
        # `_pending` is an in-flight fetch group. The first observation after
        # reset_to blocks on it (exact); a step observation adopts it when done
        # and renders with the previous tiles meanwhile -- Neuroglancer-
        # equivalent semantics (Chrome renders whatever chunks are loaded).
        self._tile_key = None
        self._tiles: dict[str, Any] = {}
        self._pending: tuple | None = None  # (key, {name: Fut}, ext, stage, t0)
        self._tile_stage: str = "fine"
        self._coarse_pending: tuple | None = None
        self._adopt_delay: int = 0   # random mode: steps still to wait
        self._warm_fut = None
        # root_id -> in-flight mesh fetch; a selected segment appears in the
        # 3D pane on whichever step its mesh lands, the way Chrome streams.
        self._mesh_futs: dict[str, Any] = {}
        self._mesh_due: dict[str, int] = {}
        self._mesh_t0: dict[str, tuple] = {}
        # Segments showing a COARSE mesh, still owed the full-resolution one.
        self._mesh_fine: set[str] = set()
        # Reset-ahead prefetch for the state the environment said comes next:
        # measured 38 s reset tail = mesh download/decode/normals + cold tiles,
        # all prefetchable during the current episode.
        self._prefetch: dict[str, Any] | None = None

    # ------------------------------------------------------------------ protocol

    def open(self) -> None:
        if self._renderer is None:
            self._renderer = MeshRenderer(PANE, PANE_H, self._mesh_budget,
                                          cuda_ipc=self.cuda_ipc)
            logger.info("simulator GL: %s", self._renderer.ctx.info["GL_RENDERER"])
        if self._meshes is None:
            self._meshes = MeshStore(self.source)

    def close(self) -> None:
        # The class-level tile pools outlive individual renderers on purpose
        # (shared by the process's env fleet; reaped at interpreter exit).
        self._pending = None
        self._prefetch = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._meshes = None

    def default_state(self) -> dict[str, Any]:
        return ngl_state_from_viewer(self._base_state)

    def warm(self, state: dict[str, Any]) -> None:
        """Start the next episode's mesh and tile fetches now."""
        if self._renderer is None:
            return
        try:
            rid = self._first_segment(state)
            mesh_fut = (None if rid is None or self._renderer.has_mesh(rid)
                        else self._tile_pool().submit(worker_mesh, self.source, rid))
            tiles = self._submit_tile_group(state["position"], state["crossSectionScale"])
        except Exception as e:
            logger.warning("reset-ahead prefetch submit failed (%s)", e)
            return
        self._prefetch = {"state": state, "mesh_fut": mesh_fut, "tiles": tiles}

    def reset_to(self, state: dict[str, Any] | str) -> None:
        if isinstance(state, str):
            state = ngl_state_from_viewer(split_state_url(state)[1])
        st = S.coerce(state)
        self.open()
        pf = self._prefetch
        self._prefetch = None
        if pf is not None and pf["state"] != st:
            pf = None  # warmed something else; pay the cold path
        self._state = st
        self._tile_key = None
        self._coarse_pending = None
        # Cancel, don't just drop: an abandoned mesh fetch would keep a pool
        # worker busy for the new episode's first seconds.
        for _lod, fut in self._mesh_futs.values():
            fut.cancel()
        self._mesh_futs.clear()
        self._mesh_due.clear()
        self._mesh_t0.clear()
        self._mesh_fine.clear()
        if self._warm_fut is not None:
            self._warm_fut.cancel()
            self._warm_fut = None
        self._steps = 0
        # RANDOM mode: one latency draw per episode, so an episode has a
        # consistent "network speed" rather than per-step jitter. Seeded from
        # the state itself, so a seeded eval replays the same draw for the
        # same episode without the renderer needing the environment's rng.
        self._adopt_delay = 0
        if self.pane_mode == "random":
            seed = hash((tuple(st["position"]), tuple(st["segments"]))) & 0xFFFFFFFF
            self._adopt_delay = int(np.random.default_rng(seed).integers(0, 5))

        rid = self._first_segment(st)
        if rid is not None and not self._renderer.has_mesh(rid):
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
        # A reset state may already carry several selected segments; only
        # the first one has a prefetched mesh.
        self._ensure_meshes(st["segments"], block=True)
        if pf is not None and pf.get("tiles") is not None:
            # Adopt the prefetched tile group; the blocking observe resolves
            # it (usually already done).
            self._pending = pf["tiles"]
        self._block_next = True

    def set_state(self, state: dict[str, Any]) -> None:
        self._state = S.coerce(state)

    def click(self, kind: str, x: float, y: float, modifiers: str) -> None:
        """NG bindings on a rendered data panel (default_input_event_bindings):
        `at:mousedown0` (left_click) -> a DRAG binding, so a click with no
        drag is a state no-op on both panes; `at:mousedown2` (right_click) ->
        move-to-mouse-position; `at:dblclick0` (double_click) -> select.
        Background = no-op. Modifiers are accepted for signature parity with
        Chrome and ignored: no default binding on these events uses them."""
        if kind == "left_click":
            return
        if y < PANEL_TOP_CLICK:
            return
        st = self._state
        if kind == "double_click":
            rid = (self._segment_under_3d(x, y) if x >= CSS_PANE
                   else self._segment_under_2d(x, y))
            if rid is not None:
                self._state = S.toggle_select(st, rid)
            return
        if kind != "right_click":
            raise ValueError(f"unknown click kind {kind!r}")
        if x >= CSS_PANE:
            hit = self._pick_3d_world(x, y)
            if hit is not None:
                self._state = S.move_to(st, hit)
        else:
            # 2D xy slice: orthographic -- clicked point maps linearly to the
            # z-plane at crossSectionScale canonical units per CSS px, about
            # the PANEL centre in click coordinates (see pane2d).
            xs = float(st["crossSectionScale"])
            self._state = S.move_to(st, [
                st["position"][0] + (x - PANEL_CX_CLICK) * xs,
                st["position"][1] + (y - PANEL_CY_CLICK) * xs,
                st["position"][2]])

    def observe(self) -> tuple[dict[str, Any], np.ndarray]:
        block = self._block_next
        self._block_next = False
        if not block:
            self._steps += 1
        return copy.deepcopy(self._state), self._render(block_tiles=block)

    @property
    def state(self) -> dict[str, Any] | None:
        """The state currently rendered from (a copy)."""
        return copy.deepcopy(self._state)

    @staticmethod
    def _first_segment(state: dict[str, Any]) -> str | None:
        """The segment whose mesh a reset needs first: the first VISIBLE one,
        else the first listed (hidden), else none."""
        vis = S.visible_segments(state["segments"])
        if vis:
            return vis[0]
        return str(state["segments"][0]).lstrip("!") if state["segments"] else None

    # ------------------------------------------------------------------ pools

    @classmethod
    def _pools(cls) -> list:
        """N single-worker pools, NOT one N-worker pool.

        The chunk LRU lives in the worker PROCESS, so which worker serves a
        fetch decides whether a move costs 0.03 s (chunks resident) or 0.20 s+
        (refetch). Sharing one N-worker pool scatters an env's consecutive
        fetches across all N, and the cache is useless. Measured with the verb
        sweep at n=8, 2D response to move-to-mouse-position:

          shared 6-worker pool   32.5 steps  (21-35)
          one worker, affine     5.0  steps  (5,5,5,5,6,5,5,5)

        Partitioning keeps the SAME total worker count -- no extra processes --
        while giving each env a stable worker, so its own chunk history is
        worth something. Envs sharing a shard still interleave, which is why
        this is a shard count and not a global switch.
        """
        if cls._TILE_POOLS is None:
            n = max(1, int(os.environ.get("NGL_NATIVE_FETCH_WORKERS", "6")))
            cls._TILE_POOLS = [
                ProcessPoolExecutor(
                    max_workers=1, mp_context=multiprocessing.get_context("spawn"))
                for _ in range(n)]
        return cls._TILE_POOLS

    def _tile_pool(self) -> ProcessPoolExecutor:
        pools = self._pools()
        return pools[self._pool_shard % len(pools)]

    # ------------------------------------------------------------------ picking

    def _segment_under_2d(self, x_css: float, y_css: float):
        """Segment id under a 2D-pane pixel via a point query on the
        segmentation volume, at the SAME mip the pane displays (label_tile
        uses extent/out_px) so a pick agrees with the tint that is drawn."""
        st = self._state
        xs = float(st["crossSectionScale"])
        world = [st["position"][0] + (x_css - PANEL_CX_CLICK) * xs,
                 st["position"][1] + (y_css - PANEL_CY_CLICK) * xs,
                 st["position"][2]]
        # NG picks against the slice it RENDERS, which is the CSS-resolution
        # pane (900 px wide), not the downscaled capture -- so the pick mip is
        # canonical*xs nm/px, one step finer than label_tile's capture-sized tile.
        res_nm = xs * self._canonical_nm
        return self._pick_em_tiles().segment_at(
            np.asarray(world, dtype=np.float64) * self._voxel_nm, res_nm)

    def _segment_under_3d(self, x_css: float, y_css: float):
        """Segment id under a 3D-pane pixel: the front-most SELECTED mesh at
        that pixel. Only selected segments have meshes in the pane, so a 3D
        double-click can only ever deselect -- which is exactly what NG does.

        Depth-per-segment rather than a volume query at the unprojected point:
        the hit lies ON a mesh surface, where a single-voxel lookup easily
        lands in the neighbouring segment or in background.
        """
        st = self._state
        ix = int(round(x_css * self.layout.capture_scale - PANE))
        iy = int(round(y_css * self.layout.capture_scale - TOOLBAR))
        if not (0 <= ix < PANE and 0 <= iy < PANE_H):
            return None
        ids = S.visible_segments(st["segments"])
        if not ids:
            return None
        self._ensure_meshes(ids)   # non-blocking: pick what has arrived
        pos_nm = np.asarray(st["position"], dtype=np.float64) * self._voxel_nm
        zoom_nm = float(st["projectionScale"]) * SCALE_CAL_NM
        best, best_d = None, 0.9999
        for rid in ids:
            depth, _, _ = self._renderer.pick_depth(
                [rid], pos_nm, st["projectionOrientation"], zoom_nm)
            # NG picks over a small radius; take the front-most hit nearby.
            y0, y1 = max(0, iy - 3), min(PANE_H, iy + 4)
            x0, x1 = max(0, ix - 3), min(PANE, ix + 4)
            d = float(depth[y0:y1, x0:x1].min())
            if d < best_d:
                best, best_d = rid, d
        return best

    def _pick_em_tiles(self):
        """EMTiles handle for client-side point queries (selecting is rare, so
        this deliberately does not go through the fetch pool)."""
        if self._pick_em is None:
            from .em import EMTiles

            self._pick_em = EMTiles(self.source)
        return self._pick_em

    def _pick_3d_world(self, x_css: float, y_css: float):
        """World point (voxels) under a 3D-pane pixel, or None for background.

        Shared by move-to-mouse-position and select so the two cannot drift on
        what "under the cursor" means.

        NOTE (2026-09-10): the `- TOOLBAR` below carries the same off-by-17-CSS-px
        error the 2D branch had before PANEL_*_CLICK -- the DOM measurement says
        a panel starts at click y=23 and is 853 CSS px tall, so the row should be
        (y_css - 23) * PANE_H / 853, which puts the panel centre at PANE_H/2
        rather than ~8.7 rows above it. Left alone deliberately: it changes where
        every 3D click lands and so alters every existing run, and unlike the 2D
        pick it has no measurement backing it yet. Fix behind its own gate.
        """
        st = self._state
        pos_nm = np.asarray(st["position"], dtype=np.float64) * self._voxel_nm
        quat = st["projectionOrientation"]
        zoom_nm = float(st["projectionScale"]) * SCALE_CAL_NM
        ext = pane_extents_nm(st["crossSectionScale"], self._canonical_nm)
        depth, view, proj = self._renderer.pick_depth(
            S.visible_segments(st["segments"]), pos_nm, quat, zoom_nm, plane_extent_nm=ext)
        fx = x_css * self.layout.capture_scale - PANE
        fy = y_css * self.layout.capture_scale - TOOLBAR
        ix, iy = int(round(fx)), int(round(fy))
        if not (0 <= ix < PANE and 0 <= iy < PANE_H):
            return None
        d = depth[iy, ix]
        px, py = ix, iy
        if d >= 0.9999:
            # NG issues the pick over a small radius; take the front-most hit
            # in a 3px window (matches the browser-validated harness).
            y0, y1 = max(0, iy - 3), min(PANE_H, iy + 4)
            x0, x1 = max(0, ix - 3), min(PANE, ix + 4)
            win = depth[y0:y1, x0:x1]
            if not (win < 0.9999).any():
                return None  # background: no-op
            yy, xx = np.unravel_index(np.argmin(win), win.shape)
            d, px, py = win[yy, xx], x0 + xx, y0 + yy
        ndc = np.array([2 * (px + 0.5) / PANE - 1,
                        1 - 2 * (py + 0.5) / PANE_H,
                        2 * d - 1, 1.0])
        w = np.linalg.inv(proj @ view) @ ndc
        return [float(v) for v in (w[:3] / w[3]) / self._voxel_nm]

    # ------------------------------------------------------------------ tiles

    def _tile_key_for(self, pos, xs):
        """GEOMETRY only -- deliberately not the selection.

        The fetched parts (EM raster, id map, plane tile) depend on where the
        viewer is, not on which segments are highlighted; the tint is applied
        client-side in _render_left. Keying on the selection instead forced a
        full refetch per click.
        """
        return (round(pos[0], 2), round(pos[1], 2), round(pos[2], 2),
                round(float(xs), 5), self.layout.left_pane)

    def _tile_state_key(self):
        st = self._state
        return self._tile_key_for(st["position"], st["crossSectionScale"])

    def _submit_tile_group(self, pos, xs, stage: str = "fine") -> tuple:
        """(key, futs, ext, stage, t0) for an arbitrary state -- used for the
        current state and for reset-ahead prefetch. stage='coarse' fetches
        the fast low-mip untinted preview; 'fine' the full tile."""
        pos_nm = np.asarray(pos, dtype=np.float64) * self._voxel_nm
        ext = pane_extents_nm(xs, self._canonical_nm)
        pool = self._tile_pool()
        src = self.source
        mx = self._coarse_px if stage == "coarse" else self._fine_px
        if self.layout.left_pane:
            # TWO jobs, run in PARALLEL, adopted independently. They read
            # different volumes and share no chunks -- 0.84 s for the EM half
            # and 0.92 s for the ids -- so running them back to back made the
            # 2D pane wait 1.77 s for 0.92 s of work (probe_parts_breakdown).
            # The plane stays with the EM tile: it reuses those chunks and
            # costs 0.01 s there against 0.84 s anywhere else. Two jobs fit one
            # wave through a 2-worker pool; the three-way split that preceded
            # this needed two and left the pane never refreshing (1ee8cef).
            if self._parallel_parts:
                futs = {"emplane": pool.submit(worker_em_plane, src, list(pos), float(xs), mx)}
                if stage != "coarse":
                    futs["ids"] = pool.submit(worker_ids, src, list(pos), float(xs))
            else:
                # NGL_NATIVE_PARALLEL_PARTS=0: the old single bundled job, kept
                # so the split can be A/B'd on identical states rather than
                # argued from the fetch timings alone.
                futs = {"parts": pool.submit(
                    worker_pane_parts, src, list(pos), float(xs), mx, stage != "coarse")}
        else:
            futs = {"plane": pool.submit(worker_tile, src, pos_nm, ext[0], ext[1], mx, False)}
        return (self._tile_key_for(pos, xs), futs, ext, stage, time.monotonic())

    def _submit_tile_fetch(self, stage: str = "fine"):
        st = self._state
        self._pending = self._submit_tile_group(st["position"], st["crossSectionScale"], stage)

    def _adopt_pending(self, timeout_s: float = 180.0) -> None:
        self._adopt_group(*self._pending, timeout_s=timeout_s)
        self._pending = None

    def _adopt_group(self, key, futs, ext, stage, t0=None, timeout_s: float = 180.0) -> None:
        tiles: dict[str, Any] = {"ext": ext, "plane": None, "em": None, "ids": None}
        for name, fut in futs.items():
            try:
                if name == "parts":
                    em_gray, ids_packed, tiles["plane"] = fut.result(timeout=timeout_s)
                    tiles["em"] = em_gray
                    tiles["ids"] = unpack_ids(ids_packed)
                elif name == "emplane":
                    tiles["em"], tiles["plane"] = fut.result(timeout=timeout_s)
                elif name == "ids":
                    tiles["ids"] = unpack_ids(fut.result(timeout=timeout_s))
                else:
                    tiles[name] = fut.result(timeout=timeout_s)
            except FuturesTimeout:
                logger.warning("EM %s tile fetch timed out; skipped", name)
            except Exception as e:
                logger.warning("EM %s tile fetch failed (%s); skipped", name, e)
        self._tile_key, self._tiles = key, tiles
        self._tile_stage = stage
        if t0 is not None:
            # submit -> on screen for the 2D pane, readable at production
            # density where several envs share a shard.
            logger.info("tiles %s on screen after %.2fs", stage, time.monotonic() - t0)
        self._schedule_warm()

    def _schedule_warm(self) -> None:
        """Pull the region AROUND the pane into the worker's chunk cache.

        A move's cost is almost all new edge chunks: 0.20 s for a 15% move
        cold, 0.03 s once a 1.6x region is resident (probe_tile_locality). This
        buys that without touching a pixel -- the pane is still produced by the
        exact fetch, and the warm job's output is discarded. One in flight at a
        time, and never in place of a real fetch.
        """
        if self._warm_factor <= 1.0:
            return
        if self._warm_fut is not None and not self._warm_fut.done():
            return
        st = self._state
        try:
            self._warm_fut = self._tile_pool().submit(
                worker_warm, self.source, list(st["position"]),
                float(st["crossSectionScale"]), self._warm_factor)
        except Exception as e:  # noqa: BLE001
            logger.warning("warm submit failed (%s)", e)

    def _fetch_tiles(self, block: bool) -> dict[str, Any]:
        """Tiles for the current state. block=True (reset) waits for exact
        tiles; block=False (step) returns the last completed set while the
        fetch streams in -- see the pipeline note in __init__."""
        key = self._tile_state_key()
        # Drain a finished fetch FIRST, whatever else happens: it refreshes the
        # plane and adds its region to the cache, and leaving it pending would
        # block every later fetch (only one is in flight at a time).
        if (self._pending is not None
                and all(f.done() for f in self._pending[1].values())):
            self._adopt_pending()
        if key == self._tile_key and self._tile_stage == "fine":
            return self._tiles  # settled at full resolution
        if self._pending is not None and self._pending[0] != key:
            # Superseded in-flight fetch: adopt it only if it already
            # finished (warms the LRU either way), then refetch.
            if all(f.done() for f in self._pending[1].values()):
                self._adopt_pending()
                if key == self._tile_key and self._tile_stage == "fine":
                    return self._tiles
            elif block:
                self._adopt_pending()  # drain before the exact fetch
            else:
                return self._tiles  # keep rendering stale; let it land
        mode = self.pane_mode
        # CONCURRENT: a coarse companion fetch runs ALONGSIDE the fine one
        # (Neuroglancer's filterVisibleSources yields every scale at once),
        # so the preview does not delay the full tile the way the sequential
        # 'progressive' staging did.
        if mode == "concurrent" and not block and self._coarse_pending is not None:
            ck, cfuts, cext, cstage, ct0 = self._coarse_pending
            if ck != key:
                self._coarse_pending = None          # superseded; drop it
            elif all(f.done() for f in cfuts.values()):
                if not (self._tile_key == key and self._tile_stage == "fine"):
                    self._adopt_group(ck, cfuts, cext, cstage, ct0)
                self._coarse_pending = None
        if self._pending is None:
            stage = "fine" if (block or mode in ("atomic", "random", "concurrent")) else "coarse"
            self._submit_tile_fetch(stage)
            if mode == "concurrent" and not block:
                st = self._state
                self._coarse_pending = self._submit_tile_group(
                    st["position"], st["crossSectionScale"], "coarse")
        if block or all(f.done() for f in self._pending[1].values()):
            # RANDOM: hold a landed tile for a per-episode number of extra
            # steps. Fetch latency is a networking artifact -- Chrome's own
            # step-0 fidelity spans 38-80% on identical states -- so train
            # across the distribution instead of one fixed lag. Panes stay
            # SHARP: 'progressive' showed that a blurry current pane costs
            # 11pp against a sharp stale one.
            if not block and mode == "random" and self._adopt_delay > 0:
                self._adopt_delay -= 1
                return self._tiles
            self._adopt_pending()
            if (not block and mode == "progressive"
                    and self._tile_key == key and self._tile_stage == "coarse"):
                self._submit_tile_fetch("fine")
        return self._tiles

    # ------------------------------------------------------------------ meshes

    def _ensure_meshes(self, segments, block: bool = False) -> None:
        """Make selected segments' meshes resident, STREAMING like Chrome:
        a COARSE level first, refined when the full one lands.

        Both halves are measured. Fetching inline made the simulator strictly
        faster than Chrome (response step 0 against ~9); fetching only the full
        mesh made it much slower -- 71.5 steps against Chrome's 2.0 under
        production stepping, ~1.0 s against ~0.08 s (883638). The cause is that
        NG requests the coarsest adequate level of the multi-resolution mesh
        and refines later, while we fetched 73k-340k vertices every time:
        measured 0.71 s median at lod 0 against 0.29-0.48 s at the coarsest,
        and Chrome's own misses land in ~0.08-0.35 s.

        So a select shows a coarse mesh at roughly Chrome's latency and sharpens
        after, instead of showing nothing for a second. `block=True` at reset
        goes straight to the full mesh: the first observation has to be
        complete, and Chrome has likewise settled before an episode starts.
        """
        for rid in S.visible_segments(segments):
            if rid in self._mesh_futs or (self._renderer.has_mesh(rid)
                                          and rid not in self._mesh_fine):
                continue
            # lod 0 when blocking (reset) or when this is the refinement
            # pass for a segment already showing its coarse level.
            lod = 0 if (block or rid in self._mesh_fine) else self.MESH_COARSE_LOD
            try:
                self._mesh_futs[rid] = (lod, self._tile_pool().submit(
                    worker_mesh, self.source, rid, lod))
                self._mesh_due[rid] = self._steps + self._mesh_lag
                self._mesh_t0[rid] = (time.monotonic(), self._steps)
            except Exception as e:  # noqa: BLE001
                logger.warning("mesh submit for %s failed (%s)", rid, e)
        for rid in list(self._mesh_futs):
            lod, fut = self._mesh_futs[rid]
            if not (block or fut.done()):
                continue
            if not block and self._steps < self._mesh_due.get(rid, 0):
                continue      # landed early; hold it to the modelled lag
            del self._mesh_futs[rid]
            self._mesh_due.pop(rid, None)
            t0 = self._mesh_t0.pop(rid, None)
            if t0 is not None:
                logger.info("mesh %s lod%d on screen after %.2fs / %d steps",
                            rid, lod, time.monotonic() - t0[0], self._steps - t0[1])
            try:
                v, vn, f = fut.result(timeout=240 if block else None)
                self._renderer.load_mesh(rid, v, f, normals=vn, replace=lod == 0)
            except Exception as e:  # noqa: BLE001
                logger.warning("mesh fetch for segment %s failed (%s)", rid, e)
                continue
            if lod == 0:
                self._mesh_fine.discard(rid)
            else:
                # Coarse level is on screen; queue the refinement.
                self._mesh_fine.add(rid)

    # ------------------------------------------------------------------ frames

    def _render_left(self, tiles: dict[str, Any]) -> np.ndarray:
        """2D xy EM pane, composed from the cached raster + id map.

        Memoized on the VISIBLE SET, not just on the tiles: a selection change
        re-tints from data already in hand, with no fetch, which is what
        Neuroglancer does. The EM resample (the priciest per-step CPU) already
        happened in the fetch worker, so this is a mask-and-blend.
        """
        vis = S.visible_segments(self._state["segments"])
        cached = tiles.get("left_canvas")
        if cached is not None and tiles.get("left_vis") == vis:
            return cached
        canvas = compose_left_parts(tiles.get("em"), tiles.get("ids"), vis)
        tiles["left_canvas"], tiles["left_vis"] = canvas, vis
        return canvas

    def _render_right(self, tiles: dict[str, Any]) -> np.ndarray:
        st = self._state
        ids = S.visible_segments(st["segments"])
        # A segment selected mid-episode (double-click) has no mesh yet.
        self._ensure_meshes(ids)
        pos_nm = np.asarray(st["position"], dtype=np.float64) * self._voxel_nm
        plane = tiles["plane"]
        if plane is not None and tiles.get("ids") is not None:
            key = ("plane_rgb", ids)
            if tiles.get("plane_key") != key:
                tiles["plane_rgb"] = tint_plane(plane, tiles["ids"], ids)
                tiles["plane_key"] = key
            plane = tiles["plane_rgb"]
        pane = self._renderer.render(
            ids, pos_nm, st["projectionOrientation"],
            float(st["projectionScale"]) * SCALE_CAL_NM,
            [segment_color(int(r)) for r in ids],
            em_tile=plane,
            em_extent_nm=(tiles["ext"][0] * pane2d_mod.PLANE_EXT_SCALE,
                          tiles["ext"][1] * pane2d_mod.PLANE_EXT_SCALE),
            em_gain=EM_GAIN, to_cuda=self.cuda_ipc)
        if self.cuda_ipc:
            # render(to_cuda=True) already returns the STABLE CUDA-IPC (rebuild,
            # args) payload for the GPU-resident frame (built once, reused). Ship
            # it as-is; the DINO server rebuilds/caches it in VRAM. No toolbar pad.
            return pane
        out = np.zeros((PANE, PANE, 3), dtype=np.uint8)
        out[TOOLBAR:] = pane
        return out

    def _render(self, block_tiles: bool) -> np.ndarray:
        tiles = self._fetch_tiles(block=block_tiles)
        panes = []
        if self.layout.left_pane:
            panes.append(self._render_left(tiles))
        if self.layout.right_pane:
            panes.append(self._render_right(tiles))
        return panes[0] if len(panes) == 1 else np.concatenate(panes, axis=1)
