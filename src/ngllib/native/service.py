"""Per-node render+encode service core (the measured-optimal architecture).

One process owns ONE GL context + ONE frozen encoder and serves many
lightweight client envs: clients submit viewer STATES (~100B) and receive
DINO FEATURES (2*D f32); clicks resolve through a pick pass. Probe-validated
at 255 sps/node (K=64 clients, pipelined; serial alternation caps ~176).

Structure (matches native/probe_render_service.py, productionized):
- GL thread: drains the request queue in small windows; answers picks
  inline; renders 3D panes; hands (frames, requests) to the encode thread.
- Encode thread: one batched encoder forward per group; resolves futures.
- 2D panes are composed in the fetch process pool (worker_left_canvas) and
  cached per tile-key with browser-equivalent stale semantics: a state
  whose canvas is still fetching is served with the client's previous
  canvas (Chrome renders loaded chunks while streaming the rest).

Framework-free: the encoder is injected (anything with
`.encode(list[img]) -> (B, D)` / `.feature_dim`); Ray/actor wrapping lives
in ngllib_agent.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np

from . import pane2d
from .colors import segment_color
from .em import MeshStore, worker_left_canvas
from .render3d import MeshRenderer

logger = logging.getLogger(__name__)


class RenderEncodeService:
    def __init__(self, encoder, cache_dir: str | None = None,
                 fetch_workers: int = 8, window_ms: float = 5.0,
                 mesh_budget_bytes: int = 4 << 30,
                 canvas_cache: int = 512):
        self._encoder = encoder
        self.feature_dim = int(encoder.feature_dim)
        self._cache_dir = cache_dir
        self._window_s = window_ms / 1000.0
        # GL context is created ON the GL thread (moderngl contexts are
        # thread-affine; creating it here and using it there fails with
        # "cannot create buffer").
        self._rend: MeshRenderer | None = None
        self._mesh_budget = mesh_budget_bytes
        self._gl_ready = threading.Event()
        self._meshes = MeshStore(cache_dir)
        # THREADS, not processes: nested multiprocessing inside a Ray actor
        # breaks (resource-tracker KeyErrors kill the actor). Canvas work is
        # network waits + PIL/zlib C calls that release the GIL.
        from .em import _worker_em

        _worker_em(cache_dir)  # eager init so threads only read the cache
        self._fetch_pool = ThreadPoolExecutor(
            max_workers=fetch_workers,
            thread_name_prefix="ngl-svc-fetch")
        self._req_q: queue.Queue = queue.Queue()
        self._enc_q: queue.Queue = queue.Queue(maxsize=2)
        # Composed-canvas cache: tile_key -> canvas; misses submit a worker
        # job and serve the client's last canvas meanwhile.
        self._canvases: dict[tuple, np.ndarray] = {}
        self._canvas_order: list[tuple] = []
        self._canvas_cap = canvas_cache
        self._canvas_pending: dict[tuple, Any] = {}
        self._client_last: dict[Any, np.ndarray] = {}
        self._blank = np.zeros((pane2d.PANE, pane2d.PANE, 3), dtype=np.uint8)
        self._stop = False
        self._gl_thread = threading.Thread(target=self._gl_loop, daemon=True)
        self._enc_thread = threading.Thread(target=self._enc_loop, daemon=True)
        self._gl_thread.start()
        self._enc_thread.start()

    # ------------------------------------------------------------ public API

    def features(self, client_id, state: dict[str, Any],
                 block_canvas: bool = False) -> np.ndarray:
        """(2*D,) float32 features for the state. block_canvas=True (resets)
        waits for the exact 2D canvas instead of serving a stale one."""
        fut: Future = Future()
        self._req_q.put(("obs", client_id, state, block_canvas, fut))
        return fut.result(timeout=300)

    def pick(self, state: dict[str, Any], px: int, py: int):
        """Picked position (voxels) for a 3D-pane pixel, or None."""
        fut: Future = Future()
        self._req_q.put(("pick", None, state, (px, py), fut))
        return fut.result(timeout=300)

    def close(self):
        self._stop = True
        self._enc_q.put(None)
        self._fetch_pool.shutdown(wait=False)

    # ------------------------------------------------------------ internals

    def _tile_key(self, state):
        p = state["position"]
        return (round(p[0], 2), round(p[1], 2), round(p[2], 2),
                round(float(state["crossSectionScale"]), 5),
                str(state["segments"][0]))

    def _canvas_store(self, key, canvas):
        self._canvases[key] = canvas
        self._canvas_order.append(key)
        while len(self._canvas_order) > self._canvas_cap:
            self._canvases.pop(self._canvas_order.pop(0), None)

    def _canvas_for(self, client_id, state, block: bool) -> np.ndarray:
        key = self._tile_key(state)
        got = self._canvases.get(key)
        if got is None:
            fut = self._canvas_pending.get(key)
            if fut is None:
                fut = self._fetch_pool.submit(
                    worker_left_canvas, self._cache_dir, state["position"],
                    float(state["crossSectionScale"]),
                    str(state["segments"][0]))
                self._canvas_pending[key] = fut
            if block or fut.done():
                try:
                    got = fut.result(timeout=180)
                except Exception as e:
                    logger.warning("left canvas failed (%s); blank", e)
                    got = self._blank
                self._canvas_pending.pop(key, None)
                self._canvas_store(key, got)
        if got is None:  # stale path
            got = self._client_last.get(client_id, self._blank)
        self._client_last[client_id] = got
        return got

    def _ensure_mesh(self, rid: str):
        if not self._rend.has_mesh(rid):
            v, f = self._meshes.get(rid)
            self._rend.load_mesh(rid, v, f)

    def _render_right(self, state) -> np.ndarray:
        rid = str(state["segments"][0])
        self._ensure_mesh(rid)
        pos_nm = np.asarray(state["position"], dtype=np.float64) * pane2d.VOXEL_NM
        # Section-plane EM texture omitted service-side v1: the plane occupies
        # a small screen region and its EM content is the least policy-salient
        # element; revisit with a parity spot-check if features drift.
        pane = self._rend.render(
            rid, pos_nm, state["projectionOrientation"],
            float(state["projectionScale"]) * pane2d.SCALE_CAL_NM,
            segment_color(int(rid)), em_gain=pane2d.EM_GAIN)
        return pane2d.paste_right(pane)

    def _do_pick(self, state, px, py):
        rid = str(state["segments"][0])
        self._ensure_mesh(rid)
        pos_nm = np.asarray(state["position"], dtype=np.float64) * pane2d.VOXEL_NM
        ext = pane2d.pane_extents_nm(float(state["crossSectionScale"]))
        depth, view, proj = self._rend.pick_depth(
            rid, pos_nm, state["projectionOrientation"],
            float(state["projectionScale"]) * pane2d.SCALE_CAL_NM,
            plane_extent_nm=ext)
        if not (0 <= px < pane2d.PANE and 0 <= py < pane2d.PANE_H):
            return None
        d = depth[py, px]
        x, y = px, py
        if d >= 0.9999:
            y0, y1 = max(0, py - 3), min(pane2d.PANE_H, py + 4)
            x0, x1 = max(0, px - 3), min(pane2d.PANE, px + 4)
            win = depth[y0:y1, x0:x1]
            if not (win < 0.9999).any():
                return None
            yy, xx = np.unravel_index(np.argmin(win), win.shape)
            d, x, y = win[yy, xx], x0 + xx, y0 + yy
        ndc = np.array([2 * (x + 0.5) / pane2d.PANE - 1,
                        1 - 2 * (y + 0.5) / pane2d.PANE_H,
                        2 * d - 1, 1.0])
        w = np.linalg.inv(proj @ view) @ ndc
        return [float(v) for v in (w[:3] / w[3]) / pane2d.VOXEL_NM]

    def _gl_loop(self):
        self._rend = MeshRenderer(pane2d.PANE, pane2d.PANE_H,
                                  self._mesh_budget)
        self._gl_ready.set()
        while not self._stop:
            try:
                first = self._req_q.get(timeout=0.5)
            except queue.Empty:
                continue
            group = [first]
            wend = time.monotonic() + self._window_s
            while time.monotonic() < wend:
                try:
                    group.append(self._req_q.get_nowait())
                except queue.Empty:
                    time.sleep(0.0005)
            frames, futs = [], []
            for msg in group:
                kind = msg[0]
                try:
                    if kind == "pick":
                        _, _, state, (px, py), fut = msg
                        fut.set_result(self._do_pick(state, px, py))
                    else:
                        _, cid, state, block, fut = msg
                        left = self._canvas_for(cid, state, block)
                        right = self._render_right(state)
                        frames.append(left)
                        frames.append(right)
                        futs.append(fut)
                except Exception as e:  # noqa: BLE001
                    msg[-1].set_exception(e)
            if futs:
                self._enc_q.put((frames, futs))

    def _enc_loop(self):
        while True:
            item = self._enc_q.get()
            if item is None:
                return
            frames, futs = item
            try:
                feats = self._encoder.encode(frames)  # (2N, D)
                for j, fut in enumerate(futs):
                    fut.set_result(
                        feats[2 * j:2 * j + 2].reshape(-1).astype(np.float32))
            except Exception as e:  # noqa: BLE001
                for fut in futs:
                    if not fut.done():
                        fut.set_exception(e)
