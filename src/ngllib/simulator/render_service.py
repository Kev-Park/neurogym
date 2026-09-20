"""Per-process batched render service (throughput-scaling experiment).

The default topology gives every env its OWN GL context (its own MeshRenderer);
at N envs/process the GPU time-slices N GL contexts and does N separate readbacks
per step. This service instead runs ONE MeshRenderer (one GL context, one shared
mesh slot pool) on ONE dedicated thread, and coalesces the per-step render calls
from all the process's env threads into a SINGLE atlas render + a SINGLE readback
(or a single GL->CUDA interop copy). Fewer GL contexts to switch among, and the
fixed per-render bind + readback/sync overhead is amortized across the batch.

It is a per-process singleton (like SimulatorRenderer's tile pools): all the
process's envs share it. It exposes the subset of the MeshRenderer API that
SimulatorRenderer drives — has_mesh / load_mesh / render / pick_depth — with
`render` transparently batched. GL work only ever happens on the service thread
(GL contexts are single-thread-current); env threads marshal via a queue + future
and block, exactly like the DINO server's request/response.

Coalescing is dynamic (a short delay window), so a straggler env doing a blocking
reset fetch never holds the stepping envs hostage — it just renders in a later,
smaller batch. Synchronous PPO steps all envs together, so the common case is a
full batch per step.
"""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future

from .render3d import MeshRenderer

# Per-process singletons, keyed by construction params (in practice one entry:
# all a process's envs share one config).
_SERVICES: dict = {}
_SERVICES_LOCK = threading.Lock()


def get_render_service(pane: int, pane_h: int, batch_size: int, *,
                       interop: bool = False, ipc_export: bool = False,
                       mesh_budget_bytes: int | None = None,
                       max_delay_ms: float = 3.0) -> "RenderService":
    """Get (or lazily create) the process's shared RenderService.

    `interop` keeps the batched panes in VRAM (GL->CUDA). `ipc_export` controls
    the hand-off: False = raw CUDA cell views for an IN-PROCESS encoder; True =
    per-cell reduce_tensor IPC payloads for a cross-process DINO SERVER."""
    key = (pane, pane_h, batch_size, interop, ipc_export, mesh_budget_bytes)
    with _SERVICES_LOCK:
        svc = _SERVICES.get(key)
        if svc is None:
            svc = _SERVICES[key] = RenderService(
                pane, pane_h, batch_size, interop=interop, ipc_export=ipc_export,
                mesh_budget_bytes=mesh_budget_bytes, max_delay_ms=max_delay_ms)
        return svc


class RenderService:
    def __init__(self, pane: int, pane_h: int, batch_size: int, *,
                 interop: bool = False, ipc_export: bool = False,
                 mesh_budget_bytes: int | None = None, max_delay_ms: float = 3.0):
        self._pane, self._pane_h = pane, pane_h
        self._batch_cap = max(1, int(batch_size))
        self._interop = bool(interop)
        self._ipc_export = bool(ipc_export)
        self._mesh_budget = mesh_budget_bytes
        self._max_delay = float(max_delay_ms) / 1000.0
        self._q: queue.Queue = queue.Queue()
        self._resident: set = set()
        self._resident_lock = threading.Lock()
        self._mr: MeshRenderer | None = None
        self._ready = threading.Event()
        self._err: BaseException | None = None
        self._thread = threading.Thread(
            target=self._loop, name="render-service", daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._err is not None:
            raise self._err

    @property
    def ctx(self):
        return self._mr.ctx

    # ------------------------------------------------------------ env-thread API

    def has_mesh(self, root_id: str) -> bool:
        with self._resident_lock:
            return root_id in self._resident

    def load_mesh(self, *args, **kwargs) -> None:
        fut: Future = Future()
        self._q.put(("load_mesh", (args, kwargs), fut))
        return fut.result()

    def pick_depth(self, *args, **kwargs):
        fut: Future = Future()
        self._q.put(("pick", (args, kwargs), fut))
        return fut.result()

    def render(self, root_id, position_nm, quat, zoom_nm, color,
               em_tile=None, em_extent_nm=None, em_gain: float = 1.0,
               to_cuda: bool = False):
        """Submit one 3D scene; block until the batch it lands in is rendered.
        Returns this env's cell: (H, W, 3) uint8 numpy (readback) or
        ((H, W, 4) uint8 cuda view, gl_flip=True) (interop). `to_cuda` is honored
        via the service's fixed transfer mode (all envs in a process match)."""
        scene = {
            "root_id": root_id, "position_nm": position_nm, "quat": quat,
            "zoom_nm": zoom_nm, "color": color, "em_tile": em_tile,
            "em_extent_nm": em_extent_nm, "em_gain": em_gain,
        }
        fut: Future = Future()
        self._q.put(("render", scene, fut))
        return fut.result()

    def close(self) -> None:
        # Per-process singleton: outlives individual envs (reaped at exit). A
        # SimulatorRenderer.close() must NOT tear this down for its siblings.
        pass

    # ---------------------------------------------------------- service thread

    def _loop(self) -> None:
        try:
            self._mr = MeshRenderer(
                self._pane, self._pane_h, self._mesh_budget,
                cuda_ipc=self._interop, ipc_export=self._ipc_export)
            self._mr.enable_atlas(self._batch_cap)
        except BaseException as e:  # surface init failure to the constructor
            self._err = e
            self._ready.set()
            return
        self._ready.set()

        pending: list[tuple[dict, Future]] = []
        first_t = 0.0
        while True:
            timeout = None
            if pending:
                timeout = max(0.0, self._max_delay - (time.monotonic() - first_t))
            try:
                op = self._q.get(timeout=timeout)
            except queue.Empty:
                op = None  # delay window elapsed -> flush what we have
            if op is None:
                self._flush(pending); pending = []
                continue
            kind = op[0]
            if kind == "render":
                _, scene, fut = op
                if not pending:
                    first_t = time.monotonic()
                pending.append((scene, fut))
                if len(pending) >= self._batch_cap:
                    self._flush(pending); pending = []
            elif kind == "load_mesh":
                _, (args, kwargs), fut = op
                try:
                    self._mr.load_mesh(*args, **kwargs)
                    with self._resident_lock:
                        self._resident = set(self._mr._vaos.keys())
                    fut.set_result(None)
                except BaseException as e:
                    fut.set_exception(e)
            elif kind == "pick":
                _, (args, kwargs), fut = op
                try:
                    fut.set_result(self._mr.pick_depth(*args, **kwargs))
                except BaseException as e:
                    fut.set_exception(e)
            elif kind == "close":
                break

    def _flush(self, pending: list[tuple[dict, Future]]) -> None:
        if not pending:
            return
        scenes = [s for s, _ in pending]
        try:
            panes = self._mr.render_batch(scenes, to_cuda=self._interop)
            for (_, fut), pane in zip(pending, panes):
                if not fut.done():
                    fut.set_result(pane)
        except BaseException as e:
            for _, fut in pending:
                if not fut.done():
                    fut.set_exception(e)
