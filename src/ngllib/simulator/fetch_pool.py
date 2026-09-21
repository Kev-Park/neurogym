"""Pluggable fetch/decode worker pools for the simulator.

Two backends, same `.submit(fn, *args) -> future` interface (future exposes
`.done()` and `.result(timeout)`), so the renderer's submit/adopt machinery is
unchanged:

- **process** (default): N single-worker `ProcessPoolExecutor`s on THIS node, one
  spawn process each. What the fleet has always run.
- **ray**: N Ray `_FetchActor`s. Identical semantics (each actor process holds its
  own `_WORKER_EM`/`_WORKER_MESHES` chunk/mesh LRU, so env->actor affinity gives the
  same locality the process pools do), but the actors can be PLACED on GPU-less CPU
  nodes via a custom Ray resource -- this is disaggregated fetch: decode cores scale
  independently of GPUs, attacking the chunk-decompress CPU wall (SPS recipe §10).
  Each ray pool runs a DEDICATED BACKGROUND THREAD that owns the Ray round-trip
  (`ray.get`), so env threads only touch local `_LocalFuture`s -- a cross-node fetch
  never blocks the env-runner actor's env-threads (approach D). Without this, a
  blocking `ray.get` on the env thread stalled `sample()` for 266 s and timed the
  runner out.

Decoded results (numpy tiles ~0.8 MB, meshes up to ~60 MB) ship back over Ray's
object store -- the network cost is the trade to measure against the CPU gain.

Knobs (launcher-set, per-node budgets):
- NGL_NATIVE_FETCH_WORKERS   pool count N (default 6)
- NGL_NATIVE_FETCH_BACKEND   'process' (default) | 'ray'
- NGL_NATIVE_FETCH_AFFINITY  '1' (default, env->pool sticky) | '0' (round-robin, A/B)
- NGL_NATIVE_FETCH_RESOURCE  ray custom resource to pin fetch actors onto (e.g.
                             'fetch_cpu'); unset => placed anywhere (local, Stage 0)
"""
from __future__ import annotations

import multiprocessing
import os
import queue
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout


def pool_config() -> tuple[int, str, bool]:
    n = max(1, int(os.environ.get("NGL_NATIVE_FETCH_WORKERS", "6")))
    backend = os.environ.get("NGL_NATIVE_FETCH_BACKEND", "process").lower()
    affinity = os.environ.get("NGL_NATIVE_FETCH_AFFINITY", "1") != "0"
    return n, backend, affinity


def make_pools():
    """Return a list of N pools, each with `.submit(fn, *args) -> future`."""
    n, backend, _ = pool_config()
    if backend == "ray":
        return _make_ray_pools(n)
    return _make_process_pools(n)


def _make_process_pools(n: int):
    ctx = multiprocessing.get_context("spawn")
    return [ProcessPoolExecutor(max_workers=1, mp_context=ctx) for _ in range(n)]


# --------------------------------------------------------------------------- ray

_FETCH_ACTOR_CLS = None


def _fetch_actor_cls():
    """Define the actor lazily so ngllib does not hard-import ray."""
    global _FETCH_ACTOR_CLS
    if _FETCH_ACTOR_CLS is None:
        import ray

        @ray.remote
        class _FetchActor:
            """A decode worker. Runs the SAME module-level worker fns the process
            pool does, so it builds its own per-actor EMTiles/MeshStore caches."""

            def run(self, fn, *args):
                return fn(*args)

            def ping(self):
                return os.getpid()

        _FETCH_ACTOR_CLS = _FetchActor
    return _FETCH_ACTOR_CLS


class _LocalFuture:
    """A future whose done()/result() are PURE-LOCAL (a threading.Event), so the
    caller -- an RLlib env-runner actor's env-thread -- never makes a Ray call on
    the env-step path. The Ray round-trip happens on the pool's background thread
    (approach D). Same surface the renderer's submit/adopt machinery expects."""

    __slots__ = ("_ev", "_result", "_exc")

    def __init__(self):
        self._ev = threading.Event()
        self._result = None
        self._exc: BaseException | None = None

    def _set(self, result, exc):
        self._result, self._exc = result, exc
        self._ev.set()

    def done(self) -> bool:
        return self._ev.is_set()

    def result(self, timeout=None):
        if not self._ev.wait(timeout):
            raise FuturesTimeout("fetch background result timed out")
        if self._exc is not None:
            raise self._exc
        return self._result


class _RayPool:
    """One fetch actor with a DEDICATED BACKGROUND THREAD that owns all Ray I/O.

    Env threads only submit() (enqueue) and poll _LocalFuture (a local Event) --
    no ray.get / ray.wait ever runs on the env-step path, so a cross-node fetch
    can never block the env-runner actor's env-threads (that is what stalled a
    266 s sample() and timed workers out). Serial per pool (one actor), matching
    the single-worker process pool; concurrency comes from having N pools."""

    def __init__(self, actor):
        self._actor = actor
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._loop, name="ray-fetch-pool", daemon=True)
        self._thread.start()

    def submit(self, fn, *args):
        fut = _LocalFuture()
        self._q.put((fut, fn, args))
        return fut

    def _loop(self):
        import ray

        while True:
            item = self._q.get()
            if item is None:  # shutdown sentinel
                return
            fut, fn, args = item
            try:
                fut._set(ray.get(self._actor.run.remote(fn, *args)), None)
            except BaseException as e:  # noqa: BLE001 - surfaced at caller's .result()
                fut._set(None, e)

    def shutdown(self, wait=False):  # parity with ProcessPoolExecutor
        self._q.put(None)


def _make_ray_pools(n: int):
    import ray

    if not ray.is_initialized():
        ray.init(address=os.environ.get("RAY_ADDRESS", "auto"),
                 ignore_reinit_error=True)
    cls = _fetch_actor_cls()
    opts = {"num_cpus": 1}
    res = os.environ.get("NGL_NATIVE_FETCH_RESOURCE")
    if res:  # pin onto CPU-only nodes that advertise this custom resource
        opts["resources"] = {res: 1}
    return [_RayPool(cls.options(**opts).remote()) for _ in range(n)]
