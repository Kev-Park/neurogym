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


class _RayFuture:
    """Adapt a Ray ObjectRef to the concurrent.futures Future surface the renderer
    uses (`.done()`, `.result(timeout)` raising FuturesTimeout)."""

    __slots__ = ("_ref", "_cached")

    def __init__(self, ref):
        self._ref = ref
        self._cached = False

    def done(self) -> bool:
        if self._cached:
            return True
        import ray

        ready, _ = ray.wait([self._ref], num_returns=1, timeout=0)
        self._cached = bool(ready)
        return self._cached

    def result(self, timeout=None):
        import ray

        try:
            return ray.get(self._ref, timeout=timeout)
        except ray.exceptions.GetTimeoutError as e:  # normalize for _adopt_group
            raise FuturesTimeout(str(e)) from e


class _RayPool:
    """One fetch actor, exposing the ProcessPoolExecutor-compatible submit()."""

    def __init__(self, actor):
        self._actor = actor

    def submit(self, fn, *args):
        return _RayFuture(self._actor.run.remote(fn, *args))

    def shutdown(self, wait=False):  # parity with ProcessPoolExecutor
        pass


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
