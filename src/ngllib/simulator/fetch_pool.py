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

- **socket**: N persistent TCP connections to a fetch server (fetch_server.py) on the
  CPU node. The GPU node connects OUT (the only direction a directional firewall
  allows), so NO Ray cluster spans the nodes -- the GPU node runs local Ray and each
  env-runner's pool just opens a socket. Sidesteps every cross-node Ray issue.

Knobs (launcher-set, per-node budgets):
- NGL_NATIVE_FETCH_WORKERS   pool count N (default 6)
- NGL_NATIVE_FETCH_BACKEND   'process' (default) | 'ray' | 'socket'
- NGL_NATIVE_FETCH_AFFINITY  '1' (default, env->pool sticky) | '0' (round-robin, A/B)
- NGL_NATIVE_FETCH_RESOURCE  ray custom resource to pin fetch actors onto (e.g.
                             'fetch_cpu'); unset => placed anywhere (local, Stage 0)
- NGL_FETCH_SERVER_HOST/PORT socket backend: the CPU-node fetch server's IP + port
"""
from __future__ import annotations

import multiprocessing
import os
import pickle
import queue
import socket
import struct
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout


# ---------------------------------------------------------------- socket wire
# Length-prefixed frames (4-byte big-endian length + payload), shared by the
# socket fetch client (below) and fetch_server.py. Used by the 'socket' backend:
# GPU-node env-runners connect OUT to a fetch server on the CPU node (the only
# direction a directional firewall allows) and get decoded numpy back on the same
# stateful connection -- disaggregated fetch with no Ray and no login proxy.

def _recv_all(conn: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def recv_msg(conn: socket.socket) -> bytes | None:
    hdr = _recv_all(conn, 4)
    if hdr is None:
        return None
    (ln,) = struct.unpack(">I", hdr)
    return _recv_all(conn, ln)


def send_msg(conn: socket.socket, payload: bytes) -> None:
    conn.sendall(struct.pack(">I", len(payload)) + payload)


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
    if backend == "socket":
        return _make_socket_pools(n)
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

    # concurrent.futures.Future surface the renderer's fetch state machine touches
    # (it calls .cancel() on superseded/warm fetches). A fetch already dispatched to
    # the background thread cannot be un-dispatched, so cancel() reports False just
    # like a ProcessPoolExecutor future for an already-running task -- the renderer
    # then simply lets it finish and discards the result.
    def cancel(self) -> bool:
        return False

    def cancelled(self) -> bool:
        return False

    def running(self) -> bool:
        return not self._ev.is_set()

    def exception(self, timeout=None):
        if not self._ev.wait(timeout):
            raise FuturesTimeout("fetch background result timed out")
        return self._exc


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


# ------------------------------------------------------------------- socket
# A GPU-node client that connects OUT to a fetch server on the CPU node (see
# fetch_server.py). One persistent connection per pool -> the server serves it
# from one process with its own chunk/mesh LRU, so env->pool affinity gives the
# same locality as the process pools. All socket I/O runs on a dedicated
# background thread (approach D): env threads only submit + poll _LocalFuture.

class _SocketPool:
    def __init__(self, host: str, port: int):
        self._host, self._port = host, port
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._loop, name="socket-fetch-pool", daemon=True)
        self._thread.start()

    def submit(self, fn, *args):
        fut = _LocalFuture()
        self._q.put((fut, fn, args))
        return fut

    def _connect(self) -> socket.socket:
        s = socket.create_connection((self._host, self._port), timeout=60)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return s

    def _loop(self):
        conn = None
        while True:
            item = self._q.get()
            if item is None:  # shutdown sentinel
                if conn is not None:
                    conn.close()
                return
            fut, fn, args = item
            try:
                if conn is None:
                    conn = self._connect()
                send_msg(conn, pickle.dumps((fn, args), protocol=pickle.HIGHEST_PROTOCOL))
                reply = recv_msg(conn)
                if reply is None:
                    raise ConnectionError("fetch server closed the connection")
                status, payload = pickle.loads(reply)
                if status == "ok":
                    fut._set(payload, None)
                else:
                    fut._set(None, RuntimeError(f"fetch server: {payload}"))
            except Exception as e:  # noqa: BLE001 - reconnect next request, surface now
                try:
                    if conn is not None:
                        conn.close()
                except Exception:  # noqa: BLE001
                    pass
                conn = None
                fut._set(None, e)

    def shutdown(self, wait=False):  # parity with ProcessPoolExecutor
        self._q.put(None)


def _make_socket_pools(n: int):
    host = os.environ.get("NGL_FETCH_SERVER_HOST")
    port = os.environ.get("NGL_FETCH_SERVER_PORT")
    if not host or not port:
        raise RuntimeError(
            "socket fetch backend needs NGL_FETCH_SERVER_HOST + NGL_FETCH_SERVER_PORT")
    return [_SocketPool(host, int(port)) for _ in range(n)]
