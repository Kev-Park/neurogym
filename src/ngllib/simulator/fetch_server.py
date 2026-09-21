"""Socket fetch server for disaggregated fetch (runs on the CPU node).

Env-runners on the GPU node connect OUT to this server -- the only direction a
directional firewall allows (GPU->CPU open, CPU->GPU blocked) -- send pickled
(worker_fn, args), and get the decoded numpy result back on the same connection.
Each client connection is served by its OWN forked process with an isolated
chunk/mesh LRU, so a sticky client connection preserves the per-worker locality
(216x) the process pools have. Decode is GIL-heavy, hence processes not threads.

No Ray, no login proxy, direct full node-to-node bandwidth.

    NGL_FETCH_SERVER_PORT=<port> uv run --no-sync python -m ngllib.simulator.fetch_server
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import socket
import sys

from .fetch_pool import recv_msg, send_msg


def _handle(conn: socket.socket) -> None:
    """Serve one client connection until it closes. Own-process LRU (built lazily
    inside the worker fns' module-level caches, isolated by the fork)."""
    try:
        while True:
            msg = recv_msg(conn)
            if msg is None:
                return
            try:
                fn, args = pickle.loads(msg)
                result = fn(*args)
                send_msg(conn, pickle.dumps(("ok", result),
                                            protocol=pickle.HIGHEST_PROTOCOL))
            except Exception as e:  # noqa: BLE001 - report to the client, keep serving
                send_msg(conn, pickle.dumps(("err", repr(e))))
    except (ConnectionError, OSError):
        pass
    finally:
        try:
            conn.close()
        except OSError:
            pass


def serve(port: int) -> None:
    # Bind all interfaces so the GPU node can reach us; fork per connection (CPU
    # node has no CUDA/EGL state, so fork is safe and inherits the accepted fd).
    ctx = mp.get_context("fork")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(128)
    print(f"[fetch-server] listening 0.0.0.0:{port} host={socket.gethostname()} "
          f"pid={os.getpid()}", flush=True)
    children: list = []
    while True:
        conn, addr = srv.accept()
        p = ctx.Process(target=_handle, args=(conn,), daemon=True)
        p.start()
        conn.close()  # parent drops its copy; the child owns it
        children = [c for c in children if c.is_alive()]
        children.append(p)


def main() -> int:
    port = int(os.environ.get("NGL_FETCH_SERVER_PORT")
               or (sys.argv[1] if len(sys.argv) > 1 else 0))
    if not port:
        raise SystemExit("set NGL_FETCH_SERVER_PORT or pass a port arg")
    serve(port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
