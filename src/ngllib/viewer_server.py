"""A loopback HTTP server for the packaged Neuroglancer build.

Serving the viewer through Playwright's `page.route` makes every asset a round
trip Chrome -> driver -> a Python callback, on the same connection the step loop
uses. Measured on a GPU node (job 967718): ~34 route calls and ~4 MB per reset,
1.4 s of handler time against a 2.7 s reset -- about half the reset. Cache
headers do not help, because each episode recycles the BrowserContext and a
Playwright context has its own empty HTTP cache.

Serving over loopback instead lets Chrome fetch the bundle in its own network
threads, in parallel, with no Python involved. One server per process, shared by
every renderer in it (16 envs share a process in production), started on demand
and stopped at exit. Bound to 127.0.0.1 with a random port: nothing outside the
node can reach it.
"""

from __future__ import annotations

import atexit
import functools
import http.server
import logging
import socketserver
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_SERVERS: dict[Path, str] = {}
_LOCK = threading.Lock()


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Static files, no request logging, immutable caching for hashed names."""

    def log_message(self, fmt, *args):  # noqa: D102 - silence per-request logs
        pass

    def end_headers(self):
        # Webpack content-hashes every asset, so only index.html is mutable.
        # Within one context this saves nothing (fresh contexts start cold) but
        # it is correct, and it helps any renderer that reuses a context.
        if self.path.endswith("/") or self.path.endswith("index.html"):
            self.send_header("Cache-Control", "no-store")
        else:
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        super().end_headers()


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(dist: Path) -> str:
    """Origin serving `dist` on loopback, e.g. `http://127.0.0.1:54321`.

    Idempotent per directory: the first caller in the process starts the
    server, everyone else gets the same origin.
    """
    dist = Path(dist).resolve()
    with _LOCK:
        origin = _SERVERS.get(dist)
        if origin is not None:
            return origin
        handler = functools.partial(_Handler, directory=str(dist))
        httpd = _Server(("127.0.0.1", 0), handler)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, name=f"ngl-viewer:{port}",
                         daemon=True).start()
        atexit.register(httpd.shutdown)
        origin = f"http://127.0.0.1:{port}"
        _SERVERS[dist] = origin
        logger.info("serving viewer %s at %s", dist, origin)
        return origin
