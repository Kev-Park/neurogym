"""The loopback viewer server: what it serves, and that it is shared."""

import os
import urllib.error
import urllib.request

import pytest

from ngllib import viewer_server


@pytest.fixture
def dist(tmp_path):
    (tmp_path / "index.html").write_text("<html>viewer</html>")
    (tmp_path / "main.abc123.js").write_text("console.log(1)")
    return tmp_path


def get(origin, path):
    with urllib.request.urlopen(f"{origin}{path}", timeout=5) as r:
        return r.status, r.read(), dict(r.headers)


def test_serves_the_bundle_on_loopback(dist):
    origin = viewer_server.serve(dist)
    assert origin.startswith("http://127.0.0.1:")      # never reachable off-node
    status, body, _ = get(origin, "/index.html")
    assert status == 200 and b"viewer" in body
    status, body, _ = get(origin, "/main.abc123.js")
    assert status == 200 and b"console.log" in body


def test_hashed_assets_are_immutable_and_index_is_not(dist):
    origin = viewer_server.serve(dist)
    _, _, headers = get(origin, "/main.abc123.js")
    assert "immutable" in headers["Cache-Control"]
    _, _, headers = get(origin, "/index.html")
    assert headers["Cache-Control"] == "no-store"


def test_one_server_per_directory(dist):
    # 16 envs share a process in production; they must not start 16 servers.
    assert viewer_server.serve(dist) == viewer_server.serve(dist)


def test_missing_file_is_a_404(dist):
    origin = viewer_server.serve(dist)
    with pytest.raises(urllib.error.HTTPError) as e:
        get(origin, "/nope.js")
    assert e.value.code == 404


def test_a_forked_child_would_serve_its_own(dist, monkeypatch):
    """The cache is keyed by PID: a process that inherits it must not hand
    Chrome a URL only its parent can answer."""
    first = viewer_server.serve(dist)
    # Capture the real pid BEFORE patching: `os` is the shared module object,
    # so a lambda that calls os.getpid() would call itself.
    child_pid = os.getpid() + 1
    monkeypatch.setattr(viewer_server.os, "getpid", lambda: child_pid)
    second = viewer_server.serve(dist)
    assert second != first
    status, body, _ = get(second, "/index.html")
    assert status == 200 and b"viewer" in body


def test_bind_failure_names_the_cause(dist, monkeypatch):
    from ngllib.errors import RendererError

    def boom(*a, **k):
        raise OSError(98, "Address already in use")

    monkeypatch.setattr(viewer_server, "_Server", boom)
    monkeypatch.setattr(viewer_server, "_SERVERS", {})
    with pytest.raises(RendererError, match="could not bind a loopback port"):
        viewer_server.serve(dist)
