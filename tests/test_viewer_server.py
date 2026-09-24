"""The loopback viewer server: what it serves, and that it is shared."""

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
