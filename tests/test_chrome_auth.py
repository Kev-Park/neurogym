"""Viewer-bundle serving and middleauth storage_state, without a browser.

Token resolution itself lives in ngllib.auth and is tested in test_auth.py.
"""

import json

from ngllib.chrome import (
    MIDDLEAUTH_STORAGE_KEY,
    cave_storage_state,
    dist_file,
    middleauth_hosts,
)

APP = "https://prodv1.flywire-daf.com"
LOGIN = "https://global.daf-apis.com/sticky_auth"


def test_middleauth_hosts_from_sources():
    state = {"layers": [
        {"source": "precomputed://gs://flywire_em/aligned/v1"},
        {"source": f"graphene://middleauth+{APP}/segmentation/1.0/flywire_public"},
        {"source": {"url": f"graphene://middleauth+{APP}/other"}},
        {"source": ["precomputed://gs://x", "graphene://middleauth+https://other.org/s"]},
    ]}
    assert middleauth_hosts(state) == [APP, "https://other.org"]


def test_middleauth_hosts_none_for_public_state():
    assert middleauth_hosts({"layers": [{"source": "precomputed://gs://flywire_v141_m783"}]}) == []


def test_cave_storage_state_shape():
    st = cave_storage_state("https://ngl.local", LOGIN, "tok", [APP])
    (origin,) = st["origins"]
    assert origin["origin"] == "https://ngl.local"
    (item,) = origin["localStorage"]
    assert item["name"] == f"{MIDDLEAUTH_STORAGE_KEY}_{LOGIN}"
    # appUrls is load-bearing: the provider throws UnverifiedApp without it.
    assert json.loads(item["value"]) == {
        "tokenType": "Bearer", "accessToken": "tok", "url": LOGIN, "appUrls": [APP]}


def test_dist_file_resolves_and_defaults_to_index(tmp_path):
    (tmp_path / "index.html").write_text("<html>")
    (tmp_path / "main.js").write_text("//")
    assert dist_file(tmp_path, "https://ngl.local/") == (tmp_path / "index.html").resolve()
    assert dist_file(tmp_path, "https://ngl.local/main.js") == (tmp_path / "main.js").resolve()
    # the state lives in the fragment, which never reaches the server
    assert dist_file(tmp_path, 'https://ngl.local/#!{"a":1}') == (tmp_path / "index.html").resolve()


def test_dist_file_rejects_missing_and_escaping_paths(tmp_path):
    (tmp_path / "index.html").write_text("<html>")
    assert dist_file(tmp_path, "https://ngl.local/nope.js") is None
    assert dist_file(tmp_path, "https://ngl.local/../../etc/passwd") is None
