"""Viewer-bundle serving and middleauth storage_state, without a browser.

Token resolution itself lives in ngllib.auth and is tested in test_auth.py.
"""

import json

import json as _json

from ngllib.chrome import (
    MIDDLEAUTH_STORAGE_KEY,
    cave_storage_state,
    middleauth_hosts,
    storage_state_for_origin,
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


def test_storage_state_file_is_rekeyed_to_the_live_origin(tmp_path):
    """The viewer is served from a loopback port that differs per process, and
    Playwright keys localStorage by origin -- so a credential file captured once
    has to be re-keyed on load or it silently does not apply."""
    f = tmp_path / "state.json"
    f.write_text(_json.dumps({
        "cookies": [{"name": "s", "domain": ".flywire-daf.com", "value": "x"}],
        "origins": [{"origin": "https://ngl.local",
                     "localStorage": [{"name": "auth_token_v2_x", "value": "tok"}]}],
    }))
    out = storage_state_for_origin(str(f), "http://127.0.0.1:54321")
    assert out["origins"][0]["origin"] == "http://127.0.0.1:54321"
    assert out["origins"][0]["localStorage"][0]["value"] == "tok"   # entries kept
    assert out["cookies"][0]["domain"] == ".flywire-daf.com"        # cookies untouched
