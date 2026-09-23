"""Start-URL normalization: legacy states, graphene rewriting, viewer choice."""

import json
import urllib.parse
from pathlib import Path

import pytest

from ngllib.dataset import (
    is_legacy_state,
    legacy_to_modern_state,
    normalize_start_url,
    rewrite_graphene_sources,
    split_state_url,
)
from ngllib.errors import ProviderError

# The link a user actually pastes from ngl.flywire.ai: legacy state format and
# a plain graphene source (both unusable as-is).
FLYWIRE_LEGACY = {
    "layers": [
        {"source": "precomputed://gs://flywire_em/aligned/v1", "type": "image", "name": "EM"},
        {"source": "graphene://https://prodv1.flywire-daf.com/segmentation/1.0/flywire_public",
         "type": "segmentation_with_graph", "segments": ["720575940625112137"],
         "name": "flywire_public (pre-edit)"},
    ],
    "navigation": {"pose": {"position": {"voxelSize": [4, 4, 40],
                                         "voxelCoordinates": [215675.25, 61215.0, 3828.825]}},
                   "zoomFactor": 8.0},
    "perspectiveZoom": 16485.27179446025,
    "perspectiveOrientation": [-0.47, 0.80, -0.30, 0.19],
    "showDefaultAnnotations": False,
    "layout": "xy-3d",
}


def url_for(state, prefix="https://ngl.flywire.ai/"):
    return prefix + "#!" + urllib.parse.quote(json.dumps(state))


def test_detects_legacy_and_modern_states():
    assert is_legacy_state(FLYWIRE_LEGACY)
    assert not is_legacy_state({"position": [1, 2, 3], "layers": []})


def test_legacy_conversion_maps_every_field_the_environment_drives():
    st = legacy_to_modern_state(FLYWIRE_LEGACY)
    assert st["position"] == [215675.25, 61215.0, 3828.825]
    assert st["dimensions"] == {"x": [4e-9, "m"], "y": [4e-9, "m"], "z": [4e-8, "m"]}
    assert st["crossSectionScale"] == pytest.approx(2.0)          # zoomFactor 8 / 4 nm
    assert st["projectionScale"] == pytest.approx(16485.27179446025)
    assert st["projectionOrientation"] == [-0.47, 0.80, -0.30, 0.19]
    assert st["layers"][1]["type"] == "segmentation"              # ..._with_graph
    assert st["layout"] == "xy-3d" and st["showDefaultAnnotations"] is False
    assert "navigation" not in st


def test_legacy_conversion_without_coordinates_is_a_clear_error():
    with pytest.raises(ProviderError, match="cannot convert"):
        legacy_to_modern_state({"navigation": {"zoomFactor": 8.0}, "layers": []})


def test_graphene_sources_are_rewritten_for_middleauth():
    st = rewrite_graphene_sources({"layers": [
        {"source": "graphene://https://prodv1.flywire-daf.com/segmentation/1.0/x"},
        {"source": "graphene://middleauth+https://prodv1.flywire-daf.com/y"},   # already
        {"source": "precomputed://gs://bucket/x"},                              # untouched
        {"source": {"url": "graphene://https://host/z"}},
        {"source": ["precomputed://gs://b", "graphene://https://host/w"]},
    ]})
    srcs = [lay["source"] for lay in st["layers"]]
    assert srcs[0] == "graphene://middleauth+https://prodv1.flywire-daf.com/segmentation/1.0/x"
    assert srcs[1] == "graphene://middleauth+https://prodv1.flywire-daf.com/y"
    assert srcs[2] == "precomputed://gs://bucket/x"
    assert srcs[3] == {"url": "graphene://middleauth+https://host/z"}
    assert srcs[4] == ["precomputed://gs://b", "graphene://middleauth+https://host/w"]


def test_normalize_a_pasted_flywire_link():
    prefix, state = normalize_start_url(url_for(FLYWIRE_LEGACY))
    assert prefix == "https://ngl.flywire.ai/"      # the caller decides the viewer
    assert state["position"] == [215675.25, 61215.0, 3828.825]
    assert "middleauth+" in state["layers"][1]["source"]


def test_normalize_leaves_a_modern_public_link_alone():
    modern = {"position": [1.0, 2.0, 3.0], "crossSectionScale": 2.0, "projectionScale": 100.0,
              "layers": [{"source": "precomputed://gs://flywire_v141_m783", "type": "segmentation",
                          "segments": ["1"]}]}
    prefix, state = normalize_start_url(url_for(modern, "https://neuroglancer-demo.appspot.com/"))
    assert state == modern and prefix == "https://neuroglancer-demo.appspot.com/"


def test_the_committed_graphene_fixture_normalizes():
    url = (Path(__file__).parent / "graphene_start_url.txt").read_text().strip()
    _, raw = split_state_url(url)
    _, state = normalize_start_url(url)
    # Already modern and already middleauth: normalization is a no-op here.
    assert not is_legacy_state(raw)
    assert state["layers"][1]["source"].startswith("graphene://middleauth+")
    assert state["position"] == raw["position"]
