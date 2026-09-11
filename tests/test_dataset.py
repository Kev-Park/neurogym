"""DatasetSpec and start-URL handling (gate 1)."""

from __future__ import annotations

import pytest

from ngllib import DatasetSpec
from ngllib.dataset import (
    default_start_url,
    merge_state,
    ngl_state_from_viewer,
    split_state_url,
    state_to_url,
)
from ngllib.errors import ProviderError
from ngllib.simulator.pane2d import CALIBRATED_DATASET


def test_packaged_config_declares_the_calibrated_dataset():
    """The simulator's constants were fitted on exactly what config.json
    points Chrome at; the two must agree or every parity number is moot."""
    assert DatasetSpec.from_config() == CALIBRATED_DATASET
    assert CALIBRATED_DATASET.voxel_nm == (4.0, 4.0, 40.0)


def test_split_and_rebuild_round_trip():
    url = default_start_url()
    prefix, st = split_state_url(url)
    assert prefix.startswith("https://")
    assert "position" in st and "layers" in st
    prefix2, st2 = split_state_url(state_to_url(prefix, st))
    assert (prefix2, st2) == (prefix, st)


def test_ngl_state_view_reads_segments_off_the_segmentation_layer():
    _, viewer = split_state_url(default_start_url())
    st = ngl_state_from_viewer(viewer)
    assert set(st) == {"position", "crossSectionScale", "projectionOrientation",
                       "projectionScale", "segments"}
    assert any(s.startswith("!") for s in st["segments"])   # the default carries hidden ones


def test_merge_state_overlays_first_class_fields_and_segments():
    _, base = split_state_url(default_start_url())
    merged = merge_state(base, {"position": [1, 2, 3], "segments": ["9"],
                                "extra": {"layout": "4panel", "position": [7, 7, 7]}})
    assert merged["position"] == [1.0, 2.0, 3.0]      # first-class beats extra
    assert merged["layout"] == "4panel"
    seg_layers = [l for l in merged["layers"] if l["type"] == "segmentation"]
    assert seg_layers[0]["segments"] == ["9"]
    assert base["position"] != [1.0, 2.0, 3.0]         # input untouched


def test_dataset_requires_dimensions_and_both_layers():
    with pytest.raises(ProviderError):
        DatasetSpec.from_state({"layers": []})
    with pytest.raises(ProviderError):
        DatasetSpec.from_state({"dimensions": {"x": [4e-9, "m"], "y": [4e-9, "m"],
                                               "z": [4e-8, "m"]}, "layers": []})


def test_dataset_is_hashable():
    {DatasetSpec.from_config(): 1}
