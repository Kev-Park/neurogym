"""Shared viewer-state math (gate 1): the transitions both backends run."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ngllib import state as S


def _state(**over):
    st = {"position": [100.0, 200.0, 30.0], "crossSectionScale": 2.0,
          "projectionOrientation": [0.0, 0.0, 0.0, 1.0], "projectionScale": 14000.0,
          "segments": ["720575940603464672"]}
    st.update(over)
    return st


def _edit(dpos=(0, 0, 0), dxs=0.0, dorient=(0, 0, 0), dps=0.0):
    return {"action_type": 3, "delta_pos": np.asarray(dpos, np.float32),
            "delta_xs_scale": np.asarray([dxs], np.float32),
            "delta_orient": np.asarray(dorient, np.float32),
            "delta_proj_scale": np.asarray([dps], np.float32)}


def test_coerce_defaults_and_requires():
    st = S.coerce({"position": (1, 2, 3), "crossSectionScale": 1, "projectionScale": 5,
                   "segments": [1, "!2"]})
    assert st["projectionOrientation"] == S.IDENTITY_QUAT
    assert st["position"] == [1.0, 2.0, 3.0]
    assert st["segments"] == ["1", "!2"]
    with pytest.raises(ValueError, match="crossSectionScale"):
        S.coerce({"position": [0, 0, 0], "projectionScale": 1})


def test_state_edit_is_pure_and_adds_deltas():
    st = _state()
    new = S.apply_state_edit(st, _edit(dpos=(10, -5, 1), dxs=0.5, dps=-2000), "euler")
    assert st == _state()                       # input untouched
    assert new["position"] == [110.0, 195.0, 31.0]
    assert new["crossSectionScale"] == 2.5
    assert new["projectionScale"] == 12000.0


def test_zoom_to_zero_or_below_keeps_previous_value():
    """NG's verifyFinitePositiveFloat rejects the value; the old one stands."""
    st = _state(projectionScale=2000.0)
    assert S.apply_state_edit(st, _edit(dps=-2000), "euler")["projectionScale"] == 2000.0
    assert S.apply_state_edit(st, _edit(dps=-5000), "euler")["projectionScale"] == 2000.0
    assert S.apply_state_edit(st, _edit(dps=-1999), "euler")["projectionScale"] == 1.0
    assert S.next_projection_scale(7.0, float("nan")) == 7.0
    assert S.next_projection_scale(7.0, float("inf")) == 7.0


def test_zoom_upper_bound():
    assert S.next_projection_scale(1.0, 1e9) == S.PROJECTION_SCALE_MAX


def test_quaternion_mode_normalizes():
    st = _state(projectionOrientation=[0.0, 0.0, 0.0, 1.0])
    new = S.apply_state_edit(st, _edit(dorient=(0.3, 0.0, 0.0, 0.0)), "quaternion")
    q = new["projectionOrientation"]
    assert math.isclose(sum(x * x for x in q), 1.0, abs_tol=1e-12)
    assert q[0] > 0 and q[3] > 0


def test_euler_mode_rotation_round_trips_to_unit_quaternion():
    st = _state()
    new = S.apply_state_edit(st, _edit(dorient=(0.2, -0.1, 0.05)), "euler")
    q = new["projectionOrientation"]
    assert math.isclose(sum(x * x for x in q), 1.0, abs_tol=1e-9)
    assert len(S.orientation_obs(new, "euler")) == 3
    assert len(S.orientation_obs(new, "quaternion")) == 4


def test_normalized_quaternion_zero_is_identity():
    assert S.normalized_quaternion([0, 0, 0, 0]) == S.IDENTITY_QUAT


def test_visible_segments_drops_hidden():
    assert S.visible_segments(["1", "!2", 3]) == ("1", "3")


def test_toggle_select_matches_neuroglancer():
    st = _state(segments=["1", "!2"])
    assert S.toggle_select(st, "1")["segments"] == ["!1", "!2"]      # visible -> hidden
    assert S.toggle_select(st, "2")["segments"] == ["1", "2"]        # hidden -> visible
    assert S.toggle_select(st, 3)["segments"] == ["1", "!2", "3"]    # new -> appended
    assert st["segments"] == ["1", "!2"]


def test_move_to():
    new = S.move_to(_state(), np.array([1, 2, 3]))
    assert new["position"] == [1.0, 2.0, 3.0]


def test_modifiers_to_str():
    assert S.modifiers_to_str([1, 0, 1]) == "Shift, Alt"
    assert S.modifiers_to_str([0, 0, 0]) == ""


def test_next_cross_section_scale_keeps_previous_at_or_below_zero():
    """MEASURED (probe_xs_boundary, job 972299): NG leaves the whole viewer
    state unreadable for crossSectionScale <= 0, exactly as for projectionScale,
    so the 2D zoom gets the same keep-previous rule."""
    assert S.next_cross_section_scale(2.0, 3.0) == 3.0
    assert S.next_cross_section_scale(2.0, 0.0) == 2.0
    assert S.next_cross_section_scale(2.0, -5.0) == 2.0
    assert S.next_cross_section_scale(2.0, float("nan")) == 2.0
    assert S.next_cross_section_scale(2.0, float("inf")) == 2.0


def test_cross_section_scale_has_our_upper_bound_not_ng_s():
    # NG accepted 1e6 verbatim; the cap is ours, to bound a zoom-out walk.
    assert S.next_cross_section_scale(2.0, 1e6) == S.CROSS_SECTION_SCALE_MAX
    assert S.CROSS_SECTION_SCALE_MAX == 64.0


def test_state_edit_applies_the_2d_zoom_rule():
    st = {"position": [1.0, 2.0, 3.0], "crossSectionScale": 2.0,
          "projectionOrientation": [0.0, 0.0, 0.0, 1.0], "projectionScale": 1000.0,
          "segments": []}
    act = {"delta_pos": [1.0, 0.0, 0.0], "delta_xs_scale": [-5.0],
           "delta_orient": [0.0, 0.0, 0.0], "delta_proj_scale": [0.0]}
    out = S.apply_state_edit(st, act, "euler")
    # The rejected zoom keeps its previous value; the rest of the edit applies.
    assert out["crossSectionScale"] == 2.0
    assert out["position"][0] == 2.0
