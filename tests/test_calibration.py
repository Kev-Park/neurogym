"""Freeze the calibrated constants (plan step 3 / gate 1).

Every value here was FITTED against Chrome and confirmed optimal by sweep;
an accidental edit would degrade parity silently. Changing one is a
deliberate act that comes with a parity re-run (gates 5 and 7), and this
test is updated in the same commit.
"""

from __future__ import annotations

from ngllib.renderer import CALIBRATED_CAPTURE_SCALE, CALIBRATED_WINDOW, PaneLayout
from ngllib.simulator import pane2d


def test_fitted_constants():
    assert pane2d.SCALE_CAL_NM == 4.07          # sweep peak 2026-09-10; 3.95/4.19 worse
    assert pane2d.EM_GAIN == 0.978              # optimal by probe_left_pane_parity
    assert pane2d.LEFT_SHIFT_PX == (-3.0, 0.0)  # registration pixel-exact
    assert pane2d.PLANE_EXT_SCALE == 1.0        # plane-extent sweep peak


def test_capture_geometry():
    assert (pane2d.PANE, pane2d.TOOLBAR, pane2d.PANE_H) == (450, 17, 433)
    # The 3D pane's own capture origin, fitted 2026-09-18 (job 922247).
    assert (pane2d.TOOLBAR_3D, pane2d.PANE_3D_SHIFT, pane2d.PANE_H_3D) == (20, 3, 430)
    assert (pane2d.CSS_PANE, pane2d.CSS_TOOLBAR, pane2d.CSS_VIEW_H) == (900.0, 33.0, 867.0)
    assert CALIBRATED_WINDOW == (1800, 900) and CALIBRATED_CAPTURE_SCALE == 0.5
    lay = PaneLayout(window_size=CALIBRATED_WINDOW, capture_scale=CALIBRATED_CAPTURE_SCALE,
                     left_pane=True, right_pane=True)
    assert lay.is_calibrated and lay.frame_shape == (450, 900, 3)


def test_click_geometry_is_separate_from_capture_geometry():
    """Measured off the live DOM (2026-09-10): panels start at click y=23 and
    are 853 CSS px tall. Deliberately NOT the capture constants above."""
    assert (pane2d.PANEL_TOP_CLICK, pane2d.PANEL_H_CLICK) == (23.0, 853.0)
    assert pane2d.PANEL_CY_CLICK == 449.5
    assert pane2d.PANEL_CX_CLICK == 450.0


def test_ui_mask_regions():
    assert pane2d.UI_REGIONS == (
        (0, 32, 0, 900), (0, 450, 0, 16), (416, 450, 0, 80), (16, 48, 868, 900),
        (416, 450, 820, 900),
        (0, 450, 450, 466))   # 3D pane left edge (axis labels), added 2026-09-18


def test_uncalibrated_layout_warns_not_rejects():
    import warnings

    lay = PaneLayout(window_size=(1600, 800), capture_scale=0.5)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        lay.warn_if_uncalibrated("test")
    assert len(w) == 1 and "not the calibrated geometry" in str(w[0].message)
