"""Cell-bounded picking never borrows targets from neighbouring actions."""

from types import SimpleNamespace

import numpy as np

from ngllib.simulator.renderer import SimulatorRenderer
from ngllib.simulator.pane2d import PANE, PANE_H_3D, TOOLBAR, TOOLBAR_3D


def renderer():
    r = SimulatorRenderer.__new__(SimulatorRenderer)
    r.click_grid = (16, 32)
    r.click_bounds = (0, 23, 1800, 876)
    r.layout = SimpleNamespace(capture_scale=0.5)
    r._state = {"segments": ["7"], "position": [0, 0, 0],
                "crossSectionScale": 1, "projectionOrientation": [0, 0, 0, 1],
                "projectionScale": 1000}
    r._voxel_nm = np.ones(3)
    r._ensure_meshes = lambda ids: None
    return r


def test_2d_rescues_corner_but_not_adjacent_cell():
    r = renderer()
    ids = np.zeros((433, PANE), dtype=np.uint64)
    # Cell row 2, col 2 spans CSS x=[112.5,168.75), y=[129.625,182.9375).
    x, y = 140.625, 156.28125
    ids[49, 57] = 7  # inside near corner, outside old point query
    ids[61, 85] = 8  # adjacent cell, closer to cursor than the corner
    r._tiles = {"ids": ids, "em": np.zeros_like(ids)}
    assert r._segment_under_2d(x, y) == 7
    ids[49, 57] = 0
    assert r._segment_under_2d(x, y) is None


def test_2d_nearest_displayed_label_and_missing_tiles():
    r = renderer()
    ids = np.zeros((433, PANE), dtype=np.uint64)
    ids[61, 70] = 9  # centre
    ids[49, 57] = 7
    r._tiles = {"ids": ids, "em": np.zeros_like(ids)}
    assert r._segment_under_2d(140.625, 156.28125) == 9
    r._tiles["em"] = None
    assert r._segment_under_2d(140.625, 156.28125) is None


def test_3d_selection_and_movement_rescue_same_cell_only():
    r = renderer()
    depth = np.ones((433, PANE))
    # Same row, col 18: x=[1012.5,1068.75), with the 3D pane offset.
    x, y = 1040.625, 156.28125
    depth[46, 57] = 0.5
    depth[58, 85] = 0.1  # adjacent cell must not win despite smaller depth
    r._renderer = SimpleNamespace(pick_depth=lambda *a, **k: (depth, np.eye(4), np.eye(4)))
    assert r._segment_under_3d(x, y) == "7"
    assert r._pick_3d_world(x, y) is not None
    depth[46, 57] = 1
    assert r._segment_under_3d(x, y) is None
    assert r._pick_3d_world(x, y) is None


def test_half_open_boundaries_and_pane_edge():
    r = renderer()
    mask = np.ones((PANE_H_3D, PANE), dtype=bool)
    assert r._cell_sample(mask, 1800, 100, pane_x=PANE, toolbar=TOOLBAR_3D) is None
    assert r._cell_sample(mask, 900, 876, pane_x=PANE, toolbar=TOOLBAR_3D) is None
    assert r._cell_sample(mask, 899, 100, pane_x=PANE, toolbar=TOOLBAR_3D) is None
    # Exhaustively check every cell: selected pixel belongs to that cell.
    for row in range(16):
        for col in range(32):
            x = (col + 0.5) * 56.25
            y = 23 + (row + 0.5) * 853 / 16
            pane_x = PANE if col >= 16 else 0
            toolbar = TOOLBAR_3D if col >= 16 else TOOLBAR
            sample = r._cell_sample(mask, x, y, pane_x=pane_x, toolbar=toolbar)
            assert sample is not None
            py, px = sample
            sx, sy = (px + pane_x + 0.5) * 2, (py + toolbar + 0.5) * 2
            assert int(sx / 56.25) == col
            assert int((sy - 23) / (853 / 16)) == row
