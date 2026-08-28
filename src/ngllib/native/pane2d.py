"""Shared pane geometry/constants + the calibrated 2D-pane composition.

Single source of truth for the capture geometry and calibrated constants
(previously environment.py-local) so NativeEnvironment (local mode) and the
per-node render service compose pixel-identical panes.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .colors import segment_color

VOXEL_NM = np.array([4.0, 4.0, 40.0])
# Calibrated: nm per projectionScale unit (parity campaign).
SCALE_CAL_NM = 4.07
# Calibrated: browser/native EM intensity ratio on grey 2D-pane pixels.
EM_GAIN = 0.978
# Calibrated: baked 2D-pane fetch-center correction in captured px (dy, dx).
LEFT_SHIFT_PX = (-3.0, 0.0)

# Capture geometry at capture_scale 0.5 (the calibrated configuration).
PANE = 450
TOOLBAR = 17
PANE_H = PANE - TOOLBAR
CSS_PANE = 900.0
CSS_TOOLBAR = 33.0
CSS_VIEW_H = 867.0


def pane_extents_nm(xs_scale: float) -> tuple[float, float]:
    return float(xs_scale) * CSS_PANE * 4.0, float(xs_scale) * CSS_VIEW_H * 4.0


def shifted_fetch_center_nm(pos_nm: np.ndarray, ext: tuple[float, float]):
    """Apply the baked registration correction to the 2D fetch center."""
    return pos_nm + np.array([
        LEFT_SHIFT_PX[1] * ext[0] / PANE,
        LEFT_SHIFT_PX[0] * ext[1] / PANE_H, 0.0])


def compose_left(tile, label_mask, root_id) -> np.ndarray:
    """2D xy EM pane canvas (PANE x PANE x 3 uint8): calibrated filter chain
    + segment tint + one-sided crosshair + toolbar strip."""
    canvas = np.zeros((PANE, PANE, 3), dtype=np.uint8)
    if tile is None:
        return canvas
    big = Image.fromarray(tile).resize((900, 867), Image.BILINEAR)
    img = np.asarray(big.resize((PANE, PANE_H), Image.BOX)
                     ).astype(np.float32) * EM_GAIN
    rgb = np.repeat(img[..., None], 3, axis=2)
    if label_mask is not None:
        col = np.asarray(segment_color(int(root_id))) * 255.0
        rgb[label_mask] = 0.5 * col[None, :] + 0.5 * rgb[label_mask]
    cy, cx = PANE_H // 2, PANE // 2
    length = int(min(900, 867) / 4 / 2)
    row = rgb[cy, cx:cx + length]
    rgb[cy, cx:cx + length] = 0.5 * np.array([255, 0, 0]) + 0.5 * row
    colm = rgb[cy:cy + length, cx]
    rgb[cy:cy + length, cx] = 0.5 * np.array([0, 255, 0]) + 0.5 * colm
    canvas[TOOLBAR:] = np.clip(rgb, 0, 255).astype(np.uint8)
    return canvas


def paste_right(pane_below_toolbar: np.ndarray) -> np.ndarray:
    """3D pane (PANE_H x PANE) -> PANE x PANE canvas with toolbar strip."""
    out = np.zeros((PANE, PANE, 3), dtype=np.uint8)
    out[TOOLBAR:] = pane_below_toolbar
    return out
