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

# Click geometry, measured off the live DOM (probe_select_parity.py --mode
# rects, 2026-09-10). `mouse_xy` is relative to `.neuroglancer-layer-group-
# viewer` -- that is where execute_click places its events -- and that element
# sits at page y=24, while the data panels sit at page y=47 and are 853 CSS px
# tall. So in CLICK coordinates a panel starts at y=23 and its centre is at
# 449.5, NOT at CSS_TOOLBAR + CSS_VIEW_H/2 = 466.5.
#
# These are deliberately separate from CSS_TOOLBAR/CSS_VIEW_H above, which are
# CAPTURE geometry: the fetch extent and the composed canvas are calibrated
# against them together with LEFT_SHIFT_PX and EM_GAIN, and changing those
# changes the observation for every existing run.
PANEL_TOP_CLICK = 23.0
PANEL_H_CLICK = 853.0
PANEL_CX_CLICK = CSS_PANE / 2.0
PANEL_CY_CLICK = PANEL_TOP_CLICK + PANEL_H_CLICK / 2.0   # 449.5


def pane_extents_nm(xs_scale: float) -> tuple[float, float]:
    return float(xs_scale) * CSS_PANE * 4.0, float(xs_scale) * CSS_VIEW_H * 4.0


def shifted_fetch_center_nm(pos_nm: np.ndarray, ext: tuple[float, float]):
    """Apply the baked registration correction to the 2D fetch center."""
    return pos_nm + np.array([
        LEFT_SHIFT_PX[1] * ext[0] / PANE,
        LEFT_SHIFT_PX[0] * ext[1] / PANE_H, 0.0])


def tint_all(rgb: np.ndarray, ids: np.ndarray) -> None:
    """Colour every segment in an id tile, in place.

    One LUT pass rather than a mask per id: a pane at this zoom holds hundreds
    of distinct segments, and this path runs whenever the selection is empty.
    """
    uniq, inv = np.unique(ids, return_inverse=True)
    lut = np.zeros((uniq.size, 3), dtype=np.float32)
    for k, seg in enumerate(uniq):
        if seg:
            lut[k] = segment_color(int(seg))
    col = lut[inv].reshape(ids.shape + (3,)) * 255.0
    nz = ids != 0
    rgb[nz] = 0.5 * col[nz] + 0.5 * rgb[nz]


def compose_left(tile, label_mask, root_id) -> np.ndarray:
    """2D xy EM pane canvas (PANE x PANE x 3 uint8): calibrated filter chain
    + segment tint + one-sided crosshair + toolbar strip.

    Neuroglancer's `select` toggles segments into a SET and tints each with its
    own colour, so `root_id` may be a single id or a sequence, and
    `label_mask` correspondingly a single mask or a {root_id: mask} dict.
    Single-id callers keep working unchanged. With NOTHING visible, NG colours
    the whole slice instead (SHOW_ALL_SEGMENTS): pass the id tile from
    EMTiles.label_ids as `label_mask` and an empty `root_id`.
    """
    canvas = np.zeros((PANE, PANE, 3), dtype=np.uint8)
    if tile is None:
        return canvas
    big = Image.fromarray(tile).resize((900, 867), Image.BILINEAR)
    img = np.asarray(big.resize((PANE, PANE_H), Image.BOX)
                     ).astype(np.float32) * EM_GAIN
    rgb = np.repeat(img[..., None], 3, axis=2)
    if label_mask is not None:
        if isinstance(label_mask, dict):
            # Painted in the selection's own order so overlaps resolve the way
            # a caller listed them, not by dict iteration accident.
            ids = [int(r) for r in root_id] if not isinstance(
                root_id, (int, str)) else [int(root_id)]
            pairs = [(r, label_mask[r]) for r in ids if label_mask.get(r) is not None]
        elif label_mask.dtype != bool:
            # SHOW_ALL_SEGMENTS: nothing is visible, so `label_mask` is the id
            # tile and every non-zero segment paints in its own colour at the
            # same selectedAlpha (hideSegmentZero keeps 0 as background).
            tint_all(rgb, label_mask)
            pairs = []
        else:
            rid0 = root_id[0] if not isinstance(root_id, (int, str)) else root_id
            pairs = [(int(rid0), label_mask)]
        for rid, m in pairs:
            col = np.asarray(segment_color(int(rid))) * 255.0
            rgb[m] = 0.5 * col[None, :] + 0.5 * rgb[m]
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
