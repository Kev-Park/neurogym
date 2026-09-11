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


def resample_em(tile, overscan: float = 1.0) -> np.ndarray:
    """EM tile -> the pane's greyscale raster (PANE_H x PANE uint8).

    Calibrated chain: subpixel-phase tile -> GL-linear resample to the 900x867
    CSS pane -> Chrome's area-average capture downscale, then EM_GAIN.

    With `overscan` > 1 the tile covers a proportionally larger region and the
    raster is scaled up by the same factor, so the nm-per-pixel is unchanged
    and a pane-sized crop out of it is the same pixel grid a direct fetch would
    have produced. That is what lets a viewer move re-crop locally instead of
    refetching -- see NativeEnvironment._tile_cache.
    """
    w = int(round(900 * overscan))
    h = int(round(867 * overscan))
    ow = int(round(PANE * overscan))
    oh = int(round(PANE_H * overscan))
    big = Image.fromarray(tile).resize((w, h), Image.BILINEAR)
    img = np.asarray(big.resize((ow, oh), Image.BOX)).astype(np.float32)
    return np.clip(img * EM_GAIN, 0, 255).astype(np.uint8)


def crop_pane(raster, centre_nm, ext_nm, want_centre_nm):
    """Pane-sized window out of an overscanned raster, or None if it does not
    fit entirely inside.

    `raster` covers `ext_nm` about `centre_nm`; the window is PANE x PANE_H
    about `want_centre_nm`. Offsets round to whole raster pixels, which is at
    most half a raster pixel of registration error -- under a quarter of a
    captured pixel at the shipping resolution, and so below the level the
    parity metric resolves.
    """
    h, w = raster.shape[:2]
    nm_per_px_x = ext_nm[0] / w
    nm_per_px_y = ext_nm[1] / h
    dx = int(round((want_centre_nm[0] - centre_nm[0]) / nm_per_px_x))
    dy = int(round((want_centre_nm[1] - centre_nm[1]) / nm_per_px_y))
    x0 = w // 2 + dx - PANE // 2
    y0 = h // 2 + dy - PANE_H // 2
    if x0 < 0 or y0 < 0 or x0 + PANE > w or y0 + PANE_H > h:
        return None
    return raster[y0:y0 + PANE_H, x0:x0 + PANE]


def draw_crosshair(rgb: np.ndarray) -> None:
    """One-sided crosshair (red +x, green +y) at alpha 0.5, in place."""
    cy, cx = PANE_H // 2, PANE // 2
    length = int(min(900, 867) / 4 / 2)
    row = rgb[cy, cx:cx + length]
    rgb[cy, cx:cx + length] = 0.5 * np.array([255, 0, 0]) + 0.5 * row
    colm = rgb[cy:cy + length, cx]
    rgb[cy:cy + length, cx] = 0.5 * np.array([0, 255, 0]) + 0.5 * colm


def tint_plane(plane_gray, ids, visible) -> np.ndarray:
    """Colourize the 3D pane's cross-section, the way NG does.

    Neuroglancer draws the SAME segmentation layer on the perspective view's
    slice, so with a segment selected its cross-section is tinted there too,
    and with NOTHING visible the plane goes fully colourized under
    SHOW_ALL_SEGMENTS. Measured on matched frames: Chrome's plane pixels carry
    a channel spread of 5.76 where ours carried 0.00.

    The id map is the 2D pane's, sampled at the registration-shifted centre
    while the plane tile is fetched unshifted -- 3 captured px apart. The plane
    is drawn at roughly 100-200 px on screen, so that is sub-pixel there; it is
    reused rather than refetched precisely so a selection change stays free.
    """
    g = np.asarray(plane_gray)
    if ids is None or g.ndim != 2:
        return g
    h, w = g.shape
    rows = np.minimum((np.arange(h) * ids.shape[0]) // h, ids.shape[0] - 1)
    cols = np.minimum((np.arange(w) * ids.shape[1]) // w, ids.shape[1] - 1)
    pid = ids[rows][:, cols]
    rgb = np.repeat(g.astype(np.float32)[..., None], 3, axis=2)
    vis = [int(v) for v in visible]
    if not vis:
        tint_all(rgb, pid)
    else:
        for rid in vis:
            m = pid == rid
            if m.any():
                col = np.asarray(segment_color(rid)) * 255.0
                rgb[m] = 0.5 * col[None, :] + 0.5 * rgb[m]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def compose_left_parts(em_gray, ids, visible) -> np.ndarray:
    """2D pane canvas from the CACHED raster + id map and the CURRENT selection.

    Splitting composition this way is what lets a selection change render with
    no fetch at all, which is what Chrome does: it already holds the
    segmentation chunk and only re-tints. Keying the tile fetch on the
    selection instead made the simulator's 2D pane lag a click by a step in the
    ordinary case and never respond at all on the deselect-to-SHOW_ALL
    transition (probe_select_dynamics, 883367).

    `visible` is the visible segment set; empty means NG's SHOW_ALL_SEGMENTS,
    where every segment paints. Draw order matches compose_left: EM, then tint,
    then crosshair.
    """
    canvas = np.zeros((PANE, PANE, 3), dtype=np.uint8)
    if em_gray is None:
        return canvas
    rgb = np.repeat(np.asarray(em_gray, dtype=np.float32)[..., None], 3, axis=2)
    if ids is not None:
        vis = [int(v) for v in visible]
        if not vis:
            tint_all(rgb, ids)
        else:
            for rid in vis:
                m = ids == rid
                if m.any():
                    col = np.asarray(segment_color(rid)) * 255.0
                    rgb[m] = 0.5 * col[None, :] + 0.5 * rgb[m]
    draw_crosshair(rgb)
    canvas[TOOLBAR:] = np.clip(rgb, 0, 255).astype(np.uint8)
    return canvas


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
