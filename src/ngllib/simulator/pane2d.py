"""Shared pane geometry, the calibrated constants, and the 2D-pane composition.

Single source of truth for the capture geometry and every constant that was
FITTED against Chrome rather than derived. Each one below names the
measurement that fixes it; change any of them only with a parity re-run
(gates 5 and 7 in renderer_seam_plan.md), and keep the freeze test in
tests/test_calibration.py in step.

All of it was measured on ONE dataset and ONE geometry (CALIBRATED_DATASET,
window 1800x900 CSS at capture_scale 0.5). Analytic derivation was tried and
refuted (plan step 3): the DOM says the panels are 853 CSS px at y=47, and
that geometry scores 0.55/0.44/0.32/0.25 against the shipping 867/17/433 at
0.861 -- the capture and click frames are genuinely different coordinate
systems, both validated independently.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..dataset import DatasetSpec
from .colors import segment_color

# The dataset every constant in this module was fitted on. The simulator warns
# when it is pointed anywhere else: it will render, but no parity claim holds.
CALIBRATED_DATASET = DatasetSpec(
    em_url="precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14",
    seg_url="precomputed://gs://flywire_v141_m783",
    voxel_nm=(4.0, 4.0, 40.0),
)
# The calibrated dataset's voxel size and canonical (finest) voxel, for the
# calibration probes -- which are by definition about this dataset. The live
# render path takes both from the DatasetSpec it was given (em.Source).
VOXEL_NM = np.asarray(CALIBRATED_DATASET.voxel_nm)
CANONICAL_NM = float(min(CALIBRATED_DATASET.voxel_nm))

# nm per projectionScale unit. Fitted on 300 browser-collected calibration
# pairs (2026-08-27; tolerance-IoU(2px) median 0.885) and re-confirmed by
# sweep on 2026-09-10: 3.95 and 4.19 both score worse on mesh AND plane IoU.
# The analytic base would be the 4.0 nm canonical voxel; the 1.75% residual
# is real (perspective camera vs NG's orthographic unit definition) and
# has not been derived, so the fitted value ships.
SCALE_CAL_NM = 4.07
# Chrome/simulator EM intensity ratio on grey 2D-pane pixels (2026-08-28,
# re-confirmed optimal by probe_left_pane_parity 2026-09-10). Chrome's image
# layer opacity 0.5 does NOT halve on-screen EM; ~1.0 is right.
EM_GAIN = 0.978


def em_gain() -> float:
    """EM_GAIN, or the NGL_NATIVE_EM_GAIN override (calibration sweeps).

    Read per call so a sweep can change it between renders in one process;
    the constant is what ships.
    """
    import os

    v = os.environ.get("NGL_NATIVE_EM_GAIN")
    return float(v) if v else EM_GAIN
# 2D-pane fetch-centre correction in captured px (dy, dx). Registration is
# pixel-exact with it (jitter sd 0.0) and the 2026-09-10 shift search found
# no better offset. Absorbs the ~1.6% vertical over-extent of CSS_VIEW_H.
LEFT_SHIFT_PX = (-3.0, 0.0)

# CAPTURE geometry at capture_scale 0.5. TOOLBAR/CSS_VIEW_H are empirical: the
# DOM-derived alternatives (853-px panels at y=47) were tried on 2026-09-10 and
# lose to these by a wide margin on the 2D pane (0.55-0.25 vs 0.861), so 867 is
# right for the CAPTURE even though 853 is right for CLICKS (below).
PANE = 450
TOOLBAR = 17
PANE_H = PANE - TOOLBAR
# The 3D pane's capture geometry is NOT the 2D pane's. Fitted 2026-09-18 by
# sweeping the composite offset against the fork build over a 16x zoom range
# (probe_3d_calibrate.py, job 922247): k=+3 wins at every zoom and by a wide
# margin in the mean (mesh+plane IoU 0.650 at k=0 -> 0.855 at k=+3, falling
# again at k=+4). Zoom-invariance says this is the pane origin, not the
# projection -- our optical centre sat at TOOLBAR + PANE_H/2 = 233.5 where
# Chrome's is ~236.5.
# The RENDER stays PANE_H tall so the camera's aspect and scale (fitted as
# SCALE_CAL_NM) are untouched; only the composite moves down, cropping the
# bottom PANE_3D_SHIFT rows. Rendering 430 tall instead put the centre at 235
# and rescaled the content -- the re-sweep then asked for +2 more rows.
TOOLBAR_3D = 20
PANE_3D_SHIFT = TOOLBAR_3D - TOOLBAR      # 3 capture px
PANE_H_3D = PANE - TOOLBAR_3D             # displayed rows of the 3D pane
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


# Multiplier on the 3D section plane's extent only (not the 2D pane's). 1.0
# ships; exposed so the quad's size can be swept against Chrome's silhouettes
# independently of the camera, which the calibration sweep showed are separate
# failures -- mesh IoU moved 0.442-0.588 with camera scale while plane IoU
# barely moved at all.
PLANE_EXT_SCALE = 1.0


def pane_extents_nm(xs_scale: float, canonical_nm: float = CANONICAL_NM) -> tuple[float, float]:
    """World extent (x, y) of the 2D pane: crossSectionScale canonical voxels
    per CSS px, over the CSS pane. `canonical_nm` is the dataset's finest
    display dimension (4.0 nm for FlyWire)."""
    return (float(xs_scale) * CSS_PANE * canonical_nm,
            float(xs_scale) * CSS_VIEW_H * canonical_nm)


def shifted_fetch_center_nm(pos_nm: np.ndarray, ext: tuple[float, float]):
    """Apply the baked registration correction to the 2D fetch center."""
    return pos_nm + np.array([
        LEFT_SHIFT_PX[1] * ext[0] / PANE,
        LEFT_SHIFT_PX[0] * ext[1] / PANE_H, 0.0])


def resample_em(tile) -> np.ndarray:
    """EM tile -> the pane's greyscale raster (PANE_H x PANE uint8).

    Calibrated chain: subpixel-phase tile -> GL-linear resample to the 900x867
    CSS pane -> Chrome's area-average capture downscale, then EM_GAIN.
    """
    big = Image.fromarray(tile).resize((900, 867), Image.BILINEAR)
    img = np.asarray(big.resize((PANE, PANE_H), Image.BOX)).astype(np.float32)
    return np.clip(img * em_gain(), 0, 255).astype(np.uint8)


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
    where every segment paints. Draw order: EM, then tint, then crosshair.
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


# Regions of the CAPTURE where Chrome draws UI and the simulator cannot.
# Measured with probe_gap_map (per-block SSIM, 12 states): the toolbar strip
# scores 0.010 against Chrome while the 2D interior scores 0.967 and the 3D
# interior 1.000 -- so the frame's biggest disagreements are not the data at
# all, they are chrome. For a policy reading both panes those are the most
# dangerous kind of difference: constant, structured, in a fixed place, and
# perfectly reliable as a "which environment am I in" cue right up until
# deployment inverts it.
#
# Coordinates are captured px in the 900x450 two-pane frame (capture_scale 0.5).
UI_REGIONS = (
    # (y0, y1, x0, x1)
    (0, 32, 0, 900),        # top strip: layer tabs, coordinate readout, icons
    (0, 450, 0, 16),        # 2D pane left edge (disagrees at every row)
    (416, 450, 0, 80),      # 2D pane scale bar ("750 nm")
    (16, 48, 868, 900),     # 3D pane top-right buttons
    (416, 450, 820, 900),   # 3D pane "Sections" control
    # The 3D pane has a left edge too, and only the 2D pane's was masked: this
    # strip is where the 62 pixels that still differed between two CHROME
    # builds lived (2026-09-17), and its grey axis labels contaminated the
    # section-plane measurement in probe_3d_calibrate.py until it was masked.
    (0, 450, 450, 466),     # 3D pane left edge (axis labels)
)


def mask_ui_enabled() -> bool:
    """NGL_MASK_UI=0 turns the mask off for both backends at once."""
    import os

    return os.environ.get("NGL_MASK_UI", "1") != "0"


def mask_ui(image: np.ndarray) -> np.ndarray:
    """Blank the regions where Chrome draws UI, in place, on a copy.

    Applied to BOTH backends so neither carries a cue the other lacks. Masking
    only Chrome would leave the simulator showing data where Chrome shows a
    scale bar, which is the same problem mirrored.

    This costs real pixels -- the top strip and the left edge are ~7% of a pane
    -- but they are pixels the two renderers can never agree on, and a network
    will find them long before it finds the neuron.
    """
    out = np.array(image, copy=True)
    for y0, y1, x0, x1 in UI_REGIONS:
        if y0 < out.shape[0] and x0 < out.shape[1]:
            out[y0:min(y1, out.shape[0]), x0:min(x1, out.shape[1])] = 0
    return out
