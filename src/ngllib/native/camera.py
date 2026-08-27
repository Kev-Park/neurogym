"""Port of Neuroglancer's 3D-projection-pane camera model.

Source (google/neuroglancer @ master src/perspective_view/panel.ts,
retrieved 2026-08-27), perspective mode (NG default):

    fovy            = pi / 4
    f               = 1 / tan(fovy / 2)
    zoomFactor      = (projectionScale / 2) * f        # world units
    nearBound       = max(0.1, 1 - relativeDepthRange)
    farBound        = 1 + relativeDepthRange
    projectionMat   = perspective(fovy, width/height, nearBound, farBound)
    invViewMatrix   = pose.toMat4(zoomFactor)          # R(q), t=position,
                                                       # scaled by zoomFactor
    invViewMatrix  *= scale(1, -1, -1)                 # flip Y and Z
    invViewMatrix  *= translate(+Z by 1)               # camera sits one
                                                       # (scaled) unit back

i.e. an orbit camera at world distance zoomFactor from `position`, looking
along the pose's -Z after the flip, with the near/far planes defined in the
zoom-scaled space (depth range brackets the orbit target at distance 1).

Units: `position` is in voxel coordinates with NG "dimensions" scaling
(FlyWire: x,y = 4nm, z = 40nm voxels). NG's pose operates in canonical
(scaled) space — the Z anisotropy (10x) MUST be applied before this camera:
render in nm-like space (x*4, y*4, z*40, or voxels with z*10) and pass
`position` in that same space with projection_scale in x/y voxel units * 4?
CALIBRATE: the collection-pair comparison decides the exact convention;
start with canonical space = voxels scaled by (1, 1, 10) and
projection_scale in xy-voxel units.

relative_depth_range: NG default = 10 (viewer.ts: new
TrackableDepthRange(-10, ...); negative = relative). After the /f division
this gives near 0.1, far ~5.1 in orbit-distance units — near-unclipped.
"""

from __future__ import annotations

import math

import numpy as np


def _quat_to_mat3(q) -> np.ndarray:
    """Unit quaternion [x, y, z, w] (NG order) -> 3x3 rotation matrix."""
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _perspective(fovy: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(fovy / 2.0)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def projection_camera(
    position,
    orientation_quat,
    projection_scale: float,
    width: int,
    height: int,
    relative_depth_range: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """(view, projection) matrices for NG's perspective pane.

    position: orbit target in canonical space (see module docstring).
    orientation_quat: NG projectionOrientation [x, y, z, w].
    Returns float64 4x4 matrices; multiply projection @ view for clip space.
    """
    fovy = math.pi / 4.0
    f = 1.0 / math.tan(fovy / 2.0)
    zoom = (float(projection_scale) / 2.0) * f
    rdr = relative_depth_range / f
    near = max(0.1, 1.0 - rdr)
    far = 1.0 + rdr

    # invView = T(position) @ R(q) @ S(zoom) @ S(1,-1,-1) @ T(0,0,1)
    inv_view = np.eye(4, dtype=np.float64)
    inv_view[:3, :3] = _quat_to_mat3(orientation_quat) * zoom
    inv_view[:3, 3] = np.asarray(position, dtype=np.float64)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    step_back = np.eye(4)
    step_back[2, 3] = 1.0
    inv_view = inv_view @ flip @ step_back

    view = np.linalg.inv(inv_view)
    proj = _perspective(fovy, width / height, near, far)
    return view, proj
