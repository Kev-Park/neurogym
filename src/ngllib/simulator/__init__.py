"""ngllib.native — browser-free Neuroglancer-equivalent rendering (spike).

Replaces the Playwright+Chrome render path with direct data access
(CloudVolume: precomputed EM cutouts + sharded Draco meshes) and offscreen
GPU rendering (moderngl + EGL), eliminating the JPEG encode/decode round
trip. Visual parity targets the exact Neuroglancer semantics ported in
`colors` (segment color hash) and `camera` (projection-pane camera model);
parity is validated pixel-level against browser captures of identical
states (see neurogym-agent native/ collection tooling).

Status: exploration branch (native-renderer); not part of the ngllib API.
"""

from .colors import segment_color
from .camera import projection_camera

__all__ = ["segment_color", "projection_camera"]
