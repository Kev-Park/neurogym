"""ngllib.simulator — browser-free Neuroglancer-equivalent rendering.

Replaces the Playwright+Chrome render path with direct data access
(CloudVolume: precomputed EM cutouts + sharded Draco meshes) and offscreen
GPU rendering (moderngl + EGL). Visual parity targets the exact Neuroglancer
semantics ported in `colors` (segment colour hash) and `camera` (projection-
pane camera model), validated pixel-level against Chrome captures of
identical states (neurogym-agent `native/` probes).

`SimulatorRenderer` is the `ngllib.Environment` backend; the heavy imports
(moderngl, cloud-volume) happen when it is opened, not here.
"""

from .camera import projection_camera
from .colors import segment_color
from .renderer import PANE_MODES, SimulatorRenderer

__all__ = ["SimulatorRenderer", "PANE_MODES", "segment_color", "projection_camera"]
