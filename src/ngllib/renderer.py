"""The seam between `Environment` and a rendering backend.

An `Environment` owns everything that is not pixels: the spaces, the
provider/reward/termination hooks, episode bookkeeping, the viewer-state dict
and its pure transitions, and the reset/step skeleton. A `Renderer` owns
everything that IS pixels -- and for Chrome, also the copy of the state that
lives inside Neuroglancer. Two backends can therefore differ in what they draw
and in nothing else.

State flows one way per call: the environment hands a renderer a state
(`reset_to`, `set_state`) or an input event (`click`), and reads the
authoritative state back with `observe`. For the simulator the state it reads
back is the one it wrote; for Chrome it is whatever Neuroglancer did with it,
which is what makes Chrome normative.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .events import EventLog

# The geometry every calibrated constant was measured at: 1800x900 CSS window
# captured at 0.5 -> two 450x450 panes with 17 captured px of toolbar.
CALIBRATED_WINDOW = (1800, 900)
CALIBRATED_CAPTURE_SCALE = 0.5


@dataclass(frozen=True)
class PaneLayout:
    """Capture geometry shared by every renderer, so panes cannot differ by
    backend.

    `window_size` is the CSS viewport (also the action space's `mouse_xy`
    bounds); `capture_scale` shrinks the capture (the compositor does it on
    the GPU in Chrome); `image_size` (W, H) resizes the final frame after any
    UI mask. With one pane on, the frame is the left or right half.
    """

    window_size: tuple[int, int] = (1800, 900)
    capture_scale: float = 1.0
    image_size: tuple[int, int] | None = None
    left_pane: bool = False
    right_pane: bool = True

    def __post_init__(self) -> None:
        if not (self.left_pane or self.right_pane):
            raise ValueError("At least one of `left_pane` or `right_pane` must be True.")
        if not (0.0 < self.capture_scale <= 1.0):
            raise ValueError(
                f"`capture_scale` must be in (0, 1]; got {self.capture_scale!r}")
        object.__setattr__(self, "window_size", tuple(int(v) for v in self.window_size))
        if self.image_size is not None:
            object.__setattr__(self, "image_size", tuple(int(v) for v in self.image_size))

    @property
    def capture_size(self) -> tuple[int, int]:
        """(W, H) of the full two-pane capture."""
        W, H = self.window_size
        return round(W * self.capture_scale), round(H * self.capture_scale)

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """(H, W, 3) of the captured frame after the pane crop, before any
        `image_size` resize."""
        W, H = self.capture_size
        if self.left_pane and self.right_pane:
            return (H, W, 3)
        return (H, W // 2, 3)

    @property
    def image_shape(self) -> tuple[int, int, int]:
        """(H, W, 3) of the observation image."""
        if self.image_size is not None:
            iw, ih = self.image_size
            return (ih, iw, 3)
        return self.frame_shape

    @property
    def is_calibrated(self) -> bool:
        return (self.window_size == CALIBRATED_WINDOW
                and self.capture_scale == CALIBRATED_CAPTURE_SCALE)

    def warn_if_uncalibrated(self, who: str) -> None:
        """Warn, don't reject: nothing stops another geometry from rendering,
        but every calibrated constant (pane2d) and the UI mask were measured
        at the calibrated one, so parity claims do not carry."""
        if not self.is_calibrated:
            warnings.warn(
                f"{who}: window_size={self.window_size} @ capture_scale="
                f"{self.capture_scale} is not the calibrated geometry "
                f"{CALIBRATED_WINDOW} @ {CALIBRATED_CAPTURE_SCALE}; the parity "
                "constants and the UI mask were measured there and do not carry.",
                stacklevel=3)


@runtime_checkable
class Renderer(Protocol):
    """What a backend must provide. See the module docstring for the flow.

    `layout` fixes the observation image shape and the click bounds.
    `warm_after_steps` is the backend's preferred delay before the environment
    warms the next episode (Chrome pays for a second live browser context, the
    simulator does not); the environment applies it unless told otherwise.
    `events` is set by the environment before `open()` so a backend can log
    its own diagnostics into the same stream.
    """

    layout: PaneLayout
    warm_after_steps: int
    events: EventLog

    def open(self) -> None:
        """Bring the backend up. Called once, lazily, on the first reset."""

    def close(self) -> None:
        """Tear the backend down. Idempotent."""

    def default_state(self) -> dict[str, Any]:
        """The NglState the backend starts in when nobody supplies one."""

    def warm(self, state: dict[str, Any]) -> None:
        """Best-effort hint that `state` is probably the next reset. Never
        raises; correctness never depends on it."""

    def reset_to(self, state: dict[str, Any] | str) -> None:
        """Make `state` current at an episode boundary. Reuses the work of a
        matching `warm()` silently; otherwise pays the cold path. A URL string
        is accepted where the backend can navigate to one."""

    def set_state(self, state: dict[str, Any]) -> None:
        """Make `state` current mid-episode (an edit_state action)."""

    def click(self, kind: str, x: float, y: float, modifiers: str) -> None:
        """Deliver Neuroglancer's `kind` in {left_click, right_click,
        double_click} at CSS pixel (x, y) with modifier keys (\"Shift, Ctrl\"
        form), and apply what Neuroglancer does in response."""

    def observe(self) -> tuple[dict[str, Any], np.ndarray]:
        """(authoritative NglState, captured frame of shape layout.frame_shape)."""
