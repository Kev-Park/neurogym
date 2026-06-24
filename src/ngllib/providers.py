"""StateProvider protocol + NglState schema — the dataset-agnostic reset contract.

`task_info` is opaque to the library — its shape is a contract between the
provider and the consumer's reward/termination factories.
"""

from __future__ import annotations

from typing import Any, Protocol, TypedDict, runtime_checkable

import numpy as np


class NglState(TypedDict, total=False):
    """Structured Neuroglancer viewer state. All keys optional; `extra` carries
    any non-first-class fields (layers, annotations, …) merged verbatim."""

    position: tuple[float, float, float] | list[float]
    crossSectionScale: float
    projectionOrientation: tuple[float, ...] | list[float]  # length 3 (euler) or 4 (quaternion)
    projectionScale: float
    segments: list[str]
    extra: dict[str, Any]


@runtime_checkable
class StateProvider(Protocol):
    """Per-episode start-state sampling + the state->task_info mapping used when
    the caller overrides state but not task_info."""

    def __call__(
        self, rng: np.random.Generator, options: dict[str, Any] | None
    ) -> tuple[NglState | str, dict[str, Any]]:
        """Sample a (start_state, task_info) pair. Use `rng`, not the global
        `random`, so reset(seed=...) is reproducible."""
        ...

    def task_info_from_state(self, state: NglState | str) -> dict[str, Any]:
        """Derive task_info for an externally-supplied state. Raise
        NotImplementedError if your task can't derive it from state alone."""
        ...
