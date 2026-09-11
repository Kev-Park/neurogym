"""A fake renderer that satisfies the `Renderer` protocol with no GL and no
browser, so the environment's own logic is testable anywhere."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from ngllib import state as S
from ngllib.events import EventLog
from ngllib.renderer import PaneLayout


class FakeRenderer:
    warm_after_steps = 0

    def __init__(self, **layout):
        self.layout = PaneLayout(**layout)
        self.events = EventLog(path_template="")
        self.calls: list[tuple] = []
        self.opened = False
        self.closed = False
        self.state: dict[str, Any] | None = None
        self.warmed: list[dict] = []
        self.fail_next_observe = False

    def open(self):
        self.opened = True

    def close(self):
        self.closed = True

    def default_state(self):
        return {"position": [0.0, 0.0, 0.0], "crossSectionScale": 1.0,
                "projectionOrientation": list(S.IDENTITY_QUAT), "projectionScale": 1000.0,
                "segments": ["42"]}

    def warm(self, state):
        self.warmed.append(copy.deepcopy(state))

    def reset_to(self, state):
        self.calls.append(("reset_to", copy.deepcopy(state)))
        self.state = S.coerce(state) if isinstance(state, dict) else self.default_state()

    def set_state(self, state):
        self.calls.append(("set_state", copy.deepcopy(state)))
        self.state = S.coerce(state)

    def click(self, kind, x, y, modifiers):
        self.calls.append(("click", kind, x, y, modifiers))
        if kind == "right_click":
            self.state = S.move_to(self.state, [x, y, self.state["position"][2]])
        elif kind == "double_click":
            self.state = S.toggle_select(self.state, "7")

    def observe(self):
        if self.fail_next_observe:
            from ngllib.errors import RendererError

            self.fail_next_observe = False
            raise RendererError("fake failure")
        h, w, _ = self.layout.frame_shape
        frame = np.full((h, w, 3), 200, dtype=np.uint8)
        return copy.deepcopy(self.state), frame


class CountingProvider:
    """Deterministic provider whose draws are countable."""

    def __init__(self):
        self.draws = 0

    def __call__(self, rng, options):
        self.draws += 1
        return ({"position": [float(self.draws), 0.0, 0.0], "crossSectionScale": 1.0,
                 "projectionScale": 5000.0, "segments": [str(self.draws)]},
                {"segment_id": str(self.draws)})

    def task_info_from_state(self, state):
        return {"segment_id": state["segments"][0]}


@pytest.fixture
def fake_renderer():
    return FakeRenderer


@pytest.fixture
def provider():
    return CountingProvider()
