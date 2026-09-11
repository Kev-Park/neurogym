"""Pure viewer-state math shared by every backend.

Chrome is normative for state and action semantics: what lives here is what
Neuroglancer does, written once so the simulator cannot drift from it and so
the environment can drive Chrome with states Neuroglancer will accept. Every
function returns a new state and leaves its input untouched.

An NglState is the five-field dict `providers.NglState` describes: position
(voxels), crossSectionScale, projectionOrientation (quaternion x,y,z,w),
projectionScale, segments (strings; a leading "!" marks selected-but-hidden).
"""

from __future__ import annotations

import copy
import math
from typing import Any, Iterable, Literal

import numpy as np

from .utils.geom import euler_to_quaternion, quaternion_to_euler

Orientation = Literal["quaternion", "euler"]

REQUIRED_FIELDS = ("position", "crossSectionScale", "projectionScale")
IDENTITY_QUAT = [0.0, 0.0, 0.0, 1.0]

# Upper bound the harness has always applied to projectionScale.
PROJECTION_SCALE_MAX = 500_000.0


def coerce(state: dict[str, Any]) -> dict[str, Any]:
    """A deep copy with the optional fields defaulted and the required ones
    checked, so every backend starts from the same shape."""
    st = copy.deepcopy(state)
    st.setdefault("projectionOrientation", list(IDENTITY_QUAT))
    st.setdefault("segments", [])
    missing = [k for k in REQUIRED_FIELDS if k not in st]
    if missing:
        raise ValueError(f"state missing required field(s) {missing}")
    st["position"] = [float(v) for v in st["position"]]
    st["segments"] = [str(s) for s in st["segments"]]
    return st


def visible_segments(segments: Iterable[Any]) -> tuple[str, ...]:
    """The VISIBLE subset, as a stable hashable key.

    NG keeps two sets (layer/segmentation/index.ts): `selectedSegments`,
    everything in the list, and `visibleSegments`, the subset that is drawn.
    `select` toggles visibility, and a selected-but-hidden segment serializes
    as "!<id>" rather than being dropped.
    """
    return tuple(str(s) for s in segments if not str(s).startswith("!"))


def normalized_quaternion(q: Iterable[float]) -> list[float]:
    """NG normalizes on both restoreState() and toJSON() (navigation_state.ts),
    so a Chrome readback is always unit length. Applied in the shared math so
    the simulator's state is too -- in `orientation="quaternion"` mode the raw
    component adds below would otherwise drift off the sphere."""
    v = [float(x) for x in q]
    n = math.sqrt(sum(x * x for x in v))
    if n == 0.0:
        return list(IDENTITY_QUAT)
    return [x / n for x in v]


def next_projection_scale(previous: float, requested: float) -> float:
    """NG's `TrackableZoom.restoreState` runs `verifyFinitePositiveFloat`, so a
    zoom of 0 or below is REJECTED outright and the viewer keeps the value it
    had (7.1). The clamp the simulator used to carry (`max(1.0, ...)`) was a
    different rule, and Chrome carried none at all: navigating to a URL with a
    non-positive zoom just failed to parse. Both backends now get this.

    Gate 4 still owes one measurement: whether Neuroglancer discards the
    OTHER components of a rejected edit as well. If it does, the rule moves
    from this function to `apply_state_edit` -- nowhere else.
    """
    if not (requested > 0.0) or not math.isfinite(requested):
        return float(previous)
    return float(min(PROJECTION_SCALE_MAX, requested))


def apply_state_edit(state: dict[str, Any], action: dict[str, Any],
                     orientation: Orientation) -> dict[str, Any]:
    """action_type 3: add the deltas to position, crossSectionScale,
    orientation and projectionScale. Identical arithmetic on both backends
    (rotate and zoom were bit-exact against Chrome before the seam; the
    normalization and the zoom rule are the two deliberate changes)."""
    st = copy.deepcopy(state)
    dpos = action["delta_pos"]
    st["position"] = [st["position"][i] + float(dpos[i]) for i in range(3)]
    st["crossSectionScale"] = float(st["crossSectionScale"]) + float(action["delta_xs_scale"][0])

    d = action["delta_orient"]
    if orientation == "euler":
        old = quaternion_to_euler(st["projectionOrientation"])
        q = euler_to_quaternion([old[i] + float(d[i]) for i in range(3)])
    else:
        q = [float(st["projectionOrientation"][i]) + float(d[i]) for i in range(4)]
    st["projectionOrientation"] = normalized_quaternion(q)

    st["projectionScale"] = next_projection_scale(
        float(st["projectionScale"]),
        float(st["projectionScale"]) + float(action["delta_proj_scale"][0]))
    return st


def move_to(state: dict[str, Any], position: Iterable[float]) -> dict[str, Any]:
    """NG `move-to-mouse-position` (right-click on a rendered data panel)."""
    st = copy.deepcopy(state)
    st["position"] = [float(v) for v in position]
    return st


def toggle_select(state: dict[str, Any], root_id: Any) -> dict[str, Any]:
    """NG `select` (double-click): toggle VISIBILITY of the segment under the
    cursor. A visible segment becomes hidden ("!"-prefixed, still selected), a
    hidden one becomes visible, an unknown one is appended visible."""
    st = copy.deepcopy(state)
    segs = [str(s) for s in st["segments"]]
    rid = str(root_id)
    if rid in segs:
        segs[segs.index(rid)] = "!" + rid
    elif "!" + rid in segs:
        segs[segs.index("!" + rid)] = rid
    else:
        segs.append(rid)
    st["segments"] = segs
    return st


def orientation_obs(state: dict[str, Any], orientation: Orientation) -> np.ndarray:
    q = state.get("projectionOrientation", IDENTITY_QUAT)
    if orientation == "euler":
        return np.asarray(quaternion_to_euler(q), dtype=np.float32)
    return np.asarray(q, dtype=np.float32)


def modifiers_to_str(modifiers: Iterable[Any]) -> str:
    """[shift, ctrl, alt] -> the "Shift, Ctrl" form the click handlers take."""
    names = ("Shift", "Ctrl", "Alt")
    return ", ".join(n for n, m in zip(names, modifiers) if int(m))


CLICK_KINDS = {0: "left_click", 1: "right_click", 2: "double_click"}
