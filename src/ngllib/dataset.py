"""Deployment identity: the start URL, and the dataset it declares.

`config.json` holds the start URL and credentials only. Everything the two
renderers need to agree on about the DATA -- which volumes, what a voxel
measures -- is parsed out of that URL once, into `DatasetSpec`, and passed to
both as an ordinary argument. The simulator used to hardcode the same three
facts; repointing the URL then silently compared two datasets.
"""

from __future__ import annotations

import copy
import json
import urllib.parse
from dataclasses import dataclass
from typing import Any

from .errors import ProviderError

DEFAULT_NG_HOST = "https://neuroglancer-demo.appspot.com/"

# Neuroglancer "dimensions" units -> nanometres.
_TO_NM = {"m": 1e9, "mm": 1e6, "um": 1e3, "µm": 1e3, "nm": 1.0, "pm": 1e-3}


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """The packaged `config.json`, or the one at `config_path`."""
    if config_path is None:
        from importlib.resources import files

        return json.loads(files("ngllib").joinpath("config.json").read_text())
    with open(config_path) as f:
        return json.load(f)


def default_start_url(config_path: str | None = None) -> str:
    url = load_config(config_path).get("default_ngl_start_url")
    if not url:
        raise ProviderError("config has no 'default_ngl_start_url'")
    return url


def split_state_url(url: str) -> tuple[str, dict[str, Any]]:
    """`(host prefix, viewer state)` from a `...#!<url-encoded json>` URL.

    A URL with no state fragment yields the demo host and an empty state.
    """
    if "#!" in url:
        prefix, encoded = url.split("#!", 1)
        try:
            return prefix, json.loads(urllib.parse.unquote(encoded))
        except (ValueError, TypeError):
            return prefix, {}
    return DEFAULT_NG_HOST, {}


def state_to_url(prefix: str, state: dict[str, Any]) -> str:
    return prefix + "#!" + urllib.parse.quote(json.dumps(state))


def _rewrite_source(url: str) -> str:
    """`graphene://https://…` -> `graphene://middleauth+https://…`.

    The FlyWire deployment authenticates plain graphene URLs itself; the
    upstream build and the lab fork do not -- a plain URL with a valid token
    measured as 401 from prodv1 (2026-09-17). So a pasted FlyWire link has to
    be rewritten before either of our viewers can load it.
    """
    if url.startswith("graphene://") and "middleauth+" not in url:
        return "graphene://middleauth+" + url[len("graphene://"):]
    return url


def rewrite_graphene_sources(state: dict[str, Any]) -> dict[str, Any]:
    """Every graphene source in `state` addressed through middleauth."""
    out = copy.deepcopy(state)
    for layer in out.get("layers", []) or []:
        src = layer.get("source")
        if isinstance(src, str):
            layer["source"] = _rewrite_source(src)
        elif isinstance(src, list):
            layer["source"] = [
                _rewrite_source(u) if isinstance(u, str)
                else ({**u, "url": _rewrite_source(u["url"])} if isinstance(u, dict) and "url" in u else u)
                for u in src
            ]
        elif isinstance(src, dict) and "url" in src:
            layer["source"] = {**src, "url": _rewrite_source(src["url"])}
    return out


def is_legacy_state(state: dict[str, Any]) -> bool:
    """A pre-2020 viewer state (`navigation.pose`, `zoomFactor`)."""
    return "navigation" in state and "position" not in state


def legacy_to_modern_state(state: dict[str, Any]) -> dict[str, Any]:
    """Convert a legacy viewer state to the format `ngllib.state` drives.

    ngl.flywire.ai and neuromancer-seung-import serve the legacy format, which
    has no `position`/`crossSectionScale`/`projectionScale` for the environment
    to act on. The mapping below is the one used by hand to reproduce a pasted
    FlyWire link in the modern viewer (2026-09-17):

        navigation.pose.position.voxelCoordinates -> position
        navigation.pose.position.voxelSize        -> dimensions (nm -> m)
        navigation.zoomFactor / min(voxelSize)    -> crossSectionScale
        perspectiveZoom                           -> projectionScale
        perspectiveOrientation                    -> projectionOrientation
        type "segmentation_with_graph"            -> "segmentation"
    """
    nav = state.get("navigation") or {}
    pose_pos = ((nav.get("pose") or {}).get("position") or {})
    voxel_size = pose_pos.get("voxelSize") or nav.get("voxelSize")
    coords = pose_pos.get("voxelCoordinates")
    if not voxel_size or not coords:
        raise ProviderError(
            "legacy start URL without navigation.pose.position.voxelCoordinates/voxelSize; "
            "cannot convert -- paste a link from a modern Neuroglancer build instead")

    out: dict[str, Any] = {
        "dimensions": {ax: [float(v) * 1e-9, "m"] for ax, v in zip("xyz", voxel_size)},
        "position": [float(c) for c in coords],
    }
    zoom = nav.get("zoomFactor")
    if zoom is not None:
        # legacy zoomFactor is nm per screen px; modern crossSectionScale is
        # canonical voxels per px.
        out["crossSectionScale"] = float(zoom) / float(min(voxel_size))
    for legacy_key, modern_key in (("perspectiveZoom", "projectionScale"),
                                   ("perspectiveOrientation", "projectionOrientation")):
        if legacy_key in state:
            out[modern_key] = copy.deepcopy(state[legacy_key])
    layers = []
    for layer in state.get("layers", []) or []:
        lay = copy.deepcopy(layer)
        if lay.get("type") == "segmentation_with_graph":
            lay["type"] = "segmentation"
        layers.append(lay)
    out["layers"] = layers
    for key in ("layout", "showDefaultAnnotations", "selectedLayer", "crossSectionOrientation"):
        if key in state:
            out[key] = copy.deepcopy(state[key])
    return out


def normalize_start_url(url: str) -> tuple[str, dict[str, Any]]:
    """`(origin prefix, viewer state)` ready for our viewers.

    Accepts a link pasted from any Neuroglancer deployment: a legacy state is
    converted, graphene sources are addressed through middleauth. The returned
    prefix is the link's own origin -- the caller decides whether to serve the
    state from a packaged viewer instead.
    """
    prefix, state = split_state_url(url)
    if state and is_legacy_state(state):
        state = legacy_to_modern_state(state)
    return prefix, rewrite_graphene_sources(state)


def merge_state(base_state: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Overlay an NglState onto a full viewer state (the start URL's).

    First-class NglState fields land in their viewer-state slots; `extra` is
    merged verbatim first so it can never shadow them. `segments` replaces the
    list on the first segmentation layer.
    """
    merged = copy.deepcopy(base_state)
    if isinstance(state.get("extra"), dict):
        merged.update(state["extra"])
    if "position" in state:
        merged["position"] = [float(v) for v in state["position"]]
    if "crossSectionScale" in state:
        merged["crossSectionScale"] = float(state["crossSectionScale"])
    if "projectionOrientation" in state:
        merged["projectionOrientation"] = [float(v) for v in state["projectionOrientation"]]
    if "projectionScale" in state:
        merged["projectionScale"] = float(state["projectionScale"])
    if "segments" in state:
        for layer in merged.get("layers", []):
            if layer.get("type") == "segmentation":
                layer["segments"] = [str(s) for s in state["segments"]]
                break
    return merged


def ngl_state_from_viewer(viewer_state: dict[str, Any]) -> dict[str, Any]:
    """The NglState view of a full viewer state: the five fields the
    environment tracks, `segments` read off the first segmentation layer."""
    st: dict[str, Any] = {}
    for k in ("position", "crossSectionScale", "projectionOrientation", "projectionScale"):
        if k in viewer_state:
            st[k] = copy.deepcopy(viewer_state[k])
    st.setdefault("projectionOrientation", [0.0, 0.0, 0.0, 1.0])
    for layer in viewer_state.get("layers", []):
        if layer.get("type") == "segmentation":
            st["segments"] = [str(s) for s in layer.get("segments", [])]
            break
    st.setdefault("segments", [])
    return st


@dataclass(frozen=True)
class DatasetSpec:
    """What the start URL says about the data. Hashable, so fetch workers can
    key their per-dataset handles on it."""

    em_url: str
    seg_url: str
    voxel_nm: tuple[float, float, float]

    @classmethod
    def from_state(cls, viewer_state: dict[str, Any]) -> "DatasetSpec":
        dims = viewer_state.get("dimensions", {})
        try:
            voxel = tuple(
                float(dims[ax][0]) * _TO_NM[dims[ax][1]] for ax in ("x", "y", "z"))
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise ProviderError(
                f"start URL declares no usable 'dimensions' (x/y/z as [value, unit]): {e}"
            ) from e
        em = seg = None
        for layer in viewer_state.get("layers", []):
            src = layer.get("source")
            if isinstance(src, dict):
                src = src.get("url")
            if layer.get("type") == "image" and em is None:
                em = src
            elif layer.get("type") == "segmentation" and seg is None:
                seg = src
        if not em or not seg:
            raise ProviderError(
                "start URL must carry one 'image' and one 'segmentation' layer with a source")
        return cls(em_url=str(em), seg_url=str(seg), voxel_nm=voxel)

    @classmethod
    def from_start_url(cls, url: str) -> "DatasetSpec":
        return cls.from_state(split_state_url(url)[1])

    @classmethod
    def from_config(cls, config_path: str | None = None) -> "DatasetSpec":
        return cls.from_start_url(default_start_url(config_path))
