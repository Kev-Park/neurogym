"""Exact port of Neuroglancer's segment coloring.

Sources (google/neuroglancer @ master, retrieved 2026-08-27):
- src/gpu_hash/hash_function.ts  (hashCombine: murmur-style mix)
- src/segment_color.ts           (hash -> HSV mapping)

Color = HSV(h, s, 1.0) where, with H = hashCombine(hashCombine(seed, lo32),
hi32) over the two 32-bit halves of the uint64 segment id:
    hue        = (H & 0xff) / 255
    saturation = 0.5 + 0.5 * ((H >> 8) & 0xff) / 255
The default segmentation layer color seed is 0 (no "colorSeed" in our layer
JSON). Value is fixed at 1.0.
"""

from __future__ import annotations

import colorsys

_M = 0xFFFFFFFF
_K1 = 0xCC9E2D51
_K2 = 0x1B873593


def _imul32(a: int, b: int) -> int:
    """JS Math.imul: 32-bit truncated multiply (unsigned view)."""
    return (a * b) & _M


def _rotl(v: int, n: int) -> int:
    return ((v << n) | (v >> (32 - n))) & _M


def hash_combine(state: int, value: int) -> int:
    """murmur3-style combine, bit-exact with NG's hashCombine."""
    value = _imul32(value & _M, _K1)
    value = _rotl(value, 15)
    value = _imul32(value, _K2)
    state = (state ^ value) & _M
    state = _rotl(state, 13)
    return (state * 5 + 0xE6546B64) & _M


def segment_color(segment_id: int, seed: int = 0) -> tuple[float, float, float]:
    """NG display color for a segment id, as RGB floats in [0, 1]."""
    lo = segment_id & _M
    hi = (segment_id >> 32) & _M
    h = hash_combine(seed & _M, lo)
    h = hash_combine(h, hi)
    hue = (h & 0xFF) / 255.0
    sat = 0.5 + 0.5 * ((h >> 8) & 0xFF) / 255.0
    return colorsys.hsv_to_rgb(hue, sat, 1.0)
