"""Ink per glyph — the size × stroke-density variable of ``plan_band.md``.

``ink_pixels`` counts the pixels inside a box that differ from the box's own
background. The background is the median of a ring just outside the box, not
of the box itself: a tight ink bbox of a dense kanji can be more than half
ink, and a scene bubble, a grid cell and a dark jitter canvas each have their
own background level, so a fixed "dark = ink" rule reads every one differently.
"""

from __future__ import annotations

import numpy as np


def glyph_count(text: str) -> int:
    """Non-space characters (the ``_remap_band`` rule)."""
    return sum(not c.isspace() for c in text)


def ink_pixels(im, box, thresh: int = 60, ring: int = 3) -> int:
    """Pixels of ``im`` inside ``box`` (x0, y0, x1, y1, canvas px) whose
    luminance is more than ``thresh`` from the background, the median of the
    ``ring``-px band around the box (clipped at the canvas)."""
    g = np.asarray(im.convert("L"), dtype=np.int16)
    H, W = g.shape
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0 = max(0, min(x0, W)), max(0, min(y0, H))
    x1, y1 = max(x0, min(x1, W)), max(y0, min(y1, H))
    if x1 <= x0 or y1 <= y0:
        return 0
    X0, Y0 = max(0, x0 - ring), max(0, y0 - ring)
    X1, Y1 = min(W, x1 + ring), min(H, y1 + ring)
    outer = g[Y0:Y1, X0:X1]
    mask = np.ones(outer.shape, dtype=bool)
    mask[y0 - Y0 : y1 - Y0, x0 - X0 : x1 - X0] = False
    inner = g[y0:y1, x0:x1]
    bg = int(np.median(outer[mask])) if mask.any() else int(np.median(inner))
    return int((np.abs(inner - bg) > thresh).sum())


def box_area(box) -> float:
    x0, y0, x1, y1 = box
    return max(0.0, float(x1 - x0)) * max(0.0, float(y1 - y0))
