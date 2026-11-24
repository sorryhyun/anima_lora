"""Manga109 polygons as crop items for the main / sub proposals: an :class:`Item` is one
``<text>`` / COO polygon with its convex hull and glyph size; ``frac`` is the share of it a
crop quad (:func:`crops.rect_quad`) holds."""

from __future__ import annotations

import cv2
import numpy as np

INCLUDE = 0.8
"""Share of an item inside a crop for it to count as inside."""
SOFT = set("!！?？・…‥~〜ー-―.．、。")


def glyphs(t: str) -> float:
    return max(1.0, sum(0.5 if c in SOFT else 1.0 for c in t if not c.isspace()))


class Item:
    __slots__ = ("kind", "id", "text", "hull", "area", "cx", "cy", "g")

    def __init__(self, line, text: str):
        pts = np.asarray(line.poly, dtype=np.float32).reshape(-1, 2)
        self.hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.float32)
        self.area = max(float(cv2.contourArea(self.hull)), 1.0)
        self.cx, self.cy = (float(v) for v in pts.mean(0))
        self.kind, self.id, self.text = line.kind, str(line.id), text
        self.g = float(np.sqrt(self.area / glyphs(text)))

    def frac(self, quad: np.ndarray) -> float:
        a, _ = cv2.intersectConvexConvex(self.hull, quad)
        return max(0.0, float(a)) / self.area


def order(items):
    return sorted(items, key=lambda it: (-it.cx, it.cy))
