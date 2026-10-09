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


def glyph_features(text: str, px: int, font_path: str, nbins: int = 18) -> dict:
    """Shape descriptors of ``text`` drawn alone at ``px`` in ``font_path``
    (plan_kanji: complexity that is not ink). ``fill`` = ink share of the
    bbox; ``straight`` = 1 − normalised entropy of the gradient-orientation
    histogram (mod 180°, magnitude-weighted) — straight strokes at any angle
    concentrate it, curves spread it; ``axis`` = share of gradient energy
    within ± 10° of horizontal or vertical (frame-like strokes)."""
    from PIL import Image, ImageDraw, ImageFont

    im = Image.new("L", (px * 3, max(px * 3, px * (len(text) + 2))), 255)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(font_path, px, index=0)
    d.text((px, px), text, font=f, fill=0)
    x0, y0, x1, y1 = d.textbbox((px, px), text, font=f)
    g = 1.0 - np.asarray(im, dtype=np.float32)[y0:y1, x0:x1] / 255.0
    if g.size == 0 or g.max() <= 0:
        return {"fill": 0.0, "straight": 0.0, "axis": 0.0}
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    ang = np.mod(np.degrees(np.arctan2(gy, gx)), 180.0)
    hist, _ = np.histogram(ang, bins=nbins, range=(0.0, 180.0), weights=mag)
    p = hist / max(hist.sum(), 1e-9)
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(nbins))
    near_axis = (ang < 10) | (ang > 170) | (np.abs(ang - 90) < 10)
    axis = float(mag[near_axis].sum() / max(mag.sum(), 1e-9))
    return {
        "fill": float((g > 0.5).mean()),
        "straight": 1.0 - ent,
        "axis": axis,
    }
