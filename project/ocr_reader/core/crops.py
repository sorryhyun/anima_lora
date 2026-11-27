"""Crop geometry: the deskewed polygon cut, the 4 px detector-box margin, and the ink
projection the line-break and furigana tools split a crop by."""

from __future__ import annotations

import cv2
import numpy as np

SPACES = set(" 　")


def _on_page_window(rot, cx: float, cy: float, cw: int, ch: int, iw: int, ih: int):
    """``(top, bottom, left, right)`` of the largest window of a ``cw`` × ``ch``
    ``getRectSubPix`` crop of the ``rot``-warped page whose every pixel samples
    the page itself — not ``warpAffine``'s replicated border, not
    ``getRectSubPix``'s. The valid set is convex (a rectangle ∩ an affine image
    of one), so a window whose four edges are valid is valid inside, and
    peeling the worst edge one line at a time ends at such a window."""
    import cv2
    import numpy as np

    u = np.arange(cw, dtype=np.float64) + cx - (cw - 1) * 0.5
    v = np.arange(ch, dtype=np.float64) + cy - (ch - 1) * 0.5
    xr, yr = np.meshgrid(u, v)
    inv = cv2.invertAffineTransform(rot)
    xo = inv[0, 0] * xr + inv[0, 1] * yr + inv[0, 2]
    yo = inv[1, 0] * xr + inv[1, 1] * yr + inv[1, 2]
    ok = (
        (xr >= 0)
        & (xr <= iw - 1)
        & (yr >= 0)
        & (yr <= ih - 1)
        & (xo >= 0)
        & (xo <= iw - 1)
        & (yo >= 0)
        & (yo <= ih - 1)
    )
    t, b, left, r = 0, ch, 0, cw
    while t < b and left < r:
        bad = {
            "t": 1 - ok[t, left:r].mean(),
            "b": 1 - ok[b - 1, left:r].mean(),
            "l": 1 - ok[t:b, left].mean(),
            "r": 1 - ok[t:b, r - 1].mean(),
        }
        edge = max(bad, key=bad.get)
        if bad[edge] == 0:
            return t, b, left, r
        if edge == "t":
            t += 1
        elif edge == "b":
            b -= 1
        elif edge == "l":
            left += 1
        else:
            r -= 1
    return None


def deskew_crop(
    img, poly: list[float], pad_frac: float, min_side: int, *, trim: bool = True
):
    """Rotate a polygon upright and return the tight crop.

    * **Orientation is preserved.** ``minAreaRect`` reports an angle in
      ``[0, 90)``; taking it verbatim would transpose a vertical text column
      into a horizontal strip. manga-ocr reads vertical Japanese natively, so we
      pick whichever of ``angle`` / ``angle - 90`` is the smaller rotation and
      only correct the tilt.
    * **Padding.** The polygons hug the glyphs, and the recogniser was trained
      on crops with a margin; a zero-pad crop clips strokes on the outer
      characters.
    * **The margin stops at the page** (``trim``). Where the padded box runs past
      the page edge, the lines that would sample outside it are cut off, as the
      shipped reader's ``pad_box`` clips its box to the image. ``trim=False``
      fills them with ``BORDER_REPLICATE`` streaks. A crop that never leaves the
      page is the same either way.
    """
    import cv2
    import numpy as np

    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    if angle > 45:
        # OpenCV >= 4.5 reports angle in (0, 90] with (w, h) measured along the
        # rotated axes: an axis-aligned 30x120 box comes back as (120, 30) @ 90°.
        # Taking angle-90 without swapping the extents crops a transposed
        # rectangle; swap so the extents follow the rotation.
        angle -= 90
        w, h = h, w
    w = max(w, 1.0)
    h = max(h, 1.0)
    pad = pad_frac * max(w, h)
    w, h = w + 2 * pad, h + 2 * pad

    rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    ih, iw = img.shape[:2]
    rotated = cv2.warpAffine(
        img,
        rot,
        (iw, ih),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    crop = cv2.getRectSubPix(rotated, (int(round(w)), int(round(h))), (cx, cy))
    if crop is None or crop.size == 0:
        return None, None
    if trim:
        win = _on_page_window(rot, cx, cy, crop.shape[1], crop.shape[0], iw, ih)
        if win is None:
            return None, None
        t, b, left, r = win
        crop = crop[t:b, left:r].copy()
    ch, cw = crop.shape[:2]
    if min(ch, cw) < min_side:
        return None, None
    orient = (
        "vertical" if ch > cw * 1.3 else ("horizontal" if cw > ch * 1.3 else "square")
    )
    return crop, orient


def margin_box(x0, y0, x1, y1, W, H) -> tuple[int, int, int, int]:
    w, h = x1 - x0, y1 - y0
    p = min(0.12 * max(w, h), 0.3 * min(w, h), 4)
    return (
        int(round(max(0, x0 - p))),
        int(round(max(0, y0 - p))),
        int(round(min(W, x1 + p))),
        int(round(min(H, y1 + p))),
    )


def cap_frac(poly) -> float:
    """``deskew_crop``'s ``pad_frac`` (× the long side of the min-area rect) that gives
    the capped margin."""
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    _, (w, h), _ = cv2.minAreaRect(pts)
    L, S = max(w, h, 1.0), max(min(w, h), 1.0)
    return min(0.12 * L, 0.3 * S, 4) / L


def rect_quad(x0, y0, x1, y1) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def ink_mask(crop: np.ndarray, border: bool = True) -> np.ndarray:
    """Otsu ink; with ``border``, connected components touching the crop edge go
    (balloon outlines, panel lines, the neighbour's ink inside the 4 px margin)."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if not border:
        return b > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    H, W = b.shape
    x, y, w, h = st[:, 0], st[:, 1], st[:, 2], st[:, 3]
    keep = ~((x == 0) | (y == 0) | (x + w == W) | (y + h == H))
    keep[0] = False
    return keep[lab]


def runs(on: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) runs of True."""
    d = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def segment(
    ink: np.ndarray, vertical: bool, gap_frac: float, sliver: float, merge: float = 0.0
) -> list[tuple[int, int, int, int]]:
    """Columns (vertical) / rows (horizontal) in reading order as (a0, a1, b0, b1):
    ``a`` across the reading axis, ``b`` the ink extent along it."""
    m = ink if vertical else ink.T  # columns run along axis 0
    prof = m.sum(axis=0)
    if prof.max() == 0:
        return []
    on = prof > gap_frac * prof.max()
    segs = [(a, b) for a, b in runs(on)]
    if not segs:
        return []
    widest = max(b - a for a, b in segs)
    # a gap narrower than merge × the widest run sits inside a glyph (は, い, ?)
    merged = [segs[0]]
    for a, b in segs[1:]:
        if a - merged[-1][1] < merge * widest:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    widest = max(b - a for a, b in merged)
    segs = [(a, b) for a, b in merged if (b - a) >= sliver * widest]
    out = []
    for a, b in segs:
        rows = np.flatnonzero(m[:, a:b].any(axis=1))
        out.append((a, b, int(rows[0]), int(rows[-1]) + 1))
    if vertical:
        out.reverse()  # right to left
    return out


SMALL = set("ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ、。，．,")
DOTS = set(".．・…‥")


def glyph_weight(ch: str) -> float:
    """Length a glyph takes along a column, in full-width cells."""
    if ch in SPACES:
        return 0.5
    if ch in DOTS:
        return 0.35
    if ch in SMALL:
        return 0.7
    if ch in "!！?？":
        return 0.6
    if ch.isascii():
        return 0.55
    return 1.0


def guess_vertical(crop: np.ndarray, raw: str) -> bool:
    """Vertical unless the box is clearly wider than a single row of the label."""
    h, w = crop.shape[:2]
    return not (w > 1.6 * h)
