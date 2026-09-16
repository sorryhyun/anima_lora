"""Speech-bubble geometry on a generated scene (numpy + cv2, CPU).

``bubble_mask`` flood-fills the bubble's fill colour from a ring around the
detector's text box and returns the fill component (ink excluded);
``bubble_interior`` fills that component's holes (the letters) so the
result is "every pixel inside the bubble outline". The scenes filter uses
the mask for its bubble bbox / usable region, the data stage's swap erases
only inside the interior — so no erase rectangle can poke past the
outline (user, 2026-09-14) and no leaked background blob passes as a
bubble (the mask must enclose its text box).
"""

from __future__ import annotations


def ring_median(arr, box, pad: int = 4, width: int = 4):
    """Median colour of a ring ``pad``–``pad+width`` px outside ``box`` (the
    bubble's local fill colour). ``arr``: HxWx3 in the array's own channel
    order (the caller's RGB or BGR)."""
    import numpy as np

    H, W = arr.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in box)
    ring = np.zeros((H, W), dtype=bool)
    ring[
        max(0, y0 - pad - width) : min(H, y1 + pad + width),
        max(0, x0 - pad - width) : min(W, x1 + pad + width),
    ] = True
    ring[max(0, y0 - pad) : min(H, y1 + pad), max(0, x0 - pad) : min(W, x1 + pad)] = (
        False
    )
    if not ring.any():
        return (255, 255, 255)
    return tuple(int(v) for v in np.median(arr[ring].reshape(-1, 3), axis=0))


def bubble_mask(arr, box, pad: int = 4, tol: int = 24):
    """Bool HxW mask of the bubble fill around text ``box``, or ``None``.
    The fill colour is the ring median; pixels farther than ``tol`` from it
    are ink, thickened 2 px so a sketchy outline still closes. Up to 12 ring
    seeds are flooded one at a time: a fill that reaches the image border
    AND is large (> 8 %) is a leak through the outline, > 35 % of the image
    is not a bubble, a small border-touching fill is a bubble clipped by
    the canvas edge (kept); the largest survivor wins, provided it is at
    most 12× the text box (a fill many times larger ran into a
    panel-bounded background) and its bbox encloses the text box (a blob
    beside the text is not the text's bubble)."""
    import cv2
    import numpy as np

    H, W = arr.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in box)
    ring = np.zeros((H, W), dtype=bool)
    ring[
        max(0, y0 - pad - 4) : min(H, y1 + pad + 4),
        max(0, x0 - pad - 4) : min(W, x1 + pad + 4),
    ] = True
    ring[max(0, y0 - pad) : min(H, y1 + pad), max(0, x0 - pad) : min(W, x1 + pad)] = (
        False
    )
    if not ring.any():
        return None
    fill = np.median(arr[ring].reshape(-1, 3), axis=0)
    dist = np.abs(arr.astype(np.int16) - fill.astype(np.int16)).max(axis=2)
    ink = (dist > tol).astype(np.uint8)
    ink = cv2.dilate(ink, np.ones((5, 5), np.uint8))  # close 2 px gaps
    canvas = np.where(ink[..., None] > 0, 0, 255).astype(np.uint8)
    canvas = np.repeat(canvas, 3, axis=2)
    ys, xs = np.nonzero(ring & (dist <= tol))
    if len(xs) == 0:
        return None
    idx = np.linspace(0, len(xs) - 1, num=min(12, len(xs))).astype(int)
    best = None
    seen = np.zeros((H, W), dtype=np.uint8)
    for k in idx:
        sx, sy = int(xs[k]), int(ys[k])
        if seen[sy, sx]:
            continue
        mask = np.zeros((H + 2, W + 2), dtype=np.uint8)
        cv2.floodFill(
            canvas,
            mask,
            (sx, sy),
            0,
            (10, 10, 10),
            (10, 10, 10),
            cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8) | 4,
        )
        m = mask[1:-1, 1:-1]
        seen |= m
        n = int(m.sum() // 255)
        touches = bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
        if n > 0.35 * H * W or (touches and n > 0.08 * H * W):
            continue
        if best is None or n > best[0]:
            best = (n, m)
    if best is None:
        return None
    # the ink dilation ate 2 px of interior at the outline: give it back
    m = cv2.dilate(best[1], np.ones((5, 5), np.uint8)) > 0
    ys, xs = np.nonzero(m)
    bx0, by0, bx1, by1 = (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )
    if (bx1 - bx0) * (by1 - by0) > 12 * max(1, (x1 - x0) * (y1 - y0)):
        return None
    # containment: the fill must surround its text (tolerance: a quarter of
    # the box each side — detector boxes run loose on descenders)
    tx, ty = (x1 - x0) // 4, (y1 - y0) // 4
    if bx0 > x0 + tx or by0 > y0 + ty or bx1 < x1 - tx or by1 < y1 - ty:
        return None
    # a pocket between an open outline and the letters' ink encloses the box
    # by bbox yet covers none of it (s1 832: outline broken at 12 o'clock,
    # every other seed leaked) — its interior must hold the text
    if bubble_interior(m)[y0:y1, x0:x1].mean() < 0.5:
        return None
    return m


def bubble_interior(mask):
    """Every pixel inside the bubble outline: the fill mask with its holes
    (the letters) filled — the complement of what a flood from the image
    border reaches over the non-fill pixels."""
    import cv2
    import numpy as np

    H, W = mask.shape
    canvas = np.where(mask, 0, 255).astype(np.uint8)
    reach = np.zeros((H + 2, W + 2), dtype=np.uint8)
    for sx, sy in [(x, y) for x in range(W) for y in (0, H - 1)] + [
        (x, y) for x in (0, W - 1) for y in range(H)
    ]:
        if canvas[sy, sx] == 255 and reach[sy + 1, sx + 1] == 0:
            cv2.floodFill(
                canvas.copy(),
                reach,
                (sx, sy),
                0,
                (0,),
                (0,),
                cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (1 << 8) | 4,
            )
    return reach[1:-1, 1:-1] == 0


def bubble_bbox(mask):
    import numpy as np

    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
