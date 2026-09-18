"""S line (plan_synth): draw JA text into a generated scene's bubble — line
splitting with kinsoku, fit to the bubble region, the anchor erase, and the
tategaki / horizontal compositor.
"""

from __future__ import annotations

import random

from ..bubble import bubble_interior, bubble_mask, ring_median

# ----------------------------------------------------------------------------
# S line (plan_synth): draw JA text into a generated scene's bubble


# glyph cell pitch along a line / column and between lines / columns
V_PITCH, V_GAP = 1.05, 1.15  # vertical: glyphs down a column, columns apart
H_PITCH, H_GAP = 1.0, 1.2  # horizontal: glyphs along a line, lines apart
# a new line / column must not start with these (kinsoku, the common subset)
NO_HEAD = set("、。，．・ー〜～!?！？」』）)ゃゅょっャュョッァィゥェォ")
# … and must not leave these at the end of the previous one
NO_TAIL = set("「『（(")
# drawn rotated a quarter turn in a column (the long-vowel bar and dashes)
V_ROTATE = set("ー〜～…‥－-—–")
# nudged to the top-right of their cell in a column
V_PUNCT = set("、。，．")
# a layout with more lines wins over fewer only when its glyph is this much
# larger — a bubble is filled, but a phrase that fits one column stays one
MORE_LINES_GAIN = 1.4


def split_lines(text: str, k: int, cuts=None) -> list[str] | None:
    """``text`` as ``k`` near-equal lines cut only at ``cuts`` (allowed
    character offsets — the caller's piece boundaries; ``None`` = anywhere),
    each cut nudged forward past a kinsoku violation. ``None`` when ``k``
    lines are not possible."""
    n = len(text)
    if k == 1:
        return [text]
    allowed = sorted(
        set(range(1, n)) if cuts is None else {c for c in cuts if 0 < c < n}
    )
    if len(allowed) < k - 1:
        return None
    chosen: list[int] = []
    for i in range(1, k):
        target = n * i / k
        lo = chosen[-1] + 1 if chosen else 1
        cand = [c for c in allowed if c >= lo]
        if not cand:
            return None
        c = min(cand, key=lambda x: abs(x - target))
        # kinsoku: prefer not to open a line with NO_HEAD or close one on
        # NO_TAIL — nudge forward; when every later cut violates too (a
        # trailing run of ・・・), keep the nearest cut rather than refuse
        c0 = c
        while c < n and (text[c] in NO_HEAD or text[c - 1] in NO_TAIL):
            nxt = [x for x in cand if x > c]
            if not nxt:
                c = c0
                break
            c = nxt[0]
        chosen.append(c)
    bounds = [0, *chosen, n]
    lines = [text[a:b] for a, b in zip(bounds, bounds[1:])]
    return lines if all(lines) else None


def _block_size(d, font, fs, lines, vertical):
    """(w, h) of the drawn block for ``lines`` at ``fs``."""
    k = len(lines)
    if vertical:
        w = max(d.textlength(ch, font=font) for ln in lines for ch in ln)
        w = max(w, fs) + (k - 1) * fs * V_GAP
        h = max(len(ln) for ln in lines) * fs * V_PITCH
    else:
        w = max(d.textlength(ln, font=font) for ln in lines)
        h = fs + (k - 1) * fs * H_GAP
    return w, h


def fit_text(
    d,
    text: str,
    font_path: str,
    region,
    vertical: bool,
    min_glyph: int,
    fill_frac: float = 0.9,
    max_lines: int = 1,
    cuts=None,
    fewest_lines: bool = False,
):
    """Largest font size whose text block fits ``region`` (inner
    ``fill_frac`` — 0.9 fills the bubble edge to edge, ``--scene_fill`` 0.7
    leaves manga-like air around the glyphs) over 1..``max_lines`` lines
    (columns when ``vertical``), lines cut only at ``cuts``. A layout with
    more lines replaces one with fewer only when its glyph is
    ``MORE_LINES_GAIN``× larger — never when ``fewest_lines`` (the sentence
    arm, user 2026-09-16: a line that fits in one column stays one column).
    The glyph cell must be at least ``min_glyph`` px, else ``None``.
    Returns ``(font, fs, lines)``."""
    from PIL import ImageFont

    rx0, ry0, rx1, ry1 = region
    rw, rh = (rx1 - rx0) * fill_frac, (ry1 - ry0) * fill_frac
    best = None
    for k in range(1, max_lines + 1):
        lines = split_lines(text, k, cuts)
        if lines is None:
            continue
        m = max(len(ln) for ln in lines)
        if vertical:
            fs = int(min(rw / (1 + (k - 1) * V_GAP), rh / (m * V_PITCH)))
        else:
            fs = int(min(rh / (1 + (k - 1) * H_GAP), rw / (m * H_PITCH)))
        if fs < min_glyph:
            continue
        for _ in range(4):
            font = ImageFont.truetype(font_path, fs, index=0)
            tw, th = _block_size(d, font, fs, lines, vertical)
            sc = min(rw / max(tw, 1e-6), rh / max(th, 1e-6))
            if sc >= 1.0:
                if best is None or fs >= best[1] * MORE_LINES_GAIN:
                    best = (font, fs, lines)
                break
            fs = int(fs * min(sc, 0.97))
            if fs < min_glyph:
                break
        if best is not None and fewest_lines:
            return best
    return best


def region_capacity(
    region,
    min_glyph: int,
    fill_frac: float = 0.9,
    max_lines: int = 1,
    vertical_only: bool = False,
) -> int:
    """How many glyphs the region holds at ``min_glyph`` px per cell over up
    to ``max_lines`` columns (vertical) or lines (horizontal), inner
    ``fill_frac`` — the larger of the two orientations, or the columns alone
    when ``vertical_only`` (the tategaki-only sentence arm)."""
    rw, rh = (region[2] - region[0]) * fill_frac, (region[3] - region[1]) * fill_frac
    g = min_glyph
    v_cols = int((rw - g) / (g * V_GAP)) + 1 if rw >= g else 0
    v_cap = int(rh / (g * V_PITCH)) * min(v_cols, max_lines)
    if vertical_only:
        return v_cap
    h_rows = int((rh - g) / (g * H_GAP)) + 1 if rh >= g else 0
    h_cap = int(rw / (g * H_PITCH)) * min(h_rows, max_lines)
    return max(v_cap, h_cap)


def erase_paint(arr, tb, reg, open_ok: bool = False):
    """Bool HxW mask of what the composite erase paints for one anchor: the
    usable region ∪ the text box padded by a quarter of its size (detector
    boxes run tight), clipped to the bubble interior (flood mask, letter
    holes filled — a rectangle's corners would poke past a round outline).
    ``None`` when no bubble mask is found — unless ``open_ok`` (bubble-less
    frames such as `sfx`): then the plain rectangle is painted, a flat
    ring-median patch on the scene."""
    import numpy as np

    H, W = arr.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in tb)
    px, py = (x1 - x0) // 4 + 4, (y1 - y0) // 4 + 4
    ex0, ey0 = max(0, min(reg[0], x0 - px)), max(0, min(reg[1], y0 - py))
    ex1, ey1 = min(W, max(reg[2], x1 + px)), min(H, max(reg[3], y1 + py))
    m = bubble_mask(arr, tb)
    if m is None and not open_ok:
        return None
    paint = np.zeros((H, W), dtype=bool)
    paint[ey0:ey1, ex0:ex1] = True
    if m is not None:
        paint &= bubble_interior(m)
    return paint


def erase_uniform(arr, tb, reg, tol: int = 24, width: int = 3) -> float:
    """Seam test for an open (no closed bubble) erase: the share of the
    ``width``-px ring just *outside* the paint rectangle that sits within
    ``tol`` of the ring-median fill. 1.0 means the painted rectangle has no
    visible edge — whatever was inside (the letters, an ascender the
    detector box clipped) vanishes into the same colour — while an outline
    or art crossing the rectangle's edge shows as a cut and lowers it. The
    judge grows the rectangle from the smallest step, so a white bubble
    with a broken outline keeps its outline (s1 832) and a bare white
    ground keeps nothing to keep."""
    import numpy as np

    paint = erase_paint(arr, tb, reg, open_ok=True)
    H, W = arr.shape[:2]
    ys, xs = np.nonzero(paint)
    if len(xs) == 0:
        return 0.0
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    ring = np.zeros((H, W), dtype=bool)
    ring[
        max(0, y0 - width) : min(H, y1 + width), max(0, x0 - width) : min(W, x1 + width)
    ] = True
    ring[y0:y1, x0:x1] = False
    if not ring.any():
        return 0.0
    fill = np.array(ring_median(arr, tb), dtype=np.int16)
    d = np.abs(arr.astype(np.int16) - fill).max(axis=2)
    return float((d[ring] <= tol).mean())


def erase_lost(arr, tb, reg, tol: int = 24) -> float:
    """Share of an open (rectangle) erase's painted pixels *outside* the text
    box that are not the fill colour — the outline, shelf lines, hair the
    rectangle paints over. The ring seam test (``erase_uniform``) misses a
    thin outline crossing the rectangle; at <= 0.02 the erases read clean,
    from 0.03 up outlines are cut (Δ0.9 sheets: s1 627 0.16, s1 65 0.03)."""
    import numpy as np

    paint = erase_paint(arr, tb, reg, open_ok=True)
    H, W = arr.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in tb)
    paint[max(0, y0 - 2) : min(H, y1 + 2), max(0, x0 - 2) : min(W, x1 + 2)] = False
    if not paint.any():
        return 0.0
    fill = np.array(ring_median(arr, tb), dtype=np.int16)
    ink = np.abs(arr.astype(np.int16) - fill).max(axis=2) > tol
    return float((ink & paint).sum()) / float(paint.sum())


def region_offset(tb, reg) -> float:
    """Distance of the region's centre from the text box's centre, in text
    box half-sizes (the larger axis). The base centres its text in a bubble
    (median 0.11); a flood that leaked through an outline gap into a panel
    strip or a figure moves the region off the text (ja_comic 770 1.4,
    sl1w 962 5.8)."""
    tcx, tcy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
    rcx, rcy = (reg[0] + reg[2]) / 2, (reg[1] + reg[3]) / 2
    return max(
        abs(rcx - tcx) / max(1.0, (tb[2] - tb[0]) / 2),
        abs(rcy - tcy) / max(1.0, (tb[3] - tb[1]) / 2),
    )


def anchor_ink(arr, tb, tol: int = 24, min_px: int = 30):
    """Median colour of the anchor's ink inside text box ``tb`` (pixels
    farther than ``tol`` from the ring-median fill), or ``None`` when too
    few — the base's own lettering colour for this scene (s1 00006: a purple
    "hi"), inherited by the drawn glyph so ink colour is not one more
    constant the rows can absorb (user, 2026-09-15). Outline pixels are in
    the median too; with a thin outline the letter body wins."""
    import numpy as np

    H, W = arr.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in tb)
    sub = arr[max(0, y0) : min(H, y1), max(0, x0) : min(W, x1)].astype(np.int16)
    if sub.size == 0:
        return None
    fill = np.array(ring_median(arr, tb), dtype=np.int16)
    ink = sub[np.abs(sub - fill).max(axis=2) > tol]
    if len(ink) < min_px:
        return None
    return tuple(int(v) for v in np.median(ink, axis=0))


def anchor_residual(arr, tb, reg, tol: int = 24, open_ok: bool = False) -> float:
    """Share of the anchor's ink (pixels in the text box farther than ``tol``
    from the ring-median fill) that ``erase_paint`` would leave standing.
    ≈ 0 for a bubble the flood found (edge pixels only); ≈ 1 when the flood
    took another blob — the letters survive under the drawn kana and the
    usable region is not this bubble's (s0: 12 of 186 kept scenes). 1.0
    when there is no bubble mask (0 under ``open_ok``: the rectangle covers
    the box)."""
    import numpy as np

    paint = erase_paint(arr, tb, reg, open_ok)
    if paint is None:
        return 1.0
    H, W = arr.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in tb)
    fill = np.array(ring_median(arr, tb), dtype=np.int16)
    sub = np.zeros((H, W), dtype=bool)
    sub[max(0, y0 - 2) : min(H, y1 + 2), max(0, x0 - 2) : min(W, x1 + 2)] = True
    ink = (np.abs(arr.astype(np.int16) - fill).max(axis=2) > tol) & sub
    n = int(ink.sum())
    return float((ink & ~paint).sum()) / n if n else 0.0


def _draw_vertical_glyph(layer, ld, ch, x, y, fs, font, color, kw):
    """One glyph of a column at cell centre ``x``, cell top ``y``: the
    long-vowel bar / dashes a quarter turn clockwise, 、。 to the top-right
    of the cell, everything else upright and centred."""
    from PIL import Image, ImageDraw

    w = ld.textlength(ch, font=font)
    if ch in V_ROTATE:
        cell = int(fs * 1.5)
        tile = Image.new("RGBA", (cell, cell), (0, 0, 0, 0))
        td = ImageDraw.Draw(tile)
        td.text(
            ((cell - w) / 2, (cell - fs) / 2 - fs * 0.1),
            ch,
            fill=color,
            font=font,
            **kw,
        )
        tile = tile.rotate(-90, resample=Image.BICUBIC)
        layer.alpha_composite(tile, (int(x - cell / 2), int(y + fs / 2 - cell / 2)))
        return
    if ch in V_PUNCT:
        ld.text((x - w / 2 + fs * 0.4, y - fs * 0.4), ch, fill=color, font=font, **kw)
        return
    ld.text((x - w / 2, y), ch, fill=color, font=font, **kw)


def render_into_scene(
    scene: dict,
    text: str,
    font_path: str,
    rng: random.Random,
    min_glyph: int = 40,
    stroke: bool = False,
    fill_frac: float = 0.9,
    tilt_frac: float = 0.3,
    tilt_deg: float = 7.0,
    max_lines: int = 1,
    cuts=None,
    vertical_only: bool = False,
    fewest_lines: bool = False,
    ref_text: str | None = None,
):
    """Erase every anchor bubble's usable region (plus the text box padded by
    a quarter of its size — detector boxes run tight) with the bubble's
    ring-median colour, only *inside the bubble interior* (flood mask,
    letter holes filled — a rectangle's corners would poke past a round
    outline), and draw ``text`` fitted into the inner ``fill_frac`` of the
    headline region over up to ``max_lines`` columns / lines, cut only at
    ``cuts`` (character offsets — the caller's piece boundaries, so a row's
    unit is never split across lines). **Vertical first** (user,
    2026-09-16: manga lettering is tategaki): columns right-to-left
    whenever the text fits that way at ``min_glyph``, horizontal lines only
    when it does not — never when ``vertical_only`` (the sentence arm,
    2026-09-16: a multi-glyph text that does not fit as columns is a miss and
    the caller re-picks the scene; a single glyph has no orientation).
    Returns ``(image, drawn text box)`` or ``None`` when the text does not
    fit at ``min_glyph`` px per glyph (the caller draws a shorter text). Other anchor bubbles are left erased (empty bubble).
    ``stroke``: a thin outline in the fill colour around the glyphs (manga
    lettering over art).

    ``ref_text`` (ΔFM, plan_synth2): a sibling of the same glyph count drawn
    by the *same* fit — same erase, font, size, line lengths, positions,
    colour and tilt draw — with ``ref_text``'s glyphs in place of ``text``'s.
    Returns ``(image, box, ref image, ref box)``; the two images are asserted
    pixel-identical outside the union of the two boxes, so the pair differs
    by the glyphs alone."""
    import numpy as np
    from PIL import Image, ImageDraw

    im = Image.open(scene["file"]).convert("RGB")
    arr = np.array(im)
    W, H = im.size
    head = scene["regions"].index(scene["region"])
    fills = []
    # a kept scene whose bubble is None passed the judge as open (bubble-less
    # frame): paint the rectangle
    bubbles = scene.get("bubbles") or [None] * len(scene["regions"])
    ink = anchor_ink(arr, scene["boxes_anchor"][head])  # read before the erase
    for tb, reg, bub in zip(scene["boxes_anchor"], scene["regions"], bubbles):
        fill = ring_median(arr, tb)
        fills.append(fill)
        paint = erase_paint(arr, tb, reg, open_ok=bub is None)
        if paint is None:
            return None
        arr[paint] = fill
    # specks the judge kept (plan_synth2 Δ0.9): same erase, no text drawn
    for tb, reg, bub in zip(
        scene.get("boxes_speck", ()),
        scene.get("speck_regions", ()),
        scene.get("speck_bubbles", ()),
    ):
        paint = erase_paint(arr, tb, reg, open_ok=bub is None)
        if paint is None:
            return None
        arr[paint] = ring_median(arr, tb)
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    region = scene["region"]
    vertical = len(text) > 1
    fit = (
        fit_text(
            d,
            text,
            font_path,
            region,
            True,
            min_glyph,
            fill_frac,
            max_lines,
            cuts,
            fewest_lines,
        )
        if vertical
        else None
    )
    if fit is None and vertical and vertical_only:
        return None
    if fit is None:
        vertical = False
        fit = fit_text(
            d,
            text,
            font_path,
            region,
            False,
            min_glyph,
            fill_frac,
            max_lines,
            cuts,
            fewest_lines,
        )
    if fit is None:
        return None
    font, fs, lines = fit
    tw, th = _block_size(d, font, fs, lines, vertical)
    fill = fills[head]
    dark_bg = sum(fill) / 3 < 100
    # the anchor's own ink colour when it contrasts with the fill (≥ 60 on
    # some channel), else the old contrast rule
    if ink is not None and max(abs(a - b) for a, b in zip(ink, fill)) >= 60:
        color = ink
    else:
        color = rng.choice(
            [(240, 240, 240), "white"]
            if dark_bg
            else ["black", "black", (30, 30, 30), (60, 40, 40)]
        )
    cx, cy = (region[0] + region[2]) / 2, (region[1] + region[3]) / 2
    kw = {"stroke_width": max(1, fs // 24), "stroke_fill": fill} if stroke else {}
    # the text goes on its own layer so it can be tilted a few degrees
    # (user, 2026-09-15: hand-lettered bubbles are rarely dead level) —
    # 30 % of composites, ±7°, about the text block's centre. The tilt is
    # drawn once so a sibling gets the same one.
    tilt = rng.uniform(-tilt_deg, tilt_deg) if rng.random() < tilt_frac else None

    def draw(text_lines):
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        if vertical:
            # columns right-to-left, glyphs top-down; the block is centred on
            # the region, every column centred on its own height
            x = cx + tw / 2 - fs / 2  # centre of the first (rightmost) column
            for ln in text_lines:
                y = cy - len(ln) * fs * V_PITCH / 2
                for ch in ln:
                    _draw_vertical_glyph(layer, ld, ch, x, y, fs, font, color, kw)
                    y += fs * V_PITCH
                x -= fs * V_GAP
        else:
            y = cy - th / 2 - fs * 0.1
            for ln in text_lines:
                w = ld.textlength(ln, font=font)
                ld.text((cx - w / 2, y), ln, fill=color, font=font, **kw)
                y += fs * H_GAP
        if tilt is not None:
            layer = layer.rotate(tilt, resample=Image.BICUBIC, center=(cx, cy))
        bb = layer.getbbox()  # alpha bbox: the drawn (and tilted) glyphs
        if bb is None:
            return None
        out = im.copy()
        out.paste(layer, (0, 0), layer)
        box = [
            int(max(0, bb[0] - 2)),
            int(max(0, bb[1] - 2)),
            int(min(W, bb[2] + 2)),
            int(min(H, bb[3] + 2)),
        ]
        return out, box

    drawn = draw(lines)
    if drawn is None or ref_text is None:
        return drawn
    assert len(ref_text) == len(text), (ref_text, text)
    # the sibling takes the item's line lengths (not a re-split: kinsoku
    # nudges depend on the glyphs, and the layout must be the same)
    ref_lines, off = [], 0
    for ln in lines:
        ref_lines.append(ref_text[off : off + len(ln)])
        off += len(ln)
    drawn_ref = draw(ref_lines)
    if drawn_ref is None:
        return None
    (im_b, box_b), (im_a, box_a) = drawn, drawn_ref
    ux0, uy0 = min(box_b[0], box_a[0]), min(box_b[1], box_a[1])
    ux1, uy1 = max(box_b[2], box_a[2]), max(box_b[3], box_a[3])
    diff = (np.array(im_b) != np.array(im_a)).any(axis=2)
    diff[uy0:uy1, ux0:ux1] = False
    assert not diff.any(), "sibling differs outside the union text box"
    return im_b, box_b, im_a, box_a
