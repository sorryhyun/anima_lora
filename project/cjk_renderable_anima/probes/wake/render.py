"""Font renders of kana strings, and square crops of corpus bubbles.

Bit-identity contract: a layout draws its random choices in the pre-W2a
order, so an unbalanced ``--layout v1`` data dir rebuilds identically.
"""

from __future__ import annotations

import random
from glob import glob
from pathlib import Path

from .bubble import bubble_interior, bubble_mask, ring_median
from .common import wh


FONT_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"


def find_fonts() -> list[str]:
    # Noto CJK .ttc index 0 = the JP face. DroidSansFallback (pre-S0 data
    # dirs had it) draws kanji in Chinese-styled forms and is out (user,
    # 2026-09-14); Noto Sans CJK is out too (user, 2026-09-15: reads
    # ambiguous next to the manga faces) — Noto Serif CJK (= 源ノ明朝, the
    # article's serif pick) stays and is the full-coverage fallback. Plus
    # the manga lettering set under assets/fonts (FONTS.md — antique,
    # rounded / angular gothic, logo, marker, handwriting), gitignored.
    return sorted(glob("/usr/share/fonts/opentype/noto/NotoSerifCJK*.ttc")) + sorted(
        str(p) for p in FONT_DIR.glob("*.[ot]tf")
    )


_CMAP: dict[str, set] = {}


def font_covers(font_path: str, text: str) -> bool:
    """Every char of ``text`` has a glyph in the font's cmap (index 0 of a
    .ttc). Hand-lettered fonts stop at JIS level 2 and a missing glyph
    renders as tofu, which the row would learn."""
    cm = _CMAP.get(font_path)
    if cm is None:
        from fontTools.ttLib import TTFont

        tt = TTFont(font_path, fontNumber=0, lazy=True)
        cm = set(tt.getBestCmap().keys())
        tt.close()
        _CMAP[font_path] = cm
    return all(ord(ch) in cm for ch in text)


def pick_font(text: str, fonts: list[str], rng: random.Random) -> str:
    """One draw among the fonts that cover ``text`` (Noto always does)."""
    ok = [f for f in fonts if font_covers(f, text)]
    return rng.choice(ok or fonts)


JITTER_BG_LIGHT = [
    "white",
    "white",
    (235, 235, 235),
    (245, 240, 230),
    (220, 225, 235),
    (250, 235, 200),
    (200, 215, 235),
]
JITTER_BG_DARK = [(30, 30, 30), (20, 25, 40), (60, 40, 50), (45, 45, 45), (10, 10, 10)]
JITTER_INK_DARK = [
    "black",
    "black",
    (30, 30, 30),
    (140, 30, 30),
    (30, 40, 140),
    (20, 100, 50),
    (150, 60, 120),
]
JITTER_INK_LIGHT = [
    "white",
    "white",
    (240, 235, 200),
    (250, 220, 60),
    (120, 220, 240),
    (255, 150, 160),
]


def sample_layout(
    n: int,
    rng: random.Random,
    size=512,
    mode: str = "v1",
    bubble_frac: float = 0.6,
) -> dict:
    """Every random choice of one render for an ``n``-char string, drawn in the
    pre-W2a order so unbalanced data dirs rebuild bit-identically.

    ``mode="jitter"`` (data lever, 2026-09-14) draws the v1 fields and then
    overrides what v1 held constant: glyph position (anywhere on the canvas /
    inside the bubble), size down to 60 px, ink colour + optional outline,
    dark backgrounds, a bubble of random size and place. Every row then has
    the same layout statistics and the only thing a row can explain is the
    glyph — the constant "big black glyph centred on a light canvas" was the
    shared gradient direction that drove the encoder's table to rank 1."""
    # mixed shapes (2026-09-14): glyph / bubble sizes were drawn for a 512
    # canvas; scale them by the short side so a 384² render keeps the same
    # glyph-to-canvas statistics (a 512 draw is bit-identical: int(x * 1.0)).
    W, H = wh(size)
    sc = min(W, H) / 512
    # the draw is consumed whatever the share, so 1.0 (S0b option (a): every
    # flat item inside the bubble, one flat layout for ``c_flat``) keeps the
    # rest of the sequence bit-identical to the 0.6 build
    lay = {"bubble": rng.random() < bubble_frac}
    lay["bg"] = rng.choice(
        ["white", "white", (235, 235, 235), (245, 240, 230), (220, 225, 235)]
    )
    if lay["bubble"]:
        # light screentone-ish dots + a white ellipse
        lay["dots"] = [(rng.randrange(W), rng.randrange(H)) for _ in range(900)]
        lay["pad"] = int(rng.randint(30, 70) * sc)
        lay["outline"] = rng.randint(2, 5)
    lay["vertical"] = rng.random() < 0.65 if n > 1 else rng.random() < 0.3
    lay["fs"] = int(
        (rng.randint(110, 200) if n == 1 else rng.randint(int(320 / n), int(400 / n)))
        * sc
    )
    lay["color"] = rng.choice(["black", "black", (30, 30, 30), (60, 40, 40)])
    lay["rot"] = rng.uniform(-6, 6) if rng.random() < 0.3 else None
    if mode == "jitter":
        dark = rng.random() < 0.35
        lay["bg"] = rng.choice(JITTER_BG_DARK if dark else JITTER_BG_LIGHT)
        # bubble fill stays white, so ink inside a bubble is always dark
        ink_light = dark and not lay["bubble"]
        lay["color"] = rng.choice(JITTER_INK_LIGHT if ink_light else JITTER_INK_DARK)
        if rng.random() < 0.25:
            lay["stroke"] = rng.randint(2, 6)
            lay["stroke_fill"] = "black" if ink_light else "white"
        lay["fs"] = int(
            (
                rng.randint(60, 200)
                if n == 1
                else rng.randint(int(180 / n), int(400 / n))
            )
            * sc
        )
        # normalised anchors; render_string maps them into the feasible range
        # once it knows the text extent (layout dicts stay font-free)
        lay["pos"] = (rng.random(), rng.random())
        if lay["bubble"]:
            lay["box"] = (
                rng.random(),
                rng.random(),
                rng.uniform(0.45, 1.0),
                rng.uniform(0.45, 1.0),
            )
    return lay


def render_string(
    text: str,
    font_path: str,
    rng: random.Random,
    size=512,
    layout=None,
    mode: str = "v1",
    bubble_frac: float = 0.6,
):
    """``layout`` (from ``sample_layout``) pins canvas/bubble/size/position so
    several strings render in the same layout; ``None`` draws a fresh one
    (``bubble_frac`` = share drawn inside a bubble). Returns ``(image,
    drew_bubble)``."""
    from PIL import Image, ImageDraw, ImageFont

    n = len(text)
    W, H = wh(size)
    lay = (
        layout if layout is not None else sample_layout(n, rng, size, mode, bubble_frac)
    )
    bubble, bg, fs, color = lay["bubble"], lay["bg"], lay["fs"], lay["color"]
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)

    def extent(fs):
        font = ImageFont.truetype(font_path, fs, index=0)
        if lay["vertical"]:
            tw = max(d.textlength(ch, font=font) for ch in text)
            th = n * fs * 1.05
        else:
            tw = d.textlength(text, font=font)
            th = fs
        return font, tw, th

    # text block extent (w, h) and its centre; v1 = canvas centre, jitter =
    # anchored inside the bubble's inscribed rectangle / the canvas
    font, tw, th = extent(fs)
    if bubble and "pos" not in lay:
        # v1 fit (2026-09-14, word data): a 3-char vertical string at 400/n px
        # ran past the ellipse (可愛い). Shrink only when the block does not fit
        # the inscribed rectangle of the centred ellipse — singles never do,
        # so the pre-word data dirs' single renders are unchanged.
        half = (min(W, H) / 2 - lay["pad"]) / 2**0.5 - 6
        k = min(1.0, half / max(tw / 2, 1e-6), half / max(th / 2, 1e-6))
        if k < 1.0:
            fs = max(12, int(fs * k))
            font, tw, th = extent(fs)
    cx, cy = W / 2, H / 2
    ebox = (lay["pad"], lay["pad"], W - lay["pad"], H - lay["pad"]) if bubble else None
    if "pos" in lay:
        m = 8
        if bubble:
            u, v, sw, sh = lay["box"]
            # half-axes: big enough that the inscribed rectangle holds the text
            amin = (tw / 2 + m) * 2**0.5
            bmin = (th / 2 + m) * 2**0.5
            amax, bmax = W / 2 - m, H / 2 - m
            ea = min(amax, max(amin, sw * amax))
            eb = min(bmax, max(bmin, sh * bmax))
            ecx = ea + m + u * max(0.0, W - 2 * (ea + m))
            ecy = eb + m + v * max(0.0, H - 2 * (eb + m))
            ebox = (ecx - ea, ecy - eb, ecx + ea, ecy + eb)
            rx = max(0.0, ea / 2**0.5 - tw / 2 - m / 2)
            ry = max(0.0, eb / 2**0.5 - th / 2 - m / 2)
            cx = ecx + (2 * lay["pos"][0] - 1) * rx
            cy = ecy + (2 * lay["pos"][1] - 1) * ry
        else:
            cx = tw / 2 + m + lay["pos"][0] * max(0.0, W - tw - 2 * m)
            cy = th / 2 + m + lay["pos"][1] * max(0.0, H - th - 2 * m)
    if bubble:
        for x, y in lay["dots"]:
            d.ellipse((x, y, x + 2, y + 2), fill=(150, 150, 150))
        d.ellipse(ebox, fill="white", outline="black", width=lay["outline"])
    stroke = {}
    if lay.get("stroke"):
        stroke = {"stroke_width": lay["stroke"], "stroke_fill": lay["stroke_fill"]}
    if lay["vertical"]:
        y = cy - th / 2
        for ch in text:
            w = d.textlength(ch, font=font)
            d.text((cx - w / 2, y), ch, fill=color, font=font, **stroke)
            y += fs * 1.05
    else:
        d.text(
            (cx - tw / 2, cy - fs / 2 - fs * 0.1), text, fill=color, font=font, **stroke
        )
    if lay["rot"] is not None:
        im = im.rotate(lay["rot"], fillcolor=bg, resample=Image.BICUBIC)
    return im, bubble


def crop_bubble(img_path: Path, box, size=512):
    """Square crop around a corpus bubble box (1.35× + margin), resized."""
    from PIL import Image

    im = Image.open(img_path).convert("RGB")
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    side = int(max(w, h) * 1.35) + 24
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    L = int(max(0, min(cx - side / 2, im.width - side)))
    T = int(max(0, min(cy - side / 2, im.height - side)))
    crop = im.crop((L, T, min(im.width, L + side), min(im.height, T + side)))
    canvas = Image.new("RGB", (side, side), (240, 240, 240))
    canvas.paste(crop, (0, 0))
    return canvas.resize((size, size), Image.LANCZOS)


# ----------------------------------------------------------------------------
# S line (plan_synth): draw JA text into a generated scene's bubble


def fit_text(
    d,
    text: str,
    font_path: str,
    region,
    vertical: bool,
    min_glyph: int,
    fill_frac: float = 0.9,
):
    """Largest font size whose text block fits ``region`` (inner
    ``fill_frac`` — 0.9 fills the bubble edge to edge, ``--scene_fill`` 0.7
    leaves manga-like air around the glyphs); the glyph cell must be at
    least ``min_glyph`` px, else ``None``. Returns ``(font, fs, tw, th)``."""
    from PIL import ImageFont

    rx0, ry0, rx1, ry1 = region
    rw, rh = (rx1 - rx0) * fill_frac, (ry1 - ry0) * fill_frac
    n = len(text)
    if vertical:
        fs = int(min(rw, rh / (n * 1.05)))
    else:
        fs = int(min(rh, rw / n))
    if fs < min_glyph:
        return None
    for _ in range(4):
        font = ImageFont.truetype(font_path, fs, index=0)
        if vertical:
            tw = max(d.textlength(ch, font=font) for ch in text)
            th = n * fs * 1.05
        else:
            tw = d.textlength(text, font=font)
            th = fs
        k = min(rw / max(tw, 1e-6), rh / max(th, 1e-6))
        if k >= 1.0:
            return font, fs, tw, th
        fs = int(fs * min(k, 0.97))
        if fs < min_glyph:
            return None
    return None


def region_capacity(region, min_glyph: int, fill_frac: float = 0.9) -> int:
    """How many glyphs the region holds at ``min_glyph`` px per cell along
    its long side (vertical when taller than wide), inner ``fill_frac``."""
    rw, rh = (region[2] - region[0]) * fill_frac, (region[3] - region[1]) * fill_frac
    if rh > rw:
        return int(rh / (min_glyph * 1.05)) if rw >= min_glyph else 0
    return int(rw / min_glyph) if rh >= min_glyph else 0


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
):
    """Erase every anchor bubble's usable region (plus the text box padded by
    a quarter of its size — detector boxes run tight) with the bubble's
    ring-median colour, only *inside the bubble interior* (flood mask,
    letter holes filled — a rectangle's corners would poke past a round
    outline), and draw ``text``
    fitted into the inner ``fill_frac`` of the headline region — vertical
    when the region is taller than wide (the base draws tall manga
    bubbles). Returns ``(image, drawn
    text box)`` or ``None`` when the text does not fit at ``min_glyph`` px
    per glyph (the caller draws a shorter text). Other anchor bubbles are
    left erased (empty bubble). ``stroke``: a thin outline in the fill colour
    around the glyphs (manga lettering over art)."""
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
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    region = scene["region"]
    rw, rh = region[2] - region[0], region[3] - region[1]
    vertical = len(text) > 1 and rh > rw
    fit = fit_text(d, text, font_path, region, vertical, min_glyph, fill_frac)
    if fit is None:
        return None
    font, fs, tw, th = fit
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
    # 30 % of composites, ±7°, about the text block's centre
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    if vertical:
        y = cy - th / 2
        for ch in text:
            w = ld.textlength(ch, font=font)
            ld.text((cx - w / 2, y), ch, fill=color, font=font, **kw)
            y += fs * 1.05
    else:
        ld.text(
            (cx - tw / 2, cy - fs / 2 - fs * 0.1), text, fill=color, font=font, **kw
        )
    if rng.random() < tilt_frac:
        layer = layer.rotate(
            rng.uniform(-tilt_deg, tilt_deg), resample=Image.BICUBIC, center=(cx, cy)
        )
    im.paste(layer, (0, 0), layer)
    bb = layer.getbbox()  # alpha bbox: the drawn (and tilted) glyphs
    if bb is None:
        return None
    box = [
        int(max(0, bb[0] - 2)),
        int(max(0, bb[1] - 2)),
        int(min(W, bb[2] + 2)),
        int(min(H, bb[3] + 2)),
    ]
    return im, box
