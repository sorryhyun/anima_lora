"""Font renders of kana strings.

Bit-identity contract: a layout draws its random choices in the pre-W2a
order, so an unbalanced ``--layout v1`` data dir rebuilds identically.
"""

from __future__ import annotations

import random
from glob import glob

from ..paths import FONT_DIR
from ..shapes import wh


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
    fit_text: str | None = None,
):
    """``layout`` (from ``sample_layout``) pins canvas/bubble/size/position so
    several strings render in the same layout; ``None`` draws a fresh one
    (``bubble_frac`` = share drawn inside a bubble). Returns ``(image,
    drew_bubble)``.

    ``fit_text`` (ΔFM flat sibling, plan_synth2): the block extent — the
    bubble fit, the jitter bubble's size and the block centre — is the max
    over ``text`` and ``fit_text``, so the two strings rendered with each
    other as ``fit_text`` under one ``layout`` share every pixel but the
    glyphs."""
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
        both = [text] + ([fit_text] if fit_text else [])
        if lay["vertical"]:
            tw = max(d.textlength(ch, font=font) for t in both for ch in t)
            th = max(len(t) for t in both) * fs * 1.05
        else:
            tw = max(d.textlength(t, font=font) for t in both)
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
        own = d.textlength(text, font=font)
        d.text(
            (cx - own / 2, cy - fs / 2 - fs * 0.1),
            text,
            fill=color,
            font=font,
            **stroke,
        )
    if lay["rot"] is not None:
        im = im.rotate(lay["rot"], fillcolor=bg, resample=Image.BICUBIC)
    return im, bubble
