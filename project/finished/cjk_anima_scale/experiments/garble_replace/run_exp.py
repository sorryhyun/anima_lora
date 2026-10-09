#!/usr/bin/env python
"""garble_replace — the base's own garble, only the string replaced (plan_garble_replace.md)

Canvases: the ``scenes_garble`` pool (frames ``ja_garble_saying`` /
``ja_garble_tag``, 1–2 text regions, judged as the JA frames). Per kept
region the garble's ink sets the layout — orientation (taller than wide →
columns), line count (ink runs across the box) and glyph px (the ink span
across ÷ the renderer's column / line pitch) — and a real dialogue line is
drawn into it at that px, after the stage's own erase (``erase_paint``,
ring-median, inside the bubble interior). No bubble fit, no tilt.

Lines: ``polish_seed``'s pool (``polish_b1.line_pool``: manga109s dialogue,
normalized, every glyph a seed single, no doubled glyph, no trigram of a
held word, routed) with ``LINE_LEN`` widened to 2–32 so short and long
garble regions take a line; a region no pool line fits takes two joined
(``、`` between them unless the first ends a sentence). The line is laid on
the garble's own columns: split in proportion to their lengths, column i
centred on the garble's column i and spread from its first ink to its last
(glyph pitch ``PITCH_MIN``–``PITCH_MAX`` × px). Each canvas
is drawn ``VARIANTS`` times with different lines (same canvas, same layout,
the string alone changes); lines are dealt greedily so each of the 57
trained singles appears.

Caption: the canvas prompt with every region's line quoted, right-to-left
(manga order), in one clause — ``She is saying "X" "Y".`` in place of the
saying frame's ``… is saying something.``, else `` Japanese text reads as
"X" "Y".`` (``caption()``). Band 0.6–0.85 on every item. The loss boxes
are the drawn glyph boxes (``layout: grid`` + ``boxes``: ``loss.item_boxes``
masks each region; ``box`` is their union, for ``BoxSplit`` logging only).

Short variants (``--short_variants K``): the last K of a canvas's
``VARIANTS`` draws take a 2–4-glyph pool line (``SHORT_LEN``) instead,
at the garble's px, one column centred on the garble's ink (``short_m``);
the rest of the erased region stays blank. Same canvas, px, fonts, band
and caption form — only the length moves. The short lines leave out any
line that is a piece of a held string or carries a ≤ 2-glyph held string
(はい: the trigram hold does not reach it).

Legs:
- ``data`` (CPU) → ``OUT/run0930_garble_replace/data`` + ``sheet_*.png``;
- ``mix --tag T`` (CPU) → ``data_T``: ``data_<--mix_from>``'s items plus
  3×3 ``grid_single`` items over the same singles (``--grid_frac`` of the
  items) at ``--band``, the source items at their own;
- ``no_humans --tag T`` (CPU) → ``data_T``: ``data_<--mix_from>``'s items
  and latents as they are, ``no humans`` added to every 3×3 grid caption
  that lacks it (``no_humans()``);
- ``reband --tag T --band LO HI`` (CPU) → ``data_T``: ``data_<--mix_from>``'s
  items and latents as they are, every item's σ band set to ``--band``
  (``reband()``);
- ``read --arm warm|cold`` (GPU): the ``sent`` ruler vs the seed's routed
  floor (``read``'s docstring);
- ``train --arm warm|cold`` (GPU) → ``OUT/experiments/garble_replace_<arm>``
  (``--tag T``: data in ``data_T``, arm ``garble_replace_<arm>_T``):
  the 57 singles of the ``sent`` strings, 90 steps / row, every other row
  frozen at the seed; warm starts from the seed rows (μ ``--mu``, default
  0.1 as polish_seed), cold from the pack rows (μ 0, the retrain's singles).
  ``--inv_freq``: each single's update × min(1, median / the items carrying
  it) (``inv_freq()``, ``train.train(row_step_scale=…)``).

    ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack \\
      .venv/bin/python project/cjk_anima_scale/experiments/garble_replace/run_exp.py \\
      --label g0 --legs data
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path

os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"
os.environ.setdefault("ANIMA_VOCAB_PACK", "models/vocab_packs/anima_cjk_vocab_pack")

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, SEED_ROWS, bootstrap, load_experiment  # noqa: E402

bootstrap()
from bench._common import make_run_dir, write_result  # noqa: E402

NAME = "garble_replace"
POOL = OUT / "scenes_garble"
DATA = OUT / "run0930_garble_replace" / "data"
EXP = OUT / "experiments"
BAND = (0.6, 0.85)
VARIANTS = 8  # lines per canvas
LINE_LEN = (2, 32)
SHORT_LEN = (2, 4)  # --short_variants: the short lines' glyph counts
PITCH_MIN, PITCH_MAX = 0.95, 1.25  # glyph pitch along a column, × px (renderer: 1.05)
MIN_FITS = 20  # fewer single lines fit a region → add joined pairs
JOIN_BARE = set("。！？…!?")  # a line ending in these joins the next without 、
FONT_MIN_INK = 0.14  # heavy_fonts: a face's ink coverage at 14 px
STEPS_PER_ROW = 90  # cjk_scale/train.py STEPS_PER_VOCAB (plan § 4: 5 130 steps)
MU_WARM = 0.1  # polish_seed's anchor; cold keeps the trainer's μ 0
INK_TOL = 80  # grey distance from the ring fill that counts as garble ink
MIN_RUN = 3  # px: an ink run across the box narrower than this is a speck
ACROSS_REL = 0.15  # an ink column / line below this share of the densest is a gutter


def held_and_singles() -> tuple[tuple, list]:
    """The ``sent`` strings (``ACCEPT_READ`` + the seed chain's ``read``) and
    their 57 singles — the rows this experiment trains."""
    from cjk_scale.eval import ACCEPT_READ

    S = load_experiment("polish_seed")
    rc = S.rc_of(NAME)
    held = tuple(dict.fromkeys((*ACCEPT_READ, *rc.read)))
    sing = sorted(
        {c for h in held for c in h if unicodedata.category(c) in ("Lo", "Lm")}
    )
    assert len(sing) == 57, len(sing)
    return held, sing, S


def measure(arr, tb) -> dict | None:
    """The garble's layout inside text box ``tb``: ``vertical`` (taller than
    wide), ``px`` (the ink span across ÷ the renderer's pitch for ``k``
    lines) and ``cols`` — one per ink run across the box, in reading order
    (columns right → left, lines top → bottom), each ``{c, a0, a1}``: its
    centre across and its ink extent along, in image px."""
    import numpy as np

    from common.bubble import ring_median
    from common.render.scene import H_GAP, V_GAP

    x0, y0, x1, y1 = (int(v) for v in tb)
    crop = arr[y0:y1, x0:x1].astype(np.int32)
    fill = np.array(ring_median(arr, tb), dtype=np.int32)
    ink = np.abs(crop - fill).max(axis=2) > INK_TOL
    vertical = (y1 - y0) > (x1 - x0)
    across = ink.sum(axis=0) if vertical else ink.sum(axis=1)  # per x / per y
    # relative to the densest line: the gutter between two close columns
    # still carries a few px of ink (punctuation, anti-aliasing)
    on = across >= max(2, ACROSS_REL * across.max())
    runs, s = [], None
    for i, v in enumerate(list(on) + [False]):
        if v and s is None:
            s = i
        elif not v and s is not None:
            runs.append((s, i))
            s = None
    # a 1-px gap inside a glyph is not a line break
    merged: list = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= 1:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(r)
    runs = [r for r in merged if r[1] - r[0] >= MIN_RUN]
    if not runs:
        return None
    k = len(runs)
    span = runs[-1][1] - runs[0][0]
    px = span / (1 + (k - 1) * (V_GAP if vertical else H_GAP))
    if px < 8:
        return None
    cols = []
    for a, b in runs:
        m = (ink[:, a:b] if vertical else ink[a:b, :]).any(axis=1 if vertical else 0)
        nz = np.flatnonzero(m)
        if not len(nz):
            continue
        off_c, off_a = (x0, y0) if vertical else (y0, x0)
        cols.append(
            {
                "c": off_c + (a + b) / 2,
                "a0": off_a + int(nz[0]),
                "a1": off_a + int(nz[-1]) + 1,
            }
        )
    if vertical:
        cols.reverse()  # right to left
    return {
        "vertical": bool(vertical),
        "k": len(cols),
        "px": round(float(px), 1),
        "cols": cols,
    }


def capacity(m: dict) -> tuple[int, int]:
    """Glyph counts the garble's columns hold at the pitch window."""
    fs = round(m["px"])
    L = [c["a1"] - c["a0"] for c in m["cols"]]
    lo = sum(max(1, math.ceil(x / (PITCH_MAX * fs))) for x in L)
    hi = sum(max(1, math.floor(x / (PITCH_MIN * fs))) for x in L)
    return max(2, lo), max(2, hi)


def split_to(text: str, m: dict) -> list[str] | None:
    """``text`` cut into the garble's columns in proportion to their lengths
    (largest remainder, each ≥ 1), each cut nudged forward past a kinsoku
    violation; ``None`` when a column would run denser than ``PITCH_MIN``."""
    from common.render.scene import NO_HEAD, NO_TAIL

    fs = round(m["px"])
    L = [c["a1"] - c["a0"] for c in m["cols"]]
    n, k = len(text), len(L)
    if n < k:
        return None
    raw = [n * x / sum(L) for x in L]
    cnt = [max(1, int(r)) for r in raw]
    while sum(cnt) > n:
        j = max(range(k), key=lambda i: cnt[i] - raw[i])
        cnt[j] -= 1
    while sum(cnt) < n:
        j = max(range(k), key=lambda i: raw[i] - cnt[i])
        cnt[j] += 1
    cuts, acc = [], 0
    for c in cnt[:-1]:
        acc += c
        while acc < n - 1 and (text[acc] in NO_HEAD or text[acc - 1] in NO_TAIL):
            acc += 1
        cuts.append(acc)
    bounds = [0, *cuts, n]
    lines = [text[a:b] for a, b in zip(bounds, bounds[1:])]
    if not all(lines):
        return None
    for ln, x in zip(lines, L):
        if x / len(ln) < PITCH_MIN * fs * 0.9:
            return None
    return lines


def draw_line(im, text: str, m: dict, font_path: str, color):
    """``text`` laid on the garble's own columns: column i centred where the
    garble's column i was, its glyphs spread from the garble column's first
    ink to its last (pitch capped at ``PITCH_MAX``; a shorter column starts
    at the garble's top / left). Returns the drawn glyph box, or ``None``."""
    from PIL import Image, ImageDraw, ImageFont

    from common.render.scene import _draw_vertical_glyph

    lines = split_to(text, m)
    if lines is None:
        return None
    fs = max(8, int(round(m["px"])))
    font = ImageFont.truetype(font_path, fs, index=0)
    W, H = im.size
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    for ln, col in zip(lines, m["cols"]):
        p = min((col["a1"] - col["a0"]) / len(ln), PITCH_MAX * fs)
        for j, ch in enumerate(ln):
            cell = col["a0"] + j * p + (p - fs) / 2  # cell top / left
            if m["vertical"]:
                _draw_vertical_glyph(layer, ld, ch, col["c"], cell, fs, font, color, {})
            else:
                w = ld.textlength(ch, font=font)
                ld.text(
                    (cell + (fs - w) / 2, col["c"] - fs * 0.6),
                    ch,
                    fill=color,
                    font=font,
                )
    bb = layer.getbbox()
    if bb is None:
        return None
    im.paste(layer, (0, 0), layer)
    return [max(0, bb[0] - 2), max(0, bb[1] - 2), min(W, bb[2] + 2), min(H, bb[3] + 2)]


def short_m(m: dict, n: int) -> dict | None:
    """One column for an ``n``-glyph line, centred on the garble's ink (across:
    the mid of its first and last column; along: the mid of its ink extent),
    at the garble's px and the renderer's 1.05 pitch; ``None`` when the line
    runs past the garble's ink along."""
    fs = round(m["px"])
    cs = [c["c"] for c in m["cols"]]
    a0 = min(c["a0"] for c in m["cols"])
    a1 = max(c["a1"] for c in m["cols"])
    L = n * 1.05 * fs
    if L > a1 - a0 + fs:
        return None
    mid = (a0 + a1) / 2
    col = {"c": (min(cs) + max(cs)) / 2, "a0": mid - L / 2, "a1": mid + L / 2}
    return {"vertical": m["vertical"], "k": 1, "px": m["px"], "cols": [col]}


def short_lines(lines: list, held) -> list:
    """The pool's ``SHORT_LEN`` lines minus any piece of a held string and any
    line carrying a ≤ 2-glyph held string."""
    tiny = [h for h in held if len(h) <= 2]
    return [
        ln
        for ln in lines
        if SHORT_LEN[0] <= len(ln) <= SHORT_LEN[1]
        and not any(ln in h for h in held)
        and not any(t in ln for t in tiny)
    ]


def erase(arr, scene: dict):
    """The stage's erase on every text region (``render_into_scene``'s loop);
    ``None`` when a bubble's flood fails."""
    from common.bubble import ring_median
    from common.render.scene import erase_paint

    bubbles = scene.get("bubbles") or [None] * len(scene["regions"])
    fills = []
    for tb, reg, bub in zip(scene["boxes_anchor"], scene["regions"], bubbles):
        fill = ring_median(arr, tb)
        fills.append(fill)
        paint = erase_paint(arr, tb, reg, open_ok=bub is None)
        if paint is None:
            return None
        arr[paint] = fill
    return fills


def ink_color(arr, tb, fill):
    """The garble's stroke colour: the median of its farthest-from-fill 30 %
    of ink pixels (the plain median of thin anti-aliased strokes lands on
    grey), else black / white by the fill."""
    import numpy as np

    x0, y0, x1, y1 = (int(v) for v in tb)
    crop = arr[y0:y1, x0:x1].reshape(-1, 3).astype(np.int32)
    dist = np.abs(crop - np.array(fill, dtype=np.int32)).max(axis=1)
    ink = dist > INK_TOL
    if ink.sum() >= 30:
        core = crop[ink & (dist >= np.percentile(dist[ink], 70))]
        c = np.median(core, axis=0)
        if max(abs(a - b) for a, b in zip(c, fill)) >= 60:
            return tuple(int(v) for v in c)
    return (240, 240, 240) if sum(fill) / 3 < 100 else (0, 0, 0)


def heavy_fonts(fonts: list) -> tuple[list, dict]:
    """The render set without its thin faces: ink coverage of a sample line
    at 14 px (the pool's median garble px) under ``FONT_MIN_INK`` reads pale
    next to the base's strokes (ExtraLight / Light / Regular serif,
    koyomiyuru: 0.09–0.13; the rest ≥ 0.14)."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    t, fs = "せっかく思い出つくりにきたんじゃないか", 14
    cov = {}
    for f in fonts:
        im = Image.new("L", (fs * len(t) + 20, fs * 2), 0)
        ImageDraw.Draw(im).text(
            (5, 5), t, fill=255, font=ImageFont.truetype(f, fs, index=0)
        )
        cov[Path(f).name] = round(
            float(np.asarray(im).sum()) / 255 / (fs * fs * len(t)), 3
        )
    return [f for f in fonts if cov[Path(f).name] >= FONT_MIN_INK], cov


class Dealer:
    """Lines by length; each draw prefers a line carrying the least-drawn of
    the trained singles that some fitting line carries."""

    def __init__(self, lines: list, sing: list, rng: random.Random, held=()):
        self.grams = set()
        for h in held:
            n = min(3, len(h))
            self.grams |= {h[i : i + n] for i in range(len(h) - n + 1)}
        self.lines = lines
        self.by_len: dict = {}
        for ln in lines:
            self.by_len.setdefault(len(ln), []).append(ln)
        self.sing = set(sing)
        self.cov = Counter({c: 0 for c in sing})
        self.used: Counter = Counter()
        self.rng = rng

    def draw(self, lo: int, hi: int, avoid: set) -> str | None:
        fits = [ln for n in range(lo, hi + 1) for ln in self.by_len.get(n, ())]
        fits = [ln for ln in fits if ln not in avoid]
        if len(fits) < MIN_FITS:
            fits += self.joined(lo, hi, MIN_FITS - len(fits))
        if not fits:
            return None
        have = {c for ln in fits for c in ln if c in self.sing}
        for c in sorted(have, key=lambda c: (self.cov[c], c)):
            cand = [ln for ln in fits if c in ln]
            least = min(self.used[ln] for ln in cand)
            cand = [ln for ln in cand if self.used[ln] == least]
            ln = self.rng.choice(cand)
            break
        else:
            ln = self.rng.choice(fits)
        self.used[ln] += 1
        for c in set(ln) & self.sing:
            self.cov[c] += 1
        return ln

    def joined(self, lo: int, hi: int, n: int) -> list:
        """Two pool lines as one bubble (``、`` between them unless the first
        ends a sentence), total length in ``lo``–``hi``; no doubled glyph and
        no held trigram across the join."""
        out, tries = [], 0
        while len(out) < n and tries < 50 * n:
            tries += 1
            a = self.rng.choice(self.lines)
            sep = "" if a[-1] in JOIN_BARE else "、"
            want = [x - len(a) - len(sep) for x in (lo, hi)]
            bs = [
                ln
                for L in range(max(2, want[0]), want[1] + 1)
                for ln in self.by_len.get(L, ())
            ]
            if not bs:
                continue
            t = a + sep + self.rng.choice(bs)
            j = len(a) + len(sep)
            seam = t[max(0, j - 3) : j + 3]
            if any(x == y for x, y in zip(t, t[1:])) or any(
                g in seam for g in self.grams
            ):
                continue
            out.append(t)
        return out


def caption(prompt: str, texts: list) -> str:
    """One clause for all regions, each quoted: a ``japanese text`` prompt
    with the saying frame's ``… is saying something.`` gets the lines as
    the speech (``She is saying "A" "B".``, the ``ja_saying`` form), any
    other gets ``Japanese text reads as "A" "B".``"""
    quoted = " ".join(f'"{t}"' for t in texts)
    head, dot, tail = prompt.partition(". ")
    if "japanese text" in (g.strip() for g in head.split(",")) and tail.endswith(
        " is saying something."
    ):
        return f"{head}{dot}{tail[: -len('something.')]}{quoted}."
    p = prompt if prompt.endswith(".") else prompt + "."
    return f"{p} Japanese text reads as {quoted}."


def build(
    sing: list, lines: list, held=(), seed: int = 0, short_variants: int = 0
) -> tuple[list, dict]:
    import numpy as np
    from PIL import Image

    from common.render.flat import find_fonts, pick_font

    rng = random.Random(seed)
    scenes = [
        json.loads(ln)
        for ln in (POOL / "scenes.jsonl").read_text("utf-8").splitlines()
        if ln
    ]
    fonts, font_ink = heavy_fonts(find_fonts())
    dealer = Dealer(lines, sing, rng, held)
    shorts = short_lines(lines, held)
    sdealer = Dealer(shorts, sing, rng, held)
    (DATA / "img").mkdir(parents=True, exist_ok=True)
    recs, why, regions = [], Counter(), []
    for sc in scenes:
        base = np.array(Image.open(sc["file"]).convert("RGB"))
        ms = [measure(base, tb) for tb in sc["boxes_anchor"]]
        if any(m is None for m in ms):
            why["no_ink"] += 1
            continue
        regions += [{"i": sc["i"], **m} for m in ms]
        arr = base.copy()
        fills = erase(arr, sc)
        if fills is None:
            why["erase"] += 1
            continue
        colors = [ink_color(base, tb, f) for tb, f in zip(sc["boxes_anchor"], fills)]
        # manga order: right to left, then top to bottom
        order = sorted(
            range(len(ms)),
            key=lambda j: (-sc["boxes_anchor"][j][2], sc["boxes_anchor"][j][1]),
        )
        for v in range(VARIANTS):
            short = v >= VARIANTS - short_variants
            im = Image.fromarray(arr.copy())
            texts, boxes, used, faces = [], [], set(), []
            for j in order:
                m = ms[j]
                lo, hi = SHORT_LEN if short else capacity(m)
                box = None
                for _try in range(6):
                    t = (sdealer if short else dealer).draw(lo, hi, used)
                    if t is None:
                        break
                    font = pick_font(t, fonts, rng)
                    mm = short_m(m, len(t)) if short else m
                    if mm is None:
                        continue
                    box = draw_line(im, t, mm, font, colors[j])
                    if box is not None:
                        break
                if box is None:
                    break
                used.add(t)
                texts.append(t)
                faces.append(Path(font).name)
                boxes.append(box)
            if len(texts) < len(ms):
                why["no_line"] += 1
                continue
            f = DATA / "img" / f"garble_{sc['i']:05d}_{v}.png"
            im.save(f)
            cap = caption(sc["prompt"], texts)
            recs.append(
                {
                    "file": str(f),
                    "text": "".join(texts),
                    "texts": texts,
                    "caption": cap,
                    "src": "scene",
                    "kind": "garble_replace",
                    "recipe": "garble_replace",
                    "group": "g0685",
                    # grid: loss.item_boxes masks every region's box
                    "layout": "grid",
                    "boxes": boxes,
                    "box": [
                        min(b[0] for b in boxes),
                        min(b[1] for b in boxes),
                        max(b[2] for b in boxes),
                        max(b[3] for b in boxes),
                    ],
                    "units": sorted(set("".join(texts)) & set(sing)),
                    "shape": sc["shape"],
                    "band": list(BAND),
                    "window": list(BAND),
                    "scene": sc["i"],
                    "scene_pool": "garble",
                    "frame": sc["frame"],
                    "layout_garble": [
                        {
                            **{k: m[k] for k in ("vertical", "k", "px")},
                            "cap": capacity(m),
                        }
                        for m in (ms[j] for j in order)
                    ],
                    "glyphs": len("".join(texts)),
                    "short": short,
                    "fonts": faces,
                }
            )
    stats = {
        "scenes": len(scenes),
        "items": len(recs),
        "dropped": dict(why),
        "regions": len(regions),
        "vertical": sum(r["vertical"] for r in regions),
        "k": dict(sorted(Counter(r["k"] for r in regions).items())),
        "px": sorted(r["px"] for r in regions),
        "cap_hi": dict(sorted(Counter(capacity(r)[1] for r in regions).items())),
        "distinct_lines": len({t for r in recs for t in r["texts"]}),
        "short_variants": short_variants,
        "short_items": sum(r["short"] for r in recs),
        "short_pool": len(shorts),
        "short_distinct": len({t for r in recs if r["short"] for t in r["texts"]}),
        "short_single_draws": dict(sorted(sdealer.cov.items(), key=lambda x: x[1])),
        "font_ink": font_ink,
        "fonts": dict(Counter(f for r in recs for f in r["fonts"])),
        "single_draws": dict(sorted(dealer.cov.items(), key=lambda x: x[1])),
    }
    return recs, stats


def sheet(recs: list, path: Path):
    from PIL import Image, ImageDraw

    from common.readers import contact_sheet

    rows = []
    for r in recs:
        im = Image.open(r["file"]).convert("RGB")
        d = ImageDraw.Draw(im)
        for b in r["boxes"]:
            d.rectangle(b, outline="lime", width=2)
        lay = " / ".join(
            f"{'V' if g['vertical'] else 'H'}{g['k']} {g['cap'][0]}–{g['cap'][1]} {g['px']:.0f}px"
            for g in r["layout_garble"]
        )
        rows.append(
            (im, [f"{r['scene']:05d} {lay}", *(t[:18] for t in r["texts"][:2]), ""])
        )
    contact_sheet(rows, path, thumb=256, cols=8)


def canvases_sheet(path: Path):
    """The garble canvases with their measured layout, one per scene."""
    import numpy as np
    from PIL import Image, ImageDraw

    from common.readers import contact_sheet

    rows = []
    for ln in (POOL / "scenes.jsonl").read_text("utf-8").splitlines():
        sc = json.loads(ln)
        im = Image.open(sc["file"]).convert("RGB")
        arr = np.array(im)
        d = ImageDraw.Draw(im)
        labs = []
        for tb in sc["boxes_anchor"]:
            m = measure(arr, tb)
            d.rectangle(tb, outline="red", width=2)
            labs.append(
                "none"
                if m is None
                else f"{'V' if m['vertical'] else 'H'}{m['k']} {capacity(m)[0]}–{capacity(m)[1]} {m['px']:.0f}px"
            )
        rows.append((im, [f"{sc['i']:05d}", " / ".join(labs), "", ""]))
    contact_sheet(rows, path, thumb=256, cols=8)


def read(arm: str, name: str) -> dict:
    """The ``read`` leg: the arm's rows on the ``sent`` ruler (the retrain_read
    grid: first 4 prompts × 2 seeds × 23 strings, ``en`` clause, 512²,
    routed), paired per render against the seed's routed floor cache —
    official / ≤ 1 edit / dup (``cjk_scale.reads``), sigma_split's placement
    (box, box_h, flat_white) + the native stage's en_cos / box_iou, and one
    sheet per string (EN ref | floor | arm)."""
    import statistics as st

    from cjk_scale import reads as R
    from cjk_scale.eval import TRAINED_ARM, ruler_args
    from stages import run as run_stage

    SS = load_experiment("sigma_split")
    held, _sing, _S = held_and_singles()
    arm_path = EXP / name
    assert (arm_path / "trained.pt").exists(), f"{arm_path}: not trained yet"
    rc = rc_of(name, _sing, held)
    a = ruler_args(rc, TRAINED_ARM, "sent")
    a.arm_path, a.data_path, a.native_limit = str(arm_path), str(DATA), SS.PROMPTS
    run_stage("native", a)
    f = arm_path / "native_sent" / "native_reads.json"
    items = SS.floor_items()
    keys = {(it["text"], it["clause"], it["pi"], it["seed"]) for it in items}
    chars = sorted({it["text"] for it in items})
    floor_h = {
        k: v for k, v in R.hits(SS.FLOOR_READS, chars, SS.CLAUSE).items() if k in keys
    }
    arm_h = {k: v for k, v in R.hits(f, chars, SS.CLAUSE).items() if k in keys}
    print("===== floor", flush=True)
    out: dict = {"floor_tally": R.tally(floor_h)}
    print(f"===== {name}", flush=True)
    out["tally"] = R.tally(arm_h)
    out["vs_floor"] = R.paired(arm_h, floor_h)
    print(f"  vs floor {out['vs_floor']}", flush=True)
    recs = {
        "floor": [
            m
            for m in json.loads(SS.FLOOR_READS.read_text("utf-8"))
            if (m["text"], m["clause"], m["pi"], m["seed"]) in keys
        ],
        arm: [
            m
            for m in json.loads(f.read_text("utf-8"))
            if (m["text"], m["clause"], m["pi"], m["seed"]) in keys
        ],
    }
    place = {}
    for k, ms in recs.items():
        ps = [SS.placement(m) for m in ms]
        place[k] = {
            x: round(st.mean(p[x] for p in ps), 4)
            for x in ("box", "box_h", "flat_white")
        } | {
            x: round(st.mean(m[x] for m in ms if m.get(x) is not None), 4)
            for x in ("en_cos", "en_cos_out", "box_iou")
        }
        print(f"  placement {k:<6} {place[k]}", flush=True)
    out["placement"] = place
    SS.sheets(items, [arm], recs, arm_path / "sheets_sent")
    out["reads"] = str(f)
    return out


def mix(src: Path, sing: list, frac: float, band: tuple, seed: int = 0) -> dict:
    """The ``mix`` leg: ``src``'s items (images reused, captions as built)
    plus 3×3 ``grid_single`` items over the same singles — the b0709 tier's
    params (fill 0.3–0.8 of the cell, half in bubbles, line cells marked) —
    so they are ``frac`` of the items. The grids train at ``band``; ``src``'s
    items keep their own (the control's)."""
    import shutil
    from types import SimpleNamespace

    from cjk_scale.recipes import grid as grid_single  # its name until 2026-10-02

    from common.render.flat import find_fonts

    rng = random.Random(seed)
    recs = [
        json.loads(ln)
        for ln in (src / "train.jsonl").read_text("utf-8").splitlines()
        if ln
    ]
    n = round(frac / (1 - frac) * len(recs))
    pools = SimpleNamespace(
        singles=list(sing),
        decks={},
        fonts=find_fonts(),
        shapes=None,
        horizontal_frac=0.3,
    )
    params = {
        "grids": "3x3",
        "fill": [0.3, 0.8],
        "bubble_frac": 0.5,
        "mark_horizontal": True,
    }
    (DATA / "img").mkdir(parents=True, exist_ok=True)
    grids = []
    for i in range(n):
        item = grid_single(pools, rng, params)
        f = DATA / "img" / f"grid3x3_{i:05d}.png"
        item.image.save(f)
        grids.append(
            {
                "file": str(f),
                "text": item.text,
                "caption": item.caption,
                "src": item.src,
                "kind": "grid_single",
                "recipe": "grid_single",
                "group": "b0709",
                "layout": item.layout,
                "units": item.vocabs,
                "shape": list(item.shape),
                "px": round(item.px(), 1),
                "boxes": item.boxes,
                "band": list(band),
                "window": list(band),
                **item.extra,
            }
        )
    out = recs + grids
    (DATA / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out), encoding="utf-8"
    )
    for f in ("vocabs.json", "eval.json"):
        shutil.copy2(src / f, DATA / f)
    bj = json.loads((src / "build.json").read_text())
    (DATA / "build.json").write_text(
        json.dumps(
            bj | {"grid_band": list(band), "mix_from": str(src), "grid_frac": frac}
        )
    )
    from PIL import Image

    from common.readers import contact_sheet

    contact_sheet(
        [
            (
                Image.open(g["file"]).convert("RGB"),
                ["".join(g["units"]), f"{g['px']:.0f}px", "", ""],
            )
            for g in grids[:16]
        ],
        DATA / "sheet_grids.png",
        thumb=256,
        cols=8,
    )
    units = Counter(u for g in grids for u in g["units"])
    return {
        "from": str(src),
        "items": len(out),
        "garble": len(recs),
        "grid3x3": n,
        "grid_frac": round(n / len(out), 3),
        "grid_units": {
            "min": min(units.values()),
            "max": max(units.values()),
            "n": len(units),
        },
        "px": sorted(g["px"] for g in grids),
    }


def no_humans(src: Path) -> dict:
    """The ``no_humans`` leg: ``src``'s items as they are — images, boxes,
    bands, order, so its latents are copied — with ``no humans`` added to
    every 3×3 grid caption that lacks it: the bubble half
    (``manga, multiple speech bubbles, …``); the flat half already carries it.
    The garble items keep their captions. The TE cache is not copied."""
    import shutil

    recs = [
        json.loads(ln)
        for ln in (src / "train.jsonl").read_text("utf-8").splitlines()
        if ln
    ]
    bubble = "manga, multiple speech bubbles, "
    n = 0
    for r in recs:
        if r["kind"] != "grid_single" or "no humans" in r["caption"]:
            continue
        assert r["caption"].startswith(bubble), r["caption"]
        r["caption"] = (
            "manga, no humans, multiple speech bubbles, " + r["caption"][len(bubble) :]
        )
        n += 1
    (DATA / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8"
    )
    for f in ("vocabs.json", "eval.json"):
        shutil.copy2(src / f, DATA / f)
    for d in src.glob("latents_*"):
        if d.is_dir() and not (DATA / d.name).exists():
            shutil.copytree(d, DATA / d.name)
    bj = json.loads((src / "build.json").read_text())
    (DATA / "build.json").write_text(json.dumps(bj | {"no_humans_from": str(src)}))
    grids = [r for r in recs if r["kind"] == "grid_single"]
    return {
        "from": str(src),
        "items": len(recs),
        "grid3x3": len(grids),
        "captions_changed": n,
        "grid_no_humans": sum("no humans" in r["caption"] for r in grids),
        "garble_no_humans": sum(
            "no humans" in r["caption"] for r in recs if r["kind"] != "grid_single"
        ),
    }


def reband(src: Path, band: tuple) -> dict:
    """The ``reband`` leg: ``src``'s items as they are — images, boxes,
    captions, order, so its latents are copied — every item's σ band set to
    ``band``. The TE cache is not copied."""
    import shutil

    recs = [
        json.loads(ln)
        for ln in (src / "train.jsonl").read_text("utf-8").splitlines()
        if ln
    ]
    old = Counter(tuple(r["band"]) for r in recs)
    for r in recs:
        r["band"] = list(band)
    (DATA / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8"
    )
    for f in ("vocabs.json", "eval.json"):
        shutil.copy2(src / f, DATA / f)
    for d in src.glob("latents_*"):
        if d.is_dir() and not (DATA / d.name).exists():
            shutil.copytree(d, DATA / d.name)
    bj = json.loads((src / "build.json").read_text())
    (DATA / "build.json").write_text(
        json.dumps(bj | {"band": list(band), "reband_from": str(src)})
    )
    return {
        "from": str(src),
        "items": len(recs),
        "old_bands": {f"{a}-{b}": n for (a, b), n in old.items()},
        "band": list(band),
    }


def inv_freq(data: Path) -> dict:
    """Per single: ``min(1, median / n)``, ``n`` = the items carrying it — the
    over-drawn singles (っ な い ん て …) step slower, the rest at the lr."""
    import statistics

    recs = [
        json.loads(ln)
        for ln in (data / "train.jsonl").read_text("utf-8").splitlines()
        if ln
    ]
    n = Counter(u for r in recs for u in r["units"])
    med = statistics.median(n.values())
    return {g: round(min(1.0, med / c), 4) for g, c in sorted(n.items())}


def rc_of(name: str, sing: list, held: tuple):
    from cjk_scale.config import RunConfig

    return RunConfig(
        name=name, path=Path(__file__), vocabs=("chars:" + "".join(sing),), read=held
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument(
        "--legs",
        nargs="+",
        default=["data"],
        choices=["data", "mix", "no_humans", "reband", "train", "read"],
    )
    p.add_argument("--arm", choices=["warm", "cold"])
    p.add_argument("--mu", type=float, default=MU_WARM, help="warm arm's anchor μ")
    p.add_argument(
        "--tag",
        default="",
        help="suffix for the data dir (data_<tag>) and the arm (<arm>_<tag>), "
        "so a variant run leaves the untagged data and rows in place",
    )
    p.add_argument(
        "--data_tag",
        default=None,
        help="the data dir's tag when it differs from --tag ('' = the untagged data)",
    )
    p.add_argument(
        "--short_variants",
        type=int,
        default=0,
        help=f"data: how many of a canvas's {VARIANTS} draws take a {SHORT_LEN} line",
    )
    p.add_argument(
        "--inv_freq",
        action="store_true",
        help="train: each single's update × min(1, median / its item count) (row_step_scale)",
    )
    p.add_argument(
        "--mix_from",
        default="quoted",
        help="mix / no_humans / reband: the source data tag",
    )
    p.add_argument(
        "--grid_frac", type=float, default=0.2, help="mix: 3×3 grid share of the items"
    )
    p.add_argument(
        "--band",
        type=float,
        nargs=2,
        default=list(BAND),
        help="mix: the grids' σ band; reband: every item's",
    )
    args = p.parse_args()
    assert not {"train", "read"} & set(args.legs) or args.arm, (
        "--legs train / read need --arm"
    )
    global DATA
    dtag = args.tag if args.data_tag is None else args.data_tag
    if dtag:
        DATA = DATA.parent / f"data_{dtag}"
    DATA.mkdir(parents=True, exist_ok=True)
    name = f"{NAME}_{args.arm}" + (f"_{args.tag}" if args.tag else "")

    held, sing, S = held_and_singles()
    metrics: dict = {"held": list(held), "singles": "".join(sing), "band": list(BAND)}
    run_dir = make_run_dir(
        NAME, label=args.label, root=LINE / "experiments" / NAME / "results"
    )
    if "data" in args.legs:
        P = S.P
        P.LINE_LEN = LINE_LEN
        lines, why = P.line_pool(held)
        print(f"lines: {len(lines)} ({LINE_LEN}), dropped {why}", flush=True)
        recs, stats = build(sing, lines, held, short_variants=args.short_variants)
        (DATA / "train.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8"
        )
        (DATA / "vocabs.json").write_text(json.dumps(sing, ensure_ascii=False))
        (DATA / "eval.json").write_text(
            json.dumps(
                [
                    {
                        "group": "sent",
                        "text": h,
                        "caption": f'manga, speech bubble, japanese text. Japanese text reads as "{h}".',
                    }
                    for h in held
                ],
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        (DATA / "build.json").write_text(
            json.dumps(
                {
                    "run": NAME,
                    "glyph_route": True,
                    "variants": VARIANTS,
                    "band": list(BAND),
                    "short_variants": args.short_variants,
                    "short_len": list(SHORT_LEN),
                }
            )
        )
        sheet(
            [r for r in recs if r["file"].endswith("_0.png")], DATA / "sheet_items.png"
        )
        sheet(
            [r for r in recs if r["scene"] == recs[0]["scene"]],
            DATA / "sheet_variants.png",
        )
        canvases_sheet(DATA / "sheet_canvases.png")
        if args.short_variants:
            sheet([r for r in recs if r["short"]][:64], DATA / "sheet_short.png")
        metrics["data"] = stats
        print(
            json.dumps(
                {k: v for k, v in stats.items() if k != "px"}, ensure_ascii=False
            ),
            flush=True,
        )
        px = stats["px"]
        print(
            f"px: min {px[0]} p25 {px[len(px) // 4]} median {px[len(px) // 2]} max {px[-1]}",
            flush=True,
        )
    if "mix" in args.legs:
        assert args.tag and args.tag != args.mix_from, "--legs mix needs its own --tag"
        src = DATA.parent / (f"data_{args.mix_from}" if args.mix_from else "data")
        metrics["mix"] = mix(src, sing, args.grid_frac, tuple(args.band))
        metrics["grid_band"] = list(args.band)
        print(
            json.dumps(
                {k: v for k, v in metrics["mix"].items() if k != "px"},
                ensure_ascii=False,
            ),
            flush=True,
        )
    if "no_humans" in args.legs:
        assert args.tag and args.tag != args.mix_from, (
            "--legs no_humans needs its own --tag"
        )
        src = DATA.parent / (f"data_{args.mix_from}" if args.mix_from else "data")
        metrics["no_humans"] = no_humans(src)
        print(json.dumps(metrics["no_humans"], ensure_ascii=False), flush=True)
    if "reband" in args.legs:
        assert args.tag and args.tag != args.mix_from, (
            "--legs reband needs its own --tag"
        )
        src = DATA.parent / (f"data_{args.mix_from}" if args.mix_from else "data")
        metrics["reband"] = reband(src, tuple(args.band))
        metrics["band"] = list(args.band)
        print(json.dumps(metrics["reband"], ensure_ascii=False), flush=True)
    if "train" in args.legs:
        from cjk_scale import train as T

        rc = rc_of(name, sing, held)
        T.INIT_ANCHOR = args.mu if args.arm == "warm" else 0.0
        metrics.update(arm=args.arm, mu=T.INIT_ANCHOR, steps_per_row=STEPS_PER_ROW)
        scale = None
        if args.inv_freq:
            scale = inv_freq(DATA)
            metrics["row_step_scale"] = scale
        T.train(
            rc,
            data=DATA,
            out=EXP / name,
            cold=args.arm == "cold",
            steps_per_row=STEPS_PER_ROW,
            context=SEED_ROWS,
            row_step_scale=scale,
        )
    if "read" in args.legs:
        metrics["read"] = read(args.arm, name)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(DATA)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
