"""The item generators. A recipe draws one item from the pools — glyph(s),
image, caption, ink box(es) — and returns an ``Item``, or ``None`` on a
render miss. Bands are the table's: a recipe knows nothing of σ.

    bubble1   one glyph in a scene bubble: ``fit`` = the share of the
              bubble's one-glyph fit it fills; or ``glyph_px`` (font px)
              in a bubble whose one-glyph fit it fills ``fill`` [lo, hi] of
    bubbleN   a window in a scene bubble, unspaced, routed per glyph at
              encode; drawn glyph-first (exposure per row). A column is
              lettered as Japanese: the turned marks on its axis, small kana
              as the font's vertical forms. ``fill`` [lo, hi] of the
              bubble fit; or ``glyph_px`` with ``fill_min`` (the text's
              length over the bubble's) and ``cross_min`` (its font px over
              the bubble's width across it)
    sent      a dialogue line in 2–3 columns of a scene bubble at
              ``glyph_px`` (``pools.sentences``; ``lengths`` [lo, hi] cells)
    grid      one glyph per cell at ``glyph_px``; ``1x1`` = the lone glyph.
              The plain caption: no language, one ``GRID_CLAUSES`` wording
              per item
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .pools import Pools, lang_caption
from . import table as T

MIN_FIT_SCENES = (
    10  # a text goes to the scenes that hold it in one column when this many do
)
SCENE_TRIES = 8


@dataclass
class Item:
    image: object
    vocabs: list
    layout: str  # scene | flat | grid
    boxes: list  # ink boxes: one per cell, or one for a scene
    caption: str
    src: str  # scene | font | grid
    shape: tuple
    extra: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(self.vocabs)

    def px(self) -> float:
        """√(box area / glyphs)."""
        from common.render.ink import box_area, glyph_count

        area = sum(box_area(b) for b in self.boxes)
        return (area / max(1, sum(glyph_count(u) for u in self.vocabs))) ** 0.5


# ----------------------------------------------------------------------------
# scenes


def _fit_px(region, n_glyphs: int, vertical: bool) -> float:
    """The font px ``fit_text`` gives ``n_glyphs`` in one column (a line when
    horizontal) of ``region`` at fill 1: the bubble's capacity in px."""
    from common.render.scene import H_PITCH, V_PITCH

    rw, rh = region[2] - region[0], region[3] - region[1]
    if vertical:
        return min(rw, rh / (n_glyphs * V_PITCH))
    return min(rh, rw / (n_glyphs * H_PITCH))


def _opted(pools: Pools, p: dict) -> set:
    return set().union(*(pools.opt_in[t] for t in p.get("scene_pools", ())))


def _weights(pools: Pools, cands: list) -> list:
    """``1 / (1 + uses)``, rescaled so the mono scenes carry ``MONO_SHARE``."""
    ws = [1.0 / (1 + pools.used[x]) for x in cands]
    m = sum(w for x, w in zip(cands, ws) if x in pools.mono)
    c = sum(ws) - m
    if not m or not c:
        return ws
    s = T.MONO_SHARE
    return [w * (s / m if x in pools.mono else (1 - s) / c) for x, w in zip(cands, ws)]


def _draw_scene(
    pools: Pools,
    rng: random.Random,
    text: str,
    *,
    min_glyph: int,
    fill: float,
    cands: list,
    target_px: float | None = None,
    fill_min: float = 0.0,
    cross_min: float = 0.0,
    lettering: bool = False,
):
    """Text first, then a scene of ``cands`` whose bubble holds it in one
    column (one line when drawn horizontal), weighted by use and colour.
    With a ``target_px``, the fill is set per scene to land it, and
    ``fill_min`` / ``cross_min`` keep only the bubbles it fills along /
    across. ``HORIZONTAL_FRAC`` of the multi-glyph texts are drawn as a
    left-to-right line on the ``HORIZONTAL_SCENES`` only (marked in the
    caption)."""
    from common.render.flat import pick_font
    from common.render.scene import region_capacity, render_into_scene
    from data.synth import scene_caption

    n = len(text)
    horiz = n > 1 and rng.random() < T.HORIZONTAL_FRAC
    if horiz:
        cands = sorted(pools.horiz_idx)
    cap = pools.scene_cap
    fitting = [
        j
        for j in cands
        if (cap is None or pools.tier_used[j] < cap)
        and region_capacity(
            pools.scenes[j]["region"],
            min_glyph,
            fill,
            1,
            not horiz,
            horizontal_only=horiz,
        )
        >= n
    ]
    if target_px is not None:

        def keeps(r) -> bool:
            along = target_px / max(_fit_px(r, n, not horiz), 1e-6)
            across = target_px / max(r[2] - r[0] if not horiz else r[3] - r[1], 1)
            return along >= fill_min and across >= cross_min

        fitting = [j for j in fitting if keeps(pools.scenes[j]["region"])]
    if not fitting:
        return None
    cuts = None
    if n > 1:
        from data.inventory import pieces as qpieces

        cuts, off = [], 0
        for p, _row in qpieces(*pools.tokq, text):
            off += len(p)
            cuts.append(off)
    pool, tries = list(fitting), []
    while pool and len(tries) < SCENE_TRIES:
        j = rng.choices(pool, weights=_weights(pools, pool))[0]
        pool.remove(j)
        tries.append(j)
    for j in tries:
        sc = pools.scenes[j]
        f = fill
        if target_px is not None:
            best = _fit_px(sc["region"], n, not horiz)
            f = max(0.15, min(fill, target_px / max(best, 1e-6)))
        drawn = render_into_scene(
            scene=sc,
            text=text,
            font_path=pick_font(text, pools.fonts, rng),
            rng=rng,
            min_glyph=min_glyph,
            stroke=False,
            fill_frac=f,
            max_lines=1,
            cuts=cuts,
            vertical_only=not horiz,
            horizontal=horiz,
            tategaki=lettering,
            vert_forms=lettering,
            keep_outline=True,
        )
        if drawn is None:
            continue
        im, box = drawn
        assert list(im.size) == list(sc["shape"]), (sc["i"], im.size, sc["shape"])
        pools.used[j] += 1
        pools.tier_used[j] += 1
        return Item(
            image=im,
            vocabs=[text],
            layout="scene",
            boxes=[box],
            caption=lang_caption(
                pools, scene_caption(sc, text, horizontal=horiz), text
            ),
            src="scene",
            shape=im.size,
            extra={
                "scene": sc["i"],
                "scene_pool": sc.get("pool"),
                "mono": j in pools.mono,
                "fill": round(f, 3),
                "horizontal": horiz,
            },
        )
    return None


def _target(rng, p: dict):
    px = p.get("glyph_px")
    return rng.uniform(float(px[0]), float(px[1])) if px else None


def _lone(pools: Pools) -> list:
    """The rows drawn alone: ``pools.lone`` (a mark whose lone spelling misses
    its row is left out), else every single."""
    return pools.singles if pools.lone is None else pools.lone


def bubble1(pools: Pools, rng: random.Random, p: dict):
    glyph = rng.choice(_lone(pools))
    target = _target(rng, p)
    if target is None:
        cands, fill, lo = sorted(pools.single_idx), float(p["fit"]), 0.0
    else:
        lo, fill = (float(x) for x in p["fill"])

        def ok(r) -> bool:
            w, h = r[2] - r[0], r[3] - r[1]
            share = target / max(_fit_px(r, 1, True), 1e-6)
            return (
                max(w, h) <= T.SINGLE_MAX_AR * max(1, min(w, h)) and lo <= share <= fill
            )

        cands = sorted(
            j
            for j in pools.single_idx | _opted(pools, p)
            if ok(pools.scenes[j]["region"])
        )
    return _draw_scene(
        pools,
        rng,
        glyph,
        min_glyph=int(p.get("min_glyph", 28)),
        fill=fill,
        cands=cands,
        target_px=target,
        fill_min=lo,
    )


def bubbleN(pools: Pools, rng: random.Random, p: dict):
    glyph = rng.choice(list(pools.windows))
    lengths = p.get("lengths")  # window length → share (else the pool's own mix)
    if lengths:
        by = pools.windows_len[glyph]
        ks = [k for k in lengths if k in by]
        word = rng.choice(by[rng.choices(ks, weights=[lengths[k] for k in ks])[0]])
    else:
        word = rng.choice(pools.windows[glyph])
    f = p.get("fill", [0.7, 1.0])
    lo, hi = f if isinstance(f, list) else (f, f)
    held = set().union(*pools.opt_in.values())
    cands = [j for j in range(len(pools.scenes)) if j not in held]
    cands += sorted(_opted(pools, p))
    item = _draw_scene(
        pools,
        rng,
        word,
        min_glyph=int(p.get("min_glyph", 28)),
        fill=rng.uniform(float(lo), float(hi)),
        cands=cands,
        target_px=_target(rng, p),
        fill_min=float(p.get("fill_min", 0)),
        cross_min=float(p.get("cross_min", 0)),
        lettering=True,
    )
    if item is not None:
        assert item.caption.count(f'"{word}"') == 1, item.caption
    return item


# ----------------------------------------------------------------------------
# sentences


def _sent_cuts(pools: Pools, text: str) -> list:
    """Column breaks at Qwen piece boundaries, none opening a column on ``…``
    (a leader opens no column, ``……`` is never split)."""
    from data.inventory import pieces as qpieces

    cuts, off = [], 0
    for p, _row in qpieces(*pools.tokq, text):
        off += len(p)
        cuts.append(off)
    return [c for c in cuts if c < len(text) and text[c] != "…"]


def _sent_plan(region, text: str, cuts: list, target: float, min_glyph: int):
    """``(columns, fill)`` that lands ``text`` at ``target`` font px in
    ``region`` as 2–3 columns no square (``SENT_BLOCK_AR``) — the fewest
    columns ``fit_text(fewest_lines=True)`` will take at that fill, every
    fewer one missing ``min_glyph`` there — or ``None`` (one column holds it,
    or the region is too wide, too small, or too roomy)."""
    from common.render.scene import V_GAP, V_PITCH, split_lines

    rw, rh = region[2] - region[0], region[3] - region[1]
    if rw > T.SENT_REGION_AR * rh:
        return None
    lo, hi = T.SENT_FILL
    fs1 = {}  # columns → font px at fill 1
    for k in (1, 2, 3):
        lines = split_lines(text, k, cuts)
        if lines is None:
            continue
        m = max(len(ln) for ln in lines)
        fs1[k] = (min(rw / (1 + (k - 1) * V_GAP), rh / (m * V_PITCH)), m)
        f = target / max(fs1[k][0], 1e-6)
        if f > hi:
            continue  # this many columns cannot reach the target here
        if k == 1 or f < lo:
            return None
        if any(fs1[j][0] * f >= min_glyph for j in fs1 if j < k):
            return None  # the fit would stop at fewer columns
        if m * V_PITCH / (1 + (k - 1) * V_GAP) < T.SENT_BLOCK_AR:
            return None
        return k, f
    return None


def sent(pools: Pools, rng: random.Random, p: dict):
    """A dialogue line (``pools.sentences``, length uniform over
    ``lengths``) lettered as Japanese in 2–3 columns of one scene bubble at
    ``glyph_px`` font px — with mark rows, a mark drawn first and a line
    holding it (``pools.sent_marks``) — on the scenes ``_sent_plan`` places it in (and
    ``scene_pools``'), none framed as a sign (``SENT_FRAMES_OUT``); routed
    per glyph at encode like the windows."""
    from common.render.flat import pick_font
    from common.render.scene import render_into_scene
    from data.synth import scene_caption

    lo, hi = p["lengths"]
    by = pools.sentences
    if pools.sent_marks:  # a mark first, then a line holding it
        by = pools.sent_marks[rng.choice(sorted(pools.sent_marks))]
    ns = [n for n in range(int(lo), int(hi) + 1) if by.get(n)]
    text = rng.choice(by[rng.choice(ns)])
    target = _target(rng, p)
    min_glyph = int(0.85 * target)
    cuts = _sent_cuts(pools, text)
    held = set().union(*pools.opt_in.values())
    cands = [j for j in range(len(pools.scenes)) if j not in held]
    cands += sorted(_opted(pools, p))
    cands = [j for j in cands if pools.scenes[j].get("frame") not in T.SENT_FRAMES_OUT]
    cap = pools.scene_cap
    plans = {}
    for j in cands:
        if cap is not None and pools.tier_used[j] >= cap:
            continue
        got = _sent_plan(pools.scenes[j]["region"], text, cuts, target, min_glyph)
        if got is not None:
            plans[j] = got
    pool, tries = sorted(plans), []
    while pool and len(tries) < SCENE_TRIES:
        j = rng.choices(pool, weights=_weights(pools, pool))[0]
        pool.remove(j)
        tries.append(j)
    for j in tries:
        sc = pools.scenes[j]
        k, f = plans[j]
        drawn = render_into_scene(
            scene=sc,
            text=text,
            font_path=pick_font(text, pools.fonts, rng),
            rng=rng,
            min_glyph=min_glyph,
            stroke=False,
            fill_frac=f,
            max_lines=k,
            cuts=cuts,
            vertical_only=True,
            fewest_lines=True,
            tategaki=True,
            vert_forms=True,
            keep_outline=True,
        )
        if drawn is None:
            continue
        im, box = drawn
        assert list(im.size) == list(sc["shape"]), (sc["i"], im.size, sc["shape"])
        pools.used[j] += 1
        pools.tier_used[j] += 1
        caption = lang_caption(pools, scene_caption(sc, text), text)
        assert caption.count(f'"{text}"') == 1, caption
        return Item(
            image=im,
            vocabs=[text],
            layout="scene",
            boxes=[box],
            caption=caption,
            src="scene",
            shape=im.size,
            extra={
                "scene": sc["i"],
                "scene_pool": sc.get("pool"),
                "mono": j in pools.mono,
                "fill": round(f, 3),
                "glyph_px": round(target, 1),
                "columns": k,
                "horizontal": False,
            },
        )
    return None


# ----------------------------------------------------------------------------
# grids

GRIDS = {  # cols, rows, canvas (None: a lone canvas from SHAPES)
    "1x1": (1, 1, None),
    "2x2": (2, 2, (512, 512)),
    "3x3": (3, 3, (512, 512)),
    "2x3": (2, 3, (416, 624)),
    "3x2": (3, 2, (624, 416)),
}


def parse_grids(spec: str) -> list:
    out = []
    for tok in spec.split(","):
        name, _, w = tok.strip().partition(":")
        assert name in GRIDS, f"grid {name}: one of {', '.join(GRIDS)}"
        out.append((name, float(w) if w else 1.0))
    return out


def grid(pools: Pools, rng: random.Random, p: dict):
    from common.prompts import GRID_CLAUSES, GRID_CLAUSES_BUBBLE, grid_caption
    from data.grid import _Deck, render_grid

    deck = pools.decks.get("singles")
    if deck is None:
        deck = pools.decks["singles"] = _Deck(_lone(pools), rng)
    grids = [
        (g, w)
        for g, w in parse_grids(p["grids"])
        if GRIDS[g][0] * GRIDS[g][1] <= len(_lone(pools))
    ]
    assert grids, "grid: no grid the rows can fill"
    name = rng.choices([g for g, _ in grids], weights=[w for _, w in grids])[0]
    cols, rows, size = GRIDS[name]
    got = deck.deal(cols * rows)
    bubble = rng.random() < float(p.get("bubble_frac", 0.5))
    size = size or pools.shapes.draw() or (512, 512)
    cell = min(size[0] / cols, size[1] / rows)
    fill = _target(rng, p) / cell  # render_grid starts at fill × cell and only shrinks
    kw = {}
    if p.get("bubble_fit"):
        kw["bubble_fit"] = tuple(p["bubble_fit"])
    if p.get("cell_jitter") is not None:
        kw["cell_jitter"] = float(p["cell_jitter"])
    bgs = []
    im, boxes = render_grid(
        got, cols, rows, size, pools.fonts, rng, bubble, (fill, fill), bgs=bgs, **kw
    )
    # a flat canvas not full white is captioned `simple background` alone
    frame = "bubble" if bubble else "flat" if bgs[0] == "white" else "tint"
    clause = rng.choice(
        sorted(c for c in GRID_CLAUSES if bubble or c not in GRID_CLAUSES_BUBBLE)
    )
    lone = cols * rows == 1
    return Item(
        image=im,
        vocabs=list(got),
        layout="flat" if lone else "grid",
        boxes=boxes,
        caption=grid_caption(frame, cols, rows, got, clause=clause),
        src="font" if lone else "grid",
        shape=tuple(size),
        extra={
            "grid": name,
            "bubble": bubble,
            "bg": bgs[0] if isinstance(bgs[0], str) else list(bgs[0]),
            "fill": round(fill, 3),
            "clause": clause,
        },
    )


RECIPES = {"bubble1": bubble1, "bubbleN": bubbleN, "sent": sent, "grid": grid}
