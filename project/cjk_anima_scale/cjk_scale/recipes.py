"""recipes — the item generators (design § 4), over the probe line's render
primitives (``common/render/``) and its inventory / scene / phrase readers.

A recipe draws one item — unit(s), px, layout — and returns an ``Item`` or
``None`` on a render miss. It knows nothing about the stage: the builder
measures the item's px, looks up its window and keeps or re-draws it.

    scene_single    one glyph in a bubble; px = the bubble fit (fill 0.7 → 48–53)
                    or a ``glyph_px`` range that sets the fill per item;
                    ``fill_min`` (any scene recipe with ``glyph_px``) keeps only
                    scenes whose bubble the text fills to that share
    grid_single     1×1 … 3×3 grid, one glyph per cell, one fill draw per item
                    or a ``glyph_px`` range (fill = px / cell, as grid_string);
                    1×1 is the flat single (bare or ellipse, plain template)
    scene_piece     one piece (one token, 2+ glyphs) in a bubble, fill 0.7–1.0
    scene_short     a 2–5-piece corpus line (multi), one column
    scene_sentence  a Manga109-s dialogue line, ``min_glyph`` drawn per item
    grid_string     pieces / short lines in 2×2 … 3×2 word cells at a target px

Records follow the probe's ``train.jsonl`` schema (``file`` / ``text`` /
``caption`` / ``src`` / ``kind`` / ``shape`` / ``box`` or ``boxes`` /
``units``), plus ``recipe`` / ``layout`` / ``px`` / ``window`` (design § 4).
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .paths import UNITS_DIR
from .windows import glyph_count

# scene-fit rules the probe settled (data/synth.py): a text goes to the
# scenes that hold it in one column whenever this many do; tries per text
MIN_FIT_SCENES = 10
SCENE_TRIES = 8


@dataclass
class Item:
    image: object  # PIL image
    units: list
    layout: str  # scene | flat | grid
    boxes: list  # ink boxes, one per unit (scene: one box)
    caption: str
    src: str  # scene | font | grid
    shape: tuple
    extra: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(self.units)

    def px(self) -> float:
        """√(box area / glyphs) — the ink-stat px of ``data/stage.py``."""
        from common.render.ink import box_area

        area = sum(box_area(b) for b in self.boxes)
        return (area / max(1, sum(glyph_count(u) for u in self.units))) ** 0.5


@dataclass
class Pools:
    """Everything the recipes draw from, built once per data dir."""

    fonts: list
    tokq: tuple
    inv: object  # data.units.Inventory
    scenes: list
    single_idx: set  # scenes a lone glyph may go to
    horiz_idx: set  # scenes a left-to-right item may go to
    shapes: object  # data.stage.ShapePool
    singles: list  # weighted pool: one token, one glyph
    pieces: list  # weighted pool: one token, ≥ 2 glyphs (one ext row)
    digraphs: list  # weighted pool: `small` digraphs (host + small row → multi)
    phrase: dict  # kind → training lines (short / sentence); every line is multi
    held: dict  # kind → held lines
    vertical: bool
    stroke: float
    horizontal_frac: float  # share of multi-glyph items / grid cells drawn as lines
    used: Counter = field(default_factory=Counter)  # scene index → items drawn
    decks: dict = field(default_factory=dict)
    balanced: dict = field(default_factory=dict)
    _n_tokens: dict = field(default_factory=dict)

    def n_tokens(self, text: str) -> int:
        """Qwen token count of ``text`` (memoised) — the kind's first axis."""
        from data.inventory import pieces as qpieces

        n = self._n_tokens.get(text)
        if n is None:
            tok, qmap = self.tokq
            n = self._n_tokens[text] = len(qpieces(tok, qmap, text))
        return n


# ----------------------------------------------------------------------------
# pools


def _quietly(fn, *args, width: int = 160):
    """Run a probe resolver with its stdout captured; print each of its
    lines cut to ``width`` (they are provenance, not a unit dump)."""
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*args)
    for ln in buf.getvalue().splitlines():
        if ln.strip():
            print(ln if len(ln) <= width else ln[: width - 1] + "…", flush=True)
    return out


def build_pools(cfg, out: Path, rng: random.Random) -> Pools:
    """Inventory (``--units`` semantics), scenes, phrase kinds, eval groups —
    the probe's resolvers on a small namespace, so a unit means what it
    meant in every read of record."""
    from types import SimpleNamespace

    from data.inventory import pieces as qpieces
    from data.inventory import phrase_file_lines, qwen_pieces
    from data.stage import (
        ShapePool,
        _base_inventory,
        _eval_strings,
        _resolve_singles,
        _word_set,
    )
    from data.synth import _SENT_DISTINCT, _SHORT_DISTINCT, _letters, load_scenes
    from common.render.flat import find_fonts

    d = cfg.data
    units = list(d["units"])
    if d["pieces"]:
        # the inventory keeps ONE `list:` source (Inventory.source), so the
        # file rides on the punctuation list, as the probe's step-2 data stage had it
        p = d["pieces"]
        path = Path(p) if "/" in p else UNITS_DIR / p
        assert path.is_file(), f"pieces file {path}"
        lists = [i for i, u in enumerate(units) if u.startswith("list:")]
        assert len(lists) <= 1, (
            "one list: source in data.units — the pieces file joins it"
        )
        if lists:
            spec = units[lists[0]]
            body, star, weight = spec.partition("*")
            units[lists[0]] = f"{body},@{path}{star}{weight}"
        else:
            units.append(f"list:@{path}*1")
    a = SimpleNamespace(
        units=units,
        balanced=0,
        seed=int(d["seed"]),
        phrase_file="",
        phrase_pieces=0,
        phrase_min_pieces=int(d["phrase_min_pieces"]),
        phrase_max_pieces=int(d["phrase_max_pieces"]),
        phrase_norm=bool(d["phrase_norm"]),
        word_min_len=2,
        n_word_eval=0,
        line_max_len=16,
        n_line_eval=0,
    )
    fonts = find_fonts()
    inv = _base_inventory(a)
    tokq = qwen_pieces()
    _eval_strings(a, rng, inv)
    _quietly(
        _resolve_singles, a, out, tokq, inv
    )  # its `extra units:` line lists every piece
    _quietly(_word_set, a, out, tokq, inv)  # no words source here: `words: 0` noise
    for g in ("combo", "corpus", "line", "word", "word_held"):
        inv.evals.pop(g, None)
    # the pool by kind: single (1 token, 1 glyph), piece (1 token, ≥ 2 glyphs),
    # digraph (the `small` source: host + small row, 2 tokens → multi)
    tok, qmap = tokq
    pool = inv.pool()
    singles, pieces, digraphs = [], [], []
    for u in pool:
        n = len(qpieces(tok, qmap, u))
        if n >= 2:
            digraphs.append(u)
        elif glyph_count(u) == 1:
            singles.append(u)
        else:
            pieces.append(u)
    assert singles, "the unit pool has no one-glyph unit"
    small = {d for ds in inv.small_of.values() for d in ds}
    stray = sorted(set(digraphs) - small)
    assert not stray, f"≥ 2-token units outside the small digraphs: {stray[:10]}"
    scenes = load_scenes(d["scenes"], 0.0, 0, "", d["scene_one_bubble"])
    single_pools = {t for t in d["single_scenes"].split(",") if t}
    max_ar = float(d["single_max_ar"])

    def single_ok(sc) -> bool:
        if single_pools and sc["pool"] not in single_pools:
            return False
        if max_ar > 0:
            w = sc["region"][2] - sc["region"][0]
            h = sc["region"][3] - sc["region"][1]
            return max(w, h) <= max_ar * max(1, min(w, h))
        return True

    single_idx = {j for j, sc in enumerate(scenes) if single_ok(sc)}
    assert single_idx, "single_scenes / single_max_ar leave no scene for a glyph"
    horiz_pools = {t for t in d["horizontal_scenes"].split(",") if t}
    horiz_idx = {
        j for j, sc in enumerate(scenes) if not horiz_pools or sc["pool"] in horiz_pools
    }
    assert horiz_idx or not float(d["horizontal_frac"]), (
        f"horizontal_scenes {sorted(horiz_pools)}: no scene for a horizontal item"
    )

    phrase: dict = {"short": [], "sentence": []}
    held: dict = {"short": [], "sentence": []}
    if d["phrase_file"]:
        tok, qmap = tokq
        plines = phrase_file_lines(
            Path(d["phrase_file"]),
            a.phrase_min_pieces,
            a.phrase_max_pieces,
            norm=a.phrase_norm,
        )
        books = sorted({b for _t, b, _n in plines})
        hrng = random.Random(a.seed + 41)
        held_books = set(
            hrng.sample(books, min(int(d["phrase_held_books"]), len(books)))
        )
        n_of = {
            t: n if n is not None else len(qpieces(tok, qmap, t)) for t, _b, n in plines
        }
        lo, hi = (int(x) for x in str(d.get("short_pieces", "2-5")).split("-"))
        min_letters = int(d.get("sentence_min_letters", 6))

        def kind_of(t):
            ls = _letters(t)
            if len(ls) >= min_letters and len(set(ls)) >= _SENT_DISTINCT:
                return "sentence"
            if lo <= n_of[t] <= hi and len(set(ls)) >= _SHORT_DISTINCT:
                return "short"
            return None

        line_ok = inv.piece_ok
        context = cfg.context_table()
        if context is not None:
            line_ok = _context_line_ok(inv.piece_ok, tokq, context)
        train_set = set()
        for t, b, _n in plines:
            if not line_ok(t):
                continue
            k = kind_of(t)
            if k is None:
                continue
            if b in held_books:
                held[k].append(t)
            else:
                phrase[k].append(t)
                train_set.add(t)
        for k in held:
            held[k] = sorted({t for t in held[k] if t not in train_set})
        for k in phrase:
            phrase[k] = sorted(set(phrase[k]))
        prng = random.Random(a.seed + 37)
        n_ev = int(d["n_phrase_eval"])
        inv.evals["phrase"] = sorted(
            prng.sample(phrase["sentence"], min(n_ev, len(phrase["sentence"])))
        )
        inv.evals["phrase_held"] = held["sentence"][:n_ev]
        inv.evals["short"] = sorted(
            prng.sample(phrase["short"], min(n_ev, len(phrase["short"])))
        )
        inv.evals["short_held"] = held["short"][:n_ev]
        print(
            f"phrases ({d['phrase_file']}): sentence {len(phrase['sentence'])} "
            f"(>= {min_letters} letters), short {len(phrase['short'])} ({lo}-{hi} pieces); "
            f"held: sentence {len(held['sentence'])}, short {len(held['short'])} "
            f"(books {sorted(held_books)})",
            flush=True,
        )
    if pieces:
        # the piece rows' own exact ruler, under the probe's `word` group
        # (single-piece multi-glyph words — what a piece is)
        erng = random.Random(a.seed + 43)
        distinct = list(dict.fromkeys(pieces))
        inv.evals["word"] = sorted(
            erng.sample(distinct, min(int(d["n_piece_eval"]), len(distinct)))
        )
    print(
        f"pools: {len(set(singles))} singles ({len(singles)} weighted), "
        f"{len(set(pieces))} pieces ({len(pieces)} weighted), "
        f"{len(set(digraphs))} digraphs ({len(digraphs)} weighted, multi), "
        f"{len(scenes)} scenes ({len(single_idx)} take a lone glyph)",
        flush=True,
    )
    return Pools(
        fonts=fonts,
        tokq=tokq,
        inv=inv,
        scenes=scenes,
        single_idx=single_idx,
        horiz_idx=horiz_idx,
        shapes=ShapePool(d["shapes"], a.seed),
        singles=singles,
        pieces=pieces,
        digraphs=digraphs,
        phrase=phrase,
        held=held,
        vertical=bool(d["vertical"]),
        stroke=float(d["stroke"]),
        horizontal_frac=float(d["horizontal_frac"]),
    )


def _context_line_ok(piece_ok, tokq, context: Path):
    """``context = "seed"``: a line is drawable when every piece has a row
    that is either an inventory row or a row of the context table (it rides
    frozen at that value), and at least one piece is the inventory's — a
    line of context rows only trains nothing."""
    import torch

    from data.inventory import pieces as qpieces

    ctx = {
        int(e)
        for e in torch.load(context, map_location="cpu", weights_only=False)["delta"][
            "ext_ids"
        ]
    }
    tok, qmap = tokq
    own: dict = {}

    def ok(text: str) -> bool:
        hit = False
        for p, row in qpieces(tok, qmap, text):
            if row is None:
                return False
            mine = own.get(p)
            if mine is None:
                mine = own[p] = piece_ok(p)
            if mine:
                hit = True
            elif row not in ctx:
                return False
        return hit

    print(f"phrase lines: context rows from {context} ({len(ctx)} rows)", flush=True)
    return ok


# ----------------------------------------------------------------------------
# scene recipes


def _cuts(pools: Pools, text: str) -> list:
    """Line cuts at Qwen piece boundaries: a row's unit is never split."""
    from data.inventory import pieces as qpieces

    tok, qmap = pools.tokq
    cuts, off = [], 0
    for p, _row in qpieces(tok, qmap, text):
        off += len(p)
        cuts.append(off)
    return cuts


def _fill_for_px(
    region,
    n_glyphs: int,
    target_px: float,
    vertical: bool,
    fill_max: float,
    max_lines: int = 1,
) -> float:
    """The ``fill_frac`` that makes ``fit_text`` land a text of ``n_glyphs``
    at about ``target_px`` font px in ``region`` — the knob that moves an
    item's px (design § 4). The fill scales the fit-1 px (``_fit_px``)."""
    best = _fit_px(region, n_glyphs, vertical, max_lines)
    return max(0.15, min(fill_max, target_px / max(best, 1e-6)))


def _fit_px(region, n_glyphs: int, vertical: bool, max_lines: int = 1) -> float:
    """The font px ``fit_text`` gives ``n_glyphs`` in ``region`` at fill 1 —
    the largest over 1..``max_lines`` columns (lines when horizontal). The
    bubble's capacity in px; ``target_px / _fit_px`` is the share of the
    bubble the text will take."""
    from common.render.scene import H_GAP, H_PITCH, V_GAP, V_PITCH

    rw, rh = region[2] - region[0], region[3] - region[1]
    best = 0.0
    for k in range(1, max(1, max_lines) + 1):
        m = -(-n_glyphs // k)  # glyphs in the longest column / line
        if vertical:
            fs = min(rw / (1 + (k - 1) * V_GAP), rh / (m * V_PITCH))
        else:
            fs = min(rh / (1 + (k - 1) * H_GAP), rw / (m * H_PITCH))
        best = max(best, fs)
    return best


def _draw_scene(
    pools: Pools,
    rng: random.Random,
    text: str,
    *,
    min_glyph: int,
    fill: float,
    max_lines: int,
    fewest_lines: bool = False,
    singles_only: bool = False,
    target_px: float | None = None,
    fill_max: float = 1.0,
    fill_min: float = 0.0,
):
    """Text first, then a scene whose capacity holds it (one column when
    enough scenes do), weighted ``1 / (1 + uses)`` — the probe's
    ``_quota_composites`` draw. Orientation is drawn per item before the
    scene: ``horizontal_frac`` of multi-glyph items are left-to-right lines
    (marked in the caption) on the ``horizontal_scenes`` pools only, the
    rest columns on any pool; a miss in the drawn
    orientation re-picks the scene, never the orientation. With a
    ``target_px``, ``fill_min`` keeps only the scenes whose bubble the text
    fills to at least that share (``target_px / _fit_px``) — small text
    goes to small bubbles instead of floating in a big one; no such scene
    is a miss (``None``, the recipe re-draws). Returns an ``Item`` or
    ``None``."""
    from common.render.flat import pick_font
    from common.render.scene import region_capacity, render_into_scene
    from data.synth import scene_caption

    n = len(text)
    horiz = n > 1 and rng.random() < pools.horizontal_frac
    vert = pools.vertical and not horiz
    if singles_only:
        cands = list(pools.single_idx)
    elif horiz:
        cands = list(pools.horiz_idx)
    else:
        cands = range(len(pools.scenes))

    def cap(sc, lines):
        return region_capacity(
            sc["region"], min_glyph, fill, lines, vert, horizontal_only=horiz
        )

    one = [j for j in cands if cap(pools.scenes[j], 1) >= n]
    fitting = (
        one
        if len(one) >= MIN_FIT_SCENES or max_lines == 1
        else [j for j in cands if cap(pools.scenes[j], max_lines) >= n]
    )
    if target_px is not None and fill_min > 0:
        fitting = [
            j
            for j in fitting
            if target_px
            / max(_fit_px(pools.scenes[j]["region"], n, vert, max_lines), 1e-6)
            >= fill_min
        ]
    if not fitting:
        return None
    cuts = _cuts(pools, text) if n > 1 else None
    pool, tries = list(fitting), []
    while pool and len(tries) < SCENE_TRIES:
        j = rng.choices(pool, weights=[1.0 / (1 + pools.used[x]) for x in pool])[0]
        pool.remove(j)
        tries.append(j)
    for j in tries:
        sc = pools.scenes[j]
        f = fill
        if target_px is not None:
            f = _fill_for_px(sc["region"], n, target_px, not horiz, fill_max, max_lines)
        drawn = render_into_scene(
            scene=sc,
            text=text,
            font_path=pick_font(text, pools.fonts, rng),
            rng=rng,
            min_glyph=min_glyph,
            stroke=rng.random() < pools.stroke,
            fill_frac=f,
            max_lines=max_lines,
            cuts=cuts,
            vertical_only=vert,
            fewest_lines=fewest_lines,
            horizontal=horiz,
        )
        if drawn is None:
            continue
        im, box = drawn
        W, H = im.size
        assert [W, H] == list(sc["shape"]), (sc["i"], im.size, sc["shape"])
        pools.used[j] += 1
        return Item(
            image=im,
            units=[text],
            layout="scene",
            boxes=[box],
            caption=scene_caption(sc, text, horizontal=horiz),
            src="scene",
            shape=(W, H),
            extra={
                "scene": sc["i"],
                "scene_pool": sc.get("pool"),
                "fill": round(f, 3),
                "horizontal": horiz,
            },
        )
    return None


def _balanced_line(pools: Pools, rng: random.Random, kind: str) -> str | None:
    """A training line of ``kind``, least-drawn first (``--text_draw balanced``)."""
    lines = pools.phrase.get(kind) or []
    if not lines:
        return None
    used = pools.balanced.setdefault(kind, Counter())
    least = min(used[t] for t in lines)
    t = rng.choice([t for t in lines if used[t] == least])
    used[t] += 1
    return t


def scene_single(pools: Pools, rng: random.Random, p: dict):
    # `digraphs = true`: the small-kana digraphs ride along (kind multi — the
    # gate keeps them only where a multi row holds the stage band)
    pool = pools.singles + (pools.digraphs if p.get("digraphs") else [])
    unit = rng.choice(pool)
    px = p.get("glyph_px")
    target = rng.uniform(*px) if px else None
    return _draw_scene(
        pools,
        rng,
        unit,
        min_glyph=int(p.get("min_glyph", 28)),
        fill=float(p.get("fill", 0.7)),
        max_lines=1,
        singles_only=True,
        target_px=target,
        fill_max=float(p.get("fill", 0.7)),
        fill_min=float(p.get("fill_min", 0)),
    )


def _target(rng, p: dict):
    """``glyph_px = [lo, hi]`` → a target px per item (None: the bubble fit)."""
    px = p.get("glyph_px")
    return rng.uniform(float(px[0]), float(px[1])) if px else None


def scene_piece(pools: Pools, rng: random.Random, p: dict):
    assert pools.pieces, "scene_piece needs data.pieces"
    unit = rng.choice(pools.pieces)
    f = p.get("fill", [0.7, 1.0])
    lo, hi = f if isinstance(f, list) else (f, f)
    fill = rng.uniform(float(lo), float(hi))
    return _draw_scene(
        pools,
        rng,
        unit,
        min_glyph=int(p.get("min_glyph", 28)),
        fill=fill,
        max_lines=1,
        target_px=_target(rng, p),
        fill_max=fill,
        fill_min=float(p.get("fill_min", 0)),
    )


def scene_short(pools: Pools, rng: random.Random, p: dict):
    text = _balanced_line(pools, rng, "short")
    if text is None:
        return None
    fill = float(p.get("fill", 0.7))
    return _draw_scene(
        pools,
        rng,
        text,
        min_glyph=int(p.get("min_glyph", 28)),
        fill=fill,
        max_lines=int(p.get("max_lines", 1)),
        target_px=_target(rng, p),
        fill_max=fill,
        fill_min=float(p.get("fill_min", 0)),
    )


def scene_sentence(pools: Pools, rng: random.Random, p: dict):
    """``min_glyph`` floors the glyph; ``glyph_px`` (when given) sets it —
    the fit otherwise grows to the bubble, so a 16 px floor draws 23 px
    text on the median bubble."""
    text = _balanced_line(pools, rng, "sentence")
    if text is None:
        return None
    mg = p.get("min_glyph", [16, 28])
    min_glyph = rng.randint(int(mg[0]), int(mg[1])) if isinstance(mg, list) else int(mg)
    fill = float(p.get("fill", 0.9))
    return _draw_scene(
        pools,
        rng,
        text,
        min_glyph=min_glyph,
        fill=fill,
        max_lines=int(p.get("max_lines", 2)),
        fewest_lines=True,
        target_px=_target(rng, p),
        fill_max=fill,
        fill_min=float(p.get("fill_min", 0)),
    )


# ----------------------------------------------------------------------------
# grid recipes (1×1 = the flat single)

GRIDS = {
    "1x1": (1, 1, None),  # canvas from the shapes pool
    "2x2": (2, 2, (512, 512)),
    "3x3": (3, 3, (512, 512)),
    "2x3": (2, 3, (416, 624)),
    "3x2": (3, 2, (624, 416)),
}


def parse_grids(spec: str) -> list:
    out = []
    for tok in str(spec).split(","):
        if not tok.strip():
            continue
        name, _, w = tok.strip().partition(":")
        assert name in GRIDS, f"grid {name}: one of {', '.join(GRIDS)}"
        out.append((name, float(w) if w else 1.0))
    return out


def _deck(pools: Pools, key: str, pool: list, rng: random.Random):
    from data.grid import _Deck

    if key not in pools.decks:
        pools.decks[key] = _Deck(pool, rng)
    return pools.decks[key]


def _grid_item(
    pools,
    rng,
    name,
    got,
    bubble,
    fill,
    box: bool,
    mark_horizontal: bool,
    pad=None,
    size=None,
):
    from common.prompts import TPL_BUBBLE, TPL_PLAIN, grid_caption
    from data.grid import WORD_PAD, render_grid

    cols, rows, gsize = GRIDS[name]
    size = size or gsize or pools.shapes.draw() or (512, 512)
    lines: list = []
    kw = {"box": True, "pad": WORD_PAD} if box else {}
    im, boxes = render_grid(
        got,
        cols,
        rows,
        size,
        pools.fonts,
        rng,
        bubble,
        (fill, fill),
        lines=lines,
        horizontal_frac=pools.horizontal_frac,
        **kw,
    )
    if cols * rows == 1:
        caption = (TPL_BUBBLE if bubble else TPL_PLAIN).format(got[0])
        layout, src = "flat", "font"
    else:
        caption = grid_caption(
            "bubble" if bubble else "flat",
            cols,
            rows,
            got,
            horizontal=set(lines) if mark_horizontal else set(),
        )
        layout, src = "grid", "grid"
    return Item(
        image=im,
        units=list(got),
        layout=layout,
        boxes=boxes,
        caption=caption,
        src=src,
        shape=tuple(size),
        extra={
            "grid": name,
            "bubble": bubble,
            "fill": round(fill, 3),
            "horizontal": sorted(lines),  # cells drawn as lines
        },
    )


def _fillable_grids(grids: list, n_distinct: int) -> list:
    """The grids whose cell count the inventory can fill (``_Deck.deal``
    asserts ``cells ≤ distinct units``); a small run drops the big ones."""
    return [(g, w) for g, w in grids if GRIDS[g][0] * GRIDS[g][1] <= n_distinct]


def grid_single(pools: Pools, rng: random.Random, p: dict):
    if p.get("digraphs"):
        pool = pools.singles + pools.digraphs
        deck = _deck(pools, "singles+digraphs", pool, rng)
    else:
        pool = pools.singles
        deck = _deck(pools, "singles", pool, rng)
    grids = _fillable_grids(
        parse_grids(p.get("grids", "1x1:2,2x2,3x3,2x3,3x2")), len(set(pool))
    )
    assert grids, "grid_single: no grid the singles can fill"
    name = rng.choices([g for g, _ in grids], weights=[w for _, w in grids])[0]
    cols, rows, size = GRIDS[name]
    got = deck.deal(cols * rows)
    bubble = rng.random() < float(p.get("bubble_frac", 0.5))
    if p.get("glyph_px"):
        # a px target, as grid_string: render_grid starts at fill × cell short
        # side and only shrinks, so fill = px / cell sets the px (1×1 draws its
        # canvas here so the cell is known)
        size = size or pools.shapes.draw() or (512, 512)
        cell = min(size[0] / cols, size[1] / rows)
        fill = _target(rng, p) / cell
    else:
        lo, hi = p.get("fill", [0.15, 0.8])
        fill = rng.uniform(float(lo), float(hi))
    return _grid_item(
        pools,
        rng,
        name,
        got,
        bubble,
        fill,
        False,
        bool(p.get("mark_horizontal", True)),
        size=size,
    )


def grid_string(pools: Pools, rng: random.Random, p: dict):
    """Strings in word cells at a target px: ``render_grid`` starts at the
    fill's font px and only shrinks, so ``fill = px / cell`` sets the px."""
    from data.grid import _NO_COLUMN  # noqa: F401  (documents the column rule)

    grids = parse_grids(p.get("grids", "2x2,2x3,3x2"))
    name = rng.choices([g for g, _ in grids], weights=[w for _, w in grids])[0]
    cols, rows, size = GRIDS[name]
    assert size is not None, "grid_string takes 2x2 and up"
    # source: pieces (kind piece) | short (lines: kind multi) | both (a mixed
    # grid takes the heavier kind, multi — windows.kind_of)
    src = p.get("source", "both")
    pool = _grid_string_pool(pools, p)
    assert pool, f"grid_string source {src!r} has no strings"
    lo, hi = p.get("glyph_px", [12, 24])
    px = rng.uniform(float(lo), float(hi))
    max_len = int(p.get("max_glyphs", 8))
    deck = _deck(pools, f"strings_{src}", pool, rng)
    got = deck.deal(cols * rows, max_len=max_len)
    bubble = rng.random() < float(p.get("bubble_frac", 0.5))
    cell = min(size[0] / cols, size[1] / rows)
    return _grid_item(
        pools,
        rng,
        name,
        got,
        bubble,
        px / cell,
        True,
        bool(p.get("mark_horizontal", True)),
    )


def _grid_string_pool(pools: Pools, p: dict) -> list:
    src = p.get("source", "both")
    pool = []
    if src in ("pieces", "both"):
        pool += pools.pieces
    if src in ("short", "both"):
        pool += pools.phrase.get("short", [])
    return pool


def missing_source(name: str, p: dict, pools: Pools) -> str | None:
    """Why ``name`` cannot draw from ``pools`` (None when it can). A recipe
    whose source is empty under the run's inventory is dropped by the
    builder and its share renormalised over the rest (``build.json``
    ``dropped``) — a 24-row run has no corpus line, production has all of
    them; the stage file is never edited for it."""
    if name == "scene_piece" and not pools.pieces:
        return "no pieces"
    if name == "scene_short" and not pools.phrase.get("short"):
        return "no short lines"
    if name == "scene_sentence" and not pools.phrase.get("sentence"):
        return "no sentence lines"
    if name == "grid_string":
        if not _grid_string_pool(pools, p):
            return f"no strings for source {p.get('source', 'both')!r}"
        cells = min(
            GRIDS[g][0] * GRIDS[g][1]
            for g, _ in parse_grids(p.get("grids", "2x2,2x3,3x2"))
        )
        if len(set(_grid_string_pool(pools, p))) < cells:
            return f"fewer strings than the smallest grid's {cells} cells"
    if name == "grid_single":
        pool = pools.singles + (pools.digraphs if p.get("digraphs") else [])
        if not _fillable_grids(
            parse_grids(p.get("grids", "1x1:2,2x2,3x3,2x3,3x2")), len(set(pool))
        ):
            return "no grid the singles can fill"
    return None


RECIPES = {
    "scene_single": scene_single,
    "grid_single": grid_single,
    "scene_piece": scene_piece,
    "scene_short": scene_short,
    "scene_sentence": scene_sentence,
    "grid_string": grid_string,
}
