"""``--grid`` items: k single units drawn mechanically in a cols × rows grid.

One item pays k ext rows. Every cell holds one unit of the singles pool
(``Inventory.pool()``, de-duplicated) at an identity-capable size, and the
caption carries one position clause per cell in reading order — the shape
``probe/position_probe.py`` read on the base model (the clause routes a quoted
text to its cell; reports/position_probe_2026_09_20.md).

- **Rows are dealt, not sampled**: the singles pool (every unit repeated by
  its weight, so a small kana's digraphs share one unit's mass as they do in
  the scene singles) is shuffled into a deck and dealt k at a time, so a row's
  draw count differs by at most one deck pass and a unit never repeats inside
  an item.
- **Position shuffle** is the deal itself: a unit lands in whatever cell its
  turn gives it, so across items it visits every cell of every grid.
- The canvas follows the grid so cells stay square: 2x2 / 3x3 on 512², 2x3
  (cols × rows) on 416×624, 3x2 on 624×416 — all in the 512² token family
  (1 024 / 1 014 tokens, two static block graphs).

Own rng stream (seed + 29): switching ``--grid`` on leaves every other draw of
the data dir untouched.

Multi-glyph units (``--grid_unit_min_glyph``, plan_z8): the deck's next unit
picks the item's (grid, frame) among those whose cell holds its glyph count
(``unit_fits`` — the word-cell rule; 3x3 takes one glyph or a digraph), and the
other cells are dealt from the units that fit there. Every unit is still dealt
once per deck pass; what moves is the grid share, toward the roomy cells. 0 =
every unit into every grid, the builds before 2026-09-21.

Word cells (``--grid_words``, plan_grid S1): ``--grid_word_frac`` of the items
hold a 2–4-piece real word per cell instead of a single unit. Words are drawn
**by row** (``_WordDeck``: rows cycled, the least-used word carrying the row,
``--grid_word_cap`` items per word), which sets distinct strings per row
directly. The cell-size rule (``max_glyphs``) keeps a word's glyphs at
``--grid_word_min_glyph`` px: never 3x3, and a bubble cell (a rounded box, so
the room stays near the cell) takes one glyph fewer than a flat one. Held
strings → eval ``gword_held``, as many drawn ones → ``gword``. Everything word
draws from its own stream (seed + 31).
"""

from __future__ import annotations

import random
import statistics as st
from collections import Counter

from common.prompts import grid_caption
from common.render.flat import JITTER_BG_LIGHT, JITTER_INK_DARK, pick_font
from common.render.scene import V_ROTATE

# a stacked column draws these unrotated / misplaced (PIL has no vertical
# forms), so a word holding one is always a line
_NO_COLUMN = V_ROTATE | set("「」、。")

# spec name → (cols, rows, (W, H))
GRIDS = {
    "2x2": (2, 2, (512, 512)),
    "3x3": (3, 3, (512, 512)),
    "2x3": (2, 3, (416, 624)),
    "3x2": (3, 2, (624, 416)),
}


def parse_grid(spec: str) -> list[tuple[str, float]]:
    """``'2x2,3x3:2,2x3,3x2'`` → ``[(name, weight)]`` (weight = item share)."""
    out = []
    for tok in spec.split(","):
        if not tok.strip():
            continue
        name, _, w = tok.strip().partition(":")
        assert name in GRIDS, f"--grid {name}: one of {', '.join(GRIDS)}"
        out.append((name, float(w) if w else 1.0))
    return out


class _Deck:
    """Shuffled passes over ``pool``; ``deal(k)`` returns k distinct units. A
    unit repeated in the pool (its ``--units`` weight; every unit but the
    ``small`` digraphs, SMALL_PER times) is dealt that many times per pass."""

    def __init__(self, pool: list, rng: random.Random):
        self.pool, self.rng, self.deck = pool, rng, []
        self.n_distinct = len(set(pool))

    def _fill(self):
        if not self.deck:
            self.deck = self.pool[:]
            self.rng.shuffle(self.deck)

    def peek(self) -> str:
        """The unit the next ``deal`` starts with."""
        self._fill()
        return self.deck[-1]

    def deal(self, k: int, max_len: int = 0) -> list:
        """``max_len``: units of more glyphs wait for a later item (the first
        unit is dealt whatever its length — it chose the grid)."""
        assert k <= self.n_distinct, f"grid of {k} cells over {self.n_distinct} units"
        got, aside = [], []
        while len(got) < k:
            self._fill()
            u = self.deck.pop()
            skip = u in got or (max_len and got and len(u) > max_len)
            (aside if skip else got).append(u)
            assert len(aside) <= 3 * len(self.pool), (
                f"no {k} units of ≤ {max_len} glyphs"
            )
        self.deck += aside  # a repeat / a too-long unit waits for the next item
        return got


# word cells: pad range and the rounded-box inset, as shares of the cell's short
# side — ``max_glyphs`` reads the room off the same numbers
WORD_PAD = (0.03, 0.05)
WORD_BOX_INSET = 0.05


def max_glyphs(name: str, bubble: bool, min_glyph: int) -> int:
    """The longest word (glyphs) a cell of grid ``name`` holds at ``min_glyph``
    px per glyph; 0 = no word cells (3x3)."""
    cols, rows, (W, H) = GRIDS[name]
    if cols * rows > 6:
        return 0
    cell = min(W / cols, H / rows)
    room = cell * (1 - 2 * WORD_PAD[1] - (2 * WORD_BOX_INSET if bubble else 0))
    return int(room / min_glyph)


def unit_fits(name: str, bubble: bool, glyphs: int, min_glyph: int) -> bool:
    """Whether a single unit of ``glyphs`` glyphs goes into a cell of grid
    ``name``: one glyph anywhere, a digraph in 3x3, else the word-cell room."""
    cols, rows, _ = GRIDS[name]
    if glyphs <= 1:
        return True
    if cols * rows > 6:
        return glyphs <= 2
    return glyphs <= max_glyphs(name, bubble, min_glyph)


def _fit_frame(a, grids, glyphs: int, rng) -> tuple[str, bool]:
    """(grid, bubble) for an item led by a unit of ``glyphs`` glyphs, by item
    share × frame share; a unit no cell holds goes to the roomiest flat one."""
    frames = [
        (g, b, w * (a.grid_bubble_frac if b else 1 - a.grid_bubble_frac))
        for g, w in grids
        for b in (False, True)
        if unit_fits(g, b, glyphs, a.grid_unit_min_glyph)
    ]
    frames = [f for f in frames if f[2] > 0]
    if not frames:
        return max((g for g, _ in grids), key=lambda g: max_glyphs(g, False, 1)), False
    g, b, _ = rng.choices(frames, weights=[f[2] for f in frames])[0]
    return g, b


def _cell_max(name: str, bubble: bool, min_glyph: int) -> int:
    cols, rows, _ = GRIDS[name]
    return 2 if cols * rows > 6 else max(1, max_glyphs(name, bubble, min_glyph))


class _WordDeck:
    """``--grid_words`` by-row draw: rows are cycled in shuffled passes and
    each row is carried by its least-used word still under ``cap``."""

    def __init__(self, words: dict, cap: int, rng: random.Random):
        self.words, self.cap, self.rng = words, cap, rng
        self.by_row: dict = {}
        for w, ps in words.items():
            for p in dict.fromkeys(ps):
                self.by_row.setdefault(p, []).append(w)
        self.rows = sorted(self.by_row)
        self.used: Counter = Counter()
        self.deck: list = []

    def deal(self, k: int, max_len: int) -> list | None:
        """k distinct words of at most ``max_len`` glyphs, or None when the
        rows cannot fill the item any more (caps reached)."""
        got, misses = [], 0
        while len(got) < k:
            if not self.deck:
                self.deck = [
                    r
                    for r in self.rows
                    if any(self.used[w] < self.cap for w in self.by_row[r])
                ]
                self.rng.shuffle(self.deck)
                if not self.deck:
                    return self._undo(got)
            r = self.deck.pop()
            cand = [
                w
                for w in self.by_row[r]
                if self.used[w] < self.cap and len(w) <= max_len and w not in got
            ]
            if not cand:
                misses += 1
                if misses > 2 * len(self.rows):
                    return self._undo(got)
                continue
            low = min(self.used[w] for w in cand)
            w = self.rng.choice([w for w in cand if self.used[w] == low])
            self.used[w] += 1
            got.append(w)
        return got

    def _undo(self, got):
        for w in got:
            self.used[w] -= 1
        return None


def word_supply(a, inv, tokq, rng) -> tuple[dict, list]:
    """``--grid_words`` → ``({trained word: [piece]}, held strings)``: the
    file's lines of ``--grid_word_pieces`` pieces, every piece a trained row,
    at least two distinct letters (not かー / ははは-as-one-letter)."""
    from pathlib import Path

    from .inventory import phrase_file_lines, pieces
    from .synth import _SHORT_DISTINCT, _letters

    assert inv.piece_ok is not None, "--grid_words needs the piece-coverage test"
    lo, hi = (int(x) for x in a.grid_word_pieces.split("-"))
    tok, qmap = tokq
    words: dict = {}
    for t, _book, _n in phrase_file_lines(Path(a.grid_words), lo, hi, norm=True):
        if t in words or len(set(_letters(t))) < _SHORT_DISTINCT:
            continue
        ps = [p for p, _ in pieces(tok, qmap, t)]
        if lo <= len(ps) <= hi and inv.piece_ok(t):
            words[t] = ps
    assert len(words) > a.grid_word_held, (
        f"--grid_words: {len(words)} covered words, --grid_word_held {a.grid_word_held}"
    )
    held = sorted(rng.sample(sorted(words), a.grid_word_held))
    for w in held:
        del words[w]
    return words, held


def render_grid(
    units,
    cols,
    rows,
    size,
    fonts,
    rng,
    bubble: bool,
    fill,
    pad=(0.04, 0.10),
    box=False,
    sizes=None,
    lines=None,
):
    """Draw ``units[i]`` in cell ``i`` (row-major). Returns ``(image, boxes)``,
    ``boxes[i]`` the ink bbox of cell i's unit in canvas pixels. ``box`` = the
    bubble is a rounded box (word cells) instead of an ellipse; ``sizes``
    collects every cell's font px, ``lines`` the cells drawn as a horizontal line."""
    from PIL import Image, ImageDraw, ImageFont

    W, H = size
    cw, ch = W / cols, H / rows
    pad_range = pad
    bg = rng.choice(JITTER_BG_LIGHT)
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    if bubble:
        for _ in range(900):
            x, y = rng.randrange(W), rng.randrange(H)
            d.ellipse((x, y, x + 2, y + 2), fill=(150, 150, 150))
    boxes = []
    for i, u in enumerate(units):
        r, c = divmod(i, cols)
        x0, y0 = c * cw, r * ch
        font_path = pick_font(u, fonts, rng)
        vertical = len(u) > 1 and rng.random() < 0.5 and not (set(u) & _NO_COLUMN)
        # the room the ink may take: the cell, or the inscribed rectangle of
        # the cell's bubble
        pad = rng.uniform(*pad_range) * min(cw, ch)
        room_w, room_h = cw - 2 * pad, ch - 2 * pad
        if bubble and box:
            inset = WORD_BOX_INSET * min(cw, ch)
            room_w, room_h = room_w - 2 * inset, room_h - 2 * inset
        elif bubble:
            room_w, room_h = room_w / 2**0.5 - 6, room_h / 2**0.5 - 6
        fs = int(rng.uniform(*fill) * min(cw, ch))
        while True:
            font = ImageFont.truetype(font_path, fs, index=0)
            txt = "\n".join(u) if vertical else u
            bx = d.multiline_textbbox((0, 0), txt, font=font, spacing=fs * 0.05)
            tw, th = bx[2] - bx[0], bx[3] - bx[1]
            if (tw <= room_w and th <= room_h) or fs <= 12:
                break
            fs = max(12, int(fs * min(room_w / tw, room_h / th) * 0.98))
        if sizes is not None:
            sizes.append(fs)
        if lines is not None and len(u) > 1 and not vertical:
            lines.append(i)
        # block centre: anywhere the ink stays inside its room
        cx = x0 + cw / 2 + (rng.random() - 0.5) * max(0.0, room_w - tw)
        cy = y0 + ch / 2 + (rng.random() - 0.5) * max(0.0, room_h - th)
        if bubble:
            frame = (x0 + pad, y0 + pad, x0 + cw - pad, y0 + ch - pad)
            kwb = {"fill": "white", "outline": "black", "width": rng.randint(2, 5)}
            if box:
                d.rounded_rectangle(frame, radius=0.18 * min(cw, ch), **kwb)
            else:
                d.ellipse(frame, **kwb)
        ox, oy = cx - tw / 2 - bx[0], cy - th / 2 - bx[1]
        kw = {"font": font, "spacing": fs * 0.05, "align": "center"}
        d.multiline_text((ox, oy), txt, fill=rng.choice(JITTER_INK_DARK), **kw)
        # the ink's own bbox: textbbox is off by a font's bearings (デ, ロ)
        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).multiline_text((ox, oy), txt, fill=255, **kw)
        boxes.append(
            list(mask.getbbox() or (int(x0), int(y0), int(x0 + cw), int(y0 + ch)))
        )
    return im, boxes


def _rec(fn, got, bubble, name, boxes, word: bool, lines=()) -> dict:
    cols, rows, size = GRIDS[name]
    return {
        "file": str(fn),
        # space-joined: set(text) is what the held-singles filter reads
        "text": " ".join(got),
        "caption": grid_caption(
            "bubble" if bubble else "flat", cols, rows, got, horizontal=set(lines)
        ),
        "src": "grid",
        "kind": f"grid{'w' if word else ''}{name}",
        "units": got,
        "boxes": boxes,
        "shape": list(size),
    }


def grid_recs(a, inv, fonts, out, tokq=None) -> list[dict]:
    rng = random.Random(a.seed + 29)
    pool = inv.pool()
    units = list(dict.fromkeys(pool))
    grids = parse_grid(a.grid)
    deck = _Deck(pool, rng)
    fill = (a.grid_fill_min, a.grid_fill_max)
    n_word, wdeck = 0, None
    if a.grid_words:
        # own stream: the single-cell items draw what they drew without words
        wrng = random.Random(a.seed + 31)
        words, held = word_supply(a, inv, tokq, wrng)
        wdeck = _WordDeck(words, a.grid_word_cap, wrng)
        wgrids = [
            (g, w) for g, w in grids if max_glyphs(g, False, a.grid_word_min_glyph)
        ]
        assert wgrids, "--grid_words: no word-capable grid in --grid (2x2, 2x3, 3x2)"
        n_word = round(a.n_grid * a.grid_word_frac)
    is_word = [True] * n_word + [False] * (a.n_grid - n_word)
    if n_word:
        wrng.shuffle(is_word)
    recs, draws, wsizes = [], Counter(), []
    for i, word in enumerate(is_word):
        fn = out / "img" / f"grid_{i:05d}.png"
        lines = [] if a.grid_mark_horizontal else None
        if word:
            if wdeck is None:
                continue  # caps reached: the word items stop here
            name = wrng.choices([g for g, _ in wgrids], weights=[w for _, w in wgrids])[
                0
            ]
            bubble = wrng.random() < a.grid_bubble_frac
            k = GRIDS[name][0] * GRIDS[name][1]
            got = wdeck.deal(k, max_glyphs(name, bubble, a.grid_word_min_glyph))
            if got is None:  # the roomiest cell once, then the supply is spent
                name, bubble = (
                    max(
                        (g for g, _ in wgrids),
                        key=lambda g: max_glyphs(g, False, a.grid_word_min_glyph),
                    ),
                    False,
                )
                k = GRIDS[name][0] * GRIDS[name][1]
                got = wdeck.deal(k, max_glyphs(name, False, a.grid_word_min_glyph))
            if got is None:
                wdeck = None
                continue
            cols, rows, size = GRIDS[name]
            im, boxes = render_grid(
                got,
                cols,
                rows,
                size,
                fonts,
                wrng,
                bubble,
                (0.9, 0.9),
                WORD_PAD,
                box=True,
                sizes=wsizes,
                lines=lines,
            )
        else:
            if a.grid_unit_min_glyph:
                name, bubble = _fit_frame(a, grids, len(deck.peek()), rng)
                cols, rows, size = GRIDS[name]
                got = deck.deal(
                    cols * rows, _cell_max(name, bubble, a.grid_unit_min_glyph)
                )
            else:
                name = rng.choices(
                    [g for g, _ in grids], weights=[w for _, w in grids]
                )[0]
                cols, rows, size = GRIDS[name]
                got = deck.deal(cols * rows)
                bubble = rng.random() < a.grid_bubble_frac
            im, boxes = render_grid(
                got, cols, rows, size, fonts, rng, bubble, fill, lines=lines
            )
            draws.update(got)
        im.save(fn)
        recs.append(_rec(fn, got, bubble, name, boxes, word, lines or ()))
    singles = [r for r in recs if not r["kind"].startswith("gridw")]
    if singles:
        n = [draws[u] for u in units]
        hs = [b[3] - b[1] for r in singles for b in r["boxes"]]
        print(
            f"grid: {len(singles)} items {dict(Counter(r['kind'] for r in singles))} "
            f"over {len(units)} units — draws per unit min {min(n)} / median "
            f"{st.median(n):g} / max {max(n)}; ink height px p05 "
            f"{sorted(hs)[len(hs) // 20]} / median {st.median(hs):g}; "
            f"{sum(h < 64 for h in hs)} of {len(hs)} under 64 px",
            flush=True,
        )
    if a.grid_words:
        _word_report(a, inv, recs, words, held, wrng, n_word, wsizes)
    return recs


def _word_report(a, inv, recs, words, held, wrng, n_word, sizes):
    """The word-cell log line, and the ``gword`` / ``gword_held`` eval groups
    (``gword`` = as many *drawn* words as there are held ones)."""
    wrecs = [r for r in recs if r["kind"].startswith("gridw")]
    used = Counter(w for r in wrecs for w in r["units"])
    strings: dict = {}
    occ: Counter = Counter()
    for w, c in used.items():
        for p in dict.fromkeys(words[w]):
            strings.setdefault(p, set()).add(w)
            occ[p] += c
    rows = sorted({p for ps in words.values() for p in ps})
    ns = [len(strings.get(p, ())) for p in rows]
    no = [occ[p] for p in rows]
    gs = sorted(sizes)  # font px per word cell
    inv.evals["gword"] = sorted(wrng.sample(sorted(used), min(len(held), len(used))))
    inv.evals["gword_held"] = held
    print(
        f"grid words: {len(wrecs)}/{n_word} items {dict(Counter(r['kind'] for r in wrecs))}; "
        f"{len(used)}/{len(words)} words drawn ({sum(c >= a.grid_word_cap for c in used.values())} "
        f"at the cap of {a.grid_word_cap}), held {len(held)}; {len(rows)} rows — distinct "
        f"strings per row min {min(ns)} / median {st.median(ns):g} / max {max(ns)} "
        f"({sum(x < 5 for x in ns)} rows under 5), draws per row min {min(no)} / median "
        f"{st.median(no):g} / max {max(no)}; glyph px p05 {gs[len(gs) // 20]} / median "
        f"{st.median(gs):g}, {sum(g < a.grid_word_min_glyph for g in gs)} of {len(gs)} "
        f"under {a.grid_word_min_glyph}",
        flush=True,
    )
