"""The pools every recipe draws from: the rows (single glyphs), the scenes,
the windowed word pool, the dialogue lines, the lone canvases."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import table as T

WINDOW_LEN = (2, 6)  # P1b's in-word lines were 2–6 glyphs


@dataclass
class Pools:
    fonts: list
    tokq: tuple
    inv: object  # data.vocabs.Inventory: the eval groups
    singles: list  # the rows' glyphs
    scenes: list
    single_idx: set  # scenes a lone glyph may go to
    horiz_idx: set  # scenes a left-to-right window may go to
    opt_in: dict  # pool tag → scenes only a tier naming it in scene_pools draws
    mono: set  # greyscale / line-art scenes
    shapes: object  # data.stage.ShapePool
    windows: dict = field(default_factory=dict)  # glyph → its windows
    windows_len: dict = field(default_factory=dict)  # glyph → length → its windows
    sentences: dict = field(default_factory=dict)  # cells → dialogue lines
    # with mark rows: mark → cells → the lines holding it (``sent`` draws the
    # mark first, as bubbleN its glyph: 、。 sit in most lines)
    sent_marks: dict = field(default_factory=dict)
    # the rows drawn alone (bubble1 / grid): a row whose lone quoted spelling
    # encodes to it (``…`` alone is T5's ``...``); ``None`` = every single
    lone: list | None = None
    synth: list = field(default_factory=list)  # synthesised dialogue lines (hearts)
    used: Counter = field(default_factory=Counter)  # scene → items drawn on it
    tier_used: Counter = field(
        default_factory=Counter
    )  # the draw loop's: scene → items
    scene_cap: int | None = None  # the draw loop's: items a scene may take
    decks: dict = field(default_factory=dict)
    lang: dict = field(default_factory=dict)  # glyph → language (a ``lang`` run's rows)


def _quietly(fn, *args):
    """A stage resolver with its stdout cut to 160 columns a line."""
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*args)
    for ln in buf.getvalue().splitlines():
        if ln.strip():
            print(ln if len(ln) <= 160 else ln[:159] + "…", flush=True)
    return out


def build_pools(rows: list, rng: random.Random, lang: dict | None = None) -> Pools:
    """``rows``: ``data.vocabs`` specs (``chars:…``), single glyphs only —
    the stage resolvers on them, so a row means what it meant in every read
    of record (and the eval groups ``eval.json`` carries). ``lang`` (glyph →
    language): the ``KOZH_FONTS`` faces join the fonts (``lang_fonts``)."""
    import tempfile
    from types import SimpleNamespace

    from cjk_scale.config import DATA
    from common.render.flat import find_fonts
    from data.inventory import pieces as qpieces
    from data.inventory import qwen_pieces
    from data.stage import (
        ShapePool,
        _base_inventory,
        _eval_strings,
        _resolve_singles,
        _word_set,
    )
    from data.synth import load_scenes
    from cjk_scale.windows import glyph_count

    a = SimpleNamespace(
        vocabs=list(rows),
        balanced=0,
        seed=T.SEED,
        phrase_min_pieces=int(DATA["phrase_min_pieces"]),
        phrase_max_pieces=int(DATA["phrase_max_pieces"]),
        phrase_norm=bool(DATA["phrase_norm"]),
        word_min_len=2,
        n_word_eval=0,
        line_max_len=16,
        n_line_eval=0,
    )
    inv = _base_inventory(a)
    tokq = qwen_pieces(char_rows=True)
    _eval_strings(a, rng, inv)
    with tempfile.TemporaryDirectory() as scratch:
        _quietly(_resolve_singles, a, Path(scratch), tokq, inv)
        _quietly(_word_set, a, Path(scratch), tokq, inv)
    for g in ("combo", "corpus", "line", "word", "word_held"):
        inv.evals.pop(g, None)
    tok, qmap = tokq
    pool = inv.pool()
    other = [u for u in pool if len(qpieces(tok, qmap, u)) != 1 or glyph_count(u) != 1]
    assert not other, f"reseed draws single glyphs only: {sorted(set(other))[:10]}"
    singles = list(dict.fromkeys(pool))
    if not inv.evals.get("single"):
        srng = random.Random(T.SEED + 47)
        inv.evals["single"] = sorted(srng.sample(singles, min(18, len(singles))))

    scenes = whole_bubbles(load_scenes(T.SCENES, 0.0, 0, "", T.ONE_BUBBLE))
    single_pools = set(T.SINGLE_SCENES.split(","))

    def single_ok(sc) -> bool:
        w = sc["region"][2] - sc["region"][0]
        h = sc["region"][3] - sc["region"][1]
        return sc["pool"] in single_pools and max(w, h) <= T.SINGLE_MAX_AR * max(
            1, min(w, h)
        )

    single_idx = {j for j, sc in enumerate(scenes) if single_ok(sc)}
    horiz = set(T.HORIZONTAL_SCENES.split(","))
    horiz_idx = {j for j, sc in enumerate(scenes) if sc["pool"] in horiz}
    n0 = len(scenes)
    scenes += whole_bubbles(load_scenes(T.SMALL_POOL, 0.0, 0, "", ""))
    opt_in = {T.SMALL_POOL: set(range(n0, len(scenes)))}
    mono = mono_scenes(scenes)
    print(
        f"pools: {len(singles)} rows; {n0} scenes ({len(single_idx)} take a lone "
        f"glyph, {len(horiz_idx)} a line) + {len(scenes) - n0} {T.SMALL_POOL}; "
        f"mono {len(mono)} / {len(scenes)} drawn at {T.MONO_SHARE}",
        flush=True,
    )
    fonts = find_fonts()
    if lang:
        fonts = lang_fonts(fonts, singles)
    return Pools(
        fonts=fonts,
        tokq=tokq,
        inv=inv,
        singles=singles,
        scenes=scenes,
        single_idx=single_idx,
        horiz_idx=horiz_idx,
        opt_in=opt_in,
        mono=mono,
        shapes=ShapePool(T.SHAPES, T.SEED),
        lang=dict(lang or {}),
    )


# ----------------------------------------------------------------------------
# rows lettered in another language (a ``lang`` run)

# the KO / ZH faces (task_report.md § 4; FONTS.md): out of the top-level
# assets/fonts that ``find_fonts()`` globs for every JA run — several ZH faces
# cover kana too
KOZH_FONTS = "kozh"


def empty_glyphs(font: str, glyphs) -> str:
    """The glyphs ``font``'s cmap maps to an empty outline (``TanukiMagic.ttf``'s
    你: ``font_covers`` passes it, ``render_grid`` divides by its zero width)."""
    from common.render.flat import font_covers
    from PIL import ImageFont

    pf = ImageFont.truetype(font, 64, index=0)
    out = ""
    for g in glyphs:
        if font_covers(font, g):
            x0, y0, x1, y1 = pf.getbbox(g)
            if x1 <= x0 or y1 <= y0:
                out += g
    return out


def lang_fonts(fonts: list, singles) -> list:
    """``fonts`` + the ``KOZH_FONTS`` faces, less every face that maps a row
    to an empty outline."""
    from common.render.flat import FONT_DIR

    kozh = sorted(str(p) for p in (FONT_DIR / KOZH_FONTS).glob("*.[ot]tf"))
    assert kozh, f"no faces in {FONT_DIR / KOZH_FONTS} (FONTS.md: re-fetch)"
    out = []
    for f in fonts + kozh:
        bad = empty_glyphs(f, singles)
        if bad:
            print(f"fonts: {Path(f).name} out (an empty outline for {bad})", flush=True)
        else:
            out.append(f)
    print(f"fonts: {len(out)} ({len(kozh)} KO / ZH faces joined)", flush=True)
    return out


LANG_TAGS = (("japanese text", "{l} text"), ("Japanese ", "{L} "))


def relang(caption: str, language: str) -> str:
    """A scene caption for an item lettered in ``language``: ``japanese text``
    → ``korean text`` in the tags, ``Japanese text / SFX reads as`` →
    ``Korean …`` in the clause."""
    assert "korean" not in caption.lower() and "chinese" not in caption.lower(), caption
    for a, b in LANG_TAGS:
        caption = caption.replace(a, b.format(l=language, L=language.capitalize()))
    return caption


def item_lang(pools: Pools, text: str) -> str | None:
    """The language of ``text``'s first glyph named in ``pools.lang`` (``None``:
    lettered as Japanese)."""
    return next((pools.lang[c] for c in text if c in pools.lang), None)


# ----------------------------------------------------------------------------
# whole bubbles: the outline inside the canvas, the erase sparing it


def bubble_check(sc: dict) -> dict:
    """``edge``: the least px between an anchor bubble's interior and the
    canvas edge (``None``: no anchor bubble); ``left``: the largest share of
    an anchor's letter ink the outline-keeping erase leaves."""
    import numpy as np
    from common.bubble import ring_median
    from common.render.scene import erase_paint, outline_ink
    from PIL import Image

    arr = np.asarray(Image.open(sc["file"]).convert("RGB")).copy()
    H, W = arr.shape[:2]
    bubbles = sc.get("bubbles") or [None] * len(sc["regions"])
    edge = [min(b[0], b[1], W - b[2], H - b[3]) for b in bubbles if b]
    left = 0.0
    for tb, reg, bub in zip(sc["boxes_anchor"], sc["regions"], bubbles):
        paint = erase_paint(arr, tb, reg, open_ok=bub is None)
        if paint is None:
            continue
        x0, y0, x1, y1 = (int(v) for v in tb)
        box = np.zeros((H, W), dtype=bool)
        box[y0:y1, x0:x1] = True
        fill = np.array(ring_median(arr, tb), dtype=np.int16)
        ink = box & (np.abs(arr.astype(np.int16) - fill).max(axis=2) > 24)
        kept = outline_ink(arr, tb, paint) & ink
        left = max(left, float(kept.sum() / max(1, ink.sum())))
    return {"edge": min(edge) if edge else None, "left": round(left, 4)}


def whole_bubbles(scenes: list) -> list:
    """``scenes`` less those whose bubble the canvas cuts (``BUBBLE_EDGE_MIN``)
    or whose erase would leave the letters (``ERASE_LEFT_MAX``)."""
    from cjk_scale.paths import OUT

    path = OUT / "experiments" / "scene_bubble_check.json"
    cache = json.loads(path.read_text("utf-8")) if path.exists() else {}
    miss = [s for s in scenes if s["file"] not in cache]
    for s in miss:
        cache[s["file"]] = bubble_check(s)
    if miss:
        path.write_text(json.dumps(cache, indent=0), encoding="utf-8")

    def cut(s) -> bool:
        e = cache[s["file"]]["edge"]
        return e is not None and e < T.BUBBLE_EDGE_MIN

    def left(s) -> bool:
        return cache[s["file"]]["left"] > T.ERASE_LEFT_MAX

    n_cut = sum(map(cut, scenes))
    n_left = sum(left(s) and not cut(s) for s in scenes)
    keep = [s for s in scenes if not cut(s) and not left(s)]
    print(
        f"whole bubbles: {len(keep)} / {len(scenes)} scenes kept — {n_cut} cut by "
        f"the canvas (< {T.BUBBLE_EDGE_MIN} px), {n_left} the erase leaves "
        f"lettered (> {T.ERASE_LEFT_MAX:g})",
        flush=True,
    )
    return keep


# ----------------------------------------------------------------------------
# scene colour (polish_seed's test; its cache is shared)


def colorful(file: str) -> float:
    """The share of a 128² thumbnail's pixels with HSV saturation and value
    over 0.15."""
    import numpy as np
    from PIL import Image

    im = np.asarray(Image.open(file).convert("RGB").resize((128, 128)), np.float32)
    im /= 255
    mx, mn = im.max(-1), im.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-3)
    return float(((sat > 0.15) & (mx > 0.15)).mean())


def mono_scenes(scenes: list) -> set:
    from cjk_scale.paths import OUT

    path = OUT / "experiments" / "scene_colorful.json"
    cache = json.loads(path.read_text("utf-8")) if path.exists() else {}
    miss = [s["file"] for s in scenes if s["file"] not in cache]
    for f in miss:
        cache[f] = colorful(f)
    if miss:
        path.write_text(json.dumps(cache, indent=0), encoding="utf-8")
    return {j for j, s in enumerate(scenes) if cache[s["file"]] < T.COLOR_MIN}


# ----------------------------------------------------------------------------
# the windowed word pool


def window_glyphs(singles) -> set:
    """The rows a window may hold: letters (kana, kanji, ー), not
    punctuation (``・`` trains lone only)."""
    import unicodedata

    return {g for g in set(singles) if unicodedata.category(g) in ("Lo", "Lm")}


def window_pool(glyphs: set, lines, held=(), length: tuple = WINDOW_LEN) -> list:
    """Every substring of ``lines`` of ``length`` glyphs, all in ``glyphs``,
    no glyph repeated (training must not teach doubling), none opening on
    ``scene.NO_HEAD`` or a small kana (``V_SMALL``: NO_HEAD lacks ぁぃぅぇぉゎ),
    none holding a trigram of a ``held`` string. A window may cross a word
    boundary (C3)."""
    from common.render.scene import NO_HEAD, V_SMALL

    no_head = NO_HEAD | V_SMALL
    grams = set()
    for h in held:
        n = min(3, len(h))
        grams |= {h[i : i + n] for i in range(len(h) - n + 1)}
    glens = sorted({len(g) for g in grams})  # look the window's own pieces up
    lo, hi = length
    out = set()
    for ln in lines:
        run = ""
        for c in ln + "\n":
            if c in glyphs:
                run += c
                continue
            for i in range(len(run)):
                if run[i] in no_head:
                    continue
                for n in range(lo, hi + 1):
                    w = run[i : i + n]
                    if len(w) < n:
                        break
                    if len(set(w)) == n and not any(
                        w[j : j + k] in grams
                        for k in glens
                        for j in range(len(w) - k + 1)
                    ):
                        out.add(w)
            run = ""
    return sorted(out)


# a mark row trains inside words (user, 10-05: the green leaf was `〜` trained
# lone in grid cells); `・` stays lone (the punct pack writes a lone one `.`)
MARK_LONE_ONLY = set("・")


def mark_singles(singles) -> list:
    """The run's rows that are not letters, less ``MARK_LONE_ONLY``."""
    letters = window_glyphs(singles)
    return [g for g in singles if g not in letters and g not in MARK_LONE_ONLY]


def context_letters() -> set:
    """The letters a mark window may hold around its mark: seed_retrain_0930's
    trained singles (kana, kanji, 々 — ``table.MARK_CONTEXT``), frozen at the
    seed while the marks train."""
    from library.env import resolve_under_home

    t = json.loads(resolve_under_home(T.MARK_CONTEXT).read_text("utf-8"))
    chars = "".join(t[k] for k in ("hiragana", "katakana", "kanji", "marks"))
    return window_glyphs(chars)


def held_grams(read: tuple) -> tuple:
    """``(trigrams of the read strings, 5-grams of the dialogue ruler's)``: no
    window or line holds one."""
    from . import OUT

    grams = set()
    for h in read:
        n = min(3, len(h))
        grams |= {h[i : i + n] for i in range(len(h) - n + 1)}
    ruler_file = OUT / "ruler" / "ruler.json"
    ruler = (
        [r["text"] for r in json.loads(ruler_file.read_text("utf-8"))["items"]]
        if ruler_file.exists()
        else []
    )
    return grams, {r[i : i + 5] for r in ruler for i in range(len(r) - 4)}


def heart_lines(lines: list, hearts: str, rng: random.Random, n: int) -> list:
    """``n`` dialogue lines with a heart (Manga109 letters none): at the end in
    place of the line's closing ``！？。`` (``すき♡``), or at a phrase break inside
    it at ``HEART_MID`` — in place of a ``！？`` (``あっ！だめ`` → ``あっ♡だめ``) or
    after a ``〜～…`` (``あ〜♡``) — never inside a word; doubled at
    ``HEART_DOUBLE``."""
    out = []
    for ln in rng.sample(lines, min(n, len(lines))):
        core = ln.rstrip("！!？?。")
        if len(core) < 2:
            continue
        h = rng.choice(hearts) * (2 if rng.random() < T.HEART_DOUBLE else 1)
        bang = [(i, i + 1) for i, c in enumerate(core) if c in "！!？?"]
        after = [(i + 1, i + 1) for i, c in enumerate(core[:-1]) if c in "〜～…"]
        after = [(i, j) for i, j in after if core[i] not in "〜～…"]
        mid = bang + after
        if mid and rng.random() < T.HEART_MID:
            i, j = rng.choice(mid)
            out.append(core[:i] + h + core[j:])
        else:
            out.append(core + h)
    return out


def mark_window_pool(
    letters: set, forms: dict, lines, held: tuple, length: tuple = WINDOW_LEN
) -> list:
    """Every substring of ``lines`` of ``length`` chars made of ``letters`` and
    mark spellings (``forms``) holding at least one of each: letters not
    repeated, no char three times running (``〜〜`` / ``……`` / ``♡♡`` stay),
    none opening on ``scene.NO_HEAD`` or a small kana, none holding a
    ``held_grams`` gram."""
    from common.render.scene import NO_HEAD, V_SMALL

    # a spelling opens no window its mark may not (`~` as `～`)
    no_head = NO_HEAD | V_SMALL | {c for c, m in forms.items() if m in NO_HEAD}
    grams, r5 = held
    ok = letters | set(forms)
    lo, hi = length
    out = set()
    for ln in lines:
        run = ""
        for c in ln + "\n":
            if c in ok:
                run += c
                continue
            for i in range(len(run)):
                if run[i] in no_head:
                    continue
                for n in range(lo, hi + 1):
                    w = run[i : i + n]
                    if len(w) < n:
                        break
                    ls = [c for c in w if c in letters]
                    if (
                        ls
                        and len(ls) < n
                        and len(set(ls)) == len(ls)
                        and not re.search(r"(.)\1\1", w)
                        and not any(g in w for g in grams)
                        and not any(w[k : k + 5] in r5 for k in range(n - 4))
                    ):
                        out.add(w)
            run = ""
    return sorted(out)


def ext_encoder():
    """``ext(route, text)``: the ext rows the pack's encoder gives ``text``
    in a caption clause, routed per glyph or not."""
    from transformers import AutoTokenizer

    from common.models import checkpoints
    from library.anima import ext_vocab
    from library.anima.ext_vocab import T5_TABLE_SIZE, HybridT5Encoder
    from library.anima.vocab_pack import resolve_pack_prefix
    from library.env import resolve_under_home

    t5 = AutoTokenizer.from_pretrained(
        resolve_under_home("library/anima/configs/t5_old")
    )
    qw = AutoTokenizer.from_pretrained(
        resolve_under_home("library/anima/configs/qwen3_06b")
    )
    _, mapping = ext_vocab.load_ext_assets(
        resolve_pack_prefix(checkpoints().vocab_pack)
    )
    encs = {
        r: HybridT5Encoder.from_mapping(t5, qw, mapping, glyph_route=r)
        for r in (False, True)
    }

    def ext(route: bool, text: str) -> list:
        ids, mask = encs[route].encode(f'Japanese text reads as "{text}".', 512)
        return [
            i - T5_TABLE_SIZE for i, m in zip(ids, mask) if m and i >= T5_TABLE_SIZE
        ]

    rows: dict = {}

    def glyph_row(c: str) -> int | None:
        """The one row ``c`` takes inside a word, routed (between two あ: a
        dot run rewrites only beside a routed char); ``None`` if not one."""
        if c not in rows:
            a = ext(True, "あ")
            got = ext(True, f"あ{c}あ")
            ok = len(a) == 1 and len(got) == 3 and got[0] == got[2] == a[0]
            rows[c] = got[1] if ok else None
        return rows[c]

    ext.glyph_row = glyph_row
    return ext


# fork-inherited (set before the pool forks; never pickled)
_EXT: dict = {}


def _routed_chunk(texts: list) -> list:
    return [_EXT["ext"](True, t) for t in texts]


def routed(ext, texts: list, workers: int = 1) -> list:
    """``ext(True, t)`` for every ``t``, in order, forked over ``workers``
    processes (the encoding check: ~0.16 ms a window, 1.3 M windows on
    sent_kanji's 1 348 rows)."""
    import multiprocessing as mp

    if workers <= 1 or len(texts) < 20_000:
        return [ext(True, t) for t in texts]
    _EXT["ext"] = ext
    k = -(-len(texts) // (workers * 8))
    chunks = [texts[i : i + k] for i in range(0, len(texts), k)]
    try:
        with mp.get_context("fork").Pool(workers) as pool:
            return [r for part in pool.map(_routed_chunk, chunks) for r in part]
    finally:
        _EXT.clear()


def add_windows(
    pools: Pools,
    read: tuple,
    out: Path,
    phrase: str,
    workers: int = 1,
    held: tuple = (),
) -> dict:
    """``pools.windows`` (glyph → its windows) over the dialogue lines (``phrase``) and
    the training set's own JA text, the read strings and ``held`` held out by
    trigram, every window routed to its glyphs' rows and nothing else (else
    dropped). Writes ``windows.json``; returns the stats for ``build.json``."""
    from cjk_scale.config import dataset_ja_lines

    glyphs = window_glyphs(pools.singles)
    lines = [
        ln.split("\t")[0]
        for ln in Path(phrase).read_text(encoding="utf-8").splitlines()
    ]
    ds = dataset_ja_lines()
    ws = window_pool(glyphs, lines + ds, (*read, *held))
    ext = ext_encoder()
    ids = {}
    for c in sorted(glyphs):
        a, b = ext(False, c), ext(True, c)
        assert len(a) == 1 and a == b, (c, a, b)
        ids[c] = a[0]
    got = routed(ext, ws, workers)
    ok = [w for w, g in zip(ws, got) if g == [ids[c] for c in w]]
    by_glyph: dict = {}  # one pass over ok (glyph × window was 2 G `in`s on 1 573 rows)
    for w in ok:
        for g in w:
            by_glyph.setdefault(g, []).append(w)
    pools.windows = {g: by_glyph[g] for g in sorted(glyphs) if g in by_glyph}
    marks = mark_singles(pools.singles)
    pools.lone = [
        g
        for g in pools.singles
        if g not in marks
        or (g not in T.MARK_NOT_LONE and ext(True, g) == [ext.glyph_row(g)])
    ]
    mstats = (
        add_mark_windows(pools, marks, lines + ds, read, ext, workers)
        if marks
        else None
    )
    pools.windows_len = {
        g: {k: [w for w in v if len(w) == k] for k in sorted({len(w) for w in v})}
        for g, v in pools.windows.items()
    }
    n = sorted(len(v) for v in pools.windows.values())
    stats = {
        "length": list(WINDOW_LEN),
        "held": list(read),
        **({"held_file": len(held)} if held else {}),
        "lines": {"dialogue": len(lines), "dataset": len(ds)},
        "n": len(ok),
        "dropped_by_encoding": len(ws) - len(ok),
        "glyphs": len(pools.windows),
        "glyphs_without": sorted(glyphs - set(pools.windows)),
        "per_glyph_min": n[0] if n else 0,
        "per_glyph_median": n[len(n) // 2] if n else 0,
        "by_length": dict(sorted(Counter(map(len, ok)).items())),
        **({"marks": mstats} if mstats else {}),
    }
    ok += sorted({w for m in marks for w in pools.windows.get(m, ())})
    (out / "windows.json").write_text(
        json.dumps(ok, ensure_ascii=False, indent=0), encoding="utf-8"
    )
    print(
        f"windows: {len(ok)} ({stats['dropped_by_encoding']} dropped by the "
        f"encoding check), {len(pools.windows)} glyphs, per glyph min "
        f"{stats['per_glyph_min']} median {stats['per_glyph_median']}; none for "
        f"{''.join(stats['glyphs_without']) or '-'} (lone only)",
        flush=True,
    )
    return stats


def add_mark_windows(
    pools: Pools, marks: list, lines: list, read: tuple, ext, workers: int = 1
) -> dict:
    """``pools.windows`` / ``windows_len`` for the mark rows: windows of the
    dialogue lines (ellipses normalised), the training set's text and
    ``HEART_LINES`` synthesised heart lines (``pools.synth``, when a heart is a
    row) around the seed's letters (``context_letters``), each window routed
    to its chars' rows and nothing else. A mark's spellings are the chars
    whose in-word row is its row (``〜`` and ``～`` for ``～``)."""
    rng = random.Random(T.SEED + 61)
    norm = [t for ln in lines if (t := norm_ellipsis(ln))]
    hearts = "".join(m for m in marks if m in T.HEARTS)
    pools.synth = heart_lines(norm, hearts, rng, T.HEART_LINES) if hearts else []
    letters = {c for c in context_letters() if ext.glyph_row(c) is not None}
    by_row = {ext.glyph_row(m): m for m in marks}
    assert None not in by_row and len(by_row) == len(marks), by_row
    cands = {c for ln in norm + pools.synth for c in ln if c not in letters}
    forms = {c: by_row[r] for c in sorted(cands) if (r := ext.glyph_row(c)) in by_row}
    forms.update({m: m for m in marks})
    ws = mark_window_pool(letters, forms, norm + pools.synth, held_grams(read))
    got = routed(ext, ws, workers)
    ok = [w for w, g in zip(ws, got) if g == [ext.glyph_row(c) for c in w]]
    for m in marks:
        v = [w for w in ok if any(forms.get(c) == m for c in w)]
        assert v, f"no window holds the mark {m}"
        pools.windows[m] = v
        pools.windows_len[m] = {
            k: [w for w in v if len(w) == k] for k in sorted({len(w) for w in v})
        }
    stats = {
        "forms": forms,
        "letters": len(letters),
        "synth_lines": len(pools.synth),
        "n": len(ok),
        "dropped_by_encoding": len(ws) - len(ok),
        "per_mark": {m: len(pools.windows[m]) for m in marks},
        "lone": [m for m in marks if m in pools.lone],
    }
    print(
        f"mark windows: {len(ok)} ({stats['dropped_by_encoding']} dropped by the "
        f"encoding check) on {len(letters)} context letters, per mark "
        f"{stats['per_mark']}; {len(pools.synth)} heart lines; spellings "
        f"{''.join(forms)}; drawn alone {''.join(stats['lone']) or '-'}",
        flush=True,
    )
    return stats


def kanji_count(s: str) -> int:
    """The CJK ideographs in ``s``."""
    import unicodedata

    return sum(unicodedata.name(c, "").startswith("CJK UNIFIED") for c in s)


def focus_pools(pools: Pools, focus: tuple, window_kanji: int) -> dict:
    """A ``focus`` run's draws: bubble1 / grid draw only ``focus`` alone
    (``pools.lone``), bubbleN only their windows of ≤ ``window_kanji`` kanji
    (glyph-first over ``focus``). The ``sent`` lines are cut in
    ``add_sentences``. Returns the stats."""
    fs = set(focus)
    lone = pools.singles if pools.lone is None else pools.lone
    pools.lone = [g for g in lone if g in fs]
    pools.windows = {
        g: v
        for g in focus
        if (
            v := [w for w in pools.windows.get(g, ()) if kanji_count(w) <= window_kanji]
        )
    }
    pools.windows_len = {
        g: {k: [w for w in v if len(w) == k] for k in sorted({len(w) for w in v})}
        for g, v in pools.windows.items()
    }
    n = sorted(len(v) for v in pools.windows.values())
    ws = {w for v in pools.windows.values() for w in v}
    stats = {
        "rows": len(focus),
        "lone": len(pools.lone),
        "window_kanji": window_kanji,
        "windows": len(ws),
        "windows_kanji_share": round(
            sum(map(kanji_count, ws)) / max(1, sum(map(len, ws))), 3
        ),
        "glyphs_without_windows": "".join(g for g in focus if g not in pools.windows),
        "per_glyph_min": n[0] if n else 0,
        "per_glyph_median": n[len(n) // 2] if n else 0,
    }
    print(
        f"focus: {len(focus)} rows, {len(pools.lone)} alone; {len(ws)} windows of ≤ "
        f"{window_kanji} kanji (kanji share {stats['windows_kanji_share']}), per "
        f"glyph min {stats['per_glyph_min']} median {stats['per_glyph_median']}; none "
        f"for {stats['glyphs_without_windows'] or '-'}",
        flush=True,
    )
    return stats


# ----------------------------------------------------------------------------
# the dialogue lines (``sent``)

# an ellipsis is drawn ``…`` under 4 dots, ``……`` at 4 or more (user, 10-05:
# Manga109 spells it ・・ / ･･･ / ・・・・・・, the page draws the leader)
_DOT_RUN = re.compile("[・･.．‥…]+")
_DOTS = {"‥": 2, "…": 3}
# what a line may hold off the pack's rows: the leader (T5's `...` on the raw
# pack; its own row on the punct pack) and the marks the fold sends to T5's ! / ?
SENT_BASE = set("…！？!?")


def norm_ellipsis(t: str) -> str | None:
    """``t`` with every dot run an ellipsis (``…`` under 4 dots, ``……`` at 4
    or more); a lone ・ / ･ is a 中黒 and stays ・; ``None`` for a lone . / ．
    (not a mark dialogue uses)."""
    out, at = [], 0
    for m in _DOT_RUN.finditer(t):
        run = m.group()
        if len(run) == 1 and run in "・･":
            rep = "・"
        elif len(run) == 1 and run in ".．":
            return None
        else:
            rep = "…" if sum(_DOTS.get(c, 1) for c in run) < 4 else "……"
        out += [t[at : m.start()], rep]
        at = m.end()
    return "".join(out) + t[at:]


def sentence_ok(s: str, lengths: tuple) -> str | None:
    """Why ``s`` (normalised) is not a ``sent`` line, else ``None``."""
    from common.render.scene import NO_HEAD
    from data.synth import _SENT_DISTINCT, _letters

    lo, hi = lengths
    if not lo <= len(s) <= hi:
        return "length"
    ls = _letters(s)
    if len(ls) < T.SENT_MIN_LETTERS or len(set(ls)) < _SENT_DISTINCT:
        return "letters"
    if s[0] in NO_HEAD:
        return "head"
    if re.search(r"([^…])\1\1", s):
        return "run"  # ああああ: a glyph three times running
    return None


def add_sentences(
    pools: Pools,
    read: tuple,
    lengths: tuple,
    out: Path,
    phrase: str,
    held: tuple = (),
    focus: tuple = (),
    line_kanji: int = 0,
) -> dict:
    """``pools.sentences`` (cells → lines): the dialogue lines (``phrase``) with their
    ellipses normalised, every char routed to its own single row (per glyph,
    as the windows are) or one of ``SENT_BASE`` with none; held out: a
    ``read`` string by trigram (``window_pool``'s rule), the dialogue
    ruler's 5+ glyph strings and ``held`` by 5-gram. With mark rows (``mark_singles``), the
    synthesised heart lines (``pools.synth``) join and a line must hold a mark;
    with ``focus``, a line holds one of them and ≤ ``line_kanji`` kanji. Writes
    ``sentences.json``; returns the stats."""
    lines = [
        ln.split("\t")[0].strip()
        for ln in Path(phrase).read_text(encoding="utf-8").splitlines()
    ] + pools.synth
    grams, r5 = held_grams(read)
    r5 |= {h[i : i + 5] for h in held for i in range(len(h) - 4)}
    fs = set(focus)
    ext = ext_encoder()
    mark_of = {ext.glyph_row(m): m for m in mark_singles(pools.singles)}
    mark_rows = set(mark_of)

    drop, keep = Counter(), set()
    for t in dict.fromkeys(lines):
        s = norm_ellipsis(t)
        why = "dot" if s is None else sentence_ok(s, lengths)
        if why is None:
            ids = [ext.glyph_row(c) for c in s]
            if any(i is None and c not in SENT_BASE for c, i in zip(s, ids)):
                why = "char"
            elif mark_rows and not mark_rows & set(ids):
                why = "no_mark"
            elif fs and not fs & set(s):
                why = "no_focus"
            elif fs and kanji_count(s) > line_kanji:
                why = "kanji"
            elif any(g in s for g in grams):
                why = "read"
            elif any(s[i : i + 5] in r5 for i in range(len(s) - 4)):
                why = "ruler"
            elif ext(True, s) != [i for i in ids if i is not None]:
                why = "route"
        if why is None:
            keep.add(s)
        else:
            drop[why] += 1
    pools.sentences, pools.sent_marks = {}, {}
    for s in sorted(keep):
        pools.sentences.setdefault(len(s), []).append(s)
        for m in sorted({mark_of[i] for c in s if (i := ext.glyph_row(c)) in mark_of}):
            pools.sent_marks.setdefault(m, {}).setdefault(len(s), []).append(s)
    stats = {
        "lengths": list(lengths),
        "lines": len(lines),
        "n": len(keep),
        "by_length": {n: len(v) for n, v in sorted(pools.sentences.items())},
        "ellipsis": sum("…" in s for s in keep),
        **(
            {
                "focus_kanji_share": round(
                    sum(map(kanji_count, keep)) / max(1, sum(map(len, keep))), 3
                ),
                "focus_without": "".join(
                    g for g in focus if not any(g in s for s in keep)
                ),
            }
            if fs
            else {}
        ),
        **(
            {
                "per_mark": {
                    m: sum(map(len, v.values())) for m, v in pools.sent_marks.items()
                }
            }
            if pools.sent_marks
            else {}
        ),
        "dropped": dict(drop),
        "ruler_5grams": len(r5),
    }
    (out / "sentences.json").write_text(
        json.dumps(sorted(keep), ensure_ascii=False, indent=0), encoding="utf-8"
    )
    print(
        f"sentences: {len(keep)} of {len(lines)} lines ({stats['ellipsis']} with "
        f"an ellipsis), {lengths[0]}–{lengths[1]} cells; dropped {dict(drop)}",
        flush=True,
    )
    return stats


def lang_caption(pools: Pools, caption: str, text: str) -> str:
    """``caption`` in the language of ``text``'s rows (a JA item's unchanged)."""
    lang = item_lang(pools, text)
    return relang(caption, lang) if lang else caption
