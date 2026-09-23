"""Stage ``data`` — build the glyph training set and the eval prompt set.

Writes ``<data_dir>/{img/, train.jsonl, eval.json, sheet_train.png}`` (+
``words.json`` / ``kanji.json`` when those inventories are on).

What goes into the table is one flag — ``--units`` (``data/units.py``), a
repeatable source spec. This module resolves those sources against the corpus
and the tokenizer, then draws the items.

Bit-identity contract: the main ``rng`` (seed 0) is consumed in a fixed order
— eval singles, eval combos, held corpus shuffle, font items, renders, corpus
shuffle, sheet draw. Every later lever draws from its own stream (shapes
seed+17, kana_ext seed+19, kanji seed+23, words seed+13, grid seed+29), so switching a lever
off rebuilds the older data dirs identically. The ``--units`` sources are
resolved in the canonical order of ``data/units.py``, never the typed order,
for the same reason. Keep it that way.
"""

from __future__ import annotations

import json
import random
from collections import Counter

from common.paths import CORPUS_HELD, CORPUS_TRAIN, data_dir
from common.prompts import EN_WORDS, TPL_BUBBLE, TPL_EN, TPL_PLAIN
from common.readers import contact_sheet
from common.render.flat import (
    crop_bubble,
    find_fonts,
    pick_font,
    render_string,
    sample_layout,
)
from common.shapes import parse_shapes
from common.text import (
    HIRA,
    KANA,
    KANA_EXT,
    KANA_EXT_HIRA,
    KANA_EXT_KATA,
    KANA_SMALL,
    KATA,
)

from .inventory import (
    clean_kana_strings,
    corpus_lines,
    kanji_inventory,
    pieces,
    qwen_pieces,
    small_digraphs,
    word_inventory,
)
from .units import SMALL_PER, Inventory, parse_units

# eval.json group order (skipped when empty)
_EVAL_ORDER = (
    "single",
    "combo",
    "corpus",
    "en",
    "word",
    "word_held",
    "line",
    "single_kanji",
    "single_ext",
    "single_small",
    "single_extra",
    "flip",
    "str3",
    "phrase",
    "phrase_held",
    "short",
    "short_held",
    "gword",
    "gword_held",
)


class ShapePool:
    """``--shapes`` canvas draws on their own rng stream (an unset ``--shapes``
    draws nothing and every render is 512²)."""

    def __init__(self, spec: str, seed: int):
        self.shapes = parse_shapes(spec)
        self.rng = random.Random(seed + 17)

    def draw(self, square: bool = False):
        if not self.shapes:
            return None
        pool = [x for x in self.shapes if not square or x[0] == x[1]] or [
            (min(x[:2]), min(x[:2]), x[2]) for x in self.shapes
        ]
        W, H, _ = self.rng.choices(pool, weights=[x[2] for x in pool])[0]
        return (W, H)


def _rec(fn, text, caption, src, shp, **extra) -> dict:
    return {
        "file": str(fn),
        "text": text,
        "caption": caption,
        "src": src,
        **extra,
        **({"shape": list(shp)} if shp else {}),
    }


def stage_data(a):
    # --seed moves every stream, so shards of one recipe (seed 0..K-1, train.jsonl
    # joined) draw different items; seed 0 is every build before 2026-09-21
    rng = random.Random(a.seed)
    out = data_dir(a)
    (out / "img").mkdir(parents=True, exist_ok=True)
    fonts = find_fonts()
    print(f"fonts: {len(fonts)}", flush=True)
    # mixed shapes (2026-09-14): every font item draws its canvas (W, H) from
    # --shapes; corpus crops are square, so they draw from the pool's squares
    shapes = ShapePool(a.shapes, a.seed)
    inv = _base_inventory(a)
    # --scenes needs the piece map for piece_ok even with no words source
    # (micro arms); a `list:` source needs it to check each unit is one piece
    tokq = (
        qwen_pieces() if (inv.needs_tokenizer() or a.scenes or a.grid_words) else None
    )
    # eval strings first so the training pool can exclude the combos
    combos_eval, n_possible = _eval_strings(a, rng, inv)
    _resolve_singles(a, out, tokq, inv)
    if inv.has("words") or a.scenes or a.grid_words:
        _word_set(a, out, tokq, inv)

    n_target = min(a.n_combo, 50 * (n_possible - len(combos_eval)))
    if a.scenes:
        # S line (plan_synth): the whole mix comes from synth.py; no corpus crops
        from .synth import synth_recs

        recs = synth_recs(a, rng, inv, combos_eval, fonts, shapes, out, tokq)
    elif a.grid:
        recs = []  # a grid-only dir: no font items, no corpus crops
    elif a.balanced:
        recs = _balanced_font_recs(
            a, rng, inv, combos_eval, n_target, fonts, shapes, out
        )
    else:
        texts = _font_texts(a, rng, inv, combos_eval, n_target, tokq)
        recs = []
        for i, s in enumerate(texts):
            shp = shapes.draw()
            im, bubble = render_string(
                s, pick_font(s, fonts, rng), rng, size=shp or 512, mode=a.layout
            )
            fn = out / "img" / f"font_{i:05d}.png"
            im.save(fn)
            recs.append(
                _rec(
                    fn, s, (TPL_BUBBLE if bubble else TPL_PLAIN).format(s), "font", shp
                )
            )
    if a.grid:
        # 2026-09-20: k units per canvas, one position clause per cell; with
        # --scenes it is added to the S-line mix (which must be a --shapes build)
        from .grid import grid_recs

        assert all("shape" in r for r in recs), "--grid beside --scenes needs --shapes"
        recs += grid_recs(a, inv, fonts, out, tokq)
    if not a.scenes and not a.grid and inv.kana:
        # corpus crops are kana bubble lines: under --no_kana they would put
        # the dropped kana rows back into the trained table through the captions
        recs += _corpus_recs(a, rng, inv, shapes, out, first_layout_id=len(recs))

    _ink_stats(recs)
    (out / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in recs)
    )
    ev = [
        {
            "group": g,
            "text": s,
            "caption": (TPL_EN if g == "en" else TPL_BUBBLE).format(s),
        }
        for g in _EVAL_ORDER
        for s in inv.evals.get(g, ())
    ]
    (out / "eval.json").write_text(json.dumps(ev, ensure_ascii=False, indent=1))
    c = Counter(r["src"] for r in recs)
    print(
        f"data: {len(recs)} train items {dict(c)}; eval {len(ev)} prompts", flush=True
    )
    if shapes.shapes:
        cs = Counter("x".join(map(str, r["shape"])) for r in recs)
        print(f"shapes: {dict(sorted(cs.items()))}", flush=True)
    _train_sheet(a, rng, recs, out)


def _ink_stats(recs) -> dict:
    """plan_band § 3: every boxed record gets ``glyphs`` / ``ink`` (ink px
    inside its box(es)) / ``box_area`` (px²), and the build prints, per
    kind, the median (p10–p90) glyph px — √(box area / glyphs) — and ink per
    glyph in latent cells² (ink px / 64 / glyphs). An arm runs only if the
    median px lands in its cell's target ± 20 %. Returns ``{kind: (px, ink)}``
    medians for tests."""
    import statistics as st

    from PIL import Image

    from common.render.ink import box_area, glyph_count, ink_pixels

    px_by, ink_by = {}, {}
    for r in recs:
        boxes = r.get("boxes") or ([r["box"]] if r.get("box") else None)
        if not boxes:
            continue
        units = r.get("units") or [r["text"]]
        glyphs = max(1, sum(glyph_count(u) for u in units))
        with Image.open(r["file"]) as im:
            g = im.convert("L")
            ink = sum(ink_pixels(g, b) for b in boxes)
        area = sum(box_area(b) for b in boxes)
        r["glyphs"], r["ink"], r["box_area"] = glyphs, ink, area
        px_by.setdefault(r["kind"], []).append((area / glyphs) ** 0.5)
        ink_by.setdefault(r["kind"], []).append(ink / 64 / glyphs)

    def q(xs):
        xs = sorted(xs)
        p10, p90 = xs[int(0.1 * (len(xs) - 1))], xs[int(0.9 * (len(xs) - 1))]
        return f"{st.median(xs):.0f} ({p10:.0f}–{p90:.0f})"

    out = {}
    for kind in sorted(px_by):
        out[kind] = (st.median(px_by[kind]), st.median(ink_by[kind]))
        print(
            f"ink {kind}: n {len(px_by[kind])}, glyph px {q(px_by[kind])}, "
            f"ink/glyph cells² {q(ink_by[kind])}",
            flush=True,
        )
    return out


# ----------------------------------------------------------------------------
# inventories and eval strings


def _base_inventory(a) -> Inventory:
    """``--units`` → the Inventory, with the base (``kana`` / ``chars:``)
    sources resolved. The rest need the tokenizer and are done later."""
    inv = Inventory(sources=parse_units(a.units))
    for s in inv.sources:
        if s.kind == "kana":
            s.units = list(KANA)
        elif s.kind != "chars":
            continue
        inv.kana += [c for c in s.units if c not in inv.kana]
    # a hand-picked base (chars: with no kana beside it): the eval singles are
    # the base itself and corpus lines are filtered down to it
    inv.restricted = inv.has("chars") and not inv.has("kana")
    print(
        f"units: --units {inv.describe()} → base {len(inv.kana)}"
        + (" (restricted)" if inv.restricted else ""),
        flush=True,
    )
    return inv


def _eval_strings(a, rng, inv: Inventory):
    """Eval singles / combos / held corpus lines (main rng) and the EN control.
    Returns ``(combos_eval, number of possible 2–3 kana strings)``."""
    kana = inv.kana
    if not kana:  # no base source: no kana singles / combos / corpus lines
        singles_eval: list = []
    elif inv.restricted:
        singles_eval = kana[:18]
    else:
        singles_eval = rng.sample(list(HIRA), 12) + rng.sample(list(KATA), 6)
    combos_eval: set = set()
    n_possible = len(kana) ** 2 + len(kana) ** 3
    # a tiny alphabet cannot fill 18 (smoke runs)
    n_eval_combos = min(18, n_possible // 2)
    while len(combos_eval) < n_eval_combos:
        k = rng.choice([2, 3])
        combos_eval.add("".join(rng.choice(kana) for _ in range(k)))
    held = corpus_lines(CORPUS_HELD / "boxes.jsonl", 4)
    rng.shuffle(held)
    if inv.restricted or not kana:
        held = [ln for ln in held if all(c in kana for c in ln[0] if c in KANA)]
    corpus_eval: list = []
    for t, _rel, _box in held:
        if t not in corpus_eval and len(corpus_eval) < 10:
            corpus_eval.append(t)
    inv.evals.update(
        single=singles_eval,
        combo=sorted(combos_eval),
        corpus=corpus_eval,
        en=EN_WORDS,
    )
    return combos_eval, n_possible


def _resolve_singles(a, out, tokq, inv: Inventory):
    """The singles-only sources: ``kana_ext`` (68 voiced / handakuten / small),
    ``kanji:N`` (the N most frequent single-row corpus kanji) and ``list:``
    (literal ext-row units — the punctuation arm).

    Each keeps its own rng stream (kana_ext seed+19, kanji seed+23), so adding
    or dropping one leaves the other's draw untouched.
    """
    if inv.has("kana_ext"):
        assert not inv.restricted and not a.balanced, (
            "--units kana_ext: full inventory, unbalanced"
        )
        src = inv.source("kana_ext")
        src.units = list(KANA_EXT)
        inv.kana_ext = src.units
        erng = random.Random(a.seed + 19)
        # lone small kana are never drawn as singles (they train inside
        # `small` digraphs and read on single_small), so they scored 0 by
        # construction here — 8 of step1_0919's 36
        hira = [c for c in KANA_EXT_HIRA if c not in KANA_SMALL]
        kata = [c for c in KANA_EXT_KATA if c not in KANA_SMALL]
        inv.evals["single_ext"] = erng.sample(hira, 12) + erng.sample(kata, 6)
    ssrc = inv.source("small")
    if ssrc:
        # 2026-09-19: the small kana cannot be drawn as singles, so the 53 k
        # seed table had no row for them and the sentence step met them cold
        assert inv.has("kana") and not inv.restricted and not a.balanced, (
            "--units small: hosts are the full kana inventory, unbalanced"
        )
        inv.small_of = small_digraphs(*tokq, inv.kana + inv.kana_ext, SMALL_PER)
        ssrc.units = [d for ds in inv.small_of.values() for d in dict.fromkeys(ds)]
        inv.evals["single_small"] = [ds[0] for ds in inv.small_of.values()]
        (out / "small.json").write_text(json.dumps(inv.small_of, ensure_ascii=False))
        print(
            f"small kana: {len(inv.small_of)} rows in {len(ssrc.units)} digraphs; "
            + " ".join(
                f"{k}:{'/'.join(dict.fromkeys(v))}" for k, v in inv.small_of.items()
            ),
            flush=True,
        )
    ksrc = inv.source("kanji")
    if ksrc and ksrc.n:
        assert not inv.restricted and not a.balanced, (
            "--units kanji: full inventory, unbalanced"
        )
        ksrc.freq = kanji_inventory(*tokq, ksrc.n)
        ksrc.units = [c for c, _ in ksrc.freq]
        inv.kanji = ksrc.units
        krng = random.Random(a.seed + 23)
        inv.evals["single_kanji"] = krng.sample(inv.kanji, min(18, len(inv.kanji)))
        (out / "kanji.json").write_text(json.dumps(ksrc.freq, ensure_ascii=False))
        print(
            f"kanji: {len(inv.kanji)} singles "
            f"(last {ksrc.freq[-1][0]}:{ksrc.freq[-1][1]}); {''.join(inv.kanji)}",
            flush=True,
        )
    lsrc = inv.source("list")
    if lsrc:
        # punctuation arm (2026-09-16): every unit must be one Qwen piece with
        # an ext row (、 。 ・ ー ～ ！ ？ 「 」 ！！ ・・・ っ ッ …); all of them form
        # eval group single_extra — read on the sheets, since the readers'
        # norm() strips punctuation before matching
        tok, qmap = tokq
        bad = [u for u in lsrc.units if len(pieces(tok, qmap, u)) != 1]
        assert not bad, f"--units list: not one Qwen piece: {bad}"
        norow = [u for u in lsrc.units if pieces(tok, qmap, u)[0][1] is None]
        assert not norow, f"--units list: pretrained piece, no ext row: {norow}"
        inv.extra = lsrc.units
        inv.evals["single_extra"] = inv.extra[:18]
        print(
            f"extra units: {len(inv.extra)} singles {' '.join(inv.extra)}", flush=True
        )


def _word_set(a, out, tokq, inv: Inventory):
    """2026-09-14 word addresses (``--units words:N/held=K``): the inventory
    gains the corpus's most frequent single-piece words (each its own pack
    row), K of them held out; corpus lines are kept only when every piece is a
    trained row (kana single or word) so ``line`` evaluates addresses *in
    sequence*, not coverage.

    Runs with no words source too (``--scenes`` needs ``piece_ok`` for the
    micro arms): N is then 0 and the coverage test is the singles alone.
    """
    tok, qmap = tokq
    src = inv.source("words")
    n_words = src.n if src else 0
    n_held = src.held if src else 0
    freq = word_inventory(tok, qmap, n_words, a.word_min_len)
    inv.words = [w for w, _ in freq]
    wrng = random.Random(a.seed + 13)
    inv.words_held = sorted(wrng.sample(inv.words, n_held)) if n_held else []
    inv.words_train = [w for w in inv.words if w not in inv.words_held]
    if src:
        src.freq, src.units, src.held_units = freq, inv.words, inv.words_held
    kana_rows = {
        p
        for c in inv.kana + inv.kana_ext + inv.kanji + inv.extra
        for p, row in pieces(tok, qmap, c)
        if row is not None
    }
    trained_pieces = kana_rows | set(inv.words_train)
    if a.phrase_file and a.phrase_pieces:
        # sentence line (2026-09-16): the phrase file's most frequent pieces
        # outside the inventory become rows too, so more of its lines are
        # drawable; they are trained only through the phrases that carry them
        from pathlib import Path

        from .inventory import phrase_file_lines, phrase_pieces

        plines = phrase_file_lines(
            Path(a.phrase_file),
            a.phrase_min_pieces,
            a.phrase_max_pieces,
            norm=bool(a.phrase_norm),
        )
        # held words stay held: they must not come back as phrase rows
        extra = phrase_pieces(
            tok, qmap, plines, trained_pieces | set(inv.words_held), a.phrase_pieces
        )
        inv.phrase_pieces = [p for p, _ in extra]
        trained_pieces |= set(inv.phrase_pieces)
        print(
            f"phrase pieces: +{len(extra)} rows from {a.phrase_file} "
            f"({len(plines)} lines; last {extra[-1][0]}:{extra[-1][1]}): "
            + " ".join(p for p, _ in extra[:40])
            + (" …" if len(extra) > 40 else ""),
            flush=True,
        )

    def piece_ok(text: str) -> bool:
        return all(
            row is not None and p in trained_pieces
            for p, row in pieces(tok, qmap, text)
        )

    inv.piece_ok = piece_ok
    (out / "words.json").write_text(
        json.dumps(
            {
                "freq": freq,
                "held": inv.words_held,
                "kana_pieces": sorted(kana_rows),
                "phrase_pieces": inv.phrase_pieces,
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    print(
        f"words: {len(inv.words)} (held {len(inv.words_held)}: {' '.join(inv.words_held)}); "
        f"top {' '.join(f'{w}:{c}' for w, c in freq[:20])}",
        flush=True,
    )
    # eval: trained words, held-out words, covered held-out corpus lines
    words_eval = wrng.sample(inv.words_train, min(a.n_word_eval, len(inv.words_train)))
    held_lines = corpus_lines(CORPUS_HELD / "boxes.jsonl", a.line_max_len)
    wrng.shuffle(held_lines)
    lines_eval: list = []
    for t, _rel, _box in held_lines:
        if t in lines_eval or not (2 <= len(pieces(tok, qmap, t)) <= 3):
            continue
        if not piece_ok(t):
            continue
        lines_eval.append(t)
        if len(lines_eval) >= a.n_line_eval:
            break
    inv.evals.update(word=words_eval, word_held=inv.words_held, line=lines_eval)
    print(
        f"words eval: {len(words_eval)} trained, {len(inv.words_held)} held, "
        f"{len(lines_eval)} covered 2-3 piece lines",
        flush=True,
    )


# ----------------------------------------------------------------------------
# training items


def _font_texts(a, rng, inv: Inventory, combos_eval, n_target, tokq) -> list[str]:
    """Unbalanced font-render texts: the strings arm, or singles ×n_single +
    random 2–3 kana combos (+ trained words, extended kana, kanji)."""
    kana = inv.kana
    texts: list[str] = []
    if a.strings_only:
        # strings arm (2026-09-14): no singles at all — every item is a
        # 2–4-piece random-order string of trained rows (kana, and trained
        # words at --word_frac per slot), so no row can carry a single-unit
        # layout and the loss asks the rows to be contextualisable
        assert inv.piece_ok is not None, "--strings_only needs --units words:N"
        tok, qmap = tokq
        flip_pairs = clean_kana_strings(
            tok, qmap, kana, rng, a.n_flip_eval, 2, excl=combos_eval
        )
        flip_eval = [x for s_ in flip_pairs for x in (s_, s_[::-1])]
        str3_eval = clean_kana_strings(
            tok, qmap, kana, rng, a.n_str3_eval, 3, excl=combos_eval
        )
        inv.evals.update(flip=flip_eval, str3=str3_eval)
        excl = combos_eval | set(flip_eval) | set(str3_eval)
        tries = 0
        while len(texts) < a.n_strings and tries < 50 * a.n_strings:
            tries += 1
            if a.single_frac > 0 and rng.random() < a.single_frac:
                # mixed distribution (plan P1): a single kana or trained word,
                # so unit count is only predictable from the caption
                texts.append(
                    rng.choice(kana + inv.kana_ext + inv.kanji + inv.words_train)
                )
                continue
            k = rng.choices([2, 3, 4], weights=[45, 35, 20])[0]
            parts = [
                rng.choice(inv.words_train)
                if inv.words_train and rng.random() < a.word_frac
                else rng.choice(kana)
                for _ in range(k)
            ]
            s_ = "".join(parts)
            if s_ in excl or [p for p, _ in pieces(tok, qmap, s_)] != parts:
                continue
            if not inv.piece_ok(s_):
                continue
            texts.append(s_)
        print(
            f"strings: {len(texts)} items (tries {tries}); flip eval {len(flip_eval)}, "
            f"str3 eval {len(str3_eval)}",
            flush=True,
        )
        return texts
    for ch in kana:
        texts += [ch] * a.n_single
    for w in inv.words_train:
        texts += [w] * a.n_single
    # P0b: extended kana / kanji, singles only; `list:` units the same way
    # (2026-09-16 — without this they reach the flat stream only via --scenes)
    for ch in inv.kana_ext + inv.kanji + inv.extra:
        texts += [ch] * a.n_single
    n_combo = 0
    while n_combo < n_target:
        k = rng.choice([2, 3])
        s = "".join(rng.choice(kana) for _ in range(k))
        if s in combos_eval:
            continue
        texts.append(s)
        n_combo += 1
    return texts


def _balanced_font_recs(
    a, rng, inv: Inventory, combos_eval, n_target, fonts, shapes, out
):
    """W2a: groups of `g` distinct strings rendered in ONE layout (font, canvas,
    bubble, glyph size/position) — layout cancels inside the batch, identity is
    the only gradient. Singles: each round partitions the shuffled inventory
    (every kana exactly n_single times when g | len(kana))."""
    kana = inv.kana
    g = a.balanced
    assert len(kana) >= g, f"--balanced {g} needs ≥ {g} chars"
    groups = []
    for _ in range(a.n_single):
        perm = kana[:]
        rng.shuffle(perm)
        for j in range(0, len(perm), g):
            grp = perm[j : j + g]
            if len(grp) < g:
                grp += rng.sample([c for c in kana if c not in grp], g - len(grp))
            groups.append(grp)
    n_combo = 0
    while n_combo < n_target:
        k = rng.choice([2, 3])
        grp: list = []
        for _ in range(1000):
            s = "".join(rng.choice(kana) for _ in range(k))
            if s not in combos_eval and s not in grp:
                grp.append(s)
                if len(grp) == g:
                    break
        if len(grp) < g:  # tiny alphabet (smoke runs)
            break
        groups.append(grp)
        n_combo += g
    recs = []
    for lid, grp in enumerate(groups):
        font = pick_font("".join(grp), fonts, rng)
        shp = shapes.draw()  # one shape per layout group: a batch is one shape
        lay = sample_layout(len(grp[0]), rng, size=shp or 512, mode=a.layout)
        for s in grp:
            im, bubble = render_string(s, font, rng, size=shp or 512, layout=lay)
            fn = out / "img" / f"font_{len(recs):05d}.png"
            im.save(fn)
            caption = (TPL_BUBBLE if bubble else TPL_PLAIN).format(s)
            recs.append(_rec(fn, s, caption, "font", shp, layout_id=lid))
    return recs


def _corpus_recs(a, rng, inv: Inventory, shapes, out, first_layout_id: int):
    """Square crops of kana-only training-corpus bubbles (word mode: only lines
    every piece of which is a trained row). Balanced data puts them in plain
    shuffled groups of g past every font layout id."""
    lines = corpus_lines(
        CORPUS_TRAIN / "boxes.jsonl", a.line_max_len if inv.has("words") else 6
    )
    if inv.restricted:
        lines = [ln for ln in lines if all(c in inv.kana for c in ln[0] if c in KANA)]
    if inv.piece_ok is not None:
        n0 = len(lines)
        lines = [ln for ln in lines if inv.piece_ok(ln[0])]
        print(
            f"corpus: {len(lines)}/{n0} lines fully covered by trained rows", flush=True
        )
    rng.shuffle(lines)
    lines = lines[: a.n_corpus]
    recs = []
    shp = None
    for j, (t, rel, box) in enumerate(lines):
        # balanced: a corpus group of g consecutive crops shares one shape
        if not a.balanced or len(recs) % a.balanced == 0:
            shp = shapes.draw(square=True)
        try:
            im = crop_bubble(CORPUS_TRAIN / rel, box, size=shp[0] if shp else 512)
        except Exception as e:  # noqa: BLE001
            print("skip", rel, e)
            continue
        fn = out / "img" / f"corpus_{j:05d}.png"
        im.save(fn)
        recs.append(_rec(fn, t, TPL_BUBBLE.format(t), "corpus", shp))
    if a.balanced:
        # corpus crops have no shared layout: the tail that does not fill a
        # group is dropped so every batch is one group
        recs = recs[: len(recs) - len(recs) % a.balanced]
        for j, r in enumerate(recs):
            r["layout_id"] = first_layout_id + j // a.balanced
    return recs


def _train_sheet(a, rng, recs, out):
    from PIL import Image

    if a.balanced:  # whole groups side by side: layout must match within a row
        by_lid: dict = {}
        for r in recs:
            by_lid.setdefault(r["layout_id"], []).append(r)
        sheet_recs = [
            r
            for lid in rng.sample(sorted(by_lid), min(10, len(by_lid)))
            for r in by_lid[lid]
        ]
    else:
        sheet_recs = rng.sample(recs, min(40, len(recs)))
    contact_sheet(
        [(Image.open(r["file"]), [r["text"], r["src"]]) for r in sheet_recs],
        out / "sheet_train.png",
        thumb=160,
        cols=a.balanced * 2 if a.balanced else 8,
    )
