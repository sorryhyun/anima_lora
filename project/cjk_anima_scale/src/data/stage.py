"""The old ``data`` stage's resolvers — what ``cjk_scale/recipes.py`` builds a
run's pools with, and ``cjk_scale/builder.py``'s ink stats. The stage itself
(``stage_data``) is gone (pruned 2026-09-25; the line's builder replaced it).

What goes into the table is the vocab specs (``data/vocabs.py``). This module
resolves those sources against the corpus and the tokenizer.

Bit-identity contract: the main ``rng`` is consumed in a fixed order (eval
singles, eval combos, held corpus shuffle); every later lever draws from its
own stream (shapes seed+17, kana_ext seed+19, kanji seed+23, words seed+13).
The sources are resolved in the canonical order of ``data/vocabs.py``, never
the typed order, for the same reason. Keep it that way.
"""

from __future__ import annotations

import json
import random

from common.paths import CORPUS_HELD
from common.prompts import EN_WORDS
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
    corpus_lines,
    kanji_inventory,
    pieces,
    small_digraphs,
    word_inventory,
)
from .vocabs import SMALL_PER, Inventory, parse_vocabs

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
        vocabs = r.get("units") or [r["text"]]  # the on-disk record key stays
        glyphs = max(1, sum(glyph_count(u) for u in vocabs))
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
    """``a.vocabs`` (vocab specs) → the Inventory, with the base (``kana`` /
    ``chars:``) sources resolved. The rest need the tokenizer and are done later."""
    inv = Inventory(sources=parse_vocabs(a.vocabs))
    for s in inv.sources:
        if s.kind == "kana":
            s.vocabs = list(KANA)
        elif s.kind != "chars":
            continue
        inv.kana += [c for c in s.vocabs if c not in inv.kana]
    # a hand-picked base (chars: with no kana beside it): the eval singles are
    # the base itself and corpus lines are filtered down to it
    inv.restricted = inv.has("chars") and not inv.has("kana")
    print(
        f"vocabs: {inv.describe()} → base {len(inv.kana)}"
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
    (literal ext-row vocabs — the punctuation arm).

    Each keeps its own rng stream (kana_ext seed+19, kanji seed+23), so adding
    or dropping one leaves the other's draw untouched.
    """
    if inv.has("kana_ext"):
        assert not inv.restricted and not a.balanced, (
            "vocab spec kana_ext: full inventory, unbalanced"
        )
        src = inv.source("kana_ext")
        src.vocabs = list(KANA_EXT)
        inv.kana_ext = src.vocabs
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
            "vocab spec small: hosts are the full kana inventory, unbalanced"
        )
        inv.small_of = small_digraphs(*tokq, inv.kana + inv.kana_ext, SMALL_PER)
        ssrc.vocabs = [d for ds in inv.small_of.values() for d in dict.fromkeys(ds)]
        inv.evals["single_small"] = [ds[0] for ds in inv.small_of.values()]
        (out / "small.json").write_text(json.dumps(inv.small_of, ensure_ascii=False))
        print(
            f"small kana: {len(inv.small_of)} rows in {len(ssrc.vocabs)} digraphs; "
            + " ".join(
                f"{k}:{'/'.join(dict.fromkeys(v))}" for k, v in inv.small_of.items()
            ),
            flush=True,
        )
    ksrc = inv.source("kanji")
    if ksrc and ksrc.n:
        assert not inv.restricted and not a.balanced, (
            "vocab spec kanji: full inventory, unbalanced"
        )
        ksrc.freq = kanji_inventory(*tokq, ksrc.n)
        ksrc.vocabs = [c for c, _ in ksrc.freq]
        inv.kanji = ksrc.vocabs
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
        # punctuation arm (2026-09-16): every vocab must be one Qwen piece with
        # an ext row (、 。 ・ ー ～ ！ ？ 「 」 ！！ ・・・ っ ッ …); all of them form
        # eval group single_extra — read on the sheets, since the readers'
        # norm() strips punctuation before matching
        tok, qmap = tokq
        bad = [u for u in lsrc.vocabs if len(pieces(tok, qmap, u)) != 1]
        assert not bad, f"vocab spec list: not one Qwen piece: {bad}"
        norow = [u for u in lsrc.vocabs if pieces(tok, qmap, u)[0][1] is None]
        assert not norow, f"vocab spec list: pretrained piece, no ext row: {norow}"
        inv.extra = lsrc.vocabs
        inv.evals["single_extra"] = inv.extra[:18]
        print(
            f"extra vocabs: {len(inv.extra)} singles {' '.join(inv.extra)}", flush=True
        )


def _word_set(a, out, tokq, inv: Inventory):
    """2026-09-14 word addresses (the ``words:N/held=K`` spec): the inventory
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
        src.freq, src.vocabs, src.held_vocabs = freq, inv.words, inv.words_held
    kana_rows = {
        p
        for c in inv.kana + inv.kana_ext + inv.kanji + inv.extra
        for p, row in pieces(tok, qmap, c)
        if row is not None
    }
    trained_pieces = kana_rows | set(inv.words_train)

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
