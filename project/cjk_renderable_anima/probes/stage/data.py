"""Stage ``data`` — build the glyph training set and the eval prompt set.

Writes ``<data_dir>/{img/, train.jsonl, eval.json, sheet_train.png}`` (+
``words.json`` / ``kanji.json`` when those inventories are on).

Bit-identity contract: the main ``rng`` (seed 0) is consumed in a fixed order
— eval singles, eval combos, held corpus shuffle, font items, renders, corpus
shuffle, sheet draw. Every later lever draws from its own stream (shapes
seed+17, kana_ext seed+19, kanji seed+23, words seed+13), so switching a lever
off rebuilds the older data dirs identically. Keep it that way.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from wake.common import (
    CORPUS_HELD,
    CORPUS_TRAIN,
    EN_WORDS,
    HIRA,
    KANA,
    KANA_EXT,
    KANA_EXT_HIRA,
    KANA_EXT_KATA,
    KATA,
    TPL_BUBBLE,
    TPL_EN,
    TPL_PLAIN,
    data_dir,
    parse_shapes,
)
from wake.inventory import (
    clean_kana_strings,
    corpus_lines,
    kanji_inventory,
    pieces,
    qwen_pieces,
    word_inventory,
)
from wake.readers import contact_sheet
from wake.render import crop_bubble, find_fonts, pick_font, render_string, sample_layout

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
    "single_extra",
    "flip",
    "str3",
    "phrase",
    "phrase_held",
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


@dataclass
class Inventory:
    """Everything a training item may contain, and the eval strings drawn."""

    kana: list
    kana_ext: list = field(default_factory=list)
    kanji: list = field(default_factory=list)
    words: list = field(default_factory=list)
    words_held: list = field(default_factory=list)
    words_train: list = field(default_factory=list)
    # --phrase_pieces: rows a phrase file needs beyond the singles inventory
    # (trained through the phrases only — never drawn as singles / evals)
    phrase_pieces: list = field(default_factory=list)
    # --extra_units: pieces drawn as singles like kana (punctuation arm,
    # user 2026-09-16: 、。ー！？ and small っ ッ as single-letter addresses)
    extra: list = field(default_factory=list)
    # word mode: a string is usable only when every piece is a trained row
    piece_ok: Callable[[str], bool] | None = None
    evals: dict = field(default_factory=dict)  # group → [text]


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
    rng = random.Random(0)
    out = data_dir(a)
    (out / "img").mkdir(parents=True, exist_ok=True)
    fonts = find_fonts()
    print(f"fonts: {len(fonts)}", flush=True)
    # mixed shapes (2026-09-14): every font item draws its canvas (W, H) from
    # --shapes; corpus crops are square, so they draw from the pool's squares
    shapes = ShapePool(a.shapes, a.seed)
    # --scenes needs the piece map for piece_ok even at --words 0 (micro arms)
    tokq = qwen_pieces() if (a.kanji or a.words or a.scenes) else None

    inv = Inventory(kana=list(a.only_chars) if a.only_chars else list(KANA))
    # eval strings first so the training pool can exclude the combos
    combos_eval, n_possible = _eval_strings(a, rng, inv)
    _extra_singles(a, out, tokq, inv)
    if a.words or a.scenes:
        _word_set(a, out, tokq, inv)

    n_target = min(a.n_combo, 50 * (n_possible - len(combos_eval)))
    if a.scenes:
        # S line (plan_synth): the whole mix comes from synth.py; no corpus crops
        from .synth import synth_recs

        recs = synth_recs(a, rng, inv, combos_eval, fonts, shapes, out, tokq)
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
    if not a.scenes:
        recs += _corpus_recs(a, rng, inv, shapes, out, first_layout_id=len(recs))

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


# ----------------------------------------------------------------------------
# inventories and eval strings


def _eval_strings(a, rng, inv: Inventory):
    """Eval singles / combos / held corpus lines (main rng) and the EN control.
    Returns ``(combos_eval, number of possible 2–3 kana strings)``."""
    kana = inv.kana
    if a.only_chars:
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
    if a.only_chars:
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


def _extra_singles(a, out, tokq, inv: Inventory):
    """P0b inventory extensions, singles only: the N most frequent single-row
    corpus kanji (``--kanji``) and voiced / small kana (``--kana_ext``)."""
    if a.kanji:
        assert not a.only_chars and not a.balanced, "--kanji: full inventory"
        kfreq = kanji_inventory(*tokq, a.kanji)
        inv.kanji = [c for c, _ in kfreq]
        krng = random.Random(a.seed + 23)
        inv.evals["single_kanji"] = krng.sample(inv.kanji, min(18, len(inv.kanji)))
        (out / "kanji.json").write_text(json.dumps(kfreq, ensure_ascii=False))
        print(
            f"kanji: {len(inv.kanji)} singles (last {kfreq[-1][0]}:{kfreq[-1][1]}); "
            f"{''.join(inv.kanji)}",
            flush=True,
        )
    if a.kana_ext:
        assert not a.only_chars and not a.balanced, (
            "--kana_ext: full inventory, unbalanced"
        )
        inv.kana_ext = list(KANA_EXT)
        erng = random.Random(a.seed + 19)
        inv.evals["single_ext"] = erng.sample(list(KANA_EXT_HIRA), 12) + erng.sample(
            list(KANA_EXT_KATA), 6
        )
    if a.extra_units:
        # punctuation arm (2026-09-16): every unit must be one Qwen piece
        # with an ext row (、 。 ・ ー ～ ！ ？ 「 」 ！！ ・・・ っ ッ …); all of
        # them form eval group single_extra — read on the sheets, since the
        # readers' norm() strips punctuation before matching
        inv.extra = [u for u in a.extra_units.split(",") if u]
        tok, qmap = tokq
        bad = [u for u in inv.extra if len(pieces(tok, qmap, u)) != 1]
        assert not bad, f"--extra_units: not one Qwen piece: {bad}"
        norow = [u for u in inv.extra if pieces(tok, qmap, u)[0][1] is None]
        assert not norow, f"--extra_units: pretrained piece, no ext row: {norow}"
        inv.evals["single_extra"] = inv.extra[:18]
        print(
            f"extra units: {len(inv.extra)} singles {' '.join(inv.extra)}", flush=True
        )


def _word_set(a, out, tokq, inv: Inventory):
    """2026-09-14 word addresses: the inventory gains the corpus's most frequent
    single-piece words (each its own pack row), K of them held out; corpus
    lines are kept only when every piece is a trained row (kana single or
    word) so ``line`` evaluates addresses *in sequence*, not coverage."""
    tok, qmap = tokq
    freq = word_inventory(tok, qmap, a.words, a.word_min_len)
    inv.words = [w for w, _ in freq]
    wrng = random.Random(a.seed + 13)
    inv.words_held = (
        sorted(wrng.sample(inv.words, a.held_out_words)) if a.held_out_words else []
    )
    inv.words_train = [w for w in inv.words if w not in inv.words_held]
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

        from wake.inventory import phrase_file_lines, phrase_pieces

        plines = phrase_file_lines(
            Path(a.phrase_file), a.phrase_min_pieces, a.phrase_max_pieces
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
        assert inv.piece_ok is not None, "--strings_only needs --words"
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
    for ch in inv.kana_ext + inv.kanji:  # P0b: extended kana / kanji, singles only
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
    lines = corpus_lines(CORPUS_TRAIN / "boxes.jsonl", a.line_max_len if a.words else 6)
    if a.only_chars:
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
