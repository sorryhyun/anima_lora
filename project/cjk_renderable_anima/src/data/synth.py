"""S-line data mix (plan_synth): flat singles + scene composites + natural
phrases + random-order strings, built from a kept ``scenes_<tag>`` run.

Called by the ``data`` stage when ``--scenes <tag>`` is set; replaces the
singles×n_single + combos + corpus-crop mix. Every item is a font render:

  font     flat-canvas single unit (kana / ext non-small ×2 / kanji ×2 / word)
  phrase   covered training-corpus line on a flat canvas (``--natural_frac``)
  strings  2–4-piece random-order string, strings-arm recipe (``--strings_frac``)
  scene    a kept scene with its anchor bubble erased and JA text drawn into
           the usable region (``--scene_frac``); caption = the scene's own
           tags (``english text`` → ``japanese text``) + the trained clause.
           Composites carry the flat items' text distribution (single /
           phrase / string in the flat shares). Records carry ``box`` (the
           drawn text box, for the train stage's ``--box_weight``) and the
           scene index.

Eval groups added: ``flip`` / ``str3`` (strings) and ``phrase_held`` —
covered held-out corpus lines whose text never appears in a training item.

Sentence arm (2026-09-16, ``--scene_mix``): the composites' kinds are hard
quotas over ``single`` (one unit) / ``short`` (a ``--short_pieces`` phrase
line under the sentence floor) / ``sentence`` (>= ``--sentence_min_letters``
kana + kanji glyphs). The text is drawn uniformly among the kind's lines that
fit the bubble (the seed path took the first of 40 random lines that fit,
which made 2 092 "phrase" composites mostly 3–4-glyph interjections); a kind
that fits nothing on a scene re-picks the scene and is never demoted. Adds
eval groups ``short`` / ``short_held`` (the ``phrase`` pair reads sentences).
"""

from __future__ import annotations

import bisect
import json
import random
from collections import Counter
from pathlib import Path

from common.paths import CORPUS_HELD, CORPUS_TRAIN, OUT
from common.prompts import TPL_BUBBLE, TPL_PLAIN, TPL_SCENE_JA
from common.readers import contact_sheet
from common.render.flat import pick_font, render_string, sample_layout
from common.render.scene import region_capacity, render_into_scene
from common.text import KANJI_RE, WORD_RE

from .inventory import clean_kana_strings, corpus_lines, phrase_file_lines, pieces
from .pair import RefPool


def load_scenes(
    tags: str,
    min_ar: float = 0.0,
    min_tokens: int = 0,
    drop: str = "",
    one_bubble: str = "",
) -> list[dict]:
    """Kept scenes of every ``scenes_<tag>`` run in the comma list (s0 + a
    frame-mix run compose). ``min_ar`` (``--scene_tall_ar``) keeps only
    scenes whose headline region is at least that tall for its width —
    the sentence line's tategaki pool (user, 2026-09-16: tall bubbles
    first; regenerate when they run short). ``min_tokens``
    (``--scene_min_tokens``) drops canvases under that many DiT tokens
    (900: the 512² family only — user, 2026-09-16). ``drop`` (``--scene_drop``,
    ``tag:i,i;tag:i``) removes kept scenes by index — sl1w 332 / 957 are
    bubble-less tall regions (a hooded sketch's body, a box beside a
    figure) that the sentence quota reused 12–13 times per 400 items (user,
    2026-09-16). ``one_bubble`` (``--scene_one_bubble``, comma tags) keeps
    only scenes with one anchor box in those runs. Every scene is tagged
    ``pool`` = its run tag (indices ``i`` repeat across runs)."""
    one = {t for t in one_bubble.split(",") if t}
    unknown = one - set(tags.split(","))
    assert not unknown, f"--scene_one_bubble {sorted(unknown)}: not in --scenes {tags}"
    dropped = {}
    for part in [x for x in drop.split(";") if x]:
        tag, ids = part.split(":")
        dropped[tag] = {int(x) for x in ids.split(",") if x}
    scenes = []
    for tag in [t for t in tags.split(",") if t]:
        path = OUT / f"scenes_{tag}" / "scenes.jsonl"
        got = [json.loads(ln) for ln in path.read_text().splitlines() if ln]
        assert got, f"--scenes {tag}: no kept scenes in {path}"
        if dropped.get(tag):
            got = [s for s in got if s["i"] not in dropped[tag]]
            print(f"scenes {tag}: dropped {sorted(dropped[tag])}", flush=True)
        for s in got:
            s["pool"] = tag
        if tag in one:
            single = [s for s in got if len(s["boxes_anchor"]) == 1]
            print(
                f"scenes {tag}: {len(single)}/{len(got)} kept scenes with one anchor bubble",
                flush=True,
            )
            got = single
        if min_tokens > 0:
            big = [
                s
                for s in got
                if (s["shape"][0] // 16) * (s["shape"][1] // 16) >= min_tokens
            ]
            print(
                f"scenes {tag}: {len(big)}/{len(got)} kept scenes at >= {min_tokens} tokens",
                flush=True,
            )
            got = big
        if min_ar > 0:
            tall = [
                s
                for s in got
                if (s["region"][3] - s["region"][1])
                >= min_ar * (s["region"][2] - s["region"][0])
            ]
            print(
                f"scenes {tag}: {len(tall)}/{len(got)} kept scenes with region "
                f"AR >= {min_ar}",
                flush=True,
            )
            got = tall
        scenes += got
    assert scenes, f"--scenes {tags}: no scenes left (--scene_tall_ar {min_ar})"
    return scenes


def scene_caption(scene: dict, text: str) -> str:
    """The scene prompt with the anchor swapped for the JA text *in the frame
    the scene was drawn under* (`clause_tpl`; s0 records predate it and are
    the `reads as` frame): `english text` → `japanese text` in the tags,
    `English text reads as` → `Japanese text reads as` in the clause, every
    other frame (`She is saying "…"`, `holding a sign that reads "…"`) keeps
    its words and only the quote changes."""
    generals = [
        "japanese text" if g == "english text" else g for g in scene["generals"]
    ]
    tags = ", ".join(scene["head"] + sorted(set(generals)))
    tpl = scene.get("clause_tpl")
    if not tpl:
        return TPL_SCENE_JA.format(tags=tags, text=text)
    clause = (
        tpl.replace("English text reads as", "Japanese text reads as")
        .replace("English SFX reads as", "Japanese SFX reads as")
        .format(a=text)
    )
    return f"{tags}. {clause}"


def synth_recs(a, rng, inv, combos_eval, fonts, shapes, out, tokq) -> list[dict]:
    assert inv.piece_ok is not None, "--scenes needs the piece-coverage test"
    tok, qmap = tokq
    kana = inv.kana
    scenes = load_scenes(
        a.scenes, a.scene_tall_ar, a.scene_min_tokens, a.scene_drop, a.scene_one_bubble
    )
    assert _parse_mix(a.scene_mix) or not (a.single_scenes or a.single_max_ar), (
        "--single_scenes / --single_max_ar route the --scene_mix draw only"
    )

    # -- eval strings: flip / str3 (strings-arm recipe, only with strings in)
    # and phrase_held
    excl = set(combos_eval)
    if a.strings_frac > 0:
        flip_pairs = clean_kana_strings(
            tok, qmap, kana, rng, a.n_flip_eval, 2, excl=combos_eval
        )
        flip_eval = [x for s_ in flip_pairs for x in (s_, s_[::-1])]
        str3_eval = clean_kana_strings(
            tok, qmap, kana, rng, a.n_str3_eval, 3, excl=combos_eval
        )
        inv.evals.update(flip=flip_eval, str3=str3_eval)
        excl |= set(flip_eval) | set(str3_eval)

    # natural phrases: covered training-corpus lines (frequency-weighted list),
    # or — sentence line (2026-09-16) — the lines of ``--phrase_file`` with the
    # held set taken by *book* (``--phrase_held_books``), so phrase_held is
    # never-trained text from never-trained pages
    if a.phrase_file:
        plines = phrase_file_lines(
            Path(a.phrase_file), a.phrase_min_pieces, a.phrase_max_pieces
        )
        books = sorted({b for _t, b, _n in plines})
        hrng = random.Random(a.seed + 41)
        held_books = set(hrng.sample(books, min(a.phrase_held_books, len(books))))
        train_lines = [
            t for t, b, _n in plines if b not in held_books and inv.piece_ok(t)
        ]
        train_set = set(train_lines)
        held_lines = sorted(
            {
                t
                for t, b, _n in plines
                if b in held_books and t not in train_set and inv.piece_ok(t)
            }
        )
        src = f"{a.phrase_file} ({len(plines)} lines, {len(books)} books, held {sorted(held_books)})"
    else:
        train_lines = [
            t
            for t, _rel, _box in corpus_lines(
                CORPUS_TRAIN / "boxes.jsonl", a.line_max_len
            )
            if len(pieces(tok, qmap, t)) >= 2 and inv.piece_ok(t)
        ]
        train_set = set(train_lines)
        held_lines = []
        for t, _rel, _box in corpus_lines(CORPUS_HELD / "boxes.jsonl", a.line_max_len):
            if (
                t in train_set
                or t in held_lines
                or t in inv.evals.get("line", ())
                or len(pieces(tok, qmap, t)) < 2
                or not inv.piece_ok(t)
            ):
                continue
            held_lines.append(t)
        src = "corpus"
    prng = random.Random(a.seed + 37)
    prng.shuffle(held_lines)
    mix = _parse_mix(a.scene_mix)
    if mix.get("short") or mix.get("sentence"):
        # sentence arm: the phrase pair reads sentences, the short pair the
        # 2–5-piece lines; a line in neither kind trains in no composite
        # (a singles-only mix, Δ1, needs no phrase source)
        assert a.phrase_file, "--scene_mix short / sentence need --phrase_file"
        n_of = {t: n for t, _b, n in plines}
        lo, hi = (int(x) for x in a.short_pieces.split("-"))

        def kind_of(t):
            ls = _letters(t)
            if len(ls) >= a.sentence_min_letters and len(set(ls)) >= _SENT_DISTINCT:
                return "sentence"
            # a short item is a word or phrase, not a stretched vowel (かー /
            # ふー): at least two distinct letters, and (--short_lexical) a
            # piece that is a word — a multi-glyph kana piece or a kanji —
            # so あっ / うっ / ぎゃああ are out (user, 2026-09-16: combined
            # glyphs must make words; costs きつね-type words the tokenizer
            # splits into single glyphs, 25 % of the lines)
            if (
                lo <= n_of[t] <= hi
                and len(set(ls)) >= _SHORT_DISTINCT
                and (not a.short_lexical or _lexical(tok, qmap, t))
            ):
                return "short"
            return None

        by_kind = {"short": [], "sentence": []}
        for t in train_lines:
            k = kind_of(t)
            if k:
                by_kind[k].append(t)
        held_by = {"short": [], "sentence": []}
        for t in held_lines:
            k = kind_of(t)
            if k:
                held_by[k].append(t)
        inv.evals["phrase_held"] = sorted(held_by["sentence"][: a.n_phrase_eval])
        inv.evals["phrase"] = sorted(
            prng.sample(
                by_kind["sentence"], min(a.n_phrase_eval, len(by_kind["sentence"]))
            )
        )
        inv.evals["short_held"] = sorted(held_by["short"][: a.n_phrase_eval])
        inv.evals["short"] = sorted(
            prng.sample(by_kind["short"], min(a.n_phrase_eval, len(by_kind["short"])))
        )
        print(
            f"phrases ({src}): {len(train_lines)} covered training lines — "
            f"sentence {len(by_kind['sentence'])} (>= {a.sentence_min_letters} letters), "
            f"short {len(by_kind['short'])} ({a.short_pieces} pieces), "
            f"{len(train_lines) - len(by_kind['sentence']) - len(by_kind['short'])} in no kind; "
            f"held: sentence {len(held_by['sentence'])}, short {len(held_by['short'])}",
            flush=True,
        )
    else:
        by_kind = {}
        inv.evals["phrase_held"] = sorted(held_lines[: a.n_phrase_eval])
        if a.phrase_file:
            # trained lines too, so memorisation and generalisation read apart
            inv.evals["phrase"] = sorted(
                prng.sample(sorted(train_set), min(a.n_phrase_eval, len(train_set)))
            )
        print(
            f"phrases ({src}): {len(train_lines)} covered training lines ({len(train_set)} distinct); "
            f"phrase_held {len(inv.evals['phrase_held'])}/{len(held_lines)} never-trained held lines",
            flush=True,
        )

    # -- unit pool for singles: every trained unit repeated by its --units
    # weight, in the canonical source order (data/units.py)
    units = inv.pool()
    assert units, (
        f"--scenes: the unit pool is empty — `--units {inv.describe()}` "
        "resolved to no drawable unit"
    )

    def draw_string() -> str:
        for _ in range(200):
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
            if inv.piece_ok(s_):
                return s_
        return rng.choice(kana)

    n = a.n_items
    n_scene = round(n * a.scene_frac)
    n_phr = round(n * a.natural_frac)
    n_str = round(n * a.strings_frac)
    n_flat = n - n_scene - n_phr - n_str
    assert n_flat >= 0, "--scenes: shares exceed --n_items"
    assert kana or not n_str, (
        "--strings_frac draws random kana strings, so it needs a base "
        "inventory — no `kana` / `chars:` source is in --units"
    )
    # composites mirror the flat kind distribution; with flat 0 (composite
    # only, 2026-09-15) they are singles unless phrases / strings are in
    kinds = (["single"] * n_flat + ["phrase"] * n_phr + ["string"] * n_str) or [
        "single"
    ]
    draws = {
        "single": lambda: rng.choice(units),
        "phrase": lambda: rng.choice(train_lines),
        "string": draw_string,
    }
    # uniform among the texts of a kind that fit a bubble of `cap` glyphs
    # (sorted by length, bisect) — the seed path's first-of-40-that-fits
    # draw is what made the phrase composites interjections
    fit_pool = {
        "single": _LenPool(units, weighted=True),
        "short": _LenPool(by_kind.get("short", [])),
        "sentence": _LenPool(by_kind.get("sentence", [])),
    }
    recs: list[dict] = []

    # ΔFM (plan_synth2): every composite gets a Latin sibling by the same
    # fit, and so does every flat item (one layout, two draws)
    refs = RefPool(a.pair_ref_pool, rng) if a.pair_ref == "en" else None

    def flat(kind: str, src: str, i: int):
        s = draws[kind]()
        shp = shapes.draw()
        font = pick_font(s, fonts, rng)
        fn = out / "img" / f"{src}_{i:05d}.png"
        rec = {"text": s, "src": src, "kind": kind}
        if refs is None:
            im, bubble = render_string(
                s, font, rng, size=shp or 512, mode=a.layout, bubble_frac=a.flat_bubble
            )
        else:
            from PIL import ImageChops

            # no scene to key the pool on: 16 slots per glyph count keep the
            # sibling captions bounded (2 templates × ≤ 64 strings a length)
            ref = refs.draw(-1 - i % 16, s)
            lay = sample_layout(len(s), rng, shp or 512, a.layout, a.flat_bubble)
            im, bubble = render_string(
                s, font, rng, size=shp or 512, layout=lay, fit_text=ref
            )
            im_a, _ = render_string(
                ref, font, rng, size=shp or 512, layout=lay, fit_text=s
            )
            fn_a = fn.with_name(fn.stem + "_ref.png")
            im_a.save(fn_a)
            # the two renders differ under the glyphs only — that is the box
            box = ImageChops.difference(im, im_a).getbbox() or (0, 0, *im.size)
            rec.update(
                ref_file=str(fn_a),
                ref_text=ref,
                ref_caption=(TPL_BUBBLE if bubble else TPL_PLAIN).format(ref),
                box=list(box),
            )
        im.save(fn)
        recs.append(
            {
                "file": str(fn),
                **rec,
                "caption": (TPL_BUBBLE if bubble else TPL_PLAIN).format(s),
                **({"shape": list(shp)} if shp else {}),
            }
        )

    for i in range(n_flat):
        flat("single", "font", i)
    for i in range(n_phr):
        flat("phrase", "phrase", i)
    for i in range(n_str):
        flat("string", "strings", i)

    # -- composites ------------------------------------------------------------
    order = list(range(len(scenes)))
    rng.shuffle(order)
    n_short = 0
    kind_c: Counter = Counter()
    if mix:
        recs += _quota_composites(
            a, rng, scenes, order, mix, n_scene, fit_pool, fonts, tokq, out, refs
        )
        _scene_sheet(rng, [r for r in recs if r["src"] == "scene"], out)
        _pair_report(refs, recs)
        return recs
    for i in range(n_scene):
        sc = scenes[order[i % len(order)]]
        kind = rng.choice(kinds)
        # the region holds `cap` glyphs at --scene_min_glyph over up to
        # --scene_max_lines columns: draw texts of the kind until one is
        # short enough (cheap, no render), singles when the kind never fits;
        # the render can still refuse (font width, piece cuts)
        cap = region_capacity(
            sc["region"],
            a.scene_min_glyph,
            a.scene_fill,
            a.scene_max_lines,
            bool(a.scene_vertical),
        )
        drawn = None
        for attempt in range(6):
            text = None
            for _ in range(40):
                t = draws[kind]()
                if len(t) <= cap:
                    text = t
                    break
            if text is None:
                n_short += 1
                kind = "single"
                continue
            # lines break only between Qwen pieces — a row's unit stays whole
            cuts, off = [], 0
            for p, _row in pieces(tok, qmap, text):
                off += len(p)
                cuts.append(off)
            ref = refs.draw(sc["i"], text) if refs else None
            drawn = render_into_scene(
                sc,
                text,
                pick_font(text, fonts, rng),
                rng,
                min_glyph=a.scene_min_glyph,
                stroke=rng.random() < a.scene_stroke,
                fill_frac=a.scene_fill,
                max_lines=a.scene_max_lines,
                cuts=cuts,
                vertical_only=bool(a.scene_vertical),
                ref_text=ref,
            )
            if drawn is not None:
                break
        if drawn is None:
            continue
        kind_c[kind] += 1
        recs.append(
            _scene_record(drawn, sc, text, kind, out / "img" / f"scene_{i:05d}", ref)
        )
    print(
        f"composites: {kind_c.get('single', 0) + kind_c.get('phrase', 0) + kind_c.get('string', 0)} "
        f"over {len(scenes)} scenes ({dict(kind_c)}); {n_short} kinds fell back to single",
        flush=True,
    )
    _scene_sheet(rng, [r for r in recs if r["src"] == "scene"], out)
    _pair_report(refs, recs)
    return recs


def _scene_record(drawn, sc, text, kind, stem: Path, ref_text=None) -> dict:
    """Save a composite (and its ΔFM sibling when drawn) and build its
    record. With a sibling, ``box`` is the union of the two drawn boxes and
    the record carries ``ref_file`` / ``ref_text`` / ``ref_caption`` /
    ``ref_box``."""
    im, box = drawn[:2]
    W, H = im.size
    assert [W, H] == list(sc["shape"]), (
        f"scene {sc['i']}: image {W}x{H} vs {sc['shape']}"
    )
    fn = stem.with_suffix(".png")
    im.save(fn)
    rec = {
        "file": str(fn),
        "text": text,
        "caption": scene_caption(sc, text),
        "src": "scene",
        "kind": kind,
        "shape": [W, H],
        "box": box,
        "scene": sc["i"],
        "scene_pool": sc.get("pool"),
    }
    if len(drawn) == 4:
        im_a, box_a = drawn[2:]
        fn_a = stem.with_name(stem.name + "_ref.png")
        im_a.save(fn_a)
        rec.update(
            ref_file=str(fn_a),
            ref_text=ref_text,
            ref_caption=scene_caption(sc, ref_text),
            ref_box=box_a,
            box=[
                min(box[0], box_a[0]),
                min(box[1], box_a[1]),
                max(box[2], box_a[2]),
                max(box[3], box_a[3]),
            ],
        )
    return rec


def _pair_report(refs, recs):
    if refs is None:
        return
    paired = [r for r in recs if "ref_file" in r]
    caps = len({r["ref_caption"] for r in paired})
    n_flat = sum(r["src"] != "scene" for r in paired)
    print(
        f"pairs: {len(paired) - n_flat} composites + {n_flat} flat items with a Latin sibling, "
        f"{refs.n_strings()} reference strings, {caps} distinct reference captions",
        flush=True,
    )


def _parse_mix(spec: str) -> dict[str, float]:
    """``single=0.1,short=0.5,sentence=0.4`` → shares (must sum to 1)."""
    if not spec:
        return {}
    mix = {}
    for part in spec.split(","):
        k, v = part.split("=")
        assert k in ("single", "short", "sentence"), f"--scene_mix: unknown kind {k}"
        mix[k] = float(v)
    assert abs(sum(mix.values()) - 1.0) < 1e-6, (
        f"--scene_mix shares sum to {sum(mix.values())}"
    )
    return mix


def _letters(t: str) -> list[str]:
    """Kana + kanji glyphs of ``t`` — the sentence floor's unit (punctuation,
    digits, Latin and the prolonged-sound mark do not count)."""
    return [c for c in t if "ぁ" <= c <= "ゖ" or "ァ" <= c <= "ヺ" or KANJI_RE.match(c)]


# the length floor alone let ハハハハハハ / おやおやおや / うわああああ through
# as sentences (smoke, 2026-09-16): a sentence has >= 4 distinct letters, a
# short item >= 2
_SENT_DISTINCT = 4
_SHORT_DISTINCT = 2


def _lexical(tok, qmap, t: str) -> bool:
    """Some Qwen piece of ``t`` is a word: >= 2 glyphs matching WORD_RE, or
    carries a kanji."""
    return any(
        (len(p) >= 2 and WORD_RE.match(p)) or any(KANJI_RE.match(c) for c in p)
        for p, _row in pieces(tok, qmap, t)
    )


class _LenPool:
    """Texts sorted by glyph length; ``draw(rng, cap)`` picks a length
    uniformly among the lengths present at most ``cap`` glyphs, then a text
    of that length (``None`` when nothing fits) — so the 2-glyph lines, the
    most numerous, do not dominate the short kind."""

    def __init__(self, texts, weighted: bool = False):
        # weighted: keep repeats (the singles pool repeats a unit per its
        # --units weight) so an as-is draw honours them
        self.texts = sorted(
            texts if weighted else set(texts), key=lambda t: (len(t), t)
        )
        self.lens = [len(t) for t in self.texts]
        self.starts = {}
        for i, n in enumerate(self.lens):
            self.starts.setdefault(n, i)
        self.lengths = sorted(self.starts)

    def draw(self, rng, cap: int, by_length: bool = True):
        if not by_length:
            # the singles pool carries the --units weights: draw it as is
            n = bisect.bisect_right(self.lens, cap)
            return self.texts[rng.randrange(n)] if n else None
        k = bisect.bisect_right(self.lengths, cap)
        if not k:
            return None
        n = self.lengths[rng.randrange(k)]
        lo, hi = self.starts[n], bisect.bisect_right(self.lens, n)
        return self.texts[rng.randrange(lo, hi)]


def _quota_composites(
    a, rng, scenes, order, mix, n_scene, fit_pool, fonts, tokq, out, refs=None
):
    """The sentence arm's composites: ``n_scene`` items whose kinds are the
    ``--scene_mix`` shares as hard counts (rounding to the largest share),
    in a shuffled order. Per item the **text comes first** (a length
    uniformly among the kind's lengths that at least ``_MIN_FIT_SCENES``
    scenes hold, then a text of that length; singles as the weighted pool),
    then a scene among those whose capacity holds it, weighted by
    ``1 / (1 + uses)`` so the tall bubbles are not one scene — scene-first
    with a text that fits piled the short kind at 2 glyphs, the one-column
    capacity of most bubbles, and plain uniform-among-fitting put 124 of 400
    items on one scene. Up to ``_SCENE_TRIES`` fitting scenes per text and
    3 texts per item; then the item is a recorded miss, never another
    kind."""
    tok, qmap = tokq
    counts = {k: int(n_scene * f) for k, f in mix.items()}
    top = max(mix, key=mix.get)
    counts[top] += n_scene - sum(counts.values())
    todo = [k for k, c in counts.items() for _ in range(c)]
    rng.shuffle(todo)
    assert all(fit_pool[k].texts for k in counts if counts[k]), (
        f"--scene_mix: a kind with a share has no texts: "
        f"{ {k: len(fit_pool[k].texts) for k in counts} }"
    )
    # columns per kind: a short item is one column (user, 2026-09-16: 2–5
    # tokens read as a single vertical line), sentences wrap
    lines_of = {"single": 1, "short": a.short_max_lines, "sentence": a.scene_max_lines}
    # sentences may draw smaller and fill more of the bubble (user,
    # 2026-09-16: 20 px / 0.9 so a 6-glyph line is one column on about half
    # the sl1w bubbles; at 28 px / 0.7 two of 276 hold it)
    glyph_of = {k: a.scene_min_glyph for k in lines_of}
    fill_of = {k: a.scene_fill for k in lines_of}
    if a.sentence_min_glyph:
        glyph_of["sentence"] = a.sentence_min_glyph
    if a.sentence_fill:
        fill_of["sentence"] = a.sentence_fill
    caps = {
        k: [
            region_capacity(
                sc["region"],
                glyph_of[k],
                fill_of[k],
                lines_of[k],
                bool(a.scene_vertical),
            )
            for sc in scenes
        ]
        for k in lines_of
    }
    # one-column capacity per kind: a text goes to the scenes that hold it
    # in one column whenever >= _MIN_FIT_SCENES do (user, 2026-09-16: a
    # 6-glyph line is one column), else to any scene that holds it wrapped
    caps1 = {
        k: [
            region_capacity(
                sc["region"], glyph_of[k], fill_of[k], 1, bool(a.scene_vertical)
            )
            for sc in scenes
        ]
        for k in lines_of
    }
    # the longest text a kind may draw: the capacity that _MIN_FIT_SCENES
    # scenes reach at its column count
    cap_of = {
        k: sorted(cs, reverse=True)[min(_MIN_FIT_SCENES, len(cs)) - 1]
        for k, cs in caps.items()
    }
    # plan_synth2 Δ0.9: a one-glyph text only on --single_scenes pools and
    # regions no longer than --single_max_ar for their short side — a lone
    # glyph in a sentence-sized strip floats on blank canvas (sl1w 192)
    single_pools = {t for t in a.single_scenes.split(",") if t}

    def single_ok(sc) -> bool:
        if single_pools and sc["pool"] not in single_pools:
            return False
        if a.single_max_ar > 0:
            w = sc["region"][2] - sc["region"][0]
            h = sc["region"][3] - sc["region"][1]
            return max(w, h) <= a.single_max_ar * max(1, min(w, h))
        return True

    single_idx = {j for j, sc in enumerate(scenes) if single_ok(sc)}
    print(
        f"one-glyph texts: {len(single_idx)}/{len(scenes)} scenes eligible "
        f"(pools {sorted(single_pools) or 'all'}, max AR {a.single_max_ar or 'off'})",
        flush=True,
    )
    assert single_idx, "--single_scenes / --single_max_ar leave no scene for a glyph"
    print(
        f"quota caps (glyphs held by >= {_MIN_FIT_SCENES} scenes): "
        + ", ".join(
            f"{k} {c} ({lines_of[k]} col, {glyph_of[k]} px, fill {fill_of[k]})"
            for k, c in cap_of.items()
        ),
        flush=True,
    )
    recs, miss = [], Counter()
    used: Counter = Counter()
    for i, kind in enumerate(todo):
        max_lines = lines_of[kind]
        drawn, text = None, None
        for _ in range(3):
            text = fit_pool[kind].draw(rng, cap_of[kind], by_length=kind != "single")
            if text is None:
                break
            fitting = [j for j, c in enumerate(caps1[kind]) if c >= len(text)]
            if len(fitting) < _MIN_FIT_SCENES:
                fitting = [j for j, c in enumerate(caps[kind]) if c >= len(text)]
            if len(text) == 1:
                fitting = [j for j in fitting if j in single_idx]
            cuts, off = [], 0
            for p, _row in pieces(tok, qmap, text):
                off += len(p)
                cuts.append(off)
            tries, pool = [], list(fitting)
            while pool and len(tries) < _SCENE_TRIES:
                j = rng.choices(
                    pool, weights=[1.0 / (1 + used[x]) for x in pool]
                )[0]
                pool.remove(j)
                tries.append(j)
            for j in tries:
                sc = scenes[j]
                ref = refs.draw(sc["i"], text) if refs else None
                drawn = render_into_scene(
                    sc,
                    text,
                    pick_font(text, fonts, rng),
                    rng,
                    min_glyph=glyph_of[kind],
                    stroke=rng.random() < a.scene_stroke,
                    fill_frac=fill_of[kind],
                    max_lines=max_lines,
                    cuts=cuts,
                    vertical_only=bool(a.scene_vertical),
                    fewest_lines=bool(a.scene_fewest_lines),
                    ref_text=ref,
                )
                if drawn is not None:
                    break
            if drawn is not None:
                break
        if drawn is None:
            miss[kind] += 1
            continue
        used[j] += 1
        recs.append(
            _scene_record(drawn, sc, text, kind, out / "img" / f"scene_{i:05d}", ref)
        )
    got = Counter(r["kind"] for r in recs)
    # columns drawn, from the box: width / height × glyphs ≈ 1 for one
    # column, ≈ 4 for two
    one_col = {
        k: sum(
            1
            for r in recs
            if r["kind"] == k
            and len(r["text"]) > 1
            and (r["box"][2] - r["box"][0]) * len(r["text"])
            < 2.2 * (r["box"][3] - r["box"][1])
        )
        for k in counts
    }
    print(f"  one-column items per kind (multi-glyph): {one_col}", flush=True)
    glyphs = {
        k: sorted(Counter(len(r["text"]) for r in recs if r["kind"] == k).items())
        for k in counts
    }
    print(
        f"composites (quota): {len(recs)}/{n_scene} over {len(used)}/{len(scenes)} scenes "
        f"(busiest scene {max(used.values()) if used else 0} items); "
        f"planned {dict(counts)}, drawn {dict(got)}, missed {dict(miss)}",
        flush=True,
    )
    for k, hist in glyphs.items():
        print(f"  {k} glyph lengths: {hist}", flush=True)
    return recs


_SCENE_TRIES = 8
_MIN_FIT_SCENES = 10


def _scene_sheet(rng, recs, out):
    from PIL import Image, ImageDraw

    sample = rng.sample(recs, min(40, len(recs)))
    tiles = []
    for r in sample:
        im = Image.open(r["file"]).convert("RGB")
        ImageDraw.Draw(im).rectangle(r["box"], outline=(0, 255, 0), width=2)
        tiles.append((im, [r["text"], r["kind"]]))
    contact_sheet(tiles, out / "sheet_scene.png", thumb=192, cols=8)
    pairs = [r for r in sample if "ref_file" in r]
    if pairs:
        # ΔFM: item beside its sibling, union box on both — the read before
        # launch is that the two differ by the glyphs alone
        tiles = []
        for r in pairs[:20]:
            for f, t in ((r["file"], r["text"]), (r["ref_file"], r["ref_text"])):
                im = Image.open(f).convert("RGB")
                ImageDraw.Draw(im).rectangle(r["box"], outline=(0, 255, 0), width=2)
                tiles.append((im, [t, r["kind"]]))
        contact_sheet(tiles, out / "sheet_scene_pair.png", thumb=192, cols=8)
