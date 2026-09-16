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
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from wake.common import (
    CORPUS_HELD,
    CORPUS_TRAIN,
    KANA_SMALL,
    OUT,
    TPL_BUBBLE,
    TPL_PLAIN,
    TPL_SCENE_JA,
)
from wake.inventory import clean_kana_strings, corpus_lines, phrase_file_lines, pieces
from wake.readers import contact_sheet
from wake.render import pick_font, region_capacity, render_into_scene, render_string


def load_scenes(tags: str, min_ar: float = 0.0) -> list[dict]:
    """Kept scenes of every ``scenes_<tag>`` run in the comma list (s0 + a
    frame-mix run compose). ``min_ar`` (``--scene_tall_ar``) keeps only
    scenes whose headline region is at least that tall for its width —
    the sentence line's tategaki pool (user, 2026-09-16: tall bubbles
    first; regenerate when they run short)."""
    scenes = []
    for tag in [t for t in tags.split(",") if t]:
        path = OUT / f"scenes_{tag}" / "scenes.jsonl"
        got = [json.loads(ln) for ln in path.read_text().splitlines() if ln]
        assert got, f"--scenes {tag}: no kept scenes in {path}"
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
    assert inv.piece_ok is not None, "--scenes needs --words (piece coverage)"
    tok, qmap = tokq
    kana = inv.kana
    scenes = load_scenes(a.scenes, a.scene_tall_ar)

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

    # -- unit pool for singles -------------------------------------------------
    ext_ns = [c for c in inv.kana_ext if c not in KANA_SMALL]
    units = (
        list(kana)
        + ext_ns * 2
        + list(inv.kanji) * 2
        + list(inv.words_train)
        + list(inv.extra) * 2
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
    recs: list[dict] = []

    def flat(kind: str, src: str, i: int):
        s = draws[kind]()
        shp = shapes.draw()
        im, bubble = render_string(
            s,
            pick_font(s, fonts, rng),
            rng,
            size=shp or 512,
            mode=a.layout,
            bubble_frac=a.flat_bubble,
        )
        fn = out / "img" / f"{src}_{i:05d}.png"
        im.save(fn)
        recs.append(
            {
                "file": str(fn),
                "text": s,
                "caption": (TPL_BUBBLE if bubble else TPL_PLAIN).format(s),
                "src": src,
                "kind": kind,
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
    for i in range(n_scene):
        sc = scenes[order[i % len(order)]]
        kind = rng.choice(kinds)
        # the region holds `cap` glyphs at --scene_min_glyph over up to
        # --scene_max_lines columns: draw texts of the kind until one is
        # short enough (cheap, no render), singles when the kind never fits;
        # the render can still refuse (font width, piece cuts)
        cap = region_capacity(
            sc["region"], a.scene_min_glyph, a.scene_fill, a.scene_max_lines
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
            )
            if drawn is not None:
                break
        if drawn is None:
            continue
        im, box = drawn
        W, H = im.size
        assert [W, H] == list(sc["shape"]), (
            f"scene {sc['i']}: image {W}x{H} vs {sc['shape']}"
        )
        fn = out / "img" / f"scene_{i:05d}.png"
        im.save(fn)
        kind_c[kind] += 1
        recs.append(
            {
                "file": str(fn),
                "text": text,
                "caption": scene_caption(sc, text),
                "src": "scene",
                "kind": kind,
                "shape": [W, H],
                "box": box,
                "scene": sc["i"],
            }
        )
    print(
        f"composites: {kind_c.get('single', 0) + kind_c.get('phrase', 0) + kind_c.get('string', 0)} "
        f"over {len(scenes)} scenes ({dict(kind_c)}); {n_short} kinds fell back to single",
        flush=True,
    )
    _scene_sheet(rng, [r for r in recs if r["src"] == "scene"], out)
    return recs


def _scene_sheet(rng, recs, out):
    from PIL import Image, ImageDraw

    sample = rng.sample(recs, min(40, len(recs)))
    tiles = []
    for r in sample:
        im = Image.open(r["file"]).convert("RGB")
        ImageDraw.Draw(im).rectangle(r["box"], outline=(0, 255, 0), width=2)
        tiles.append((im, [r["text"], r["kind"]]))
    contact_sheet(tiles, out / "sheet_scene.png", thumb=192, cols=8)
