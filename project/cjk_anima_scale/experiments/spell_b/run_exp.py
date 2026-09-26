#!/usr/bin/env python
"""spell_b — five single rows trained on in-line strings, read spelled (2026-09-26)

The seed's singles あ り が と う render natively (native_spell/ on
``rows_step1_0921_merged``: 3–14 / 16 official, contained 9–16), but the
spelled caption ``"あ り が と う"`` (five single ext ids — a half-width space
is dropped by the pack encoder) renders one big あ on a single canvas, 0 / 32:
the rows carry the 0.7–0.9 single-canvas layout they were trained on. This
arm trains the same five rows on **multi-glyph lines** instead — each item a
real word made of the five glyphs, rendered unspaced in a bubble, captioned
spaced so every glyph is its single row — in the piece kind's bands
(0.5–0.7 at ≈ 40 px, 0.3–0.5 at 12–24 px; no 0.7–0.9 group), every other
row frozen at the seed. ありがとう is held out: り→が and が→と never occur
in a training word.

Read: native (en + swap, 8 prompts × 2 seeds) on the trained rows — the
held-out spelled ``あ り が と う``, three spelled training words, the five
singles alone — against the seed floor's ``native_spell/`` cache (the keys
it lacks render once). No other ruler.

legs: data (CPU), train (GPU), eval (GPU) — the GPU legs through
``make daemon-run``. ``--dry_run`` prints the table, the words and their
spelled encodings without rendering.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, SEED_ROWS, bootstrap  # noqa: E402

bootstrap()

from bench._common import make_run_dir, write_result  # noqa: E402

NAME = "run0926_spell_b"
GLYPHS = "ありがとう"
# real words spelled from the five glyphs; ありがとう (and ありがと) held out
WORDS = ("あり", "あと", "あう", "うり", "とり", "とう", "がり", "あがり")
HELD = "ありがとう"
READ_WORDS = ("あり", "とう", "あがり")  # spelled training words the eval reads


def spell(s: str) -> str:
    return " ".join(s)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument(
        "--legs", nargs="+", default=["data"], choices=["data", "train", "eval"]
    )
    p.add_argument("--workers", type=int, help="data: render processes")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def arm_rc():
    from cjk_scale.config import RunConfig

    return RunConfig(
        name=NAME, path=Path(__file__), vocabs=(f"chars:{GLYPHS}",), read=()
    )


def scene_spelled(pools, rng, p: dict):
    """A training word in one bubble, unspaced on the image, spaced in the
    caption — the scene_piece draw with the word list as its pool."""
    from cjk_scale.recipes import _draw_scene, _target

    word = rng.choice(WORDS)
    f = p.get("fill", [0.7, 1.0])
    lo, hi = f if isinstance(f, list) else (f, f)
    fill = rng.uniform(float(lo), float(hi))
    item = _draw_scene(
        pools,
        rng,
        word,
        min_glyph=int(p.get("min_glyph", 28)),
        fill=fill,
        max_lines=1,
        target_px=_target(rng, p),
        fill_max=fill,
        fill_min=float(p.get("fill_min", 0)),
    )
    if item is None:
        return None
    quoted = f'"{word}"'
    assert item.caption.count(quoted) == 1, item.caption
    item.caption = item.caption.replace(quoted, f'"{spell(word)}"')
    return item


def spell_table() -> tuple:
    """The two piece band groups' ``scene_piece`` tiers, brought in by the
    single kind and drawing the word list (``builder.TABLE``'s b0709 is
    dropped: no single-canvas draw)."""
    from cjk_scale.builder import TABLE, Group, Tier

    tiers = {
        g.name: next(t for t in g.tiers if t.recipe == "scene_piece")
        for g in TABLE
        if g.kind == "piece"
    }
    return tuple(
        Group(
            name, "single", band, 0.5, (Tier("scene_spelled", 1.0, tiers[name].params),)
        )
        for name, band in (("b0507", (0.5, 0.7)), ("b0305", (0.3, 0.5)))
    )


def check_spelling() -> dict:
    """Every spelled word / the held-out string encodes to its glyphs' single
    ext ids and nothing else (the pack encoder, as the TE cache sees it)."""
    from transformers import AutoTokenizer

    from library.anima import ext_vocab
    from library.anima.ext_vocab import T5_TABLE_SIZE, HybridT5Encoder
    from library.env import resolve_under_home
    from library.anima.vocab_pack import resolve_pack_prefix
    import os

    t5 = AutoTokenizer.from_pretrained(
        resolve_under_home("library/anima/configs/t5_old")
    )
    qw = AutoTokenizer.from_pretrained(
        resolve_under_home("library/anima/configs/qwen3_06b")
    )
    prefix = resolve_pack_prefix(os.environ["ANIMA_VOCAB_PACK"])
    _, mapping = ext_vocab.load_ext_assets(prefix)
    enc = HybridT5Encoder.from_mapping(t5, qw, mapping)

    def ext(text):
        ids, mask = enc.encode(f'Japanese text reads as "{text}".', 512)
        return [
            i - T5_TABLE_SIZE for i, m in zip(ids, mask) if m and i >= T5_TABLE_SIZE
        ]

    single = {c: ext(c) for c in GLYPHS}
    assert all(len(v) == 1 for v in single.values()), single
    out = {}
    for w in (*WORDS, HELD):
        got = ext(spell(w))
        want = [single[c][0] for c in w]
        assert got == want, (w, got, want)
        out[w] = got
    print(f"spelled encodings: {json.dumps(out, ensure_ascii=False)}", flush=True)
    return out


def native_read(rc, arm: str, chars: list, tag: str) -> Path:
    from cjk_scale.eval import arm_out, probe_args
    from stages import run as run_stage

    a = probe_args(
        rc, arm, ["native"], ["--eval_tag", tag, "--native_chars", ",".join(chars)]
    )
    run_stage("native", a)
    return arm_out(rc, arm) / f"native_{tag}" / "native_reads.json"


def tally(path: Path, chars) -> dict:
    from common.readers import norm

    out: dict = {}
    for m in json.loads(path.read_text("utf-8")):
        if m["text"] not in chars:
            continue
        k = f"{m['text']}|{m['clause']}"
        t = norm(m["text"])
        c = out.setdefault(k, {"n": 0, "official": 0, "loose": 0, "contained": 0})
        c["n"] += 1
        c["official"] += bool(m.get("hit_sfx")) and bool(m.get("hit_vl"))
        c["loose"] += bool(m.get("hit_sfx")) or bool(m.get("hit_vl"))
        c["contained"] += any(
            t in norm(r.get("sfx") or "") or t in norm(r.get("vl") or "")
            for r in m.get("reads", [])
        )
    for k, c in out.items():
        print(
            f"  {k:<14} official {c['official']:>2}  loose {c['loose']:>2}  "
            f"contained {c['contained']:>2} / {c['n']}",
            flush=True,
        )
    return out


def main():
    args = parse_args()
    from cjk_scale import recipes

    recipes.RECIPES["scene_spelled"] = scene_spelled
    rc = arm_rc()
    table = spell_table()
    for g in table:
        print(
            f"{g.name} σ {g.band}: {[(t.recipe, t.params) for t in g.tiers]}",
            flush=True,
        )
    print(f"words {WORDS}, held {HELD}", flush=True)
    metrics: dict = {"words": list(WORDS), "held": HELD, "spelled": check_spelling()}
    if args.dry_run:
        return
    run_dir = make_run_dir(
        "spell_b", label=args.label, root=LINE / "experiments" / "spell_b" / "results"
    )
    if "data" in args.legs:
        from cjk_scale.builder import build

        build(rc, workers=args.workers, table=table)
    if "train" in args.legs:
        from cjk_scale.train import train

        train(rc)
    if "eval" in args.legs:
        from cjk_scale.eval import TRAINED_ARM, ensure_native_floor
        from cjk_scale.paths import floor_dir

        chars = [spell(HELD), *(spell(w) for w in READ_WORDS), *GLYPHS]
        print("floor (the seed dir's native_spell/ cache):", flush=True)
        ensure_native_floor(rc, "native_spell", chars, "en,swap")
        metrics["floor"] = tally(
            floor_dir() / "native_spell" / "native_reads.json", chars
        )
        print("trained:", flush=True)
        metrics["trained"] = tally(native_read(rc, TRAINED_ARM, chars, "spell"), chars)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(OUT / NAME)],
        extra={"seed_floor_rest": str(SEED_ROWS.parent / "native_spell")},
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
