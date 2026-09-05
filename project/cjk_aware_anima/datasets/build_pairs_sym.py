"""Symbol register ``tags_sym`` (plan_zh2 U5): a teacher for rows that have none.

The symbol block (findings §11) gives ``^^^`` / ``:<`` / ``☆`` ext rows instead
of one shared T5 ``<unk>``, but nothing teaches those rows anything: the EN
caption's own side is ``<unk>`` too, so every existing pair supervises a symbol
row toward the ``<unk>`` embedding at best. This register is the fix the
glossary registers used for CJK — **teacher text = the tag's Danbooru wiki
definition, one short EN clause; student text = the symbol verbatim** — with
the whole rest of the caption shared, so only the symbol span differs.

Corpus premise, measured 2026-09-04 (``scratchpad/sym_census2.py``): across the
11,869 training images only ``^^^`` (13) and ``^ ^`` (4) occur as real symbol
tags — the plan's "~40 tags with > 20 occurrences" is not what this corpus
holds — so the register is minted entirely from templates: every entry of
``assets/tag_glossary_sym.json`` gets ``--per-target`` captions regardless of
its corpus count (row visits are ~0 for all of them, the ``synth_names``
floor-filling allocation degenerates to a constant).

Minting (the ``synth_tags`` recipe): pick a random ``image_dataset`` caption
template, replace one random non-leading *general* segment — teacher side gets
the definition, student side the symbol (``via: wiki_verified``, 1.0). Context
is either kept EN on both sides (``en_pinned`` spans, the ``names``-register
exactness argument: teacher and student agree everywhere but the symbol) or
composed through the ja / ko / zh glossary like the ``tags_*`` registers, so
the symbol also sits in the CJK context the gate prompts use;
``--en-context-frac`` splits the two, the CJK share is spread evenly over
``--langs``. Decorative symbols (``☆`` ``♪`` ``♡`` ``×``) get the same slot —
in captions they are tags like ``star (symbol)`` / ``musical note`` / ``heart``.

Every student symbol is checked against the pack's row mapping and must touch
at least one row inside ``mapping["sym_rows"]`` (a tag T5 *can* spell has no
row to teach and is skipped with a warning); every teacher clause must touch
none. ``id`` is ``SYM/<template image>/<tag>/<k>`` so the cache's image-grouped
holdout spreads each symbol over both splits (``tags_sym`` then reads out
per-register at eval like any other).

Usage (CPU)::

    python project/cjk_aware_anima/datasets/build_pairs_sym.py --dry-run
    python project/cjk_aware_anima/datasets/build_pairs_sym.py
    make daemon-run ARGS="-m scripts.distill_cjk.cache \
        --pairs post_image_dataset/cjk_distill/pairs_sym.jsonl \
        --cache_dir post_image_dataset/cjk_distill/cache_sym --holdout 500"
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
for p in (str(ROOT), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import build_pairs  # noqa: E402
import synth_tags  # noqa: E402
import tag_glossary  # noqa: E402

REGISTER = "tags_sym"
DEFAULT_GLOSSARY = ROOT / "assets" / "tag_glossary_sym.json"
DEFAULT_OUT = REPO / "post_image_dataset" / "cjk_distill" / "pairs_sym.jsonl"
DEFAULT_PACK = REPO / "output" / "ckpt" / "cjk_vocab_pack_synthjakozh1sym_r256.json"
LANG_GLOSSARY = {
    "ja": tag_glossary.DEFAULT_OUT,
    "ko": tag_glossary.ASSETS / "tag_glossary_ko.json",
    "zh": tag_glossary.ASSETS / "tag_glossary_zh.json",
}


def load_sym_glossary(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tags = data["tags"]
    for sym, e in tags.items():
        if not e.get("en"):
            raise SystemExit(f"{path}: {sym!r} has no `en` teacher clause")
    return tags


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--glossary", type=Path, default=DEFAULT_GLOSSARY)
    ap.add_argument("--captions", type=Path, default=synth_tags.DEFAULT_CAPTIONS)
    ap.add_argument(
        "--pack",
        type=Path,
        default=DEFAULT_PACK,
        help="pack json whose row mapping (with `sym_rows`) defines the ext encoding",
    )
    ap.add_argument("--per-target", type=int, default=300, help="captions per symbol")
    ap.add_argument(
        "--en-context-frac",
        type=float,
        default=0.5,
        help="share of captions whose context stays EN on both sides (en_pinned); "
        "the rest is composed through the --langs glossaries, evenly",
    )
    ap.add_argument("--langs", default="ja,ko,zh")
    ap.add_argument("--min-segs", type=int, default=6)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    for lang in langs:
        if lang not in LANG_GLOSSARY:
            raise SystemExit(f"unknown --langs entry {lang!r}")

    sym_tags = load_sym_glossary(args.glossary)
    rows = synth_tags.load_encoder(args.pack)
    lo, hi = json.loads(args.pack.read_text(encoding="utf-8"))["sym_rows"]

    def sym_rows(text: str) -> list[int]:
        return [i for i in rows(text) if lo <= i < hi]

    targets: list[tuple[str, str]] = []
    print(f"{len(sym_tags)} symbol entries in {args.glossary.name}:")
    for sym, e in sym_tags.items():
        en = e["en"]
        s_rows, t_rows = sym_rows(sym), sym_rows(en)
        flag = ""
        if not s_rows:
            flag = "  SKIP: student touches no symbol row (T5 spells it)"
        elif t_rows:
            flag = f"  SKIP: teacher clause touches symbol rows {t_rows}"
        print(f"   {sym!r:14s} -> {en!r:42s} rows={s_rows}{flag}")
        if not flag:
            targets.append((sym, en))
    n_total = len(targets) * args.per_target
    print(
        f"allocation: {n_total} pairs over {len(targets)} symbols "
        f"({args.per_target} each; en-context {args.en_context_frac:.2f}, "
        f"cjk context over {langs})"
    )
    if args.dry_run:
        return

    glossaries = {
        lang: json.loads(LANG_GLOSSARY[lang].read_text(encoding="utf-8"))["tags"]
        for lang in langs
    }
    templates = [
        (image_id, segs)
        for image_id, text in build_pairs.load_captions([(args.captions, False)])
        if len(segs := build_pairs.split_caption(text)) >= args.min_segs
    ]
    if not templates:
        raise SystemExit("no caption templates")
    # Slots are general tags only — never a name / artist (other registers'
    # supervision) — judged with the JA glossary's axis, which covers the most.
    axis_of = glossaries.get("ja") or next(iter(glossaries.values()), {})

    minted: list[dict] = []
    by_ctx: collections.Counter = collections.Counter()
    for sym, en in targets:
        for k in range(args.per_target):
            image_id, segs = rng.choice(templates)
            segs = list(segs)
            slots = [
                i
                for i in range(1, len(segs))
                if (axis_of.get(segs[i]) or {}).get("axis") in (None, "general")
            ] or [len(segs) - 1]
            i = rng.choice(slots)
            if rng.random() < args.en_context_frac:
                lang = "en"
                student = list(segs)
                missing: list[str] = []
                spans = [
                    {"en": s, "ja": s, "via": "en_pinned", "f1": 0.0} for s in segs
                ]
                joiner = build_pairs.JOINER
            else:
                lang = rng.choice(langs)
                student, missing, spans = build_pairs.compose(
                    segs, glossaries[lang], alt=False, rng=rng, min_f1=0.5
                )
                joiner = build_pairs.pick_joiner(rng if lang == "ja" else None)
            teacher = list(segs)
            teacher[i] = en
            student[i] = sym
            spans[i] = {"en": en, "ja": sym, "via": "wiki_verified", "f1": 0.0}
            minted.append(
                {
                    "id": f"SYM/{image_id}/{sym}/{k}",
                    "source": "SYM",
                    "register": REGISTER,
                    "lang": lang,
                    "symbol": sym,
                    "en": ", ".join(teacher),
                    "ja": joiner.join(student),
                    "joiner": joiner,
                    "n_missing": len(missing),
                    "spans": spans,
                }
            )
            by_ctx[lang] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for rec in minted:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {args.out} ({len(minted)} pairs; context {dict(by_ctx)})")
    for rec in rng.sample(minted, min(4, len(minted))):
        print(f"   EN: {rec['en']}\n   ST: {rec['ja']}")


if __name__ == "__main__":
    main()
