"""Synthetic under-floor *general-tag* register (plan §5a, report 0827 §6.1).

The 2c grid fails exactly where an eval prompt's content rows are thin in the
span supervision (`鎧` 193 visits, `照明` 2, `巫` 188 against the measured 300
render floor), and the mechanism that fixed every other coverage-bound prompt
— mint span visits by substituting the wording into real caption templates
(`names_synth_ja`) — applies to general tags unchanged. This is "targeted
caption widening" without a crawl: text-only, CPU.

Target selection is prompt-driven, like `gates/coverage.py`: take every
tag-style eval prompt (t*/c* ids; n1/n2 are the name registers' job and n3's
en/ja segs don't align), split both sides on ``", "`` into aligned
``(en_seg, ja_seg)`` wordings, encode the JA side with the ext encoder, and
keep the segs owning at least one ext row under ``--floor`` in the current
corpus. The *prompt's* wording is what gets trained — the glossary may bind
the tag to another register (``armor`` → アーマー while users type 鎧; the
README's "glossary bound the tag to another wording" corollary), so the
minted span pins the user-facing form (``via: eval_pinned``, trust 1.0 by
default — the wording is hand-authored, not machine-chosen).

Minting: pick a random `image_dataset` caption template, compose the JA
context through the glossary exactly like the `tags` register
(`build_pairs.compose`), then replace one random non-leading general segment
with the target — one target per caption, EN side stays the faithful EN seg.
Register ``tags_synth_ja``; the per-pair joiner is recorded like every
span-carrying register. Allocation is the `synth_names.py` greedy: seed row
visits from the trained registers, size each target so its rarest row reaches
the floor (each minted caption carries the wording once), rarest first.

Output merges ``--pairs`` + minted pairs into ``pairs_synth_tags.jsonl``
next to it, plus ``tags_synth.jsonl`` holding just the minted rows.

Usage (CPU)::

    python project/cjk_aware_anima/datasets/synth_tags.py --dry-run
    python project/cjk_aware_anima/datasets/synth_tags.py
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
import tag_glossary  # noqa: E402

DEFAULT_PROMPTS = (
    REPO / "project" / "cjk_aware_anima" / "assets" / "ja_eval_prompts.json"
)
DEFAULT_PAIRS = REPO / "post_image_dataset" / "cjk_distill" / "pairs_synth.jsonl"
DEFAULT_CAPTIONS = REPO / "image_dataset"
REGISTER = "tags_synth_ja"
SEED_REGISTERS = ("tags", "tags_alt", "names", "names_synth_ja")
# --lang ko: same machinery, KO surfaces (plan_ko3 M1 loanword-tag widening).
KO_PROMPTS = REPO / "project" / "cjk_aware_anima" / "assets" / "ko_eval_prompts.json"
KO_PAIRS = REPO / "post_image_dataset" / "cjk_distill" / "pairs_synth_ko.jsonl"
KO_REGISTER = "tags_synth_ko"
KO_SEED_REGISTERS = ("tags_ko", "tags_alt_ko", "names_ko", "names_synth_ko")
# --lang zh (plan_zh.md Z1): same machinery, zh surfaces.
ZH_PROMPTS = REPO / "project" / "cjk_aware_anima" / "assets" / "zh_eval_prompts.json"
ZH_PAIRS = REPO / "post_image_dataset" / "cjk_distill" / "pairs_zh.jsonl"
ZH_REGISTER = "tags_synth_zh"
ZH_SEED_REGISTERS = ("tags_zh", "tags_alt_zh", "tags_zh_hant", "names_zh")
LANG_CFG = {
    "ja": (DEFAULT_PROMPTS, DEFAULT_PAIRS, REGISTER, SEED_REGISTERS),
    "ko": (KO_PROMPTS, KO_PAIRS, KO_REGISTER, KO_SEED_REGISTERS),
    "zh": (ZH_PROMPTS, ZH_PAIRS, ZH_REGISTER, ZH_SEED_REGISTERS),
}


def load_encoder(pack: Path):
    from anima_lora import default_checkpoints  # noqa: PLC0415
    from bench.cjk_adapter import ext_vocab  # noqa: PLC0415
    from library.anima.weights import (  # noqa: PLC0415
        load_qwen3_tokenizer,
        load_t5_tokenizer,
    )

    ck = default_checkpoints()
    enc = ext_vocab.HybridT5Encoder.from_mapping(
        load_t5_tokenizer(),
        load_qwen3_tokenizer(str(ck.text_encoder)),
        json.loads(pack.read_text(encoding="utf-8")),
    )
    t5 = ext_vocab.T5_TABLE_SIZE
    cache: dict[str, tuple[int, ...]] = {}

    def rows(text: str) -> tuple[int, ...]:
        r = cache.get(text)
        if r is None:
            raw, _ = enc.encode(text, 128)
            r = cache[text] = tuple(i - t5 for i in raw if i >= t5)
        return r

    return rows


def eval_targets(prompts: Path, key: str = "ja") -> list[tuple[str, str, str]]:
    """Aligned (prompt_id, en_seg, ja_seg) for every tag-style prompt seg."""
    data = json.loads(prompts.read_text(encoding="utf-8"))
    out, seen = [], set()
    for pid, rec in data.items():
        if pid.startswith("_") or not isinstance(rec, dict):
            continue
        if pid[0] not in "tc":  # tag-style + composition registers only
            continue
        en, ja = rec["en"].split(", "), rec[key].split(", ")
        if len(en) != len(ja):
            raise SystemExit(f"{pid}: en/ja segment count mismatch")
        for e, j in zip(en, ja):
            if j not in seen:
                seen.add(j)
                out.append((pid, e.strip(), j.strip()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--captions", type=Path, default=DEFAULT_CAPTIONS)
    ap.add_argument(
        "--pack",
        type=Path,
        default=REPO / "output" / "ckpt" / "cjk_vocab_pack_synthja_v2.json",
        help="pack whose row mapping defines the ext encoding (rows, not weights)",
    )
    ap.add_argument("--glossary", type=Path, default=tag_glossary.DEFAULT_OUT)
    ap.add_argument("--floor", type=int, default=300)
    ap.add_argument("--per-target", type=int, default=24, help="minimum captions/seg")
    ap.add_argument("--max-per-target", type=int, default=500)
    ap.add_argument(
        "--extra-terms",
        nargs="*",
        default=[],
        help="extra 'en|ja' wordings to include regardless of the prompt files",
    )
    ap.add_argument(
        "--lang",
        default="ja",
        choices=["ja", "ko", "zh"],
        help="student-side language: ko reads ko_eval_prompts / the _ko "
        "registers, mints register tags_synth_ko, joins rng-free with ', '",
    )
    ap.add_argument(
        "--rows-from",
        type=int,
        default=0,
        help="keep only targets whose encoding touches an ext row >= this "
        "index (plan_ko3 M1: scope allocation to minted word rows — under "
        "--span_focus_from, pairs for any other target are dead weight)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="merged output (default pairs_synth_tags<suffix>.jsonl next to --pairs); "
        "the no-synth-names JA corpus is `--pairs pairs.jsonl --out pairs_tags.jsonl`",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    prompts_default, pairs_default, register, seed_registers = LANG_CFG[args.lang]
    if args.lang != "ja":
        if args.prompts == DEFAULT_PROMPTS:
            args.prompts = prompts_default
        if args.pairs == DEFAULT_PAIRS:
            args.pairs = pairs_default
        if args.glossary == tag_glossary.DEFAULT_OUT:
            args.glossary = tag_glossary.ASSETS / f"tag_glossary_{args.lang}.json"

    rows = load_encoder(args.pack)

    # Row visits over the trained registers of the current corpus.
    visits: collections.Counter = collections.Counter()
    base: list[str] = []
    with args.pairs.open(encoding="utf-8") as f:
        for line in f:
            base.append(line.rstrip("\n"))
            rec = json.loads(line)
            if rec.get("register") not in seed_registers:
                continue
            for sp in rec.get("spans") or []:
                if sp.get("via") != "en_pinned":
                    visits.update(rows(sp["ja"]))

    cand = eval_targets(args.prompts, key=args.lang)
    cand += [("extra", *t.split("|", 1)) for t in args.extra_terms]
    targets = []
    for pid, en, ja in cand:
        r = rows(ja)
        if args.rows_from and not any(i >= args.rows_from for i in r):
            continue
        under = [i for i in r if visits[i] < args.floor]
        if under:
            targets.append((pid, en, ja, min(visits[i] for i in under)))
    targets.sort(key=lambda t: t[3])  # rarest first, like synth_names
    print(f"eval segs {len(cand)}, under-floor targets {len(targets)}:")
    for pid, en, ja, v in targets:
        print(f"   {pid:16s} {en:28s} {ja:14s} min_visits={v}")

    alloc: dict[str, int] = {}
    for _, _, ja, _ in targets:
        need = max((args.floor - visits[i] for i in rows(ja)), default=0)
        n = max(args.per_target, min(args.max_per_target, need))
        alloc[ja] = n
        for i in rows(ja):
            visits[i] += n
    n_total = sum(alloc.values())
    print(
        f"allocation: {n_total} pairs over {len(alloc)} targets "
        f"(min {min(alloc.values(), default=0)}, max {max(alloc.values(), default=0)})"
    )
    if args.dry_run:
        return

    glossary = json.loads(args.glossary.read_text(encoding="utf-8"))["tags"]
    templates = [
        segs
        for _, text in build_pairs.load_captions([(args.captions, False)])
        if len(segs := build_pairs.split_caption(text)) >= 6
    ]
    if not templates:
        raise SystemExit("no caption templates")

    minted = []
    by_target = {(en, ja): 0 for _, en, ja, _ in targets}
    for (pid, en, ja, _), k_total in ((t, alloc[t[2]]) for t in targets):
        for k in range(k_total):
            segs = list(rng.choice(templates))
            ja_out, _missing, spans = build_pairs.compose(
                segs, glossary, alt=False, rng=rng, min_f1=0.5
            )
            # replace one random non-leading segment whose axis is general
            # (never a name/artist — those are other registers' supervision)
            slots = [
                i
                for i in range(1, len(segs))
                if (glossary.get(segs[i]) or {}).get("axis") in (None, "general")
                and segs[i] != en
            ] or [len(segs) - 1]
            i = rng.choice(slots)
            segs[i] = en
            ja_out[i] = ja
            spans[i] = {"en": en, "ja": ja, "via": "eval_pinned", "f1": 0.0}
            joiner = build_pairs.pick_joiner(rng if args.lang == "ja" else None)
            minted.append(
                {
                    "id": f"SYNT/{pid}/{en}/{k}",
                    "source": "SYNT",
                    "register": register,
                    "lang": args.lang,
                    "en": ", ".join(segs),
                    "ja": joiner.join(ja_out),
                    "joiner": joiner,
                    "n_missing": len(_missing),
                    "spans": spans,
                }
            )
            by_target[(en, ja)] += 1

    suffix = "" if args.lang == "ja" else f"_{args.lang}"
    out_dir = args.pairs.parent
    only = out_dir / f"tags_synth{suffix}.jsonl"
    merged = args.out or out_dir / f"pairs_synth_tags{suffix}.jsonl"
    with only.open("w", encoding="utf-8") as f:
        for rec in minted:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with merged.open("w", encoding="utf-8") as f:
        for line in base:
            f.write(line + "\n")
        for rec in minted:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {only} ({len(minted)}) and {merged} ({len(base) + len(minted)})")


if __name__ == "__main__":
    main()
