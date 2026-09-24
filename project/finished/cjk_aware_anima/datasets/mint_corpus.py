"""Build the minted-word training corpus (plan_ko3 M1; smoke rule, committed).

The smoke corpus was an ad-hoc filter: every pair from the KO corpus files
whose student-side text contains a minted surface (1,273 pairs for the 10
smoke words). This reproduces that rule and layers the M1 densification on
top:

* base = pairs_ko + pairs_synth_ko + pairs_desc_ko, dedup by id;
* densified targets' old synthetic pairs are REPLACED, not appended
  (``--replace-prefix SYN/hakurei reimu/`` drops the stale 24 so the fresh
  200 own the id space);
* extra files (the ``--only`` synth_names run, the tags_synth_ko run) merge
  in before the filter;
* surfaces come from the mint mapping (``mapping["word"]``) so the corpus
  and the encoder can never disagree on what counts as minted;
* ``--respace FROM|TO`` rewrites a wording in ja + spans (the wiki KO label
  for touhou is spaced ``동방 프로젝트`` while the minted surface and the n1
  eval prompt use ``동방프로젝트`` — the spaced form would never fire the row).

Usage::

    python project/finished/cjk_aware_anima/datasets/mint_corpus.py \
        --mapping bench/cjk_adapter/assets/ext_embed_mint_m1.json \
        --extra <scratch>/names_synth_ko.jsonl \
                post_image_dataset/cjk_distill/tags_synth_ko.jsonl \
        --replace-prefix "SYN/hakurei reimu/" \
        --respace "동방 프로젝트|동방프로젝트" \
        --out post_image_dataset/cjk_distill/pairs_mint_m1.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BASE = REPO / "post_image_dataset" / "cjk_distill"
DEFAULT_SOURCES = ("pairs_ko.jsonl", "pairs_synth_ko.jsonl", "pairs_desc_ko.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mapping", type=Path, required=True, help="mint mapping json")
    ap.add_argument(
        "--sources",
        nargs="*",
        type=Path,
        default=[BASE / f for f in DEFAULT_SOURCES],
    )
    ap.add_argument("--extra", nargs="*", type=Path, default=[])
    ap.add_argument(
        "--replace-prefix",
        nargs="*",
        default=[],
        help="drop base pairs whose id starts with any of these (their "
        "densified replacements arrive via --extra)",
    )
    ap.add_argument(
        "--respace",
        nargs="*",
        default=[],
        metavar="FROM|TO",
        help="rewrite a wording in ja + span ja (extra files only — the base "
        "corpus keeps its shipped wordings)",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    words = list(
        (json.loads(args.mapping.read_text(encoding="utf-8")).get("word") or {})
    )
    if not words:
        raise SystemExit(f"{args.mapping}: no minted words")
    respace = [tuple(r.split("|", 1)) for r in args.respace]

    def rewrite(rec: dict) -> dict:
        for a, b in respace:
            if a in rec.get("ja", ""):
                rec["ja"] = rec["ja"].replace(a, b)
                for sp in rec.get("spans") or []:
                    sp["ja"] = sp["ja"].replace(a, b)
        return rec

    seen: set[str] = set()
    kept: list[dict] = []
    dropped = 0
    per_register: collections.Counter = collections.Counter()
    per_word: collections.Counter = collections.Counter()

    def ingest(path: Path, *, extra: bool):
        nonlocal dropped
        for line in path.open(encoding="utf-8"):
            rec = json.loads(line)
            if rec["id"] in seen:
                continue
            seen.add(rec["id"])
            if not extra and any(rec["id"].startswith(p) for p in args.replace_prefix):
                dropped += 1
                continue
            if extra:
                rec = rewrite(rec)
            if not any(w in rec.get("ja", "") for w in words):
                continue
            kept.append(rec)
            per_register[rec.get("register")] += 1
            for w in words:
                if w in rec.get("ja", ""):
                    per_word[w] += 1

    # extra files first so a densified replacement wins any id tie
    for p in args.extra:
        ingest(p, extra=True)
    for p in args.sources:
        ingest(p, extra=False)

    with args.out.open("w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{args.out}: {len(kept)} pairs ({dropped} replaced-prefix pairs dropped)")
    print("per register:", dict(per_register))
    print("per surface:", {w: per_word[w] for w in words})


if __name__ == "__main__":
    main()
