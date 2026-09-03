#!/usr/bin/env python3
"""Ext-row visit statistics over the *training pool* of a CJK distill run.

Answers "which ext rows never get looked up, and why" for a given cache set
(the numbers in ``docs/experimental/cjk_ext_vocab_coverage.md``). Reads only
the ``.sids`` tensors out of the staged shards (``safe_open`` — no hidden
states are loaded), applies the same ``--train_registers`` filter
``scripts/distill_cjk/distill.py::make_pool`` applies, and buckets every row
by (block, script, visit band). Also reports the two structural checks:
rows whose surface never re-tokenizes to itself (unreachable), and
``<unk>`` positions left in the student stream (chars with no row at all).

CPU only, ~2 min over the four synthjakozh1 caches::

    .venv/bin/python project/cjk_aware_anima/probes/visit_stats.py \\
        --cache_dir cache_tags,cache_ko,cache_desc_ko,cache_zh \\
        --train_registers tags,tags_alt,names,tags_synth_ja,tags_ko,tags_alt_ko,\\
names_ko,names_synth_ko,desc_ko,tags_zh,tags_alt_zh,names_zh,tags_zh_hant,tags_synth_zh
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from library.anima.ext_vocab import T5_TABLE_SIZE, T5_UNK_ID, Route  # noqa: E402

from scripts.distill_cjk.rows import (  # noqa: E402
    BANDS,
    band,
    row_surfaces,
    script_of,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--cache_dir", required=True, help="comma list, relative to cjk_distill/"
    )
    ap.add_argument("--train_registers", default="", help="comma list; '' = all")
    ap.add_argument("--split", default="train")
    ap.add_argument(
        "--ext_prefix", type=Path, default=REPO / "bench/cjk_adapter/assets/ext_embed"
    )
    ap.add_argument(
        "--out", type=Path, default=None, help="write the row table as JSON"
    )
    args = ap.parse_args()

    from anima_lora import default_checkpoints
    from library.anima import strategy as strategy_anima

    mapping = json.loads(
        args.ext_prefix.with_suffix(".json").read_text(encoding="utf-8")
    )
    n_rows = int(mapping["rows"])
    regs = {r.strip() for r in args.train_registers.split(",") if r.strip()}

    visits = torch.zeros(n_rows, dtype=torch.long)
    unk_total = 0
    per_cache: dict[str, torch.Tensor] = {}
    base = REPO / "post_image_dataset" / "cjk_distill"
    for c in args.cache_dir.split(","):
        d = base / c.strip() / args.split
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        v = torch.zeros(n_rows, dtype=torch.long)
        by_shard: dict[str, list[str]] = collections.defaultdict(list)
        n = 0
        for rec in meta["pairs"]:
            if not regs or rec["register"] in regs:
                by_shard[rec["shard"]].append(rec["key"])
                n += 1
        unk = 0
        for sh, keys in by_shard.items():
            with safe_open(str(d / sh), "pt") as f:
                for k in keys:
                    ids = f.get_tensor(f"{k}.sids").long()
                    unk += int((ids == T5_UNK_ID).sum())
                    e = ids[ids >= T5_TABLE_SIZE] - T5_TABLE_SIZE
                    e = e[e < n_rows]
                    if e.numel():
                        v.index_add_(0, e, torch.ones_like(e))
        per_cache[c] = v
        visits += v
        unk_total += unk
        print(
            f"{c}: pairs {n}, rows visited {int((v > 0).sum())}, "
            f"visits {int(v.sum())}, <unk> tokens {unk}",
            flush=True,
        )
    print(
        f"JOINT: {int((visits > 0).sum())} / {n_rows} rows visited, <unk> {unk_total}"
    )

    ckpt = default_checkpoints()
    tok = strategy_anima.AnimaTokenizeStrategy(
        qwen3_path=ckpt.text_encoder, qwen3_max_length=512, t5_max_length=512
    )
    qtok = tok.qwen3_tokenizer
    surfaces = row_surfaces(mapping, qtok)

    tab: collections.Counter = collections.Counter()
    for r in range(n_rows):
        block, s = surfaces[r]
        tab[(block, script_of(s), band(int(visits[r])))] += 1
    print(
        f"\n{'block':9s} {'script':18s} {'total':>7s} "
        + " ".join(f"{b:>7s}" for b in BANDS)
    )
    for block, sc in sorted({(b, s) for b, s, _ in tab}):
        row = [tab[(block, sc, b)] for b in BANDS]
        print(f"{block:9s} {sc:18s} {sum(row):7d} " + " ".join(f"{x:7d}" for x in row))

    # Structural reachability: does a token row's own surface re-tokenize to it?
    unreach = []
    for block in ("qwen", "sym"):
        for k, r in (mapping.get(block) or {}).items():
            s = surfaces[r][1]
            if qtok(s, add_special_tokens=False)["input_ids"] != [int(k)]:
                unreach.append((block, s, int(visits[r])))
    print(f"\nrows whose surface never re-tokenizes to itself: {len(unreach)}")
    for block, s, v in unreach[:20]:
        print(f"  {block} {s!r} visits={v}")

    seen = visits > 0
    print("\nper-cache distinct rows by (block, script):")
    for c, v in per_cache.items():
        cc: collections.Counter = collections.Counter()
        for r in (v > 0).nonzero().flatten().tolist():
            cc[surfaces[r][0] + "/" + script_of(surfaces[r][1])] += 1
        print(f"  {c}: {dict(cc.most_common(6))}")

    visited = visits[seen]
    if visited.numel():
        q = torch.quantile(visited.float(), torch.tensor([0.5, 0.9, 0.99]))
        top = torch.topk(visits, 10)
        print(
            f"\nvisited rows: median {int(q[0])}, p90 {int(q[1])}, p99 {int(q[2])}, "
            f"max {int(visits.max())}; top-100 rows carry "
            f"{float(torch.topk(visits, 100).values.sum() / visits.sum()):.1%} of visits"
        )
        print(
            "top rows:",
            [(surfaces[int(r)][1], int(v)) for v, r in zip(top.values, top.indices)],
        )
    if Route.from_mapping(mapping).chars:
        print(f"\npack routes {len(Route.from_mapping(mapping).chars)} symbol chars")

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    str(r): [surfaces[r][0], surfaces[r][1], int(visits[r])]
                    for r in range(n_rows)
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
