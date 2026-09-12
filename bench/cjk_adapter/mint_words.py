#!/usr/bin/env python3
"""Mint word-level ext rows on top of a trained vocab pack (ko2 O2 smoke).

The composition wall: distilled per-char/per-piece ext rows converge in
embedding space but the frozen consumer never *composes* a multi-row name
into one identity (plan3: "metrics restorable, renders never compose").
This sidesteps composition: one appended row per curated surface (a name
word, a loanword tag), emitted by the encoder via greedy longest-match
(``HybridT5Encoder._encode_cjk_words``), initialised from the mean of the
rows its current spelled-out encoding produces, then distilled with the
ordinary span loss while every base row stays frozen
(``--param row --tunable_rows_from <old table size>``).

Usage::

    python bench/cjk_adapter/mint_words.py \
        --pack output/ckpt/cjk_vocab/cjk_vocab_pack_synthjako2 \
        --words 하쿠레이 레이무 쌍둥이 ... \
        --out bench/cjk_adapter/assets/ext_embed_mint

The output prefix is a full ext-asset pair (table + mapping with a "word"
dict) usable as --ext_prefix for cache staging, training, and run_bench.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bench.cjk_adapter import ext_vocab  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path, required=True, help="trained pack prefix")
    ap.add_argument("--words", nargs="*", default=[])
    ap.add_argument(
        "--subs",
        nargs="*",
        default=[],
        metavar="표기=en text",
        help="C-fallback substitutions: encode the surface as the EN text's "
        "stock spiece ids instead of ext rows (zero training — the pretrained "
        "rows carry identity no minted row learns; plan_ko3 M1-fail branch). "
        "A surface in both --subs and the word map resolves to the sub.",
    )
    ap.add_argument("--out", type=Path, required=True, help="output asset prefix")
    ap.add_argument("--text_encoder", default=None)
    args = ap.parse_args()
    if not args.words and not args.subs:
        ap.error("nothing to do: pass --words and/or --subs")

    from anima_lora import default_checkpoints
    from library.anima import strategy as strategy_anima

    ckpt = default_checkpoints()
    tok = strategy_anima.AnimaTokenizeStrategy(
        qwen3_path=args.text_encoder or ckpt.text_encoder,
        qwen3_max_length=512,
        t5_max_length=512,
    )

    table, mapping = ext_vocab.load_ext_assets(args.pack)
    enc = ext_vocab.HybridT5Encoder.from_mapping(
        tok.t5_tokenizer, tok.qwen3_tokenizer, mapping
    )

    word_map: dict[str, int] = dict(mapping.get("word") or {})
    rows = [table]
    for w in args.words:
        if w in word_map:
            print(f"{w}: already minted (row {word_map[w]}), skipping")
            continue
        ids, _ = enc._encode_cjk(w)
        ext_ids = [
            i - ext_vocab.T5_TABLE_SIZE for i in ids if i >= ext_vocab.T5_TABLE_SIZE
        ]
        if len(ids) == 1 and ext_ids:
            print(f"{w}: already a single ext token (row {ext_ids[0]}), skipping")
            continue
        if not ext_ids:
            print(f"{w}: no ext rows in its encoding, skipping")
            continue
        init = table[ext_ids].mean(dim=0, keepdim=True)
        row = table.shape[0] + len(rows) - 1
        word_map[w] = row
        rows.append(init)
        print(
            f"{w}: minted row {row} (init = mean of {len(ext_ids)} rows, {len(ids)} slots → 1)"
        )

    word_sub: dict[str, list[int]] = dict(mapping.get("word_sub") or {})
    for s in args.subs:
        surface, en = s.split("=", 1)
        ids = tok.t5_tokenizer(en, add_special_tokens=False)["input_ids"]
        word_sub[surface] = [int(i) for i in ids]
        print(f"{surface}: sub → {en!r} ({len(ids)} stock tokens)")

    if len(rows) == 1 and not args.subs:
        print("nothing to mint")
        return

    out_table = torch.cat(rows)
    out_map = dict(mapping)
    out_map["word"] = word_map
    if word_sub:
        out_map["word_sub"] = word_sub
    out_map["rows"] = out_table.shape[0]
    out_map["minted_from"] = str(args.pack)

    from safetensors.torch import save_file

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file({"ext_embed": out_table}, str(args.out.with_suffix(".safetensors")))
    args.out.with_suffix(".json").write_text(
        json.dumps(out_map, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"→ {args.out}.safetensors ({out_table.shape[0]} rows, base {table.shape[0]})"
    )
    print(
        f"train with: --ext_prefix {args.out} --param row --tunable_rows_from {table.shape[0]}"
    )


if __name__ == "__main__":
    main()
