"""Per-prompt ext-row coverage vs the training spans (CPU-only, no model).

The 2c render grid failed exactly where the eval prompt's *content* tokens are
thin in the training corpus (騎/鎧 = 0 visits, 博麗 = 2), and worked where they
are dense (t1's every content token ≥ 300). This diagnostic makes that readable
per prompt: tokenize each eval prompt with the ext encoder, count how often
each ext row appears in the span supervision, and print the zero-visit tokens
decoded — the literal list of what the pack cannot know.

Two uses:
  * ranking a render grid's failures before eyeballing it;
  * sizing corpus work — the D1-widening / name-register target is "no
    user-facing content token below a visit floor", and this prints the gap.

Usage (CPU)::

    python project/cjk_aware_anima/gates/coverage.py \
        --pack output/ckpt/cjk_vocab/cjk_vocab_pack_global.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from anima_lora import default_checkpoints  # noqa: E402
from bench.cjk_adapter import ext_vocab  # noqa: E402
from library.anima.weights import load_qwen3_tokenizer, load_t5_tokenizer  # noqa: E402

T5 = ext_vocab.T5_TABLE_SIZE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", default="output/ckpt/cjk_vocab/cjk_vocab_pack_global.json")
    ap.add_argument("--pairs", default="post_image_dataset/cjk_distill/pairs.jsonl")
    ap.add_argument(
        "--prompts",
        default="project/cjk_aware_anima/assets/ja_eval_prompts.json",
    )
    ap.add_argument("--registers", default="tags,tags_alt")
    ap.add_argument("--lang", default="ja", help="prompt-file key holding the native text (ja/ko/zh)")
    ap.add_argument("--floor", type=int, default=5, help="low-visit threshold")
    args = ap.parse_args()
    registers = set(args.registers.split(","))

    ck = default_checkpoints()
    qwen_tok = load_qwen3_tokenizer(str(ck.text_encoder))
    t5_tok = load_t5_tokenizer()
    pack = json.load(open(REPO / args.pack))
    enc = ext_vocab.HybridT5Encoder.from_mapping(t5_tok, qwen_tok, pack)

    # Visit counts over the span supervision. Span texts repeat massively
    # across images, so cache text -> rows.
    visits: collections.Counter = collections.Counter()
    span_rows: dict[str, list[int]] = {}
    n_pairs = 0
    with open(REPO / args.pairs) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("register") not in registers:
                continue
            n_pairs += 1
            for s in rec.get("spans") or []:
                ja = s["ja"]
                rows = span_rows.get(ja)
                if rows is None:
                    raw, _ = enc.encode(ja, 128)
                    rows = [i - T5 for i in raw if i >= T5]
                    span_rows[ja] = rows
                for r in rows:
                    visits[r] += 1
    print(
        f"pairs={n_pairs} ({args.registers}) unique_span_texts={len(span_rows)} "
        f"rows_visited={len(visits)} / {pack['rows']}"
    )

    inv_char = {v: k for k, v in pack["char"].items()}
    inv_qwen = {v: int(k) for k, v in pack["qwen"].items()}

    def row_str(row: int) -> str:
        if row in inv_qwen:
            return qwen_tok.decode([inv_qwen[row]])
        return inv_char.get(row, f"#{row}")

    prompts = json.load(open(REPO / args.prompts))
    lo = args.floor
    print(
        f"\n{'prompt':<18}{'ext':>4}{'unk':>4}{'v=0':>5}{'v<' + str(lo):>5}"
        f"  zero/low-visit tokens (token:visits)"
    )
    for key, entry in prompts.items():
        if key.startswith("_"):
            continue
        raw, mask = enc.encode(entry[args.lang], 512)
        n = sum(mask)
        rows = [i - T5 for i in raw[:n] if i >= T5]
        unk = sum(1 for i in raw[:n] if i == ext_vocab.T5_UNK_ID)
        thin = [(r, visits[r]) for r in rows if visits[r] < lo]
        v0 = sum(1 for _, v in thin if v == 0)
        detail = " ".join(f"{row_str(r)}:{v}" for r, v in thin)
        print(f"{key:<18}{len(rows):>4}{unk:>4}{v0:>5}{len(thin):>5}  {detail}")


if __name__ == "__main__":
    main()
