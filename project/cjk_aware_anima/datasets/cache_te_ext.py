#!/usr/bin/env python3
"""Arm-C TE cache: OCR-quoted captions encoded through the ext vocab pack.

Builds a temp caption mirror (symlinked resized images + ``.txt`` /
``.variants.txt`` with the OCR quote tags appended to every variant's flat
bag), then runs the standard ``cache_text_embeddings`` stage with a tokenize
strategy whose T5 side always goes through ``HybridT5Encoder`` (the trained
synthjako2 rows appended to the frozen adapter embed, exactly the run_bench
``*_ext`` arm). Output caches land in a sidecar dir for the arm-C dataset's
``text_cache_dir`` redirect — production captions, caches and masters are
untouched. Captions with no CJK encode bit-identically to the stock strategy,
so copied-through images double as an in-run control.

Usage (GPU -> daemon)::

    make daemon-run ARGS="project/cjk_aware_anima/datasets/cache_te_ext.py \
        --shard sincos"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from anima_lora import default_checkpoints  # noqa: E402
from bench.cjk_adapter import ext_vocab  # noqa: E402

TEXT_TAG_RE = re.compile(r"\b(text|speech bubble|sound effects)\b")


class ExtTokenizeStrategy:
    """AnimaTokenizeStrategy with the T5 side routed through the ext encoder."""

    def __init__(self, base, ext_encoder):
        self.base = base
        self.ext_encoder = ext_encoder
        # cache_text_embeddings reads these for the erasure-token pool.
        self.qwen3_tokenizer = base.qwen3_tokenizer
        self.t5_tokenizer = base.t5_tokenizer

    def tokenize(self, text):
        texts = [text] if isinstance(text, str) else list(text)
        t5_rows = []
        for t in texts:
            ids, mask = self.ext_encoder.encode(t, self.base.t5_max_length)
            t5_rows.append((torch.tensor(ids), torch.tensor(mask)))
        enc = self.base.qwen3_tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.base.qwen3_max_length,
        )
        return [
            enc["input_ids"],
            enc["attention_mask"],
            torch.stack([r[0] for r in t5_rows]),
            torch.stack([r[1] for r in t5_rows]),
        ]


def ocr_tags_by_stem(records: Path, max_lines: int) -> dict[str, list[str]]:
    by_stem: dict[str, list[str]] = {}
    with records.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_stem.setdefault(r["stem"], []).append(f"「{r['text']}」")
    return {k: v[:max_lines] for k, v in by_stem.items()}


def append_tags(caption: str, tags: list[str]) -> str:
    from anime_tools.captions.position_clauses import compose_caption, parse_caption

    parsed = parse_caption(caption)
    extra = [] if TEXT_TAG_RE.search(caption) else ["japanese text"]
    return compose_caption(
        tuple(parsed.flat_tags) + tuple(extra) + tuple(tags), parsed.clauses
    )


def build_mirror(
    resized: Path, mirror: Path, tags: dict[str, list[str]]
) -> tuple[int, int]:
    from anime_tools.captions.variants import read_variants_sidecar

    mirror.mkdir(parents=True, exist_ok=True)
    n_text = n_plain = 0
    for img in sorted(resized.glob("*.png")):
        link = mirror / img.name
        if not link.exists():
            link.symlink_to(img.resolve())
        stem_tags = tags.get(img.stem, [])
        cap_src = resized / f"{img.stem}.txt"
        caption = (
            cap_src.read_text(encoding="utf-8").strip() if cap_src.exists() else ""
        )
        var_src = resized / f"{img.stem}.variants.txt"
        rows = read_variants_sidecar(var_src) if var_src.exists() else []
        if stem_tags:
            caption = append_tags(caption, stem_tags)
            rows = [(label, append_tags(text, stem_tags)) for label, text in rows]
            n_text += 1
        else:
            n_plain += 1
        (mirror / f"{img.stem}.txt").write_text(caption + "\n", encoding="utf-8")
        if rows:
            out = ["# anima caption variants — auto-generated, do not hand-edit"]
            out += [f"{label}\t{text}" for label, text in rows]
            (mirror / f"{img.stem}.variants.txt").write_text(
                "\n".join(out) + "\n", encoding="utf-8"
            )
    return n_text, n_plain


def main() -> None:
    ckpt = default_checkpoints()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", default="sincos")
    ap.add_argument("--records", type=Path, default=None)
    ap.add_argument(
        "--ext_prefix",
        type=Path,
        default=REPO / "output" / "ckpt" / "cjk_vocab_pack_synthjako2",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--mirror",
        type=Path,
        default=None,
        help="Mirror dir (default mirror_<shard>); pass a fresh one when re-caching "
        "with a different OCR engine so the arm-C mirror stays as trained.",
    )
    ap.add_argument("--dit", default=ckpt.dit)
    ap.add_argument("--qwen3", default=ckpt.text_encoder)
    ap.add_argument("--max_lines", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    opts = ap.parse_args()

    base_dir = REPO / "post_image_dataset" / "cjk_unmask"
    records = opts.records or base_dir / f"ocr_records_{opts.shard}.jsonl"
    out = opts.out or base_dir / "te" / opts.shard
    mirror = opts.mirror or base_dir / f"mirror_{opts.shard}"
    resized = REPO / "post_image_dataset" / "resized" / opts.shard

    tags = ocr_tags_by_stem(records, opts.max_lines)
    n_text, n_plain = build_mirror(resized, mirror, tags)
    print(f"mirror: {n_text} captions carry OCR tags, {n_plain} plain -> {mirror}")

    from library.anima import weights as anima_utils
    from library.anima.strategy import AnimaTextEncodingStrategy, AnimaTokenizeStrategy
    from library.preprocess.text import cache_text_embeddings

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder, qwen3_tokenizer = anima_utils.load_qwen3_text_encoder(
        opts.qwen3, dtype=torch.bfloat16, device=str(device)
    )
    t5_tokenizer = anima_utils.load_t5_tokenizer(None)
    llm_adapter = anima_utils.load_llm_adapter(
        opts.dit, dtype=torch.bfloat16, device=str(device)
    )

    ext_table, mapping = ext_vocab.load_ext_assets(opts.ext_prefix)
    emb = llm_adapter.embed
    new_w = torch.cat(
        [emb.weight.data, ext_table.to(emb.weight.dtype).to(emb.weight.device)]
    )
    llm_adapter.embed = torch.nn.Embedding.from_pretrained(new_w)
    base_strategy = AnimaTokenizeStrategy(
        qwen3_tokenizer=qwen3_tokenizer, t5_tokenizer=t5_tokenizer
    )
    hybrid = ext_vocab.HybridT5Encoder.from_mapping(
        t5_tokenizer, qwen3_tokenizer, mapping
    )
    strategy = ExtTokenizeStrategy(base_strategy, hybrid)
    print(f"ext vocab: +{ext_table.shape[0]} rows from {opts.ext_prefix}")

    # Sanity: one OCR'd caption must actually hit ext rows.
    for stem, stem_tags in list(tags.items())[:1]:
        cap = (mirror / f"{stem}.txt").read_text(encoding="utf-8").strip()
        ids, mask = hybrid.encode(cap, 512)
        n_ext = sum(1 for i in ids[: sum(mask)] if i >= ext_vocab.T5_TABLE_SIZE)
        print(f"sanity {stem}: {sum(mask)} t5 tokens, {n_ext} ext — {stem_tags[:2]}")
        assert n_ext > 0, "OCR caption produced no ext rows — pack/encoder mismatch"

    out.mkdir(parents=True, exist_ok=True)
    stats = cache_text_embeddings(
        mirror,
        strategy,
        AnimaTextEncodingStrategy(),
        text_encoder,
        llm_adapter=llm_adapter,
        device=device,
        cache_dir=out,
        batch_size=opts.batch_size,
        overwrite=opts.overwrite,
    )
    print(f"cached -> {out}: {stats}")


if __name__ == "__main__":
    main()
