#!/usr/bin/env python3
"""Build BYG edit-tuple sidecars for unpaired instruction-editing training.

For each captioned image this emits a ``<stem>_byg.safetensors`` holding the
*encoded* (post-LLM-adapter ``crossattn_emb`` + mask) for the four BYG roles:
``src_caption``, ``tgt_caption``, ``instruction``, ``reverse_instruction``. The
BYG dataset (``library/datasets/base.py``) loads these into
``batch["byg_{role}_emb"]`` / ``["byg_{role}_mask"]`` and the
``BYGMethodAdapter`` consumes them.

Generation is **tag-swap** (paper App. D shortcut): pick a color word present in
the caption and swap it for another, which mechanically yields a self-contained
reverse instruction and a temporal-language-free target caption. It cannot
express style edits; ``--vlm`` (paper App. D.1, Qwen3-VL) is not implemented.

Usage::

    python scripts/byg/build_edit_tuples.py \
        --dir image_dataset --cache_dir post_image_dataset/byg \
        --qwen3 models/text_encoders/qwen_3_06b_base.safetensors \
        --dit  models/diffusion_models/anima-base-v1.0.safetensors

Stems are matched to the training VAE/TE caches by basename, so the sidecar dir
is flat (``post_image_dataset/byg/<stem>_byg.safetensors``) — the same default
``--byg_text_dir`` the trainer reads.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import torch
from safetensors.torch import save_file
from tqdm import tqdm

# Tag-swap color vocabulary; multi-word colors first so the regex matches longest.
COLORS = [
    "light blue",
    "dark blue",
    "light green",
    "dark green",
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "pink",
    "orange",
    "black",
    "white",
    "brown",
    "grey",
    "gray",
    "silver",
    "gold",
    "blonde",
]
_COLOR_RE = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in COLORS) + r")\b", re.IGNORECASE
)


def _iter_caption_files(src: Path, recursive: bool):
    globber = src.rglob if recursive else src.glob
    for txt in sorted(globber("*.txt")):
        try:
            caption = txt.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if caption:
            yield txt.stem, caption


def build_tag_swap_tuple(caption: str, rng: random.Random):
    """Return (src, tgt, instruction, reverse, edit_type) or None if no color tag."""
    m = _COLOR_RE.search(caption)
    if m is None:
        return None
    old = m.group(1)
    choices = [c for c in COLORS if c.lower() != old.lower()]
    new = rng.choice(choices)
    # Swap only the first occurrence so the edit stays localized.
    tgt = caption[: m.start()] + new + caption[m.end() :]
    instruction = f"change {old} to {new}"
    reverse = f"change {new} to {old}"
    return caption, tgt, instruction, reverse, "color"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dir", type=str, default="image_dataset", help="Caption source dir"
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default="post_image_dataset/byg",
        help="Output dir for <stem>_byg.safetensors (default: post_image_dataset/byg).",
    )
    p.add_argument("--qwen3", type=str, required=True, help="Qwen3 text encoder path")
    p.add_argument(
        "--dit", type=str, required=True, help="DiT path (for the LLM adapter)"
    )
    p.add_argument("--t5_tokenizer_path", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--recursive", action="store_true", default=True)
    p.add_argument(
        "--limit", type=int, default=0, help="Cap images (0 = all). Smoke aid."
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--vlm",
        action="store_true",
        help="(not implemented in v1) VLM tuple generation for the style tail.",
    )
    args = p.parse_args()

    if args.vlm:
        raise NotImplementedError(
            "VLM tuple generation is a later phase; run tag-swap (drop --vlm) for v1."
        )

    rng = random.Random(args.seed)
    src = Path(args.dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from library.anima import weights as anima_utils
    from library.anima.strategy import AnimaTextEncodingStrategy, AnimaTokenizeStrategy
    from library.preprocess.text import _encode_batch

    print(f"Loading Qwen3 text encoder from {args.qwen3} ...")
    text_encoder, qwen3_tokenizer = anima_utils.load_qwen3_text_encoder(
        args.qwen3, dtype=torch.bfloat16, device=str(device)
    )
    t5_tokenizer = anima_utils.load_t5_tokenizer(args.t5_tokenizer_path)
    print(f"Loading LLM adapter from {args.dit} ...")
    llm_adapter = anima_utils.load_llm_adapter(
        args.dit, dtype=torch.bfloat16, device=str(device)
    )
    tokenize_strategy = AnimaTokenizeStrategy(
        qwen3_tokenizer=qwen3_tokenizer, t5_tokenizer=t5_tokenizer
    )
    encoding_strategy = AnimaTextEncodingStrategy()

    roles = ("src_caption", "tgt_caption", "instruction", "reverse_instruction")
    written = skipped = no_color = 0
    captions = list(_iter_caption_files(src, args.recursive))
    pbar = tqdm(captions, desc="BYG tuples", unit="img")
    for stem, caption in pbar:
        if args.limit and written >= args.limit:
            break
        out_path = cache_dir / f"{stem}_byg.safetensors"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        tup = build_tag_swap_tuple(caption, rng)
        if tup is None:
            no_color += 1
            continue
        src_cap, tgt_cap, instruction, reverse, edit_type = tup
        texts = [src_cap, tgt_cap, instruction, reverse]

        # _encode_batch returns (prompt_embeds, attn_mask, t5_ids, t5_mask, crossattn).
        _, _, _, t5_mask, crossattn = _encode_batch(
            texts,
            tokenize_strategy,
            encoding_strategy,
            text_encoder,
            llm_adapter,
            device,
        )
        if crossattn is None:
            raise RuntimeError("LLM adapter produced no crossattn_emb; check --dit")

        save_dict = {}
        for i, role in enumerate(roles):
            save_dict[f"{role}_emb"] = crossattn[i].contiguous()
            save_dict[f"{role}_mask"] = t5_mask[i].contiguous()
        save_file(
            save_dict,
            str(out_path),
            metadata={"edit_type": edit_type, "instruction": instruction},
        )
        written += 1
        pbar.set_postfix(written=written, skipped=skipped, no_color=no_color)

    print(
        f"\nBYG tuples done: {written} written, {skipped} skipped (existed), "
        f"{no_color} without a color tag (no tuple)."
    )
    text_encoder.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
