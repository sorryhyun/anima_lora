#!/usr/bin/env python3
"""Cache text encoder (Qwen3) outputs for all captioned images in a dataset directory.

Reads .txt caption sidecars, tokenizes with Qwen3 + T5, encodes through the
Qwen3 text encoder, and optionally runs the LLM adapter to produce crossattn_emb.
Saves results as *_anima_te.safetensors alongside each image.

Supports caption shuffle variants: with --caption_shuffle_variants N, generates
N shuffled permutations of each caption and caches all variants in one file.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library.cache_utils import TE_CACHE_SUFFIX
from library.datasets.image_utils import IMAGE_EXTENSIONS


def _generate_shuffled_captions(caption: str, num_variants: int) -> list[str]:
    """Generate N shuffled caption variants using smart shuffle logic."""
    from library import anima_train_utils

    tags = [t.strip() for t in caption.split(",")]
    return [
        ", ".join(anima_train_utils.anima_smart_shuffle_caption(tags.copy()))
        for _ in range(num_variants)
    ]


def _encode_batch(
    captions: list[str],
    tokenize_strategy,
    encoding_strategy,
    text_encoder,
    llm_adapter,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Tokenize, encode through Qwen3, optionally run LLM adapter. Returns CPU tensors."""
    tokens_and_masks = tokenize_strategy.tokenize(captions)
    with torch.no_grad():
        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = (
            encoding_strategy.encode_tokens(
                tokenize_strategy, [text_encoder], tokens_and_masks
            )
        )

        crossattn_emb = None
        if llm_adapter is not None:
            crossattn_emb = llm_adapter(
                source_hidden_states=prompt_embeds,
                target_input_ids=t5_input_ids.to(device, dtype=torch.long),
                target_attention_mask=t5_attn_mask.to(device),
                source_attention_mask=attn_mask,
            )
            crossattn_emb[~t5_attn_mask.to(device).bool()] = 0
            crossattn_emb = crossattn_emb.to(dtype=torch.bfloat16).cpu()

    return (
        prompt_embeds.to(dtype=torch.bfloat16).cpu(),
        attn_mask.to(dtype=torch.int32).cpu(),
        t5_input_ids.to(dtype=torch.long).cpu(),
        t5_attn_mask.to(dtype=torch.int32).cpu(),
        crossattn_emb,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=str, required=True, help="Dataset directory")
    parser.add_argument(
        "--qwen3", type=str, required=True, help="Path to Qwen3 text encoder"
    )
    parser.add_argument(
        "--dit",
        type=str,
        default=None,
        help="Path to DiT model for LLM adapter crossattn_emb caching",
    )
    parser.add_argument(
        "--t5_tokenizer_path",
        type=str,
        default=None,
        help="Path to T5 tokenizer (default: configs/t5_old/)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Text encoder batch size (default: 16)",
    )
    parser.add_argument(
        "--caption_shuffle_variants",
        type=int,
        default=0,
        help="Number of shuffled caption variants per image (0 = single caption)",
    )
    args = parser.parse_args()

    from safetensors.torch import save_file as _save_safetensors

    from library import anima_utils
    from library.strategy_anima import AnimaTextEncodingStrategy, AnimaTokenizeStrategy

    data_dir = Path(args.dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = args.caption_shuffle_variants

    # Load text encoder + tokenizers
    print(f"Loading Qwen3 text encoder from {args.qwen3} ...")
    text_encoder, qwen3_tokenizer = anima_utils.load_qwen3_text_encoder(
        args.qwen3, dtype=torch.bfloat16, device=str(device)
    )
    t5_tokenizer = anima_utils.load_t5_tokenizer(args.t5_tokenizer_path)

    # Optionally load LLM adapter for crossattn_emb caching
    llm_adapter = None
    if args.dit:
        print(f"Loading LLM adapter from {args.dit} ...")
        llm_adapter = anima_utils.load_llm_adapter(
            args.dit, dtype=torch.bfloat16, device=str(device)
        )

    tokenize_strategy = AnimaTokenizeStrategy(
        qwen3_tokenizer=qwen3_tokenizer, t5_tokenizer=t5_tokenizer
    )
    encoding_strategy = AnimaTextEncodingStrategy()

    # Collect images that have caption sidecars
    entries: list[tuple[Path, str]] = []
    for p in sorted(data_dir.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        caption_path = p.with_suffix(".txt")
        if not caption_path.exists():
            continue
        caption = caption_path.read_text(encoding="utf-8").strip().split("\n")[0]
        if caption:
            entries.append((p, caption))

    total = len(entries)
    cached = 0
    skipped = 0
    caption_dropout_rate = torch.tensor(0.0, dtype=torch.float32)

    if N > 0:
        print(f"Caption shuffle variants: {N}")

    pbar = tqdm(total=total, desc="Caching text embeddings")
    for batch_start in range(0, total, args.batch_size):
        batch = entries[batch_start : batch_start + args.batch_size]

        # Skip already-cached entries
        to_encode: list[tuple[Path, str, Path]] = []
        for img_path, caption in batch:
            cache_path = img_path.with_name(
                img_path.stem + TE_CACHE_SUFFIX
            )
            if cache_path.exists():
                skipped += 1
                pbar.update(1)
                pbar.set_postfix_str(f"skip {img_path.name}")
            else:
                to_encode.append((img_path, caption, cache_path))

        if not to_encode:
            continue

        if N <= 0:
            # Single variant: encode original captions
            captions = [c for _, c, _ in to_encode]
            prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, crossattn_emb = (
                _encode_batch(
                    captions,
                    tokenize_strategy,
                    encoding_strategy,
                    text_encoder,
                    llm_adapter,
                    device,
                )
            )

            for i, (img_path, _, cache_path) in enumerate(to_encode):
                save_dict = {
                    "prompt_embeds": prompt_embeds[i],
                    "attn_mask": attn_mask[i],
                    "t5_input_ids": t5_input_ids[i],
                    "t5_attn_mask": t5_attn_mask[i],
                    "caption_dropout_rate": caption_dropout_rate,
                }
                if crossattn_emb is not None:
                    save_dict["crossattn_emb"] = crossattn_emb[i]
                _save_safetensors(save_dict, str(cache_path))
                cached += 1
                pbar.update(1)
                pbar.set_postfix_str(f"{img_path.name}")
        else:
            # Multi-variant: generate N shuffled captions per image, encode all at once
            all_captions: list[str] = []
            for _, caption, _ in to_encode:
                all_captions.extend(_generate_shuffled_captions(caption, N))

            prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, crossattn_emb = (
                _encode_batch(
                    all_captions,
                    tokenize_strategy,
                    encoding_strategy,
                    text_encoder,
                    llm_adapter,
                    device,
                )
            )

            for i, (img_path, _, cache_path) in enumerate(to_encode):
                save_dict = {
                    "num_variants": torch.tensor(N, dtype=torch.int64),
                    "caption_dropout_rate": caption_dropout_rate,
                }
                for vi in range(N):
                    flat_idx = i * N + vi
                    save_dict[f"prompt_embeds_v{vi}"] = prompt_embeds[flat_idx]
                    save_dict[f"attn_mask_v{vi}"] = attn_mask[flat_idx]
                    save_dict[f"t5_input_ids_v{vi}"] = t5_input_ids[flat_idx]
                    save_dict[f"t5_attn_mask_v{vi}"] = t5_attn_mask[flat_idx]
                    if crossattn_emb is not None:
                        save_dict[f"crossattn_emb_v{vi}"] = crossattn_emb[flat_idx]
                _save_safetensors(save_dict, str(cache_path))
                cached += 1
                pbar.update(1)
                pbar.set_postfix_str(f"{img_path.name} ({N}v)")

    pbar.close()
    print(
        f"\nText embedding caching complete: {cached} cached, {skipped} skipped (already existed)"
    )

    text_encoder.to("cpu")
    del text_encoder, llm_adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
