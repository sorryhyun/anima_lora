#!/usr/bin/env python3
"""Cache text encoder (Qwen3) outputs for all captioned images in a dataset directory.

Reads .txt caption sidecars, tokenizes with Qwen3 + T5, encodes through the
Qwen3 text encoder, and optionally runs the LLM adapter to produce crossattn_emb.
Saves results as *_anima_te.safetensors alongside each image (or under
``--cache_dir``).

Supports caption shuffle variants: with --caption_shuffle_variants N, generates
N variants per image and caches them all in one file. v0 is the pristine
original caption (no shuffle, no dropout); v1..v{N-1} are smart-shuffled and,
if --caption_tag_dropout_rate > 0, have non-prefix tags independently dropped
at that rate, and if --caption_tag_randomize_rate > 0, have surviving tags'
identities erased (replaced by meaningless tokens, prefix included) at that
rate. The strategy loader picks v0 with 20% probability and uniform
v1..v{N-1} with 80% probability when use_shuffled_caption_variants is on.

The encode loop lives in ``library/preprocess/text.py``; this file is argparse +
model load + the one-time uncond sidecar staging.
"""

import argparse
from pathlib import Path

import torch


from library.preprocess import (
    cache_text_embeddings,
    count_pending_text,
    tqdm_progress,
)
from library.runtime.argparse_groups import add_io_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_io_args(
        parser,
        cache_noun="text-encoder caches",
        include_batch_size=True,
        batch_size_default=16,
    )
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
        help="Path to T5 tokenizer (default: library/anima/configs/t5_old/)",
    )
    parser.add_argument(
        "--caption_shuffle_variants",
        type=int,
        default=0,
        help=(
            "Number of caption variants per image (0 = single caption). v0 is "
            "the pristine original; v1..v{N-1} are shuffled (and tag-dropped "
            "if --caption_tag_dropout_rate > 0)."
        ),
    )
    parser.add_argument(
        "--caption_tag_dropout_rate",
        type=float,
        default=0.0,
        help=(
            "Per-tag dropout probability applied to v1..v{N-1} only. Tags up "
            "to and including the first @artist marker are never dropped. "
            "Ignored when --caption_shuffle_variants <= 0."
        ),
    )
    parser.add_argument(
        "--caption_tag_randomize_rate",
        type=float,
        default=0.0,
        help=(
            "Lexinvariant tag regularization (arXiv:2305.16349): per-tag "
            "probability of *erasing a tag's identity* — replacing it with a "
            "fresh meaningless token while keeping its slot — applied to "
            "v1..v{N-1} only. Unlike dropout this DOES include the @artist "
            "prefix (the concept/trigger tag is the target). Section headers "
            "and the @no-artist sentinel are never randomized. Paper's prior is "
            "p~=0.2. Ignored when --caption_shuffle_variants <= 0. Build a "
            "randomized cache into a SEPARATE --cache_dir to A/B against the "
            "baseline cache."
        ),
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=500_000,
        help=(
            "Skip images with fewer than this many pixels (default: 500_000 "
            "= 0.5MP). Mirrors the same filter in the resize stage (make preprocess-resize) "
            "so TE caches don't accumulate for images that get dropped at "
            "resize time. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--path_pattern",
        "--path-pattern",
        dest="path_pattern",
        default="*",
        help=(
            "Only cache images whose path relative to --dir matches this "
            "fnmatch glob. Use | to separate alternatives. Default: *"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Re-encode every cache even if it already exists. Use after changing "
            "the variant count / dropout / randomize rate, which the existence "
            "check can't detect (existing caches are otherwise skipped)."
        ),
    )
    parser.add_argument(
        "--match_images_from",
        type=str,
        default=None,
        help=(
            "Only cache text for source images whose relative stem also exists "
            "under this image directory. Used by the preprocess chain to mirror "
            "the already-resized/curated image set while still reading captions "
            "from the original source directory."
        ),
    )
    args = parser.parse_args()

    from library.anima import weights as anima_utils
    from library.anima.strategy import AnimaTextEncodingStrategy, AnimaTokenizeStrategy

    data_dir = Path(args.dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    keep_rel_stems = None
    if args.match_images_from:
        match_dir = Path(args.match_images_from)
        if match_dir.is_dir():
            from library.preprocess import walk_images

            keep_rel_stems = {
                p.relative_to(match_dir).with_suffix("").as_posix()
                for p in walk_images(
                    match_dir,
                    recursive=args.recursive,
                    pattern=args.path_pattern,
                )
            }

    # Pre-flight: skip the (slow) Qwen3 + LLM-adapter load when every TE cache
    # already exists. When --dit is set the run also stages the one-time uncond
    # sidecar (needs the loaded model), so a missing sidecar still forces a load.
    from library.anima.uncond import default_uncond_path

    pending, total = count_pending_text(
        data_dir,
        cache_dir=cache_dir,
        recursive=args.recursive,
        path_pattern=args.path_pattern,
        keep_rel_stems=keep_rel_stems,
        min_pixels=args.min_pixels,
        overwrite=args.overwrite,
    )
    uncond_needed = bool(args.dit) and not default_uncond_path().exists()
    if pending == 0 and not uncond_needed:
        print(
            f"Text embedding caching: all {total} captions already cached "
            "— skipping text-encoder load."
        )
        return

    from library.runtime.backend import warn_if_cuda_unavailable

    warn_if_cuda_unavailable(torch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = args.caption_shuffle_variants

    if pending:
        print(f"{pending}/{total} captions need encoding.")
    print(f"Loading Qwen3 text encoder from {args.qwen3} ...")
    text_encoder, qwen3_tokenizer = anima_utils.load_qwen3_text_encoder(
        args.qwen3, dtype=torch.bfloat16, device=str(device)
    )
    t5_tokenizer = anima_utils.load_t5_tokenizer(args.t5_tokenizer_path)

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

    # Stage the T5("") CFG-uncond sidecar while the models are on device; every
    # training/distill run reuses this file (matches library/inference/text.py).
    # Skipped without --dit (no llm_adapter → can't produce crossattn here).
    if llm_adapter is not None:
        from library.preprocess.uncond import (
            DEFAULT_UNCOND_DIR,
            stage_uncond_sidecar_with_models,
        )

        stage_uncond_sidecar_with_models(
            DEFAULT_UNCOND_DIR,
            text_encoder,
            tokenize_strategy,
            encoding_strategy,
            llm_adapter,
            device=device,
            overwrite=bool(getattr(args, "force_recache_uncond", False)),
        )

    tag_dropout_rate = float(args.caption_tag_dropout_rate)
    tag_randomize_rate = float(args.caption_tag_randomize_rate)
    if N > 0:
        print(
            f"Caption shuffle variants: {N} "
            f"(v0=pristine, v1..v{N - 1}=shuffled"
            + (
                f" + tag dropout p={tag_dropout_rate:.3f}"
                if tag_dropout_rate > 0.0
                else ""
            )
            + (
                f" + identity-randomize p={tag_randomize_rate:.3f}"
                if tag_randomize_rate > 0.0
                else ""
            )
            + ")"
        )
    elif tag_dropout_rate > 0.0 or tag_randomize_rate > 0.0:
        print(
            "warn: --caption_tag_dropout_rate / --caption_tag_randomize_rate "
            "ignored because --caption_shuffle_variants <= 0 (single-variant cache)."
        )

    stats = cache_text_embeddings(
        data_dir,
        tokenize_strategy,
        encoding_strategy,
        text_encoder,
        llm_adapter=llm_adapter,
        device=device,
        cache_dir=cache_dir,
        recursive=args.recursive,
        path_pattern=args.path_pattern,
        keep_rel_stems=keep_rel_stems,
        batch_size=args.batch_size,
        caption_shuffle_variants=N,
        caption_tag_dropout_rate=tag_dropout_rate,
        caption_tag_randomize_rate=tag_randomize_rate,
        min_pixels=args.min_pixels,
        overwrite=args.overwrite,
        progress=tqdm_progress("Caching text embeddings"),
    )
    print(
        f"\nText embedding caching complete: {stats.written} cached, "
        f"{stats.skipped} skipped (already existed)"
    )

    text_encoder.to("cpu")
    del text_encoder, llm_adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
