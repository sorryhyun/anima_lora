#!/usr/bin/env python3
"""Cache VAE latents for all images in a dataset directory.

Encodes images through the Qwen Image VAE and saves latent caches (.npz)
alongside the images (or under ``--cache_dir``).  Skips already-cached
entries (idempotent).

The walk → group-by-resolution → encode → save loop lives in
``library/preprocess/latents.py``; this file is argparse + VAE load + reporting.
"""

import argparse
from pathlib import Path

import torch


from library.preprocess import (
    cache_demoted_latents,
    cache_latents,
    count_pending_demoted,
    count_pending_latents,
    tqdm_progress,
)
from library.runtime.argparse_groups import add_io_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_io_args(
        parser,
        cache_noun="latent caches",
        include_batch_size=True,
        batch_size_default=2,
    )
    parser.add_argument("--vae", type=str, required=True, help="Path to VAE weights")
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=64,
        help="VAE spatial chunk size (default: 64)",
    )
    parser.add_argument(
        "--disable_cache",
        action="store_true",
        default=True,
        help="Disable VAE internal cache (default: True)",
    )
    # torch.compile(dynamic=True) on the encoder: ~33% faster steady-state encode
    # at +~0.8GB peak, after a ~70s one-time warmup (dynamic shapes mean ONE
    # compile covers every free-fit (W,H) — no per-shape recompile). Opt-in
    # because the warmup is a net loss for tiny incremental re-caches. Forces
    # --chunk_size 0: the chunked-conv Python loop is compile-hostile (never
    # finishes compiling). See scratch bench 2026-06-28.
    parser.add_argument(
        "--compile_vae",
        action="store_true",
        help="torch.compile the VAE encoder (dynamic=True). Forces chunk_size=0.",
    )
    # 2D VAE fold is ON by default: image-only pipeline, ~2x faster encode at
    # ~0.65-0.7x peak VRAM, latents equivalent within bf16 noise. See
    # _archive/bench/qwen_vae_2d/. Opt out with --no_vae_2d for the stock 3D causal VAE.
    parser.add_argument(
        "--qwen_image_vae_2d",
        "--vae_2d",
        dest="vae_2d",
        action="store_true",
        default=True,
        help="Fold the causal Conv3d VAE into 2D convs (image-only). Default ON.",
    )
    parser.add_argument(
        "--no_vae_2d",
        "--qwen_image_vae_3d",
        dest="vae_2d",
        action="store_false",
        help="Use the stock 3D causal-Conv3d VAE instead of the 2D fold.",
    )
    # Encode in fp32 instead of bf16. The fp32 save already happens regardless;
    # this also encodes in fp32, removing the (structured but ~17 dB-below-recon)
    # bf16 accumulation error and making the 2D fold bit-exact. See
    # _archive/bench/qwen_vae_2d/encode_dtype_probe.py — quality-neutral, hygiene only.
    parser.add_argument(
        "--no_half_vae",
        "--fp32_vae",
        dest="no_half_vae",
        action="store_true",
        help="Encode latents in fp32 (default: bf16). Bit-exact 2D fold; slower.",
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
            "Re-encode every latent even if its (W,H) cache already exists. Use "
            "after a VAE / encode-dtype change, which the per-resolution skip "
            "can't detect (existing caches are otherwise skipped)."
        ),
    )
    parser.add_argument(
        "--sigma_demote",
        default=None,
        metavar="NATIVE:DEMOTE",
        help=(
            "Emit σ-demote sibling latents instead of the normal pass "
            "(sigma_lowres Phase 1b): for each image in NATIVE's free-fit band, "
            "downscale the resized PNG to its DEMOTE-tier bucket, VAE-encode, "
            "and append a demoted_{H}x{W} key inside the existing native npz. "
            "Requires the native latents to be cached first. E.g. 1024:896 "
            "(the only measured-safe route)."
        ),
    )
    args = parser.parse_args()

    from library.models import qwen_vae as qwen_image_autoencoder_kl

    data_dir = Path(args.dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    demote_route = None
    if args.sigma_demote:
        native_s, _, demote_s = args.sigma_demote.partition(":")
        try:
            demote_route = (int(native_s), int(demote_s))
        except ValueError:
            # A comma list belongs to the task runner (which expands it into one
            # pass per route) — say so instead of dying inside int().
            raise SystemExit(
                f"--sigma_demote expects a single NATIVE:DEMOTE route, got "
                f"{args.sigma_demote!r}. For several routes run one pass each, "
                f"or use `make preprocess-demote` (it expands the comma list in "
                f"configs/preprocess.toml)."
            ) from None

    # Pre-flight: a fully-cached dataset needs no VAE — skip the (slow) load.
    if demote_route is not None:
        pending, total = count_pending_demoted(
            data_dir,
            native_edge=demote_route[0],
            demote_edge=demote_route[1],
            cache_dir=cache_dir,
            recursive=args.recursive,
            path_pattern=args.path_pattern,
            overwrite=args.overwrite,
        )
        if pending == 0:
            print(
                f"σ-demote caching: all {total} eligible images already carry "
                "their demoted key — skipping VAE load."
            )
            return
    else:
        pending, total = count_pending_latents(
            data_dir,
            cache_dir=cache_dir,
            recursive=args.recursive,
            path_pattern=args.path_pattern,
            overwrite=args.overwrite,
        )
        if pending == 0:
            print(
                f"Latent caching: all {total} images already cached — skipping VAE load."
            )
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.no_half_vae else torch.bfloat16

    # Chunking is incompatible with compile (shape-dependent Python loop), and
    # buys no memory at the resized resolutions anyway — drop it when compiling.
    chunk_size = args.chunk_size
    if args.compile_vae and chunk_size:
        print("--compile_vae: forcing chunk_size=0 (chunking is compile-hostile).")
        chunk_size = 0

    print(f"{pending}/{total} images need latents.")
    print(f"Loading VAE from {args.vae} (encode dtype: {dtype}) ...")
    vae = qwen_image_autoencoder_kl.load_vae(
        args.vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=chunk_size,
        disable_cache=args.disable_cache,
    )
    vae.to(device, dtype=dtype)
    if args.vae_2d:
        n = vae.convert_to_2d()
        print(f"Folded VAE to 2D (image-only): {n} Conv3d -> Conv2d")
    vae.requires_grad_(False)
    vae.eval()
    if args.compile_vae:
        # dynamic=True: one compile covers every free-fit (W,H); ~70s warmup on
        # the first batch, then ~33% faster encode.
        print("Compiling VAE encoder (dynamic=True) — first batch warms up (~70s)...")
        vae.encoder = torch.compile(vae.encoder, dynamic=True)

    if demote_route is not None:
        stats = cache_demoted_latents(
            data_dir,
            vae,
            native_edge=demote_route[0],
            demote_edge=demote_route[1],
            cache_dir=cache_dir,
            recursive=args.recursive,
            path_pattern=args.path_pattern,
            batch_size=args.batch_size,
            progress=tqdm_progress("Caching demoted latents"),
            overwrite=args.overwrite,
        )
        print(
            f"\nσ-demote caching complete: {stats.written} emitted, "
            f"{stats.skipped} skipped (already present), "
            f"{stats.failed} failed (no native npz / undecodable)"
        )
    else:
        stats = cache_latents(
            data_dir,
            vae,
            cache_dir=cache_dir,
            recursive=args.recursive,
            path_pattern=args.path_pattern,
            batch_size=args.batch_size,
            progress=tqdm_progress("Caching latents"),
            overwrite=args.overwrite,
        )
        print(
            f"\nLatent caching complete: {stats.written} cached, "
            f"{stats.skipped} skipped (already existed)"
        )

    vae.to("cpu")
    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
