#!/usr/bin/env python3
"""Text-to-image generation with the Anima base model — optionally with LoRA.

The programmatic equivalent of:

    python inference.py --dit <…> --vae <…> --text_encoder <…> \
        --prompt "…" --save_path output/tests/example_01.png

Run from the repo root (anima_lora/) after `make download-models`:

    # base model, no adapter
    python examples/01_generate.py --prompt "a red fox in a snowy forest"

    # with one or more trained LoRA adapters attached
    python examples/01_generate.py \
        --lora_weight output/ckpt/my_lora.safetensors \
        --prompt "a portrait of <subject>"

generate_image() shows the three steps: settings → generate → decode.

Adapters are optional: pass one or more paths to `--lora_weight` (with matching
`--multiplier` values) to stack them. generate() → load_dit_model() builds the
network from the checkpoint and either merges it (plain LoRA / OrthoLoRA /
T-LoRA) or keeps it live (HydraLoRA / FeRA) based on the checkpoint's metadata.
"""

from __future__ import annotations

import argparse


import torch

# `GenerationRequest.to_args()` feeds the request through `inference.parse_args`,
# so every optional knob gets its default (a hand-built Namespace misses them).
from anima_lora import (
    GenerationRequest,
    default_checkpoints,
    generate,
    get_generation_settings,
    load_vae,
    save_output,
)
from library.runtime.device import clean_memory_on_device

# env (ANIMA_DIT / ANIMA_VAE / ANIMA_TEXT_ENCODER, incl. a project-root `.env`)
# → configs/base.toml → built-in fallbacks. See `.env.example`.
_ckpt = default_checkpoints()
DIT = _ckpt.dit
VAE = _ckpt.vae
TEXT_ENCODER = _ckpt.text_encoder


def build_request(opts: argparse.Namespace) -> GenerationRequest:
    """Describe the generation as a typed request (the CLI is one consumer).

    lora_weight / lora_multiplier accept sequences; empty lists mean a plain
    base-model run.
    """
    return GenerationRequest(
        dit=DIT,
        vae=VAE,
        text_encoder=TEXT_ENCODER,
        prompt=opts.prompt,
        save_path=opts.save_path,
        infer_steps=opts.steps,
        guidance_scale=opts.cfg,
        image_size=tuple(opts.size),  # (H, W)
        seed=opts.seed,
        lora_weight=opts.lora_weight,
        lora_multiplier=opts.multiplier,
    )


def generate_image(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # 1. Settings carry the device + DiT weight dtype (bf16).
    gen_settings = get_generation_settings(args)

    # 2. generate() loads the DiT (attaching any adapter from args.lora_weight),
    #    encodes the prompt (max-padded — trimming gives black images), and runs
    #    the sampler. Returns the clean latent.
    latent = generate(args, gen_settings)

    # 3. Free the DiT before bringing up the VAE to keep peak VRAM down.
    clean_memory_on_device(device)

    # 4. Decode latent → PNG (+ generation metadata) via save_output.
    vae = load_vae(
        args.vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=args.vae_chunk_size,
        disable_cache=args.vae_disable_cache,
        dtype=torch.bfloat16,
        eval=True,
    )
    save_output(args, vae, latent, device)
    if args.lora_weight:
        print(f"saved → {args.save_path}  (adapters: {', '.join(args.lora_weight)})")
    else:
        print(f"saved → {args.save_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--prompt", default="a red fox sitting in a snowy forest, golden hour"
    )
    p.add_argument("--save_path", default="output/tests/example_01.png")
    p.add_argument(
        "--lora_weight",
        nargs="+",
        default=[],
        help="zero or more adapter .safetensors (omit for a base-model run)",
    )
    p.add_argument("--multiplier", type=float, nargs="+", default=[1.0])
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=3.5)
    p.add_argument(
        "--size", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W")
    )
    p.add_argument("--seed", type=int, default=42)
    opts = p.parse_args()

    # Broadcast a single multiplier across all adapters.
    if len(opts.multiplier) == 1 and len(opts.lora_weight) > 1:
        opts.multiplier = opts.multiplier * len(opts.lora_weight)

    generate_image(build_request(opts).to_args())


if __name__ == "__main__":
    main()
