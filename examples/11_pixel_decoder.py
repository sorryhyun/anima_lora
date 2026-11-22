#!/usr/bin/env python3
"""Decode with the pixel head instead of the VAE.

The programmatic equivalent of:

    python inference.py … --decoder pixel_head [--pixel_head <file>] [--pixel_scale 2]

The pixel head (~25M params, `docs/anima/inference/pixel_head.md`) reads the DiT's
last-step features and writes RGB directly — no VAE is loaded. Its weights are the
catalog rows `anima_pixel_head` (×1) and `anima_pixel_head_x2` (×2, a 1024 render
comes out at 2048), fetched on first use.

Run from the repo root (anima_lora/) after `make download-models`:

    python examples/11_pixel_decoder.py
    python examples/11_pixel_decoder.py --scale 2                 # the ×2 head
    python examples/11_pixel_decoder.py --compare_vae             # also VAE-decode the same latent
    python examples/11_pixel_decoder.py --sampler er_sde --cns --compare_vae
    python examples/11_pixel_decoder.py --turbo_lora models/loras/anima_turbo_4step.safetensors

Three steps, same as 01 but with a different decode:

  1. `GenerationRequest(decoder="pixel_head")` — `generate()` calls
     `ensure_pixel_decoder`, which hangs the head and its feature tap on the DiT
     (`anima._pixel_decoder`) before sampling.
  2. `generate(args, settings, shared)` — pass a `shared` dict so the DiT stays
     reachable afterwards (`shared["model"]`); the tap records the last step.
  3. `pixel_decode(args, dit)` → (B, 3, H, W) in [-1, 1] on the CPU, saved with
     `save_decoded`.

The head was trained at fixed decode points: the base DiT at shift-3 Euler 20 … 50
steps, and the 4-step turbo LoRA at CFG 1 (×1 head only). Elsewhere it still decodes
and logs a warning. `--cfgpp` / `--smc_cfg` warn (the feature is combined as plain CFG);
`--tiled_diffusion` is refused.
"""

from __future__ import annotations

import argparse

import torch

from anima_lora import (
    GenerationRequest,
    default_checkpoints,
    generate,
    get_generation_settings,
    load_vae,
    save_output,
)
from anima_lora.inference import pixel_decode, save_decoded
from library.runtime.device import clean_memory_on_device

_ckpt = default_checkpoints()
DIT = _ckpt.dit
VAE = _ckpt.vae
TEXT_ENCODER = _ckpt.text_encoder


def build_request(opts: argparse.Namespace) -> GenerationRequest:
    turbo = opts.turbo_lora is not None
    return GenerationRequest(
        dit=DIT,
        vae=VAE,  # only read by --compare_vae
        text_encoder=TEXT_ENCODER,
        prompt=opts.prompt,
        save_path=opts.save_dir,
        # Base: shift-3 Euler at 28 steps (inside the head's 20 … 50 range).
        # Turbo: 4 steps at CFG 1, the head's second trained point.
        infer_steps=4 if turbo else opts.steps,
        guidance_scale=1.0 if turbo else opts.cfg,
        flow_shift=3.0,
        sampler=opts.sampler,
        # CNS recolors ER-SDE's injected noise; untyped, so it rides extra_argv.
        extra_argv=["--cns", "auto"] if opts.cns else (),
        image_size=tuple(opts.size),  # (H, W)
        seed=opts.seed,
        lora_weight=[opts.turbo_lora] if turbo else None,
        lora_multiplier=1.0 if turbo else None,
        decoder="pixel_head",
        pixel_head=opts.pixel_head,  # None → catalog file for the scale
        pixel_scale=opts.scale,  # None → the file's own scale
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--prompt", default="a red fox sitting in a snowy forest, golden hour"
    )
    p.add_argument("--save_dir", default="output/tests/example_11")
    p.add_argument(
        "--scale",
        type=int,
        choices=[1, 2],
        default=None,
        help="1 or 2; unset decodes at the head file's own scale. "
        "With no --pixel_head, 2 takes the catalog's ×2 file.",
    )
    p.add_argument(
        "--pixel_head",
        default=None,
        help="a head .safetensors (default: the catalog row)",
    )
    p.add_argument(
        "--turbo_lora",
        default=None,
        help="4-step turbo LoRA; sets 4 steps and CFG 1 (the ×1 head's turbo point)",
    )
    p.add_argument(
        "--compare_vae",
        action="store_true",
        help="also decode the same latent with the VAE, saved beside it",
    )
    p.add_argument(
        "--sampler",
        choices=["euler", "er_sde"],
        default="euler",
        help="er_sde ends on the same x̂0 step as euler, so the head reads it the same way",
    )
    p.add_argument(
        "--cns",
        action="store_true",
        help="CNS-recolored ER-SDE noise (--cns auto); needs --sampler er_sde",
    )
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument(
        "--size", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W")
    )
    p.add_argument("--seed", type=int, default=42)
    opts = p.parse_args()

    args = build_request(opts).to_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # 1–2. Sample; the head and tap are attached inside generate().
    shared: dict = {}
    latent = generate(args, get_generation_settings(args), shared)

    # 3. Pixel head readout of the last step → PNG named <time>_<seed>_pixel.png.
    pixels = pixel_decode(args, shared["model"])  # (B, 3, s·H, s·W), [-1, 1]
    save_decoded(args, latent, pixels[0], original_base_name="pixel")
    print(f"pixel head: {tuple(pixels.shape[-2:])} → {args.save_path}")

    if opts.compare_vae:
        # Free the DiT first: the VAE decode does not need it.
        shared.clear()
        clean_memory_on_device(device)
        vae = load_vae(
            args.vae,
            device="cpu",
            disable_mmap=True,
            spatial_chunk_size=args.vae_chunk_size,
            disable_cache=args.vae_disable_cache,
            dtype=torch.bfloat16,
            eval=True,
            vae_2d=args.vae_2d,
        )
        save_output(args, vae, latent, device, original_base_name="vae")
        print(f"VAE: same latent → {args.save_path}")


if __name__ == "__main__":
    main()
