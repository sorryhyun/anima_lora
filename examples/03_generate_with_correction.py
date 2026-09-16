#!/usr/bin/env python3
"""Generate with a training-free sampler correction (SMC-CFG or Spectrum).

Same flow as 01_generate.py, plus `extra_argv` — the channel for method knobs
`GenerationRequest` doesn't model as typed fields. Anything you'd pass on the
`inference.py` command line —
`--smc_cfg`, `--spectrum`, their sub-knobs — goes here as verbatim CLI tokens,
and `.to_args()` feeds them through `inference.parse_args` so the generation
code sees them exactly as a CLI run would. `extra_argv` is appended last, so it
can also override a structured field.

Two training-free corrections are wired up:

  * **smc_cfg**  — Sliding-Mode Control CFG (arXiv:2603.03281). `--smc_cfg`
                   turns it on; `--smc_cfg_alpha` is the adaptive gain. Modifies
                   the cond/uncond combine; no extra forwards. See
                   docs/inference/smc_cfg.md.
  * **spectrum** — Chebyshev feature-forecasting acceleration. `--spectrum`
                   turns it on; cached steps skip the transformer blocks.
                   `--spectrum_warmup` is the full-forward warmup count.

Any other untyped knob works the same way.

Run from the repo root (anima_lora/):

    python examples/03_generate_with_correction.py --correction spectrum
    python examples/03_generate_with_correction.py --correction spectrum --spectrum_warmup 6
    python examples/03_generate_with_correction.py --correction smc_cfg --smc_cfg_alpha 0.2
    python examples/03_generate_with_correction.py --correction none   # baseline
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
from library.runtime.device import clean_memory_on_device

# env (ANIMA_DIT / ANIMA_VAE / ANIMA_TEXT_ENCODER, incl. a project-root `.env`)
# → configs/base.toml → built-in fallbacks. See `.env.example`.
_ckpt = default_checkpoints()
DIT = _ckpt.dit
VAE = _ckpt.vae
TEXT_ENCODER = _ckpt.text_encoder


def correction_argv(opts: argparse.Namespace) -> list[str]:
    """Build the verbatim `inference.py` tokens for the chosen correction.

    The same tokens you'd pass to `python inference.py …`.
    """
    if opts.correction == "smc_cfg":
        # store_true flag + one sub-knob (adaptive gain α).
        return ["--smc_cfg", "--smc_cfg_alpha", str(opts.smc_cfg_alpha)]
    if opts.correction == "spectrum":
        return ["--spectrum", "--spectrum_warmup", str(opts.spectrum_warmup)]
    return []  # "none" → plain sampler, no correction


def build_request(opts: argparse.Namespace) -> GenerationRequest:
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
        # Untyped method flags as raw CLI tokens.
        extra_argv=correction_argv(opts),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--correction",
        choices=["smc_cfg", "spectrum", "none"],
        default="spectrum",
        help="Which training-free correction to enable via extra_argv.",
    )
    p.add_argument(
        "--smc_cfg_alpha",
        type=float,
        default=0.2,
        help="SMC-CFG adaptive gain α. Used when --correction smc_cfg.",
    )
    p.add_argument(
        "--spectrum_warmup",
        type=int,
        default=6,
        help="Spectrum full-forward warmup steps. Used when --correction spectrum.",
    )
    p.add_argument(
        "--prompt", default="a red fox sitting in a snowy forest, golden hour"
    )
    p.add_argument("--save_path", default="output/tests/example_03.png")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=3.5)
    p.add_argument(
        "--size", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W")
    )
    p.add_argument("--seed", type=int, default=42)
    opts = p.parse_args()

    request = build_request(opts)
    extra = request.extra_argv
    print(f"correction={opts.correction!r}  extra_argv={list(extra)}")

    args = request.to_args()  # routes extra_argv through inference.parse_args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    gen_settings = get_generation_settings(args)
    latent = generate(args, gen_settings)
    clean_memory_on_device(device)

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
    print(f"saved → {args.save_path}  (correction: {opts.correction})")


if __name__ == "__main__":
    main()
