#!/usr/bin/env python3
"""Stack two LoRA-family variants — OrthoInit + T-LoRA — from Python.

`06_frozen_dit_training_build.py` shows the *plainest* programmatic build (default
LoRA, no variant kwargs). Everything beyond that is documented only as the
comment-toggle blocks in `configs/methods/lora.toml`. This script demonstrates
the three facts an embedder (or a bespoke `scripts/` trainer) needs to compose
variants *without* a config file:

  1. Variant selection is **kwargs → resolve_network_spec** — the exact keys the
     TOML carries pass straight through `create_network(**kwargs)`. Here:
       use_ortho_init=True  → OrthoInitLoRAModule (trainable SVD-seeded bases,
                              λ-gated so ΔW=0 at init; distills to plain LoRA).
       use_timestep_mask=True → T-LoRA power-law rank schedule.
     (use_ortho_init=True + use_moe_style would raise in resolve_network_spec —
      impossible combos fail loudly at build, they don't silently degrade.)

  2. T-LoRA is **not a class**. It is the `_timestep_mask` buffer every variant
     inherits from `BaseLoRAModule`, rebound per-step to one shared GPU tensor by
     `LoRANetwork.set_timestep_mask`. Stacking it on OrthoInit is therefore free:
     the mask gates the singular-value gate `lambda_layer` inside the OrthoInit
     forward (`networks/lora_modules/ortho.py`).

  3. The per-step driving is ONE call — `apply_router_conditioning(...)` — which
     `hasattr`-probes the network and fires `set_timestep_mask` / `set_sigma` /
     `set_fei` in a stable order. A bespoke loop should call that, not hand-roll
     the individual setters. For this stack only `set_timestep_mask` does work;
     the rest no-op.

Caveats this script is deliberately demonstrating around:

  - **T-LoRA is training-only.** Inference runs full rank at every `t`; never call
    `set_timestep_mask` in a sampling loop (`docs/methods/timestep_mask.md`).
  - **The schedule barely moves learned effective rank on Anima**
    (`bench/timestep_mask/`). This shows the *mechanism*; don't expect the
    schedule to be a big quality lever — question `network_dim` before tuning
    `alpha_rank_scale`.
  - **OrthoInit distills to a standard LoRA at save** (sqrt-split λ → down/up), so
    the output checkpoint loads anywhere a plain LoRA does. The whole stack is a
    *training-time* composition with no inference-side footprint.
  - The variant matrix (which combos exist) lives in `networks/CLAUDE.md`
    (three-axis table).

No dataset — synthetic latents + text features drive a few real DiT forwards so
the mask is exercised live (and, with --compile, the compile-after-apply ordering
is too). It stops after a handful of steps and prints the live mask rank each one.

    python examples/07_stack_ortho_init_tlora.py --steps 3
    python examples/07_stack_ortho_init_tlora.py --steps 3 --compile
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from library.anima import weights as anima_weights
from library.env import default_checkpoints
from library.runtime.harness import (
    compile_dit_blocks,
    enable_training_grad_ckpt,
    place_dit_for_training,
)
from library.training.forward.router_conditioning import apply_router_conditioning
from networks.lora_anima.factory import create_network

# env ANIMA_DIT (incl. a project-root `.env`) → configs/base.toml → fallback.
DIT = default_checkpoints().dit

# Anima DiT contract (library/anima/weights.py + models.py): 16 latent channels,
# 1024-d cross-attn text features, patch_spatial=2 (latent edge → edge/2 patches).
LATENT_CHANNELS = 16
CROSSATTN_EMB_DIM = 1024


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--network_dim", type=int, default=32, help="max rank R_max")
    p.add_argument("--network_alpha", type=float, default=16.0)
    p.add_argument("--steps", type=int, default=3, help="synthetic training steps")
    p.add_argument("--min_rank", type=int, default=1, help="T-LoRA floor at clean end")
    p.add_argument(
        "--alpha_rank_scale",
        type=float,
        default=1.0,
        help="T-LoRA power-law exponent (1.0 linear, >1 steeper)",
    )
    p.add_argument(
        "--latent_hw", type=int, default=64, help="synthetic latent H=W (÷ 2 = patches)"
    )
    p.add_argument("--text_len", type=int, default=512, help="synthetic text tokens")
    p.add_argument("--blocks_to_swap", type=int, default=0)
    p.add_argument("--grad_ckpt", action="store_true", help="unsloth grad ckpt (CUDA)")
    p.add_argument("--compile", action="store_true", help="torch.compile blocks")
    opts = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # 1. Load the base DiT (identical to example 06).
    model = anima_weights.load_anima_model(
        device=device,
        dit_path=DIT,
        attn_mode="torch",
        loading_device="cpu" if opts.blocks_to_swap > 0 else device,
        dit_weight_dtype=dtype,
    )

    # 2. Attach a fresh network — VARIANT STACKING HAPPENS HERE. These are the
    #    same keys lora.toml carries, passed as kwargs straight to
    #    resolve_network_spec + LoRANetworkCfg.from_kwargs. No config file.
    network = create_network(
        multiplier=1.0,
        network_dim=opts.network_dim,
        network_alpha=opts.network_alpha,
        vae=None,
        text_encoders=[],
        unet=model,
        use_ortho_init=True,  # ortho (init-only) axis → OrthoInitLoRAModule
        use_timestep_mask=True,  # T-LoRA axis — mask gates lambda_layer per step
        min_rank=opts.min_rank,
        alpha_rank_scale=opts.alpha_rank_scale,
    )
    network.apply_to(
        text_encoders=[], unet=model, apply_text_encoder=False, apply_unet=True
    )

    # 3. Place on device (arms the training swap path), then compile LAST — the
    #    apply_to monkey-patches above must already be installed (the same
    #    compile-after-apply invariant build_anima encodes).
    place_dit_for_training(model, device, blocks_to_swap=opts.blocks_to_swap)
    compile_dit_blocks(model, enabled=opts.compile, mode="default")

    # 4. Grad checkpointing (opt-in — unsloth offload needs CUDA), then train().
    enable_training_grad_ckpt(model, enabled=opts.grad_ckpt)
    model.train()

    # 5. Freeze the base DiT; only the adapter trains.
    for param in model.parameters():
        param.requires_grad_(False)
    network.to(device=device, dtype=dtype)
    network.prepare_grad_etc(None, model)  # sets adapter params requires_grad

    trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"\nready to train — {trainable:,} trainable adapter params over "
        f"{len(network.unet_loras)} modules "
        f"({trainable / total * 100:.3f}% of the {total / 1e9:.2f}B-param DiT)"
    )
    print(
        f"stack: OrthoInit (trainable SVD-seeded bases) + T-LoRA "
        f"(R_max={opts.network_dim}, min_rank={opts.min_rank}, "
        f"alpha_rank_scale={opts.alpha_rank_scale})\n"
    )

    optimizer = torch.optim.AdamW(
        [p for p in network.parameters() if p.requires_grad], lr=1e-4
    )

    # 6. A few synthetic steps. Descending t (high noise → clean) so the T-LoRA
    #    schedule's effect on effective rank is visible: low rank at high noise,
    #    full rank at the clean end.
    H = W = opts.latent_hw
    gen = torch.Generator(device=device).manual_seed(0)
    timestep_grid = torch.linspace(0.9, 0.1, opts.steps)

    warmup_step = 0
    for step in range(opts.steps):
        t = timestep_grid[step]
        noisy_latents = torch.randn(
            1, LATENT_CHANNELS, 1, H, W, generator=gen, device=device, dtype=dtype
        )
        timesteps = t.to(device).reshape(1)  # flow t ∈ [0, 1]
        context = torch.randn(
            1,
            opts.text_len,
            CROSSATTN_EMB_DIM,
            generator=gen,
            device=device,
            dtype=dtype,
        )
        padding_mask = torch.ones(1, 1, H, W, device=device, dtype=dtype)
        target = torch.randn(
            1, LATENT_CHANNELS, 1, H, W, generator=gen, device=device, dtype=dtype
        )

        # THE per-step hook — drives set_timestep_mask (+ set_sigma/set_fei, which
        # no-op for this stack). Returns the next warmup_step.
        warmup_step = apply_router_conditioning(
            network=network,
            noisy_model_input=noisy_latents,
            timesteps=timesteps,
            is_train=True,
            warmup_step=warmup_step,
            max_train_steps=opts.steps,
        )

        # The mask is now live on every adapted Linear (one shared GPU buffer).
        # Its sum is the effective rank gating the OrthoInit λ this step.
        mask = network._shared_timestep_mask
        eff_rank = int(mask.sum().item())
        print(
            f"step {step + 1}: t={t.item():.2f} → effective rank "
            f"{eff_rank}/{opts.network_dim}"
        )

        # autocast(bf16) is what the real trainer runs under — it also keeps the
        # adapter's compute-dtype policy (org_forwarded.dtype) on its intended
        # bf16 path. Enabled on both CUDA and CPU so the example is portable.
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype):
            model_pred = model(
                noisy_latents, timesteps, context, padding_mask=padding_mask
            )
            loss = F.mse_loss(model_pred.float(), target.float())
        loss.backward()
        optimizer.step()

    print("\nstack trained for a few synthetic steps — save with the usual LoRA")
    print("pipeline (OrthoInit distills to standard LoRA at save). See example 02.")


if __name__ == "__main__":
    main()
