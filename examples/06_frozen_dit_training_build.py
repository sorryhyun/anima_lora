#!/usr/bin/env python3
"""Build a frozen DiT + fresh adapter for a *training* run — the harness helpers.

`build_anima` (shown indirectly by 01/04) owns the inference path: it loads an
*existing* adapter checkpoint and applies it. When you write a distillation /
fine-tuning trainer you instead attach a *fresh, untrained* network and drive
your own optimizer — so you can't call `build_anima` wholesale. The three
composable helpers in `library.runtime.harness` factor out the model-side
boilerplate the trainers used to copy verbatim:

  - place_dit_for_training(model, device, blocks_to_swap=...)
        block-swap placement (or a plain .to(device)) with the *training* swap
        path armed (forward + backward block movement).
  - compile_dit_blocks(model, enabled=..., cache_size_limit=..., mode=...)
        native-shape-flatten torch.compile of each block._forward. COMPILE LAST
        — run it only after the network's apply_to, or it traces the wrong
        forward (the same invariant build_anima encodes).
  - enable_training_grad_ckpt(model, enabled=...)
        unsloth CPU-offload gradient checkpointing (model must stay in train()).

This is exactly the sequence scripts/distill_{mod,turbo}.py run; this script
distills it to the smallest demonstrable build (no dataset, no optimizer step —
it stops once the model is ready to train and prints the trainable-param split).

    python examples/06_frozen_dit_training_build.py
    python examples/06_frozen_dit_training_build.py --blocks_to_swap 16 --compile
"""

from __future__ import annotations

import argparse


import torch

from library.anima import weights as anima_weights
from library.env import default_checkpoints
from library.runtime.harness import (
    compile_dit_blocks,
    enable_training_grad_ckpt,
    place_dit_for_training,
)
from networks.lora_anima.factory import create_network

# env ANIMA_DIT (incl. a project-root `.env`) → configs/base.toml → fallback.
DIT = default_checkpoints().dit


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--network_dim", type=int, default=16)
    p.add_argument("--network_alpha", type=float, default=16.0)
    p.add_argument("--blocks_to_swap", type=int, default=0)
    p.add_argument("--grad_ckpt", action="store_true", default=True)
    p.add_argument("--no_grad_ckpt", dest="grad_ckpt", action="store_false")
    p.add_argument("--compile", action="store_true", help="torch.compile blocks")
    opts = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # 1. Load the base DiT. With block swap, stage the swapped blocks on CPU.
    model = anima_weights.load_anima_model(
        device=device,
        dit_path=DIT,
        attn_mode="torch",
        loading_device="cpu" if opts.blocks_to_swap > 0 else device,
        dit_weight_dtype=dtype,
    )

    # 2. Attach a fresh (untrained) LoRA network — this is what build_anima
    #    *can't* do for you (it loads weights from a checkpoint instead).
    network = create_network(
        multiplier=1.0,
        network_dim=opts.network_dim,
        network_alpha=opts.network_alpha,
        vae=None,
        text_encoders=[],
        unet=model,
    )
    network.apply_to(
        text_encoders=[], unet=model, apply_text_encoder=False, apply_unet=True
    )

    # 3. Place on device (arms the training swap path), then compile LAST — the
    #    apply_to monkey-patches above must already be installed.
    place_dit_for_training(model, device, blocks_to_swap=opts.blocks_to_swap)
    compile_dit_blocks(model, enabled=opts.compile, mode="default")

    # 4. Grad checkpointing, then train() — Block.forward gates checkpointing on
    #    self.training, so the train() call has to come after.
    enable_training_grad_ckpt(model, enabled=opts.grad_ckpt)
    model.train()

    # 5. Freeze the base DiT; only the adapter trains. apply_to add_module'd the
    #    LoRA submodules onto the unet, so freeze-all then re-enable the network.
    for param in model.parameters():
        param.requires_grad_(False)
    network.to(device=device, dtype=dtype)
    network.prepare_grad_etc(None, model)  # sets the adapter params requires_grad

    trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"\nready to train — {trainable:,} trainable adapter params over "
        f"{len(network.unet_loras)} modules "
        f"({trainable / total * 100:.3f}% of the {total / 1e9:.2f}B-param DiT)"
    )
    print(
        "plug in your dataset + optimizer from here (see scripts/distill_turbo/setup.py)."
    )


if __name__ == "__main__":
    main()
