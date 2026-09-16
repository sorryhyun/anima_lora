#!/usr/bin/env python3
"""Access the three core models directly — the primitives a scripts/ tool builds on.

inference.py and train.py wrap these loaders; a custom script uses them directly:

  - DiT  : library.anima.weights.load_anima_model()
  - VAE  : library.models.qwen_vae.load_vae()
  - Text : library.inference.models.load_text_encoder()  (Qwen3)

This script loads all three, then encodes a prompt to the DiT-ready cross-attn
embedding via prepare_text_inputs(), which applies the text-encoder padding
invariant (max-pad to 512). Encoding needs the DiT too: it projects the encoder
hidden states through `_preprocess_text_embeds`.

    python examples/04_load_models.py --prompt "a lighthouse at dusk"
"""

from __future__ import annotations

import argparse


import torch

from library.anima import weights as anima_weights
from library.env import default_checkpoints
from library.inference.models import load_text_encoder
from library.inference.text import (
    MAX_CROSSATTN_TOKENS,
    ensure_text_strategies,
    prepare_text_inputs,
)
from library.models import qwen_vae

# env (ANIMA_DIT / ANIMA_VAE / ANIMA_TEXT_ENCODER, incl. a project-root `.env`)
# → configs/base.toml → built-in fallbacks. See `.env.example`.
_ckpt = default_checkpoints()
DIT = _ckpt.dit
VAE = _ckpt.vae
TEXT_ENCODER = _ckpt.text_encoder


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", default="a lighthouse at dusk, dramatic clouds")
    opts = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- DiT ----------------------------------------------------------------
    # load_anima_model takes explicit paths/dtypes (no args namespace). attn_mode
    # "torch" is the portable default; "flash"/"flex" need the matching kernels.
    dit = anima_weights.load_anima_model(
        device=device,
        dit_path=DIT,
        attn_mode="torch",
        loading_device=device,
        dit_weight_dtype=torch.bfloat16,
    )
    dit.eval()
    n_dit = sum(p.numel() for p in dit.parameters())
    print(
        f"DiT   : {type(dit).__name__}  {n_dit / 1e9:.2f}B params  patch_spatial={dit.patch_spatial}"
    )

    # --- VAE ----------------------------------------------------------------
    vae = qwen_vae.load_vae(
        VAE, device=device, disable_mmap=True, dtype=torch.bfloat16, eval=True
    )
    print(f"VAE   : {type(vae).__name__}  z_dim={vae.z_dim}")

    # --- Text encoder + encode a prompt -------------------------------------
    # Prompt encoding goes through two process-global strategy singletons
    # (library/anima/strategy.py). ensure_text_strategies() installs them from
    # the text-encoder path (no-op if already set; prepare_text_inputs() calls it
    # internally) and returns them. Without them tokenize() fails on None.
    tokenize_strategy, encoding_strategy = ensure_text_strategies(
        TEXT_ENCODER, max_length=MAX_CROSSATTN_TOKENS
    )
    print(
        f"Strat : {type(tokenize_strategy).__name__} + "
        f"{type(encoding_strategy).__name__}"
    )

    # Loading the encoder needs only its path.
    text_encoder = load_text_encoder(
        text_encoder=TEXT_ENCODER,
        dtype=torch.bfloat16,
        device=device,
    )
    text_encoder.eval()
    print(f"Text  : {type(text_encoder).__name__} (Qwen3)")

    # prepare_text_inputs returns (context, context_null); context['embed'][0] is
    # the cross-attn embedding the DiT consumes, max-padded to MAX_CROSSATTN_TOKENS.
    # shared_models hands in the loaded encoder so it isn't reloaded.
    context, _context_null = prepare_text_inputs(
        device=device,
        anima=dit,
        prompt=opts.prompt,
        shared_models={"text_encoder": text_encoder},
    )
    crossattn = context["embed"][0]
    print(
        f"\nprompt → cross-attn embedding: shape={tuple(crossattn.shape)} "
        f"(padded to {MAX_CROSSATTN_TOKENS} tokens — do NOT trim, padding acts as "
        f"attention sinks)"
    )


if __name__ == "__main__":
    main()
