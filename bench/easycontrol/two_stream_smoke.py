#!/usr/bin/env python
"""Two-stream EasyControl smoke test.

Constructs a real Anima DiT, applies the rewritten EasyControlNetwork, runs a
single forward+backward with a synthetic cond, and reports peak GPU memory.
This is the M1 / M3 gate from the rewrite plan: confirms the patched
``Block.forward`` wires up end-to-end without crashing, the cond stream flows
through all blocks, autograd reaches the cond LoRAs, and the peak memory is
within the expected envelope.

Usage::

    uv run python bench/easycontrol/two_stream_smoke.py
    uv run python bench/easycontrol/two_stream_smoke.py --gradient_checkpointing
    uv run python bench/easycontrol/two_stream_smoke.py --no_cond  # baseline
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from library.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


def build_small_dit(device, dtype):
    """Construct an Anima DiT at the live config dimensions."""
    from library.anima.models import Anima

    dit = Anima(
        max_img_h=2048,
        max_img_w=2048,
        max_frames=1,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=True,
        model_channels=2048,
        num_blocks=28,
        num_heads=16,
        mlp_ratio=4.0,
        crossattn_emb_channels=1024,
        use_adaln_lora=True,
        adaln_lora_dim=256,
        attn_mode="flash",
        split_attn=False,
    )
    dit.set_static_token_count(4096)
    dit = dit.to(device=device, dtype=dtype)
    dit.eval()  # frozen
    for p in dit.parameters():
        p.requires_grad_(False)
    return dit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument(
        "--target_h",
        type=int,
        default=64,
        help="target latent H (in latent-pixels). 64 → 128 patches "
        "horizontally — well below 4096 static tokens.",
    )
    p.add_argument("--target_w", type=int, default=64)
    p.add_argument("--cond_h", type=int, default=64)
    p.add_argument("--cond_w", type=int, default=64)
    p.add_argument("--cond_token_count", type=int, default=1024)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument(
        "--no_cond",
        action="store_true",
        help="Skip set_cond — measures baseline DiT memory for A/B.",
    )
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    logger.info(f"building DiT (device={device}, dtype={dtype})...")
    dit = build_small_dit(device, dtype)

    if args.gradient_checkpointing:
        dit.enable_gradient_checkpointing(unsloth_offload=False)
        # Mirror what the real trainer does (train.py:2062): explicitly put
        # the DiT into train mode so Block.forward takes the gradient-
        # checkpoint branch. Params still have requires_grad=False, so this
        # only flips the .training flag and dropout state.
        dit.train()
        logger.info(
            "gradient checkpointing ON (plain torch_checkpoint), DiT in train mode"
        )

    from networks.methods.easycontrol import create_network

    net = create_network(
        multiplier=1.0,
        network_dim=16,
        network_alpha=16.0,
        vae=None,
        text_encoders=None,
        unet=dit,
        b_cond_init="-10.0",
        cond_scale="1.0",
        apply_ffn_lora="1",
        cond_token_count=str(args.cond_token_count),
    )
    net = net.to(device=device, dtype=dtype)
    net.apply_to(text_encoders=None, unet=dit, apply_unet=True)
    net.train()

    B = 1
    C = 16
    H = args.target_h
    W = args.target_w
    H_c = args.cond_h
    W_c = args.cond_w

    # Realistic-ish target latent + cond latent (clean reference image VAE output).
    target_latent = torch.randn(B, C, H, W, device=device, dtype=dtype)
    cond_latent = torch.randn(B, C, H_c, W_c, device=device, dtype=dtype)
    timesteps = torch.full((B,), 0.5, device=device, dtype=dtype)
    crossattn_emb = torch.randn(
        B, 256, 1024, device=device, dtype=dtype
    )  # (B, max_text_len, D_text)
    crossattn_seqlens = torch.full((B,), 256, device=device, dtype=torch.long)
    padding_mask = torch.ones(B, 1, H, W, device=device, dtype=dtype)

    # Prime cond on the network. set_cond builds cond_x, cond_rope, cond_temb
    # and stashes on net._cond_state + block 0's side channel.
    if not args.no_cond:
        net.set_cond(
            cond_latent,
            padding_mask=torch.ones(B, 1, H_c, W_c, device=device, dtype=dtype),
        )
        logger.info("cond primed via set_cond")
    else:
        logger.info("no cond (--no_cond) — measuring baseline DiT memory")

    torch.cuda.reset_peak_memory_stats()
    mem_pre = torch.cuda.memory_allocated() / 1024**2

    out = dit.forward_mini_train_dit(
        target_latent.unsqueeze(2),  # (B,C,1,H,W) for the 5D path
        timesteps,
        crossattn_emb,
        padding_mask=padding_mask,
        crossattn_seqlens=crossattn_seqlens,
    )
    logger.info(f"forward done: out shape={tuple(out.shape)}")

    loss = out.float().pow(2).mean()
    logger.info(f"loss={loss.item():.6f}")
    loss.backward()
    logger.info("backward done")

    # Confirm cond LoRAs received gradients (only meaningful when cond was set).
    if not args.no_cond:
        n_with_grad = 0
        n_total = 0
        for name, p in net.named_parameters():
            n_total += 1
            if p.grad is not None and p.grad.abs().sum().item() != 0.0:
                n_with_grad += 1
        logger.info(
            f"cond LoRA gradients: {n_with_grad}/{n_total} parameters "
            f"received non-zero grad"
        )
        # b_cond gets a non-trivial gradient because target output depends on
        # b_cond via α/β; the rest (cond LoRAs) start with up=0 → delta=0 →
        # output insensitive at this step. Expect roughly the b_cond entries.
        logger.info(
            "  (LoRA up.weight is zero-init at step 0 — only b_cond entries "
            "are expected to have non-zero grad on the very first step.)"
        )

    mem_peak = torch.cuda.max_memory_allocated() / 1024**2
    mem_post = torch.cuda.memory_allocated() / 1024**2
    print()
    print(
        f"GPU memory:  pre={mem_pre:.1f} MiB   peak={mem_peak:.1f} MiB   "
        f"post={mem_post:.1f} MiB"
    )


if __name__ == "__main__":
    main()
