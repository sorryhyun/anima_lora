"""Does a LoRA backward fit on the card? — the gate before any data work.

Everything upstream of this (VAE latent cache, Qwen3-VL text-embedding cache) is
wasted effort if the transformer cannot take a backward under the block swapper,
so this runs the training step on random tensors of the right shape and reports
what it costs. Nothing here reads the dataset.

Three things are being checked at once:

1. ``ModelOffloader(supports_backward=True)`` drives the backward direction from
   its own hooks, so the forward must only half-swap — see
   ``blockswap.swap_schedule(restore=False)``. The block devices are asserted
   back at their starting placement after every step.
2. Every adapter gets a finite gradient. ``lora.LoRANetwork`` keeps the
   trainable weights off the swapped blocks; if that ever slips, the optimizer
   step raises on a CPU/CUDA mismatch here rather than mid-run.
3. Peak VRAM and s/step at the training sequence length, which is what decides
   the resolution the line can afford.

    make daemon-run ARGS="project/qwen21_lora/src/backward_smoke.py \
        --resolution 512 --compile"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

sys.path.insert(0, str(Path(__file__).parent))

import blockswap  # noqa: E402
from accel import compile_blocks, recompile_report  # noqa: E402
from loader import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    TRANSFORMER_BLOCKS,
    empty_cache,
    free_vram_gb,
    place,
)
from lora import DEFAULT_TARGETS, LoRANetwork  # noqa: E402

# 4 spatial downsamples in the VAE's dim_mult, as the pipeline's
# `vae_scale_factor`. 512 px -> 32x32 latent tokens, 1024 px -> 64x64.
VAE_SCALE_FACTOR = 16


def load_transformer(model_dir: Path | str, dtype: torch.dtype = torch.bfloat16):
    """The 14.2 GB transformer alone — no text encoder, no VAE."""
    from diffusers import QwenImage21Transformer2DModel

    return QwenImage21Transformer2DModel.from_pretrained(
        Path(model_dir) / "transformer", dtype=dtype
    )


def dummy_batch(
    transformer,
    resolution: int,
    text_len: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
):
    """One text-to-image training sample, shaped like the pipeline's own.

    Text-to-image means no condition images, so the vision-language mask is all
    text and the pipeline appends one image slot per 2x2 group of target latent
    tokens (``append_target_slots``).
    """
    side = resolution // VAE_SCALE_FACTOR
    tokens = side * side
    channels = transformer.config.in_channels

    latents = torch.randn(
        (1, tokens, channels), generator=generator, device=device, dtype=dtype
    )
    encoder_hidden_states = torch.randn(
        (1, text_len, transformer.config.context_in_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    encoder_hidden_states_mask = torch.ones(
        (1, text_len), dtype=torch.long, device=device
    )
    img_mask = torch.cat(
        [
            torch.zeros((1, text_len), dtype=torch.bool, device=device),
            torch.ones((1, tokens // 4), dtype=torch.bool, device=device),
        ],
        dim=1,
    )
    return {
        "latents": latents,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_hidden_states_mask": encoder_hidden_states_mask,
        "img_shapes": [[(1, side, side)]],
        "img_mask": img_mask,
    }


def flow_matching_step(transformer, batch, generator: torch.Generator):
    """One flow-matching loss on the batch's latents as the clean sample."""
    latents = batch["latents"]
    noise = torch.randn(
        latents.shape, generator=generator, device=latents.device, dtype=latents.dtype
    )
    sigma = torch.rand((1,), generator=generator, device=latents.device).to(
        latents.dtype
    )
    noisy = (1.0 - sigma) * latents + sigma * noise
    target = noise - latents

    pred = transformer(
        hidden_states=noisy,
        encoder_hidden_states=batch["encoder_hidden_states"],
        encoder_hidden_states_mask=batch["encoder_hidden_states_mask"],
        timestep=sigma,
        img_shapes=batch["img_shapes"],
        img_mask=batch["img_mask"],
        return_dict=False,
    )[0]
    # The model returns the whole joint sequence; the target image is its tail.
    pred = pred[:, -latents.shape[1] :]
    return F.mse_loss(pred.float(), target.float())


def block_devices(blocks) -> list[str]:
    return [next(block.parameters()).device.type for block in blocks]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--text_len", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--targets", default=DEFAULT_TARGETS)
    # None = size the swap against what is free; 0 = keep everything resident.
    ap.add_argument("--blocks_to_swap", type=int, default=None)
    ap.add_argument(
        "--activation_reserve_gb",
        type=float,
        default=6.0,
        help="VRAM the swap sizer leaves for activations — a backward holds "
        "far more of them than the 2.5 GB an inference forward needs",
    )
    ap.add_argument(
        "--compile", action="store_true", help="torch.compile each block's forward"
    )
    ap.add_argument("--compile_mode", default=None)
    ap.add_argument(
        "--grad_checkpointing",
        action="store_true",
        help="recompute block activations in the backward; the recompute runs "
        "under blockswap.checkpoint_context_fn so it does not re-drive the swap",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="project/qwen21_lora/out/backward_smoke.json")
    args = ap.parse_args()

    torch.cuda.init()
    device = torch.device("cuda")
    print(f"free VRAM before load: {free_vram_gb():.2f} GB", flush=True)

    t0 = time.time()
    transformer = load_transformer(args.model_dir)
    transformer.requires_grad_(False)
    print(f"transformer loaded in {time.time() - t0:.1f}s", flush=True)

    network = LoRANetwork(transformer, rank=args.rank, targets=args.targets)
    patched = network.apply_to()
    network.to(device)
    print(
        f"lora: rank {args.rank} on {patched} linears, "
        f"{network.num_parameters / 1e6:.1f}M params fp32",
        flush=True,
    )

    if args.grad_checkpointing:

        def checkpointing_func(module, *inputs):
            return torch.utils.checkpoint.checkpoint(
                module.__call__,
                *inputs,
                use_reentrant=False,
                context_fn=blockswap.checkpoint_context_fn,
            )

        transformer.enable_gradient_checkpointing(checkpointing_func)
        print("gradient checkpointing: on", flush=True)

    attached = place(
        transformer,
        TRANSFORMER_BLOCKS,
        device,
        blocks_to_swap=args.blocks_to_swap,
        supports_backward=True,
        activation_reserve_gb=args.activation_reserve_gb,
        label="transformer",
    )
    blocks = transformer.transformer_blocks
    placement = block_devices(blocks)

    if args.compile:
        # Training only ever runs the prefill shape (`segments is not None`),
        # which the decode-only dispatch routes to the eager forward — so the
        # whole point here is to compile every shape.
        compile_blocks(blocks, mode=args.compile_mode, decode_only=False)

    empty_cache()
    print(f"free VRAM before step: {free_vram_gb():.2f} GB", flush=True)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    batch = dummy_batch(
        transformer, args.resolution, args.text_len, device, torch.bfloat16, generator
    )
    params = list(network.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    image_tokens = batch["latents"].shape[1]
    print(
        f"sequence: {image_tokens} image + {args.text_len} text = "
        f"{image_tokens + args.text_len} joint tokens",
        flush=True,
    )

    records = []
    for step in range(args.steps):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        loss = flow_matching_step(transformer, batch, generator)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)

        # Read the gradients before the optimizer clears them. `up` is zero-init,
        # so on step 0 `down`'s gradient is legitimately zero (it arrives through
        # `up`) while every parameter must still have a finite one.
        no_grad = [n for n, p in network.named_parameters() if p.grad is None]
        nonfinite = [
            n
            for n, p in network.named_parameters()
            if p.grad is not None and not torch.isfinite(p.grad).all()
        ]
        zero_grad = [
            n
            for n, p in network.named_parameters()
            if p.grad is not None and not p.grad.any()
        ]
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        record = {
            "step": step,
            "loss": loss.item(),
            "grad_norm": grad_norm.item(),
            "seconds": time.time() - t0,
            "peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "params_without_grad": len(no_grad),
            "params_nonfinite_grad": len(nonfinite),
            "params_zero_grad": len(zero_grad),
            "placement_restored": block_devices(blocks) == placement,
        }
        records.append(record)
        print(
            f"step {step}: loss {record['loss']:.4f}  |g| {record['grad_norm']:.4f}  "
            f"{record['seconds']:.2f}s  peak {record['peak_gb']:.2f} GB  "
            f"grads none/nonfinite/zero "
            f"{len(no_grad)}/{len(nonfinite)}/{len(zero_grad)} of {len(params)}  "
            f"placement {'ok' if record['placement_restored'] else 'DRIFTED'}",
            flush=True,
        )

    moved = [n for n, p in network.named_parameters() if p.device.type != device.type]
    if args.compile:
        print(recompile_report(), flush=True)

    result = {
        "resolution": args.resolution,
        "image_tokens": batch["latents"].shape[1],
        "text_len": args.text_len,
        "rank": args.rank,
        "lora_params": network.num_parameters,
        "patched_linears": patched,
        "blocks_to_swap": attached.offloader.blocks_to_swap if attached else 0,
        "compile": args.compile,
        "grad_checkpointing": args.grad_checkpointing,
        "steps": records,
        "params_off_device": moved,
        # The first step pays compile and allocator growth; the rest is what a
        # run would actually cost.
        "steady_seconds": (
            min(r["seconds"] for r in records[1:]) if len(records) > 1 else None
        ),
        "peak_gb": max(r["peak_gb"] for r in records),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}", flush=True)

    if attached is not None:
        attached.detach()


if __name__ == "__main__":
    main()
