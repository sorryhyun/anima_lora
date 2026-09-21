"""One-image smoke test for Qwen-Image-2.1 on a 16 GB card.

Runs the split strategy from ``loader.py`` — stream the text encoder for the
encode pass, drop it, then denoise with the transformer resident — and prints
peak VRAM per phase for the training script to budget against.

    .venv/bin/python project/qwen21_lora/src/smoke_t2i.py \
        --prompt "..." --resolution 1024 --steps 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from accel import compile_blocks, recompile_report, set_attention_backend  # noqa: E402
from loader import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    TEXT_ENCODER_BLOCKS,
    TRANSFORMER_BLOCKS,
    decode_latents,
    drop_text_encoder,
    empty_cache,
    encode_prompts,
    free_vram_gb,
    load_pipeline,
    load_text_encoder,
    place,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument(
        "--prompt",
        default='A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, '
        "reflections on wet pavement",
    )
    ap.add_argument("--negative_prompt", default=None)
    ap.add_argument("--true_cfg_scale", type=float, default=1.0)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    # None = size the swap against free VRAM; 0 = keep everything resident.
    ap.add_argument("--blocks_to_swap", type=int, default=None)
    ap.add_argument("--te_blocks_to_swap", type=int, default=None)
    ap.add_argument(
        "--attn_backend",
        default="native",
        help="diffusers attention backend for the DiT decode path: native, "
        "flash, flash_varlen, sage, ... (see AttentionBackendName)",
    )
    ap.add_argument(
        "--te_attn_implementation",
        default=None,
        help="transformers attn_implementation for the text encoder "
        "(sdpa / flash_attention_2 / eager); default = the model's own",
    )
    ap.add_argument(
        "--compile", action="store_true", help="torch.compile each DiT block's forward"
    )
    ap.add_argument(
        "--compile_mode", default=None, help="inductor preset for --compile"
    )
    ap.add_argument(
        "--compile_all_shapes",
        action="store_true",
        help="also compile the prefill shape (default: decode shape)",
    )
    ap.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="denoise this many times; with --compile the first pass pays the "
        "compile and the rest are the steady-state number",
    )
    ap.add_argument("--out", default="project/qwen21_lora/out/smoke.png")
    args = ap.parse_args()

    torch.cuda.init()
    device = torch.device("cuda")
    print(f"free VRAM before load: {free_vram_gb():.2f} GB", flush=True)

    t0 = time.time()
    te = load_text_encoder(
        args.model_dir, attn_implementation=args.te_attn_implementation
    )
    pipe = load_pipeline(args.model_dir, text_encoder=te)
    print(f"pipeline built in {time.time() - t0:.1f}s", flush=True)

    # ── phase 1: text encode, encoder block-swapped onto the card ─────
    torch.cuda.reset_peak_memory_stats()
    te_attached = place(
        te,
        TEXT_ENCODER_BLOCKS,
        device,
        blocks_to_swap=args.te_blocks_to_swap,
        label="text_encoder",
    )
    t0 = time.time()
    prompts = [args.prompt] + (
        [args.negative_prompt] if args.negative_prompt is not None else []
    )
    encoded = encode_prompts(pipe, prompts, device="cuda")
    print(
        f"encode: {time.time() - t0:.1f}s for {len(prompts)} prompt(s)  "
        f"peak {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB",
        flush=True,
    )
    if te_attached is not None:
        te_attached.detach()
    te_attached = None
    del te
    drop_text_encoder(pipe)
    print(
        f"after dropping TE: alloc {torch.cuda.memory_allocated() / 1024**3:.2f} "
        f"reserved {torch.cuda.memory_reserved() / 1024**3:.2f} "
        f"free {free_vram_gb():.2f} GB",
        flush=True,
    )

    # ── phase 2: denoise to latents, transformer block-swapped ────────
    torch.cuda.reset_peak_memory_stats()
    dit_attached = place(
        pipe.transformer,
        TRANSFORMER_BLOCKS,
        device,
        blocks_to_swap=args.blocks_to_swap,
        label="transformer",
    )
    embeds, mask, _pad = encoded[0]
    set_attention_backend(
        pipe.transformer, args.attn_backend, padded_prompt=mask is not None
    )
    if args.compile:
        compile_blocks(
            pipe.transformer.transformer_blocks,
            mode=args.compile_mode,
            decode_only=not args.compile_all_shapes,
        )
    empty_cache()
    print(f"free VRAM with DiT placed: {free_vram_gb():.2f} GB", flush=True)

    call = dict(
        prompt_embeds=embeds.to("cuda"),
        prompt_embeds_mask=None if mask is None else mask.to("cuda"),
        num_inference_steps=args.steps,
        output_resolution=args.resolution,
        true_cfg_scale=args.true_cfg_scale,
    )
    if len(encoded) > 1:
        neg_embeds, neg_mask, _ = encoded[1]
        call["negative_prompt_embeds"] = neg_embeds.to("cuda")
        call["negative_prompt_embeds_mask"] = (
            None if neg_mask is None else neg_mask.to("cuda")
        )

    for run in range(args.repeat):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        latents = pipe(
            output_type="latent",
            generator=torch.Generator("cuda").manual_seed(args.seed),
            **call,
        ).images
        dt = time.time() - t0
        tag = "" if args.repeat == 1 else f" [{run + 1}/{args.repeat}]"
        print(
            f"denoise{tag}: {dt:.1f}s ({dt / args.steps:.2f}s/step)  "
            f"peak {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB",
            flush=True,
        )
    if args.compile:
        print(recompile_report(), flush=True)

    # ── phase 3: decode, with the DiT off the card ────────────────────
    # The VAE wants several GB to itself at 1024 (see loader.decode_latents).
    if dit_attached is not None:
        dit_attached.detach()
    pipe.transformer = None
    pipe.register_to_config(transformer=None)
    empty_cache()
    pipe.vae.to(device)
    print(f"free VRAM for decode: {free_vram_gb():.2f} GB", flush=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        image = decode_latents(pipe, latents, args.resolution, args.resolution)[0]
    print(
        f"decode: {time.time() - t0:.1f}s  "
        f"peak {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB",
        flush=True,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    print(f"wrote {out}  ({image.size[0]}x{image.size[1]}, mode={image.mode})")


if __name__ == "__main__":
    main()
