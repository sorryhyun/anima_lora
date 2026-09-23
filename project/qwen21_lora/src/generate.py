"""Generate an eval set with and without the LoRA, same seed per prompt.

The point is the A/B, not the pictures: ``--multipliers 1.0,0.0`` renders each
prompt twice from one model, because ``LoRANetwork.set_multiplier(0)`` makes
every adapter short-circuit and the transformer is bit-identical to the base.
No reload, no second copy on a 16 GB card, and the only difference between the
pair is the adapter.

Three phases, as ``smoke_t2i.py``: encode every prompt with the text encoder
block-swapped on, drop its 17.5 GB, denoise with the transformer, then take the
transformer off the card and decode. Latents are held between phases because
the VAE wants several GB to itself at 1024.

    make daemon-run ARGS="project/qwen21_lora/src/generate.py \
        --lora project/qwen21_lora/out/lora_channel_caststation.safetensors"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from library.qwen21.loader import (
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
from library.qwen21.lora import load_network


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=None)
    ap.add_argument(
        "--lora",
        default="project/qwen21_lora/out/lora_channel_caststation.safetensors",
        help="empty string = base model only",
    )
    ap.add_argument("--prompts", default="project/qwen21_lora/eval_prompts.txt")
    ap.add_argument("--out_dir", default="project/qwen21_lora/out/eval")
    ap.add_argument(
        "--multipliers",
        default="1.0,0.0",
        help="comma-separated adapter scales; 0.0 is the base model",
    )
    ap.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="square edge, used only when --width/--height are not given",
    )
    ap.add_argument(
        "--width",
        type=int,
        default=None,
        help="explicit size, a multiple of 32. Without it the pipeline renders a "
        "square (`height = height or output_resolution`), which is off the "
        "training distribution for a portrait folder",
    )
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument(
        "--true_cfg_scale",
        type=float,
        default=1.0,
        help="the pipeline's own default — 2.1 runs without CFG (no guidance "
        "embedding in the transformer either), and >1 costs a second forward "
        "per step. Raising it stacks an untested setting on top of the A/B",
    )
    ap.add_argument("--negative_prompt", default="")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--blocks_to_swap", type=int, default=None)
    ap.add_argument("--te_blocks_to_swap", type=int, default=None)
    args = ap.parse_args()

    prompts = [
        line.strip()
        for line in Path(args.prompts).read_text().splitlines()
        if line.strip()
    ]
    if not prompts:
        raise SystemExit(f"no prompts in {args.prompts}")
    multipliers = [float(m) for m in args.multipliers.split(",")]
    if (args.width is None) != (args.height is None):
        raise SystemExit("--width and --height go together")
    width = args.width or args.resolution
    height = args.height or args.resolution
    for name, value in (("width", width), ("height", height)):
        if value % 32:
            raise SystemExit(f"--{name} {value} is not a multiple of 32")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"{len(prompts)} prompts x {len(multipliers)} multipliers "
        f"{multipliers} = {len(prompts) * len(multipliers)} images "
        f"at {width}x{height} ({width * height / 1e6:.2f} MP, "
        f"{(width // 16) * (height // 16)} latent tokens), "
        f"{args.steps} steps, cfg {args.true_cfg_scale}",
        flush=True,
    )

    torch.cuda.init()
    device = torch.device("cuda")
    do_cfg = args.true_cfg_scale > 1

    # ── phase 1: encode every prompt, text encoder swapped on ─────────
    te = load_text_encoder(args.model_dir)
    pipe = load_pipeline(args.model_dir, text_encoder=te)
    te_attached = place(
        te,
        TEXT_ENCODER_BLOCKS,
        device,
        blocks_to_swap=args.te_blocks_to_swap,
        label="text_encoder",
    )
    t0 = time.time()
    to_encode = prompts + ([args.negative_prompt] if do_cfg else [])
    encoded = encode_prompts(pipe, to_encode, device="cuda")
    negative = encoded.pop() if do_cfg else None
    print(
        f"encode: {len(to_encode)} prompts in {time.time() - t0:.1f}s  "
        f"peak {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB",
        flush=True,
    )
    if te_attached is not None:
        te_attached.detach()
    del te
    drop_text_encoder(pipe)
    empty_cache()

    # ── phase 2: denoise, transformer swapped on, adapter attached ────
    network, meta = (None, {})
    if args.lora:
        network, meta = load_network(pipe.transformer, args.lora)
        network.apply_to()
        network.to(device)
        print(
            f"lora: {args.lora} rank {meta['rank']} alpha {meta['alpha']} "
            f"({meta.get('epochs', '?')} epochs, {meta.get('samples', '?')} samples)",
            flush=True,
        )
    elif multipliers != [0.0]:
        raise SystemExit("--lora is empty, so only --multipliers 0.0 is meaningful")

    dit_attached = place(
        pipe.transformer,
        TRANSFORMER_BLOCKS,
        device,
        blocks_to_swap=args.blocks_to_swap,
        label="transformer",
    )
    empty_cache()
    print(f"free VRAM with DiT placed: {free_vram_gb():.2f} GB", flush=True)

    renders = []
    t_all = time.time()
    for multiplier in multipliers:
        if network is not None:
            network.set_multiplier(multiplier)
        for index, (prompt, (embeds, mask, _pad)) in enumerate(zip(prompts, encoded)):
            call = dict(
                prompt_embeds=embeds.to(device),
                prompt_embeds_mask=None if mask is None else mask.to(device),
                num_inference_steps=args.steps,
                width=width,
                height=height,
                true_cfg_scale=args.true_cfg_scale,
            )
            if do_cfg:
                neg_embeds, neg_mask, _ = negative
                call["negative_prompt_embeds"] = neg_embeds.to(device)
                call["negative_prompt_embeds_mask"] = (
                    None if neg_mask is None else neg_mask.to(device)
                )
            t0 = time.time()
            latents = pipe(
                output_type="latent",
                # Same seed per prompt across multipliers: the pair differs by
                # the adapter and nothing else.
                generator=torch.Generator("cuda").manual_seed(args.seed + index),
                **call,
            ).images
            renders.append(
                {
                    "index": index,
                    "multiplier": multiplier,
                    "prompt": prompt,
                    "seed": args.seed + index,
                    "latents": latents,
                    "seconds": time.time() - t0,
                }
            )
            print(
                f"  m{multiplier} [{index + 1}/{len(prompts)}] "
                f"{renders[-1]['seconds']:.1f}s",
                flush=True,
            )
    print(
        f"denoise: {len(renders)} images in {(time.time() - t_all) / 60:.1f} min  "
        f"peak {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB",
        flush=True,
    )

    # ── phase 3: DiT off the card, then decode ────────────────────────
    if dit_attached is not None:
        dit_attached.detach()
    pipe.transformer = None
    pipe.register_to_config(transformer=None)
    empty_cache()
    pipe.vae.to(device)
    print(f"free VRAM for decode: {free_vram_gb():.2f} GB", flush=True)

    manifest = []
    t0 = time.time()
    with torch.no_grad():
        for render in renders:
            image = decode_latents(pipe, render["latents"], height, width)[0]
            name = f"{render['index']:02d}_m{render['multiplier']:g}.png"
            image.save(out_dir / name)
            manifest.append(
                {
                    "file": name,
                    "index": render["index"],
                    "multiplier": render["multiplier"],
                    "seed": render["seed"],
                    "prompt": render["prompt"],
                    "denoise_seconds": render["seconds"],
                }
            )
    print(f"decode: {len(renders)} images in {time.time() - t0:.1f}s", flush=True)

    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "lora": args.lora,
                "lora_metadata": meta,
                "size": f"{width}x{height}",
                "steps": args.steps,
                "true_cfg_scale": args.true_cfg_scale,
                "negative_prompt": args.negative_prompt,
                "multipliers": multipliers,
                "images": manifest,
            },
            indent=2,
        )
    )
    print(f"wrote {len(manifest)} images + manifest.json to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
