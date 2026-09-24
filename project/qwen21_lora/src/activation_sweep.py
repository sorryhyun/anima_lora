"""How much VRAM does a checkpointed LoRA backward hold, as a function of tokens?

``--activation_reserve_gb`` sizes the swap with one constant. If the activation
peak is affine in the joint token count (image + text), the cache already knows
the largest sample before the model loads, and the reserve could follow it. This
measures that curve: one model load, a fixed swap so the weights on the card do
not move between points, and one fwd+bwd per token count on random tensors.

activation = max_memory_allocated during the step - memory_allocated at idle
(weights + LoRA + Adam states). The first step of the sweep warms the allocator
and the optimizer states and is thrown away.

    make daemon-run ARGS="--stall-timeout 900 project/qwen21_lora/src/activation_sweep.py"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.utils.checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backward_smoke import flow_matching_step  # noqa: E402

from library.qwen21 import blockswap  # noqa: E402
from library.qwen21.loader import (  # noqa: E402
    TRANSFORMER_BLOCKS,
    empty_cache,
    free_vram_gb,
    load_transformer,
    place,
)
from library.qwen21.lora import DEFAULT_TARGETS, LoRANetwork  # noqa: E402


def dummy_batch(transformer, side_h: int, side_w: int, text_len: int, device, dtype):
    tokens = side_h * side_w
    latents = torch.randn((1, tokens, transformer.config.in_channels), device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(
        (1, text_len, transformer.config.context_in_dim), device=device, dtype=dtype
    )
    encoder_hidden_states_mask = torch.ones((1, text_len), dtype=torch.long, device=device)
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
        "img_shapes": [[(1, side_h, side_w)]],
        "img_mask": img_mask,
    }


def fit_line(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = max(abs(y - (a + b * x)) for x, y in zip(xs, ys))
    return a, b, resid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=None)
    # latent side lengths (tokens = side²); 64 = 1024², 32 = 512²
    ap.add_argument("--sides", default="32,40,48,56,64,72,80")
    ap.add_argument("--text_len", type=int, default=346)
    # second axis: text length at a fixed image, to check a text token costs an image token
    ap.add_argument("--text_lens_at_64", default="64,200,346,512")
    ap.add_argument("--blocks_to_swap", type=int, default=14)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--targets", default=DEFAULT_TARGETS)
    ap.add_argument("--no_grad_checkpointing", action="store_true")
    ap.add_argument("--out", default="project/qwen21_lora/out/activation_sweep.json")
    args = ap.parse_args()

    torch.cuda.init()
    device = torch.device("cuda")
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"free VRAM before load: {free_vram_gb():.2f} of {total_gb:.2f} GB", flush=True)

    t0 = time.time()
    transformer = load_transformer(args.model_dir)
    transformer.requires_grad_(False)
    print(f"transformer loaded in {time.time() - t0:.1f}s", flush=True)

    network = LoRANetwork(transformer, rank=args.rank, targets=args.targets)
    patched = network.apply_to()
    network.to(device)
    print(f"lora: rank {args.rank} fp32 master on {patched} linears", flush=True)

    if not args.no_grad_checkpointing:

        def checkpointing_func(module, *inputs):
            return torch.utils.checkpoint.checkpoint(
                module.__call__,
                *inputs,
                use_reentrant=False,
                context_fn=blockswap.checkpoint_context_fn,
            )

        transformer.enable_gradient_checkpointing(checkpointing_func)
    print(f"gradient checkpointing: {not args.no_grad_checkpointing}", flush=True)

    attached = place(
        transformer,
        TRANSFORMER_BLOCKS,
        device,
        blocks_to_swap=args.blocks_to_swap,
        supports_backward=True,
        label="transformer",
    )
    empty_cache()

    params = list(network.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    generator = torch.Generator(device=device).manual_seed(0)

    sides = [int(s) for s in args.sides.split(",")]
    text_lens = [int(s) for s in args.text_lens_at_64.split(",")]
    # (h, w, text). Warm-up point first, thrown away.
    configs = [(sides[0], sides[0], args.text_len, "warmup")]
    configs += [(s, s, args.text_len, "image") for s in sides]
    configs += [(64, 64, t, "text") for t in text_lens]

    records = []
    for h, w, text_len, axis in configs:
        empty_cache()
        torch.cuda.synchronize()
        idle_alloc = torch.cuda.memory_allocated() / 1024**3
        idle_free = free_vram_gb()
        torch.cuda.reset_peak_memory_stats()
        batch = dummy_batch(transformer, h, w, text_len, device, torch.bfloat16)
        joint = h * w + text_len
        t0 = time.time()
        try:
            loss = flow_matching_step(transformer, batch, generator)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            oom = False
        except torch.cuda.OutOfMemoryError:
            oom = True
            optimizer.zero_grad(set_to_none=True)
        peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
        del batch
        rec = {
            "axis": axis,
            "latent_h": h,
            "latent_w": w,
            "image_tokens": h * w,
            "text_len": text_len,
            "joint_tokens": joint,
            "idle_alloc_gb": idle_alloc,
            "idle_free_gb": idle_free,
            "peak_alloc_gb": peak_alloc,
            "peak_reserved_gb": peak_reserved,
            "activation_gb": peak_alloc - idle_alloc,
            "reserved_over_idle_gb": peak_reserved - idle_alloc,
            "seconds": time.time() - t0,
            "oom": oom,
        }
        records.append(rec)
        print(
            f"{axis:6s} {h}x{w}+{text_len} = {joint:5d} tok  "
            f"idle {idle_alloc:.2f}  peak {peak_alloc:.2f} alloc / {peak_reserved:.2f} reserved  "
            f"activation {rec['activation_gb']:.2f} GB (reserved {rec['reserved_over_idle_gb']:.2f})  "
            f"{rec['seconds']:.1f}s{'  OOM' if oom else ''}",
            flush=True,
        )

    fits = {}
    for axis in ("image", "text"):
        pts = [r for r in records if r["axis"] == axis and not r["oom"]]
        if len(pts) >= 2:
            for key in ("activation_gb", "reserved_over_idle_gb"):
                a, b, resid = fit_line(
                    [r["joint_tokens"] for r in pts], [r[key] for r in pts]
                )
                fits[f"{axis}/{key}"] = {"a_gb": a, "b_mb_per_token": b * 1024, "max_resid_gb": resid}
                print(
                    f"fit {axis:5s} {key:22s}: {a:.2f} GB + {b * 1024:.3f} MB/token  "
                    f"(max resid {resid:.2f} GB)",
                    flush=True,
                )

    result = {
        "blocks_to_swap": args.blocks_to_swap,
        "grad_checkpointing": not args.no_grad_checkpointing,
        "rank": args.rank,
        "total_gb": total_gb,
        "records": records,
        "fits": fits,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}", flush=True)
    if attached is not None:
        attached.detach()


if __name__ == "__main__":
    main()
