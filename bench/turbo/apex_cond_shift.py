#!/usr/bin/env python3
"""Does ``a·c`` separate from ``c`` on the DiT? (Turbo APEX Phase 0b)

project/anima_turbo/proposal/turbo_apex.md. On N dataset items, each τ band and each scale ``a``,
with the same (x_t, τ) across branches, on the int8 base + fp32 stream the loop runs on:

    v_c   F(x_t, τ, c)     v_u   F(x_t, τ, T5(""))     v_a   F(x_t, τ, a·c)

The shared ε − x part keeps every raw velocity cosine near 1, so the gaps are also read
in units of the guidance signal ``g = rms(v_c − v_u)``:

    sep_c    rms(v_a − v_c) / g    0 = the fake branch shares the student's path
    sep_u    rms(v_a − v_u) / g    0 = the fake branch is uncond
    cos_gu   cos(v_a − v_u, v_c − v_u)    −1 = anti-guidance, +1 = weaker guidance

``d_rms`` and ``d_clip`` are what L_score sees: ``d = v_a − v_c`` and the fraction of its
coordinates beyond the ±1 clamp. ``--lora`` reads the same on a warm-start delta.

    make daemon-run ARGS="--stall-timeout 300 bench/turbo/apex_cond_shift.py"
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from anima_lora import GenerationRequest
from bench._anima import add_model_args
from bench._common import make_run_dir, write_result
from library.anima.uncond import default_uncond_path, load_uncond_crossattn
from library.env import resolve_under_home
from library.inference.models import load_dit_model
from library.io.cache import (
    discover_cached_pairs,
    load_cached_latents,
    load_cached_text_features,
)

ROWS = (
    "cos_c_a",
    "cos_a_u",
    "cos_c_u",
    "sep_c",
    "sep_u",
    "cos_gu",
    "g_rms",
    "v_rms",
    "d_rms",
    "d_clip",
)


def make_args(opts, device) -> argparse.Namespace:
    req = GenerationRequest(
        dit=opts.dit,
        vae=opts.vae,
        text_encoder=opts.text_encoder,
        prompt="",
        save_path="output/tests/_apex_cond_shift.png",
        infer_steps=4,
        guidance_scale=1.0,
        image_size=(1024, 1024),
        seed=opts.seed,
    )
    args = req.to_args()
    args.device = device
    args.dtype = "bf16"
    args.attn_mode = getattr(args, "attn_mode", "flash") or "flash"
    args.compile = False
    args.compile_blocks = False
    args.compile_dynamic_seq = False
    args.int8 = True
    args.int8_file = opts.int8_file
    args.stream = "fp32"
    args.lora_weight = [opts.lora] if opts.lora else None
    args.lora_multiplier = 1.0
    return args


def velocity(model, x, tau, c) -> torch.Tensor:
    pad = torch.zeros(1, 1, *x.shape[-2:], dtype=torch.bfloat16, device=x.device)
    t = torch.full((1,), tau, dtype=torch.float32, device=x.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        v = model.forward_mini_train_dit(
            x.unsqueeze(2), t, c, padding_mask=pad, skip_pooled_text_proj=True
        )
    return v.squeeze(2).float()


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.flatten(), b.flatten(), dim=0
    ).item()


def rms(a: torch.Tensor) -> float:
    return a.pow(2).mean().sqrt().item()


def read(v_c, v_u, v_a) -> dict[str, float]:
    guidance = v_c - v_u
    g = rms(guidance)
    d = v_a - v_c
    return {
        "cos_c_a": cos(v_c, v_a),
        "cos_a_u": cos(v_a, v_u),
        "cos_c_u": cos(v_c, v_u),
        "sep_c": rms(d) / g,
        "sep_u": rms(v_a - v_u) / g,
        "cos_gu": cos(v_a - v_u, guidance),
        "g_rms": g,
        "v_rms": rms(v_c),
        "d_rms": rms(d),
        "d_clip": (d.abs() > 1.0).float().mean().item(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", default="post_image_dataset/lora")
    p.add_argument("--int8_file", default=None)
    p.add_argument("--lora", default=None, help="Read on this delta instead of base.")
    p.add_argument("--scales", type=float, nargs="+", default=[-1.0, -0.5])
    p.add_argument(
        "--taus",
        type=float,
        nargs="+",
        default=[0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
    )
    p.add_argument("--label", default="apex-cond-shift")
    opts = p.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    run_dir = make_run_dir("turbo", label=opts.label)

    data_dir = Path(resolve_under_home(opts.data_dir)).resolve()
    pairs = discover_cached_pairs(str(data_dir))
    random.Random(opts.seed).shuffle(pairs)
    pairs = pairs[: opts.n]

    model = load_dit_model(make_args(opts, device), device, torch.bfloat16)
    c_null = load_uncond_crossattn(str(default_uncond_path()), device, torch.bfloat16)
    if c_null.ndim == 2:
        c_null = c_null.unsqueeze(0)

    rows = {a: {tau: [] for tau in opts.taus} for a in opts.scales}
    for i, pair in enumerate(pairs):
        latent = load_cached_latents(pair.npz_path)[0].to(device).float().unsqueeze(0)
        c = load_cached_text_features(pair.te_path, variant=0)[0]
        c = c.to(device, torch.bfloat16).unsqueeze(0)
        gen = torch.Generator("cpu").manual_seed(opts.seed + i)
        eps = torch.randn(latent.shape, generator=gen).to(device)
        for tau in opts.taus:
            x_t = (1.0 - tau) * latent + tau * eps
            v_c = velocity(model, x_t, tau, c)
            v_u = velocity(model, x_t, tau, c_null)
            for a in opts.scales:
                rows[a][tau].append(read(v_c, v_u, velocity(model, x_t, tau, a * c)))
        print(f"[0b] item {i + 1}/{len(pairs)} {tuple(latent.shape)}", flush=True)

    metrics: dict = {"N": len(pairs), "lora": opts.lora, "scales": {}}
    print("\n" + f"{'a':>5} {'tau':>5} " + " ".join(f"{k:>8}" for k in ROWS))
    for a in opts.scales:
        by_tau = {}
        for tau in opts.taus:
            mean = {k: float(np.mean([r[k] for r in rows[a][tau]])) for k in ROWS}
            mean["sep_c_min"] = float(min(r["sep_c"] for r in rows[a][tau]))
            by_tau[str(tau)] = mean
            print(
                f"{a:>5} {tau:>5} " + " ".join(f"{mean[k]:>8.4f}" for k in ROWS),
                flush=True,
            )
        metrics["scales"][str(a)] = by_tau
    write_result(run_dir, script=__file__, args=opts, metrics=metrics)
    print(f"[0b] wrote {run_dir}", flush=True)


if __name__ == "__main__":
    main()
