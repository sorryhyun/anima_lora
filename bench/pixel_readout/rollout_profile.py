#!/usr/bin/env python3
"""Where a P2 training rollout's time goes (read-only profile, no training run).

On N cached train images: load (latent + TE from disk), one DiT forward at σ_last, the two
VAE decodes (x0, x̂0), and ``--opt_steps`` head updates on the `1106` recipe head (fresh init),
eager and under ``torch.compile(dynamic=True)``. ``--stream fp32`` applies ``stream_fp32_``
(the `make test` default) before compiling the blocks; the training bench runs bf16 today.
Every phase is timed with a CUDA sync after a warm pass over the same images.

Usage::

    make daemon-run ARGS="--queue --stall-timeout 900 bench/pixel_readout/rollout_profile.py \
        --compile --stream bf16 --label prof_mmdd_hhmm"
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

from bench.pixel_readout._compile import compile_blocks_isolated
from bench._anima import add_common_args, add_model_args, build_anima
from bench._common import make_run_dir, start_heartbeat, write_result
from bench.pixel_readout.convnext_head import (
    PixelHead,
    lat_patches,
    rec_loss,
    token_count,
)
from library.datasets.buckets import cluster_token_bands
from library.env import resolve_under_home
from library.inference.sampling import get_timesteps_sigmas
from library.io.cache import (
    discover_cached_pairs,
    load_cached_latents,
    load_cached_text_features,
)
from library.models import qwen_vae


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(p, text_encoder=False)
    add_common_args(p)
    p.add_argument("--cache_root", default="post_image_dataset/lora")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--opt_steps", type=int, default=4)
    p.add_argument("--stream", choices=["bf16", "fp32"], default="bf16")
    args = p.parse_args()
    start_heartbeat(label="prof")
    device = torch.device(args.device)

    root = resolve_under_home(args.cache_root)
    artists = sorted(d for d in root.iterdir() if d.is_dir())
    items = [discover_cached_pairs(str(a))[0] for a in artists[: args.n]]
    bands = cluster_token_bands([token_count(ci) for ci in items])
    _, sigmas = get_timesteps_sigmas(50, 3.0, device)
    sigma = float(sigmas[-2])

    vae = qwen_vae.load_vae(
        args.vae, device="cpu", dtype=torch.bfloat16, eval=True, vae_2d=True
    ).to(device)
    compile_blocks, args.compile = args.compile, False
    anima = build_anima(args, adapter=None, train_mode=False).anima
    if args.stream == "fp32":
        from library.anima.stream import stream_fp32_

        stream_fp32_(anima)
    if compile_blocks:
        compile_blocks_isolated(anima, bands)
    tap: dict = {}
    anima.final_layer.linear.register_forward_pre_hook(
        lambda _m, inp: tap.__setitem__("x", inp[0].detach())
    )

    D = anima.final_layer.hidden_size
    head = PixelHead(D, 64, 1024, 3, 1.0, use_skip=False, up=1).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-5, betas=(0.9, 0.99))
    head_c = torch.compile(head, dynamic=True)

    def load(ci):
        lat = load_cached_latents(ci.npz_path)[0].unsqueeze(0).to(device)
        cross, _ = load_cached_text_features(ci.te_path, variant=0)
        return lat, cross.unsqueeze(0).to(device, torch.bfloat16)

    @torch.no_grad()
    def dit(lat, cross):
        g = torch.Generator(device=device).manual_seed(0)
        noise = torch.randn(lat.shape, generator=g, device=device, dtype=lat.dtype)
        x_t = (1 - sigma) * lat + sigma * noise
        pad = torch.zeros(1, 1, *lat.shape[-2:], dtype=torch.bfloat16, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = anima(
                x_t.unsqueeze(2).to(torch.bfloat16),
                torch.tensor([sigma], device=device),
                cross,
                padding_mask=pad,
            )
        return tap.pop("x")[0, 0].float(), x_t, v

    @torch.no_grad()
    def decode(lat):
        return vae.decode_to_pixels(lat)[0].float()

    def steps(h, f, xp, tgt):
        for _ in range(args.opt_steps):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = h(f, xp)
            loss, _ = rec_loss(pred, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()

    rows = []
    for rep in ("warm", "timed"):
        for ci in items:
            (lat, cross), t_load = timed(lambda: load(ci))
            (f, x_t, v), t_dit = timed(lambda: dit(lat, cross))
            tgt, t_x0 = timed(lambda: decode(lat))
            _, t_xh = timed(lambda: decode(x_t - sigma * v.float().squeeze(2)))
            xp = lat_patches(x_t).float()
            _, t_eager = timed(lambda: steps(head, f, xp, tgt))
            _, t_comp = timed(lambda: steps(head_c, f, xp, tgt))
            if rep == "timed":
                rows.append(
                    {
                        "tokens": token_count(ci),
                        "load_s": t_load,
                        "dit_s": t_dit,
                        "vae_x0_s": t_x0,
                        "vae_xhat0_s": t_xh,
                        "head_eager_s": t_eager,
                        "head_compiled_s": t_comp,
                    }
                )
        print(f"{rep} pass done", flush=True)

    summary = {
        k: statistics.median(r[k] for r in rows) for k in rows[0] if k.endswith("_s")
    }
    for k, v in summary.items():
        print(f"{k:18s} {v * 1e3:8.1f} ms", flush=True)
    run_dir = make_run_dir("pixel_readout", label=args.label)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={"sigma": sigma, "median": summary, "per_image": rows},
        label=args.label,
        device=device,
    )


if __name__ == "__main__":
    main()
