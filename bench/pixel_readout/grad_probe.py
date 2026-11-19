#!/usr/bin/env python3
"""Where the P2 head's gradient goes (pixel readout, forward + backward only, no step).

For a P2 head (fresh init at ``--ls_init`` or a trained ``head.safetensors``), on
``--n`` training images at σ_last: per-group gradient norms of the reconstruction
loss (P1 skip / proj / lat_embed / blocks / out), the clip factor ``min(1, 1/‖g‖)``
that training applied, the residual path's output RMS against the skip's, and the
mean pairwise cosine of per-image ``out.weight`` gradients (a consistent signal across
images reads high; a per-image one reads near 0).

Usage::

    make daemon-run ARGS="--queue --stall-timeout 600 bench/pixel_readout/grad_probe.py --label gradprobe_mmdd_hhmm"
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
from safetensors.torch import load_file

from bench._anima import add_common_args, add_model_args, build_anima
from bench._common import make_run_dir, start_heartbeat, write_result
from bench.pixel_readout.convnext_head import P1_RUN, PixelHead, lat_patches, rec_loss
from bench.pixel_readout.linear_probe import pick_samples
from library.env import resolve_under_home
from library.inference.sampling import get_timesteps_sigmas
from library.io.cache import load_cached_latents, load_cached_text_features
from library.models import qwen_vae

GROUPS = ("linear", "proj", "lat_embed", "blocks", "norm_out", "out")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(p, text_encoder=False)
    add_common_args(p)
    p.add_argument("--cache_root", default="post_image_dataset/lora")
    p.add_argument(
        "--heads",
        default="init:1e-6,init:0.1,run:bench/pixel_readout/results/20261119-0900-convnext_1119_0835",
        help="comma list of init:<ls_init> | run:<results dir>",
    )
    p.add_argument("--n", type=int, default=16)
    args = p.parse_args()
    start_heartbeat(label="gradprobe")
    device = torch.device(args.device)

    cache_root = Path(resolve_under_home(args.cache_root))
    train, _, _ = pick_samples(
        cache_root, 2000, 8, 4, 4608, 0
    )  # P1's split, train side
    random.Random(7).shuffle(train)
    items = train[: args.n]
    _, sigmas = get_timesteps_sigmas(50, 3.0, device)
    sigma_last = float(sigmas[-2])

    vae = qwen_vae.load_vae(
        args.vae, device="cpu", dtype=torch.bfloat16, eval=True, vae_2d=True
    ).to(device)
    args.compile = False
    anima = build_anima(args, adapter=None, train_mode=False).anima
    tap: dict[str, torch.Tensor] = {}
    anima.final_layer.linear.register_forward_pre_hook(
        lambda _m, inp: tap.__setitem__("x", inp[0].detach())
    )
    D = anima.final_layer.hidden_size

    data = []
    with torch.no_grad():
        for i, ci in enumerate(items):
            lat = load_cached_latents(ci.npz_path)[0].unsqueeze(0).to(device)
            tgt = vae.decode_to_pixels(lat)[0].float()
            cross, _ = load_cached_text_features(ci.te_path, variant=0)
            g = torch.Generator(device=device).manual_seed(i)
            noise = torch.randn(lat.shape, generator=g, device=device, dtype=lat.dtype)
            x_t = (1 - sigma_last) * lat + sigma_last * noise
            pad = torch.zeros(
                1, 1, *lat.shape[-2:], dtype=torch.bfloat16, device=device
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                anima(
                    x_t.unsqueeze(2).to(torch.bfloat16),
                    torch.tensor([sigma_last], device=device),
                    cross.unsqueeze(0).to(device, torch.bfloat16),
                    padding_mask=pad,
                )
            data.append((tap.pop("x")[0, 0].float(), lat_patches(x_t).float(), tgt))
    del vae, anima
    torch.cuda.empty_cache()

    w1 = load_file(
        str(Path(resolve_under_home(P1_RUN)) / "readout_linear.safetensors")
    )["weight"]
    results = {}
    for spec in args.heads.split(","):
        kind, val = spec.split(":", 1)
        if kind == "init":
            head = PixelHead(D, 64, 1024, 3, float(val)).to(device)
            with torch.no_grad():
                head.linear.weight.copy_(w1[:, :D])
                head.linear.bias.copy_(w1[:, D])
        else:
            sd = load_file(str(Path(resolve_under_home(val)) / "head.safetensors"))
            head = PixelHead(D, 64, 1024, 3).to(device)
            head.load_state_dict(sd)
        norms = {g: [] for g in GROUPS}
        totals, out_grads, rms_res, rms_skip = [], [], [], []
        for f, lat, tgt in data:
            head.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = head(f, lat)
            loss, _ = rec_loss(pred, tgt)
            loss.backward()
            for g in GROUPS:
                sq = sum(
                    p.grad.float().pow(2).sum().item()
                    for n, p in head.named_parameters()
                    if n.split(".")[0] == g and p.grad is not None
                )
                norms[g].append(sq**0.5)
            totals.append(sum(v[-1] ** 2 for v in norms.values()) ** 0.5)
            out_grads.append(head.out.weight.grad.flatten().float().clone())
            with torch.no_grad():
                skip = head.linear(f)
                h = (head.proj(f) + head.lat_embed(lat)).permute(2, 0, 1)[None]
                for b in head.blocks:
                    h = b(h)
                res = head.out(head.norm_out(h[0].permute(1, 2, 0)))
                rms_res.append(res.pow(2).mean().sqrt().item())
                rms_skip.append(skip.pow(2).mean().sqrt().item())
        G = torch.stack(out_grads)
        G = G / G.norm(dim=1, keepdim=True).clamp_min(1e-30)
        C = G @ G.T
        n = C.shape[0]
        cos = ((C.sum() - n) / (n * (n - 1))).item()
        tot = np.array(totals)
        results[spec] = {
            "grad_norm": {g: float(np.mean(v)) for g, v in norms.items()},
            "grad_share": {
                g: float(np.mean(np.array(v) ** 2 / tot**2)) for g, v in norms.items()
            },
            "total_norm_mean": float(tot.mean()),
            "clip_factor_mean": float(np.mean(np.minimum(1.0, 1.0 / tot))),
            "out_grad_pairwise_cos": cos,
            "rms_residual": float(np.mean(rms_res)),
            "rms_skip": float(np.mean(rms_skip)),
        }
        r = results[spec]
        print(
            f"[{spec}] total ‖g‖ {r['total_norm_mean']:.3g}  clip× {r['clip_factor_mean']:.3f}  "
            f"out-grad cos {cos:.3f}  rms res/skip {r['rms_residual']:.3g}/{r['rms_skip']:.3g}",
            flush=True,
        )
        for g in GROUPS:
            print(
                f"    {g:10s} ‖g‖ {r['grad_norm'][g]:.3g}  share {r['grad_share'][g]:.3f}",
                flush=True,
            )

    run_dir = make_run_dir("pixel_readout", label=args.label)
    out = write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={"sigma_last": sigma_last, "n": len(items), "heads": results},
        label=args.label,
        artifacts=[],
        device=device,
    )
    print(f"envelope: {out}")


if __name__ == "__main__":
    main()
