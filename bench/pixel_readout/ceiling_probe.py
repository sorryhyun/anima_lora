#!/usr/bin/env python3
"""Which target can the head reach? (pixel readout, forward only)

P1 / P2 regress VAE(x0) — the decode of the cached clean latent — from features the
DiT computed on x_t = (1 − σ)·x0 + σ·ε at σ_last. FastFrames regresses the decode of
the latent the DiT itself produced. On the P2 held-out set (30 test + 8 text, same
picks and noise seeds) this scores, per image:

    VAE(x_t)  vs VAE(x0)   noise left in the input
    VAE(x̂0)  vs VAE(x0)   x̂0 = x_t − σ·v, the DiT's own one-step estimate
    P1, P2    vs VAE(x0)   as trained
    P1, P2    vs VAE(x̂0)  against the target FastFrames uses

Usage::

    make daemon-run ARGS="--queue --stall-timeout 600 bench/pixel_readout/ceiling_probe.py \
        --head_run bench/pixel_readout/results/<run> --label ceiling_mmdd_hhmm"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file

from bench._anima import add_common_args, add_model_args, build_anima
from bench._common import make_run_dir, start_heartbeat, write_result
from bench.pixel_readout.convnext_head import (
    P1_RUN,
    PixelHead,
    lat_patches,
    pick_text,
)
from bench.pixel_readout.linear_probe import PATCH, from_patches, pick_samples, psnr
from library.env import resolve_under_home
from library.inference.sampling import get_timesteps_sigmas
from library.io.cache import load_cached_latents, load_cached_text_features
from library.models import qwen_vae

PAIRS = (
    ("vae_xt", "vae_x0"),
    ("vae_x0hat", "vae_x0"),
    ("p1", "vae_x0"),
    ("p2", "vae_x0"),
    ("p1", "vae_x0hat"),
    ("p2", "vae_x0hat"),
)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(p, text_encoder=False)
    add_common_args(p)
    p.add_argument("--head_run", required=True)
    p.add_argument("--p1_run", default=P1_RUN)
    p.add_argument("--cache_root", default="post_image_dataset/lora")
    p.add_argument("--infer_steps", type=int, default=50)
    p.add_argument("--flow_shift", type=float, default=3.0)
    args = p.parse_args()
    start_heartbeat(label="ceiling")
    device = torch.device(args.device)

    cache_root = Path(resolve_under_home(args.cache_root))
    p1_res = json.loads(
        (Path(resolve_under_home(args.p1_run)) / "result.json").read_text()
    )
    a = p1_res["args"]
    p1_train, test, test_artists = pick_samples(
        cache_root,
        a["n_train"],
        a["n_test_artists"],
        a["per_artist"],
        a["max_tokens"],
        a["seed"],
    )
    assert test_artists == p1_res["metrics"]["test_artists"]
    text = pick_text(cache_root, {ci.npz_path for ci in p1_train})
    _, sigmas = get_timesteps_sigmas(args.infer_steps, args.flow_shift, device)
    sigma_last = float(sigmas[-2])

    vae = qwen_vae.load_vae(
        args.vae, device="cpu", dtype=torch.bfloat16, eval=True, vae_2d=True
    ).to(device)
    args.compile = False  # 38 single forwards
    anima = build_anima(args, adapter=None, train_mode=False).anima
    tap: dict[str, torch.Tensor] = {}
    anima.final_layer.linear.register_forward_pre_hook(
        lambda _m, inp: tap.__setitem__("x", inp[0].detach())
    )

    head_path = Path(resolve_under_home(args.head_run)) / "head.safetensors"
    with safe_open(str(head_path), "pt") as fh:
        meta = fh.metadata()
    D = anima.final_layer.hidden_size
    head = PixelHead(
        D,
        64,
        int(meta["width"]),
        int(meta["depth"]),
        use_skip=meta.get("skip", "1") == "1",
    ).to(device)
    head.load_state_dict(load_file(str(head_path)))
    head.eval()
    w1 = load_file(
        str(Path(resolve_under_home(args.p1_run)) / "readout_linear.safetensors")
    )["weight"]
    p1 = nn.Linear(D, PATCH * PATCH * 3).to(device)
    with torch.no_grad():
        p1.weight.copy_(w1[:, :D])
        p1.bias.copy_(w1[:, D])

    rows = []
    for j, ci in enumerate(test + text):
        with torch.no_grad():
            lat = load_cached_latents(ci.npz_path)[0].unsqueeze(0).to(device)
            cross, _ = load_cached_text_features(ci.te_path, variant=0)
            cross = cross.unsqueeze(0).to(device, torch.bfloat16)
            # same noise rule as convnext_head.py's held-out features
            g = torch.Generator(device=device).manual_seed(
                args.seed * 100003 + 10_000 + j
            )
            noise = torch.randn(lat.shape, generator=g, device=device, dtype=lat.dtype)
            x_t = (1 - sigma_last) * lat + sigma_last * noise
            sig = torch.tensor([sigma_last], device=device)
            pad = torch.zeros(
                1, 1, *lat.shape[-2:], dtype=torch.bfloat16, device=device
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = anima(
                    x_t.unsqueeze(2).to(torch.bfloat16), sig, cross, padding_mask=pad
                )
            v = v.float().squeeze(2)
            x0_hat = x_t.float() - sigma_last * v
            f = tap.pop("x")[0, 0].float()
            gh, gw = f.shape[:2]
            img = {
                "vae_x0": vae.decode_to_pixels(lat)[0].float(),
                "vae_xt": vae.decode_to_pixels(x_t)[0].float(),
                "vae_x0hat": vae.decode_to_pixels(x0_hat.to(lat.dtype))[0].float(),
                "p1": from_patches(p1(f).reshape(gh * gw, -1), gh, gw),
            }
            with torch.autocast("cuda", dtype=torch.bfloat16):
                img["p2"] = head(f, lat_patches(x_t).float()).float()
            img = {k: t.clamp(-1, 1) for k, t in img.items()}
        row = {
            "stem": ci.stem,
            "artist": Path(ci.npz_path).parent.name,
            "set": "test" if j < len(test) else "text",
            "latent_rmse_xt": float((x_t.float() - lat.float()).pow(2).mean().sqrt()),
            "latent_rmse_x0hat": float((x0_hat - lat.float()).pow(2).mean().sqrt()),
            "latent_rms_x0": float(lat.float().pow(2).mean().sqrt()),
        }
        for src, ref in PAIRS:
            row[f"{src}__vs__{ref}"] = psnr(img[src], img[ref])
        rows.append(row)
        print(
            f"  {row['set']} {ci.stem}: "
            + "  ".join(f"{s}|{r[4:]} {row[f'{s}__vs__{r}']:.2f}" for s, r in PAIRS),
            flush=True,
        )

    keys = [k for k in rows[0] if k not in ("stem", "artist", "set")]
    summary = {
        s: {k: float(np.mean([r[k] for r in rows if r["set"] == s])) for k in keys}
        for s in ("test", "text")
    }
    for s, d in summary.items():
        print(s, {k: round(v, 3) for k, v in d.items()}, flush=True)
    run_dir = make_run_dir("pixel_readout", label=args.label)
    out = write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={"sigma_last": sigma_last, "summary": summary, "per_image": rows},
        label=args.label,
        artifacts=[],
        device=device,
    )
    print(f"envelope: {out}")


if __name__ == "__main__":
    main()
