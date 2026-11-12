"""Reference DiT outputs for the browser DiT check (dit_onnx_eval.py).

At the 512 bucket (64x64 latent): a cached latent area-downsampled to 64x64, noised at
each σ of the 4-step shift-3 grid, its own cached text features; the DiT (``--dit``) in
fp32. Inputs and outputs go to one npz, in the ONNX graph's layout (latent (1,16,1,64,64),
timestep (1,) = σ, encoder_hidden_states (1,512,1024), noise_pred like latent).

    make daemon-run ARGS="--stall-timeout 600 project/webgpu_int8/src/dit_ref_dump.py \
        --dit models/diffusion_models/anima_turbo_4step_v4_preview.safetensors"
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F

from bench._anima import add_common_args, add_model_args, build_anima
from library.env import resolve_under_home
from library.io.cache import (
    discover_cached_pairs,
    load_cached_latents,
    load_cached_text_features,
)

SIGMAS = (1.0, 0.9, 0.75, 0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(ap, vae=False, text_encoder=True)
    add_common_args(ap, include_compile=False, include_checkpointing=False)
    ap.add_argument("--cache_root", default="post_image_dataset/lora")
    ap.add_argument("--n", type=int, default=8, help="images (each at every σ)")
    ap.add_argument("--out", default="output/webgpu_int8/dit_ref_512.npz")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    bundle = build_anima(args, adapter=None, train_mode=False)
    anima, device = bundle.anima, bundle.device
    anima.float()
    root = resolve_under_home(args.cache_root)
    pairs = [
        ci
        for d in sorted(p for p in root.iterdir() if p.is_dir())
        for ci in discover_cached_pairs(str(d))
    ]
    random.Random(1).shuffle(pairs)
    lat, ts, ctx, out = [], [], [], []
    for i, ci in enumerate(pairs[: args.n]):
        x0 = load_cached_latents(ci.npz_path)[0].unsqueeze(0).float().to(device)
        x0 = F.adaptive_avg_pool2d(x0, (64, 64))
        c, _ = load_cached_text_features(ci.te_path, variant=0)
        c = c.unsqueeze(0).float().to(device)
        g = torch.Generator(device=device).manual_seed(i)
        noise = torch.randn(x0.shape, generator=g, device=device)
        for s in SIGMAS:
            x = ((1 - s) * x0 + s * noise).unsqueeze(2)
            sig = torch.tensor([s], device=device)
            pad = torch.zeros(1, 1, 64, 64, device=device)
            y = anima(x, sig, c, padding_mask=pad)
            lat.append(x.cpu().numpy())
            ts.append(np.array([s], np.float32))
            ctx.append(c.cpu().numpy())
            out.append(y.float().cpu().numpy())
    o = resolve_under_home(args.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        o,
        latent=np.concatenate(lat),
        timestep=np.concatenate(ts),
        encoder_hidden_states=np.concatenate(ctx),
        noise_pred=np.concatenate(out),
    )
    print("wrote", o, np.concatenate(out).shape, flush=True)


if __name__ == "__main__":
    main()
