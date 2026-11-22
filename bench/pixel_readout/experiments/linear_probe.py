#!/usr/bin/env python3
"""Linear pixel readout from the frozen Anima DiT's pre-head features (FastFrames probe 1).

FastFrames (arXiv:2611.23226 §3.2) replaces the VAE decoder with a readout on the
frozen DiT's pre-head features; a single linear layer, fit with MSE to the VAE
output, gave "blurry yet recognizable" frames on Wan 2.1. This probe asks the
same of Anima: does a per-token linear map from the features ``final_layer.linear``
reads (the LayerNorm'd, adaLN-modulated 2048-dim token) to a 16×16 RGB patch
reproduce the VAE decode?

Per image: VAE-decode the cached clean latent x0 (the target), noise x0 to the
last σ of the default sampler grid (50 steps, shift 3), run one DiT forward with
the image's cached caption, and tap the final layer's input. Train images
accumulate XᵀX and XᵀY; the readout is the ridge least-squares solution (the MSE
optimum of a linear layer, no SGD). Test images come from artists held out of the
fit and are scored against the VAE decode and against a token-mean reference (the
VAE image averaged per 16×16 patch, i.e. one colour per token).

Usage::

    make daemon-run ARGS="--queue --stall-timeout 600 bench/pixel_readout/experiments/linear_probe.py --compile --label readout_1118_HHMM"
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from bench._anima import add_common_args, add_model_args, build_anima
from bench._common import make_run_dir, start_heartbeat, write_result
from library.anima.pixel_head import PATCH, from_patches
from library.anima.pixel_head_train import compile_dit, psnr, zoom_sheet
from library.anima.pixel_head_train_x1 import pick_samples
from library.env import resolve_under_home
from library.inference.sampling import get_timesteps_sigmas
from library.io.cache import (
    load_cached_latents,
    load_cached_text_features,
)
from library.models import qwen_vae


def to_patches(img: torch.Tensor) -> torch.Tensor:
    """(3, H, W) → (H/16 · W/16, 768), row-major tokens, (py, px, c) within a patch."""
    c, h, w = img.shape
    return (
        img.view(c, h // PATCH, PATCH, w // PATCH, PATCH)
        .permute(1, 3, 2, 4, 0)
        .reshape((h // PATCH) * (w // PATCH), PATCH * PATCH * c)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(p, text_encoder=False)
    add_common_args(p)
    p.add_argument("--cache_root", default="post_image_dataset/lora")
    p.add_argument("--n_train", type=int, default=256)
    p.add_argument("--n_test_artists", type=int, default=8)
    p.add_argument("--per_artist", type=int, default=4)
    p.add_argument("--max_tokens", type=int, default=4608)
    p.add_argument("--infer_steps", type=int, default=50)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument(
        "--ridge",
        default="1e-6,1e-4,1e-2",
        help="relative ridge λ (× mean diag of XᵀX), comma list",
    )
    p.add_argument("--n_sheets", type=int, default=12)
    args = p.parse_args()
    start_heartbeat(label="readout")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    cache_root = Path(resolve_under_home(args.cache_root))
    train, test, test_artists = pick_samples(
        cache_root,
        args.n_train,
        args.n_test_artists,
        args.per_artist,
        args.max_tokens,
        args.seed,
    )
    _, sigmas = get_timesteps_sigmas(args.infer_steps, args.flow_shift, device)
    sigma_last = float(sigmas[-2])
    print(
        f"train {len(train)}  test {len(test)} ({', '.join(test_artists)})  "
        f"sigma_last {sigma_last:.4f}",
        flush=True,
    )

    # --- stage 1: VAE targets (VAE → free → DiT, the lazy-loading order) ---
    vae = qwen_vae.load_vae(
        args.vae, device="cpu", dtype=torch.bfloat16, eval=True, vae_2d=True
    )
    vae.to(device)
    targets: dict[str, torch.Tensor] = {}
    t0 = time.perf_counter()
    for i, ci in enumerate(train + test):
        lat = load_cached_latents(ci.npz_path)[0].unsqueeze(0).to(device)
        with torch.no_grad():
            px = vae.decode_to_pixels(lat)[0]
        targets[ci.npz_path] = px.to("cpu", torch.float16)
        if (i + 1) % 32 == 0:
            print(
                f"  vae {i + 1}/{len(train) + len(test)} "
                f"{(time.perf_counter() - t0) / (i + 1):.2f}s/img",
                flush=True,
            )
    del vae
    torch.cuda.empty_cache()

    # --- stage 2: DiT features at sigma_last ---
    # compile here, not in build_anima: the samples mix bucket shapes, so mark the
    # seq axis dynamic (no adapter, so compile-after-apply holds trivially)
    compile_blocks, args.compile = args.compile, False
    bundle = build_anima(args, adapter=None, train_mode=False)
    args.compile = compile_blocks
    anima = bundle.anima
    if compile_blocks:
        compile_dit(anima)
    tap: dict[str, torch.Tensor] = {}
    anima.final_layer.linear.register_forward_pre_hook(
        lambda _m, inp: tap.__setitem__("x", inp[0].detach())
    )

    def features(ci, idx: int):
        lat = load_cached_latents(ci.npz_path)[0].unsqueeze(0).to(device)
        cross, _ = load_cached_text_features(ci.te_path, variant=0)
        cross = cross.unsqueeze(0).to(device, torch.bfloat16)
        g = torch.Generator(device=device).manual_seed(args.seed * 100003 + idx)
        noise = torch.randn(lat.shape, generator=g, device=device, dtype=lat.dtype)
        x_t = (
            ((1 - sigma_last) * lat + sigma_last * noise)
            .unsqueeze(2)
            .to(torch.bfloat16)
        )
        sig = torch.tensor([sigma_last], device=device)
        pad = torch.zeros(1, 1, *lat.shape[-2:], dtype=torch.bfloat16, device=device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            anima(x_t, sig, cross, padding_mask=pad)
        x = tap.pop("x")  # (1, 1, gh, gw, D)
        gh, gw = x.shape[2], x.shape[3]
        return x.reshape(gh * gw, -1).float(), gh, gw

    D = anima.final_layer.hidden_size
    xtx = torch.zeros(D + 1, D + 1, dtype=torch.float64, device=device)
    xty = torch.zeros(D + 1, PATCH * PATCH * 3, dtype=torch.float64, device=device)
    n_tok = 0
    t0 = time.perf_counter()
    for i, ci in enumerate(train):
        x, gh, gw = features(ci, i)
        x = torch.cat([x, torch.ones_like(x[:, :1])], 1)
        y = to_patches(targets.pop(ci.npz_path).to(device).float())
        xtx += (x.T @ x).double()
        xty += (x.T @ y).double()
        n_tok += x.shape[0]
        if (i + 1) % 32 == 0:
            print(
                f"  fit {i + 1}/{len(train)} {(time.perf_counter() - t0) / (i + 1):.2f}s/img",
                flush=True,
            )

    test_feats = []
    for j, ci in enumerate(test):
        x, gh, gw = features(ci, 10_000 + j)
        test_feats.append((ci, x.to("cpu", torch.bfloat16), gh, gw))

    # --- solve and score ---
    mean_diag = xtx.diagonal()[:D].mean().item()
    eye = torch.eye(D + 1, dtype=torch.float64, device=device)
    eye[D, D] = 0  # no ridge on the bias
    ridges = [float(r) for r in args.ridge.split(",")]
    by_ridge, weights = {}, {}
    for r in ridges:
        w = torch.linalg.solve(xtx + r * mean_diag * eye, xty).float()
        weights[r] = w
        scores = []
        for ci, x, gh, gw in test_feats:
            x = torch.cat(
                [x.to(device).float(), torch.ones(x.shape[0], 1, device=device)], 1
            )
            pred = from_patches(x @ w, gh, gw).clamp(-1, 1)
            scores.append(psnr(pred, targets[ci.npz_path].to(device)))
        by_ridge[r] = float(np.mean(scores))
        print(f"  ridge {r:g}: test PSNR vs VAE {by_ridge[r]:.2f} dB", flush=True)
    best = max(by_ridge, key=by_ridge.get)
    w = weights[best]

    run_dir = make_run_dir("pixel_readout", label=args.label)
    rows, artifacts = [], []
    for k, (ci, x, gh, gw) in enumerate(test_feats):
        x = torch.cat(
            [x.to(device).float(), torch.ones(x.shape[0], 1, device=device)], 1
        )
        pred = from_patches(x @ w, gh, gw).clamp(-1, 1)
        tgt = targets[ci.npz_path].to(device).float()
        tok_mean = F.interpolate(
            F.avg_pool2d(tgt[None], PATCH), scale_factor=PATCH, mode="nearest"
        )[0]
        tok_bilin = F.interpolate(
            F.avg_pool2d(tgt[None], PATCH),
            scale_factor=PATCH,
            mode="bilinear",
            align_corners=False,
        )[0]
        rows.append(
            {
                "stem": ci.stem,
                "artist": Path(ci.npz_path).parent.name,
                "psnr_linear": psnr(pred, tgt),
                "psnr_token_mean": psnr(tok_mean, tgt),
                "psnr_token_bilinear": psnr(tok_bilin, tgt),
            }
        )
        if k < args.n_sheets:
            name = f"sheet_{k:02d}_{ci.stem}.png"
            zoom_sheet(
                [
                    ("VAE decode", tgt),
                    ("linear readout", pred),
                    ("token mean, bilinear", tok_bilin),
                ],
                run_dir / name,
            )
            artifacts.append(name)

    save_file(
        {"weight": w.T.contiguous().cpu(), "ridge": torch.tensor(best)},
        str(run_dir / "readout_linear.safetensors"),
    )
    artifacts.append("readout_linear.safetensors")

    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    metrics = {
        "sigma_last": sigma_last,
        "n_train": len(train),
        "n_train_tokens": n_tok,
        "n_test": len(test),
        "test_artists": test_artists,
        "ridge_rel": best,
        "test_psnr_by_ridge": by_ridge,
        "test_psnr_linear": mean("psnr_linear"),
        "test_psnr_token_mean": mean("psnr_token_mean"),
        "test_psnr_token_bilinear": mean("psnr_token_bilinear"),
        "per_image": rows,
    }
    print(
        f"test PSNR vs VAE: linear {metrics['test_psnr_linear']:.2f}  "
        f"token-mean {metrics['test_psnr_token_mean']:.2f}  "
        f"token-bilinear {metrics['test_psnr_token_bilinear']:.2f}",
        flush=True,
    )
    out = write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=artifacts,
        device=device,
    )
    print(f"envelope: {out}")


if __name__ == "__main__":
    main()
