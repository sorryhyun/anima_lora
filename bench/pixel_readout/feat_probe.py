#!/usr/bin/env python3
"""Which DiT features hold pixel information: the last step's, the step before, a mid block.

P1's ridge readout (``linear_probe.py``: a per-token linear map to a 16×16 RGB patch, fit
in closed form on train images, scored on held-out artists) with the input changed. Three
feature groups per token, 2048-d each:

* ``L``: the final layer's input at the sampler's last σ (what the pixel head reads);
* ``P``: the same tap one step earlier on the grid, same image and same noise draw;
* ``M``: the output of block ``--mid_block`` at the last σ, layer-normed per token.

Arms are ``L``, ``P``, ``M``, ``L+P``, ``L+M``; targets are the VAE decode of the clean
latent and the image the latent was encoded from. Every arm is scored on the same test
images, so the read is the paired per-image gap to ``L``.

Usage::

    make daemon-run ARGS="--queue --stall-timeout 600 bench/pixel_readout/feat_probe.py \
        --compile --label featprobe_mmdd_hhmm"
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

from bench._anima import add_common_args, add_model_args, build_anima
from bench._common import make_run_dir, start_heartbeat, write_result
from bench.pixel_readout._compile import compile_blocks_isolated
from bench.pixel_readout.convnext_head import real_pixels
from bench.pixel_readout.linear_probe import pick_samples, psnr, to_patches
from library.anima.pixel_head import PATCH, from_patches
from library.env import resolve_under_home
from library.inference.sampling import get_timesteps_sigmas
from library.io.cache import load_cached_latents, load_cached_text_features
from library.models import qwen_vae

ARMS = {"L": "L", "P": "P", "M": "M", "L+P": "LP", "L+M": "LM"}
GROUPS = "LPM"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(p, text_encoder=False)
    add_common_args(p)
    p.add_argument("--cache_root", default="post_image_dataset/lora")
    p.add_argument("--n_train", type=int, default=256)
    p.add_argument("--n_test_artists", type=int, default=8)
    p.add_argument("--per_artist", type=int, default=4)
    p.add_argument("--max_tokens", type=int, default=4608)
    p.add_argument("--infer_steps", type=int, default=28)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument("--mid_block", type=int, default=None, help="default: 2/3 depth")
    p.add_argument("--ridge", default="1e-6,1e-4,1e-2")
    p.add_argument("--stream", choices=["bf16", "fp32"], default="fp32")
    args = p.parse_args()
    start_heartbeat(label="featprobe")
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
    sigma_last, sigma_prev = float(sigmas[-2]), float(sigmas[-3])
    print(
        f"train {len(train)}  test {len(test)}  sigma last {sigma_last:.4f} "
        f"prev {sigma_prev:.4f}",
        flush=True,
    )

    # --- targets: VAE(x0) and the image each latent was encoded from ---
    vae = qwen_vae.load_vae(
        args.vae, device="cpu", dtype=torch.bfloat16, eval=True, vae_2d=True
    ).to(device)
    targets: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for ci in train + test:
        lat = load_cached_latents(ci.npz_path)[0].unsqueeze(0).to(device)
        with torch.no_grad():
            px = vae.decode_to_pixels(lat)[0].float().clamp(-1, 1)
        orig = real_pixels(ci).to(device).float() / 127.5 - 1
        assert orig.shape == px.shape, (ci.npz_path, orig.shape, px.shape)
        targets[ci.npz_path] = (
            px.to("cpu", torch.float16),
            orig.to("cpu", torch.float16),
        )
    del vae
    torch.cuda.empty_cache()

    # --- DiT with three taps ---
    compile_blocks, args.compile = args.compile, False
    anima = build_anima(args, adapter=None, train_mode=False).anima
    args.compile = compile_blocks
    if args.stream == "fp32":
        from library.anima.stream import stream_fp32_

        stream_fp32_(anima)
    if compile_blocks:
        compile_blocks_isolated(anima)
    D = anima.final_layer.hidden_size
    n_blocks = len(anima.blocks)
    mid = args.mid_block if args.mid_block is not None else (2 * n_blocks) // 3
    tap: dict[str, torch.Tensor] = {}
    anima.final_layer.linear.register_forward_pre_hook(
        lambda _m, inp: tap.__setitem__("last", inp[0].detach())
    )
    anima.blocks[mid].register_forward_hook(
        lambda _m, _inp, out: tap.__setitem__("mid", out.detach())
    )
    seen_shape = []

    @torch.no_grad()
    def features(ci, idx: int) -> tuple[torch.Tensor, int, int]:
        """→ ((gh·gw, 3D) = [L, P, M], gh, gw)."""
        lat = load_cached_latents(ci.npz_path)[0].unsqueeze(0).to(device)
        cross, _ = load_cached_text_features(ci.te_path, variant=0)
        cross = cross.unsqueeze(0).to(device, torch.bfloat16)
        g = torch.Generator(device=device).manual_seed(args.seed * 100003 + idx)
        noise = torch.randn(lat.shape, generator=g, device=device, dtype=lat.dtype)
        pad = torch.zeros(1, 1, *lat.shape[-2:], dtype=torch.bfloat16, device=device)
        out = {}
        for key, sigma in (("P", sigma_prev), ("L", sigma_last)):
            x_t = ((1 - sigma) * lat + sigma * noise).unsqueeze(2).to(torch.bfloat16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                anima(
                    x_t, torch.tensor([sigma], device=device), cross, padding_mask=pad
                )
            x = tap.pop("last")  # (1, 1, gh, gw, D)
            gh, gw = x.shape[2], x.shape[3]
            out[key] = x.reshape(gh * gw, D).float()
        m = tap.pop("mid")
        if not seen_shape:
            seen_shape.append(tuple(m.shape))
            print(f"block {mid} output {tuple(m.shape)}, grid {gh} × {gw}", flush=True)
        m = m.reshape(-1, D).float()
        assert m.shape[0] == gh * gw, (m.shape, gh, gw)
        out["M"] = F.layer_norm(m, (D,))
        return torch.cat([out[k] for k in GROUPS], 1), gh, gw

    n = 3 * D + 1
    xtx = torch.zeros(n, n, dtype=torch.float64, device=device)
    xty = {
        k: torch.zeros(n, PATCH * PATCH * 3, dtype=torch.float64, device=device)
        for k in ("vae", "orig")
    }
    t0 = time.perf_counter()
    for i, ci in enumerate(train):
        x, gh, gw = features(ci, i)
        x = torch.cat([x, torch.ones_like(x[:, :1])], 1).double()
        px, orig = targets.pop(ci.npz_path)
        xtx += x.T @ x
        xty["vae"] += x.T @ to_patches(px.to(device).float()).double()
        xty["orig"] += x.T @ to_patches(orig.to(device).float()).double()
        if (i + 1) % 32 == 0:
            print(
                f"  fit {i + 1}/{len(train)} "
                f"{(time.perf_counter() - t0) / (i + 1):.2f}s/img",
                flush=True,
            )
    test_feats = []
    for j, ci in enumerate(test):
        x, gh, gw = features(ci, 10_000 + j)
        test_feats.append((ci, x.to("cpu", torch.bfloat16), gh, gw))

    # --- solve each arm on its sub-block of the normal equations ---
    ridges = [float(r) for r in args.ridge.split(",")]
    results: dict[str, dict] = {}
    per_image: dict[str, dict[str, list[float]]] = {}
    for arm, groups in ARMS.items():
        cols = torch.cat(
            [
                torch.arange(GROUPS.index(g) * D, (GROUPS.index(g) + 1) * D)
                for g in groups
            ]
            + [torch.tensor([3 * D])]
        ).to(device)
        a = xtx[cols][:, cols]
        d = len(cols) - 1
        eye = torch.eye(d + 1, dtype=torch.float64, device=device)
        eye[d, d] = 0  # no ridge on the bias
        mean_diag = a.diagonal()[:d].mean().item()
        results[arm], per_image[arm] = {}, {}
        for tk in ("vae", "orig"):
            best = None
            for r in ridges:
                w = torch.linalg.solve(a + r * mean_diag * eye, xty[tk][cols]).float()
                scores = []
                for ci, x, gh, gw in test_feats:
                    xd = x.to(device).float()
                    xd = torch.cat([xd, torch.ones_like(xd[:, :1])], 1)[:, cols]
                    pred = from_patches(xd @ w, gh, gw).clamp(-1, 1)
                    tgt = targets[ci.npz_path][0 if tk == "vae" else 1]
                    scores.append(psnr(pred, tgt.to(device).float()))
                if best is None or np.mean(scores) > np.mean(best[1]):
                    best = (r, scores)
            results[arm][tk] = {"ridge": best[0], "psnr": float(np.mean(best[1]))}
            per_image[arm][tk] = best[1]
        print(
            f"  {arm:4s} vs VAE {results[arm]['vae']['psnr']:.2f} (ridge "
            f"{results[arm]['vae']['ridge']:g})  vs original "
            f"{results[arm]['orig']['psnr']:.2f} (ridge {results[arm]['orig']['ridge']:g})",
            flush=True,
        )

    paired = {}
    for arm in ARMS:
        if arm == "L":
            continue
        paired[arm] = {}
        for tk in ("vae", "orig"):
            dlt = np.array(per_image[arm][tk]) - np.array(per_image["L"][tk])
            paired[arm][tk] = {
                "mean": float(dlt.mean()),
                "min": float(dlt.min()),
                "max": float(dlt.max()),
                "n_up": int((dlt > 0).sum()),
                "n": len(dlt),
            }
            q = paired[arm][tk]
            print(
                f"  {arm:4s} − L, {tk}: {q['mean']:+.2f} dB ({q['n_up']} / {q['n']} up, "
                f"{q['min']:+.2f} … {q['max']:+.2f})",
                flush=True,
            )

    run_dir = make_run_dir("pixel_readout", label=args.label)
    out = write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={
            "sigma_last": sigma_last,
            "sigma_prev": sigma_prev,
            "mid_block": mid,
            "n_blocks": n_blocks,
            "n_train": len(train),
            "n_test": len(test),
            "test_artists": test_artists,
            "arms": results,
            "paired_vs_L": paired,
            "per_image": per_image,
        },
        label=args.label,
        artifacts=[],
        device=device,
    )
    print(f"envelope: {out}")


if __name__ == "__main__":
    main()
