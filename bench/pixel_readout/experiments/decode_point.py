#!/usr/bin/env python3
"""How much fine detail does the decode point's noise cost? (pixel_decoder, ×1)

For each named cached image: the original, VAE(z0) (the decode of the clean cached
latent), and over a σ sweep VAE(x̂0) and the ×1 heads read on one DiT forward at that σ.
Windows are the ``--n`` where VAE(z0)'s high-pass is furthest from the original's, so
they do not depend on σ or on a head. Per σ and column: dB, ``amp``, ``hp_corr`` and
``hp_err`` against the original (``x1_overfit.stroke_stats``), the mean over ``--draws``
noise draws and their range. Sheets ``<stem>_win<k>_<column>.png``: original | VAE(z0)
| the column at each σ (draw 0), ×2 nearest.

Images are named by stem; the artist folder is found in the cache.

Usage::

    make daemon-run ARGS="--queue --stall-timeout 600 \
        bench/pixel_readout/experiments/decode_point.py \
        --config project/pixel_decoder/configs/<run>.toml"
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bench._anima import add_common_args, add_model_args, build_anima
from bench._common import make_run_dir, start_heartbeat, write_result
from bench.pixel_readout.experiments.x1_overfit import (
    best_windows,
    decode,
    high_pass,
    stroke_stats,
)
from library.anima.pixel_head import PATCH, load_pixel_head
from library.env import resolve_under_home
from library.io.cache import discover_cached_pairs
from library.models import qwen_vae
from scripts.pixel_head.train._common import pair_sheet
from scripts.pixel_head.train._x1_trainer import X1Rollout


def find_item(root: Path, stem: str):
    hits = sorted(root.glob(f"*/{stem}_*_anima.npz"))
    assert len(hits) == 1, f"{stem}: {len(hits)} cached latents under {root}"
    pairs = discover_cached_pairs(str(hits[0].parent))
    return next(ci for ci in pairs if Path(ci.npz_path) == hits[0])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(p, text_encoder=False)
    add_common_args(p)
    p.add_argument("--config", default=None, help="a flat TOML of these flags")
    p.add_argument("--output_name", default=None, help="the run's name (the label)")
    p.add_argument("--cache_root", default="post_image_dataset/lora")
    p.add_argument("--items", nargs="+", default=[], help="image stems")
    p.add_argument("--heads", nargs="+", default=[], help="×1 head files")
    p.add_argument("--sigmas", nargs="+", type=float, default=[0.0577])
    p.add_argument("--draws", type=int, default=2, help="noise draws per σ")
    p.add_argument("--window", type=int, default=12, help="tokens a side")
    p.add_argument("--n", type=int, default=4, help="windows an image")
    p.add_argument("--stream", choices=["bf16", "fp32"], default="fp32")
    args = p.parse_args()
    if args.config:
        cfg = tomllib.loads(Path(resolve_under_home(args.config)).read_text())
        unknown = set(cfg) - set(vars(args))
        assert not unknown, f"{args.config}: unknown keys {sorted(unknown)}"
        p.set_defaults(**cfg)
        args = p.parse_args()
    label = args.output_name or args.label
    start_heartbeat(label="decode_point")
    device = torch.device(args.device)
    root = Path(resolve_under_home(args.cache_root))

    vae = qwen_vae.load_vae(
        args.vae, device="cpu", dtype=torch.bfloat16, eval=True, vae_2d=True
    ).to(device)
    args.compile = False  # one forward per (image, σ, draw), a shape per image
    anima = build_anima(args, adapter=None, train_mode=False).anima
    if args.stream == "fp32":
        from library.anima.stream import stream_fp32_

        stream_fp32_(anima)
    rollout = X1Rollout(anima, vae, device, args.seed, None, None)
    heads = {}
    for h in args.heads:
        q = Path(resolve_under_home(h))
        heads[q.parent.name] = load_pixel_head(q, device)
    names = ["VAE(xhat0)", *heads]

    run_dir = make_run_dir("pixel_readout", label=label)
    w, artifacts, metrics = args.window, [], {}

    def up2(t):
        return F.interpolate(t[None], scale_factor=2, mode="nearest")[0]

    with torch.no_grad():
        for stem in args.items:
            item = find_item(root, stem)
            _, _, tg = rollout(item, 0, args.sigmas[0], targets=("x0", "real"))
            real, z0 = tg["real"], tg["x0"].clamp(-1, 1)
            d = (high_pass(z0) - high_pass(real)) ** 2
            wins = best_windows(F.avg_pool2d(d[None, None], PATCH)[0, 0], w, args.n)
            regs = {"whole": (slice(None),) * 3}
            for k, (_, y0, x0) in enumerate(wins):
                regs[f"win{k}"] = (
                    slice(None),
                    slice(y0 * PATCH, (y0 + w) * PATCH),
                    slice(x0 * PATCH, (x0 + w) * PATCH),
                )
            rec = {
                "size": list(real.shape[-2:]),
                "windows": [[y0 * PATCH, x0 * PATCH, w * PATCH] for _, y0, x0 in wins],
                "vae_z0": {r: stroke_stats(z0[sl], real[sl]) for r, sl in regs.items()},
                "sigmas": {},
            }
            keep = {n: [] for n in names}  # draw 0 of each σ, for the sheets
            for sigma in args.sigmas:
                rows = {n: {r: [] for r in regs} for n in names}
                for draw in range(args.draws):
                    f, latp, tg = rollout(item, draw, sigma, targets=("xhat0",))
                    cols = {"VAE(xhat0)": tg["xhat0"].clamp(-1, 1)}
                    for n, h in heads.items():
                        cols[n] = decode(h, f, latp).float().clamp(-1, 1)
                    for n, img in cols.items():
                        for r, sl in regs.items():
                            rows[n][r].append(stroke_stats(img[sl], real[sl]))
                        if draw == 0:
                            keep[n].append(img.cpu())
                rec["sigmas"][f"{sigma:g}"] = {
                    n: {
                        r: {
                            k: [
                                float(np.mean([s[k] for s in ss])),
                                float(np.ptp([s[k] for s in ss])),
                            ]
                            for k in ss[0]
                        }
                        for r, ss in by.items()
                    }
                    for n, by in rows.items()
                }
            metrics[stem] = rec
            for r in regs:
                v = rec["vae_z0"][r]
                print(f"{stem} {r}: VAE(z0) {v['db']:.2f} dB corr {v['hp_corr']:.3f}")
                for sg, by in rec["sigmas"].items():
                    print(
                        f"  σ {sg:>6s}  "
                        + "  ".join(
                            f"{n} {by[n][r]['db'][0]:.2f} (±{by[n][r]['db'][1] / 2:.2f}) "
                            f"corr {by[n][r]['hp_corr'][0]:.3f} amp {by[n][r]['amp'][0]:.3f}"
                            for n in names
                        ),
                        flush=True,
                    )
            for k in range(len(wins)):
                sl = regs[f"win{k}"]
                for n in names:
                    cols = [
                        ("original", up2(real[sl].cpu())),
                        ("VAE(z0)", up2(z0[sl].cpu())),
                    ]
                    cols += [
                        (f"s {sg:g}", up2(img[sl]))
                        for sg, img in zip(args.sigmas, keep[n])
                    ]
                    fn = f"{stem}_win{k}_{n.replace('(', '_').replace(')', '')}.png"
                    pair_sheet(cols, run_dir / fn)
                    artifacts.append(fn)
    out = write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=label,
        artifacts=artifacts,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
