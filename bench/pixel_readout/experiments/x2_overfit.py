#!/usr/bin/env python3
"""Can the ×2 head fit 16 windows? (pixel_decoder P5 ×2, capacity ceiling of arm A)

``x2a_1119_1802`` ended 30.70 dB from the teacher on test with a second teacher draw at
42.45. This probe takes the data size, the DiT and the teacher out of the loop: features,
x_t, VAE(x̂0) and the teacher's ×2 of it are computed once for ``--n_fit`` train windows
(decode points in turn, every fourth a manga page), then each arm trains a fresh ×2 stage
on the released base head against those pairs with the run's optimizer and loss.

    x2a      2 blocks at width 256, base trainable        (= x2a_1119_1802)
    frozen   2 blocks at width 256, base frozen           (×1 output untouched)
    deep     6 blocks at width 256, base trainable
    wide     4 blocks at width 512, base trainable

Read: PSNR of the ×2 output against the teacher on the fit windows (loss region), per arm
and decode point, beside bicubic ×2 of VAE(x̂0) and a second teacher draw on the same
windows; the ×1 output against VAE(x̂0).

Usage::

    make daemon-run ARGS="--queue --stall-timeout 600 bench/pixel_readout/experiments/x2_overfit.py \
        --init mixr_1119_1455 --adapter models/loras/anima_turbo_4step.safetensors \
        --points 0.0577-0.1364 0.5:adapter --pages post_image_dataset/pixel_pages \
        --label x2fit_mmdd_hhmm"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
from safetensors.torch import load_file

from bench._anima import (
    DEFAULT_TEXT_ENCODER,
    add_common_args,
    add_model_args,
    build_anima,
)
from bench._common import make_run_dir, start_heartbeat, write_result
from bench.pixel_readout._shared import P1_RUN, bicubic2, resolve_run
from bench.pixel_readout._teacher import TEACHER_DIR, TEACHER_REPO, load_teacher
from library.anima.pixel_head import (
    PATCH,
    PixelHeadX2,
    lat_patches,
    load_pixel_head,
)
from library.anima.pixel_head_train import (
    attach_adapter,
    crop_margin,
    pair_sheet,
    psnr,
    rec_loss,
)
from library.anima.pixel_head_train_x1 import pick_samples, pick_text
from library.env import resolve_under_home
from library.inference.text import prepare_text_inputs
from library.io.cache import (
    discover_cached_pairs,
    load_cached_latents,
    load_cached_text_features,
)
from library.models import qwen_vae

ARMS = {  # name: (depth2, width2, base frozen)
    "x2a": (2, 256, False),
    "frozen": (2, 256, True),
    "deep": (6, 256, False),
    "wide": (4, 512, False),
}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(p, text_encoder=False)
    add_common_args(p)
    p.add_argument("--cache_root", default="post_image_dataset/lora")
    p.add_argument("--p1_run", default=P1_RUN)
    p.add_argument("--init", required=True)
    p.add_argument("--arms", default="x2a,frozen,deep,wide")
    p.add_argument("--n_fit", type=int, default=16)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--window", type=int, default=24)
    p.add_argument("--margin", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--pages", default=None)
    p.add_argument(
        "--page_caption",
        default="monochrome, greyscale, comic, speech bubble, japanese text",
    )
    p.add_argument("--stream", choices=["bf16", "fp32"], default="fp32")
    p.add_argument("--text_encoder", default=DEFAULT_TEXT_ENCODER)
    p.add_argument("--adapter", default=None)
    p.add_argument("--points", nargs="+", required=True)
    p.add_argument("--teacher_repo", default=TEACHER_REPO)
    p.add_argument("--teacher_dir", default=TEACHER_DIR)
    args = p.parse_args()
    start_heartbeat(label="x2fit")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed + 1)

    # --- data: train items of the parent split, every fourth window a manga page ---
    cache_root = Path(resolve_under_home(args.cache_root))
    p1 = json.loads((Path(resolve_under_home(args.p1_run)) / "result.json").read_text())
    a1 = p1["args"]
    p1_train, _, test_artists = pick_samples(
        cache_root,
        a1["n_train"],
        a1["n_test_artists"],
        a1["per_artist"],
        a1["max_tokens"],
        a1["seed"],
    )
    text = pick_text(cache_root, {ci.npz_path for ci in p1_train})
    held_paths = {ci.npz_path for ci in text}
    train = [
        ci
        for a in sorted(d.name for d in cache_root.iterdir() if d.is_dir())
        if a not in set(test_artists)
        for ci in discover_cached_pairs(str(cache_root / a))
        if ci.npz_path not in held_paths
    ]
    rng.shuffle(train)
    pages = []
    if args.pages:
        proot = Path(resolve_under_home(args.pages))
        pages = sorted((proot / "train").glob("*.safetensors"))
        rng.shuffle(pages)
    items = [pages[i] if pages and i % 4 == 3 else train[i] for i in range(args.n_fit)]
    points = []  # (name, σ lo, σ hi, adapter on)
    for spec in args.points:
        s, _, tag = spec.partition(":")
        lo, _, hi = s.partition("-")
        lo, hi = float(lo), float(hi or lo)
        points.append(("lora" if tag else "base", lo, hi, bool(tag)))

    # --- models ---
    vae = qwen_vae.load_vae(
        args.vae, device="cpu", dtype=torch.bfloat16, eval=True, vae_2d=True
    ).to(device)
    args.compile = False  # 16 forwards
    anima = build_anima(args, adapter=None, train_mode=False).anima
    if args.stream == "fp32":
        from library.anima.stream import stream_fp32_

        stream_fp32_(anima)
    network = attach_adapter(anima, args.adapter, device) if args.adapter else None
    tap: dict[str, torch.Tensor] = {}
    anima.final_layer.linear.register_forward_pre_hook(
        lambda _m, inp: tap.__setitem__("x", inp[0].detach())
    )
    up2 = load_teacher(
        Path(resolve_under_home(args.teacher_repo)),
        Path(resolve_under_home(args.teacher_dir)),
        device,
    )
    page_cross = None
    if pages:
        arg_c, _ = prepare_text_inputs(
            device=device,
            anima=anima,
            prompt=args.page_caption,
            text_encoder_path=str(resolve_under_home(args.text_encoder)),
        )
        page_cross = arg_c["embed"][0].to(device, torch.bfloat16)

    # --- the fit windows, once ---
    sig_rng = random.Random(args.seed + 4)
    win_rng = random.Random(args.seed + 5)
    wt = args.window + 2 * args.margin
    fit = []  # (point, f, lat, VAE(x̂0) loss region, teacher ×2 loss region, margins)
    refs: dict[str, list] = {}
    with torch.no_grad():
        for i, item in enumerate(items):
            name, lo, hi, adapter_on = points[i % len(points)]
            sigma = lo if hi == lo else sig_rng.uniform(lo, hi)
            if network is not None:
                network.set_enabled(adapter_on)
            if isinstance(item, Path):
                lat = load_file(str(item))["latent"].float()[None].to(device)
                cross = page_cross
            else:
                lat = load_cached_latents(item.npz_path)[0].unsqueeze(0).to(device)
                cross, _ = load_cached_text_features(item.te_path, variant=0)
                cross = cross.unsqueeze(0).to(device, torch.bfloat16)
            g = torch.Generator(device=device).manual_seed(args.seed * 100003 + i)
            noise = torch.randn(lat.shape, generator=g, device=device, dtype=lat.dtype)
            x_t = (1 - sigma) * lat + sigma * noise
            sig = torch.tensor([sigma], device=device)
            pad = torch.zeros(
                1, 1, *lat.shape[-2:], dtype=torch.bfloat16, device=device
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = anima(
                    x_t.unsqueeze(2).to(torch.bfloat16), sig, cross, padding_mask=pad
                )
            f = tap.pop("x")[0, 0].float()
            latp = lat_patches(x_t).float()
            img = vae.decode_to_pixels(x_t - sigma * v.float().squeeze(2))[0]
            img = img.float().clamp(-1, 1)
            gh, gw = f.shape[:2]
            wh, ww = min(gh, wt), min(gw, wt)
            y0, x0 = win_rng.randint(0, gh - wh), win_rng.randint(0, gw - ww)
            m = args.margin
            mar = (
                0 if y0 == 0 else m,
                0 if y0 + wh == gh else m,
                0 if x0 == 0 else m,
                0 if x0 + ww == gw else m,
            )
            win = img[:, y0 * PATCH : (y0 + wh) * PATCH, x0 * PATCH : (x0 + ww) * PATCH]
            t2 = crop_margin(up2(win, i), mar, 2 * PATCH)
            refs.setdefault(f"{name}_bicubic", []).append(
                psnr(crop_margin(bicubic2(win), mar, 2 * PATCH), t2)
            )
            refs.setdefault(f"{name}_teacher_twin", []).append(
                psnr(crop_margin(up2(win, 10_000 + i), mar, 2 * PATCH), t2)
            )
            fit.append(
                (
                    name,
                    f[y0 : y0 + wh, x0 : x0 + ww].clone(),
                    latp[y0 : y0 + wh, x0 : x0 + ww].clone(),
                    crop_margin(win, mar, PATCH).clone(),
                    t2,
                    mar,
                )
            )
    refs = {k: float(np.mean(v)) for k, v in refs.items()}
    del anima, vae, network
    torch.cuda.empty_cache()
    names = list(dict.fromkeys(n for n, *_ in fit))
    print(
        f"fit windows {len(fit)} ({', '.join(names)}); vs teacher: "
        + "  ".join(f"{k} {v:.2f}" for k, v in refs.items()),
        flush=True,
    )

    # --- arms ---
    init_path = resolve_run(args.init) / "head.safetensors"
    run_dir = make_run_dir("pixel_readout", label=args.label)
    arms_out = {}
    artifacts = []
    for arm in args.arms.split(","):
        depth2, width2, frozen = ARMS[arm]
        torch.manual_seed(args.seed)
        base = load_pixel_head(init_path, device).train()
        head = PixelHeadX2(base, depth2, 1.0, width2).to(device)
        if frozen:
            base.requires_grad_(False)
        params = [q for q in head.parameters() if q.requires_grad]
        n_new = sum(q.numel() for n, q in head.named_parameters() if "base." not in n)
        opt = torch.optim.AdamW(
            params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01
        )
        draw = random.Random(args.seed + 3)

        @torch.no_grad()
        def evaluate(head=head):
            head.eval()
            out: dict[str, list] = {}
            for name, f, latp, b1, b2, mar in fit:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    x1, x2 = head(f, latp)
                x1 = crop_margin(x1.float().clamp(-1, 1), mar, PATCH)
                x2 = crop_margin(x2.float().clamp(-1, 1), mar, 2 * PATCH)
                out.setdefault(f"{name}_x2", []).append(psnr(x2, b2))
                out.setdefault(f"{name}_x1", []).append(psnr(x1, b1))
            head.train()
            return {k: float(np.mean(v)) for k, v in out.items()}

        hist = [{"step": 0, **evaluate()}]
        t0 = time.perf_counter()
        for step in range(args.steps):
            s = (
                min(1.0, (step + 1) / args.warmup)
                * 0.5
                * (1 + math.cos(math.pi * step / args.steps))
            )
            for g in opt.param_groups:
                g["lr"] = args.lr * s
            opt.zero_grad(set_to_none=True)
            _, f, latp, b1, b2, mar = draw.choice(fit)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x1, x2 = head(f, latp)
            l2, _ = rec_loss(crop_margin(x2, mar, 2 * PATCH), b2)
            loss = l2
            if not frozen:
                l1, _ = rec_loss(crop_margin(x1, mar, PATCH), b1)
                loss = l2 + l1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if (step + 1) % args.eval_every == 0:
                hist.append({"step": step + 1, **evaluate()})
                print(
                    f"  {arm} @ {step + 1}: "
                    + "  ".join(
                        f"{k} {v:.2f}" for k, v in hist[-1].items() if k != "step"
                    ),
                    flush=True,
                )
        sec = time.perf_counter() - t0
        arms_out[arm] = {
            "depth2": depth2,
            "width2": width2,
            "base_frozen": frozen,
            "new_params_M": n_new / 1e6,
            "sec": sec,
            "history": hist,
            **{f"final_{k}": v for k, v in hist[-1].items() if k != "step"},
        }
        print(f"{arm}: {n_new / 1e6:.1f}M new params, {sec:.0f}s", flush=True)
        head.eval()
        for name in names:  # one window per decode point
            _, f, latp, _, b2, mar = next(w for w in fit if w[0] == name)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _, x2 = head(f, latp)
            x2 = crop_margin(x2.float().clamp(-1, 1), mar, 2 * PATCH)
            fn = f"{arm}_{name}.png"
            pair_sheet(
                [("VAE -> ResShift x2", b2), (f"x2 head, {arm} (fit)", x2)],
                run_dir / fn,
            )
            artifacts.append(fn)
        del head, base, opt
        torch.cuda.empty_cache()

    print("arm: ×2 vs teacher / ×1 vs VAE(x̂0), fit windows, per point")
    for arm, a in arms_out.items():
        print(
            f"  {arm}: "
            + "  ".join(
                f"{n} {a[f'final_{n}_x2']:.2f} / {a[f'final_{n}_x1']:.2f}"
                for n in names
            ),
            flush=True,
        )
    out = write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={"n_fit": len(fit), "refs": refs, "arms": arms_out},
        label=args.label,
        artifacts=artifacts,
        device=device,
    )
    print(f"envelope: {out}")


if __name__ == "__main__":
    main()
