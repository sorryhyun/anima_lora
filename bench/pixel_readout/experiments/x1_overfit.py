#!/usr/bin/env python3
"""Can the ×1 head's form fit fine strokes? (pixel_decoder, capacity of the ×1 form)

The windows where the base head loses the most fine detail against the VAE decode are
found on the held-out images, then each arm overfits them against the original image.
Data size, the DiT and generalisation are out of the loop: features, x_t, the original
and VAE(x̂0) are computed once.

Scan: every held-out image (test, text, eval pages) gets one DiT forward at a decode
point drawn from ``--sigma``; the ``--init`` head and the VAE decode x̂0. A window's score
is the mean of (hp(head) − hp(original))² − (hp(VAE) − hp(original))², hp = luma minus a
Gaussian blur at σ 1.5 px, 8-bit levels. The ``--n_fit`` highest windows are kept, at
most ``--per_image`` an image and not overlapping.

Arms, each started from ``--init`` and trained whole with the recipe's regression loss:

    x1      the released form: 2 blocks at 16 px, 1 block at 8 px, an 8×8 patch per position
    deep8   + 3 blocks at 8 px
    fine4   + a 4-px stage (pixel shuffle, 2 blocks at width 256), a 4×4 patch per position

Read, on the fit windows' loss region, per arm beside VAE(x̂0) and the init head: dB
against the original; ``amp``, the high-pass std as a share of the original's;
``hp_corr``, the correlation of the two high-pass images; ``hp_err``, their RMS
difference in levels. Sheets ``win_<k>.png``: original | VAE(x̂0) | init | arms, ×2
nearest.

``--read_heads`` reads other ×1 head files on the same windows (and on the whole
held-out images) without fitting them; ``--arms ""`` runs that read alone. The windows
depend on ``--init``, ``--seed`` and the scan settings only.

Usage::

    make daemon-run ARGS="--queue --stall-timeout 600 \
        bench/pixel_readout/experiments/x1_overfit.py \
        --config project/pixel_decoder/configs/<run>.toml"
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
import tomllib
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from bench._anima import (
    DEFAULT_TEXT_ENCODER,
    add_common_args,
    add_model_args,
    build_anima,
)
from bench._common import make_run_dir, start_heartbeat, write_result
from library.anima.pixel_head import (
    PATCH,
    ConvNeXtBlock,
    PixelHeadX2,
    load_pixel_head,
)
from library.env import resolve_under_home
from library.inference.text import prepare_text_inputs
from library.models import qwen_vae
from scripts.pixel_head.train._common import luma8, pair_sheet, psnr, rec_loss
from scripts.pixel_head.train._x1_data import held_out_split
from scripts.pixel_head.train._x1_trainer import X1Rollout
from scripts.pixel_head.train._x2_trainer import crop_margin

ARMS = {  # name: (blocks added at 8 px, depth of a 4-px stage)
    "x1": (0, 0),
    "deep8": (3, 0),
    "fine4": (0, 2),
}


def build_arm(init: Path, arm: str, device) -> nn.Module:
    extra8, depth4 = ARMS[arm]
    base = load_pixel_head(init, device)
    w = base.norm_out.normalized_shape[0]
    for _ in range(extra8):
        base.up_blocks.append(ConvNeXtBlock(w, 1.0).to(device))
    if not depth4:
        return base.train()
    # the ×2 head's second stage, writing the ×1 image: a 4×4 patch per 4-px position
    head = PixelHeadX2(base, depth4, 1.0, scales=(2,)).to(device)
    head.patch2 = PATCH // 4
    head.out2 = nn.Linear(head.out2.in_features, 3 * head.patch2**2).to(device)
    return head.train()


def decode(head: nn.Module, f, latp) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if isinstance(head, PixelHeadX2):
            return head(f, latp, scale=2)
        return head(f, latp)


def high_pass(img: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    """(3, H, W) in [-1, 1] → luma minus its Gaussian blur, (H, W) in 8-bit levels."""
    y = luma8(img.clamp(-1, 1))[None, None]
    r = math.ceil(3 * sigma)
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=y.device) / sigma) ** 2)
    k = (k / k.sum()).to(y.dtype)
    b = F.conv2d(F.pad(y, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    b = F.conv2d(F.pad(b, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return (y - b)[0, 0]


def stroke_stats(pred: torch.Tensor, real: torch.Tensor) -> dict[str, float]:
    hp, hr = high_pass(pred), high_pass(real)
    a, b = hp - hp.mean(), hr - hr.mean()
    return {
        "db": psnr(pred.clamp(-1, 1), real),
        "amp": (hp.std() / hr.std().clamp_min(1e-6)).item(),
        "hp_corr": ((a * b).mean() / (a.std() * b.std()).clamp_min(1e-6)).item(),
        "hp_err": (hp - hr).pow(2).mean().sqrt().item(),
    }


def best_windows(score: torch.Tensor, w: int, n: int) -> list[tuple[float, int, int]]:
    """The ``n`` highest non-overlapping ``w``-token windows of a per-token score map,
    on a ``w``/2 grid → [(score, y0, x0)]."""
    gh, gw = score.shape
    wh, ww = min(gh, w), min(gw, w)
    s = F.avg_pool2d(score[None, None], (wh, ww), stride=max(1, w // 2))[0, 0]
    out: list[tuple[float, int, int]] = []
    for i in s.flatten().argsort(descending=True).tolist():
        y0, x0 = (i // s.shape[1]) * max(1, w // 2), (i % s.shape[1]) * max(1, w // 2)
        if all(abs(y0 - y) >= wh or abs(x0 - x) >= ww for _, y, x in out):
            out.append((s.flatten()[i].item(), y0, x0))
        if len(out) == n:
            break
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(p, text_encoder=False)
    add_common_args(p)
    p.add_argument("--config", default=None, help="a flat TOML of these flags")
    p.add_argument("--output_name", default=None, help="the run's name (the label)")
    p.add_argument("--cache_root", default="post_image_dataset/lora")
    p.add_argument(
        "--init", default="output/pixel_head/x1dll_1123_0821/head.safetensors"
    )
    p.add_argument("--arms", default="x1,deep8,fine4", help="empty = no fit")
    p.add_argument(
        "--read_heads",
        nargs="*",
        default=[],
        help="other ×1 head files, read on the same windows without fitting",
    )
    p.add_argument("--sigma", default="0.0577-0.1364")
    p.add_argument("--n_fit", type=int, default=16)
    p.add_argument("--per_image", type=int, default=2)
    p.add_argument("--window", type=int, default=16, help="tokens a side")
    p.add_argument("--margin", type=int, default=3, help="context tokens, off the loss")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--pages", default="post_image_dataset/pixel_pages")
    p.add_argument(
        "--page_caption",
        default="monochrome, greyscale, comic, speech bubble, japanese text",
    )
    p.add_argument("--stream", choices=["bf16", "fp32"], default="fp32")
    p.add_argument("--text_encoder", default=DEFAULT_TEXT_ENCODER)
    args = p.parse_args()
    if args.config:
        cfg = tomllib.loads(Path(resolve_under_home(args.config)).read_text())
        unknown = set(cfg) - set(vars(args))
        assert not unknown, f"{args.config}: unknown keys {sorted(unknown)}"
        p.set_defaults(**cfg)
        args = p.parse_args()
    label = args.output_name or args.label
    start_heartbeat(label="x1fit")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    # --- data: the held-out images ---
    data = held_out_split(Path(resolve_under_home(args.cache_root)))
    items = [(ci, "test") for ci in data["test"]] + [
        (ci, "text") for ci in data["text"]
    ]
    if args.pages:
        root = Path(resolve_under_home(args.pages))
        items += [(q, "manga") for q in sorted((root / "eval").glob("*.safetensors"))]

    # --- models ---
    vae = qwen_vae.load_vae(
        args.vae, device="cpu", dtype=torch.bfloat16, eval=True, vae_2d=True
    ).to(device)
    args.compile = False  # one forward per image, a shape per image
    anima = build_anima(args, adapter=None, train_mode=False).anima
    if args.stream == "fp32":
        from library.anima.stream import stream_fp32_

        stream_fp32_(anima)
    page_cross = None
    if args.pages:
        arg_c, _ = prepare_text_inputs(
            device=device,
            anima=anima,
            prompt=args.page_caption,
            text_encoder_path=str(resolve_under_home(args.text_encoder)),
        )
        page_cross = arg_c["embed"][0].to(device, torch.bfloat16)
    rollout = X1Rollout(anima, vae, device, args.seed, None, page_cross)
    init_path = Path(resolve_under_home(args.init))
    init_head = load_pixel_head(init_path, device)
    others = {}  # name: head, read beside the init
    for r in args.read_heads:
        q = Path(resolve_under_home(r))
        others[q.parent.name if q.stem == "head" else q.stem] = load_pixel_head(
            q, device
        )

    # --- scan: per image, its windows where the init head is furthest under the VAE ---
    lo, _, hi = args.sigma.partition("-")
    lo, hi = float(lo), float(hi or lo)
    sig_rng = random.Random(args.seed + 4)
    m = args.margin
    cands = []  # (score, name, set, σ, f, x_t patches, original, VAE(x̂0), init, margins)
    whole: dict[str, list] = {}
    with torch.no_grad():
        for i, (item, kind) in enumerate(items):
            sigma = sig_rng.uniform(lo, hi)
            f, latp, tg = rollout(item, i, sigma, targets=("xhat0", "real"))
            real, dec = tg["real"], tg["xhat0"].clamp(-1, 1)
            out0 = decode(init_head, f, latp).float().clamp(-1, 1)
            whole.setdefault(f"{kind}_vae", []).append(psnr(dec, real))
            whole.setdefault(f"{kind}_init", []).append(psnr(out0, real))
            for k, h in others.items():
                o = decode(h, f, latp).float().clamp(-1, 1)
                whole.setdefault(f"{kind}_{k}", []).append(psnr(o, real))
            hr = high_pass(real)
            d = (high_pass(out0) - hr) ** 2 - (high_pass(dec) - hr) ** 2
            score = F.avg_pool2d(d[None, None], PATCH)[0, 0]
            gh, gw = f.shape[:2]
            wh, ww = min(gh, args.window), min(gw, args.window)
            name = item.stem
            for s, y0, x0 in best_windows(score, args.window, args.per_image):
                ya, yb = max(0, y0 - m), min(gh, y0 + wh + m)
                xa, xb = max(0, x0 - m), min(gw, x0 + ww + m)
                mar = (y0 - ya, yb - (y0 + wh), x0 - xa, xb - (x0 + ww))
                px = (slice(ya * PATCH, yb * PATCH), slice(xa * PATCH, xb * PATCH))
                cands.append(
                    (
                        s,
                        f"{name}@{y0},{x0}",
                        kind,
                        sigma,
                        f[ya:yb, xa:xb].clone(),
                        latp[ya:yb, xa:xb].clone(),
                        crop_margin(real[:, px[0], px[1]], mar, PATCH).clone(),
                        crop_margin(dec[:, px[0], px[1]], mar, PATCH).clone(),
                        crop_margin(out0[:, px[0], px[1]], mar, PATCH).clone(),
                        mar,
                    )
                )
    whole = {k: float(np.mean(v)) for k, v in whole.items()}
    del anima, vae, rollout, init_head
    torch.cuda.empty_cache()
    cands.sort(key=lambda c: -c[0])
    fit = cands[: args.n_fit]
    read_outs = {}  # name: the head's decode of each fit window (with its context)
    with torch.no_grad():
        for k, h in others.items():
            read_outs[k] = [
                crop_margin(decode(h, c[4], c[5]).float().clamp(-1, 1), c[9], PATCH)
                for c in fit
            ]
    del others
    kinds = {k: sum(c[2] == k for c in fit) for k in ("test", "text", "manga")}
    print(
        f"scanned {len(items)} images, {len(cands)} windows; fit {len(fit)} ({kinds}), "
        f"score {fit[0][0]:.1f} … {fit[-1][0]:.1f} (median of all {cands[len(cands) // 2][0]:.1f})",
        flush=True,
    )

    def mean_stats(rows: list[dict]) -> dict[str, float]:
        return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

    refs = {
        "vae": [stroke_stats(c[7], c[6]) for c in fit],
        "init": [stroke_stats(c[8], c[6]) for c in fit],
    }
    for k, outs_k in read_outs.items():
        refs[k] = [stroke_stats(q, c[6]) for q, c in zip(outs_k, fit)]
    for k, rows in refs.items():
        print(
            f"  {k}: " + "  ".join(f"{a} {b:.3f}" for a, b in mean_stats(rows).items()),
            flush=True,
        )

    # --- arms ---
    run_dir = make_run_dir("pixel_readout", label=label)
    arms_out, outs = {}, {}
    for arm in filter(None, args.arms.split(",")):
        torch.manual_seed(args.seed)
        head = build_arm(init_path, arm, device)
        params = list(head.parameters())
        n_params = sum(q.numel() for q in params)
        opt = torch.optim.AdamW(
            params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01
        )
        draw = random.Random(args.seed + 3)

        @torch.no_grad()
        def evaluate(head=head):
            head.eval()
            preds = [
                crop_margin(decode(head, c[4], c[5]).float().clamp(-1, 1), c[9], PATCH)
                for c in fit
            ]
            head.train()
            return preds, [stroke_stats(q, c[6]) for q, c in zip(preds, fit)]

        hist = [{"step": 0, **mean_stats(evaluate()[1])}]
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
            c = draw.choice(fit)
            pred = crop_margin(decode(head, c[4], c[5]), c[9], PATCH)
            loss, _ = rec_loss(pred.float(), c[6])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if (step + 1) % args.eval_every == 0:
                hist.append({"step": step + 1, **mean_stats(evaluate()[1])})
                print(
                    f"  {arm} @ {step + 1}: "
                    + "  ".join(
                        f"{k} {v:.3f}" for k, v in hist[-1].items() if k != "step"
                    ),
                    flush=True,
                )
        sec = time.perf_counter() - t0
        outs[arm], rows = evaluate()
        arms_out[arm] = {
            "params_M": n_params / 1e6,
            "sec": sec,
            "history": hist,
            "final": mean_stats(rows),
            "windows": rows,
        }
        print(f"{arm}: {n_params / 1e6:.1f}M params, {sec:.0f}s", flush=True)
        del head, opt
        torch.cuda.empty_cache()

    def up2(t):
        return F.interpolate(t[None], scale_factor=2, mode="nearest")[0]

    artifacts = []
    for k, c in enumerate(fit):
        cols = [("original", c[6]), ("VAE(x^0)", c[7]), ("init head", c[8])]
        cols += [(name, o[k]) for name, o in read_outs.items()]
        cols += [(f"{arm} (fit)", outs[arm][k]) for arm in outs]
        fn = f"win_{k:02d}_{c[2]}.png"
        pair_sheet([(n, up2(t)) for n, t in cols], run_dir / fn)
        artifacts.append(fn)

    print("fit windows, mean: dB / amp / hp_corr / hp_err")
    table = {k: mean_stats(v) for k, v in refs.items()}
    table.update({arm: a["final"] for arm, a in arms_out.items()})
    for k, r in table.items():
        print(
            f"  {k:6s} {r['db']:.2f} / {r['amp']:.3f} / {r['hp_corr']:.3f} / {r['hp_err']:.2f}",
            flush=True,
        )
    out = write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={
            "n_scanned": len(items),
            "whole_image_db": whole,
            "fit": [
                {"window": c[1], "set": c[2], "sigma": c[3], "score": c[0]} for c in fit
            ],
            "refs": {k: {"mean": mean_stats(v), "windows": v} for k, v in refs.items()},
            "arms": arms_out,
        },
        label=label,
        artifacts=artifacts,
        device=device,
    )
    print(f"envelope: {out}")


if __name__ == "__main__":
    main()
