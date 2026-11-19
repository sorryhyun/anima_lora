#!/usr/bin/env python3
"""ConvNeXt pixel head on the frozen Anima DiT's pre-head features (pixel readout P2).

``project/pixel_decoder/proposal.md`` P2. The head reads the same tap as
probe 1 (the input of ``final_layer.linear``: LayerNorm + adaLN-modulated, 2048-dim
per token) plus x_t's 2×2 latent patch (64 values), and predicts a 16×16 RGB patch
per token:

    pred = P1_linear(f) + out(ConvNeXt³(proj(f) + embed(x_t)))

``P1_linear`` starts at probe 1's ridge solution (``readout_1118_2219``) and trains at a
lower LR; ``out`` is zero-initialised, so step 0 reproduces P1. The three ConvNeXt
blocks (3×3 depthwise, LayerNorm, 4× GELU MLP, layer scale) run at token resolution,
no upsampler (FastFrames App. C, 2D).

Training: anima-base cache minus P1's 8 held-out artists and the 8 text images below;
per rollout one DiT forward at σ_last (50 steps, shift 3) from the cached latent, one VAE
decode of the clean latent (target), then ``--opt_steps`` head updates on that pair
(FastFrames: 4 per rollout). Loss 0.5 L1 + 5 MSE + 0.05 L_grad (finite differences).

Read: P1's 30 test images (same picker, same seed → same images; sheets for the first 12)
and 8 images with Japanese text, one per artist from the most text-heavy artists, held out
of training (sheets cropped at the densest-edge window, 2×). Columns VAE | P1 linear |
P2 head, labelled.

Usage::

    make daemon-run ARGS="--queue --stall-timeout 900 bench/pixel_readout/convnext_head.py --compile --label convnext_mmdd_hhmm"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from safetensors.torch import load_file, save_file

from bench._anima import add_common_args, add_model_args, build_anima
from bench._common import make_run_dir, start_heartbeat, write_result
from bench.pixel_readout.linear_probe import (
    PATCH,
    from_patches,
    pick_samples,
    psnr,
    to_pil,
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

P1_RUN = "bench/pixel_readout/results/20261118-2221-readout_1118_2219"
TEXT_ARTISTS = (  # most "Japanese text" captions in post_image_dataset/resized
    "sincos",
    "suujiniku",
    "ie_(raarami)",
    "tottotonero",
    "kase_daiki",
    "ama_mitsuki",
    "sumiyao_(amam)",
    "kat_(bu-kunn)",
)


# --------------------------------------------------------------------------- head


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, ls_init: float = 1e-6):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.full((dim,), ls_init))

    def forward(self, x):  # (B, C, H, W)
        h = self.dw(x).permute(0, 2, 3, 1)
        h = self.fc2(F.gelu(self.fc1(self.norm(h))))
        return x + (self.gamma * h).permute(0, 3, 1, 2)


class PixelHead(nn.Module):
    def __init__(
        self,
        d_feat: int,
        d_lat: int,
        width: int,
        depth: int,
        ls_init: float = 1e-6,
        use_skip: bool = True,
    ):
        super().__init__()
        out = PATCH * PATCH * 3
        self.use_skip = use_skip
        self.linear = nn.Linear(d_feat, out)  # P1 skip (unused when use_skip is off)
        self.proj = nn.Linear(d_feat, width)
        self.lat_embed = nn.Linear(d_lat, width)
        self.blocks = nn.ModuleList(ConvNeXtBlock(width, ls_init) for _ in range(depth))
        self.norm_out = nn.LayerNorm(width, eps=1e-6)
        self.out = nn.Linear(width, out)
        if (
            use_skip
        ):  # step 0 reproduces P1; without the skip, out keeps its default init
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, f, lat):  # f (gh, gw, D), lat (gh, gw, 64) → (3, H, W)
        gh, gw = f.shape[:2]
        h = (self.proj(f) + self.lat_embed(lat)).permute(2, 0, 1)[None]
        for b in self.blocks:
            h = b(h)
        h = self.norm_out(h[0].permute(1, 2, 0))
        # the ridge solution loses ~dB under bf16 weights; keep the skip in fp32
        with torch.autocast("cuda", enabled=False):
            tok = self.out(h).float()
            if self.use_skip:
                tok = tok + self.linear(f.float())
        return from_patches(tok.reshape(gh * gw, -1).float(), gh, gw)


def rec_loss(pred, tgt):
    l1 = (pred - tgt).abs().mean()
    mse = ((pred - tgt) ** 2).mean()
    gh = (pred.diff(dim=-2) - tgt.diff(dim=-2)).abs().mean()
    gw = (pred.diff(dim=-1) - tgt.diff(dim=-1)).abs().mean()
    return 0.5 * l1 + 5.0 * mse + 0.05 * (gh + gw), mse


# --------------------------------------------------------------------------- data


def caption_path(ci) -> Path:
    artist = Path(ci.npz_path).parent.name
    return Path(
        resolve_under_home(f"post_image_dataset/resized/{artist}/{ci.stem}.txt")
    )


def pick_text(cache_root: Path, exclude: set[str]) -> list:
    """Per text artist, the image whose quoted caption text is longest."""
    import re

    out = []
    for a in TEXT_ARTISTS:
        best, best_len = None, 0
        for ci in discover_cached_pairs(str(cache_root / a)):
            if ci.npz_path in exclude:
                continue
            p = caption_path(ci)
            if not p.exists():
                continue
            cap = p.read_text(encoding="utf-8")
            if "japanese text" not in cap.lower():
                continue
            n = sum(len(s) for s in re.findall(r'"([^"]*)"', cap))
            if n > best_len:
                best, best_len = ci, n
        if best is not None:
            out.append(best)
    return out


def lat_patches(lat: torch.Tensor) -> torch.Tensor:
    """(1, C, h, w) latent → (h/2, w/2, 4C)."""
    _, c, h, w = lat.shape
    return (
        lat[0]
        .view(c, h // 2, 2, w // 2, 2)
        .permute(1, 3, 0, 2, 4)
        .reshape(h // 2, w // 2, 4 * c)
    )


def token_count(ci) -> int:
    w, h = (int(v) for v in ci.npz_path.rsplit("_", 2)[-2].split("x"))
    return (w // PATCH) * (h // PATCH)


# --------------------------------------------------------------------------- sheets


def densest_window(img: torch.Tensor, crop: int) -> tuple[int, int]:
    g = img.float().mean(0)
    e = (g.diff(dim=0).abs()[:, :-1] + g.diff(dim=1).abs()[:-1, :])[None, None]
    s = F.avg_pool2d(e, crop, stride=crop // 4)[0, 0]
    iy, ix = divmod(int(s.argmax()), s.shape[1])
    return ix * (crop // 4), iy * (crop // 4)


def sheet(cols, path: Path, crop: int = 384, at: tuple[int, int] | None = None):
    pils = [to_pil(t) for _, t in cols]
    w, h = pils[0].size
    cx, cy = at if at is not None else ((w - crop) // 2, (h - crop) // 2)
    crops = [
        p.crop((cx, cy, cx + crop, cy + crop)).resize(
            (2 * crop, 2 * crop), Image.NEAREST
        )
        for p in pils
    ]
    cw = max(w, 2 * crop)
    canvas = Image.new("RGB", (cw * len(cols), 24 + h + 2 * crop), "white")
    d = ImageDraw.Draw(canvas)
    for i, ((name, _), p, c) in enumerate(zip(cols, pils, crops)):
        d.text((i * cw + 6, 4), name, fill="black")
        canvas.paste(p, (i * cw, 24))
        d.rectangle(
            (i * cw + cx, 24 + cy, i * cw + cx + crop, 24 + cy + crop), outline="red"
        )
        canvas.paste(c, (i * cw, 24 + h))
    canvas.save(path)


# --------------------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(p, text_encoder=False)
    add_common_args(p)
    p.add_argument("--cache_root", default="post_image_dataset/lora")
    p.add_argument("--p1_run", default=P1_RUN)
    p.add_argument("--rollouts", type=int, default=3600)
    p.add_argument("--opt_steps", type=int, default=4)
    p.add_argument(
        "--accum",
        type=int,
        default=1,
        help="each update averages the gradient over the last N rollouts (images "
        "differ in shape, so this is accumulation, not a batch axis)",
    )
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--ls_init", type=float, default=1e-6, help="layer-scale γ init")
    p.add_argument(
        "--no_skip",
        action="store_true",
        help="drop the P1 linear skip; out keeps its default init (FastFrames' form)",
    )
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr_linear", type=float, default=2e-5)
    p.add_argument("--warmup", type=int, default=200, help="opt steps")
    p.add_argument("--eval_every", type=int, default=600, help="rollouts")
    p.add_argument("--infer_steps", type=int, default=50)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument("--n_sheets", type=int, default=12)
    args = p.parse_args()
    start_heartbeat(label="convnext")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed + 1)

    # --- data: P1's picker for the held-out set, everything else trains ---
    cache_root = Path(resolve_under_home(args.cache_root))
    p1 = json.loads((Path(resolve_under_home(args.p1_run)) / "result.json").read_text())
    p1_args = p1["args"]
    p1_train, test, test_artists = pick_samples(
        cache_root,
        p1_args["n_train"],
        p1_args["n_test_artists"],
        p1_args["per_artist"],
        p1_args["max_tokens"],
        p1_args["seed"],
    )
    assert test_artists == p1["metrics"]["test_artists"], (
        "P1 test set did not reproduce"
    )
    text = pick_text(cache_root, {ci.npz_path for ci in p1_train})
    held = set(test_artists)
    held_paths = {ci.npz_path for ci in text}
    train = [
        ci
        for a in sorted(d.name for d in cache_root.iterdir() if d.is_dir())
        if a not in held
        for ci in discover_cached_pairs(str(cache_root / a))
        if ci.npz_path not in held_paths
    ]
    seq_bands = cluster_token_bands(token_count(ci) for ci in train + test + text)
    _, sigmas = get_timesteps_sigmas(args.infer_steps, args.flow_shift, device)
    sigma_last = float(sigmas[-2])
    print(
        f"train {len(train)}  test {len(test)} ({', '.join(test_artists)})  "
        f"text {len(text)}  bands {seq_bands}  sigma_last {sigma_last:.4f}",
        flush=True,
    )

    # --- models: VAE (targets, stays on card) + frozen DiT ---
    vae = qwen_vae.load_vae(
        args.vae, device="cpu", dtype=torch.bfloat16, eval=True, vae_2d=True
    ).to(device)
    compile_blocks, args.compile = args.compile, False
    anima = build_anima(args, adapter=None, train_mode=False).anima
    args.compile = compile_blocks
    if compile_blocks:
        anima.compile_blocks(dynamic_seq=True, seq_bands=seq_bands)
    tap: dict[str, torch.Tensor] = {}
    anima.final_layer.linear.register_forward_pre_hook(
        lambda _m, inp: tap.__setitem__("x", inp[0].detach())
    )

    @torch.no_grad()
    def rollout(ci, idx: int):
        lat = load_cached_latents(ci.npz_path)[0].unsqueeze(0).to(device)
        tgt = vae.decode_to_pixels(lat)[0].float()
        cross, _ = load_cached_text_features(ci.te_path, variant=0)
        cross = cross.unsqueeze(0).to(device, torch.bfloat16)
        g = torch.Generator(device=device).manual_seed(args.seed * 100003 + idx)
        noise = torch.randn(lat.shape, generator=g, device=device, dtype=lat.dtype)
        x_t = (1 - sigma_last) * lat + sigma_last * noise
        sig = torch.tensor([sigma_last], device=device)
        pad = torch.zeros(1, 1, *lat.shape[-2:], dtype=torch.bfloat16, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            anima(x_t.unsqueeze(2).to(torch.bfloat16), sig, cross, padding_mask=pad)
        f = tap.pop("x")[0, 0]  # (gh, gw, D)
        return f.float(), lat_patches(x_t).float(), tgt

    # --- head ---
    D = anima.final_layer.hidden_size
    head = PixelHead(
        D, 64, args.width, args.depth, args.ls_init, use_skip=not args.no_skip
    ).to(device)
    w1 = load_file(
        str(Path(resolve_under_home(args.p1_run)) / "readout_linear.safetensors")
    )
    w1 = w1["weight"]  # (768, D+1), last column = bias
    with torch.no_grad():
        head.linear.weight.copy_(w1[:, :D])
        head.linear.bias.copy_(w1[:, D])
    n_params = sum(p.numel() for p in head.parameters())
    print(f"head {n_params / 1e6:.1f}M params", flush=True)
    lin_params = list(head.linear.parameters())
    rest = [p for n, p in head.named_parameters() if not n.startswith("linear.")]
    groups = [{"params": rest, "lr": args.lr, "weight_decay": 0.01}]
    if head.use_skip:
        groups.append({"params": lin_params, "lr": args.lr_linear, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.99))
    total = args.rollouts * args.opt_steps
    base_lrs = [g["lr"] for g in opt.param_groups]

    def set_lr(step):
        s = (
            min(1.0, (step + 1) / args.warmup)
            * 0.5
            * (1 + math.cos(math.pi * step / total))
        )
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * s

    # --- held-out features once (CPU), so evals cost no DiT forward ---
    t0 = time.perf_counter()
    held_feats = []
    for j, ci in enumerate(test + text):
        f, lat, tgt = rollout(ci, 10_000 + j)
        held_feats.append(
            (ci, f.to("cpu", torch.bfloat16), lat.cpu(), tgt.to("cpu", torch.float16))
        )
    print(
        f"held-out features {len(held_feats)} in {time.perf_counter() - t0:.0f}s",
        flush=True,
    )

    @torch.no_grad()
    def predict(f, lat, use_head=True):
        f, lat = f.to(device).float(), lat.to(device)
        if use_head:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = head(f, lat)
        else:
            gh, gw = f.shape[:2]
            pred = from_patches(head.linear(f).reshape(gh * gw, -1), gh, gw)
        return pred.float().clamp(-1, 1)

    def evaluate():
        head.eval()
        out = {"test": [], "text": []}
        for k, (ci, f, lat, tgt) in enumerate(held_feats):
            key = "test" if k < len(test) else "text"
            out[key].append(psnr(predict(f, lat), tgt.to(device)))
        head.train()
        return {k: float(np.mean(v)) for k, v in out.items()}

    p1_lin = [
        psnr(predict(f, lat, use_head=False), tgt.to(device))
        for _, f, lat, tgt in held_feats
    ]
    p1_scores = {
        "test": float(np.mean(p1_lin[: len(test)])),
        "text": float(np.mean(p1_lin[len(test) :])),
    }
    print(
        f"P1 linear: test {p1_scores['test']:.2f}  text {p1_scores['text']:.2f}",
        flush=True,
    )
    # P1 linear weights are about to train; keep a frozen copy for the sheets
    p1_linear = nn.Linear(D, PATCH * PATCH * 3).to(device)
    p1_linear.load_state_dict(head.linear.state_dict())

    # --- train ---
    history = []
    order: list = []
    step = 0
    t0 = time.perf_counter()
    loss_ema = mse_ema = None
    window: deque = deque(maxlen=args.accum)
    for r in range(args.rollouts):
        if not order:
            order = train[:]
            rng.shuffle(order)
        ci = order.pop()
        window.append(rollout(ci, r))
        for _ in range(args.opt_steps):
            set_lr(step)
            opt.zero_grad(set_to_none=True)
            # newest rollout last, so the logged loss / mse are this rollout's
            for f, lat, tgt in window:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = head(f, lat)
                loss, mse = rec_loss(pred, tgt)
                (loss / len(window)).backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            step += 1
        lv, mv = loss.item(), mse.item()
        loss_ema = lv if loss_ema is None else 0.98 * loss_ema + 0.02 * lv
        mse_ema = mv if mse_ema is None else 0.98 * mse_ema + 0.02 * mv
        if (r + 1) % 50 == 0:
            el = time.perf_counter() - t0
            print(
                f"  rollout {r + 1}/{args.rollouts}  loss {loss_ema:.4f}  "
                f"psnr~{10 * math.log10(4 / mse_ema):.2f}  "
                f"{el / (r + 1):.2f}s/rollout  eta {(args.rollouts - r - 1) * el / (r + 1) / 60:.0f}m",
                flush=True,
            )
        if (r + 1) % args.eval_every == 0 or r + 1 == args.rollouts:
            ev = evaluate()
            history.append({"rollout": r + 1, "step": step, **ev})
            print(
                f"  eval @ {r + 1}: test {ev['test']:.2f}  text {ev['text']:.2f}",
                flush=True,
            )

    # --- read ---
    run_dir = make_run_dir("pixel_readout", label=args.label)
    rows, artifacts = [], []
    head.eval()
    text_k = 0
    for k, (ci, f, lat, tgt) in enumerate(held_feats):
        tgt = tgt.to(device).float()
        pred = predict(f, lat)
        with torch.no_grad():
            fd = f.to(device).float()
            gh, gw = fd.shape[:2]
            lin = from_patches(
                p1_linear(fd).reshape(gh * gw, -1).float(), gh, gw
            ).clamp(-1, 1)
        is_text = k >= len(test)
        rows.append(
            {
                "stem": ci.stem,
                "artist": Path(ci.npz_path).parent.name,
                "set": "text" if is_text else "test",
                "psnr_p1_linear": psnr(lin, tgt),
                "psnr_head": psnr(pred, tgt),
            }
        )
        cols = [("VAE decode", tgt), ("P1 linear", lin), ("P2 ConvNeXt head", pred)]
        if is_text:
            name = f"text_{text_k:02d}_{ci.stem}.png"
            text_k += 1
            sheet(cols, run_dir / name, at=densest_window(tgt, 384))
        elif k < args.n_sheets:
            name = f"sheet_{k:02d}_{ci.stem}.png"
            sheet(cols, run_dir / name)
        else:
            continue
        artifacts.append(name)

    save_file(
        {k: v.detach().contiguous().cpu() for k, v in head.state_dict().items()},
        str(run_dir / "head.safetensors"),
        metadata={
            "width": str(args.width),
            "depth": str(args.depth),
            "ls_init": str(args.ls_init),
            "skip": str(int(head.use_skip)),
            "sigma": f"{sigma_last}",
        },
    )
    artifacts.append("head.safetensors")

    def mean(s, key):
        return float(np.mean([r[key] for r in rows if r["set"] == s]))

    metrics = {
        "sigma_last": sigma_last,
        "n_train": len(train),
        "n_test": len(test),
        "n_text": len(text),
        "test_artists": test_artists,
        "text_images": [f"{Path(ci.npz_path).parent.name}/{ci.stem}" for ci in text],
        "head_params_M": n_params / 1e6,
        "opt_steps_total": step,
        "sec_per_rollout": (time.perf_counter() - t0) / args.rollouts,
        "test_psnr_p1_linear": mean("test", "psnr_p1_linear"),
        "test_psnr_head": mean("test", "psnr_head"),
        "text_psnr_p1_linear": mean("text", "psnr_p1_linear"),
        "text_psnr_head": mean("text", "psnr_head"),
        "eval_history": history,
        "per_image": rows,
    }
    print(
        f"test PSNR vs VAE: P1 {metrics['test_psnr_p1_linear']:.2f} → head "
        f"{metrics['test_psnr_head']:.2f} | text: P1 {metrics['text_psnr_p1_linear']:.2f} → "
        f"head {metrics['text_psnr_head']:.2f}",
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
