#!/usr/bin/env python3
"""×2 pixel head: decode and 2× upscale in one readout (pixel_decoder P5, arm A).

A released head (``PixelHead`` with one upsample stage, 8-px spacing at width 512) gets a
second stage: LayerNorm + per-position linear 512 → 4 × 256, pixel shuffle, ConvNeXt blocks
at 4-px spacing (width 256), and a linear that writes an 8×8 RGB patch per position, i.e.
32×32 output pixels per DiT token. The ×1 output of the base head is kept and trained
beside it, so one file decodes at ×1 and ×2.

Target (arm A, teacher distillation): the released 1-step ResShift ×2 student
(``ComfyUI-Distilled-ResShift``, its ``upscale`` at the node's defaults) applied to
VAE(x̂0), x̂0 = x_t − σ·v from the same DiT forward. The ×1 output keeps its recipe target,
VAE(x̂0). Both use the recipe loss (0.5 L1 + 5 MSE + 0.05 L_grad).

Training is windowed: the DiT forward and the VAE decode run on the whole image; one
random window of ``--window`` + 2 × ``--margin`` tokens per rollout goes to the teacher and
into the replay buffer, and the loss leaves ``--margin`` tokens out on every window side
that is not an image edge (the head's receptive field is 3 tokens; the same strip is the
teacher's context). Evals run the head and the teacher on whole images.

Usage::

    make daemon-run ARGS="--queue --stall-timeout 900 bench/pixel_readout/x2_head.py \
        --compile --init mixr_1119_1455 --adapter models/loras/anima_turbo_4step.safetensors \
        --points 0.0577-0.1364 0.5:adapter --pages post_image_dataset/pixel_pages \
        --page_share 0.33 --label x2a_mmdd_hhmm"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from bench._anima import (
    DEFAULT_TEXT_ENCODER,
    add_common_args,
    add_model_args,
    build_anima,
)
from bench._common import make_run_dir, start_heartbeat, write_result
from bench.pixel_readout._compile import compile_blocks_isolated
from bench.pixel_readout.convnext_head import (
    P1_RUN,
    attach_adapter,
    densest_window,
    page_token_count,
    pick_text,
    real_pixels,
    rec_loss,
    resolve_run,
    token_count,
)
from bench.pixel_readout.linear_probe import PATCH, pick_samples, psnr, to_pil
from bench.pixel_readout.x2_dmd import X2_TEACHER, X2Critic
from library.anima.pixel_head import (
    PixelHeadX2,
    lat_patches,
    load_pixel_head,
)
from library.datasets.buckets import cluster_token_bands
from library.env import resolve_under_home
from library.inference.sampling import get_timesteps_sigmas
from library.inference.text import prepare_text_inputs
from library.io.cache import (
    discover_cached_pairs,
    load_cached_latents,
    load_cached_text_features,
)
from library.models import qwen_vae

TEACHER_REPO = "../comfy/custom_nodes/comfyui-distilled-resshift"
TEACHER_DIR = "../comfy/models/resshift"
TEACHER_FILE = "rsd_student_18k.safetensors"  # Hub sorryhyun/Distilled-ResShift-2x
VQGAN_FILE = "autoencoder_vq_f4.pth"


# --------------------------------------------------------------------------- teacher


def load_teacher(repo: Path, weights: Path, device):
    """The released ResShift ×2 student as ``up2(img (3,H,W) in [-1,1], seed) → (3,2H,2W)``.

    Loaded the way the ComfyUI node loads it (student fp32 under bf16 autocast, VQGAN
    fp16) and run through the node's ``upscale`` at its defaults (chop 512, overlap 64,
    tile batch 4, shared noise)."""
    spec = importlib.util.spec_from_file_location(
        "resshift_infer", str(repo / "resshift_infer.py")
    )
    R = importlib.util.module_from_spec(spec)
    sys.modules["resshift_infer"] = R
    spec.loader.exec_module(R)
    cfg = R.load_configs(str(weights / VQGAN_FILE), config_path=R.CONFIGS["x2"])
    diff = R.build_diffusion(cfg)
    path = str(weights / TEACHER_FILE)
    with safe_open(path, "pt") as fh:
        meta = fh.metadata() or {}
    nc = meta.get("noise_channels")
    student = R.build_student(
        cfg,
        device,
        noise_mode=meta.get("noise_mode", "concat"),
        noise_channels=int(nc) if nc is not None else None,
    )
    R.load_student_weights(student, load_file(path))
    vqgan = R.build_autoencoder(cfg, device, dtype=torch.float16)
    align = R.swin_align(cfg)
    cond = meta.get("cond_lq", "latent")
    scale = int(cfg.diffusion.params.sf)
    assert scale == 2, f"teacher scale {scale}"

    @torch.no_grad()
    def up2(img: torch.Tensor, seed: int) -> torch.Tensor:
        return R.upscale(
            cfg,
            diff,
            vqgan,
            student,
            img[None].float().clamp(-1, 1),
            device,
            scale,
            align,
            overlap=64,
            chop=512,
            tile_batch=4,
            amp=True,
            seed=seed,
            shared_noise=True,
            cond_mode=cond,
        )[0].float()

    # the pieces a critic on the same VQ-f4 latent builds on (x2_dmd.X2Critic)
    up2.resshift = (R, cfg, diff, cond)
    return up2


# --------------------------------------------------------------------------- helpers


def crop_margin(img: torch.Tensor, mar: tuple[int, int, int, int], px: int):
    """Drop ``mar`` = (top, bottom, left, right) tokens, ``px`` pixels per token."""
    t, b, lf, r = (k * px for k in mar)
    h, w = img.shape[-2:]
    return img[..., t : h - b, lf : w - r]


def bicubic2(img: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        img[None].float(), scale_factor=2, mode="bicubic", align_corners=False
    )[0].clamp(-1, 1)


def down2(img: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(img[None].float(), 2)[0]


def dc_loss(a: torch.Tensor, b: torch.Tensor, k: int) -> torch.Tensor:
    """L1 on ``k``-px block means (the SR line's tone term,
    ``project/finished/sr/distill_rsd/train.py``); both in [-1, 1]."""
    return F.l1_loss(F.avg_pool2d(a[None], k), F.avg_pool2d(b[None], k))


def luma8(img: torch.Tensor) -> torch.Tensor:
    """(3, H, W) in [-1, 1] → Rec.601 luma in 8-bit levels."""
    w = img.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
    return ((img.float() + 1) * 127.5 * w).sum(0)


def tone_stats(pred: torch.Tensor, tgt: torch.Tensor) -> tuple[float, float | None]:
    """(whole-image luma, luma over the pixels ``tgt`` has under 64), ``pred`` −
    ``tgt`` in 8-bit levels: the dark-pixel lift read in ``x2_plan.md``."""
    lp, lt = luma8(pred), luma8(tgt)
    d = lp - lt
    dark = lt < 64
    return d.mean().item(), (d[dark].mean().item() if dark.any() else None)


def mean_or_none(v: list) -> float | None:
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def pair_sheet(cols, path: Path, at: tuple[int, int, int] | None = None):
    """Two labelled columns at native pixels; ``at`` = (x, y, size) crops both."""
    pils = [to_pil(t) for _, t in cols]
    if at is not None:
        x, y, s = at
        pils = [p.crop((x, y, x + s, y + s)) for p in pils]
    w, h = pils[0].size
    canvas = Image.new("RGB", (w * len(cols), 24 + h), "white")
    d = ImageDraw.Draw(canvas)
    for i, ((name, _), p) in enumerate(zip(cols, pils)):
        d.text((i * w + 6, 4), name, fill="black")
        canvas.paste(p, (i * w, 24))
    canvas.save(path)


# --------------------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(p, text_encoder=False)
    add_common_args(p)
    p.add_argument("--cache_root", default="post_image_dataset/lora")
    p.add_argument("--p1_run", default=P1_RUN, help="held-out picker's args")
    p.add_argument("--init", required=True, help="base head: <run>/head.safetensors")
    p.add_argument("--depth2", type=int, default=2, help="blocks at 4-px spacing")
    p.add_argument("--ls_init", type=float, default=1.0)
    p.add_argument("--rollouts", type=int, default=1000)
    p.add_argument("--opt_steps", type=int, default=16)
    p.add_argument("--replay", type=int, default=64)
    p.add_argument("--window", type=int, default=32, help="loss window, tokens")
    p.add_argument("--margin", type=int, default=3, help="context strip, tokens")
    p.add_argument("--x1_weight", type=float, default=1.0, help="×1 output's loss")
    p.add_argument(
        "--lambda_lpips",
        type=float,
        default=0.0,
        help="VGG-LPIPS on the ×2 output against the target (the SR line's 2.0 was "
        "beside a DMD loss, not this recipe loss)",
    )
    p.add_argument(
        "--lambda_dc",
        type=float,
        default=0.0,
        help="tone term (dc_loss) on the ×2 output",
    )
    p.add_argument("--dc_block", type=int, default=32, help="dc_loss block, ×2 px")
    p.add_argument(
        "--lambda_lpips_x1",
        type=float,
        default=0.0,
        help="VGG-LPIPS on the ×1 output at the :real points (convnext_head.py's "
        "--lambda_lpips, the base head's own recipe)",
    )
    p.add_argument(
        "--lpips_crop",
        type=int,
        default=0,
        help="LPIPS on one random CxC crop of the loss window per step (0 = the whole "
        "window; lpips_speed.py: 768² 19.6 ms, 512² 9.0 ms compiled)",
    )
    p.add_argument(
        "--lambda_dmd",
        type=float,
        default=0.0,
        help="RSD's distribution-matching term on a --critic_crop crop of the ×2 "
        "output, in the teacher's VQ-f4 latent (x2_dmd.X2Critic)",
    )
    p.add_argument(
        "--lambda_gan",
        type=float,
        default=0.0,
        help="GAN term on the fake critic's bottleneck (the SR line's 3e-3 sat beside "
        "its own DMD and LPIPS weights)",
    )
    p.add_argument(
        "--critic",
        action="store_true",
        help="build and train the critic even at --lambda_dmd 0 --lambda_gan 0 (with "
        "--term_log: size the two weights)",
    )
    p.add_argument(
        "--critic_crop", type=int, default=256, help="×2 px, multiple of 256"
    )
    p.add_argument(
        "--critic_k", type=int, default=1, help="critic updates per head update"
    )
    p.add_argument("--critic_lr", type=float, default=5e-5)
    p.add_argument(
        "--critic_gan_d",
        type=float,
        default=3e-3,
        help="weight of the discriminator loss inside the critic's update",
    )
    p.add_argument("--critic_teacher", default=X2_TEACHER)
    p.add_argument(
        "--term_log",
        action="store_true",
        help="compute LPIPS and dc_loss in the graph even at weight 0 and log every "
        "term's loss and gradient norm every 50 rollouts (to size the weights)",
    )
    p.add_argument(
        "--resume_x2",
        default=None,
        help="<run> whose head_x2.safetensors is loaded after the held-out build "
        "(with --rollouts 0: read a trained ×2 head on this script's meters)",
    )
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=200, help="opt steps")
    p.add_argument("--eval_every", type=int, default=250, help="rollouts")
    p.add_argument("--n_test", type=int, default=12, help="test images read (of 30)")
    p.add_argument("--n_manga", type=int, default=4)
    p.add_argument("--n_sheets", type=int, default=4, help="per point and set")
    p.add_argument("--crop", type=int, default=768, help="zoom sheet side, ×2 pixels")
    p.add_argument("--pages", default=None)
    p.add_argument("--page_share", type=float, default=0.0)
    p.add_argument(
        "--page_caption",
        default="monochrome, greyscale, comic, speech bubble, japanese text",
    )
    p.add_argument("--stream", choices=["bf16", "fp32"], default="fp32")
    p.add_argument("--text_encoder", default=DEFAULT_TEXT_ENCODER)
    p.add_argument("--infer_steps", type=int, default=50)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument("--adapter", default=None)
    p.add_argument(
        "--points",
        nargs="+",
        default=None,
        metavar="SIGMA[-SIGMA][:adapter][:real]",
        help="decode points, in turn per rollout (convnext_head.py's form); a LO-HI "
        "range trains on uniform σ and is read at its midpoint. :real trains that "
        "point's ×1 output against the original image (with --lambda_lpips_x1), as the "
        "base head was; the ×2 target stays the teacher on VAE(x̂0)",
    )
    p.add_argument(
        "--chain",
        action="store_true",
        help="read the chain too: the teacher on the --init head's ×1 output (as "
        "loaded, before training) against the teacher on VAE(x̂0); the sheets become "
        "chain | ×2 head",
    )
    p.add_argument("--teacher_repo", default=TEACHER_REPO)
    p.add_argument("--teacher_dir", default=TEACHER_DIR)
    args = p.parse_args()
    start_heartbeat(label="x2head")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed + 1)

    # --- data: the parent recipe's split (convnext_head.py) ---
    cache_root = Path(resolve_under_home(args.cache_root))
    p1 = json.loads((Path(resolve_under_home(args.p1_run)) / "result.json").read_text())
    a1 = p1["args"]
    p1_train, test, test_artists = pick_samples(
        cache_root,
        a1["n_train"],
        a1["n_test_artists"],
        a1["per_artist"],
        a1["max_tokens"],
        a1["seed"],
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
    page_train: list[Path] = []
    page_eval: list[Path] = []
    if args.pages:
        proot = Path(resolve_under_home(args.pages))
        page_train = sorted((proot / "train").glob("*.safetensors"))
        page_eval = sorted((proot / "eval").glob("*.safetensors"))
    seq_bands = cluster_token_bands(
        [token_count(ci) for ci in train + test + text]
        + [page_token_count(pth) for pth in page_train + page_eval]
    )
    _, sigmas = get_timesteps_sigmas(args.infer_steps, args.flow_shift, device)
    sigma_last = float(sigmas[-2])
    train_points: list[tuple[float, float, bool, bool]] = [
        (sigma_last, sigma_last, False, False)
    ]
    points: list[tuple[str, float, bool]] = [
        (f"base_s{sigma_last:.4g}", sigma_last, False)
    ]
    if args.points:
        train_points, points = [], []
        for spec in args.points:
            s, *tags = spec.split(":")
            assert set(tags) <= {"adapter", "real"}, (
                f"--points {spec}: SIGMA[:adapter][:real]"
            )
            tag = "adapter" in tags
            assert not tag or args.adapter, f"--points {spec} needs --adapter"
            lo, _, hi = s.partition("-")
            lo, hi = float(lo), float(hi or lo)
            train_points.append((lo, hi, tag, "real" in tags))
            mid = (lo + hi) / 2
            points.append((f"{'lora' if tag else 'base'}_s{mid:.4g}", mid, bool(tag)))
    sig_rng = random.Random(args.seed + 4)
    print(
        f"train {len(train)}  test {len(test)} (read {args.n_test})  text {len(text)}  "
        f"pages {len(page_train)} / eval {len(page_eval)}  bands {seq_bands}  "
        f"points {', '.join(n for n, _, _ in points)}",
        flush=True,
    )

    # --- models: VAE + frozen DiT + teacher ---
    vae = qwen_vae.load_vae(
        args.vae, device="cpu", dtype=torch.bfloat16, eval=True, vae_2d=True
    ).to(device)
    compile_blocks, args.compile = args.compile, False
    anima = build_anima(args, adapter=None, train_mode=False).anima
    args.compile = compile_blocks
    if args.stream == "fp32":
        from library.anima.stream import stream_fp32_

        stream_fp32_(anima)
    network = attach_adapter(anima, args.adapter, device) if args.adapter else None
    if compile_blocks:
        compile_blocks_isolated(anima, seq_bands)
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
    if args.pages:
        arg_c, _ = prepare_text_inputs(
            device=device,
            anima=anima,
            prompt=args.page_caption,
            text_encoder_path=str(resolve_under_home(args.text_encoder)),
        )
        page_cross = arg_c["embed"][0].to(device, torch.bfloat16)

    def load_item(item):
        if isinstance(item, Path):  # a build_pages.py page
            lat = load_file(str(item))["latent"].float()[None].to(device)
            return lat, page_cross
        lat = load_cached_latents(item.npz_path)[0].unsqueeze(0).to(device)
        cross, _ = load_cached_text_features(item.te_path, variant=0)
        return lat, cross.unsqueeze(0).to(device, torch.bfloat16)

    @torch.no_grad()
    def rollout(item, idx: int, sigma: float, adapter_on: bool):
        """→ (features (gh, gw, D), x_t patches, VAE(x̂0) (3, H, W)) at one decode point."""
        if network is not None:
            network.set_enabled(adapter_on)
        lat, cross = load_item(item)
        g = torch.Generator(device=device).manual_seed(args.seed * 100003 + idx)
        noise = torch.randn(lat.shape, generator=g, device=device, dtype=lat.dtype)
        x_t = (1 - sigma) * lat + sigma * noise
        sig = torch.tensor([sigma], device=device)
        pad = torch.zeros(1, 1, *lat.shape[-2:], dtype=torch.bfloat16, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = anima(x_t.unsqueeze(2).to(torch.bfloat16), sig, cross, padding_mask=pad)
        f = tap.pop("x")[0, 0]
        x0_hat = x_t - sigma * v.float().squeeze(2)
        img = vae.decode_to_pixels(x0_hat)[0].float().clamp(-1, 1)
        return f.float(), lat_patches(x_t).float(), img

    # --- head ---
    D = anima.final_layer.hidden_size
    init_path = resolve_run(args.init) / "head.safetensors"
    base = load_pixel_head(init_path, device, D).train()
    head = PixelHeadX2(base, args.depth2, args.ls_init).to(device)
    n_params = sum(q.numel() for q in head.parameters())
    n_new = n_params - sum(q.numel() for q in base.parameters())
    print(
        f"head {n_params / 1e6:.1f}M params ({n_new / 1e6:.1f}M new), base {init_path}",
        flush=True,
    )
    head_train = torch.compile(head, dynamic=True) if compile_blocks else head
    opt = torch.optim.AdamW(
        head.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01
    )
    total_steps = args.rollouts * args.opt_steps

    def set_lr(step):
        s = (
            min(1.0, (step + 1) / args.warmup)
            * 0.5
            * (1 + math.cos(math.pi * step / total_steps))
        )
        for g in opt.param_groups:
            g["lr"] = args.lr * s

    # --- held-out: whole-image features, VAE(x̂0), and the teacher's ×2 of it ---
    t0 = time.perf_counter()
    held_items = (
        [(ci, "test") for ci in test[: args.n_test]]
        + [(ci, "text") for ci in text]
        + [(pth, "manga") for pth in page_eval[: args.n_manga]]
    )
    import lpips

    lp = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    for q in lp.parameters():
        q.requires_grad_(False)
    # channels_last + compile, pred and target as two inputs: 40.3 → 19.6 ms per
    # 768² step (bench/pixel_readout/lpips_speed.py)
    lp.to(memory_format=torch.channels_last)
    lp_train = torch.compile(lp) if compile_blocks else lp

    @torch.no_grad()
    def lpips_img(a: torch.Tensor, b: torch.Tensor, tile: int = 768) -> float:
        """Whole-image LPIPS as the area-weighted mean over ``tile``-px tiles (a
        2560 × 1664 image in one VGG pass runs out of memory beside training)."""
        h, w = a.shape[-2:]
        tot = 0.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for y in range(0, h, tile):
                for x in range(0, w, tile):
                    ta = a[None, :, y : y + tile, x : x + tile].float()
                    tb = b[None, :, y : y + tile, x : x + tile].float()
                    tot += lp(ta, tb).mean().item() * ta.shape[-2] * ta.shape[-1]
        return tot / (h * w)

    # (item, set key, f, lat, VAE(x̂0), teacher ×2, bicubic dB, twin dB, chain | None)
    held_feats = []
    chain_rows = []  # (set key, dB, LPIPS, luma, dark luma) vs the teacher
    t_teacher = []
    head.eval()
    for name, sigma, adapter_on in points:  # the same noise per item at every point
        for j, (item, key) in enumerate(held_items):
            f, lat, img = rollout(item, 10_000 + j, sigma, adapter_on)
            chain = None
            if args.chain:  # the teacher on the --init head's ×1 output, as loaded
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    x1, _ = head(f, lat)
                chain = up2(x1.float().clamp(-1, 1), 0)
            torch.cuda.synchronize()
            tt = time.perf_counter()
            t2 = up2(img, 0)
            torch.cuda.synchronize()
            t_teacher.append(time.perf_counter() - tt)
            # two draws of the teacher on one input: how far the target is from a
            # function of the input (the ceiling a regression can reach)
            first = j == 0 or held_items[j - 1][1] != key  # each set's first image
            twin = psnr(up2(img, 1), t2) if first or j % 4 == 0 else None
            held_feats.append(
                (
                    item,
                    f"{name}.{key}",
                    f.to("cpu", torch.bfloat16),
                    lat.cpu(),
                    img.to("cpu", torch.float16),
                    t2.to("cpu", torch.float16),
                    psnr(bicubic2(img), t2),
                    twin,
                    None if chain is None else chain.to("cpu", torch.float16),
                )
            )
            if chain is not None:  # chain meters, once
                lum, dark = tone_stats(chain, t2)
                chain_rows.append(
                    (
                        f"{name}.{key}",
                        psnr(chain, t2),
                        lpips_img(chain, t2),
                        lum,
                        dark,
                    )
                )
    head.train()
    sets = list(dict.fromkeys(h[1] for h in held_feats))
    refs = {}
    for st in sets:
        rows_ = [h for h in held_feats if h[1] == st]
        refs[f"{st}_psnr_bicubic"] = float(np.mean([h[6] for h in rows_]))
        refs[f"{st}_psnr_teacher_twin"] = float(
            np.mean([h[7] for h in rows_ if h[7] is not None])
        )
        cr = [c for c in chain_rows if c[0] == st]
        if cr:
            for i, col in enumerate(("psnr", "lpips", "luma", "dark"), 1):
                refs[f"{st}_chain_{col}"] = mean_or_none([c[i] for c in cr])
    if args.resume_x2:
        rx = resolve_run(args.resume_x2) / "head_x2.safetensors"
        head.load_state_dict(load_file(str(rx), device=str(device)))
        print(f"loaded ×2 head {rx}", flush=True)
    refs["teacher_sec_per_image"] = float(np.mean(t_teacher))
    print(
        f"held-out {len(held_feats)} in {time.perf_counter() - t0:.0f}s, teacher "
        f"{refs['teacher_sec_per_image']:.2f} s / image",
        flush=True,
    )
    print(
        "vs teacher ×2 — bicubic / second teacher draw: "
        + "  ".join(
            f"{st} {refs[f'{st}_psnr_bicubic']:.2f} / {refs[f'{st}_psnr_teacher_twin']:.2f}"
            for st in sets
        ),
        flush=True,
    )

    def okey(item) -> str:
        return str(item) if isinstance(item, Path) else item.npz_path

    # the ×1 output is also read against the image each latent was encoded from
    origs = {
        okey(it): (real_pixels(it).float() / 127.5 - 1).to(torch.float16)
        for it, _ in held_items
    }

    @torch.no_grad()
    def predict(f, lat):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x1, x2 = head(f.to(device).float(), lat.to(device))
        return x1.float().clamp(-1, 1), x2.float().clamp(-1, 1)

    def evaluate():
        """Per set: ×2 vs the teacher (dB, LPIPS, luma, dark-pixel luma), ×1 vs
        VAE(x̂0), 2×2-mean of ×2 vs VAE(x̂0)."""
        head.eval()
        out: dict[str, list] = {}
        for item, key, f, lat, img, t2, *_ in held_feats:
            x1, x2 = predict(f, lat)
            img, t2 = img.to(device).float(), t2.to(device).float()
            o = origs[okey(item)].to(device).float()
            out.setdefault(f"{key}_x1_orig", []).append(psnr(x1, o))
            out.setdefault(f"{key}_x1_lpips_orig", []).append(lpips_img(x1, o))
            lum, dark = tone_stats(x2, t2)
            out.setdefault(f"{key}_x2", []).append(psnr(x2, t2))
            out.setdefault(f"{key}_x2_lpips", []).append(lpips_img(x2, t2))
            out.setdefault(f"{key}_x2_luma", []).append(lum)
            out.setdefault(f"{key}_x2_dark", []).append(dark)
            out.setdefault(f"{key}_x1", []).append(psnr(x1, img))
            out.setdefault(f"{key}_down", []).append(psnr(down2(x2), img))
        head.train()
        return {k: mean_or_none(v) for k, v in out.items()}

    def fmt(ev):
        return "  ".join(
            f"{k} ×2 {ev[k + '_x2']:.2f} lpips {ev[k + '_x2_lpips']:.4f} "
            f"dark {ev[k + '_x2_dark'] or 0:+.2f} ×1 {ev[k + '_x1']:.2f} "
            f"(orig {ev[k + '_x1_orig']:.2f} lpips {ev[k + '_x1_lpips_orig']:.4f}) "
            f"down {ev[k + '_down']:.2f}"
            for k in sets
        )

    ev0 = evaluate()
    history = [{"rollout": 0, "step": 0, **ev0}]
    print(f"  eval @ 0 (×1 = the base head as loaded): {fmt(ev0)}", flush=True)

    # --- critic (arm 3): RSD's DMD + GAN terms on a crop of the ×2 output ---
    critic = None
    if args.lambda_dmd > 0 or args.lambda_gan > 0 or args.critic:
        assert args.critic_crop % 256 == 0, "--critic_crop: a multiple of 256"
        R, rcfg, rdiff, rcond = up2.resshift
        assert rcond == "latent", f"critic conditioning {rcond}"
        critic = X2Critic(
            R,
            rcfg,
            rdiff,
            Path(resolve_under_home(args.critic_teacher)),
            device,
            lr=args.critic_lr,
            lambda_gan_d=args.critic_gan_d,
            seed=args.seed + 7,
            use_dmd=args.lambda_dmd > 0 or args.critic,
            use_gan=args.lambda_gan > 0 or args.critic,
        )
    peak_setup = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    print(f"peak VRAM before training {peak_setup:.2f} GB", flush=True)

    # --- train ---
    use_lpips = args.lambda_lpips > 0 or args.term_log
    use_dc = args.lambda_dc > 0 or args.term_log
    weights = {"rec2": 1.0, "rec1": args.x1_weight}
    if use_lpips:
        weights["lpips"] = args.lambda_lpips
    if use_dc:
        weights["dc"] = args.lambda_dc
    if args.lambda_lpips_x1 > 0:
        weights["lpips1"] = args.lambda_lpips_x1
    if critic is not None:  # a term at weight 0 is in the graph only under --critic
        if critic.use_dmd:
            weights["dmd"] = args.lambda_dmd
        if critic.use_gan:
            weights["gan"] = args.lambda_gan
    crit_rng = random.Random(args.seed + 8)

    def critic_crops(p2, blq, t2_):
        """One random --critic_crop crop of the loss region → (×2 output, bicubic ×2
        of VAE(x̂0), teacher ×2), each (1, 3, c, c). The bicubic is the teacher's own
        upsample (``resshift_infer.upscale``), taken on the region before the crop."""
        c = args.critic_crop
        h, w = p2.shape[-2:]
        assert min(h, w) >= c, (h, w, c)
        y, x = crit_rng.randint(0, h - c), crit_rng.randint(0, w - c)
        up = F.interpolate(
            blq[None].float(), scale_factor=2, mode="bicubic", align_corners=False
        )
        sl = (..., slice(y, y + c), slice(x, x + c))
        return p2[None].float()[sl], up[sl], t2_[None][sl]

    def loss_terms(
        bf, blat, b1, b2, bmar, breal=False, blq=None, model=None, lp_fn=None
    ):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x1, x2 = (model or head_train)(bf, blat)
        p2, t2_ = crop_margin(x2, bmar, 2 * PATCH), b2.float()
        l2, mse2 = rec_loss(p2, t2_)
        l1, mse1 = rec_loss(crop_margin(x1, bmar, PATCH), b1.float())
        out = {"rec2": l2, "rec1": l1, "mse2": mse2.detach(), "mse1": mse1.detach()}
        if use_lpips:
            a, b = p2[None], t2_[None]
            c = args.lpips_crop
            if c and min(a.shape[-2:]) > c:
                y = lp_rng.randint(0, a.shape[-2] - c)
                x = lp_rng.randint(0, a.shape[-1] - c)
                a, b = a[..., y : y + c, x : x + c], b[..., y : y + c, x : x + c]
            cl = torch.channels_last
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out["lpips"] = (
                    (lp_fn or lp_train)(
                        a.contiguous(memory_format=cl), b.contiguous(memory_format=cl)
                    )
                    .mean()
                    .float()
                )
        if use_dc:
            out["dc"] = dc_loss(p2, t2_, args.dc_block)
        if critic is not None:
            dmd, gan = critic.gen_terms(*critic_crops(p2, blq, t2_))
            out.update({k: v for k, v in (("dmd", dmd), ("gan", gan)) if v is not None})
        if breal and args.lambda_lpips_x1 > 0:
            cl = torch.channels_last
            a = crop_margin(x1, bmar, PATCH)[None].float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out["lpips1"] = (
                    (lp_fn or lp_train)(
                        a.contiguous(memory_format=cl),
                        b1.float()[None].contiguous(memory_format=cl),
                    )
                    .mean()
                    .float()
                )
        return out

    def grad_norms(sample) -> dict[str, float]:
        """Unweighted gradient norm of each loss term over all head parameters
        (eager: the compiled backward donates buffers, so no retain_graph)."""
        params = [q for q in head.parameters() if q.requires_grad]
        terms = loss_terms(*sample, model=head, lp_fn=lp)
        keys = [k for k in weights if k in terms]
        out = {}
        for i, k in enumerate(keys):
            g = torch.autograd.grad(
                terms[k], params, retain_graph=i + 1 < len(keys), allow_unused=True
            )
            out[f"g_{k}"] = torch.sqrt(
                sum((x.float() ** 2).sum() for x in g if x is not None)
            ).item()
        return out

    grad_log: list[dict] = []
    lp_rng = random.Random(args.seed + 6)
    order: list = []
    page_order: list = []
    page_rng = random.Random(args.seed + 2)
    win_rng = random.Random(args.seed + 5)
    replay: deque = deque(maxlen=args.replay)
    replay_rng = random.Random(args.seed + 3)
    n_page_rollouts = 0
    step = 0
    wt = args.window + 2 * args.margin
    ema: dict[str, float] = {}
    t_critic = 0.0
    peak_steps, evaluated = None, False
    crit_keys = ("dmd", "gan", "c_fake", "c_gan_d", "d_real", "d_fake")
    t0 = time.perf_counter()
    for r in range(args.rollouts):
        if page_train and page_rng.random() < args.page_share:
            if not page_order:
                page_order = page_train[:]
                page_rng.shuffle(page_order)
            item = page_order.pop()
            n_page_rollouts += 1
        else:
            if not order:
                order = train[:]
                rng.shuffle(order)
            item = order.pop()
        lo, hi, adapter_on, real = train_points[r % len(train_points)]
        sg = lo if hi == lo else sig_rng.uniform(lo, hi)
        f, lat, img = rollout(item, r, sg, adapter_on)
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
        ys, xs = (
            slice(y0 * PATCH, (y0 + wh) * PATCH),
            slice(x0 * PATCH, (x0 + ww) * PATCH),
        )
        win = img[:, ys, xs]
        t2 = up2(win, r)
        win1 = win  # the ×1 target: VAE(x̂0), or the original image at a :real point
        if real:
            px = real_pixels(item)
            assert px.shape[-2:] == img.shape[-2:], (item, px.shape, img.shape)
            win1 = px[:, ys, xs].to(device).float() / 127.5 - 1
        replay.append(
            (
                f[y0 : y0 + wh, x0 : x0 + ww].to(torch.bfloat16).clone(),
                lat[y0 : y0 + wh, x0 : x0 + ww].clone(),
                crop_margin(win1, mar, PATCH).half().clone(),
                crop_margin(t2, mar, 2 * PATCH).half(),
                mar,
                real,
                crop_margin(win, mar, PATCH).half().clone(),  # VAE(x̂0): the critic's LQ
            )
        )
        for _ in range(args.opt_steps):
            set_lr(step)
            opt.zero_grad(set_to_none=True)
            terms = loss_terms(*replay_rng.choice(replay))
            loss = sum(w * terms[k] for k, w in weights.items() if k in terms)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            step += 1
            if critic is not None:
                if args.term_log:
                    torch.cuda.synchronize()
                tc = time.perf_counter()
                clog = critic.step()  # on this step's ×2 crop, detached
                for _ in range(args.critic_k - 1):
                    cf, clat, _, cb2, cmar, _, clq = crit_rng.choice(replay)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        _, cx2 = head(cf.float(), clat)
                    clog = critic.update(
                        *critic_crops(
                            crop_margin(cx2, cmar, 2 * PATCH), clq, cb2.float()
                        )
                    )
                if args.term_log:
                    torch.cuda.synchronize()
                t_critic += time.perf_counter() - tc
        for k, v in terms.items():
            v = v.item()
            ema[k] = v if k not in ema else 0.98 * ema[k] + 0.02 * v
        if critic is not None:
            for k, v in clog.items():
                ema[k] = v if k not in ema else 0.98 * ema[k] + 0.02 * v
        if (r + 1) % 50 == 0:
            el = time.perf_counter() - t0
            extra = ""
            if use_lpips or use_dc or "lpips1" in ema:
                extra = "  " + " ".join(
                    f"{k} {ema[k]:.4f}" for k in ("lpips", "dc", "lpips1") if k in ema
                )
            if critic is not None:
                extra += "  " + " ".join(
                    f"{k} {ema[k]:.4g}" for k in crit_keys if k in ema
                )
                extra += f"  critic {t_critic / step * 1e3:.0f} ms/step"
            print(
                f"  rollout {r + 1}/{args.rollouts}  psnr~ ×2 "
                f"{10 * math.log10(4 / ema['mse2']):.2f} ×1 "
                f"{10 * math.log10(4 / ema['mse1']):.2f}{extra}  "
                f"{el / (r + 1):.2f}s/rollout  "
                f"eta {(args.rollouts - r - 1) * el / (r + 1) / 60:.0f}m",
                flush=True,
            )
            if args.term_log:
                gn = grad_norms(replay_rng.choice(replay))
                grad_log.append({"rollout": r + 1, **gn, **{k: ema[k] for k in ema}})
                print(
                    "    grad norm per term (unweighted): "
                    + "  ".join(f"{k} {v:.3e}" for k, v in gn.items()),
                    flush=True,
                )
        if not evaluated:  # rollouts and updates only, before any whole-image eval
            peak_steps = torch.cuda.max_memory_allocated() / 1e9
        if (r + 1) % args.eval_every == 0 or r + 1 == args.rollouts:
            evaluated = True
            ev = evaluate()
            history.append({"rollout": r + 1, "step": step, **ev})
            print(f"  eval @ {r + 1}: {fmt(ev)}", flush=True)
    sec_per_rollout = (time.perf_counter() - t0) / max(1, args.rollouts)

    # --- read ---
    run_dir = make_run_dir("pixel_readout", label=args.label)
    head.eval()
    rows, artifacts = [], []
    counters: dict[str, int] = {}
    for item, key, f, lat, img, t2, bic, twin, chain in held_feats:
        x1, x2 = predict(f, lat)
        img, t2 = img.to(device).float(), t2.to(device).float()
        lum, dark = tone_stats(x2, t2)
        is_page = isinstance(item, Path)
        rows.append(
            {
                "stem": item.stem,
                "artist": "pages" if is_page else Path(item.npz_path).parent.name,
                "set": key,
                "psnr_x2_vs_teacher": psnr(x2, t2),
                "lpips_x2_vs_teacher": lpips_img(x2, t2),
                "luma_x2_vs_teacher": lum,
                "dark_x2_vs_teacher": dark,
                "psnr_bicubic_vs_teacher": bic,
                "psnr_teacher_twin": twin,
                "psnr_x1_vs_vae": psnr(x1, img),
                "psnr_down_vs_vae": psnr(down2(x2), img),
                "psnr_teacher_down_vs_vae": psnr(down2(t2), img),
            }
        )
        if chain is not None:
            chain = chain.to(device).float()
            rows[-1]["psnr_chain_vs_teacher"] = psnr(chain, t2)
        n = counters.get(key, 0)
        counters[key] = n + 1
        if n >= args.n_sheets and not key.endswith(".text"):
            continue
        cols = [("VAE -> ResShift x2", t2), ("x2 head", x2)]
        if chain is not None:
            cols = [("chain: head x1 -> ResShift x2", chain), ("x2 head", x2)]
        stem = f"{key.replace('.', '_')}_{n:02d}_{item.stem}"
        pair_sheet(cols, run_dir / f"{stem}_full.png")
        cx, cy = densest_window(t2, args.crop)
        pair_sheet(cols, run_dir / f"{stem}_crop.png", at=(cx, cy, args.crop))
        artifacts += [f"{stem}_full.png", f"{stem}_crop.png"]

    # head time on one 1024-tier image, eager, bf16 autocast (whole-image ×1 + ×2)
    _, _, f, lat, *_ = held_feats[0]
    fd, ld = f.to(device).float(), lat.to(device)
    for _ in range(3):
        predict(fd, ld)
    torch.cuda.synchronize()
    tt = time.perf_counter()
    for _ in range(10):
        predict(fd, ld)
    torch.cuda.synchronize()
    head_ms = (time.perf_counter() - tt) / 10 * 1e3

    b = head.base
    save_file(
        {k: v.detach().contiguous().cpu() for k, v in head.state_dict().items()},
        str(run_dir / "head_x2.safetensors"),
        metadata={
            "width": str(b.proj.out_features),
            "depth": str(len(b.blocks) + len(b.up_blocks)),
            "up": str(b.up),
            "skip": "0",
            "x2_depth": str(args.depth2),
            "sigma": ",".join(
                f"{lo}" if lo == hi else f"{lo}-{hi}" for lo, hi, _, _ in train_points
            ),
            "points": " ".join(args.points or []),
            "stream": args.stream,
            "adapter": args.adapter or "",
            "init": args.init,
            "lambda_lpips_x1": str(args.lambda_lpips_x1),
            "teacher": TEACHER_FILE,
        },
    )
    artifacts.append("head_x2.safetensors")

    set_metrics = {}
    for st in sets:
        for col in (
            "psnr_x2_vs_teacher",
            "lpips_x2_vs_teacher",
            "luma_x2_vs_teacher",
            "dark_x2_vs_teacher",
            "psnr_bicubic_vs_teacher",
            "psnr_x1_vs_vae",
            "psnr_down_vs_vae",
            "psnr_teacher_down_vs_vae",
            *(["psnr_chain_vs_teacher"] if args.chain else []),
        ):
            set_metrics[f"{st}_{col}"] = mean_or_none(
                [row[col] for row in rows if row["set"] == st]
            )
    metrics = {
        "train_points": [
            {"sigma_lo": lo, "sigma_hi": hi, "adapter": a, "real_x1": rl}
            for lo, hi, a, rl in train_points
        ],
        "eval_points": [{"name": n, "sigma": s, "adapter": a} for n, s, a in points],
        "n_train": len(train),
        "n_page_train": len(page_train),
        "n_page_rollouts": n_page_rollouts,
        "head_params_M": n_params / 1e6,
        "head_new_params_M": n_new / 1e6,
        "opt_steps_total": step,
        "sec_per_rollout": sec_per_rollout,
        "head_ms_whole_image": head_ms,
        "head_ms_image_tokens": list(f.shape[:2]),
        "peak_vram_gb": max(peak_setup, torch.cuda.max_memory_allocated() / 1e9),
        "peak_vram_setup_gb": peak_setup,
        "peak_vram_train_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_vram_steps_gb": peak_steps,
        "peak_vram_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        "critic_ms_per_step": t_critic / max(1, step) * 1e3,
        **refs,
        **set_metrics,
        "loss_weights": weights,
        "term_log": grad_log,
        "eval_history": history,
        "per_image": rows,
    }
    print(
        "×2 head vs teacher (bicubic | second teacher draw): "
        + " | ".join(
            f"{st}: {set_metrics[f'{st}_psnr_x2_vs_teacher']:.2f} "
            f"lpips {set_metrics[f'{st}_lpips_x2_vs_teacher']:.4f} "
            f"({refs[f'{st}_psnr_bicubic']:.2f} | {refs[f'{st}_psnr_teacher_twin']:.2f})"
            for st in sets
        ),
        flush=True,
    )
    if args.chain:
        print(
            "chain (teacher on the --init head's ×1 output) vs teacher: "
            + " | ".join(
                f"{st}: {refs[f'{st}_chain_psnr']:.2f} lpips {refs[f'{st}_chain_lpips']:.4f} "
                f"dark {refs[f'{st}_chain_dark'] or 0:+.2f}"
                for st in sets
            ),
            flush=True,
        )
    print(
        f"head {head_ms:.1f} ms on {tuple(f.shape[:2])} tokens, "
        f"{sec_per_rollout:.2f} s / rollout, peak {metrics['peak_vram_gb']:.1f} GB "
        f"(setup {peak_setup:.1f}, rollouts + updates {peak_steps or 0:.1f}, reserved "
        f"{metrics['peak_vram_reserved_gb']:.1f})",
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
