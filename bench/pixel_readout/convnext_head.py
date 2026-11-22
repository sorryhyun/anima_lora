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
from glob import escape as glob_escape
import os
import random
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from bench.pixel_readout._compile import compile_blocks_isolated
from bench._anima import (
    DEFAULT_TEXT_ENCODER,
    add_common_args,
    add_model_args,
    build_anima,
)
from bench._common import make_run_dir, start_heartbeat, write_result
from bench.pixel_readout.linear_probe import (
    PATCH,
    from_patches,
    pick_samples,
    psnr,
    to_pil,
)
from library.anima.pixel_head import (  # noqa: F401  (re-exported for the bench scripts)
    ConvNeXtBlock,
    PixelHead,
    lat_patches,
    load_pixel_head,
)
from library.anima.pixel_head_train import (  # noqa: F401  (re-exported for the scripts here)
    densest_window,
    rec_loss,
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

RESULTS = "bench/pixel_readout/results"
P1_RUN = f"{RESULTS}/20261118-2221-readout_1118_2219"
def _local_split(key: str) -> tuple[str, ...]:
    """An artist list of the local split file (``output/pixel_head/x1_split.json``, kept
    out of the repo); () when it is absent."""
    import json

    from library.env import resolve_under_home

    path = resolve_under_home("output/pixel_head/x1_split.json")
    return tuple(json.loads(path.read_text())[key]) if path.exists() else ()


TEXT_ARTISTS = _local_split("text_artists")  # most "Japanese text" captions in post_image_dataset/resized


# --------------------------------------------------------------------------- head


# ConvNeXtBlock / PixelHead live in library/anima/pixel_head.py (promoted 2026-11-19).


def resolve_run(run: str | Path) -> Path:
    """A results directory, or a bare label → the newest ``results/*-<label>``."""
    path = Path(resolve_under_home(str(run)))
    if path.is_dir():
        return path
    hits = sorted(Path(resolve_under_home(RESULTS)).glob(f"*-{run}"))
    if not hits:
        raise FileNotFoundError(
            f"no run {run} (neither a path nor a label in {RESULTS})"
        )
    return hits[-1]


def load_head(run: str | Path, d_feat: int, device) -> PixelHead:
    """A trained head from ``<run>/head.safetensors`` (shape from its metadata)."""
    return load_pixel_head(resolve_run(run) / "head.safetensors", device, d_feat)


def attach_adapter(anima, path: str, device):
    """A LoRA-family adapter as hooks on a built DiT (``set_multiplier`` switches it).

    Call after ``stream_fp32_``: each hook captures its module's forward when it is
    applied, so the reverse order would keep the bf16 linears under the adapter.
    ``inference.py`` merges a plain LoRA before the stream change instead; same function."""
    from networks.lora_anima.factory import create_network_from_weights

    path = str(resolve_under_home(path))
    network, _ = create_network_from_weights(
        1.0, path, None, None, anima, for_inference=True
    )
    network.apply_to([], anima, apply_text_encoder=False, apply_unet=True)
    print(f"adapter {path}: {network.load_weights(path)}", flush=True)
    network.to(device=device, dtype=torch.bfloat16).requires_grad_(False)
    return network


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


def real_pixels(item) -> torch.Tensor:
    """The image an item's latent was encoded from, (3, H, W) uint8: a cached pair's
    resized image, or a build_pages.py page (stored pixels, else its source resized
    as the builder did)."""
    if isinstance(item, Path):
        with safe_open(str(item), "pt") as fh:
            if "pixels" in fh.keys():
                return fh.get_tensor("pixels")
            meta = fh.metadata()
        w, h = (int(v) for v in meta["size"].split("x"))
        im = Image.open(meta["source"]).convert("RGB").resize((w, h), Image.LANCZOS)
        return torch.from_numpy(np.array(im)).permute(2, 0, 1)
    if item.image_path:  # x2_flex.py's rows name their ×1 image
        im = Image.open(item.image_path).convert("RGB")
        return torch.from_numpy(np.array(im)).permute(2, 0, 1)
    src = caption_path(item)
    hits = [
        q
        for q in src.parent.glob(f"{glob_escape(src.stem)}.*")
        if q.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    ]
    assert hits, f"no resized image for {item.npz_path}"
    return torch.from_numpy(np.array(Image.open(hits[0]).convert("RGB"))).permute(
        2, 0, 1
    )


def page_token_count(path: Path) -> int:
    """Tokens of a build_pages.py page, from its stored ``size`` (WxH pixels)."""
    with safe_open(str(path), "pt") as fh:
        w, h = (int(v) for v in fh.metadata()["size"].split("x"))
    return (w // PATCH) * (h // PATCH)


def token_count(ci) -> int:
    w, h = (int(v) for v in ci.npz_path.rsplit("_", 2)[-2].split("x"))
    return (w // PATCH) * (h // PATCH)


# --------------------------------------------------------------------------- sheets


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
    p.add_argument(
        "--up",
        type=int,
        default=0,
        help="the last N of --depth blocks run after one 2× upsample at width/2",
    )
    p.add_argument(
        "--target",
        choices=["x0", "xhat0"],
        default="x0",
        help="train against VAE(x0) (clean latent) or VAE(x̂0) (x_t − σ·v, the DiT's own "
        "estimate from the same forward); evals report both",
    )
    p.add_argument(
        "--pages",
        default=None,
        help="build_pages.py output (train/, eval/): manga pages in the mix, eval/ "
        "pages scored as a third held-out set",
    )
    p.add_argument("--page_share", type=float, default=0.0, help="P(rollout is a page)")
    p.add_argument(
        "--page_caption",
        default="monochrome, greyscale, comic, speech bubble, japanese text",
    )
    p.add_argument(
        "--stream",
        choices=["bf16", "fp32"],
        default="fp32",
        help="DiT residual stream (stream_fp32_); fp32 is inference.py's default. Runs up "
        "to convnext_1119_1106 used bf16",
    )
    p.add_argument(
        "--replay",
        type=int,
        default=0,
        help="> 0: keep the last N rollouts on the card and draw every update's sample "
        "uniformly from them (--opt_steps updates per rollout); 0 = the newest rollout(s)",
    )
    p.add_argument("--text_encoder", default=DEFAULT_TEXT_ENCODER)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr_linear", type=float, default=2e-5)
    p.add_argument("--warmup", type=int, default=200, help="opt steps")
    p.add_argument("--eval_every", type=int, default=600, help="rollouts")
    p.add_argument("--infer_steps", type=int, default=50)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="train and read at this σ instead of the grid's last (turbo 4-step: 0.5)",
    )
    p.add_argument(
        "--adapter",
        default=None,
        help="LoRA-family adapter on the frozen DiT (hooks, after the stream change)",
    )
    p.add_argument(
        "--points",
        nargs="+",
        default=None,
        metavar="SIGMA[-SIGMA][:adapter][:real]",
        help="decode points for one head, taken in turn per rollout: σ on the base DiT, "
        "or σ:adapter with --adapter switched on (e.g. 0.1 0.5:adapter). LO-HI draws σ "
        "uniformly per rollout and is evaluated at LO, the midpoint and HI. :real trains "
        "that point against the original image instead of --target, with "
        "--lambda_lpips. Evals, sheets and metrics are per evaluated point. Overrides "
        "--sigma",
    )
    p.add_argument(
        "--eval_points",
        nargs="+",
        default=None,
        metavar="SIGMA[:adapter]",
        help="with --points: more σ the head is read at and not trained on (a point "
        "--points already reads is skipped)",
    )
    p.add_argument(
        "--lambda_lpips",
        type=float,
        default=0.0,
        help="weight of LPIPS (VGG) on the :real points, beside the reconstruction loss",
    )
    p.add_argument(
        "--lpips_crop",
        type=int,
        default=512,
        help="LPIPS on one random crop of this side per update (lpips_speed.py, archived: 512² "
        "9.0 ms, 768² 19.6 ms compiled)",
    )
    p.add_argument(
        "--grad_ratio",
        type=int,
        default=0,
        help="print the LPIPS / reconstruction gradient-norm ratio (unweighted, over the "
        "head's parameters) on the first N :real rollouts, before any update",
    )
    p.add_argument(
        "--init", default=None, help="start the head from <run>/head.safetensors"
    )
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
    page_train: list[Path] = []
    page_eval: list[Path] = []
    if args.pages:
        proot = Path(resolve_under_home(args.pages))
        page_train = sorted((proot / "train").glob("*.safetensors"))
        page_eval = sorted((proot / "eval").glob("*.safetensors"))
    page_tokens = [page_token_count(pth) for pth in page_train + page_eval]
    seq_bands = cluster_token_bands(
        [token_count(ci) for ci in train + test + text] + page_tokens
    )
    _, sigmas = get_timesteps_sigmas(args.infer_steps, args.flow_shift, device)
    sigma_last = float(sigmas[-2]) if args.sigma is None else args.sigma
    # training points (σ lo, σ hi, adapter on) and the fixed points they are read at
    # (name, σ, adapter on); one unnamed point = the single-σ run (adapter on if given)
    # a point's training target: --target, or "real" (the original image) when tagged
    train_points: list[tuple[float, float, bool, str]] = [
        (sigma_last, sigma_last, True, args.target)
    ]
    points: list[tuple[str, float, bool]] = [("", sigma_last, True)]
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
            assert lo <= hi, f"--points {spec}: LO-HI"
            train_points.append(
                (lo, hi, tag, "real" if "real" in tags else args.target)
            )
            for sg in dict.fromkeys((lo, (lo + hi) / 2, hi)):
                name = f"{'lora' if tag else 'base'}_s{sg:.4g}"
                points.append((name, sg, tag))
        assert len({n for n, _, _ in points}) == len(points), "duplicate --points"
        for spec in args.eval_points or []:
            s, *tags = spec.split(":")
            assert set(tags) <= {"adapter"}, f"--eval_points {spec}: SIGMA[:adapter]"
            tag = "adapter" in tags
            assert not tag or args.adapter, f"--eval_points {spec} needs --adapter"
            name = f"{'lora' if tag else 'base'}_s{float(s):.4g}"
            if name not in {n for n, _, _ in points}:
                points.append((name, float(s), tag))
    else:
        assert not args.eval_points, "--eval_points needs --points"
    multi = bool(points[0][0])
    sig_rng = random.Random(args.seed + 4)

    def skey(pt: str, key: str) -> str:
        return f"{pt}.{key}" if pt else key

    print(
        f"train {len(train)}  test {len(test)} ({', '.join(test_artists)})  "
        f"text {len(text)}  pages {len(page_train)} / eval {len(page_eval)}  "
        f"bands {seq_bands}  "
        + (
            "points " + ", ".join(n for n, _, _ in points)
            if multi
            else f"sigma_last {sigma_last:.4f}"
        ),
        flush=True,
    )

    # --- models: VAE (targets, stays on card) + frozen DiT ---
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

    def load_real(item, hw: tuple[int, int]) -> torch.Tensor:
        """The image the latent was encoded from, (3, H, W) in [-1, 1] on the card."""
        px = real_pixels(item)
        assert tuple(px.shape[-2:]) == hw, (item, tuple(px.shape), hw)
        return px.to(device).float() / 127.5 - 1

    @torch.no_grad()
    def rollout(item, idx: int, targets=None, point=None):
        """→ (features, x_t patches, {target name: VAE decode}) at one decode point."""
        targets = targets or (args.target,)
        _, sigma_last, adapter_on = point or points[0]
        if network is not None:
            # off = each module returns its org_forward: the base DiT's own output
            network.set_enabled(adapter_on)
        lat, cross = load_item(item)
        g = torch.Generator(device=device).manual_seed(args.seed * 100003 + idx)
        noise = torch.randn(lat.shape, generator=g, device=device, dtype=lat.dtype)
        x_t = (1 - sigma_last) * lat + sigma_last * noise
        sig = torch.tensor([sigma_last], device=device)
        pad = torch.zeros(1, 1, *lat.shape[-2:], dtype=torch.bfloat16, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = anima(x_t.unsqueeze(2).to(torch.bfloat16), sig, cross, padding_mask=pad)
        f = tap.pop("x")[0, 0]  # (gh, gw, D)
        tgts = {}
        if "x0" in targets:
            tgts["x0"] = vae.decode_to_pixels(lat)[0].float()
        if "xhat0" in targets:
            x0_hat = x_t - sigma_last * v.float().squeeze(2)
            tgts["xhat0"] = vae.decode_to_pixels(x0_hat)[0].float()
        if "real" in targets:
            tgts["real"] = load_real(item, (lat.shape[-2] * 8, lat.shape[-1] * 8))
        return f.float(), lat_patches(x_t).float(), tgts

    # --- head ---
    D = anima.final_layer.hidden_size
    head = PixelHead(
        D,
        64,
        args.width,
        args.depth,
        args.ls_init,
        use_skip=not args.no_skip,
        up=args.up,
    ).to(device)
    w1 = load_file(
        str(Path(resolve_under_home(args.p1_run)) / "readout_linear.safetensors")
    )
    w1 = w1["weight"]  # (768, D+1), last column = bias
    with torch.no_grad():
        head.linear.weight.copy_(w1[:, :D])
        head.linear.bias.copy_(w1[:, D])
    if args.init:
        head.load_state_dict(
            load_file(str(resolve_run(args.init) / "head.safetensors"))
        )
        print(f"head init from {args.init}", flush=True)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"head {n_params / 1e6:.1f}M params", flush=True)
    # training forward only; evals and sheets stay eager (rollout_profile: −15 % per update)
    head_train = torch.compile(head, dynamic=True) if compile_blocks else head
    import lpips

    lp = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    lp.requires_grad_(False)
    lp.to(
        memory_format=torch.channels_last
    )  # + compile: lpips_speed.py (archived), x2_head.py
    lp_train = torch.compile(lp) if compile_blocks else lp
    lp_rng = random.Random(args.seed + 5)

    def lpips_term(pred, tgt, fn=None):
        """LPIPS of one random --lpips_crop window (the whole image when smaller)."""
        a, b = pred[None], tgt[None]
        c = args.lpips_crop
        if c and min(a.shape[-2:]) > c:
            y = lp_rng.randint(0, a.shape[-2] - c)
            x = lp_rng.randint(0, a.shape[-1] - c)
            a, b = a[..., y : y + c, x : x + c], b[..., y : y + c, x : x + c]
        cl = torch.channels_last
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = (fn or lp_train)(
                a.contiguous(memory_format=cl), b.contiguous(memory_format=cl)
            )
        return out.mean().float()

    @torch.no_grad()
    def lpips_img(a: torch.Tensor, b: torch.Tensor, tile: int = 768) -> float:
        """Whole-image LPIPS, the area-weighted mean over ``tile``-px tiles."""
        h, w = a.shape[-2:]
        tot = 0.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for y in range(0, h, tile):
                for x in range(0, w, tile):
                    ta = a[None, :, y : y + tile, x : x + tile].float()
                    tb = b[None, :, y : y + tile, x : x + tile].float()
                    tot += lp(ta, tb).mean().item() * ta.shape[-2] * ta.shape[-1]
        return tot / (h * w)

    def grad_ratio(f, lat, tgt) -> tuple[float, float]:
        """Unweighted gradient norms (reconstruction, LPIPS) over the head (eager)."""
        params = [q for q in head.parameters() if q.requires_grad]
        norms = []
        for term in ("rec", "lpips"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = head(f.float(), lat)
            loss = (
                rec_loss(pred, tgt)[0]
                if term == "rec"
                else lpips_term(pred.float(), tgt, fn=lp)
            )
            gs = torch.autograd.grad(loss, params, allow_unused=True)
            norms.append(
                math.sqrt(
                    sum(float(g.float().pow(2).sum()) for g in gs if g is not None)
                )
            )
        return norms[0], norms[1]

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
    # (item, set, features, x_t patches, {target: decode}, original | None, point name)
    held_feats = []
    held_items = (
        [(ci, "test") for ci in test]
        + [(ci, "text") for ci in text]
        + [(pth, "manga") for pth in page_eval]
    )
    origs: dict[int, torch.Tensor] = {}
    for point in points:  # the same noise per item at every point
        for j, (item, key) in enumerate(held_items):
            f, lat, tgts = rollout(
                item, 10_000 + j, targets=("x0", "xhat0"), point=point
            )
            if j not in origs:  # one copy per item, shared by its points
                origs[j] = load_real(item, tuple(tgts["x0"].shape[-2:])).to(
                    "cpu", torch.float16
                )
            orig = origs[j]
            held_feats.append(
                (
                    item,
                    key,
                    f.to("cpu", torch.bfloat16),
                    lat.cpu(),
                    {k: v.to("cpu", torch.float16) for k, v in tgts.items()},
                    orig,
                    point[0],
                )
            )
    sets = list(dict.fromkeys(skey(h[6], h[1]) for h in held_feats))
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
        """{set: mean PSNR vs VAE(x0)}, {set_xhat0: … vs VAE(x̂0)}, and against the
        original image {set_orig: PSNR, set_lpips_orig: LPIPS}."""
        head.eval()
        out: dict[str, list] = {}
        for _, key, f, lat, tgts, orig, pt in held_feats:
            key = skey(pt, key)
            pred = predict(f, lat)
            o = orig.to(device).float()
            out.setdefault(key, []).append(psnr(pred, tgts["x0"].to(device)))
            out.setdefault(f"{key}_xhat0", []).append(
                psnr(pred, tgts["xhat0"].to(device))
            )
            out.setdefault(f"{key}_orig", []).append(psnr(pred, o))
            out.setdefault(f"{key}_lpips_orig", []).append(lpips_img(pred, o))
        head.train()
        return {k: float(np.mean(v)) for k, v in out.items()}

    def fmt(ev):
        return "  ".join(
            f"{k} {ev[k]:.2f} (x̂0 {ev[k + '_xhat0']:.2f}, orig {ev[k + '_orig']:.2f} "
            f"lpips {ev[k + '_lpips_orig']:.4f})"
            for k in sets
        )

    p1_lin: dict[str, list] = {}
    for _, key, f, lat, tgts, _, pt in held_feats:
        p1_lin.setdefault(skey(pt, key), []).append(
            psnr(predict(f, lat, use_head=False), tgts["x0"].to(device))
        )
    p1_scores = {k: float(np.mean(v)) for k, v in p1_lin.items()}
    print(
        "P1 linear: " + "  ".join(f"{k} {v:.2f}" for k, v in p1_scores.items()),
        flush=True,
    )
    # P1 linear weights are about to train; keep a frozen copy for the sheets
    p1_linear = nn.Linear(D, PATCH * PATCH * 3).to(device)
    p1_linear.load_state_dict(head.linear.state_dict())

    # --- train ---
    history = []
    order: list = []
    page_order: list = []
    page_rng = random.Random(args.seed + 2)  # own stream: colour order as without pages
    n_page_rollouts = 0
    step = 0
    t0 = time.perf_counter()
    loss_ema = mse_ema = lp_ema = None
    ratios: list[tuple[float, float]] = []
    window: deque = deque(maxlen=args.accum)
    assert not (args.replay and args.accum > 1), "--replay and --accum do not combine"
    replay: deque = deque(maxlen=max(1, args.replay))
    replay_rng = random.Random(args.seed + 3)
    if args.init:  # the loaded head on this run's points and meters
        ev = evaluate()
        history.append({"rollout": 0, "step": 0, **ev})
        print(f"  eval @ 0 ({args.init} as loaded): {fmt(ev)}", flush=True)
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
        lo, hi, adapter_on, tname = train_points[r % len(train_points)]
        sg = lo if hi == lo else sig_rng.uniform(lo, hi)
        f, lat, tgts = rollout(item, r, targets=(tname,), point=("", sg, adapter_on))
        perc = tname == "real" and args.lambda_lpips > 0
        if tname == "real" and len(ratios) < args.grad_ratio:
            ratios.append(grad_ratio(f, lat, tgts[tname]))
            if len(ratios) == args.grad_ratio:
                rr = sorted(lpg / max(rg, 1e-12) for rg, lpg in ratios)
                print(
                    f"  grad norm LPIPS / rec over {len(rr)} real rollouts: median "
                    f"{rr[len(rr) // 2]:.2f} (min {rr[0]:.2f}, max {rr[-1]:.2f}); rec "
                    f"median {sorted(g for g, _ in ratios)[len(rr) // 2]:.4f}",
                    flush=True,
                )
        if args.replay:
            # bf16 features lose nothing: the head's first linear runs under bf16 autocast
            replay.append((f.to(torch.bfloat16), lat, tgts[tname].half(), perc))
        else:
            window.append((f, lat, tgts[tname], perc))
        for _ in range(args.opt_steps):
            set_lr(step)
            opt.zero_grad(set_to_none=True)
            # window: newest rollout last, so the logged loss / mse are this rollout's;
            # replay: the logged loss is the last drawn sample's
            batch = [replay_rng.choice(replay)] if args.replay else window
            for f, lat, tgt, perc in batch:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = head_train(f, lat)
                loss, mse = rec_loss(pred, tgt.float())
                if perc:
                    lp_v = lpips_term(pred.float(), tgt.float())
                    loss = loss + args.lambda_lpips * lp_v
                    lv_ = lp_v.item()
                    lp_ema = lv_ if lp_ema is None else 0.98 * lp_ema + 0.02 * lv_
                (loss / len(batch)).backward()
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
                + (f"lpips {lp_ema:.4f}  " if lp_ema is not None else "")
                + f"{el / (r + 1):.2f}s/rollout  eta {(args.rollouts - r - 1) * el / (r + 1) / 60:.0f}m",
                flush=True,
            )
        if (r + 1) % args.eval_every == 0 or r + 1 == args.rollouts:
            ev = evaluate()
            history.append({"rollout": r + 1, "step": step, **ev})
            print(f"  eval @ {r + 1}: {fmt(ev)}", flush=True)

    # --- read ---
    run_dir = make_run_dir("pixel_readout", label=args.label)
    rows, artifacts = [], []
    head.eval()
    counters: dict[str, int] = {}
    for item, key, f, lat, tgts, orig, pt in held_feats:
        tgt = tgts["x0"].to(device).float()
        tgt_hat = tgts["xhat0"].to(device).float()
        pred = predict(f, lat)
        with torch.no_grad():
            fd = f.to(device).float()
            gh, gw = fd.shape[:2]
            lin = from_patches(
                p1_linear(fd).reshape(gh * gw, -1).float(), gh, gw
            ).clamp(-1, 1)
        is_page = isinstance(item, Path)
        row = {
            "stem": item.stem,
            "artist": "pages" if is_page else Path(item.npz_path).parent.name,
            "set": skey(pt, key),
            "point": pt,
            "psnr_p1_linear": psnr(lin, tgt),
            "psnr_head": psnr(pred, tgt),
            "psnr_head_xhat0": psnr(pred, tgt_hat),
            "psnr_vae_xhat0": psnr(tgt_hat, tgt),
        }
        o = orig.to(device).float()
        row["psnr_vae_vs_orig"] = psnr(tgt, o)
        row["psnr_head_vs_orig"] = psnr(pred, o)
        row["lpips_vae_vs_orig"] = lpips_img(tgt, o)
        row["lpips_head_vs_orig"] = lpips_img(pred, o)
        # the VAE path at this point: the decode of the DiT's own x̂0
        row["psnr_vaehat_vs_orig"] = psnr(tgt_hat, o)
        row["lpips_vaehat_vs_orig"] = lpips_img(tgt_hat, o)
        rows.append(row)
        n = counters.get((pt, key), 0)
        counters[(pt, key)] = n + 1
        pre = f"{pt}_" if pt else ""
        if key == "manga":
            cols = [("original", o), ("VAE decode", tgt), ("head", pred)]
            name = f"{pre}manga_{n:02d}_{item.stem}.png"
            sheet(cols, run_dir / name, at=densest_window(tgt, 384))
        elif key == "text":
            cols = [("VAE decode", tgt), ("P1 linear", lin), ("P2 ConvNeXt head", pred)]
            name = f"{pre}text_{n:02d}_{item.stem}.png"
            sheet(cols, run_dir / name, at=densest_window(tgt, 384))
        elif n < args.n_sheets:
            cols = [("VAE decode", tgt), ("P1 linear", lin), ("P2 ConvNeXt head", pred)]
            name = f"{pre}sheet_{n:02d}_{item.stem}.png"
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
            "up": str(args.up),
            "target": args.target,
            "sigma": ",".join(
                f"{lo}" if lo == hi else f"{lo}-{hi}" for lo, hi, _, _ in train_points
            ),
            "lambda_lpips": str(args.lambda_lpips),
            "points": " ".join(args.points or []),
            "stream": args.stream,
            "replay": str(args.replay),
            "adapter": args.adapter or "",
            "init": args.init or "",
        },
    )
    artifacts.append("head.safetensors")

    set_metrics = {}  # <set>_<column>; a multi-point run's sets are <point>.<set>
    for st in sets:
        for col in (
            "psnr_p1_linear",
            "psnr_head",
            "psnr_head_xhat0",
            "psnr_head_vs_orig",
            "psnr_vae_vs_orig",
            "lpips_head_vs_orig",
            "lpips_vae_vs_orig",
            "psnr_vaehat_vs_orig",
            "lpips_vaehat_vs_orig",
        ):
            vals = [r[col] for r in rows if r["set"] == st and col in r]
            if vals:
                set_metrics[f"{st}_{col}"] = float(np.mean(vals))

    metrics = {
        "sigma_last": points[0][1],
        "train_points": [
            {
                "sigma_lo": lo,
                "sigma_hi": hi,
                "adapter": bool(a and args.adapter),
                "target": t,
            }
            for lo, hi, a, t in train_points
        ],
        "eval_points": [
            {"name": n, "sigma": s, "adapter": bool(a and args.adapter)}
            for n, s, a in points
        ],
        "n_train": len(train),
        "n_test": len(test),
        "n_text": len(text),
        "test_artists": test_artists,
        "text_images": [f"{Path(ci.npz_path).parent.name}/{ci.stem}" for ci in text],
        "head_params_M": n_params / 1e6,
        "opt_steps_total": step,
        "sec_per_rollout": (time.perf_counter() - t0) / args.rollouts,
        **set_metrics,
        **(
            {"n_page_train": len(page_train), "n_page_rollouts": n_page_rollouts}
            if page_eval
            else {}
        ),
        "eval_history": history,
        "per_image": rows,
    }
    print(
        "PSNR vs VAE(x0) (x̂0): "
        + " | ".join(
            f"{st}: P1 {set_metrics[f'{st}_psnr_p1_linear']:.2f} → head "
            f"{set_metrics[f'{st}_psnr_head']:.2f} "
            f"({set_metrics[f'{st}_psnr_head_xhat0']:.2f})"
            for st in sets
        ),
        flush=True,
    )
    print(
        "vs original, PSNR / LPIPS: "
        + " | ".join(
            f"{st}: VAE(x0) {set_metrics[f'{st}_psnr_vae_vs_orig']:.2f} / "
            f"{set_metrics[f'{st}_lpips_vae_vs_orig']:.4f}, VAE(x̂0) "
            f"{set_metrics[f'{st}_psnr_vaehat_vs_orig']:.2f} / "
            f"{set_metrics[f'{st}_lpips_vaehat_vs_orig']:.4f}, head "
            f"{set_metrics[f'{st}_psnr_head_vs_orig']:.2f} / "
            f"{set_metrics[f'{st}_lpips_head_vs_orig']:.4f}"
            for st in sets
        ),
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
