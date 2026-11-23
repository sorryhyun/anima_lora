#!/usr/bin/env python3
"""VAE | P1 linear | P2 head sheets for chosen images (pixel readout, read-only).

Two inputs: cached anima-base items (``--cached artist/stem``, cached latent + cached
caption) and raw images (``--images``, resized to ~4100 tokens at native aspect,
VAE-encoded, captioned with ``--caption`` through the Anima TE). Each image: one DiT
forward at σ_last of the 50-step shift-3 grid (same tap and noise rule as the training
scripts), then the P1 linear (from the P2 run's ``head.safetensors`` start, i.e.
``readout_1118_2219``) and the trained P2 head. PSNR is against the VAE decode.

Usage::

    make daemon-run ARGS="--queue --stall-timeout 600 bench/pixel_readout/render_sheets.py \
        --head_run bench/pixel_readout/results/<run> --cached <artist>/<stem> \
        --images a.jpg b.jpg --caption 'monochrome, greyscale, comic' --label sheets_mmdd_hhmm"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file

from bench._anima import (
    DEFAULT_TEXT_ENCODER,
    add_common_args,
    add_model_args,
    build_anima,
)
from bench._common import make_run_dir, start_heartbeat, write_result
from bench.pixel_readout._shared import P1_RUN, fit_size, load_head
from library.anima.pixel_head import PATCH, from_patches, lat_patches
from library.datasets.image_utils import IMAGE_TRANSFORMS
from library.env import resolve_under_home
from library.inference.sampling import get_timesteps_sigmas
from library.inference.text import prepare_text_inputs
from library.io.cache import (
    discover_cached_pairs,
    load_cached_latents,
    load_cached_text_features,
)
from library.models import qwen_vae
from scripts.pixel_head.train._common import densest_window, psnr, zoom_sheet


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(p, text_encoder=False)
    add_common_args(p)
    p.add_argument("--head_run", required=True)
    p.add_argument("--p1_run", default=P1_RUN)
    p.add_argument("--cache_root", default="post_image_dataset/lora")
    p.add_argument("--cached", nargs="*", default=[], help="artist/stem")
    p.add_argument("--images", nargs="*", default=[])
    p.add_argument("--caption", default="")
    p.add_argument("--text_encoder", default=DEFAULT_TEXT_ENCODER)
    p.add_argument("--infer_steps", type=int, default=50)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument(
        "--stream",
        choices=["fp32", "bf16"],
        default="fp32",
        help="DiT residual stream; fp32 is what heads train on since replay_1119_1225",
    )
    args = p.parse_args()
    start_heartbeat(label="sheets")
    device = torch.device(args.device)

    _, sigmas = get_timesteps_sigmas(args.infer_steps, args.flow_shift, device)
    sigma_last = float(sigmas[-2])
    vae = qwen_vae.load_vae(
        args.vae, device="cpu", dtype=torch.bfloat16, eval=True, vae_2d=True
    ).to(device)
    args.compile = False  # a handful of single forwards
    anima = build_anima(args, adapter=None, train_mode=False).anima
    if args.stream == "fp32":
        from library.anima.stream import stream_fp32_

        stream_fp32_(anima)
    tap: dict[str, torch.Tensor] = {}
    anima.final_layer.linear.register_forward_pre_hook(
        lambda _m, inp: tap.__setitem__("x", inp[0].detach())
    )

    D = anima.final_layer.hidden_size
    head = load_head(args.head_run, D, device)
    w1 = load_file(
        str(Path(resolve_under_home(args.p1_run)) / "readout_linear.safetensors")
    )["weight"]
    p1 = nn.Linear(D, PATCH * PATCH * 3).to(device)
    with torch.no_grad():
        p1.weight.copy_(w1[:, :D])
        p1.bias.copy_(w1[:, D])

    # --- inputs: (name, clean latent (1,C,h,w), crossattn (1,512,1024)) ---
    items = []
    cache_root = Path(resolve_under_home(args.cache_root))
    for spec in args.cached:
        artist, stem = spec.split("/", 1)
        ci = next(
            c for c in discover_cached_pairs(str(cache_root / artist)) if c.stem == stem
        )
        lat = load_cached_latents(ci.npz_path)[0].unsqueeze(0)
        cross, _ = load_cached_text_features(ci.te_path, variant=0)
        items.append((f"{artist}_{stem}", lat, cross.unsqueeze(0), None))
    if args.images:
        arg_c, _ = prepare_text_inputs(
            device=device,
            anima=anima,
            prompt=args.caption,
            text_encoder_path=str(resolve_under_home(args.text_encoder)),
        )
        cross = arg_c["embed"][0]
        for path in args.images:
            im = Image.open(path).convert("RGB")
            im = im.resize(fit_size(*im.size), Image.LANCZOS)
            px = IMAGE_TRANSFORMS(np.array(im))[None].to(device, torch.bfloat16)
            with torch.no_grad():
                lat = vae.encode_pixels_to_latents(px).squeeze(2).float().cpu()
            name = "_".join(Path(path).parts[-2:]).rsplit(".", 1)[0]
            items.append((name, lat, cross, px[0].float().cpu()))
    print(f"{len(items)} images, sigma_last {sigma_last:.4f}", flush=True)

    run_dir = make_run_dir("pixel_readout", label=args.label)
    rows, artifacts = [], []
    for k, (name, lat, cross, orig) in enumerate(items):
        lat = lat.to(device)
        with torch.no_grad():
            tgt = vae.decode_to_pixels(lat)[0].float()
            g = torch.Generator(device=device).manual_seed(
                args.seed * 100003 + 20_000 + k
            )
            noise = torch.randn(lat.shape, generator=g, device=device, dtype=lat.dtype)
            x_t = (1 - sigma_last) * lat + sigma_last * noise
            sig = torch.tensor([sigma_last], device=device)
            pad = torch.zeros(
                1, 1, *lat.shape[-2:], dtype=torch.bfloat16, device=device
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = anima(
                    x_t.unsqueeze(2).to(torch.bfloat16),
                    sig,
                    cross.to(device, torch.bfloat16),
                    padding_mask=pad,
                )
            f = tap.pop("x")[0, 0].float()
            gh, gw = f.shape[:2]
            lin = from_patches(p1(f).reshape(gh * gw, -1), gh, gw).clamp(-1, 1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = head(f, lat_patches(x_t).float())
            pred = pred.float().clamp(-1, 1)
            x0_hat = x_t - sigma_last * v.float().squeeze(2)
            dec_hat = vae.decode_to_pixels(x0_hat)[0].float()
        row = {
            "name": name,
            "psnr_p1_linear": psnr(lin, tgt),
            "psnr_head": psnr(pred, tgt),
            "psnr_vae_xhat0": psnr(dec_hat, tgt),
            "psnr_head_vs_xhat0": psnr(pred, dec_hat),
        }
        if orig is not None:  # raw images: the VAE's own loss against the pixels
            o = orig.to(device)
            row["psnr_vae_vs_orig"] = psnr(tgt, o)
            row["psnr_head_vs_orig"] = psnr(pred, o)
        rows.append(row)
        print(
            f"  {name}: "
            + "  ".join(f"{k[5:]} {v:.2f}" for k, v in row.items() if k != "name"),
            flush=True,
        )
        fn = f"sheet_{k:02d}_{name}.png"
        cols = [("VAE decode", tgt), ("P1 linear", lin), ("P2 ConvNeXt head", pred)]
        if orig is not None:
            cols = [("original", orig.to(device)), cols[0], cols[2]]
        zoom_sheet(
            cols,
            run_dir / fn,
            at=densest_window(tgt, 384),
        )
        artifacts.append(fn)

    out = write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={"sigma_last": sigma_last, "per_image": rows},
        label=args.label,
        artifacts=artifacts,
        device=device,
    )
    print(f"envelope: {out}")


if __name__ == "__main__":
    main()
