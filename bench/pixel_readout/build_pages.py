#!/usr/bin/env python3
"""Precache VAE latents of manga pages for the pixel head's training mix (P3 manga arms).

Pools: Manga109-s images (spreads) and hitocrawl JA pages. Held out: the 4 Manga109 titles
and 6 hitocrawl artists of ``manga_1119_1052``; their pages go to ``eval/`` (the same page as
in that run: Manga109 p.14, hitocrawl page 8 of each artist's first work), everything drawn
for training to ``train/``. Pages are resized to ~4100 tokens at native aspect (the
``_shared.fit_size`` rule) and VAE-encoded; one ``<source>__<name>.safetensors`` per page
holding ``latent`` (C, h, w) bf16, plus ``pixels`` (3, H, W) uint8 for eval pages. Captions are not stored: the training script encodes one
fixed tag caption.

Usage::

    make daemon-run ARGS="--queue --stall-timeout 600 bench/pixel_readout/build_pages.py \
        --n_manga109 1500 --n_hito 1500"
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

from bench._anima import add_model_args
from bench._common import start_heartbeat
from bench.pixel_readout._shared import fit_size
from library.datasets.image_utils import IMAGE_TRANSFORMS
from library.env import resolve_under_home
from library.models import qwen_vae

MANGA109 = "/media/sorryhyun/new/dataset/manga109s/Manga109s_released_2026_05_21/images"
HITO = "/media/sorryhyun/new/dataset/hitocrawl/retrieved"
EVAL_M109 = ("ARMS", "DollGun", "AosugiruHaru", "BakuretsuKungFuGirl")
EVAL_HITO = (
    "amatani-haruka",
    "gosaiji",
    "haruhisky",
    "ishikei",
    "misaka12003",
    "ohtomo-takuji",
)


def first_work(artist_dir: Path) -> Path:
    return sorted(p for p in artist_dir.iterdir() if p.is_dir())[0]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(p, dit=False, text_encoder=False)
    p.add_argument("--out", default="post_image_dataset/pixel_pages")
    p.add_argument("--n_manga109", type=int, default=1500)
    p.add_argument("--n_hito", type=int, default=1500)
    p.add_argument(
        "--skip_first", type=int, default=2, help="hitocrawl pages per work (covers)"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    start_heartbeat(label="pages")
    rng = random.Random(args.seed)
    out = Path(resolve_under_home(args.out))

    m109 = Path(MANGA109)
    hito = Path(HITO)
    eval_items = [("m109", m109 / t / "014.jpg") for t in EVAL_M109] + [
        ("hito", first_work(hito / a) / "ja" / "0008.webp") for a in EVAL_HITO
    ]
    m_pool = sorted(
        f
        for t in m109.iterdir()
        if t.is_dir() and t.name not in EVAL_M109
        for f in t.glob("*.jpg")
    )
    h_pool = sorted(
        f
        for a in hito.iterdir()
        if a.is_dir() and a.name not in EVAL_HITO
        for f in a.glob("*/ja/*.webp")
        if int(f.stem) > args.skip_first
    )
    train_items = [("m109", f) for f in rng.sample(m_pool, args.n_manga109)] + [
        ("hito", f) for f in rng.sample(h_pool, args.n_hito)
    ]
    print(
        f"pools: manga109 {len(m_pool)}, hito {len(h_pool)}; train {len(train_items)}, "
        f"eval {len(eval_items)}",
        flush=True,
    )

    device = torch.device(args.device)
    vae = qwen_vae.load_vae(
        args.vae, device="cpu", dtype=torch.bfloat16, eval=True, vae_2d=True
    ).to(device)
    manifest = {}
    t0 = time.perf_counter()
    for split, items in (("eval", eval_items), ("train", train_items)):
        (out / split).mkdir(parents=True, exist_ok=True)
        rows = []
        for k, (src, path) in enumerate(items):
            # Manga109: <title>/<page>; hitocrawl: <artist>/<work>/ja/<page>
            parts = (
                path.parts[-2:]
                if src == "m109"
                else (path.parts[-4], path.parts[-3], path.stem)
            )
            name = f"{src}__" + "_".join(parts).rsplit(".", 1)[0]
            dst = out / split / f"{name}.safetensors"
            rows.append({"name": name, "source": str(path)})
            if dst.exists():
                continue
            im = Image.open(path).convert("RGB")
            im = im.resize(fit_size(*im.size), Image.LANCZOS)
            px = IMAGE_TRANSFORMS(np.array(im))[None].to(device, torch.bfloat16)
            with torch.no_grad():
                lat = vae.encode_pixels_to_latents(px).squeeze(2)[0]
            tensors = {"latent": lat.to(torch.bfloat16).contiguous().cpu()}
            if split == "eval":  # original pixels for "vs original" reads
                tensors["pixels"] = (
                    torch.from_numpy(np.array(im)).permute(2, 0, 1).contiguous()
                )
            save_file(
                tensors,
                str(dst),
                metadata={"source": str(path), "size": f"{im.width}x{im.height}"},
            )
            if (k + 1) % 250 == 0:
                print(
                    f"  {split} {k + 1}/{len(items)}  {time.perf_counter() - t0:.0f}s",
                    flush=True,
                )
        manifest[split] = rows
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"done in {time.perf_counter() - t0:.0f}s → {out}", flush=True)


if __name__ == "__main__":
    main()
