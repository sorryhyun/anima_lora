#!/usr/bin/env python3
"""Real 2× targets from the masters: supply and alignment (x2_plan.md, arm B). CPU only.

A resized training image is the master cover-scaled to its bucket and anchor-cropped
(``anime_tools.stages.resize.resize_to_bucket``; anchor and margins are stamped in the
PNG). The same geometry at twice the cover size and twice the crop offset gives the 2×
image of the same crop, a real target for the ×2 head. This script reads every resized image's master, writes a manifest
(scale of the 2× resize relative to the master, master format) and, on a sample, builds the
2× target and reads its 2×2 box-mean against the resized image: the alignment check.

Usage::

    python -m bench.pixel_readout.x2_masters --label x2masters_mmdd_hhmm
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from anime_tools.stages.resize import (
    MARGINS_KEY,
    margin_box,
    normalize_crop_margins,
    oriented_size,
)
from PIL import Image

from bench._common import make_run_dir, write_result
from bench.pixel_readout._masters import master_of, target_2x
from library.env import resolve_under_home

Image.MAX_IMAGE_PIXELS = None


def psnr8(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return 99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--resized", default="post_image_dataset/resized")
    p.add_argument("--masters", default="image_dataset")
    p.add_argument("--n_check", type=int, default=48, help="alignment sample")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default=None)
    args = p.parse_args()
    resized_root = Path(resolve_under_home(args.resized))
    master_root = Path(resolve_under_home(args.masters))

    rows = []
    missing = 0
    for f in sorted(resized_root.rglob("*.png")):
        m = master_of(f, resized_root, master_root)
        if m is None:
            missing += 1
            continue
        with Image.open(f) as r:
            bw, bh = r.size
            margins = normalize_crop_margins(r.info.get(MARGINS_KEY))
        with Image.open(m) as src:
            w, h = oriented_size(src)
            fmt = (src.format or "").lower()
        box = margin_box(w, h, margins)
        w, h = box[2] - box[0], box[3] - box[1]
        scale = 2 * (bh / h if w / h > bw / bh else bw / w)
        rows.append(
            {
                "resized": str(f.relative_to(resized_root)),
                "master": str(m.relative_to(master_root)),
                "format": fmt,
                "bucket": [bw, bh],
                "scale_2x": scale,
            }
        )
    scales = np.array([r["scale_2x"] for r in rows])
    fmts = Counter(r["format"] for r in rows)
    native = [r for r in rows if r["scale_2x"] <= 1.0]
    print(
        f"{len(rows)} resized images with a master ({missing} without); the 2× target "
        f"is a downscale of the master (scale ≤ 1) on {len(native)}, ≤ 1.05 on "
        f"{int((scales <= 1.05).sum())}, ≤ 1.25 on {int((scales <= 1.25).sum())}",
        flush=True,
    )
    print(
        f"master formats: {dict(fmts)}; native set: "
        f"{dict(Counter(r['format'] for r in native))}",
        flush=True,
    )

    rng = random.Random(args.seed)
    sample = rng.sample(native, min(args.n_check, len(native)))
    checks = []
    for r in sample:
        with Image.open(resized_root / r["resized"]) as im:
            res = im.convert("RGB")
            t2, _ = target_2x(master_root / r["master"], im)
        a = np.asarray(t2, dtype=np.float64)
        box = a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2, 3).mean((1, 3))
        lz = np.asarray(t2.resize(res.size, Image.Resampling.LANCZOS))
        checks.append(
            {
                "resized": r["resized"],
                "psnr_box": psnr8(box, np.asarray(res)),
                "psnr_lanczos": psnr8(lz, np.asarray(res)),
            }
        )
    pb = np.array([c["psnr_box"] for c in checks])
    pl = np.array([c["psnr_lanczos"] for c in checks])
    print(
        f"alignment, {len(checks)} images: 2× target → ×1 against the resized image, "
        f"box mean {pb.mean():.2f} dB (min {pb.min():.2f}), LANCZOS {pl.mean():.2f} dB "
        f"(min {pl.min():.2f})",
        flush=True,
    )

    run_dir = make_run_dir("pixel_readout", label=args.label)
    (run_dir / "manifest.json").write_text(json.dumps(rows))
    out = write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={
            "n_with_master": len(rows),
            "n_without_master": missing,
            "n_native_2x": len(native),
            "n_scale_le_1_05": int((scales <= 1.05).sum()),
            "n_scale_le_1_25": int((scales <= 1.25).sum()),
            "formats": dict(fmts),
            "formats_native": dict(Counter(r["format"] for r in native)),
            "align_psnr_box_mean": float(pb.mean()),
            "align_psnr_box_min": float(pb.min()),
            "align_psnr_lanczos_mean": float(pl.mean()),
            "align_psnr_lanczos_min": float(pl.min()),
            "align_checks": checks,
        },
        label=args.label,
        artifacts=["manifest.json"],
        device="cpu",
    )
    print(f"envelope: {out}")


if __name__ == "__main__":
    main()
