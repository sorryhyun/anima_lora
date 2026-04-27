#!/usr/bin/env python3
"""Resize training images to constant-token bucket resolutions.

Reads images from a source directory, resizes and center-crops them to the
nearest bucket resolution, writes the results plus caption sidecars to an
output directory.
"""

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library.datasets.buckets import BucketManager
from library.datasets.image_utils import IMAGE_EXTENSIONS
CAPTION_EXTENSIONS = {".txt", ".caption"}


def process_image(
    image_path: Path,
    out_dir: Path,
    bucket_args: tuple,
) -> tuple[str, tuple[int, int]]:
    """Worker function — receives bucket params instead of BucketManager to be picklable."""
    max_reso, min_size, max_size, reso_steps, use_constant = bucket_args
    bucket_mgr = BucketManager(
        no_upscale=False,
        max_reso=max_reso,
        min_size=min_size,
        max_size=max_size,
        reso_steps=reso_steps,
    )
    bucket_mgr.make_buckets(constant_token_buckets=use_constant)

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    bucket_reso, _, _ = bucket_mgr.select_bucket(w, h)
    bw, bh = bucket_reso

    # Resize preserving aspect ratio so the image covers the bucket
    ar_img = w / h
    ar_bucket = bw / bh
    if ar_img > ar_bucket:
        new_h = bh
        new_w = round(bh * ar_img)
    else:
        new_w = bw
        new_h = round(bw / ar_img)

    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Center crop to bucket resolution
    left = (new_w - bw) // 2
    top = (new_h - bh) // 2
    img = img.crop((left, top, left + bw, top + bh))

    out_path = out_dir / f"{image_path.stem}.png"
    img.save(out_path, format="PNG")

    # Copy caption sidecars
    for ext in CAPTION_EXTENSIONS:
        cap = image_path.with_suffix(ext)
        if cap.exists():
            shutil.copy2(cap, out_dir / f"{image_path.stem}{ext}")

    return image_path.name, bucket_reso


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=str, required=True, help="Source image directory")
    parser.add_argument("--dst", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--resolution", type=int, default=1024, help="Max resolution (default: 1024)"
    )
    parser.add_argument(
        "--min_bucket_reso",
        type=int,
        default=512,
        help="Min bucket size (default: 512)",
    )
    parser.add_argument(
        "--max_bucket_reso",
        type=int,
        default=2048,
        help="Max bucket size (default: 2048)",
    )
    parser.add_argument(
        "--bucket_reso_steps",
        type=int,
        default=64,
        help="Bucket step size (default: 64)",
    )
    parser.add_argument(
        "--constant_token_buckets",
        action="store_true",
        default=True,
        help="Use constant-token buckets (default: True)",
    )
    parser.add_argument(
        "--no_constant_token_buckets",
        action="store_true",
        help="Disable constant-token buckets",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=500_000,
        help="Skip images with fewer than this many pixels (default: 500_000 = 0.5MP). "
        "Set to 0 to disable.",
    )
    args = parser.parse_args()

    if args.no_constant_token_buckets:
        args.constant_token_buckets = False

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    use_constant = args.constant_token_buckets
    bucket_args = (
        (args.resolution, args.resolution),
        args.min_bucket_reso,
        args.max_bucket_reso,
        args.bucket_reso_steps,
        use_constant,
    )

    image_files = sorted(
        p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if args.min_pixels > 0:
        kept: list[Path] = []
        skipped: list[tuple[Path, int, int]] = []
        for p in image_files:
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception as e:
                print(f"  warn: could not read {p.name}: {e}")
                continue
            if w * h < args.min_pixels:
                skipped.append((p, w, h))
            else:
                kept.append(p)
        if skipped:
            print(
                f"Skipping {len(skipped)} images below {args.min_pixels:,} pixels "
                f"({args.min_pixels / 1e6:.2f}MP):"
            )
            for p, w, h in skipped:
                print(f"  {p.name}  {w}x{h}  ({w * h / 1e6:.3f}MP)")
        image_files = kept

    print(
        f"Resizing {len(image_files)} images to "
        f"{'constant-token' if use_constant else 'standard'} buckets"
    )
    bucket_counts: dict[tuple[int, int], int] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_image, img_path, dst, bucket_args): img_path
            for img_path in image_files
        }
        pbar = tqdm(as_completed(futures), total=len(futures), desc="Resizing")
        for future in pbar:
            name, reso = future.result()
            bucket_counts[reso] = bucket_counts.get(reso, 0) + 1
            pbar.set_postfix_str(f"{name} → {reso[0]}x{reso[1]}")

    print("\nBucket distribution:")
    for reso in sorted(bucket_counts):
        tokens = (reso[0] // 16) * (reso[1] // 16)
        print(
            f"  {reso[0]:>4d}x{reso[1]:<4d}: {bucket_counts[reso]:>3d} images  ({tokens} tokens)"
        )


if __name__ == "__main__":
    main()
