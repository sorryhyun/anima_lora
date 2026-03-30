#!/usr/bin/env python3
"""Pre-process training images to bucket resolutions.

Reads images from a source directory, resizes and center-crops them to the
nearest bucket resolution, and writes the results (plus caption sidecars)
to an output directory.  This lets you inspect exactly what the trainer
will see and avoids repeated on-the-fly resizing.
"""

import argparse
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library.datasets.buckets import BucketManager

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
CAPTION_EXTENSIONS = {".txt", ".caption"}


def process_image(
    image_path: Path,
    out_dir: Path,
    bucket_args: tuple[tuple[int, int], int, int, int],
) -> tuple[str, tuple[int, int]]:
    """Worker function — receives bucket params instead of BucketManager to be picklable."""
    max_reso, min_size, max_size, reso_steps = bucket_args
    bucket_mgr = BucketManager(
        no_upscale=False,
        max_reso=max_reso,
        min_size=min_size,
        max_size=max_size,
        reso_steps=reso_steps,
    )
    bucket_mgr.make_buckets()
    return _process_image(image_path, out_dir, bucket_mgr)


def _process_image(
    image_path: Path,
    out_dir: Path,
    bucket_mgr: BucketManager,
) -> tuple[str, tuple[int, int]]:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    bucket_reso, _, _ = bucket_mgr.select_bucket(w, h)
    bw, bh = bucket_reso

    # Resize preserving aspect ratio so the image covers the bucket
    ar_img = w / h
    ar_bucket = bw / bh
    if ar_img > ar_bucket:
        # Image is wider → fit height
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
        default=1536,
        help="Max bucket size (default: 1536)",
    )
    parser.add_argument(
        "--bucket_reso_steps",
        type=int,
        default=128,
        help="Bucket step size (default: 64)",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers (default: 4)"
    )
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    bucket_args = (
        (args.resolution, args.resolution),
        args.min_bucket_reso,
        args.max_bucket_reso,
        args.bucket_reso_steps,
    )

    image_files = sorted(
        p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    bucket_counts: dict[tuple[int, int], int] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_image, img_path, dst, bucket_args): img_path
            for img_path in image_files
        }
        pbar = tqdm(as_completed(futures), total=len(futures), desc="Preprocessing")
        for future in pbar:
            name, reso = future.result()
            bucket_counts[reso] = bucket_counts.get(reso, 0) + 1
            pbar.set_postfix_str(f"{name} → {reso[0]}x{reso[1]}")

    print("\nBucket distribution:")
    for reso in sorted(bucket_counts):
        print(f"  {reso[0]:>4d}x{reso[1]:<4d}: {bucket_counts[reso]}")


if __name__ == "__main__":
    main()
