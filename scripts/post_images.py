#!/usr/bin/env python3
"""Pre-process training images: resize to constant-token buckets and cache VAE latents.

Reads images from a source directory, resizes and center-crops them to the
nearest bucket resolution (constant-token buckets by default), writes the
results plus caption sidecars to an output directory, then encodes all images
through the VAE and saves latent caches (.npz) to disk.
"""

import argparse
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library.datasets.buckets import CONSTANT_TOKEN_BUCKETS, BucketManager

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
CAPTION_EXTENSIONS = {".txt", ".caption"}
ANIMA_LATENTS_NPZ_SUFFIX = "_anima.npz"

IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


# ---------------------------------------------------------------------------
# Phase 1: Resize images to bucket resolutions (CPU, parallel)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Phase 2: VAE latent caching (GPU, sequential)
# ---------------------------------------------------------------------------


def get_latents_npz_path(image_path: Path, image_size: tuple[int, int]) -> Path:
    """Match the naming convention used by AnimaLatentsCachingStrategy."""
    return image_path.with_name(
        f"{image_path.stem}_{image_size[0]:04d}x{image_size[1]:04d}{ANIMA_LATENTS_NPZ_SUFFIX}"
    )


def cache_latents(
    out_dir: Path,
    vae_path: str,
    vae_batch_size: int,
    vae_chunk_size: int | None,
    vae_disable_cache: bool,
) -> None:
    """Encode all images in out_dir through the VAE and save .npz latent caches."""
    from library import qwen_image_autoencoder_kl

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"\nLoading VAE from {vae_path} ...")
    vae = qwen_image_autoencoder_kl.load_vae(
        vae_path,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=vae_chunk_size,
        disable_cache=vae_disable_cache,
    )
    vae.to(device, dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()

    # Collect images grouped by resolution for efficient batching
    image_files = sorted(p for p in out_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

    # Group by resolution
    reso_groups: dict[tuple[int, int], list[Path]] = {}
    for p in image_files:
        img = Image.open(p)
        size = img.size  # (W, H)
        img.close()
        reso_groups.setdefault(size, []).append(p)

    total = len(image_files)
    cached = 0
    skipped = 0

    pbar = tqdm(total=total, desc="Caching latents")
    for (w, h), paths in reso_groups.items():
        # Process in batches of the same resolution
        for batch_start in range(0, len(paths), vae_batch_size):
            batch_paths = paths[batch_start : batch_start + vae_batch_size]
            tensors = []

            for p in batch_paths:
                npz_path = get_latents_npz_path(p, (w, h))
                if npz_path.exists():
                    # Check if already cached with correct key
                    latents_size = (h // 8, w // 8)
                    key = f"latents_{latents_size[0]}x{latents_size[1]}"
                    try:
                        npz = np.load(npz_path)
                        if key in npz:
                            skipped += 1
                            pbar.update(1)
                            pbar.set_postfix_str(f"skip {p.name}")
                            continue
                    except Exception:
                        pass

                img = Image.open(p).convert("RGB")
                img_np = np.array(img)
                img_tensor = IMAGE_TRANSFORMS(img_np)
                tensors.append((p, img_tensor, (w, h)))

            if not tensors:
                continue

            img_batch = torch.stack([t[1] for t in tensors], dim=0)
            img_batch = img_batch.to(device=device, dtype=dtype)

            with torch.no_grad():
                latents = vae.encode_pixels_to_latents(img_batch).cpu()

            for i, (p, _, size) in enumerate(tensors):
                lat = latents[i]  # (16, H/8, W/8)
                latents_size = lat.shape[-2:]  # H/8, W/8
                key_reso_suffix = f"_{latents_size[0]}x{latents_size[1]}"

                npz_path = get_latents_npz_path(p, size)
                kwargs = {}
                if npz_path.exists():
                    npz = np.load(npz_path)
                    for key in npz.files:
                        kwargs[key] = npz[key]

                kwargs[f"latents{key_reso_suffix}"] = lat.float().numpy()
                kwargs[f"original_size{key_reso_suffix}"] = np.array(list(size))
                # Pre-cropped images: crop covers the full image
                kwargs[f"crop_ltrb{key_reso_suffix}"] = np.array([0, 0, size[0], size[1]])

                np.savez(npz_path, **kwargs)

                cached += 1
                pbar.update(1)
                pbar.set_postfix_str(f"{p.name} → {size[0]}x{size[1]}")

    pbar.close()
    print(f"\nLatent caching complete: {cached} cached, {skipped} skipped (already existed)")

    # Cleanup
    vae.to("cpu")
    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
    # VAE latent caching
    parser.add_argument(
        "--vae", type=str, default=None, help="Path to VAE weights for latent caching"
    )
    parser.add_argument(
        "--vae_batch_size", type=int, default=4, help="VAE encoding batch size (default: 4)"
    )
    parser.add_argument(
        "--vae_chunk_size", type=int, default=64, help="VAE spatial chunk size (default: 64)"
    )
    parser.add_argument(
        "--vae_disable_cache",
        action="store_true",
        default=True,
        help="Disable VAE internal cache (default: True)",
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

    # Phase 1: Resize images
    print(f"Phase 1: Resizing {len(image_files)} images to {'constant-token' if use_constant else 'standard'} buckets")
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
        print(f"  {reso[0]:>4d}x{reso[1]:<4d}: {bucket_counts[reso]:>3d} images  ({tokens} tokens)")

    # Phase 2: VAE latent caching
    if args.vae:
        print(f"\nPhase 2: VAE latent caching")
        cache_latents(
            dst,
            args.vae,
            args.vae_batch_size,
            args.vae_chunk_size,
            args.vae_disable_cache,
        )
    else:
        print("\nSkipping VAE latent caching (no --vae specified)")


if __name__ == "__main__":
    main()
