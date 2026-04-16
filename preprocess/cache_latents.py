#!/usr/bin/env python3
"""Cache VAE latents for all images in a dataset directory.

Encodes images through the Qwen Image VAE and saves latent caches (.npz)
alongside the images.  Skips already-cached entries (idempotent).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library.cache_utils import LATENT_CACHE_SUFFIX
from library.datasets.image_utils import IMAGE_EXTENSIONS, IMAGE_TRANSFORMS


def get_latents_npz_path(image_path: Path, image_size: tuple[int, int]) -> Path:
    """Match the naming convention used by AnimaLatentsCachingStrategy."""
    return image_path.with_name(
        f"{image_path.stem}_{image_size[0]:04d}x{image_size[1]:04d}{LATENT_CACHE_SUFFIX}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--vae", type=str, required=True, help="Path to VAE weights")
    parser.add_argument(
        "--batch_size", type=int, default=4, help="VAE encoding batch size (default: 4)"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=64,
        help="VAE spatial chunk size (default: 64)",
    )
    parser.add_argument(
        "--disable_cache",
        action="store_true",
        default=True,
        help="Disable VAE internal cache (default: True)",
    )
    args = parser.parse_args()

    from library import qwen_image_autoencoder_kl

    data_dir = Path(args.dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Loading VAE from {args.vae} ...")
    vae = qwen_image_autoencoder_kl.load_vae(
        args.vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=args.chunk_size,
        disable_cache=args.disable_cache,
    )
    vae.to(device, dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()

    # Collect images grouped by resolution for efficient batching
    image_files = sorted(
        p for p in data_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )

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
        for batch_start in range(0, len(paths), args.batch_size):
            batch_paths = paths[batch_start : batch_start + args.batch_size]
            tensors = []

            for p in batch_paths:
                npz_path = get_latents_npz_path(p, (w, h))
                if npz_path.exists():
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
                kwargs[f"crop_ltrb{key_reso_suffix}"] = np.array(
                    [0, 0, size[0], size[1]]
                )

                np.savez(npz_path, **kwargs)

                cached += 1
                pbar.update(1)
                pbar.set_postfix_str(f"{p.name} → {size[0]}x{size[1]}")

    pbar.close()
    print(
        f"\nLatent caching complete: {cached} cached, {skipped} skipped (already existed)"
    )

    vae.to("cpu")
    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
