#!/usr/bin/env python3
"""Cache PE-Core (or other registered vision-encoder) features for IP-Adapter.

Mirrors the live PE encoding done by ``train.py:_maybe_set_ip_tokens`` so that
``make ip-adapter`` can read patch-token features off disk instead of running
the encoder every step. Loads each pre-resized image from ``post_image_dataset/``
in [-1, 1], picks the encoder's nearest-aspect bucket, runs a single forward,
and saves ``{stem}_anima_{encoder}.safetensors`` alongside the image. Skips
already-cached entries (idempotent).

The cache key matches what the encoder produces at training time:
``encode_pe_from_imageminus1to1(bundle, x, same_bucket=True)`` -> ``[T_pe, d_enc]``.
Variable T per encoder bucket; per-image stored as a single tensor (no padding).
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

from library.datasets.image_utils import IMAGE_EXTENSIONS, IMAGE_TRANSFORMS
from library.vision.encoder import encode_pe_from_imageminus1to1, load_pe_encoder


def cache_path_for(
    image_path: Path, encoder: str, cache_dir: Path | None = None
) -> Path:
    name = f"{image_path.stem}_anima_{encoder}.safetensors"
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / name
    return image_path.with_name(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=str, required=True, help="Dataset directory")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help=(
            "Optional directory to write PE caches into (created if needed). "
            "Defaults to writing alongside each source image."
        ),
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="pe",
        help="Vision encoder registry name (default: pe). See library/vision/encoders.py.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Override the encoder's default model id / checkpoint path.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Forward batch size within each (H, W) group (default: 8).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Storage dtype for cached features (default: bfloat16, matches train-time).",
    )
    args = parser.parse_args()

    from safetensors.torch import save_file as _save_safetensors

    data_dir = Path(args.dir)
    if not data_dir.is_dir():
        print(f"--dir not found: {data_dir}", file=sys.stderr)
        sys.exit(1)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"Loading vision encoder '{args.encoder}' on {device} ...")
    bundle = load_pe_encoder(device, name=args.encoder, model_id=args.model_id)
    print(
        f"  encoder={bundle.name} d_enc={bundle.d_enc} "
        f"patch={bundle.bucket_spec.patch} cls={bundle.bucket_spec.use_cls}"
    )

    # Group images by their post-resize pixel dimensions so a single forward
    # serves the whole group (same encoder bucket -> same T_pe -> same shape).
    image_files = sorted(
        p for p in data_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        print(f"No images found in {data_dir}/", file=sys.stderr)
        sys.exit(1)

    reso_groups: dict[tuple[int, int], list[Path]] = {}
    for p in image_files:
        with Image.open(p) as img:
            size = img.size  # (W, H)
        reso_groups.setdefault(size, []).append(p)

    total = len(image_files)
    cached = 0
    skipped = 0

    metadata = {
        "encoder": bundle.name,
        "d_enc": str(bundle.d_enc),
        "patch": str(bundle.bucket_spec.patch),
    }

    pbar = tqdm(total=total, desc=f"Caching {bundle.name} features")
    for (w, h), paths in reso_groups.items():
        for batch_start in range(0, len(paths), args.batch_size):
            batch_paths = paths[batch_start : batch_start + args.batch_size]

            to_encode: list[tuple[Path, torch.Tensor, Path]] = []
            for p in batch_paths:
                out_path = cache_path_for(p, bundle.name, cache_dir=cache_dir)
                if out_path.exists():
                    skipped += 1
                    pbar.update(1)
                    pbar.set_postfix_str(f"skip {p.name}")
                    continue
                img = Image.open(p).convert("RGB")
                img_tensor = IMAGE_TRANSFORMS(np.array(img))  # [3, H, W] in [-1, 1]
                img.close()
                to_encode.append((p, img_tensor, out_path))

            if not to_encode:
                continue

            img_batch = torch.stack([t[1] for t in to_encode], dim=0)  # [B, 3, H, W]
            with torch.no_grad():
                feats_list = encode_pe_from_imageminus1to1(
                    bundle, img_batch, same_bucket=True
                )

            for (p, _, out_path), feats in zip(to_encode, feats_list):
                save_dict = {"image_features": feats.detach().to(save_dtype).cpu().contiguous()}
                _save_safetensors(save_dict, str(out_path), metadata=metadata)
                cached += 1
                pbar.update(1)
                pbar.set_postfix_str(f"{p.name} → T={feats.shape[0]}")

    pbar.close()
    print(
        f"\n{bundle.name} feature caching complete: {cached} cached, {skipped} skipped"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
