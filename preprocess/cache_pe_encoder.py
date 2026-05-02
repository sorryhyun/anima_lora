#!/usr/bin/env python3
"""Cache PE-Core (or other registered vision-encoder) features.

Mirrors the live PE encoding done at training time so callers can read
patch-token features off disk instead of running the encoder every step.
Loads each pre-resized image from ``--dir`` in [-1, 1], picks the
encoder's nearest-aspect bucket, runs a single forward, and saves
``{stem}_anima_{encoder}.safetensors`` into ``--cache_dir`` (or alongside
the image when omitted). Skips already-cached entries (idempotent).

Wrapped by ``make preprocess-pe`` (reads ``post_image_dataset/resized/``,
writes ``post_image_dataset/lora/``). The same sidecars are consumed by the
LoRA / REPA pipeline and by IP-Adapter — they share the cache directory.

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
from torch.utils.data import DataLoader, Dataset
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


class _PEImageGroup(Dataset):
    """Reads images from one ``(W, H)`` resolution group.

    Each ``__getitem__`` returns ``(str_path, str_out_path, [3, H, W] tensor in
    [-1, 1])`` so the main thread can write safetensors in batch order without
    holding the PIL.Image object across the worker boundary. We pass paths as
    strings (instead of ``Path``) because ``Path`` is picklable but heavier;
    safetensors' ``save_file`` takes a string anyway.
    """

    def __init__(self, paths: list[Path], out_paths: list[Path]):
        self._paths = [str(p) for p in paths]
        self._out_paths = [str(p) for p in out_paths]

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int):
        p = self._paths[idx]
        with Image.open(p) as img:
            tensor = IMAGE_TRANSFORMS(np.array(img.convert("RGB")))
        return p, self._out_paths[idx], tensor


def _collate(batch):
    """Stack tensors into ``[B, 3, H, W]``; group already guarantees same shape."""
    paths, out_paths, tensors = zip(*batch)
    return list(paths), list(out_paths), torch.stack(tensors, dim=0)


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
        "--num_workers",
        type=int,
        default=4,
        help=(
            "DataLoader workers for parallel PIL decode + transform. "
            "0 = single-threaded (decode on the main thread, GPU sits idle "
            "during decode + safetensors write). Default 4."
        ),
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

    # Pre-skip cached files so workers never decode them. The header read in
    # PIL.Image.open is cheap but adds up at 100k+ images; we still do it
    # below for grouping, but only on uncached entries.
    pending: list[Path] = []
    skipped = 0
    for p in image_files:
        if cache_path_for(p, bundle.name, cache_dir=cache_dir).exists():
            skipped += 1
        else:
            pending.append(p)

    reso_groups: dict[tuple[int, int], list[Path]] = {}
    for p in pending:
        with Image.open(p) as img:
            size = img.size  # (W, H)
        reso_groups.setdefault(size, []).append(p)

    cached = 0

    metadata = {
        "encoder": bundle.name,
        "d_enc": str(bundle.d_enc),
        "patch": str(bundle.bucket_spec.patch),
    }

    pbar = tqdm(
        total=len(pending),
        desc=f"Caching {bundle.name} features",
    )
    for (w, h), paths in reso_groups.items():
        out_paths = [
            cache_path_for(p, bundle.name, cache_dir=cache_dir) for p in paths
        ]
        ds = _PEImageGroup(paths, out_paths)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=_collate,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0 and len(paths) > args.batch_size),
        )
        for batch_paths, batch_out_paths, img_batch in loader:
            with torch.no_grad():
                feats_list = encode_pe_from_imageminus1to1(
                    bundle, img_batch, same_bucket=True
                )
            for src, dst, feats in zip(batch_paths, batch_out_paths, feats_list):
                save_dict = {
                    "image_features": feats.detach()
                    .to(save_dtype)
                    .cpu()
                    .contiguous()
                }
                _save_safetensors(save_dict, dst, metadata=metadata)
                cached += 1
                pbar.update(1)
                pbar.set_postfix_str(f"{Path(src).name} → T={feats.shape[0]}")

    pbar.close()
    print(
        f"\n{bundle.name} feature caching complete: {cached} cached, {skipped} skipped"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
