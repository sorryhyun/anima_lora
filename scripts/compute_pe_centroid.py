#!/usr/bin/env python
"""Compute the dataset-mean PE feature centroid for IP-Adapter.

Walks a directory of cached PE-Core features (``{stem}_anima_{encoder}.safetensors``
produced by ``preprocess/cache_pe_encoder.py``), mean-pools each over the
patch-token axis (dropping CLS), then averages across the dataset to produce
a single ``[encoder_dim]`` centroid. Saved as a small sidecar safetensors
loaded by ``IPAdapterNetwork.load_centroid_from_file`` at training start.

Why: bench/ip_adapter/analysis.md measured a participation-ratio-6 collapse
on this dataset's pooled PE features (cross-pair sim ~0.69) — i.e. most of
the per-feature variance is shared across the dataset. Subtracting the
centroid before the resampler hands it the per-image delta directly,
instead of forcing it to learn the discriminative direction in the presence
of a large shared-aesthetic background.

Usage::

    uv run python scripts/compute_pe_centroid.py
    uv run python scripts/compute_pe_centroid.py --cache_dir post_image_dataset/lora
    uv run python scripts/compute_pe_centroid.py --encoder pe --out path/to/centroid.safetensors

Output: ``post_image_dataset/lora/anima_pe_centroid_{encoder}.safetensors`` by
default (one tensor under key ``"centroid"`` of dtype fp32, shape ``[D]``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]


def _pool_pe(feats: torch.Tensor, *, drop_cls: bool = True) -> torch.Tensor:
    """Mean-over-patch-tokens pool. ``feats`` is ``[T, D]``; returns ``[D]``."""
    if drop_cls and feats.shape[0] > 1:
        feats = feats[1:]
    return feats.mean(dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--cache_dir", type=str, default="post_image_dataset/lora",
        help="Directory of cached PE features (default: post_image_dataset/lora).",
    )
    parser.add_argument(
        "--encoder", type=str, default="pe",
        help="Encoder name; matches the suffix on cached files "
        "({stem}_anima_{encoder}.safetensors). Default 'pe'.",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output path for the centroid sidecar. Defaults to "
        "post_image_dataset/ip_adapter/anima_pe_centroid_{encoder}.safetensors "
        "(separate from the shared PE cache dir so LoRA stays untouched).",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Optionally cap the number of files used (0 = use all).",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = (ROOT / cache_dir).resolve()
    if not cache_dir.is_dir():
        print(f"cache_dir not found: {cache_dir}", file=sys.stderr)
        sys.exit(1)

    suffix = f"_anima_{args.encoder}.safetensors"
    files = sorted(p for p in cache_dir.iterdir() if p.name.endswith(suffix))
    # Exclude the centroid sidecar itself if a previous run wrote it here.
    files = [p for p in files if not p.name.startswith("anima_pe_centroid")]
    if not files:
        print(f"No '{suffix}' caches under {cache_dir}", file=sys.stderr)
        sys.exit(1)
    if args.limit > 0:
        files = files[: args.limit]

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = (
            ROOT
            / "post_image_dataset"
            / "ip_adapter"
            / f"anima_pe_centroid_{args.encoder}.safetensors"
        )

    print(f"cache_dir: {cache_dir}")
    print(f"files:     {len(files)}")
    print(f"out:       {out_path}")

    # Streaming mean: avoid stacking [N, D] in memory (N=2407 × D=1024 is fine
    # but we prefer to be cheap on really big datasets too).
    centroid: torch.Tensor | None = None
    n = 0
    for p in tqdm(files, desc="pooling"):
        sd = load_file(str(p))
        feats = sd.get("image_features")
        if feats is None:
            print(f"  skip {p.name}: no 'image_features' key", file=sys.stderr)
            continue
        pool = _pool_pe(feats.to(torch.float32))
        if centroid is None:
            centroid = torch.zeros_like(pool)
        centroid += pool
        n += 1

    if n == 0 or centroid is None:
        print("No usable PE features found.", file=sys.stderr)
        sys.exit(1)
    centroid = centroid / n

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"centroid": centroid.contiguous()},
        str(out_path),
        metadata={
            "encoder": args.encoder,
            "n_images": str(n),
            "d_enc": str(centroid.shape[0]),
            "pool": "mean_over_patch_tokens",
        },
    )

    print(
        f"\ncentroid shape: {tuple(centroid.shape)}  "
        f"‖centroid‖={float(centroid.norm()):.3f}  "
        f"mean(centroid)={float(centroid.mean()):.4f}  "
        f"std(centroid)={float(centroid.std()):.4f}"
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
