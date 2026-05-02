#!/usr/bin/env python
"""PE feature analysis — is IP-Adapter's data wall fundamental or fixable?

Three feature-only diagnostics on the cached PE-Core features in
``post_image_dataset/lora/`` plus live-encoded augmented variants. The goal is
to decide *before* spending more training budget whether the IP-Adapter wall
is a data problem (need paired-but-different references / aug) or an
optimization problem (gate not opening, LR, etc.). These measurements don't
require IP-Adapter training — they probe the PE features themselves.

Hypothesis under test:
    With self-paired training (ref == target), PE features carry enough
    surface-level information that the resampler + per-block KV can solve
    the loss by lookup rather than by learning appearance. If true, no amount
    of optimization tuning will help — only paired-but-different data will.

Three measurements:

1. **Aug-invariance histogram.** For N random images, compute pooled PE for
   each + hflip / center-crop / color-jitter variants + a random *other*
   image. Tells you whether aug-pairs cluster near 1.0 (cos) or collapse
   toward the cross-pair distribution. The gap is the "aug-invariant signal"
   that the resampler could in principle extract.

2. **Crop retrieval rank.** From a 60% random crop of a query image, how
   well does pooled cosine sim retrieve the original from the full cache?
   recall@1 ≈ 100% means PE features fingerprint the source pixels —
   the IP path can always recover the target, so self-paired training has
   nothing to learn beyond memorization.

3. **Effective rank of pooled feature distribution.** SVD on the (N × D)
   pooled-feature matrix. K=16 resampler tokens × D dims is the IP path's
   capacity; if the dataset's pooled features live in a sub-space much
   smaller than that, capacity isn't the bottleneck.

All three pool patch tokens via mean (CLS dropped). The resampler attends to
all tokens, not the mean — so mean-pool similarity is a *lower bound* on the
resampler's distinguishability. Strong collapse at the mean-pool level is a
strong signal; high mean-pool separation says "the global signal is fine"
but doesn't rule out finer-grained issues.

Usage::

    uv run python bench/ip_adapter/pe_feature_analysis.py             # all 3
    uv run python bench/ip_adapter/pe_feature_analysis.py --steps 1,3 # skip retrieval
    uv run python bench/ip_adapter/pe_feature_analysis.py --n 100     # smaller sample
    uv run python bench/ip_adapter/pe_feature_analysis.py --resized_dir ... --cache_dir ...

Outputs a text report to stdout and (optional) JSON to ``--report``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file as load_safetensors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from library.datasets.image_utils import IMAGE_EXTENSIONS, IMAGE_TRANSFORMS
from library.vision.encoder import encode_pe_from_imageminus1to1, load_pe_encoder


# ----- pooling -----

def _pool_pe(feats: torch.Tensor, *, drop_cls: bool = True) -> torch.Tensor:
    """Mean-over-tokens pool. ``feats`` is ``[T, D]``; returns ``[D]``."""
    if drop_cls and feats.shape[0] > 1:
        feats = feats[1:]
    return feats.mean(dim=0)


def _l2norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))


# ----- cached-feature loading -----

def _load_cached_pool(path: Path) -> torch.Tensor:
    """Read one ``*_anima_pe.safetensors`` and return mean-pooled ``[D]`` (fp32)."""
    sd = load_safetensors(str(path))
    feats = sd["image_features"].to(torch.float32)
    return _pool_pe(feats)


def _load_all_cached_pools(cache_dir: Path) -> tuple[list[Path], torch.Tensor]:
    """Load every cached PE file and stack pooled vectors into ``[N, D]``."""
    files = sorted(cache_dir.glob("*_anima_pe.safetensors"))
    if not files:
        raise FileNotFoundError(f"No PE caches found under {cache_dir}")
    pools = [_load_cached_pool(p) for p in files]
    return files, torch.stack(pools, dim=0)


# ----- live encoding for augmented variants -----

def _read_image_minus1to1(path: Path) -> torch.Tensor:
    """Same path as preprocess/cache_pe_encoder.py — IMAGE_TRANSFORMS gives [-1, 1]."""
    with Image.open(path) as img:
        return IMAGE_TRANSFORMS(np.array(img.convert("RGB")))


def _hflip(x: torch.Tensor) -> torch.Tensor:
    """Horizontal flip on a CHW tensor."""
    return torch.flip(x, dims=[-1])


def _random_crop_resize(x: torch.Tensor, frac: float, *, rng: random.Random) -> torch.Tensor:
    """Random crop of side fraction ``frac`` then resize back to original HW."""
    _, H, W = x.shape
    crop_h = max(1, int(round(H * frac)))
    crop_w = max(1, int(round(W * frac)))
    top = rng.randint(0, H - crop_h)
    left = rng.randint(0, W - crop_w)
    cropped = x[:, top:top + crop_h, left:left + crop_w]
    return F.interpolate(
        cropped.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
    )[0]


def _color_jitter(
    x: torch.Tensor, *, brightness: float, contrast: float, saturation: float,
    rng: random.Random,
) -> torch.Tensor:
    """Cheap brightness/contrast/saturation jitter on a [-1, 1] CHW tensor.

    Operates in [0, 1] then maps back. Scale factors uniform in
    ``[1-amount, 1+amount]``.
    """
    img01 = (x.clamp(-1, 1) + 1.0) * 0.5
    b = 1.0 + rng.uniform(-brightness, brightness)
    c = 1.0 + rng.uniform(-contrast, contrast)
    s = 1.0 + rng.uniform(-saturation, saturation)
    img01 = img01 * b
    mean_pix = img01.mean(dim=(-2, -1), keepdim=True)
    img01 = (img01 - mean_pix) * c + mean_pix
    gray = img01.mean(dim=0, keepdim=True)
    img01 = (img01 - gray) * s + gray
    img01 = img01.clamp(0, 1)
    return img01 * 2.0 - 1.0


def _encode_pooled_batch(
    bundle, images: list[torch.Tensor], *, device: torch.device,
) -> torch.Tensor:
    """Encode a list of variable-shape ``[3, H, W]`` images, mean-pool each.

    Returns ``[N, D]`` fp32. Runs one-at-a-time (PE-Core supports dynamic
    resolution and our images may differ in HW from each other).
    """
    pooled = []
    for img in images:
        batch = img.unsqueeze(0).to(device)
        feats_list = encode_pe_from_imageminus1to1(bundle, batch, same_bucket=True)
        pooled.append(_pool_pe(feats_list[0]).to(torch.float32).cpu())
    return torch.stack(pooled, dim=0)


# ----- step 1: aug-invariance histogram -----

def step_aug_histogram(
    args, *, image_paths: list[Path], cached_pools: torch.Tensor, all_files: list[Path],
) -> dict:
    """Live-encode N images + 3 aug variants, compare cos-sims to cross pairs.

    Cross pair = sample's pooled feature vs a random *other* file's cached pool.
    """
    rng = random.Random(args.seed)
    N = min(args.n, len(image_paths))
    sampled = rng.sample(image_paths, N)
    print(f"\n=== Step 1: aug-invariance histogram (N={N}) ===")

    print("  Loading PE encoder...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_pe_encoder(device, name="pe")

    # Live encode original + augs. We re-encode the original (instead of using
    # the cached pool) to ensure the original/aug comparison is apples-to-apples
    # — same pooling, same dtype, same encoder run.
    originals: list[torch.Tensor] = []
    hflips: list[torch.Tensor] = []
    crops: list[torch.Tensor] = []
    jitters: list[torch.Tensor] = []
    for p in sampled:
        x = _read_image_minus1to1(p)
        originals.append(x)
        hflips.append(_hflip(x))
        crops.append(_random_crop_resize(x, frac=args.crop_frac, rng=rng))
        jitters.append(
            _color_jitter(
                x, brightness=args.jitter, contrast=args.jitter,
                saturation=args.jitter, rng=rng,
            )
        )

    print(f"  Encoding {4 * N} images (orig + hflip + crop + jitter)...")
    pe_orig = _encode_pooled_batch(bundle, originals, device=device)
    pe_hflip = _encode_pooled_batch(bundle, hflips, device=device)
    pe_crop = _encode_pooled_batch(bundle, crops, device=device)
    pe_jitter = _encode_pooled_batch(bundle, jitters, device=device)

    # Build path -> cached-index map for the cross-pair sampling.
    file_to_idx = {p.stem: i for i, p in enumerate(all_files)}
    # Stems of the cached files include the `_anima_pe` suffix removal step; map
    # by source-image stem (everything before "_anima_pe.safetensors").
    cache_stems = [p.name.removesuffix("_anima_pe.safetensors") for p in all_files]
    stem_to_cache_idx = {s: i for i, s in enumerate(cache_stems)}

    cos_orig_hflip = F.cosine_similarity(pe_orig, pe_hflip, dim=-1).numpy()
    cos_orig_crop = F.cosine_similarity(pe_orig, pe_crop, dim=-1).numpy()
    cos_orig_jitter = F.cosine_similarity(pe_orig, pe_jitter, dim=-1).numpy()

    # Cross pairs: each sample vs a random *other* cached image.
    cross_sims: list[float] = []
    n_cached = cached_pools.shape[0]
    for i, p in enumerate(sampled):
        # pick a random index that is not this sample's cached entry (if present)
        forbidden = stem_to_cache_idx.get(p.stem, -1)
        for _ in range(20):
            j = rng.randrange(n_cached)
            if j != forbidden:
                break
        sim = F.cosine_similarity(pe_orig[i:i+1], cached_pools[j:j+1], dim=-1).item()
        cross_sims.append(sim)
    cross_sims_arr = np.array(cross_sims)

    def _summarize(name: str, arr: np.ndarray) -> dict:
        return dict(
            name=name,
            mean=float(arr.mean()),
            std=float(arr.std()),
            min=float(arr.min()),
            max=float(arr.max()),
            p10=float(np.percentile(arr, 10)),
            p50=float(np.percentile(arr, 50)),
            p90=float(np.percentile(arr, 90)),
        )

    summaries = {
        "self_hflip": _summarize("self_hflip", cos_orig_hflip),
        "self_crop": _summarize(f"self_crop({args.crop_frac:.2f})", cos_orig_crop),
        "self_jitter": _summarize(f"self_jitter({args.jitter:.2f})", cos_orig_jitter),
        "cross": _summarize("cross_random", cross_sims_arr),
    }

    print()
    print(f"  {'pair':24s}  {'mean':>6s} {'std':>6s} {'p10':>6s} {'p50':>6s} {'p90':>6s} {'min':>6s} {'max':>6s}")
    for s in summaries.values():
        print(
            f"  {s['name']:24s}  {s['mean']:6.3f} {s['std']:6.3f} "
            f"{s['p10']:6.3f} {s['p50']:6.3f} {s['p90']:6.3f} "
            f"{s['min']:6.3f} {s['max']:6.3f}"
        )

    cross_mean = summaries["cross"]["mean"]
    gaps = {
        "hflip": summaries["self_hflip"]["mean"] - cross_mean,
        "crop": summaries["self_crop"]["mean"] - cross_mean,
        "jitter": summaries["self_jitter"]["mean"] - cross_mean,
    }
    print(
        f"\n  Gap (mean self-aug - mean cross): "
        f"hflip {gaps['hflip']:+.3f} / crop {gaps['crop']:+.3f} / jitter {gaps['jitter']:+.3f}"
    )

    return dict(summaries=summaries, gaps=gaps, n=N)


# ----- step 2: crop retrieval rank -----

def step_crop_retrieval(
    args, *, image_paths: list[Path], cached_pools: torch.Tensor,
    all_files: list[Path],
) -> dict:
    """For Q query images, encode a 60% random crop, find rank of self in cache."""
    rng = random.Random(args.seed + 1)
    Q = min(args.n_retrieval, len(image_paths))
    print(f"\n=== Step 2: crop retrieval rank (Q={Q}, index size={cached_pools.shape[0]}) ===")
    print(f"  Crop fraction: {args.crop_frac:.2f}")

    # Map source stem -> cached index for self-rank lookup. Skip queries whose
    # stem isn't in the cache (shouldn't happen for resized/ + lora/ pair, but
    # be defensive).
    cache_stems = [p.name.removesuffix("_anima_pe.safetensors") for p in all_files]
    stem_to_idx = {s: i for i, s in enumerate(cache_stems)}

    queryable = [p for p in image_paths if p.stem in stem_to_idx]
    if len(queryable) < Q:
        print(f"  warn: only {len(queryable)} queryable (have cached PE), reducing Q")
        Q = len(queryable)
    sampled = rng.sample(queryable, Q)

    print("  Loading PE encoder...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_pe_encoder(device, name="pe")

    crops: list[torch.Tensor] = []
    self_indices: list[int] = []
    for p in sampled:
        x = _read_image_minus1to1(p)
        crops.append(_random_crop_resize(x, frac=args.crop_frac, rng=rng))
        self_indices.append(stem_to_idx[p.stem])

    print(f"  Encoding {Q} crops...")
    pe_crops = _encode_pooled_batch(bundle, crops, device=device)

    # Cosine sim against the entire cached pool, fp32 on CPU.
    pool_n = _l2norm(cached_pools)
    crops_n = _l2norm(pe_crops)
    sims = crops_n @ pool_n.T  # [Q, N]

    # rank-of-self: how many entries have similarity ≥ self?
    self_sim = sims[torch.arange(Q), torch.tensor(self_indices)]
    ranks = (sims >= self_sim.unsqueeze(1)).sum(dim=1) - 1  # 0-indexed; 0 = top hit
    ranks = ranks.numpy()

    recall_at_1 = float((ranks == 0).mean())
    recall_at_10 = float((ranks < 10).mean())
    recall_at_100 = float((ranks < 100).mean())
    median_rank = float(np.median(ranks))

    print()
    print(f"  recall@1   = {recall_at_1:.3f}  ({int(recall_at_1*Q)}/{Q})")
    print(f"  recall@10  = {recall_at_10:.3f}")
    print(f"  recall@100 = {recall_at_100:.3f}")
    print(f"  median rank = {median_rank:.0f}  (0 = top hit; lower is more memorizable)")

    return dict(
        Q=Q,
        n_index=int(cached_pools.shape[0]),
        crop_frac=args.crop_frac,
        recall_at_1=recall_at_1,
        recall_at_10=recall_at_10,
        recall_at_100=recall_at_100,
        median_rank=median_rank,
    )


# ----- step 3: effective rank -----

def step_effective_rank(args, *, cached_pools: torch.Tensor) -> dict:
    """SVD of pooled-feature matrix. Report 95% / 99% energy and participation ratio."""
    print(f"\n=== Step 3: effective rank of PE pooled features ===")
    X = cached_pools - cached_pools.mean(dim=0, keepdim=True)
    print(f"  Matrix: {tuple(X.shape)}  (centered)")
    s = torch.linalg.svdvals(X.to(torch.float32))
    s2 = (s ** 2).numpy()
    total = s2.sum()
    cum = np.cumsum(s2) / total
    k95 = int(np.searchsorted(cum, 0.95) + 1)
    k99 = int(np.searchsorted(cum, 0.99) + 1)
    # Participation ratio of the squared singular values (a continuous "effective
    # rank"; equals N for uniform spectrum, 1 for rank-1).
    pr = float((s2.sum() ** 2) / (s2 ** 2).sum())

    print(f"  95% energy in top {k95} dims")
    print(f"  99% energy in top {k99} dims")
    print(f"  participation ratio: {pr:.1f}")

    # IP-Adapter capacity sanity check: K resampler tokens * D dims is what the
    # IP path can represent. If 95% of pooled-feature variation lives in <K dims,
    # capacity isn't the bottleneck.
    K_default = 16
    capacity_note = "below K=16" if k95 < K_default else "above K=16"
    print(f"  → 95%-energy rank {k95} is {capacity_note} resampler-tokens")

    return dict(
        n=int(X.shape[0]),
        d=int(X.shape[1]),
        k95=k95,
        k99=k99,
        participation_ratio=pr,
    )


# ----- decision rule -----

def _verdict(report: dict) -> str:
    s1 = report.get("step1")
    s2 = report.get("step2")
    if s1 is None or s2 is None:
        return "(need both step 1 and step 2 for the verdict)"

    gap_min = min(s1["gaps"].values())
    r1 = s2["recall_at_1"]

    # Calibration:
    # - gap < 0.05 means aug-pair sims are within 1 std of cross-pair sims.
    #   The IP path can't tell "same image, augmented" from "different image".
    # - r1 > 0.95 means the cache fingerprints sources almost exactly.
    # - gap > 0.20 means clear separation; cross pairs land at ~0.5 cos and aug
    #   pairs at ~0.7+ cos, leaving ~0.3 cos of headroom for the resampler.
    if gap_min < 0.05 or r1 > 0.95:
        return (
            "Verdict: DATA WALL (likely fundamental).\n"
            "  → Augmentation gap < 0.05 OR crop recall@1 > 0.95.\n"
            "  → PE features are tightly coupled to source pixels; self-paired\n"
            "    training reduces to lookup. Switching to EasyControl will hit\n"
            "    the same wall (also self-paired). Need paired-but-different\n"
            "    references (multi-view, multi-panel of same character)."
        )
    if gap_min > 0.20 and r1 < 0.5:
        return (
            "Verdict: OPTIMIZATION-SIDE problem (data is fine).\n"
            "  → Aug-invariant signal exists at the global level (gap > 0.20)\n"
            "    AND crop retrieval is non-trivial (recall@1 < 0.5).\n"
            "  → Wall is probably gate / LR / budget — not the architecture.\n"
            "    IP-Adapter can be made to work with the current data layout."
        )
    return (
        "Verdict: AMBIGUOUS — features carry some aug-invariance but also\n"
        "  fingerprint the source. Useful next step: shuffled-reference\n"
        "  validation baseline (option D from ip-adapter-0502.md) on a small\n"
        "  budget run, or pre-cache N=4 augmented PE variants per image and\n"
        "  re-bench."
    )


# ----- main -----

def parse_steps(s: str) -> list[int]:
    if not s or s == "all":
        return [1, 2, 3]
    return [int(x) for x in s.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resized_dir", type=str, default="post_image_dataset/resized",
                        help="Directory of post-resize images (default: post_image_dataset/resized).")
    parser.add_argument("--cache_dir", type=str, default="post_image_dataset/lora",
                        help="Directory of cached *_anima_pe.safetensors (default: post_image_dataset/lora).")
    parser.add_argument("--n", type=int, default=200,
                        help="Sample size for step 1 aug histogram (default: 200).")
    parser.add_argument("--n_retrieval", type=int, default=50,
                        help="Sample size for step 2 crop retrieval (default: 50).")
    parser.add_argument("--crop_frac", type=float, default=0.6,
                        help="Random-crop side fraction for step 2 (default: 0.6).")
    parser.add_argument("--jitter", type=float, default=0.2,
                        help="Color-jitter amount for step 1 (default: 0.2).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=str, default="all",
                        help="Comma-separated step numbers, or 'all' (default).")
    parser.add_argument("--report", type=str, default=None,
                        help="Optional path to write JSON report.")
    args = parser.parse_args()

    resized_dir = (ROOT / args.resized_dir).resolve() if not Path(args.resized_dir).is_absolute() else Path(args.resized_dir)
    cache_dir = (ROOT / args.cache_dir).resolve() if not Path(args.cache_dir).is_absolute() else Path(args.cache_dir)
    if not resized_dir.is_dir():
        print(f"resized_dir not found: {resized_dir}", file=sys.stderr)
        sys.exit(1)
    if not cache_dir.is_dir():
        print(f"cache_dir not found: {cache_dir}", file=sys.stderr)
        sys.exit(1)

    steps = parse_steps(args.steps)
    print(f"Resized: {resized_dir}")
    print(f"Cache:   {cache_dir}")
    print(f"Steps:   {steps}")

    image_paths = sorted(p for p in resized_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        print(f"No images under {resized_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoading {sum(1 for _ in cache_dir.glob('*_anima_pe.safetensors'))} cached PE pools...")
    all_files, cached_pools = _load_all_cached_pools(cache_dir)
    print(f"  pools: {tuple(cached_pools.shape)}  dtype={cached_pools.dtype}")

    report: dict = {
        "args": vars(args),
        "n_images": len(image_paths),
        "n_cached": cached_pools.shape[0],
        "d_enc": int(cached_pools.shape[1]),
    }

    if 1 in steps:
        report["step1"] = step_aug_histogram(
            args, image_paths=image_paths,
            cached_pools=cached_pools, all_files=all_files,
        )
    if 2 in steps:
        report["step2"] = step_crop_retrieval(
            args, image_paths=image_paths,
            cached_pools=cached_pools, all_files=all_files,
        )
    if 3 in steps:
        report["step3"] = step_effective_rank(args, cached_pools=cached_pools)

    print("\n=== Decision support ===\n")
    print(_verdict(report))

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"\nJSON report → {out}")


if __name__ == "__main__":
    main()
