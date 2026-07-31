#!/usr/bin/env python
"""sigma_lowres — build the probe-local 1280-tier cache for the 1280→1024 probe.

The roadmap's discriminating experiment (ratio-vs-capacity, record/questions.md Q1)
needs native-1280 caches, but a corpus re-preprocess at a 1280 tier would (a)
be expensive and (b) leak a new tier into training — on-disk caches are the
source of truth for training buckets. So this script builds a tiny standalone
dataset root OUTSIDE ``post_image_dataset/``::

    probe1280_cache/
      resized/<artist>/{stem}.png        # production resize chain, tier [1280]
      lora/<artist>/{stem}_{WxH}_anima.npz          # production VAE cache chain
      lora/<artist>/{stem}_anima_te.safetensors     # symlink (TE is text-only)

which ``run_sigma_probe.py --tier 1280 --data_root <this dir>`` consumes
directly. Selection: sources with native token count ≥ the 1280 band whose TE
cache already exists, redundancy-stratified (scored on their existing
native-tier latent cache — content statistic, tier-insensitive for ranking)
with a per-artist cap. Over-provision (~1.5× the probe's --num_images) so the
probe's own stratified selection still has room.

Resize + encode go through ``library.preprocess`` (NOT the probe's own
``encode_probe_latents`` chain) so the probe's reenc arm remains a genuine
encode-chain control instead of a bit-identical no-op.

Usage::

    make daemon-run ARGS="project/sigma_lowres/bench/prep_1280_probe.py"
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from library.datasets.buckets import (  # noqa: E402
    DEFAULT_FREEFIT_MAX_RATIO,
    freefit_band_for_edge,
)
from library.preprocess.images import process_image  # noqa: E402
from library.preprocess.latents import cache_latents  # noqa: E402
from library.preprocess.resize_preview import (  # noqa: E402
    DEFAULT_RESIZE_CROP_ANCHOR,
    normalize_crop_margins,
    normalize_fit_mode,
    parse_bucket_resos,
)
from project.sigma_lowres.bench.tier_routing.redundancy import (  # noqa: E402
    score_corpus,
    score_image,
)

SOURCE_ROOT = REPO_ROOT / "image_dataset"
POST_LORA = REPO_ROOT / "post_image_dataset" / "lora"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
VAE = "models/vae/qwen_image_vae.safetensors"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "probe1280_cache"),
        help="probe-local dataset root (lora/ + resized/ are created inside)",
    )
    p.add_argument("--vae", default=VAE)
    p.add_argument("--num_images", type=int, default=36)
    p.add_argument("--max_per_artist", type=int, default=3)
    p.add_argument(
        "--score_limit",
        type=int,
        default=400,
        help="seeded subsample of eligible sources scored for redundancy "
        "(caps the npz-read cost of stratification)",
    )
    p.add_argument(
        "--max_native_tokens",
        type=int,
        default=40_000,
        help="skip absurdly large sources (~10MP at the default) — decode "
        "cost, not a quality judgment",
    )
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def eligible_sources(min_tokens: int, max_tokens: int) -> list[tuple[Path, str]]:
    """(source_path, artist) for images whose native patch-token count can
    populate the 1280 tier and whose TE cache already exists. ``followlinks``
    because image_dataset holds symlinked artist dirs."""
    from PIL import Image

    out: list[tuple[Path, str]] = []
    for dirpath, _dirnames, filenames in os.walk(SOURCE_ROOT, followlinks=True):
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if p.suffix.lower() not in IMG_EXTS:
                continue
            artist = p.parent.name
            if not (POST_LORA / artist / f"{p.stem}_anima_te.safetensors").exists():
                continue
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception:
                continue
            tokens = (w // 16) * (h // 16)
            if min_tokens <= tokens <= max_tokens:
                out.append((p, artist))
    return out


def native_npz_index(artist: str) -> dict[str, Path]:
    """stem → existing native-tier latent npz in the main cache (for
    redundancy scoring only; stripped by the same pattern score_image uses)."""
    idx: dict[str, Path] = {}
    for npz in (POST_LORA / artist).glob("*_anima.npz"):
        idx[re.sub(r"_\d+x\d+_anima\.npz$", "", npz.name)] = npz
    return idx


def stratified_pick(
    scored: list[tuple[float, Path, str]], n: int, max_per_artist: int
) -> list[tuple[float, Path, str]]:
    """Evenly spaced picks over the redundancy-sorted pool with a per-artist
    cap — same walk-outward scheme as redundancy.select_probe_set."""
    import numpy as np

    pool = sorted(scored, key=lambda t: t[0])
    if len(pool) <= n:
        return pool
    picks: list[tuple[float, Path, str]] = []
    per_artist: dict[str, int] = {}
    taken: set[int] = set()
    targets = np.linspace(0, len(pool) - 1, n).round().astype(int)
    for t in targets:
        for off in range(len(pool)):
            for idx in (t + off, t - off):
                if idx < 0 or idx >= len(pool) or idx in taken:
                    continue
                _, _, artist = pool[idx]
                if per_artist.get(artist, 0) >= max_per_artist:
                    continue
                picks.append(pool[idx])
                taken.add(idx)
                per_artist[artist] = per_artist.get(artist, 0) + 1
                break
            else:
                continue
            break
    return sorted(picks, key=lambda t: t[0])


def main() -> None:
    args = parse_args()
    out_root = Path(args.out).resolve()
    resized_dir = out_root / "resized"
    lora_dir = out_root / "lora"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / ".gitignore").write_text("*\n")  # cache artifacts, never committed

    band_lo, _band_hi = freefit_band_for_edge(1280)
    print(f"scanning {SOURCE_ROOT} for sources ≥ {band_lo} native tokens…")
    sources = eligible_sources(band_lo, args.max_native_tokens)
    print(f"eligible: {len(sources)} images, {len({a for _, a in sources})} artists")
    if not sources:
        raise SystemExit("no eligible sources")

    rng = random.Random(args.seed)
    sample = sources
    if len(sample) > args.score_limit:
        sample = rng.sample(sorted(sample), args.score_limit)
    print(f"scoring redundancy on {len(sample)} existing native caches…")
    npz_idx: dict[str, dict[str, Path]] = {}
    scored: list[tuple[float, Path, str]] = []
    for p, artist in sample:
        idx = npz_idx.setdefault(artist, native_npz_index(artist))
        npz = idx.get(p.stem)
        if npz is None:
            continue
        scored.append((score_image(npz).redundancy, p, artist))
    print(f"scored: {len(scored)}")

    picks = stratified_pick(scored, args.num_images, args.max_per_artist)
    print(
        f"picked {len(picks)} images, redundancy "
        f"{picks[0][0]:.3f}…{picks[-1][0]:.3f}, "
        f"{len({a for _, _, a in picks})} artists"
    )

    # --- resize: production chain, single-tier [1280] ---------------------
    max_ratio = DEFAULT_FREEFIT_MAX_RATIO
    ppcfg = REPO_ROOT / "configs" / "preprocess.toml"
    if ppcfg.exists():
        max_ratio = float(
            tomllib.loads(ppcfg.read_text()).get("freefit_max_ratio", max_ratio)
        )
    bucket_args = (
        (1024, 1024),  # vestigial legacy slots (layout stability)
        512,
        2048,
        64,
        [1280],
        DEFAULT_RESIZE_CROP_ANCHOR,
        parse_bucket_resos(None),
        normalize_crop_margins(None),
        normalize_fit_mode(None),
        float(max_ratio),
    )
    buckets: dict[tuple[int, int], int] = {}
    for _red, p, artist in picks:
        name, reso, skipped = process_image(
            p, resized_dir, bucket_args, copy_captions=True, rel_dir=artist
        )
        buckets[reso] = buckets.get(reso, 0) + 1
        print(f"  {artist}/{name} → {reso[0]}x{reso[1]}{' (skip)' if skipped else ''}")
    print("bucket distribution:")
    for reso in sorted(buckets):
        tokens = (reso[0] // 16) * (reso[1] // 16)
        print(f"  {reso[0]:>4d}x{reso[1]:<4d}: {buckets[reso]:>3d}  ({tokens} tokens)")

    # --- VAE latents: production chain (bf16, 2D-folded) ------------------
    from library.models.qwen_vae import load_vae

    device = torch.device("cuda")
    print(f"loading VAE from {args.vae}…")
    vae = load_vae(str(REPO_ROOT / args.vae), device="cpu", disable_mmap=True)
    vae.to(device, dtype=torch.bfloat16)
    vae.convert_to_2d()
    vae.requires_grad_(False)
    vae.eval()
    stats = cache_latents(
        resized_dir,
        vae,
        cache_dir=lora_dir,
        recursive=True,
        batch_size=args.batch_size,
    )
    print(f"latents: {stats.written} written, {stats.skipped} skipped")
    del vae
    torch.cuda.empty_cache()

    # --- TE symlinks (text-only caches — resolution-independent) ----------
    linked = 0
    for _red, p, artist in picks:
        te_src = POST_LORA / artist / f"{p.stem}_anima_te.safetensors"
        te_dst = lora_dir / artist / te_src.name
        if not te_dst.exists():
            te_dst.symlink_to(te_src)
        linked += 1
    print(f"TE symlinks: {linked}")

    # --- sanity: what the probe will see ----------------------------------
    records = score_corpus(data_root=out_root)
    ok = [r for r in records if r.tier == 1280 and r.complete()]
    print(
        f"probe view: {len(records)} records, {len(ok)} complete @ tier 1280, "
        f"redundancy {min(r.redundancy for r in ok):.3f}…"
        f"{max(r.redundancy for r in ok):.3f}"
        if ok
        else "probe view: NO complete tier-1280 records — something is wrong"
    )
    bad = [r for r in records if r.tier != 1280]
    if bad:
        print(f"⚠ {len(bad)} records landed OUTSIDE the 1280 band:")
        for r in bad:
            print(f"  {r.artist}/{r.stem}  {r.cached_px}  tokens={r.tokens}")


if __name__ == "__main__":
    main()
