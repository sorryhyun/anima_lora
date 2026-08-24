#!/usr/bin/env python3
"""Select + mask + paint the condition tree for the *region* EasyControl task
("generate the character inside the painted region").

Cond = a solid-color paint blob derived from the image's own SAM3 character
mask, composited over either a white canvas (``--cond_background white``) or
the real image itself (``--cond_background image``); target = the original
image (full scene, full caption). Image mode is the character-inpaint framing:
the blob occludes the character region, so the model learns "fill the paint
with the character, keep the surrounding scene" — and no LaMa-style background
recovery is ever needed, because neither training nor inference sees the
pixels behind the paint. The paint blob is *augmented* (blur/dilate/wobble,
seeded per-stem) so rough hand-painted blobs work at inference, but always
kept a strict superset of the character mask: in image mode any character
fringe outside the paint (hair wisps, feet) would leak pose/identity the
model could cheat off — a channel that is empty at inference time.

Four idempotent stages:

1. **Select** — read the caption index, keep solo-1girl images only
   (``count == ['1girl']``, ``solo`` in the flat tag bag, ``multiple views``
   absent — a multi-view sheet would union several figures into one mask),
   and mirror them as a relative-symlink target tree under ``{base}/staging``.
   Stale symlinks (no longer selected) are pruned.
2. **Sam** (GPU) — SAM3 concept segmentation over the staging tree into
   ``{base}/masks`` via ``scripts/preprocess/generate_masks.py``
   (``focus_prompts: [girl]`` at a low threshold — 0.7 misses ~45% of solo
   images on nude/close-up/stylized art; 0.4 recovers them with no measured
   over-inclusion, probed 2026-08-23). Existing masks are skipped
   (``--sam_force`` regenerates); "focus not found" images get no mask file
   and drop out at the cond stage.
3. **Cond** — gate each raw mask on coverage / fragmentation / hole-iness
   (a dirty mask drops the image: its staging symlink is unlinked so it
   leaves the dataset), then paint the condition image into
   ``{base}/cond_images``: ``--paint_color`` over the augmented blob, on a
   white canvas or the real image (``--cond_background``). One seeded cond
   per image (``crc32(stem)``, the inpaint
   convention) so re-runs are bit-identical. Drop reasons land in
   ``{base}/report.json``.
4. **Encode** (GPU) — VAE-encode the cond images at native size into
   ``{base}/cond`` so cond latent shape == target latent shape.

Target latents, caption TE embeddings (incl. shuffled variants), and PE all
come from the shared LoRA cache (``cache_dir = post_image_dataset/lora`` in
the blueprint) — nothing is re-encoded for targets. Run the stages in order:
the cond stage prunes gate-failed symlinks that a select-only re-run would
resurrect without cond latents.

Run from the repo root::

    python easycontrol_adapters/region/prep.py --skip_encode   # staging (GPU: SAM)
    python easycontrol_adapters/region/prep.py \
        --skip_select --skip_sam --skip_cond                   # preprocess (GPU: VAE)

Via the task runner (knobs from ``configs/easycontrol/region.toml``)::

    make easycontrol-staging    EASYADAPTER=region   # stages 1-3
    make easycontrol-preprocess EASYADAPTER=region   # stage 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]


def _stable_seed(name: str) -> int:
    """Process-stable per-stem seed (Python's hash() is salted; crc32 isn't)."""
    return zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF


def _save_png_atomic(arr: np.ndarray, out: Path) -> None:
    """Temp file in the same dir + os.replace so the final name is never truncated."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out.parent, suffix=".tmp.png")
    os.close(fd)
    try:
        Image.fromarray(arr).save(tmp)
        os.replace(tmp, out)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── Stage 1: select ──────────────────────────────────────────────────────────


def stage_select(
    index_path: Path,
    resized_dir: Path,
    caption_src: Path,
    staging: Path,
    *,
    limit: int | None,
) -> tuple[int, int, int]:
    from library.captioning.position_clauses import parse_caption

    if not index_path.is_file():
        raise SystemExit(f"{index_path} not found — run `make caption-index` first.")
    image_meta = json.loads(index_path.read_text(encoding="utf-8"))["image_meta"]

    keep: list[str] = []
    for key, meta in sorted(image_meta.items()):
        if meta.get("count") != ["1girl"]:
            continue
        if not (resized_dir / f"{key}.png").exists():
            continue
        # Prefer the resized caption (what training's TE cache encodes), fall
        # back to the caption master for a not-yet-mirrored image.
        cap_path = resized_dir / f"{key}.txt"
        if not cap_path.is_file():
            cap_path = caption_src / meta["path"]
            if not cap_path.is_file():
                continue
        tags = parse_caption(cap_path.read_text(encoding="utf-8")).tag_keys
        if "solo" not in tags or "multiple views" in tags:
            continue
        keep.append(key)
    if limit:
        keep = keep[:limit]

    kept_set = set(keep)
    written = 0
    for key in keep:
        dst = staging / f"{key}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        target = os.path.relpath(resized_dir / f"{key}.png", dst.parent)
        if dst.is_symlink():
            if os.readlink(dst) == target:
                continue
            dst.unlink()
        dst.symlink_to(target)
        written += 1
    pruned = 0
    if staging.is_dir():
        for p in staging.rglob("*.png"):
            if p.relative_to(staging).with_suffix("").as_posix() not in kept_set:
                p.unlink()
                pruned += 1
    return len(keep), written, pruned


# ── Stage 2: SAM character masks ─────────────────────────────────────────────


def stage_sam(
    staging: Path,
    masks_dir: Path,
    *,
    prompts: list[str],
    threshold: float,
    sam_dilate: int,
    batch_size: int,
    checkpoint: str,
    force: bool,
) -> None:
    cfg = {
        "prompts": [],
        "focus_prompts": list(prompts),
        "threshold": threshold,
        "dilate": sam_dilate,
        "path_pattern": "*",
    }
    cfg_path = masks_dir.parent / "sam_region.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")  # JSON is valid YAML
    cmd = [
        sys.executable,
        "scripts/preprocess/generate_masks.py",
        "--config",
        str(cfg_path),
        "--image-dir",
        str(staging),
        "--mask-dir",
        str(masks_dir),
        "--checkpoint",
        checkpoint,
        "--batch-size",
        str(batch_size),
        "--recursive",
    ]
    if force:
        cmd.append("--force")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


# ── Stage 3: gate + augment + paint ──────────────────────────────────────────


def _mask_gates(
    binary: np.ndarray,
    *,
    min_coverage: float,
    max_coverage: float,
    min_dominance: float,
    max_hole_frac: float,
    min_comp_frac: float,
    max_raggedness: float,
) -> tuple[np.ndarray | None, str | None]:
    """Return ``(speck-cleaned + pinhole-closed mask, None)`` or ``(None, drop_reason)``."""
    h, w = binary.shape
    area = h * w
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    comps = [(int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, n)]
    if not comps:
        return None, "empty"
    kept = [(a, i) for a, i in comps if a >= min_comp_frac * area]
    if not kept:
        return None, "specks_only"
    total = sum(a for a, _ in kept)
    cov = total / area
    if cov < min_coverage:
        return None, "too_small"
    if cov > max_coverage:
        return None, "too_large"
    if max(a for a, _ in kept) / total < min_dominance:
        return None, "fragmented"
    clean = np.isin(labels, [i for _, i in kept]).astype(np.uint8)
    # Raggedness: contour length / sqrt(area). A circle scores ~3.5, a clean
    # figure silhouette (hair spikes included) ~10-30; a salt-noise mask whose
    # speckle background connects outward — invisible to the hole gate — scores
    # in the hundreds.
    contours, _ = cv2.findContours(clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, closed=True) for c in contours)
    if perimeter / max(1.0, float(total) ** 0.5) > max_raggedness:
        return None, "ragged"
    # Hole fraction: background flood from a padded ring; unreached non-mask
    # pixels are enclosed holes (the ragged-tights signature).
    padded = np.pad((1 - clean).astype(np.uint8), 1, constant_values=1)
    ffmask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(padded, ffmask, (0, 0), 2)
    holes = int((padded[1:-1, 1:-1] == 1).sum())
    if holes / (total + holes) > max_hole_frac:
        return None, "holey"
    # Close pinholes so the exact/smooth augment levels don't inherit
    # salt-noise texture from an otherwise-clean mask.
    k = _odd(int(max(h, w) * 0.008))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    return clean, None


def _blur(m: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur; large sigmas go through a downscale round-trip for speed."""
    if sigma <= 8:
        return cv2.GaussianBlur(m, (0, 0), sigma)
    s = max(2, int(sigma / 8))
    small = cv2.resize(
        m,
        (max(1, m.shape[1] // s), max(1, m.shape[0] // s)),
        interpolation=cv2.INTER_AREA,
    )
    small = cv2.GaussianBlur(small, (0, 0), sigma / s)
    return cv2.resize(small, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_LINEAR)


def _odd(k: int) -> int:
    return max(3, k | 1)


def _elastic_warp(
    m: np.ndarray, rng: random.Random, edge: int, amplitude_frac: float
) -> np.ndarray:
    """Smooth random displacement field — hand-painted wobble.

    Bends the contour without bridging concavities, so limb structure (spread
    legs, raised arms) survives where a heavy blur would absorb it."""
    h, w = m.shape
    grid = rng.randint(6, 10)
    amp = edge * amplitude_frac
    field = [
        cv2.resize(
            np.array(
                [[rng.uniform(-1, 1) for _ in range(grid)] for _ in range(grid)],
                dtype=np.float32,
            ),
            (w, h),
            interpolation=cv2.INTER_CUBIC,
        )
        * amp
        for _ in range(2)
    ]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(
        m,
        xs + field[0],
        ys + field[1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _augment_mask(
    mask: np.ndarray, rng: random.Random, weights: list[float]
) -> np.ndarray:
    """Exact→rough augmentation spectrum: 0 exact, 1 smoothed, 2 blob, 3 wobble.

    Every level must keep the pose readable — the paint cue carries position
    AND rough silhouette, so bridging the gap between spread legs into one fat
    blob over-simplifies the cond (user feedback 2026-08-23). Blur sigmas stay
    small enough that only sub-limb concavities close.

    The output is always a strict superset of the (slightly dilated) source
    mask: levels 1/3 can locally dip inside the true silhouette, and on a
    real-background cond any character pixel left outside the paint leaks
    pose/identity that vanishes at inference (nothing under the paint)."""
    h, w = mask.shape
    edge = max(h, w)
    level = rng.choices((0, 1, 2, 3), weights=weights)[0]
    m = mask.astype(np.float32)
    guard_k = _odd(int(edge * 0.006))
    guard = cv2.dilate(mask, np.ones((guard_k, guard_k), np.uint8))
    if level == 0:
        out = guard > 0
    elif level == 1:
        out = _blur(m, edge * rng.uniform(0.008, 0.02)) >= 0.5
    elif level == 2:
        # Rounder + slightly grown, but sigma/threshold bounded so wide limb
        # gaps survive (a sub-0.5 threshold grows the region as it smooths).
        out = _blur(m, edge * rng.uniform(0.015, 0.04)) >= rng.uniform(0.40, 0.50)
    else:
        smoothed = _blur(m, edge * 0.012)
        out = _elastic_warp(smoothed, rng, edge, rng.uniform(0.01, 0.03)) >= 0.5
    out = (out | (guard > 0)).astype(np.uint8)
    return out if out.any() else mask


def stage_cond(
    staging: Path,
    masks_dir: Path,
    cond_images: Path,
    report_path: Path,
    *,
    gate_kwargs: dict,
    paint_color: tuple[int, int, int],
    cond_background: str,
    aug_weights: list[float],
    overwrite: bool,
) -> None:
    from library.preprocess import tqdm_progress
    from library.preprocess._dataset import walk_images

    images = walk_images(staging, recursive=True)
    if not images:
        raise SystemExit(f"No staged images under {staging} — run the select stage.")
    if not any(masks_dir.rglob("*_mask.png")):
        raise SystemExit(f"No masks under {masks_dir} — run the SAM stage first.")

    report: dict = {"kept": 0, "dropped": {}, "drops": []}

    def drop(img: Path, rel: Path, reason: str) -> None:
        img.unlink()  # leaves the dataset (staging tree = dataset scope)
        stale = (cond_images / rel).with_suffix(".png")
        stale.unlink(missing_ok=True)
        report["dropped"][reason] = report["dropped"].get(reason, 0) + 1
        report["drops"].append({"image": rel.as_posix(), "reason": reason})

    progress = tqdm_progress("Painting cond")
    progress(0, total=len(images))
    for p in images:
        rel = p.relative_to(staging)
        mask_path = masks_dir / rel.parent / f"{rel.stem}_mask.png"
        out = (cond_images / rel).with_suffix(".png")
        if not mask_path.exists():
            drop(p, rel, "no_mask")  # SAM "focus not found"
            progress(1, detail=f"drop {rel.stem}")
            continue
        if out.exists() and not overwrite:
            report["kept"] += 1
            progress(1, detail=f"skip {rel.stem}")
            continue
        with Image.open(p) as im:
            iw, ih = im.size
            rgb = np.array(im.convert("RGB")) if cond_background == "image" else None
        binary = (np.array(Image.open(mask_path).convert("L")) > 127).astype(np.uint8)
        if binary.shape != (ih, iw):
            binary = cv2.resize(binary, (iw, ih), interpolation=cv2.INTER_NEAREST)
        clean, reason = _mask_gates(binary, **gate_kwargs)
        if reason:
            drop(p, rel, reason)
            progress(1, detail=f"drop {rel.stem} ({reason})")
            continue
        rng = random.Random(_stable_seed(rel.stem))
        blob = _augment_mask(clean, rng, aug_weights)
        canvas = rgb if rgb is not None else np.full((ih, iw, 3), 255, np.uint8)
        canvas[blob > 0] = paint_color
        _save_png_atomic(canvas, out)
        report["kept"] += 1
        progress(1, detail=rel.stem)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(
        f"\nCond: {report['kept']} kept, "
        f"{sum(report['dropped'].values())} dropped {report['dropped']} "
        f"(details: {report_path})"
    )


# ── Stage 4: encode ──────────────────────────────────────────────────────────


def stage_encode(
    cond_images: Path,
    cond_cache_dir: Path,
    *,
    vae_path: str,
    batch_size: int,
    chunk_size: int,
    vae_2d: bool,
):
    """VAE-encode cond images at native size (cond latent shape == target's).

    ``vae_2d`` default ON — the same fold the target latents in
    post_image_dataset/lora/ were cached with, so cond and target stay
    numerically consistent."""
    import torch

    from library.models import qwen_vae
    from library.preprocess import cache_latents, tqdm_progress

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading VAE from {vae_path} (2d_fold={vae_2d}) ...")
    vae = qwen_vae.load_vae(
        vae_path,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=chunk_size,
        disable_cache=True,
        vae_2d=vae_2d,
    )
    vae.to(device, dtype=torch.bfloat16)
    vae.requires_grad_(False)
    vae.eval()

    stats = cache_latents(
        cond_images,
        vae,
        cache_dir=cond_cache_dir,
        recursive=True,
        batch_size=batch_size,
        progress=tqdm_progress("Caching cond latents"),
    )

    vae.to("cpu")
    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="post_image_dataset/easycontrol/region")
    parser.add_argument(
        "--caption_index", default="post_image_dataset/captions/caption_index.json"
    )
    parser.add_argument("--resized_dir", default="post_image_dataset/resized")
    parser.add_argument("--caption_src", default="image_dataset")
    parser.add_argument("--limit", type=int, default=None, help="cap #images (QA)")
    # ── SAM stage ──
    parser.add_argument("--prompts", nargs="+", default=["girl"])
    parser.add_argument(
        "--threshold", type=float, default=0.4, help="SAM concept threshold"
    )
    parser.add_argument("--sam_dilate", type=int, default=0)
    parser.add_argument("--sam_batch_size", type=int, default=4)
    parser.add_argument("--checkpoint", default="models/sam3/sam3.pt")
    parser.add_argument(
        "--sam_force", action="store_true", help="regenerate existing masks"
    )
    # ── cond stage: quality gates ──
    parser.add_argument("--min_coverage", type=float, default=0.03)
    parser.add_argument("--max_coverage", type=float, default=0.85)
    parser.add_argument(
        "--min_dominance",
        type=float,
        default=0.75,
        help="largest component / total mask area floor (fragmentation gate)",
    )
    parser.add_argument("--max_hole_frac", type=float, default=0.05)
    parser.add_argument(
        "--max_raggedness",
        type=float,
        default=20.0,
        help="contour-length / sqrt(mask-area) ceiling (salt-noise mask gate)",
    )
    parser.add_argument(
        "--min_comp_frac",
        type=float,
        default=0.002,
        help="components below this image-area fraction are specks (removed)",
    )
    # ── cond stage: paint ──
    parser.add_argument(
        "--paint_color", nargs=3, type=int, default=[0, 0, 0], metavar=("R", "G", "B")
    )
    parser.add_argument(
        "--cond_background",
        choices=("white", "image"),
        default="white",
        help="paint the blob over a white canvas or over the real image "
        "(character-inpaint framing: fill the paint, keep the scene)",
    )
    parser.add_argument(
        "--aug_weights",
        nargs=4,
        type=float,
        default=[0.25, 0.35, 0.30, 0.10],
        help="sampling weights for augment levels 0 exact / 1 smooth / 2 blob / 3 hull",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="re-paint staged cond PNGs"
    )
    # ── encode stage ──
    parser.add_argument("--vae", default="models/vae/qwen_image_vae.safetensors")
    parser.add_argument("--vae_2d", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=64)
    # ── stage toggles ──
    parser.add_argument("--skip_select", action="store_true")
    parser.add_argument("--skip_sam", action="store_true")
    parser.add_argument("--skip_cond", action="store_true")
    parser.add_argument("--skip_encode", action="store_true")
    args = parser.parse_args()

    base = Path(args.base)
    staging = base / "staging"
    masks_dir = base / "masks"
    cond_images = base / "cond_images"
    cond_cache_dir = base / "cond"

    if not args.skip_select:
        kept, written, pruned = stage_select(
            Path(args.caption_index),
            Path(args.resized_dir),
            Path(args.caption_src),
            staging,
            limit=args.limit,
        )
        print(f"Select: {kept} solo-1girl targets ({written} linked, {pruned} pruned)")

    if not args.skip_sam:
        stage_sam(
            staging,
            masks_dir,
            prompts=args.prompts,
            threshold=args.threshold,
            sam_dilate=args.sam_dilate,
            batch_size=args.sam_batch_size,
            checkpoint=args.checkpoint,
            force=args.sam_force,
        )

    if not args.skip_cond:
        stage_cond(
            staging,
            masks_dir,
            cond_images,
            base / "report.json",
            gate_kwargs=dict(
                min_coverage=args.min_coverage,
                max_coverage=args.max_coverage,
                min_dominance=args.min_dominance,
                max_hole_frac=args.max_hole_frac,
                min_comp_frac=args.min_comp_frac,
                max_raggedness=args.max_raggedness,
            ),
            paint_color=tuple(args.paint_color),
            cond_background=args.cond_background,
            aug_weights=args.aug_weights,
            overwrite=args.overwrite,
        )

    if not args.skip_encode:
        stats = stage_encode(
            cond_images,
            cond_cache_dir,
            vae_path=args.vae,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            vae_2d=args.vae_2d,
        )
        print(
            f"\nCond latent caching complete: {stats.written} cached, "
            f"{stats.skipped} skipped (already existed)"
        )


if __name__ == "__main__":
    main()
