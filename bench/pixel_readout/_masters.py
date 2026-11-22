"""The master behind a resized image, and the real 2× target cut from it."""

from __future__ import annotations

from pathlib import Path

from anime_tools.stages.resize import (
    ANCHOR_KEY,
    CROP_ANCHORS,
    MARGINS_KEY,
    _flatten_alpha,
    margin_box,
    normalize_crop_anchor,
    normalize_crop_margins,
)
from PIL import Image, ImageOps

EXTS = (".png", ".jpg", ".jpeg", ".webp")
Image.MAX_IMAGE_PIXELS = None


def master_of(resized: Path, resized_root: Path, master_root: Path) -> Path | None:
    rel = resized.relative_to(resized_root).with_suffix("")
    for ext in EXTS:
        cand = master_root / rel.parent / (rel.name + ext)
        if cand.exists():
            return cand
    return None


def target_2x(master: Path, resized: Image.Image) -> tuple[Image.Image, float]:
    """→ (the master resized and cropped as the resized image was, at 2× its bucket;
    the linear scale that resize applies to the master, > 1 = an upscale)."""
    anchor = resized.info.get(ANCHOR_KEY)
    margins = normalize_crop_margins(resized.info.get(MARGINS_KEY))
    bw, bh = resized.size
    with Image.open(master) as src:
        img = ImageOps.exif_transpose(src)
        work = _flatten_alpha(img).crop(margin_box(*img.size, margins))
    # resize_to_bucket's geometry at exactly 2×: doubling the bucket instead rounds the
    # cover size and the crop offset by themselves and lands half a ×1 pixel off
    ax, ay = CROP_ANCHORS[normalize_crop_anchor(anchor)]
    w, h = work.size
    if w / h > bw / bh:
        new_h, new_w = bh, round(bh * w / h)
    else:
        new_w, new_h = bw, round(bw * h / w)
    left, top = 2 * round((new_w - bw) * ax), 2 * round((new_h - bh) * ay)
    big = work.resize((2 * new_w, 2 * new_h), Image.Resampling.LANCZOS)
    return big.crop((left, top, left + 2 * bw, top + 2 * bh)), 2 * new_w / w
