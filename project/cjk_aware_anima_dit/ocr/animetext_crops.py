#!/usr/bin/env python3
"""Unlabelled text crops from the AnimeText splits — the SSL corpus for
``ssl_tower_simmim.py`` (tower domain adaptation, 2026-09-08).

    ANIMA_ANIMETEXT_ROOT=<dir with the split parquets> \\
        python project/cjk_aware_anima_dit/ocr/animetext_crops.py \\
            [--split test|valid|train] [--shard N] [--limit 2000]

``deepghs/AnimeText`` (CC-BY-NC-SA — research build only) is detection-only:
``objects.bbox`` = normalised ``[cx, cy, w, h]``, ``category`` 0 = text, 1 =
hard negative. Text boxes are cut axis-aligned at ``--pad`` 12 % of the box's
long edge per side (the O1 crop convention, no deskew — the polygons live in a
separate file and SSL does not need them), dropped below ``--min_side`` px.

Splits are sharded on the Hub (``test`` ×1, ``valid`` ×2, ``train`` ×6); pass
``--shard`` to pick one, or ``--parquet`` for a path the naming doesn't cover.
``--start_group`` resumes a partly-cropped shard without re-decoding the row
groups already on disk.

Output, beside the parquet: ``animetext_crops/<split>/<image_id>_<k>.<ext>`` +
``animetext_crops/manifest_<split>[_<name>].parquet`` (``split, image_id, k, w,
h, box, path``, path relative to the root). No text column — there is none.
``--format webp`` is lossless (bit-exact, ~0.44× PNG) and is what the bulk
crops use; the original 140k test PNGs predate the flag.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SHARDS = {"test": 1, "valid": 2, "train": 6}
# cv2 writes lossless WebP at quality > 100.
ENCODE = {"png": [], "webp": [cv2.IMWRITE_WEBP_QUALITY, 101]}


def animetext_root(override: str | None = None) -> Path:
    env = override or os.environ.get("ANIMA_ANIMETEXT_ROOT")
    if not env:
        raise SystemExit(
            "ANIMA_ANIMETEXT_ROOT is not set — the directory holding the "
            "AnimeText split parquets (e.g. test-00000-of-00001.parquet)."
        )
    return Path(env).expanduser()


def shard_parquet(root: Path, split: str, shard: int) -> Path:
    n = SHARDS[split]
    if not 0 <= shard < n:
        raise SystemExit(f"{split} has {n} shard(s); --shard {shard} is out of range")
    p = root / f"{split}-{shard:05d}-of-{n:05d}.parquet"
    if not p.is_file():
        raise SystemExit(f"missing {p} — download it first")
    return p


def crop_box(img: np.ndarray, box, pad: float, min_side: int):
    """Axis-aligned crop of a normalised ``[cx, cy, w, h]`` box, padded and
    clamped to the image."""
    ih, iw = img.shape[:2]
    cx, cy, w, h = box
    w, h = w * iw, h * ih
    p = pad * max(w, h)
    x0 = int(round(cx * iw - w / 2 - p))
    y0 = int(round(cy * ih - h / 2 - p))
    x1 = int(round(cx * iw + w / 2 + p))
    y1 = int(round(cy * ih + h / 2 + p))
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(iw, x1), min(ih, y1)
    if min(x1 - x0, y1 - y0) < min_side:
        return None, None
    return img[y0:y1, x0:x1], [x0, y0, x1, y1]


def _do_group(args) -> list[dict]:
    parquet, gi, out_dir, split, pad, min_side, fmt, overwrite = args
    t = pq.ParquetFile(parquet).read_row_group(gi).to_pylist()
    params = ENCODE[fmt]
    rows = []
    for r in t:
        img = cv2.imdecode(
            np.frombuffer(r["image"]["bytes"], np.uint8), cv2.IMREAD_COLOR
        )
        if img is None:
            continue
        obj = r["objects"]
        k = 0
        for box, cat in zip(obj["bbox"], obj["category"]):
            if cat != 0:
                continue
            crop, xyxy = crop_box(img, box, pad, min_side)
            if crop is None:
                continue
            fn = out_dir / f"{r['image_id']}_{k}.{fmt}"
            if overwrite or not fn.exists():
                cv2.imwrite(str(fn), crop, params)
            rows.append(
                dict(
                    split=split,
                    image_id=int(r["image_id"]),
                    k=k,
                    w=int(crop.shape[1]),
                    h=int(crop.shape[0]),
                    box=xyxy,
                    path=str(fn.relative_to(out_dir.parent.parent)),
                )
            )
            k += 1
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", help="override $ANIMA_ANIMETEXT_ROOT")
    ap.add_argument("--split", default="test", choices=sorted(SHARDS))
    ap.add_argument("--shard", type=int, default=0, help="shard index within the split")
    ap.add_argument("--parquet", help="explicit parquet path (overrides split/shard)")
    ap.add_argument("--limit", type=int, help="first N images (smoke / a draw)")
    ap.add_argument(
        "--start_group", type=int, default=0, help="resume: skip row groups below this"
    )
    ap.add_argument(
        "--name",
        help="manifest suffix → manifest_<split>_<name>.parquet (default: smoke "
        "with --limit, none without)",
    )
    ap.add_argument("--format", default="png", choices=sorted(ENCODE))
    ap.add_argument("--pad", type=float, default=0.12)
    ap.add_argument("--min_side", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    root = animetext_root(a.root)
    parquet = Path(a.parquet) if a.parquet else shard_parquet(root, a.split, a.shard)
    out_dir = root / "animetext_crops" / a.split
    out_dir.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(parquet)
    groups = list(range(a.start_group, pf.metadata.num_row_groups))
    if a.limit:
        n, keep = 0, []
        for gi in groups:
            keep.append(gi)
            n += pf.metadata.row_group(gi).num_rows
            if n >= a.limit:
                break
        groups = keep
    print(
        f"{parquet.name}: {pf.metadata.num_rows} rows in {pf.metadata.num_row_groups} "
        f"row groups; cropping {len(groups)} groups (from {groups[0] if groups else '-'}) "
        f"→ {out_dir} as .{a.format}",
        flush=True,
    )
    t0 = time.time()
    rows: list[dict] = []
    jobs = [
        (str(parquet), gi, out_dir, a.split, a.pad, a.min_side, a.format, a.overwrite)
        for gi in groups
    ]
    with Pool(a.workers) as pool:
        for i, part in enumerate(pool.imap_unordered(_do_group, jobs), 1):
            rows += part
            if i % 50 == 0 or i == len(jobs):
                print(
                    f"  {i}/{len(jobs)} groups, {len(rows)} crops, {time.time() - t0:.0f}s",
                    flush=True,
                )
    df = pd.DataFrame(rows).sort_values(["image_id", "k"]).reset_index(drop=True)
    name = a.name or ("smoke" if a.limit else "")
    mpath = out_dir.parent / f"manifest_{a.split}{'_' + name if name else ''}.parquet"
    df.to_parquet(mpath, index=False)
    side = np.minimum(df.w, df.h)
    print(
        f"{len(df)} crops from {df.image_id.nunique()} images → {mpath}\n"
        f"min side p10/p50/p90 = {np.percentile(side, [10, 50, 90]).astype(int).tolist()}, "
        f"long edge p50/p90 = {np.percentile(np.maximum(df.w, df.h), [50, 90]).astype(int).tolist()}",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
