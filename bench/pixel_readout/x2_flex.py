#!/usr/bin/env python3
"""The ×2 training set for arm B: real 2× targets at a ×1 tier of 1024 / 896 / 768 (x2_plan.md). CPU.

Per resized image with a master (crop margins applied):

- ``down``: the master covers twice the stem's 1024-tier bucket. The ×1 input is the
  resized image and its cached latent; the 2× target is ``x2_masters.target_2x`` (the
  master LANCZOS-downscaled at the resize geometry, doubled).
- ``exact``: otherwise, the master floored to a multiple of 32 px on each side (cropped
  at the stem's anchor) is the 2× target as is, no resampling, and its LANCZOS ×½ is the
  ×1 input, if that ×1 holds 2,160 – 4,200 tokens (the 768 tier's band to the top of the
  1024 tier's). Its latent is not cached yet: encode ``x1/`` into ``cache/`` with
  ``scripts/preprocess/cache_latents.py --recursive`` (GPU, through the daemon).
- dropped: the ×1 would fall under 2,160 tokens.

The held-out split: every image of the P1 test artists leaves training; the read set is
up to ``--per_test_artist`` ``down`` images of each, plus, per text artist, the ``down``
image with the longest quoted caption text (out of training too). Writes, under
``--out``: ``x1/<artist>/<stem>.png`` (exact), ``x2/<artist>/<stem>.png`` (both kinds),
``manifest.json`` (paths relative to the repo home). Re-running skips targets already on disk.

Usage::

    python -m bench.pixel_readout.x2_flex --label x2flex_mmdd_hhmm
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from anime_tools.buckets import DEFAULT_FREEFIT_MAX_RATIO, band_for_tier
from anime_tools.stages.resize import (
    ANCHOR_KEY,
    CROP_ANCHORS,
    MARGINS_KEY,
    _flatten_alpha,
    margin_box,
    normalize_crop_anchor,
    normalize_crop_margins,
    oriented_size,
)
from PIL import Image, ImageOps

from bench._common import make_run_dir, write_result
from bench.pixel_readout.x2_masters import master_of, target_2x
from library.env import resolve_under_home
from library.io.cache import TE_CACHE_SUFFIX
from library.preprocess.latents import get_latents_npz_path

Image.MAX_IMAGE_PIXELS = None
P1_TEST_ARTISTS = (  # convnext_head.P1_RUN's test artists (its result.json)
    "audirenze",
    "asou_(asabu202)",
    "gin00",
    "kimoshi",
    "dorontabi",
    "butterchalk",
    "fizz_(pixiv34498626)",
    "sushispin",
)
TOK_LO = band_for_tier(768)[0]
TOK_HI = band_for_tier(1024)[1]


def exact_box(w: int, h: int, anchor: str | None) -> tuple[int, int, int, int]:
    """The 2× crop of a (w, h) master: the free-fit aspect clamp, then each side
    floored to 32 px, both cropped at the anchor."""
    ax, ay = CROP_ANCHORS[normalize_crop_anchor(anchor)]
    r = DEFAULT_FREEFIT_MAX_RATIO
    cw, ch = w, h
    if w / h > r:
        cw = round(h * r)
    elif h / w > r:
        ch = round(w * r)
    cw, ch = cw - cw % 32, ch - ch % 32
    left, top = round((w - cw) * ax), round((h - ch) * ay)
    return left, top, left + cw, top + ch


def classify(job: tuple[str, str, str, str]) -> dict:
    """Geometry only, from headers: → the manifest row without written files."""
    rel, master, resized_root, master_root = job
    with Image.open(Path(resized_root) / rel) as r:
        bw, bh = r.size
        anchor = r.info.get(ANCHOR_KEY)
        margins = normalize_crop_margins(r.info.get(MARGINS_KEY))
    with Image.open(Path(master_root) / master) as src:
        w, h = oriented_size(src)
    x0, y0, x1, y1 = margin_box(w, h, margins)
    w, h = x1 - x0, y1 - y0
    row = {"resized": rel, "master": master, "anchor": anchor}
    scale = 2 * (bh / h if w / h > bw / bh else bw / w)
    if scale <= 1.0:
        return {**row, "kind": "down", "x1_size": [bw, bh], "scale_2x": scale}
    box = exact_box(w, h, anchor)
    cw, ch = box[2] - box[0], box[3] - box[1]
    tok = (cw // 32) * (ch // 32)
    if TOK_LO <= tok <= TOK_HI and max(cw, ch) // 32 <= 256:
        return {
            **row,
            "kind": "exact",
            "x1_size": [cw // 2, ch // 2],
            "scale_2x": 1.0,
            "box": list(box),
        }
    return {**row, "kind": None, "x1_tokens": tok}


def build(job: tuple[dict, str, str, str]) -> str:
    """Write one row's target (and ×1 for exact). → 'wrote' / 'kept'."""
    row, resized_root, master_root, home = job
    x2p = Path(home) / row["x2"]
    x1p = Path(home) / row["x1"] if row["kind"] == "exact" else None
    if x2p.exists() and (x1p is None or x1p.exists()):
        return "kept"
    x2p.parent.mkdir(parents=True, exist_ok=True)
    master = Path(master_root) / row["master"]
    if row["kind"] == "down":
        with Image.open(Path(resized_root) / row["resized"]) as im:
            big, _ = target_2x(master, im)
        big.save(x2p, compress_level=3)
        return "wrote"
    with Image.open(Path(resized_root) / row["resized"]) as r:
        margins = normalize_crop_margins(r.info.get(MARGINS_KEY))
    with Image.open(master) as src:
        img = ImageOps.exif_transpose(src)
        work = _flatten_alpha(img).crop(margin_box(*img.size, margins))
    big = work.crop(tuple(row["box"]))
    small = big.resize(tuple(row["x1_size"]), Image.Resampling.LANCZOS)
    x1p.parent.mkdir(parents=True, exist_ok=True)
    big.save(x2p, compress_level=3)
    small.save(x1p, compress_level=3)
    return "wrote"


def quoted_len(caption: Path) -> int:
    if not caption.exists():
        return 0
    cap = caption.read_text(encoding="utf-8")
    if "japanese text" not in cap.lower():
        return 0
    return sum(len(s) for s in re.findall(r'"([^"]*)"', cap))


def main():
    from bench.pixel_readout.convnext_head import TEXT_ARTISTS

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--resized", default="post_image_dataset/resized")
    p.add_argument("--masters", default="image_dataset")
    p.add_argument("--lora_cache", default="post_image_dataset/lora")
    p.add_argument("--out", default="post_image_dataset/x2flex")
    p.add_argument("--per_test_artist", type=int, default=2)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry_run", action="store_true", help="manifest only, no images")
    p.add_argument("--label", default=None)
    args = p.parse_args()
    resized_root = Path(resolve_under_home(args.resized))
    master_root = Path(resolve_under_home(args.masters))
    lora = Path(resolve_under_home(args.lora_cache))
    out = Path(resolve_under_home(args.out))
    home = Path(resolve_under_home("."))

    jobs, missing = [], 0
    for f in sorted(resized_root.rglob("*.png")):
        m = master_of(f, resized_root, master_root)
        if m is None:
            missing += 1
            continue
        jobs.append(
            (
                str(f.relative_to(resized_root)),
                str(m.relative_to(master_root)),
                str(resized_root),
                str(master_root),
            )
        )
    with Pool(args.workers) as pool:
        rows = pool.map(classify, jobs, chunksize=16)

    kept_rows = []
    for row in rows:
        if row["kind"] is None:
            continue
        rel = Path(row["resized"])
        artist, stem = rel.parent.as_posix(), rel.stem
        te = lora / artist / f"{stem}{TE_CACHE_SUFFIX}"
        row["artist"], row["stem"] = artist, stem
        row["te"] = str(te.relative_to(home))
        row["x2"] = str((out / "x2" / artist / f"{stem}.png").relative_to(home))
        w1, h1 = row["x1_size"]
        if row["kind"] == "down":
            row["x1"] = str((resized_root / rel).relative_to(home))
            npz = get_latents_npz_path(resized_root / rel, (w1, h1), lora, resized_root)
        else:
            x1 = out / "x1" / artist / f"{stem}.png"
            row["x1"] = str(x1.relative_to(home))
            npz = get_latents_npz_path(x1, (w1, h1), out / "cache", out / "x1")
        row["npz"] = str(Path(npz).relative_to(home))
        row["te_exists"] = te.exists()
        row["npz_exists"] = Path(npz).exists()
        kept_rows.append(row)

    # held-out: the P1 test artists out of training; the read set from their down rows
    rng = random.Random(args.seed)
    for row in kept_rows:
        row["split"] = "heldout" if row["artist"] in P1_TEST_ARTISTS else "train"
    for a in P1_TEST_ARTISTS:
        downs = [r for r in kept_rows if r["artist"] == a and r["kind"] == "down"]
        for r in rng.sample(downs, min(args.per_test_artist, len(downs))):
            r["split"] = "test"
    for a in TEXT_ARTISTS:
        downs = [r for r in kept_rows if r["artist"] == a and r["kind"] == "down"]
        best = max(
            downs,
            key=lambda r: quoted_len(resized_root / a / f"{r['stem']}.txt"),
            default=None,
        )
        if best is not None and quoted_len(resized_root / a / f"{best['stem']}.txt"):
            best["split"] = "text"

    kinds = Counter(r["kind"] for r in rows)
    splits = Counter((r["split"], r["kind"]) for r in kept_rows)
    x1_tok = Counter()
    for r in kept_rows:
        if r["kind"] == "exact":
            t = (r["x1_size"][0] // 16) * (r["x1_size"][1] // 16)
            x1_tok["≤2160" if t <= 2160 else "2161-3024" if t <= 3024 else "3025-4200"] += 1
    print(
        f"{len(rows)} with a master ({missing} without): down {kinds['down']}, exact "
        f"{kinds['exact']}, dropped {kinds[None]}; exact ×1 tokens {dict(x1_tok)}; "
        f"no TE cache on {sum(not r['te_exists'] for r in kept_rows)}, down rows "
        f"without their latent {sum(not r['npz_exists'] for r in kept_rows if r['kind'] == 'down')}",
        flush=True,
    )
    print(f"splits: {dict(sorted(splits.items(), key=str))}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    built = Counter()
    if not args.dry_run:
        with Pool(args.workers) as pool:
            for i, res in enumerate(
                pool.imap_unordered(
                    build,
                    [(r, str(resized_root), str(master_root), str(home)) for r in kept_rows],
                    chunksize=4,
                )
            ):
                built[res] += 1
                if (i + 1) % 200 == 0:
                    print(f"  {i + 1}/{len(kept_rows)} {dict(built)}", flush=True)
    (out / "manifest.json").write_text(json.dumps(kept_rows, ensure_ascii=False))
    print(f"built {dict(built)}; manifest {out / 'manifest.json'}", flush=True)

    run_dir = make_run_dir("pixel_readout", label=args.label)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={
            "n_with_master": len(rows),
            "n_without_master": missing,
            "n_down": kinds["down"],
            "n_exact": kinds["exact"],
            "n_dropped": kinds[None],
            "exact_x1_tokens": dict(x1_tok),
            "splits": {f"{s}/{k}": n for (s, k), n in splits.items()},
            "built": dict(built),
            "manifest": str(out / "manifest.json"),
        },
        label=args.label,
        device="cpu",
    )


if __name__ == "__main__":
    main()
