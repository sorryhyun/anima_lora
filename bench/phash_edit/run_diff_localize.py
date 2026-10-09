"""phash_edit — Phase 0 for position-clause instructions: where do the pairs differ?

CPU-only pass over the miner's manifest (no GPU, no adapter). For every mined
*edit* pair, compute a coarse pixel-diff map between cond and target on a
64x64 grid and localize it: how much of the canvas changed, in how many
blobs, and which 3x3-grid header ("top left", "center", ...) the dominant
blob lands in.

The verdict this feeds (_archive/proposals/phash_edit_position_clauses.md): a
position-clause instruction (`On the upper left, -english text.`) is only
worth re-mining for if a healthy fraction of pairs have a *clean,
single-region* diff that a coarse header can address, and the header
distribution is not degenerate (everything "center"). The same pass doubles
as a purity audit: a pair whose tag delta is large but whose pixels barely
move (or vice versa) is exactly the label-noise pair suspected of teaching
the variant-axis prior.

Classes per pair (thresholds are grid-cell fractions of the 64x64 map):
    near_zero  diff_frac < 0.001 — pixels don't support the tag delta
    global     diff_frac > 0.55  — whole-canvas change (recolor/redraw)
    single     dominant blob holds >= 75% of diff mass — clause-addressable
    multi      everything else — needs multiple clauses or a drop gate

`both_directions` duplicates every pair reversed; the diff is symmetric, so
unordered dedup halves the work and the report counts unique pairs.

Usage:
    uv run python bench/phash_edit/run_diff_localize.py
    uv run python bench/phash_edit/run_diff_localize.py --limit 200 --examples 4
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from library.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

PAIRS_JSON = (
    REPO_ROOT / "post_image_dataset" / "easycontrol" / "phash_edit" / "pairs.json"
)

GRID = 64
MIN_BLOB_CELLS = 2  # single-cell hits are decode/screentone noise; 2+ cells
# can be a genuine eye/text edit on a full-body shot
NEAR_ZERO_FRAC = 0.001  # ~4 cells of 4096 — "no pixel support", not "small edit"
GLOBAL_FRAC = 0.55
SINGLE_MASS_SHARE = 0.75


def load_grid(path: str, cache: dict) -> np.ndarray | None:
    """Decode → RGB → (GRID, GRID, 3) float in [0,1], memoized per path."""
    if path in cache:
        return cache[path]
    try:
        with Image.open(path) as im:
            arr = (
                np.asarray(
                    im.convert("RGB").resize((GRID, GRID), Image.BILINEAR),
                    dtype=np.float32,
                )
                / 255.0
            )
    except Exception as e:  # missing/corrupt member — report, don't die
        logger.warning("failed to load %s: %s", path, e)
        arr = None
    cache[path] = arr
    return arr


def label_components(mask: np.ndarray) -> np.ndarray:
    """8-connected component labels (0 = background)."""
    try:
        from scipy import ndimage

        labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
        return labels
    except ImportError:
        pass
    # BFS fallback — 64x64 grid, plenty fast
    labels = np.zeros(mask.shape, dtype=np.int32)
    nxt = 0
    for sy, sx in zip(*np.nonzero(mask)):
        if labels[sy, sx]:
            continue
        nxt += 1
        stack = [(sy, sx)]
        labels[sy, sx] = nxt
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx_ = y + dy, x + dx
                    if (
                        0 <= ny < GRID
                        and 0 <= nx_ < GRID
                        and mask[ny, nx_]
                        and not labels[ny, nx_]
                    ):
                        labels[ny, nx_] = nxt
                        stack.append((ny, nx_))
    return labels


def header_for(cy: float, cx: float) -> str:
    """Quantize a centroid (grid coords) into the 3x3 position-clause vocabulary."""
    row = "top" if cy < GRID / 3 else ("bottom" if cy >= 2 * GRID / 3 else "")
    col = "left" if cx < GRID / 3 else ("right" if cx >= 2 * GRID / 3 else "")
    if row and col:
        return f"{row} {col}"
    return row or col or "center"


def analyze_pair(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a - b).mean(axis=2)  # (GRID, GRID) mean-channel diff
    # Noise floor from the unchanged majority: near-twins share most cells, so
    # the 25th percentile sits on encode/screentone noise even for large edits.
    thr = max(0.05, float(np.percentile(d, 25)) + 0.06)
    mask = d > thr
    diff_frac = float(mask.mean())

    labels = label_components(mask)
    blobs = []
    for lab in range(1, labels.max() + 1):
        ys, xs = np.nonzero(labels == lab)
        if len(ys) < MIN_BLOB_CELLS:
            continue
        mass = float(d[ys, xs].sum())
        blobs.append(
            {
                "cells": int(len(ys)),
                "mass": mass,
                "cy": float(ys.mean()),
                "cx": float(xs.mean()),
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            }
        )
    blobs.sort(key=lambda blob: blob["mass"], reverse=True)

    total_mass = sum(blob["mass"] for blob in blobs)
    top_share = blobs[0]["mass"] / total_mass if total_mass > 0 else 0.0

    if diff_frac < NEAR_ZERO_FRAC or not blobs:
        cls = "near_zero"
    elif diff_frac > GLOBAL_FRAC:
        cls = "global"
    elif top_share >= SINGLE_MASS_SHARE:
        cls = "single"
    else:
        cls = "multi"

    header = header_for(blobs[0]["cy"], blobs[0]["cx"]) if blobs else ""
    return {
        "diff_frac": round(diff_frac, 4),
        "n_blobs": len(blobs),
        "top_share": round(top_share, 3),
        "header": header,
        "class": cls,
        "top_bbox": blobs[0]["bbox"] if blobs else None,
        "mask": mask,
    }


def save_example(
    run_dir: Path, name: str, a: np.ndarray, b: np.ndarray, res: dict
) -> str:
    """Side-by-side cond|target|diff-mask panel with the dominant bbox drawn."""
    size = 192
    panel = Image.new("RGB", (size * 3, size))
    for i, arr in enumerate((a, b)):
        panel.paste(
            Image.fromarray((arr * 255).astype(np.uint8)).resize(
                (size, size), Image.NEAREST
            ),
            (i * size, 0),
        )
    mask_img = (
        Image.fromarray((res["mask"] * 255).astype(np.uint8))
        .convert("RGB")
        .resize((size, size), Image.NEAREST)
    )
    panel.paste(mask_img, (2 * size, 0))
    if res["top_bbox"]:
        x0, y0, x1, y1 = [v * size / GRID for v in res["top_bbox"]]
        draw = ImageDraw.Draw(panel)
        for off in (0, size, 2 * size):
            draw.rectangle([off + x0, y0, off + x1, y1], outline=(255, 64, 64), width=2)
    fname = f"example_{name}.png"
    panel.save(run_dir / fname)
    return fname


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", type=Path, default=PAIRS_JSON)
    ap.add_argument("--limit", type=int, default=0, help="cap unique pairs (0 = all)")
    ap.add_argument(
        "--examples", type=int, default=8, help="example panels saved per class"
    )
    ap.add_argument("--label", default="diffloc")
    args = ap.parse_args()

    manifest = json.loads(args.pairs.read_text())
    edit_pairs = [p for p in manifest["pairs"] if p.get("kind") == "edit"]

    seen: set[frozenset] = set()
    unique = []
    for p in edit_pairs:
        key = frozenset((p["cond_image"], p["target_image"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    if args.limit:
        unique = unique[: args.limit]
    logger.info("edit pairs: %d directed, %d unique", len(edit_pairs), len(unique))

    cache: dict = {}
    paths = sorted(
        {p["cond_image"] for p in unique} | {p["target_image"] for p in unique}
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda pth: load_grid(pth, cache), paths))

    run_dir = make_run_dir("phash_edit", label=args.label)
    rows, per_class_examples = [], defaultdict(list)
    for p in unique:
        a, b = cache.get(p["cond_image"]), cache.get(p["target_image"])
        if a is None or b is None:
            continue
        res = analyze_pair(a, b)
        rows.append(
            {
                "pair_id": p["pair_id"],
                "artist": p["artist"],
                "tag_delta": p["tag_delta"],
                "phash": p["phash"],
                "delta_caption": p["delta_caption"],
                **{k: v for k, v in res.items() if k != "mask"},
            }
        )
        if len(per_class_examples[res["class"]]) < args.examples:
            per_class_examples[res["class"]].append(
                save_example(run_dir, f"{res['class']}_{p['pair_id']}", a, b, res)
            )

    n = len(rows)
    by_class = Counter(r["class"] for r in rows)
    headers = Counter(r["header"] for r in rows if r["class"] == "single")

    def stratum(delta: int) -> str:
        return "d1-3" if delta <= 3 else ("d4-8" if delta <= 8 else "d9+")

    strata: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        strata[stratum(r["tag_delta"])][r["class"]] += 1

    # Label-noise suspects, both directions of the mismatch
    silent = [r for r in rows if r["class"] == "near_zero" and r["tag_delta"] >= 4]
    unlabeled = [r for r in rows if r["class"] == "global" and r["tag_delta"] <= 2]

    csv_path = run_dir / "per_pair.csv"
    with open(csv_path, "w", newline="") as f:
        fields = [k for k in rows[0] if k != "top_bbox"] if rows else []
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        "n_directed_edit_pairs": len(edit_pairs),
        "n_unique_pairs": n,
        "class_counts": dict(by_class),
        "class_frac": {k: round(v / n, 3) for k, v in by_class.items()} if n else {},
        "class_median_diff_frac": {
            k: round(
                float(np.median([r["diff_frac"] for r in rows if r["class"] == k])), 4
            )
            for k in by_class
        },
        "single_header_hist": dict(headers.most_common()),
        "by_delta_stratum": {k: dict(v) for k, v in sorted(strata.items())},
        "suspect_silent_delta": len(silent),  # tags say edit, pixels say copy
        "suspect_unlabeled_global": len(
            unlabeled
        ),  # pixels say redraw, tags say ~nothing
        "thresholds": {
            "grid": GRID,
            "near_zero_frac": NEAR_ZERO_FRAC,
            "global_frac": GLOBAL_FRAC,
            "single_mass_share": SINGLE_MASS_SHARE,
        },
    }
    artifacts = ["per_pair.csv"] + [
        f for lst in per_class_examples.values() for f in lst
    ]
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        artifacts=artifacts,
        label=args.label,
    )
    logger.info("summary: %s", json.dumps(metrics, indent=2))
    logger.info("run dir: %s", run_dir)


if __name__ == "__main__":
    main()
