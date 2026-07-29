"""E7 corpus inventory — per-artist 1024-tier counts + redundancy profile.

Feeds the E7 controlled-LoRA 2x2 style factorial: checks whether two
style clusters ("flat" = high latent redundancy vs "dirty" = low) exist
with enough artists x images to populate the cells (target: >=12 artists
x >=8 qualifying images per cluster; within-artist holdout needs >=3
held out per artist).

Qualifying image = 1024-tier cached npz + complete() (resized png + TE
cache present). Style metric = the established tier_routing redundancy
scalar (1 - unique joint-code fraction; high = flat), aggregated
per-artist as the median over qualifying images.

Writes e7_inventory.tsv (one row per artist) next to this script under
runs/<stamp>-e7-inventory/ and prints the cluster picture.

Usage:
    python project/sigma_lowres/paper_bench/e7_inventory.py
"""

from __future__ import annotations

import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bench" / "tier_routing"))

from redundancy import score_corpus  # noqa: E402

MIN_IMAGES = 8


def main() -> None:
    recs = score_corpus()
    qual = [r for r in recs if r.tier == 1024 and r.complete()]

    by_artist: dict[str, list] = {}
    for r in qual:
        by_artist.setdefault(r.artist, []).append(r)

    rows = []
    for artist, rs in sorted(by_artist.items()):
        reds = sorted(x.redundancy for x in rs)
        rows.append(
            dict(
                artist=artist,
                n_qual=len(rs),
                red_median=statistics.median(reds),
                red_min=reds[0],
                red_max=reds[-1],
                red_iqr=(
                    statistics.quantiles(reds, n=4)[2]
                    - statistics.quantiles(reds, n=4)[0]
                    if len(reds) >= 4
                    else float("nan")
                ),
            )
        )

    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M")
    out_dir = HERE / "runs" / f"{stamp}-e7-inventory"
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / "e7_inventory.tsv"
    with tsv.open("w") as f:
        f.write("artist\tn_qual\tred_median\tred_min\tred_max\tred_iqr\n")
        for row in sorted(rows, key=lambda r: -r["red_median"]):
            f.write(
                f"{row['artist']}\t{row['n_qual']}\t{row['red_median']:.4f}\t"
                f"{row['red_min']:.4f}\t{row['red_max']:.4f}\t{row['red_iqr']:.4f}\n"
            )

    n_img = len(qual)
    print(f"corpus: {len(recs)} cached images, {n_img} qualifying (1024-tier, complete)")
    print(f"artists with >= {MIN_IMAGES} qualifying images:")
    big = [r for r in rows if r["n_qual"] >= MIN_IMAGES]
    for row in sorted(big, key=lambda r: -r["red_median"]):
        bar = "#" * int(row["red_median"] * 40)
        print(
            f"  {row['artist']:<28} n={row['n_qual']:<4} "
            f"red {row['red_median']:.3f} [{row['red_min']:.3f},{row['red_max']:.3f}] {bar}"
        )
    meds = sorted((r["red_median"] for r in big), reverse=True)
    print(f"\n{len(big)} artists eligible; median-redundancy spread "
          f"{meds[-1]:.3f}..{meds[0]:.3f}")
    for k in (12, 16):
        if len(big) >= 2 * k:
            top = meds[k - 1]
            bot = meds[-k]
            print(f"  top-{k} vs bottom-{k} artist-median split: "
                  f"flat >= {top:.3f}  |  dirty <= {bot:.3f}  (gap {top - bot:+.3f})")
    print("tsv:", tsv)


if __name__ == "__main__":
    main()
