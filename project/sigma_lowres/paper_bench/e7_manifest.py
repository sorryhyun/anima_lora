"""E7 frozen train/holdout manifest generator (run ONCE, before training).

Implements the pre-registered pool composition from
`required_experiments.md` §E7:

- clusters = top-12 ("flat") / bottom-12 ("dirty") artist-median latent
  redundancy among artists with >= 8 qualifying images (1024-tier cache
  + complete sidecars) — the frozen rule behind
  `runs/20260729-1147-e7-inventory/`;
- per cluster: train on the 8 largest-n artists, reserve the 4
  smallest-n entirely (the S3s never-seen-same-style cell);
- within each trained artist: seeded ~65/35 split, train contribution
  capped at 16, holdout floor 3;
- train totals equalized across clusters by moving seeded-random stems
  from the larger cluster's largest train artists to its holdout side.

Outputs (runs/<stamp>-e7-manifest/):
- e7_manifest.json — full stem-level cells + per-stem redundancy
  (probe-cell selection later reuses these scores);
- path_pattern_flat.txt / path_pattern_dirty.txt — argv-ready fnmatch
  alternation strings ("artist/stem.*|..."), verified below against the
  real resized tree via the production filter
  (`library.datasets.path_filter.filter_paths_by_glob`).

Usage:
    python project/sigma_lowres/paper_bench/e7_manifest.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "bench" / "tier_routing"))
sys.path.insert(0, str(REPO))

from library.datasets.path_filter import filter_paths_by_glob  # noqa: E402
from redundancy import score_corpus  # noqa: E402

SEED = 20260729
MIN_IMAGES = 8
N_CLUSTER = 12
N_TRAIN_ARTISTS = 8
TRAIN_FRAC = 0.65
TRAIN_CAP = 16
HOLDOUT_FLOOR = 3
RESIZED = REPO / "post_image_dataset" / "resized"


def esc(s: str) -> str:
    """fnmatch-escape [ and ] (the only specials legal in our names)."""
    return s.replace("[", "[[]").replace("]", "[]]")


def main() -> None:
    recs = [r for r in score_corpus() if r.tier == 1024 and r.complete()]
    by_artist: dict[str, list] = {}
    for r in recs:
        by_artist.setdefault(r.artist, []).append(r)
    eligible = {a: rs for a, rs in by_artist.items() if len(rs) >= MIN_IMAGES}
    ranked = sorted(
        eligible, key=lambda a: float(np.median([r.redundancy for r in eligible[a]]))
    )
    clusters = {"dirty": ranked[:N_CLUSTER], "flat": ranked[-N_CLUSTER:]}

    rng = np.random.default_rng(SEED)
    manifest: dict[str, dict] = {}
    for name, artists in clusters.items():
        by_n = sorted(artists, key=lambda a: (-len(eligible[a]), a))
        trained, reserved = by_n[:N_TRAIN_ARTISTS], by_n[N_TRAIN_ARTISTS:]
        cell = {"train": {}, "holdout": {}, "reserved": {}, "redundancy": {}}
        for a in sorted(trained):
            stems = sorted(r.stem for r in eligible[a])
            n = len(stems)
            train_n = min(TRAIN_CAP, round(TRAIN_FRAC * n), n - HOLDOUT_FLOOR)
            order = rng.permutation(n)
            picked = sorted(stems[i] for i in order[:train_n])
            cell["train"][a] = picked
            cell["holdout"][a] = sorted(set(stems) - set(picked))
        for a in sorted(reserved):
            cell["reserved"][a] = sorted(r.stem for r in eligible[a])
        for a in artists:
            cell["redundancy"].update(
                {f"{a}/{r.stem}": round(r.redundancy, 4) for r in eligible[a]}
            )
        manifest[name] = cell

    # equalize train totals: larger cluster moves seeded-random stems from its
    # largest train artists to holdout until totals match
    def total(c):
        return sum(len(v) for v in manifest[c]["train"].values())

    while total("flat") != total("dirty"):
        big = "flat" if total("flat") > total("dirty") else "dirty"
        artist = max(manifest[big]["train"], key=lambda a: len(manifest[big]["train"][a]))
        stems = manifest[big]["train"][artist]
        moved = stems[int(rng.integers(len(stems)))]
        stems.remove(moved)
        manifest[big]["holdout"][artist] = sorted(
            manifest[big]["holdout"][artist] + [moved]
        )

    # patterns + verification against the real tree with the production filter
    all_pngs = sorted(str(p) for p in RESIZED.glob("*/*.png"))
    patterns = {}
    for name, cell in manifest.items():
        alts = [
            f"{esc(a)}/{esc(s)}.*" for a, ss in cell["train"].items() for s in ss
        ]
        pattern = "|".join(alts)
        mask = filter_paths_by_glob(all_pngs, str(RESIZED), pattern)
        matched = {
            str(Path(p).relative_to(RESIZED)).rsplit(".", 1)[0]
            for p, m in zip(all_pngs, mask)
            if m
        }
        expected = {f"{a}/{s}" for a, ss in cell["train"].items() for s in ss}
        assert matched == expected, (
            f"{name}: pattern matched {len(matched)} != manifest {len(expected)}; "
            f"diff {matched ^ expected}"
        )
        patterns[name] = pattern

    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M")
    out = HERE / "runs" / f"{stamp}-e7-manifest"
    out.mkdir(parents=True, exist_ok=True)
    (out / "e7_manifest.json").write_text(
        json.dumps(
            dict(
                seed=SEED,
                rule=(
                    f"top/bottom-{N_CLUSTER} artist-median redundancy, n>={MIN_IMAGES}; "
                    f"train {N_TRAIN_ARTISTS} largest-n artists, reserve rest; "
                    f"{TRAIN_FRAC} split, cap {TRAIN_CAP}, holdout floor {HOLDOUT_FLOOR}; "
                    "totals equalized"
                ),
                clusters=manifest,
            ),
            indent=1,
        )
    )
    for name, pattern in patterns.items():
        (out / f"path_pattern_{name}.txt").write_text(pattern + "\n")

    for name, cell in manifest.items():
        t = sum(len(v) for v in cell["train"].values())
        h = sum(len(v) for v in cell["holdout"].values())
        r = sum(len(v) for v in cell["reserved"].values())
        print(
            f"{name:>5}: train {t} ({len(cell['train'])} artists, "
            f"{ {a: len(v) for a, v in cell['train'].items()} }), "
            f"holdout {h}, reserved {r} ({sorted(cell['reserved'])})"
        )
    print("out:", out)


if __name__ == "__main__":
    main()
