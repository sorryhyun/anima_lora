#!/usr/bin/env python
"""sigma_lowres — G8 gradient-norm renormalization check (post-hoc, CPU-only).

Re-analysis of an existing σ-probe run's ``per_image.jsonl`` (no GPU, no new
forwards): tests whether the interior peak / low-σ dip of gap_e(σ) is a
cosine-denominator effect. gap is scale-free (a mismatch *fraction*), so under
``gap ≈ mismatch/G^p`` with G = ‖g‖ U-shaped, the absolute-mismatch proxy
``(gap − Floor_e)·G^p`` should be low-σ-maximal even though raw gap dips at
the lowest bin. Reports per-edge: bin-mean gap, the ·G and ·G² proxies,
Spearman(gap, 1/G) at bin level and per image, and the bin0-vs-peak flip.
Record + policy note: ``project/sigma_lowres/record/groundings.md`` G8 (the proxy is
noise-dominated at σ ≳ 0.8 — read the S1 region only; gap stays the
criterion metric).

Usage:
    python check_gap_renorm.py results/20260724-1237-phase0 \
        [--floors 896=0,768=0.127,512=0.326]

Floors default to each edge's last-bin (highest-σ) mean gap — its own
endpoint-Floor estimate — so the script is self-contained on any run.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "results_dir", type=Path, help="probe run dir containing per_image.jsonl"
    )
    ap.add_argument(
        "--floors",
        default="",
        help="comma list edge=floor (gap units); default: last-bin mean gap per edge",
    )
    args = ap.parse_args()

    rows = [json.loads(line) for line in (args.results_dir / "per_image.jsonl").open()]
    sig = np.array(rows[0]["sigma_centers"])
    edges = [
        k.removeprefix("gap_")
        for k in rows[0]
        if k.startswith("gap_") and k != "gap_reenc"
    ]
    floors = dict(kv.split("=") for kv in args.floors.split(",") if kv)

    gnorm = np.array([r["gnorm_native"] for r in rows]).mean(0)
    print(f"{args.results_dir.name}  n={len(rows)}  bins={np.round(sig, 3).tolist()}")
    print(f"mean gnorm_native: {np.round(gnorm, 2).tolist()}")

    for e in edges:
        gap = np.array([r[f"gap_{e}"] for r in rows])
        mgap = gap.mean(0)
        floor = float(floors.get(e, mgap[-1]))
        m1 = (mgap - floor) * gnorm
        m2 = (mgap - floor) * gnorm**2
        lo = sig <= 0.5
        print(f"\n-- edge {e} (Floor={floor:.3f})")
        print(f"  bin-mean gap      : {np.round(mgap, 3).tolist()}")
        print(f"  (gap-Floor)*|g|   : {np.round(m1, 3).tolist()}")
        print(f"  (gap-Floor)*|g|^2 : {np.round(m2, 3).tolist()}")
        r_all = spearmanr(mgap, 1 / gnorm).statistic
        r_lo = spearmanr(mgap[lo], 1 / gnorm[lo]).statistic
        print(f"  spearman(gap, 1/|g|): all {r_all:+.2f}  sigma<=0.5 {r_lo:+.2f}")
        per = np.array(
            [
                spearmanr(gi, 1 / np.array(r["gnorm_native"])).statistic
                for gi, r in zip(gap, rows)
                if np.all(np.isfinite(gi))
            ]
        )
        print(
            f"  per-image spearman(gap, 1/|g|): mean {per.mean():+.2f} "
            f"(SEM {per.std() / np.sqrt(len(per)):.2f}), frac>0 {np.mean(per > 0):.2f}"
        )
        k = int(np.argmax(mgap))
        flip = "FLIPPED to max" if m1[0] >= m1[k] else "not flipped"
        print(
            f"  dip check: raw bin0 {mgap[0]:.3f} vs peak bin{k} {mgap[k]:.3f}; "
            f"renorm bin0 {m1[0]:.3f} vs {m1[k]:.3f} -> {flip}"
        )


if __name__ == "__main__":
    main()
