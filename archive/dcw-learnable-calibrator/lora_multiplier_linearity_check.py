#!/usr/bin/env python
"""Cross-LoRA multiplier-linearity check.

Reads three measure_bias.py runs (base DiT, LoRA @ mult=0.5, LoRA @ mult=1.0)
and tests whether the LoRA's bias contribution is linear in multiplier:

    gap_LL@mult=m  ?≈  (1-m) · gap_LL_base + m · gap_LL@mult=1.0

If yes (residual <few % of base), the proposal can land on the cleanest
"base-model-scoped calibration, no multiplier rescaling" branch of
docs/proposal/dcw-learnable-calibrator.md §"Cross-LoRA invariance gate".

Inputs are the per_step_bands.csv files emitted by measure_bias.py.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

BANDS = ("LL", "LH", "HL", "HH")


def load_band_gaps(csv_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rows = list(csv.DictReader(open(csv_path)))
    sigmas = np.array([float(r["sigma_i"]) for r in rows])
    gaps = {b: np.array([float(r[f"baseline_gap_{b}"]) for r in rows]) for b in BANDS}
    return sigmas, gaps


def linearity_report(
    base_csv: Path, mid_csv: Path, full_csv: Path, mid_mult: float = 0.5
) -> str:
    sigmas, gb = load_band_gaps(base_csv)
    _, gm = load_band_gaps(mid_csv)
    _, gf = load_band_gaps(full_csv)
    n_steps = len(sigmas)
    half = n_steps // 2

    lines: list[str] = []
    lines.append("# LoRA multiplier-linearity check\n")
    lines.append(
        f"Inputs:\n"
        f"  base       = {base_csv}\n"
        f"  mid (m={mid_mult}) = {mid_csv}\n"
        f"  full (m=1.0) = {full_csv}\n\n"
    )

    # Predicted at mid via linear interpolation in multiplier
    def linerr(b_arr: np.ndarray, m_arr: np.ndarray, f_arr: np.ndarray) -> tuple:
        pred_mid = (1 - mid_mult) * b_arr + mid_mult * f_arr
        # Integrated signed
        sb, sm_, sf = b_arr.sum(), m_arr.sum(), f_arr.sum()
        spred = pred_mid.sum()
        # Residual / |base| (pct)
        return (sb, sm_, sf, spred, (sm_ - spred) / max(abs(sb), 1e-9) * 100)

    lines.append("## Integrated signed gap per band (linearity at mult=0.5)\n\n")
    lines.append(
        "| band | base | obs @0.5 | obs @1.0 | linear-pred @0.5 | "
        "linerr (% of base) | Δ@1.0 vs base |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    for b in BANDS:
        sb, sm_, sf, spred, err = linerr(gb[b], gm[b], gf[b])
        d10 = (sf - sb) / max(abs(sb), 1e-9) * 100
        lines.append(
            f"| **{b}** | {sb:+8.2f} | {sm_:+8.2f} | {sf:+8.2f} | "
            f"{spred:+8.2f} | **{err:+5.2f}%** | {d10:+5.2f}% |\n"
        )

    # LL is the band that matters; per-step late-half deviation
    lines.append("\n## Per-step LL linearity residual\n")
    pred_LL = (1 - mid_mult) * gb["LL"] + mid_mult * gf["LL"]
    resid = gm["LL"] - pred_LL
    rel = resid / (np.abs(gb["LL"]) + 1e-9) * 100  # pct of base step gap
    max_late = float(np.abs(rel[half:]).max())
    mean_late = float(np.abs(rel[half:]).mean())
    lines.append(
        f"- max |residual| anywhere: {float(np.abs(resid).max()):.3f}\n"
        f"- max relative residual at step 0 is meaningless (base ≈ 0 there)\n"
        f"- late-half (i ≥ {half}) max relative residual: **{max_late:.2f}%**\n"
        f"- late-half (i ≥ {half}) mean relative residual: {mean_late:.2f}%\n"
    )
    return "".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True, help="base per_step_bands.csv")
    p.add_argument("--mid", type=Path, required=True, help="LoRA @ mid mult per_step_bands.csv")
    p.add_argument("--full", type=Path, required=True, help="LoRA @ mult=1.0 per_step_bands.csv")
    p.add_argument("--mid_mult", type=float, default=0.5, help="multiplier of --mid (default 0.5)")
    p.add_argument(
        "--out_md", type=Path, default=None, help="write report to this markdown path"
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    md = linearity_report(args.base, args.mid, args.full, args.mid_mult)
    print(md)
    if args.out_md is not None:
        args.out_md.write_text(md)
        print(f"\n→ {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
