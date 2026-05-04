#!/usr/bin/env python
"""Cross-LoRA bias-signature check.

Reads a base-DiT measure_bias.py run and N LoRA runs (each at some
multiplier), reports per-band integrated gap deviation and the per-step
LL gap shape correlation against base.

Lands a config in one of the proposal's three branches
(docs/proposal/dcw-learnable-calibrator.md §"Cross-LoRA invariance gate"):

    All Δ within ±15%  → base-DiT-scoped calibration (no per-LoRA artifact)
    Δ in [15%, 30%] AND multiplier-monotone → base-DiT + multiplier rescaling
    Δ > 30% OR per-band SNR profile shifts qualitatively → per-LoRA fallback

Inputs are the per_step_bands.csv + result.json files emitted by
measure_bias.py.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

BANDS = ("LL", "LH", "HL", "HH")


def load(run_dir: Path) -> dict:
    rows = list(csv.DictReader(open(run_dir / "per_step_bands.csv")))
    sigmas = np.array([float(r["sigma_i"]) for r in rows])
    gaps = {b: np.array([float(r[f"baseline_gap_{b}"]) for r in rows]) for b in BANDS}
    meta = json.load(open(run_dir / "result.json"))
    snr = meta["metrics"]["per_band_cross_sample_snr"]["baseline"]
    return dict(sigmas=sigmas, gaps=gaps, snr=snr, meta=meta)


def render(base_dir: Path, lora_dirs: list[tuple[str, Path]]) -> str:
    base = load(base_dir)
    n_steps = len(base["sigmas"])
    half = n_steps // 2

    lines: list[str] = []
    lines.append("# Cross-LoRA bias-signature check\n\n")
    lines.append(f"Base run: `{base_dir}`\n")
    lines.append(f"n_steps={n_steps}\n\n")

    # Per-band integrated signed gap %Δ table
    lines.append("## Integrated signed gap per band (%Δ vs base)\n\n")
    lines.append("| config | LL signed | Δ_LL% | LH | Δ% | HL | Δ% | HH | Δ% | branch |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|---|\n")

    def branch_for(deltas: dict[str, float]) -> str:
        max_abs = max(abs(d) for d in deltas.values())
        if max_abs <= 15:
            return "base-DiT scope"
        if max_abs <= 30:
            return "base-DiT + mult rescaling (if monotone)"
        return "per-LoRA fallback"

    base_signed = {b: float(base["gaps"][b].sum()) for b in BANDS}
    lines.append(
        f"| **base** | {base_signed['LL']:+.2f} | 0.00% | "
        f"{base_signed['LH']:+.2f} | 0.00% | "
        f"{base_signed['HL']:+.2f} | 0.00% | "
        f"{base_signed['HH']:+.2f} | 0.00% | — |\n"
    )

    rows_for_summary: list[tuple[str, dict, float, float, str]] = []
    for name, run in lora_dirs:
        d = load(run)
        signed = {b: float(d["gaps"][b].sum()) for b in BANDS}
        deltas = {b: (signed[b] - base_signed[b]) / abs(base_signed[b]) * 100 for b in BANDS}
        # Per-step LL shape: Pearson r + late-half max relative residual
        b_LL = base["gaps"]["LL"]
        l_LL = d["gaps"]["LL"]
        r = float(np.corrcoef(b_LL, l_LL)[0, 1])
        resid = l_LL - b_LL
        rel = resid / (np.abs(b_LL) + 1e-9) * 100
        late_max = float(np.abs(rel[half:]).max())
        br = branch_for(deltas)
        lines.append(
            f"| **{name}** | {signed['LL']:+.2f} | **{deltas['LL']:+.2f}%** | "
            f"{signed['LH']:+.2f} | {deltas['LH']:+.2f}% | "
            f"{signed['HL']:+.2f} | {deltas['HL']:+.2f}% | "
            f"{signed['HH']:+.2f} | {deltas['HH']:+.2f}% | {br} |\n"
        )
        rows_for_summary.append((name, d, r, late_max, br))

    # Per-step LL shape correlation
    lines.append("\n## Per-step LL gap shape correlation vs base\n\n")
    lines.append(
        "| config | Pearson r (24 steps) | late-half max \\|residual\\|/\\|base\\| |\n|---|---|---|\n"
    )
    for name, _, r, late_max, _ in rows_for_summary:
        lines.append(f"| {name} | **r = {r:.5f}** | {late_max:.2f}% |\n")

    # SNR profile preservation
    lines.append("\n## Cross-sample SNR profile (LL/LH/HL/HH)\n\n")
    lines.append("| config | LL | LH | HL | HH |\n|---|---|---|---|---|\n")
    base_snr = base["snr"]
    lines.append(
        f"| base | {base_snr['LL']:.3f} | {base_snr['LH']:.3f} | "
        f"{base_snr['HL']:.3f} | {base_snr['HH']:.3f} |\n"
    )
    for name, d, _, _, _ in rows_for_summary:
        snr = d["snr"]
        lines.append(
            f"| {name} | {snr['LL']:.3f} | {snr['LH']:.3f} | {snr['HL']:.3f} | {snr['HH']:.3f} |\n"
        )

    # Verdict
    lines.append("\n## Branch assignment\n\n")
    branches = {br for _, _, _, _, br in rows_for_summary}
    if branches == {"base-DiT scope"}:
        lines.append(
            "All LoRA × multiplier configurations land in the **base-DiT scope** "
            "branch (max |Δ| ≤ 15% on every band). No per-LoRA artifact needed; "
            "one reference profile per base DiT release suffices.\n"
        )
    else:
        lines.append(
            "Mixed branch outcomes: "
            + ", ".join(sorted(branches))
            + ". Consult §\"Decision gates\" of the proposal for the action per "
            "config.\n"
        )

    return "".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True, help="base run dir")
    p.add_argument(
        "--lora",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        help="--lora <label> <run_dir>; repeat per LoRA config",
    )
    p.add_argument("--out_md", type=Path, default=None, help="write report to this path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.lora:
        raise SystemExit("at least one --lora <label> <path> required")
    md = render(args.base, [(label, Path(path)) for label, path in args.lora])
    print(md)
    if args.out_md is not None:
        args.out_md.write_text(md)
        print(f"\n→ {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
