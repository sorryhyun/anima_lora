#!/usr/bin/env python
"""Validate the dcw-learnable-calibrator transfer hypothesis.

Reads a ``gaps_per_sample.npz`` produced by ``measure_bias.py
--dump_per_sample_gaps`` and computes the early-vs-late per-sample
gap correlation that gates v0 (offline head) vs v0a (online controller)
in ``docs/proposal/dcw-learnable-calibrator.md``.

Decision (per proposal):
    abs(r) > 0.7 → v0a (online controller, no MLP)
    0.4 < abs(r) <= 0.7 → v0 with early-step features
    abs(r) <= 0.4 → v0 with c_pool only

Reports r computed three ways to detect seed-coupling inflation:
    all-trajectories (n_img × n_seeds): noisiest, biased upward by seed
    seed-0 only (n_img): clean per-prompt independent draws
    seed-1 only (n_img): replicate
    seed-averaged (n_img): per-prompt smoothed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BANDS = ("LL", "LH", "HL", "HH")


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """95% CI on Pearson r via Fisher z-transform."""
    if not np.isfinite(r) or abs(r) >= 1.0 or n <= 3:
        return (float("nan"), float("nan"))
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = 1.959963984540054  # 95% normal
    lo, hi = z - z_crit * se, z + z_crit * se
    return float(np.tanh(lo)), float(np.tanh(hi))


def split_correlation(gaps: np.ndarray, K: int | None = None) -> dict:
    """r = corr(early_mean, late_mean) per the proposal's one-liner."""
    n_traj, n_steps = gaps.shape
    if K is None:
        K = n_steps // 2
    early = gaps[:, :K].mean(axis=1)
    late = gaps[:, K:].mean(axis=1)
    r = corr(early, late)
    lo, hi = fisher_ci(r, n_traj)
    return dict(K=K, r=r, ci95=(lo, hi), n=n_traj)


def step_pair_corr(gaps: np.ndarray) -> np.ndarray:
    """Full (n_steps × n_steps) Pearson correlation across trajectories."""
    return np.corrcoef(gaps.T)


def rank_stability(gaps: np.ndarray, K: int | None = None) -> float:
    """Spearman rank correlation between early-mean and late-mean per traj."""
    n_traj, n_steps = gaps.shape
    if K is None:
        K = n_steps // 2
    from scipy.stats import spearmanr  # type: ignore

    early = gaps[:, :K].mean(axis=1)
    late = gaps[:, K:].mean(axis=1)
    rho, _ = spearmanr(early, late)
    return float(rho)


def decision(r_abs: float) -> str:
    if r_abs > 0.7:
        return "v0a (online controller)"
    if r_abs > 0.4:
        return "v0 with early-step features"
    return "v0 with c_pool only"


def analyze(npz_path: Path, n_seeds: int = 2) -> dict:
    z = np.load(npz_path, allow_pickle=True)
    sigmas = z["sigmas"]
    n_steps = sigmas.shape[0]

    out: dict = {
        "npz": str(npz_path),
        "n_steps": int(n_steps),
        "sigmas": [float(x) for x in sigmas.tolist()],
        "n_seeds_assumed": n_seeds,
        "bands": {},
    }

    for band in BANDS:
        gaps = z[f"gap_{band}"]  # (n_traj, n_steps)
        n_traj = gaps.shape[0]
        n_img = n_traj // n_seeds if n_traj % n_seeds == 0 else n_traj

        # Split-correlation across all trajectories
        all_r = split_correlation(gaps)

        # Per-seed splits + seed-averaged (only if seeds line up)
        per_seed: list[dict] = []
        seed_avg: dict | None = None
        if n_traj % n_seeds == 0 and n_seeds >= 1:
            gaps_reshaped = gaps.reshape(n_img, n_seeds, n_steps)
            for s in range(n_seeds):
                per_seed.append(split_correlation(gaps_reshaped[:, s, :]))
            seed_avg = split_correlation(gaps_reshaped.mean(axis=1))

        # Per-step pairwise correlation matrix → average upper-tri off-diag
        # of the early-vs-late block as a robustness check.
        cm = step_pair_corr(gaps)
        K = n_steps // 2
        early_late_block = cm[:K, K:]
        avg_pair_corr = float(np.nan_to_num(early_late_block).mean())

        # Spearman rank correlation (early vs late traj rank)
        try:
            rho = rank_stability(gaps)
        except Exception:
            rho = float("nan")

        # Per-step cross-trajectory mean and std for context
        mean_per_step = gaps.mean(axis=0)
        std_per_step = gaps.std(axis=0)

        out["bands"][band] = {
            "n_traj": int(n_traj),
            "all": all_r,
            "per_seed": per_seed,
            "seed_avg": seed_avg,
            "avg_early_late_pair_corr": avg_pair_corr,
            "spearman_rho": rho,
            "decision_from_seed_avg_r": (
                decision(abs(seed_avg["r"])) if seed_avg else None
            ),
            "decision_from_all_r": decision(abs(all_r["r"])),
            "mean_gap_per_step": [float(x) for x in mean_per_step.tolist()],
            "std_gap_per_step": [float(x) for x in std_per_step.tolist()],
            "integrated_abs_mean_gap": float(np.abs(mean_per_step).sum()),
            "mean_per_step_std": float(std_per_step.mean()),
        }
    return out


def render_markdown(result: dict) -> str:
    lines: list[str] = []
    lines.append("# Transfer-hypothesis check\n")
    lines.append(f"Source: `{result['npz']}`\n")
    lines.append(
        f"n_steps={result['n_steps']}, n_seeds_assumed={result['n_seeds_assumed']}\n"
    )
    lines.append("\n## Per-band early-vs-late correlation\n")
    lines.append(
        "Definition: r = corrcoef(gaps[:, :N/2].mean(1), gaps[:, N/2:].mean(1))\n"
    )
    lines.append(
        "\n| Band | n_traj | r (all) | 95% CI | r (seed-avg) | 95% CI | "
        "Spearman ρ | Decision (seed-avg) |\n|---|---|---|---|---|---|---|---|\n"
    )
    for band, info in result["bands"].items():
        all_r = info["all"]
        sa = info.get("seed_avg")
        sa_r = sa["r"] if sa else float("nan")
        sa_ci = sa["ci95"] if sa else (float("nan"), float("nan"))
        lines.append(
            f"| **{band}** | {info['n_traj']} | "
            f"{all_r['r']:+.3f} | [{all_r['ci95'][0]:+.2f}, {all_r['ci95'][1]:+.2f}] | "
            f"{sa_r:+.3f} | [{sa_ci[0]:+.2f}, {sa_ci[1]:+.2f}] | "
            f"{info['spearman_rho']:+.3f} | "
            f"{info.get('decision_from_seed_avg_r') or 'n/a'} |\n"
        )
    lines.append("\n## Per-seed splits (seed-coupling diagnostic)\n")
    lines.append(
        "If single-seed r ≈ all-trajectories r and seed-avg r is similar, "
        "seed coupling is weak and the all-trajectory r is trustworthy. "
        "If seed-avg r is much higher than per-seed r, per-prompt signal "
        "is real but seeds add noise, not coupling.\n"
    )
    for band, info in result["bands"].items():
        if not info.get("per_seed"):
            continue
        lines.append(f"\n### {band}\n")
        for s, ps in enumerate(info["per_seed"]):
            lines.append(
                f"- seed {s}: r = {ps['r']:+.3f} (n={ps['n']}, "
                f"95% CI [{ps['ci95'][0]:+.2f}, {ps['ci95'][1]:+.2f}])\n"
            )
    return "".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--npz",
        type=Path,
        required=True,
        help="Path to gaps_per_sample.npz produced by measure_bias.py "
        "--dump_per_sample_gaps",
    )
    p.add_argument(
        "--n_seeds",
        type=int,
        default=2,
        help="Number of seeds per image (rows are interleaved img-major).",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="If set, write transfer_check.json + transfer_check.md alongside.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.npz.exists():
        print(f"npz not found: {args.npz}", file=sys.stderr)
        return 2
    result = analyze(args.npz, n_seeds=args.n_seeds)
    md = render_markdown(result)
    print(md)
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "transfer_check.json").write_text(json.dumps(result, indent=2))
        (args.out_dir / "transfer_check.md").write_text(md)
        print(f"\n→ {args.out_dir}/transfer_check.json")
        print(f"→ {args.out_dir}/transfer_check.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
