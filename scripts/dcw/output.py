"""Output helpers: CSV writers, plot, summary printer, accumulator."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import torch

from scripts.dcw.haar import BANDS

log = logging.getLogger("dcw-bench")


def _accumulate_row(
    accum: dict,
    name: str,
    v_fwd: np.ndarray,
    fwd_bands: dict[str, np.ndarray],
    rev_norms: np.ndarray,
    rev_bands: dict[str, np.ndarray],
    per_sample_bands: dict[str, np.ndarray] | None,
    per_sample_v_rev_bands: dict[str, np.ndarray] | None,
    per_sample_stems: list[str] | None,
    img_idx: int,
    seed_idx: int,
    n_seeds: int,
    stem: str,
) -> None:
    """Fold one (img, seed, config) trajectory into the accumulator."""
    gap = rev_norms - v_fwd
    accum[name]["v_fwd"] += v_fwd
    accum[name]["v_rev"] += rev_norms
    accum[name]["gap"] += gap
    accum[name]["gap_sq"] += gap**2
    for b in BANDS:
        gap_b = rev_bands[b] - fwd_bands[b]
        accum[name]["v_fwd_bands"][b] += fwd_bands[b]
        accum[name]["v_rev_bands"][b] += rev_bands[b]
        accum[name]["gap_bands"][b] += gap_b
        if name == "baseline" and per_sample_bands is not None:
            row = img_idx * n_seeds + seed_idx
            per_sample_bands[b][row] = gap_b
            per_sample_v_rev_bands[b][row] = rev_bands[b]
            per_sample_stems[row] = stem
    accum[name]["n"] += 1


def write_per_step_csv(
    out_dir: Path, accum: dict, sigmas: torch.Tensor, n_steps: int
) -> Path:
    csv_path = out_dir / "per_step.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        headers = ["step", "sigma_i"]
        for name in accum:
            headers += [
                f"{name}_v_fwd",
                f"{name}_v_rev",
                f"{name}_gap",
                f"{name}_gap_std",
            ]
        w.writerow(headers)
        for i in range(n_steps):
            row: list = [i, float(sigmas[i])]
            for name in accum:
                row += [
                    accum[name]["v_fwd"][i],
                    accum[name]["v_rev"][i],
                    accum[name]["gap"][i],
                    accum[name]["gap_std"][i],
                ]
            w.writerow(row)
    return csv_path


def write_per_band_csv(
    out_dir: Path, accum: dict, sigmas: torch.Tensor, n_steps: int
) -> Path:
    csv_path = out_dir / "per_step_bands.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        headers = ["step", "sigma_i"]
        for name in accum:
            for b in BANDS:
                headers += [
                    f"{name}_v_fwd_{b}",
                    f"{name}_v_rev_{b}",
                    f"{name}_gap_{b}",
                ]
        w.writerow(headers)
        for i in range(n_steps):
            row: list = [i, float(sigmas[i])]
            for name in accum:
                for b in BANDS:
                    row += [
                        accum[name]["v_fwd_bands"][b][i],
                        accum[name]["v_rev_bands"][b][i],
                        accum[name]["gap_bands"][b][i],
                    ]
            w.writerow(row)
    return csv_path


def make_plot(out_dir: Path, accum: dict, n_steps: int) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping plot")
        return False

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), sharex=True)
    base = accum["baseline"]
    xs = range(n_steps)

    axes[0].plot(xs, base["v_fwd"], label="forward ‖v(x_t, t)‖", color="#2a9d8f")
    axes[0].plot(xs, base["v_rev"], label="reverse ‖v(x̂_t, t)‖", color="#e76f51")
    axes[0].set_title("Baseline forward vs reverse velocity (Fig 1c)")
    axes[0].set_xlabel("step i")
    axes[0].set_ylabel("‖v‖₂")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for name in accum:
        axes[1].plot(xs, accum[name]["gap"], label=name, alpha=0.85)
    axes[1].fill_between(
        xs,
        base["gap"] - base["gap_std"],
        base["gap"] + base["gap_std"],
        color="#888888",
        alpha=0.20,
        label="baseline ±1σ across (img×seed)",
    )
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_title("gap(i) = ‖v_rev‖ − ‖v_fwd‖  (closer to 0 = better)")
    axes[1].set_xlabel("step i")
    axes[1].set_ylabel("gap")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    band_colors = {"LL": "#264653", "LH": "#2a9d8f", "HL": "#e9c46a", "HH": "#e76f51"}
    for b in BANDS:
        axes[2].plot(xs, base["gap_bands"][b], label=b, color=band_colors[b], alpha=0.9)
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_title("Baseline gap by Haar subband")
    axes[2].set_xlabel("step i")
    axes[2].set_ylabel("gap (band)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    png_path = out_dir / "gap_curves.png"
    fig.savefig(png_path, dpi=130)
    log.info(f"plot → {png_path}")
    return True


def print_summary(accum: dict, ranked: list, dcw_sweep: bool) -> None:
    base = accum["baseline"]
    print("\n=== SNR-t bias measurement ===")
    print(
        f"baseline integrated signed gap: {base['gap'].sum():+.3f}  "
        f"(Anima flow-matching: gap is typically negative — forward > reverse — "
        f"opposite of the DDPM noise-pred sign in the paper)"
    )
    print(
        f"baseline gap std across {base['n']} (img×seed) trajectories: "
        f"mean σ_step = {base['gap_std'].mean():.3f}, "
        f"max σ_step = {base['gap_std'].max():.3f}"
    )

    print("\nbaseline integrated signed gap by Haar subband:")
    print(f"  {'band':<4s}  {'signed':>9s}  {'|gap|':>8s}")
    for b in BANDS:
        g = float(base["gap_bands"][b].sum())
        a = float(np.abs(base["gap_bands"][b]).sum())
        print(f"  {b:<4s}  {g:+9.3f}  {a:8.3f}")

    if dcw_sweep:
        print("\nconfigs ranked by integrated |gap| (smaller = closer alignment):")
        for rank, (name, a, s) in enumerate(ranked, 1):
            print(f"  {rank:>2}. {name:<24s}  |gap|={a:7.3f}  signed={s:+7.3f}")
