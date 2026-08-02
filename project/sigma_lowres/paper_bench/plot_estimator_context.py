"""Render the estimator-context panel (paper Fig. verdict-b) from a probe run.

The two curves every gap read is conditioned on: the redraw-floor cosine
cos(g_a, g_b) (bin mean +/- SEM) and the U-shaped native gradient magnitude
||g_bar_src(sigma)|| (bin mean, log axis), both straight from per_image.jsonl.
The original panel was split by hand from a two-panel E1 figure and had no
committed generator; this script replaces that with the same layout.

The sigma=1 endpoint bin is a different probe mode (exact endpoint, not a
stratified draw) — detached as open markers, matching plot_debiased_map.py.

Usage:
    python project/sigma_lowres/paper_bench/plot_estimator_context.py \
        [--run runs/20260801-2304-e14-ledger-probematched] \
        [--out ../paper_suggestion/figs/estimator_context.png]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

C_COS, C_NORM = "C2", "mediumpurple"


def bin_stats(recs: list[dict], field: str, n_bins: int):
    means, sems = [], []
    for b in range(n_bins):
        vals = [v for r in recs if (v := r[field][b]) is not None and math.isfinite(v)]
        n = len(vals)
        if n == 0:
            means.append(math.nan)
            sems.append(math.nan)
            continue
        mean = sum(vals) / n
        means.append(mean)
        if n == 1:
            sems.append(math.nan)
            continue
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        sems.append(math.sqrt(var / n))
    return means, sems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/20260801-2304-e14-ledger-probematched")
    ap.add_argument("--out", default="../paper_suggestion/figs/estimator_context.png")
    args = ap.parse_args()

    run_dir = (HERE / args.run).resolve()
    recs = [json.loads(line) for line in (run_dir / "per_image.jsonl").open()]
    sigma = recs[0]["sigma_centers"]
    n_bins = len(sigma)

    cos_m, cos_s = bin_stats(recs, "cos_floor", n_bins)
    nrm_m, nrm_s = bin_stats(recs, "gnorm_native", n_bins)

    has_endpoint = math.isclose(sigma[-1], 1.0, abs_tol=1e-9)
    n_curve = n_bins - 1 if has_endpoint else n_bins

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=150)
    ax2 = ax.twinx()

    ax.errorbar(
        sigma[:n_curve],
        cos_m[:n_curve],
        yerr=cos_s[:n_curve],
        color=C_COS,
        marker="s",
        markersize=5,
        capsize=0,
        lw=1.8,
        zorder=3,
        label="redraw floor cosine",
    )
    ax2.errorbar(
        sigma[:n_curve],
        nrm_m[:n_curve],
        yerr=nrm_s[:n_curve],
        color=C_NORM,
        marker="^",
        markersize=5,
        capsize=0,
        lw=1.8,
        linestyle="--",
        zorder=2,
        label="grad norm (log)",
    )
    if has_endpoint:
        for a, m, s, c, mk in (
            (ax, cos_m, cos_s, C_COS, "s"),
            (ax2, nrm_m, nrm_s, C_NORM, "^"),
        ):
            a.errorbar(
                sigma[-1],
                m[-1],
                yerr=s[-1],
                linestyle="none",
                marker=mk,
                markersize=6,
                markerfacecolor="white",
                markeredgewidth=1.2,
                capsize=0,
                color=c,
                zorder=4,
            )

    ax2.set_yscale("log")
    ax.set_xlabel(r"$\sigma$ (bin center)")
    ax.set_ylabel("redraw floor cosine", color=C_COS)
    ax2.set_ylabel("grad norm (log)", color=C_NORM)
    ax.tick_params(axis="y", colors=C_COS)
    ax2.tick_params(axis="y", which="both", colors=C_NORM)
    ax.set_title(
        r"Estimator context: floor + magnitude vs $\sigma$"
        + (rf" ($N{{=}}{len(recs)}$)" if recs else "")
    )

    out = (HERE / args.out).resolve()
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
