"""Render the debiased verdict map (paper Fig. 1c) from an E1b per_image.jsonl.

Reproduces Table 2's recipe exactly: per-image paired delta
(debiased_gap_<route> - debiased_gap_reenc), non-finite and |delta| > 1.5
trimmed, bin mean +/- SEM. The gray band is the bin-level instrument
resolution +/-eps* = 1.645 x route-median SEM (Eq. epsstar); exact per-route
SEMs stay in Table 2.

Usage:
    python project/sigma_lowres/paper_bench/plot_debiased_map.py \
        [--run runs/20260729-0014-e1b-debiased-map] [--out ../paper/figs/gap_debiased.png]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

ROUTES = ("896", "768", "512")
ROUTE_COLORS = {"896": "C0", "768": "C1", "512": "C3"}  # matches the raw gap_curves fig
TRIM_ABS = 1.5


def paired_stats(recs: list[dict], route: str, n_bins: int):
    means, sems = [], []
    for b in range(n_bins):
        vals = []
        for r in recs:
            d, c = r[f"debiased_gap_{route}"][b], r["debiased_gap_reenc"][b]
            if d is None or c is None:
                continue
            delta = d - c
            if not math.isfinite(delta) or abs(delta) > TRIM_ABS:
                continue
            vals.append(delta)
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        means.append(mean)
        sems.append(math.sqrt(var / n))
    return means, sems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/20260729-0014-e1b-debiased-map")
    ap.add_argument("--out", default="../paper/figs/gap_debiased.png")
    args = ap.parse_args()

    run_dir = (HERE / args.run).resolve()
    recs = [json.loads(line) for line in (run_dir / "per_image.jsonl").open()]
    sigma = recs[0]["sigma_centers"]
    n_bins = len(sigma)

    stats = {route: paired_stats(recs, route, n_bins) for route in ROUTES}
    eps_star = [
        1.645 * sorted(stats[route][1][b] for route in ROUTES)[len(ROUTES) // 2]
        for b in range(n_bins)
    ]

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=150)
    ax.fill_between(
        sigma,
        [-e for e in eps_star],
        eps_star,
        color="0.5",
        alpha=0.35,
        linewidth=0,
        label=r"$\pm\varepsilon^{*}$ (bin-level)",
        zorder=1,
    )
    ax.axhline(0.0, color="black", linewidth=0.8, zorder=2)

    for route in ROUTES:
        means, sems = stats[route]
        ax.errorbar(
            sigma,
            means,
            yerr=sems,
            color=ROUTE_COLORS[route],
            marker="o",
            markersize=4,
            capsize=3,
            label=rf"$\bar\Delta_{{{route}}}$",
            zorder=3,
        )
        # The sigma=1 endpoint bin is a separate endpoint-mode measurement.
        ax.plot(
            sigma[-1],
            means[-1],
            marker="o",
            markersize=4,
            markerfacecolor="white",
            color=ROUTE_COLORS[route],
            zorder=4,
        )

    ax.set_xlabel(r"$\sigma$ (bin center)")
    ax.set_ylabel(r"debiased paired gap $\bar\Delta$")
    ax.set_title(r"Debiased demotion gap vs $\sigma$ (paired, 40 images, SEM)")
    ax.legend(loc="upper right", framealpha=0.9)

    out = (HERE / args.out).resolve()
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
