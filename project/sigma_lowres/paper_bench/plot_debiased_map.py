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


def detect_routes(rec: dict) -> tuple[str, ...]:
    """Routes actually present in a per-image row (E7 runs carry no 512 arm)."""
    return tuple(r for r in ROUTES if f"debiased_gap_{r}" in rec)


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
    ap.add_argument("--title", default=None, help="override the axes title")
    ap.add_argument(
        "--split-ood-cell",
        default=None,
        help="cell tag treated as OOD (e.g. S3x for E7 runs): plot ID and OOD "
        "stems as separate curves per route",
    )
    ap.add_argument(
        "--ylim",
        default=None,
        help="shared y-axis limits as 'lo,hi' (for side-by-side panels)",
    )
    args = ap.parse_args()

    run_dir = (HERE / args.run).resolve()
    recs = [json.loads(line) for line in (run_dir / "per_image.jsonl").open()]
    sigma = recs[0]["sigma_centers"]
    n_bins = len(sigma)
    routes = detect_routes(recs[0])

    stats = {route: paired_stats(recs, route, n_bins) for route in routes}
    eps_star = [
        1.645 * sorted(stats[route][1][b] for route in routes)[len(routes) // 2]
        for b in range(n_bins)
    ]
    groups = [(recs, "", "-", 1.0)]
    if args.split_ood_cell:
        id_recs = [r for r in recs if r.get("cell") != args.split_ood_cell]
        ood_recs = [r for r in recs if r.get("cell") == args.split_ood_cell]
        groups = [(id_recs, " (ID)", "-", 1.0), (ood_recs, " (OOD)", "--", 0.55)]

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

    for route in routes:
        for grecs, suffix, ls, alpha in groups:
            means, sems = paired_stats(grecs, route, n_bins)
            ax.errorbar(
                sigma,
                means,
                yerr=sems,
                color=ROUTE_COLORS[route],
                linestyle=ls,
                alpha=alpha,
                marker="o",
                markersize=4,
                capsize=3,
                label=rf"$\bar\Delta_{{{route}}}${suffix}",
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
                alpha=alpha,
                zorder=4,
            )

    ax.set_xlabel(r"$\sigma$ (bin center)")
    ax.set_ylabel(r"debiased paired gap $\bar\Delta$")
    if args.ylim:
        lo, hi = (float(v) for v in args.ylim.split(","))
        ax.set_ylim(lo, hi)
    title = (
        args.title
        or rf"Debiased demotion gap vs $\sigma$ (paired, {len(recs)} images, SEM)"
    )
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.9)

    out = (HERE / args.out).resolve()
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
