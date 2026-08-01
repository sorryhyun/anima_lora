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
        # A bin can trim to empty (dense E13 segments, or an OOD split with few
        # stems). NaN it rather than dividing by zero — matplotlib skips NaN
        # points, and eps_star drops them below.
        if n == 0:
            means.append(math.nan)
            sems.append(math.nan)
            continue
        mean = sum(vals) / n
        means.append(mean)
        if n == 1:
            sems.append(math.nan)  # no variance estimate from one sample
            continue
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
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
    eps_star = []
    for b in range(n_bins):
        # median over the routes that actually have a SEM in this bin
        finite = sorted(s for route in routes if math.isfinite(s := stats[route][1][b]))
        eps_star.append(1.645 * finite[len(finite) // 2] if finite else math.nan)
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

    # The sigma=1 bin is a *different probe mode* (exact endpoint, not a
    # stratified draw inside a window). Connecting it to the last interior bin
    # draws a line segment between two estimands — most of why the last leg
    # reads as a cliff. Detach it: interior bins carry the line, the endpoint
    # is an open marker with its own error bar and no connector.
    has_endpoint = math.isclose(sigma[-1], 1.0, abs_tol=1e-9)
    n_curve = n_bins - 1 if has_endpoint else n_bins

    for route in routes:
        for grecs, suffix, ls, alpha in groups:
            means, sems = paired_stats(grecs, route, n_bins)
            ax.errorbar(
                sigma[:n_curve],
                means[:n_curve],
                yerr=sems[:n_curve],
                color=ROUTE_COLORS[route],
                linestyle=ls,
                alpha=alpha,
                marker="o",
                markersize=4,
                capsize=3,
                label=rf"$\bar\Delta_{{{route}}}${suffix}",
                zorder=3,
            )
            if has_endpoint:
                ax.errorbar(
                    sigma[-1],
                    means[-1],
                    yerr=sems[-1],
                    linestyle="none",
                    marker="o",
                    markersize=5,
                    markerfacecolor="white",
                    markeredgewidth=1.2,
                    capsize=3,
                    color=ROUTE_COLORS[route],
                    alpha=alpha,
                    zorder=4,
                )

    if has_endpoint:
        ax.plot(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.2,
            color="0.35",
            label=r"$\sigma{=}1$ endpoint mode",
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
