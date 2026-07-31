"""E7 per-route verdict-map panels: both adapters x ID/OOD in one frame.

Alternative layout to the per-adapter panels: one panel per route
(896 / 768), four curves each — high-redundancy adapter ID/OOD and
low-redundancy adapter ID/OOD. Same paired debiased estimator, trim,
and bin-level +/-eps* recipe as plot_debiased_map.py (band = 1.645 x
median of the two runs' full-N SEMs for that route). Color encodes the
adapter (not the route, unlike the main-text figures); linestyle
encodes ID (solid) vs OOD (dashed).

Usage:
    python project/sigma_lowres/paper_bench/experiments/e7/plot_e7_routes.py \
        [--out-dir ../paper/figs] [--ylim " -0.3,0.44"]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PAPER_BENCH = HERE.parents[1]  # project/sigma_lowres/paper_bench
RUN_ROOT = PAPER_BENCH / "runs"
sys.path.insert(0, str(PAPER_BENCH))  # plot_debiased_map lives at the paper_bench root

from plot_debiased_map import paired_stats  # noqa: E402

# Artifact names: "flat" = the paper's high-redundancy cluster, "dirty" = low-redundancy.
RUNS = {"high-red.": "20260729-1349", "low-red.": "20260729-1702"}
ADAPTER_COLORS = {"high-red.": "C4", "low-red.": "C2"}  # distinct from route colors
OOD_CELL = "S3x"
ROUTES = ("896", "768")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="../paper/figs")
    ap.add_argument("--ylim", default=" -0.3,0.44")
    args = ap.parse_args()

    recs = {
        name: [json.loads(line) for line in (RUN_ROOT / run / "per_image.jsonl").open()]
        for name, run in RUNS.items()
    }
    sigma = next(iter(recs.values()))[0]["sigma_centers"]
    n_bins = len(sigma)

    for route in ROUTES:
        full = {n: paired_stats(r, route, n_bins) for n, r in recs.items()}
        eps_star = [
            1.645 * sorted(full[n][1][b] for n in RUNS)[len(RUNS) // 2]
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

        for name, rows in recs.items():
            for tag, ls, alpha in (("ID", "-", 1.0), ("OOD", "--", 0.55)):
                grecs = [
                    r for r in rows if (r.get("cell") == OOD_CELL) == (tag == "OOD")
                ]
                means, sems = paired_stats(grecs, route, n_bins)
                ax.errorbar(
                    sigma,
                    means,
                    yerr=sems,
                    color=ADAPTER_COLORS[name],
                    linestyle=ls,
                    alpha=alpha,
                    marker="o",
                    markersize=4,
                    capsize=3,
                    label=f"{name} ({tag})",
                    zorder=3,
                )
                # The sigma=1 endpoint bin is a separate endpoint-mode measurement.
                ax.plot(
                    sigma[-1],
                    means[-1],
                    marker="o",
                    markersize=4,
                    markerfacecolor="white",
                    color=ADAPTER_COLORS[name],
                    alpha=alpha,
                    zorder=4,
                )

        ax.set_xlabel(r"$\sigma$ (bin center)")
        ax.set_ylabel(r"debiased paired gap $\bar\Delta$")
        lo, hi = (float(v) for v in args.ylim.split(","))
        ax.set_ylim(lo, hi)
        ax.set_title(rf"$1024{{\to}}{route}$ route, both adapters (paired, SEM)")
        ax.legend(loc="upper right", framealpha=0.9)

        out = (
            PAPER_BENCH / args.out_dir / f"gap_debiased_e7_route{route}.png"
        ).resolve()
        fig.tight_layout()
        fig.savefig(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
