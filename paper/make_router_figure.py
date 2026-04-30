"""Render the router-dynamics figure for paper/ortho_hydra.tex.

Three-way comparison of HydraLoRA routing dynamics:
    ortho     — existing 28k run (use_ortho=true, disjoint basis)
    naive     — shared basis, no symmetry-breaker (use_ortho=false, sigma=0)
    jittered  — shared basis + Gaussian on lora_up (expert_init_std=0.1)

Reads ``metrics.json`` exports under ``output/logs/`` (produced by
``scripts/export_logs_json.py``). Output: ``paper/router_dynamics.pdf``.

Single-panel layout:
* X-axis is symlog(linthresh=1000): linear 0–1k, log 1k–28k. Gives the
  deadlock window comparable visual width to the long ortho tail.
* Left y-axis: router entropy (solid) and top1−top2 margin (dashed) per
  variant. Coloured by variant.
* Right y-axis: training loss (loss/average) per variant — proof of life
  showing the network keeps training while the router stays uniform.
* Upper-right inset: 0–1k window with y-axis zoomed to [0.997, 1.0006]
  so the slow drift of the deadlocked variants is visible against the
  uniform-prior ceiling.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "output" / "logs"
RUN_GROUPS: dict[str, list[str]] = {
    "ortho":    ["20260429123534", "20260429200319", "20260430003821"],
    "naive":    ["20260430152118"],
    "jittered": ["20260430154856"],
}
PLOT_ORDER = ["naive", "jittered", "ortho"]
COLORS = {"ortho": "#e45756", "naive": "#4c78a8", "jittered": "#59a14f"}
DISPLAY = {
    "ortho":    r"ortho (disjoint)",
    "naive":    r"naive (shared, $\sigma{=}0$)",
    "jittered": r"jittered (shared, $\sigma{=}0.1$)",
}
OUT = REPO / "paper" / "router_dynamics.pdf"


def _load_tag(run: str, tag: str) -> list[tuple[int, float]]:
    p = LOGS / run / "metrics.json"
    d = json.loads(p.read_text())["tags"]
    return [(row[0], row[2]) for row in d.get(tag, [])]


def _concat(runs: list[str], tag: str) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[int, float]] = []
    for r in runs:
        rows.extend(_load_tag(r, tag))
    rows.sort(key=lambda x: x[0])
    if not rows:
        return np.array([]), np.array([])
    seen: dict[int, float] = {}
    for s, v in rows:
        seen[s] = v  # last write wins → continuation overwrites overlap
    steps = np.array(sorted(seen.keys()))
    vals = np.array([seen[s] for s in steps])
    return steps, vals


def _smooth(x: np.ndarray, win: int = 25) -> np.ndarray:
    if x.size < win:
        return x
    k = np.ones(win) / win
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, k, mode="valid")[: x.size]


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("0.55")

    # Loss first (drawn behind router curves).
    for variant in PLOT_ORDER:
        runs = RUN_GROUPS[variant]
        s_L, L = _concat(runs, "loss/average")
        if s_L.size:
            ax2.plot(s_L, _smooth(L, win=80), color=COLORS[variant],
                     linewidth=0.9, linestyle=(0, (1, 1.6)), alpha=0.55,
                     zorder=1.5)

    # Router entropy.
    for variant in PLOT_ORDER:
        runs = RUN_GROUPS[variant]
        s_H, H = _concat(runs, "hydra/router_entropy")
        c = COLORS[variant]
        ax.plot(s_H, _smooth(H), color=c, linewidth=1.5, zorder=3)

    # Symlog x with custom ticks: linear segment 0–1k, log segment 1k–28k.
    ax.set_xscale("symlog", linthresh=1000, linscale=1.0)
    ax.set_xlim(0, 30000)
    ax.set_ylim(-0.03, 1.18)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ticks = [0, 200, 500, 1000, 2000, 5000, 10000, 28000]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0", "200", "500", "1k", "2k", "5k", "10k", "28k"])
    ax.minorticks_off()
    ax.axhline(1.0, color="0.7", linestyle=":", linewidth=0.6, zorder=0)
    ax.axvline(1000, color="0.85", linestyle="-", linewidth=0.5, zorder=0)
    ax.text(1000, 1.16, r"$\leftarrow$ linear  $\mid$  log $\rightarrow$",
            fontsize=6.8, color="0.45", ha="center", va="top")
    # Balance-loss schedule: weight = 0 for the first 40% of training,
    # then flips to target. Step 11266 = 0.4 × 28164 max steps. The bump
    # in ortho's entropy/margin around this step is the router being
    # pulled back toward uniform after specialization.
    ax.axvline(11266, color=COLORS["ortho"], linestyle=":", linewidth=0.8,
               alpha=0.7, zorder=0)
    ax.text(11266, 0.04, "balance loss on\n(warmup end, 0.4)",
            fontsize=6.5, color=COLORS["ortho"], alpha=0.9,
            ha="center", va="bottom")
    ax.text(0.005, 1.005, r"uniform prior ($\bar H = 1$)",
            transform=ax.get_yaxis_transform(),
            fontsize=7, color="0.45", va="bottom", ha="left")
    ax.set_xlabel("training step")
    ax.set_ylabel("router statistic")

    # Right axis: zoomed to [0.05, 0.10] — the band all three runs settle
    # into. Continuation-run rolling-avg warmup spikes (e.g. ortho's third
    # run at step ~24k) clip out the top, which is fine — the proof-of-life
    # signal is the steady state, not the resets.
    ax2.set_ylim(0.07, 0.10)
    ax2.set_yticks([0.07, 0.08, 0.09, 0.10])
    ax2.set_ylabel("training loss (rolling avg)", color="0.35")
    ax2.tick_params(axis="y", colors="0.35", labelsize=8)

    # Inset: 0–1k window with y zoomed near the uniform-prior ceiling.
    # Naive/jittered are flat against H=1 in the main panel — magnifying
    # y reveals jittered drifting slightly faster than naive while ortho
    # already dives off-axis.
    axin = ax.inset_axes([0.70, 0.62, 0.27, 0.32], zorder=10)
    axin.set_facecolor("white")
    axin.patch.set_alpha(1.0)
    for variant in PLOT_ORDER:
        runs = RUN_GROUPS[variant]
        s_H, H = _concat(runs, "hydra/router_entropy")
        if s_H.size:
            axin.plot(s_H, _smooth(H, win=15), color=COLORS[variant],
                      linewidth=1.2)
    axin.set_xlim(0, 1000)
    axin.set_ylim(0.997, 1.0006)
    axin.set_xticks([0, 500, 1000])
    axin.set_xticklabels(["0", "500", "1k"], fontsize=6.5)
    axin.set_yticks([0.997, 0.999, 1.000])
    axin.set_yticklabels(["0.997", "0.999", "1.000"], fontsize=6.5)
    axin.axhline(1.0, color="0.7", linestyle=":", linewidth=0.5, zorder=0)
    axin.tick_params(length=2.5, width=0.5)
    for sp in axin.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color("0.55")
    axin.set_title(r"0–1k zoom of $\bar H$",
                   fontsize=6.8, pad=2, color="0.25")

    # Legends. Two compact frames in the lower-left.
    variant_handles = [
        Line2D([0], [0], color=COLORS[v], linewidth=1.6, label=DISPLAY[v])
        for v in PLOT_ORDER
    ]
    metric_handles = [
        Line2D([0], [0], color="0.3", linewidth=1.5, linestyle="-",
               label=r"router entropy $\bar H$ (left axis)"),
        Line2D([0], [0], color="0.3", linewidth=0.9,
               linestyle=(0, (1, 1.6)), alpha=0.7,
               label=r"training loss (right axis)"),
    ]
    leg1 = ax.legend(handles=variant_handles, frameon=False, fontsize=7.4,
                     loc="lower left", bbox_to_anchor=(0.005, 0.22),
                     handlelength=1.6, borderaxespad=0.0)
    ax.add_artist(leg1)
    ax.legend(handles=metric_handles, frameon=False, fontsize=7.4,
              loc="lower left", bbox_to_anchor=(0.005, 0.02),
              handlelength=1.9, borderaxespad=0.0)

    fig.tight_layout()
    fig.savefig(OUT, format="pdf", bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {OUT}")
    for v in PLOT_ORDER:
        s_H, H = _concat(RUN_GROUPS[v], "hydra/router_entropy")
        s_M, M = _concat(RUN_GROUPS[v], "hydra/router_margin")
        s_L, L = _concat(RUN_GROUPS[v], "loss/average")
        if H.size:
            print(f"  {v:<9}  steps {s_H[0]}..{s_H[-1]}  "
                  f"H_end={H[-1]:.4f}  M_end={M[-1]:.4f}  "
                  f"loss_end={L[-1]:.4f}")


if __name__ == "__main__":
    main()
