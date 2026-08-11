"""paper_v2 observation-section figures: cancellation panel + axis field.

Figure assembly only — every number is read from committed digests;
nothing is refit and no constant is tuned.

  canc_panel.png  h(B), h(C), h(B+C) by sigma per route, with a rho strip,
                  from the E14 probe-matched ledger
                  (runs/20260801-2304-e14-ledger-probematched/ledger.json,
                  reenc ref, N=40, D=12). Gate-failing bins hollow
                  (min(rel_B, rel_C) < 0.5); the sigma=1 endpoint bin is
                  drawn detached (open marker, dotted connector).

  axis_field.png  left: pairwise debiased |cos| across routes (fixed sigma)
                  and across stores (sigma=0.7) for the legs and the
                  residual, from e24_axis.json (families) and
                  e250_read.json (r_descriptive); right: the across-sigma
                  rotation, |cos| vs sigma separation. Debiased values may
                  exceed 1; clamped to 1 for display only (E24 convention).

Usage:
    uv run python project/sigma_lowres/paper_bench/fig_canc_axis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent  # project/sigma_lowres/paper_bench
SIGMA = HERE.parent
FIGS = SIGMA / "paper_v2" / "figs"

E14_LEDGER = HERE / "runs" / "20260801-2304-e14-ledger-probematched" / "ledger.json"
E24_AXIS = HERE / "experiments" / "e24" / "e24_axis.json"
E250_READ = HERE / "experiments" / "e25" / "e250_read.json"

DISP = {"896": r"$1024\to896$", "768": r"$1024\to768$", "512": r"$1024\to512$"}
C_B, C_C, C_NET = "C0", "C3", "0.15"


def canc_panel() -> None:
    led = json.loads(E14_LEDGER.read_text())
    sigmas = led["sigma_centers"]
    fig, axes = plt.subplots(
        2, 3, figsize=(9.8, 3.9), sharex=True,
        gridspec_kw={"height_ratios": [2.6, 1.0], "hspace": 0.08, "wspace": 0.28},
    )
    for j, route in enumerate(("896", "768", "512")):
        rows = led["bc_ledger"][route]
        ax, axr = axes[0][j], axes[1][j]
        for key, color, lab, lw in (
            ("h_B", C_B, r"$h(B)$", 1.6),
            ("h_C", C_C, r"$h(C)$", 1.6),
            ("h_B_plus_C", C_NET, r"$h(B{+}C)$ (realized)", 2.2),
        ):
            xs = sigmas[:-1]
            ys = [r[key] for r in rows[:-1]]
            gated = [min(r["rel_cos_B"], r["rel_cos_C"]) >= 0.5 for r in rows[:-1]]
            ax.plot(xs, ys, "-", color=color, lw=lw, label=lab, zorder=3)
            for x, y, g in zip(xs, ys, gated):
                ax.plot(x, y, "o", ms=4, color=color,
                        mfc=(color if g else "white"), mec=color, zorder=4)
            # sigma = 1 endpoint bin: distinct probe mode, detached.
            ax.plot(sigmas[-1], rows[-1][key], "o", ms=4.5, mfc="white",
                    mec=color, mew=1.2, zorder=4)
            ax.plot(sigmas[-2:], [rows[-2][key], rows[-1][key]], ":",
                    color=color, lw=0.9, zorder=2)
        ax.set_yscale("log")
        ax.set_title(DISP[route], fontsize=10.5)
        ax.grid(True, which="major", axis="both", alpha=0.25, lw=0.5)
        # rho strip (debiased; clamp to [-1.05, 0] for display only)
        rho_x = [s for s, r in zip(sigmas[:-1], rows[:-1])]
        rho_y = [max(-1.05, min(0.0, r["rho"])) for r in rows[:-1]]
        axr.plot(rho_x, rho_y, "-", color="0.35", lw=1.3)
        axr.plot(rho_x, rho_y, "o", ms=3, color="0.35")
        axr.plot(sigmas[-1], max(-1.05, min(0.0, rows[-1]["rho"])), "o", ms=3.5,
                 mfc="white", mec="0.35", mew=1.1)
        axr.axhline(-0.91, color="0.6", lw=0.8, ls=(0, (4, 3)))
        axr.set_ylim(-1.12, 0.02)
        axr.set_yticks([-1.0, -0.5, 0.0])
        axr.set_xlabel(r"$\sigma$", fontsize=10)
        axr.grid(True, axis="y", alpha=0.25, lw=0.5)
        if j == 0:
            ax.set_ylabel("counterfactual gap $h(\\cdot)$ (log)", fontsize=9.5)
            axr.set_ylabel(r"$\rho$", fontsize=10)
            axr.annotate(r"$\bar\rho=-0.91$", (0.60, -0.38), fontsize=7.5,
                         color="0.45")
    axes[0][0].legend(fontsize=8, loc="lower left", framealpha=0.9)
    fig.savefig(FIGS / "canc_panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _pairs(fam: dict, key: str = "cos") -> list[float]:
    return [min(abs(p[key if key in p else "cos_raw"]), 1.0) for p in fam["pairs"]]


def _dsig(p: dict) -> float:
    sx = float(p["x"].rsplit("/s", 1)[1])
    sy = float(p["y"].rsplit("/s", 1)[1])
    return abs(sy - sx)


def axis_field() -> None:
    ax24 = json.loads(E24_AXIS.read_text())
    r250 = json.loads(E250_READ.read_text())
    fams = ax24["families"]
    rdesc = r250["r_descriptive"]

    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(9.8, 3.1), gridspec_kw={"width_ratios": [1.0, 1.35], "wspace": 0.25}
    )

    # Left: across-route (fixed sigma) and across-store (sigma = 0.7)
    groups = [
        (r"$\widehat{B}$", fams["B"], C_B),
        (r"$\widehat{C}$", fams["C"], C_C),
        (r"$\widehat{R}$", rdesc, C_NET),
    ]
    xt, xl = [], []
    for gi, famkey in enumerate(("across_route", "across_store")):
        for li, (lab, fam, color) in enumerate(groups):
            x = gi * 4 + li
            vals = _pairs(fam[famkey])
            axl.plot([x] * len(vals), vals, "o", ms=4.5, color=color, alpha=0.75)
            med = sorted(vals)[len(vals) // 2]
            axl.plot([x - 0.28, x + 0.28], [med, med], "-", color=color, lw=2.0)
            xt.append(x)
            xl.append(lab)
    axl.axhline(0.5, color="0.75", lw=0.8, ls=(0, (4, 3)))
    axl.set_xticks(xt)
    axl.set_xticklabels(xl, fontsize=9)
    axl.text(1.0, 0.30, "across routes\n(same $\\sigma$)", ha="center", fontsize=8.5)
    axl.text(5.0, 0.30, "across runs\n($\\sigma{=}0.7$, one env.)", ha="center", fontsize=8.5)
    axl.set_ylim(0.25, 1.04)
    axl.set_ylabel(r"debiased $|\cos|$ (clamped at 1)", fontsize=9)
    axl.set_title("one direction per $\\sigma$", fontsize=10.5)
    axl.grid(True, axis="y", alpha=0.25, lw=0.5)

    # Right: across-sigma rotation, |cos| vs sigma separation
    for lab, fam, color, marker in (
        (r"$\widehat{B}$ pairs", fams["B"]["across_sigma"], C_B, "o"),
        (r"$\widehat{R}$ pairs", rdesc["across_sigma"], C_NET, "s"),
    ):
        xs = [_dsig(p) for p in fam["pairs"]]
        ys = _pairs(fam)
        axr.plot(xs, ys, marker, ms=5, color=color, alpha=0.8, ls="none", label=lab)
    axr.set_xlabel(r"$\sigma$ separation of the pair", fontsize=10)
    axr.set_ylabel(r"debiased $|\cos|$", fontsize=9)
    axr.set_title("smooth rotation across $\\sigma$", fontsize=10.5)
    axr.set_ylim(0.35, 1.02)
    axr.grid(True, alpha=0.25, lw=0.5)
    axr.legend(fontsize=8.5, loc="lower left", framealpha=0.9)
    axr.annotate(
        "hole: $768/\\sigma{=}0.43$\n($\\widehat{R}$ unreliable; excluded)",
        xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
        fontsize=8, color="0.35",
    )
    fig.savefig(FIGS / "axis_field.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    canc_panel()
    axis_field()
    print(f"wrote {FIGS / 'canc_panel.png'}")
    print(f"wrote {FIGS / 'axis_field.png'}")
