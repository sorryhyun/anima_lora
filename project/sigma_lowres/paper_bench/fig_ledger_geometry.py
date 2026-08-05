"""Ledger-geometry figure: the B/C split as a round trip on the data--graph
lattice (companion to the ``ledger's geometry'' paragraph and Table
"e9ledger").

Panel 1 is the 2x2 lattice schematic: the three realizable arms, the
unrealizable fourth corner (native content on the coarse graph), and B / C as
the partial differences along the lattice's only realizable path. Panels 2-4
draw the measured legs tip-to-tail per route at the strongest in-window bin
(sigma = 0.5625): |B_perp| and |C_perp| to scale in units of ||g_src||, the
angle from the debiased rho (clamped to [-1, 1] for drawing), and the
resultant B+C. Equal aspect and shared limits across the three routes, so the
896/768 stubs and the 512 survivor are read against one scale.

Figure assembly only: every number is read from the published vector-ledger
reanalysis (ledger.json of the N=24, D=8, two-draw-set run behind Table
"e9ledger"); nothing is refit.

Usage:
    uv run python project/sigma_lowres/paper_bench/fig_ledger_geometry.py
"""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).resolve().parent  # project/sigma_lowres/paper_bench
SIGMA = HERE.parent  # project/sigma_lowres
RUNS = HERE / "runs"
FIG_DIRS = (SIGMA / "paper_suggestion" / "figs",)

LEDGER = SIGMA / "bench" / "results" / "20260731-0721" / "ledger.json"

DISP = {"896": r"$1024\to896$", "768": r"$1024\to768$",
        "512": r"$1024\to512$"}
C_B, C_C, C_NET = "C0", "C3", "0.15"


def arrow(ax, p, q, color, lw=1.8, ls="-", zorder=3, mut=11):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=mut, color=color,
        lw=lw, linestyle=ls, zorder=zorder, shrinkA=0, shrinkB=0))


def draw_lattice(ax):
    src, rp, dem, ghost = (0.1, 0.85), (0.9, 0.85), (0.9, 0.15), (0.1, 0.15)
    ax.plot(*src, "o", ms=7, color=C_NET, zorder=4)
    ax.plot(*dem, "o", ms=7, color=C_NET, zorder=4)
    ax.plot(*rp, "o", ms=7, mfc="white", mec=C_NET, mew=1.4, zorder=4)
    ax.plot(*ghost, "x", ms=8, color="0.6", mew=1.6, zorder=4)
    arrow(ax, src, rp, C_B)
    arrow(ax, rp, dem, C_C)
    arrow(ax, src, dem, C_NET, lw=1.3, ls=(0, (4, 3)), mut=9)
    ax.annotate(r"$B$ (data)", (0.5, 0.80), ha="center", va="top",
                fontsize=8.5, color=C_B)
    ax.annotate(r"$C$ (graph)", (0.86, 0.5), ha="right", va="center",
                fontsize=8.5, color=C_C)
    ax.annotate(r"$B{+}C$", (0.38, 0.36), ha="center", fontsize=8.5,
                color=C_NET, rotation=-41)
    ax.annotate(r"$\bar g_{\rm src}$", (src[0], src[1] + 0.09),
                ha="center", va="bottom", fontsize=8.5, color=C_NET)
    ax.annotate(r"$\bar g_{\rm repromote}$", (rp[0], rp[1] + 0.09),
                ha="center", va="bottom", fontsize=8.5, color=C_NET)
    ax.annotate(r"$\bar g_{\rm dem}$", (dem[0], dem[1] - 0.10),
                ha="center", va="top", fontsize=8.5, color=C_NET)
    ax.annotate("unrealizable", (ghost[0], ghost[1] - 0.10),
                ha="center", va="top", fontsize=7.5, color="0.55")
    ax.annotate("fine graph", (0.5, 1.16), ha="center", fontsize=7.5,
                color="0.45")
    ax.annotate("coarse graph", (0.5, -0.19), ha="center", va="top",
                fontsize=7.5, color="0.45")
    ax.annotate("native data", (-0.16, 0.5), ha="center", va="center",
                fontsize=7.5, color="0.45", rotation=90)
    ax.annotate("demoted data", (1.16, 0.5), ha="center", va="center",
                fontsize=7.5, color="0.45", rotation=270)
    ax.annotate("filled = matched   open = mismatched", (0.5, -0.36),
                ha="center", va="top", fontsize=6.8, color="0.55")
    ax.set_xlim(-0.30, 1.30)
    ax.set_ylim(-0.50, 1.30)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("the data–graph lattice", fontsize=10)


def draw_route(ax, route, row, scalebar=False):
    b, c, rho = row["b_perp_rawnorm"], row["c_perp_rawnorm"], row["rho"]
    th = math.acos(max(-1.0, min(1.0, rho)))
    o = (0.0, 0.0)
    bt = (b, 0.0)
    ct = (bt[0] + c * math.cos(th), bt[1] + c * math.sin(th))
    net = math.hypot(*ct)
    arrow(ax, o, bt, C_B, lw=2.0)
    arrow(ax, bt, ct, C_C, lw=2.0)
    arrow(ax, o, ct, C_NET, lw=2.4, mut=13)
    ax.plot(0, 0, "o", ms=4, color=C_NET, zorder=4)
    ax.annotate(r"$B^{\perp}$", (b / 2, -0.055), ha="center", va="top",
                fontsize=9, color=C_B)
    ax.annotate(r"$C^{\perp}$", ((bt[0] + ct[0]) / 2 + 0.05,
                                 (bt[1] + ct[1]) / 2 + 0.03),
                ha="left", fontsize=9, color=C_C)
    ax.annotate(r"$B{+}C$", (ct[0] / 2 - 0.045, ct[1] / 2 + 0.02),
                ha="right", fontsize=9, color=C_NET)
    ax.annotate(
        f"$\\rho = {rho:+.2f}$\n"
        f"$h(B){{+}}h(C) = {row['h_B'] + row['h_C']:.2f}"
        f"\\;\\rightarrow\\; h(B{{+}}C) = {row['h_B_plus_C']:.2f}$",
        (0.20, -0.30), ha="center", va="top", fontsize=7.5, linespacing=1.6)
    if scalebar:
        ax.plot([-0.38, 0.12], [0.82, 0.82], color="0.6", lw=2)
        ax.annotate(r"$0.5\,\|\bar g_{\rm src}\|$", (-0.13, 0.76),
                    ha="center", va="top", fontsize=7.5, color="0.45")
    ax.set_xlim(-0.42, 0.82)
    ax.set_ylim(-0.62, 0.96)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(DISP[route], fontsize=10)
    return net


def main() -> None:
    ledger = json.loads(LEDGER.read_text())
    bc = ledger["bc_ledger"]
    assert abs(ledger["sigma_centers"][0] - 0.5625) < 1e-9

    fig, axes = plt.subplots(1, 4, figsize=(9.8, 2.9))
    draw_lattice(axes[0])
    nets = {}
    for ax, route in zip(axes[1:], ("896", "768", "512")):
        nets[route] = draw_route(ax, route, bc[route][0],
                                 scalebar=(route == "896"))
    fig.tight_layout(w_pad=0.4)

    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M")
    out_dir = RUNS / f"{stamp}-fig-ledger-geometry"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ledger_geometry.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    for fd in FIG_DIRS:
        if fd.is_dir():
            shutil.copy2(out, fd / "ledger_geometry.png")

    envelope = dict(
        schema_version=1,
        script="project/sigma_lowres/paper_bench/fig_ledger_geometry.py",
        label="fig-ledger-geometry",
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        sources=dict(ledger=str(LEDGER)),
        metrics=dict(net_resultant_sigma0563=nets),
        caveats=[
            "figure assembly only: all numbers read from the vector-ledger "
            "reanalysis behind Table e9ledger; nothing refit",
            "vector panels draw the sigma=0.5625 bin; the drawn angle uses "
            "the debiased rho clamped to [-1, 1] (the estimator itself is "
            "not confined to that range)",
            "lattice panel is schematic (arm identities, not magnitudes)",
        ],
    )
    (out_dir / "result.json").write_text(json.dumps(envelope, indent=2))
    print(f"wrote {out} (+ figs/ copies); resultants {nets}")


if __name__ == "__main__":
    main()
