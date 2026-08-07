"""One-picture version (user sketch): B/C leg pairs planted along a single
sigma axis, route 1024->768, E14 probe-matched ledger (reenc ref).

At each bin position (equally spaced so dense low-sigma bins don't overlap):
B_perp (blue) points straight up at true debiased scale, C_perp (red) folds
back from B's tip at the bin's arccos(rho), and the resultant B+C (black)
runs base -> C tip. A dashed envelope traces the B-tip heights. Everything
shares one scale (bar at right, units of ||g_src||).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent  # experiments/e19
LEDGER = HERE.parents[1] / "runs/20260801-2304-e14-ledger-probematched/ledger.json"
OUT = Path(__file__).parent / "bc_comb_768.png"

C_B, C_C, C_NET, C_AX = "#1f77b4", "#d62728", "0.15", "#20c9ab"
E9_WINDOW = (0.56, 0.81)
K = 0.85  # ||g|| units -> axis units

led = json.loads(LEDGER.read_text())
centers = led["sigma_centers"]
rows = led["bc_ledger"]["768"]
n = len(rows)

fig, ax = plt.subplots(figsize=(13.5, 5.2))

# sigma axis (equally spaced bin slots)
ax.annotate(
    "",
    (n + 0.6, 0),
    (-0.8, 0),
    arrowprops=dict(arrowstyle="-|>", color=C_AX, lw=2.4, shrinkA=0, shrinkB=0),
    annotation_clip=False,
)
ax.annotate(r"$\sigma$", (n + 0.55, -0.28), fontsize=13, color=C_AX, ha="center")

# E9 window shading (in slot coordinates)
w0 = min(i for i, c in enumerate(centers) if c >= E9_WINDOW[0])
w1 = max(i for i, c in enumerate(centers) if c <= E9_WINDOW[1])
ax.axvspan(w0 - 0.45, w1 + 0.45, color="0.93", zorder=0)
ax.annotate(
    "E9 window",
    ((w0 + w1) / 2, 1.9),
    ha="center",
    fontsize=8.5,
    color="0.45",
)

b_tips = []
for i, (sig, row) in enumerate(zip(centers, rows)):
    b = math.sqrt(2 * row["S"]) * K if row["S"] > 0 else float("nan")
    c = math.sqrt(2 * row["F"]) * K if row["F"] > 0 else float("nan")
    th = math.acos(max(-1.0, min(1.0, row["rho"])))
    base = (i, 0.0)
    bt = (i, b)  # B straight up
    # C at angle theta from B's direction (90 deg): folds back down-right
    phi = math.pi / 2 - th
    ct = (bt[0] + c * math.cos(phi), bt[1] + c * math.sin(phi))
    ax.annotate(
        "",
        bt,
        base,
        arrowprops=dict(arrowstyle="-|>", color=C_B, lw=2.0, shrinkA=0, shrinkB=0),
        annotation_clip=False,
    )
    ax.annotate(
        "",
        ct,
        bt,
        arrowprops=dict(arrowstyle="-|>", color=C_C, lw=2.0, shrinkA=0, shrinkB=0),
        annotation_clip=False,
    )
    ax.annotate(
        "",
        ct,
        base,
        arrowprops=dict(arrowstyle="-|>", color=C_NET, lw=1.7, shrinkA=0, shrinkB=0),
        annotation_clip=False,
    )
    ax.plot(*base, "o", ms=3, color=C_NET, zorder=5)
    b_tips.append(bt)
    lbl = f"{sig:g}" if sig != 1.0 else "1"
    ax.annotate(
        lbl,
        (i, -0.30),
        ha="center",
        va="top",
        fontsize=7.5,
        color="0.35",
        rotation=45,
    )

# envelope through B tips
ax.plot(
    [p[0] for p in b_tips],
    [p[1] for p in b_tips],
    ls=(0, (4, 3)),
    color="0.4",
    lw=1.4,
    zorder=1,
)

# labels at the tallest bin
i_max = max(range(n), key=lambda i: b_tips[i][1])
ax.annotate(
    r"$B^{\perp}$ (data)",
    (i_max - 2.6, b_tips[i_max][1] * 0.62),
    ha="right",
    fontsize=10,
    color=C_B,
)
ax.annotate(
    r"$C^{\perp}$ (graph)",
    (i_max + 0.9, b_tips[i_max][1] * 0.95),
    ha="left",
    fontsize=10,
    color=C_C,
)
ax.annotate(
    r"$B{+}C$ (realized gap)",
    (i_max + 1.1, -0.85),
    ha="left",
    fontsize=10,
    color=C_NET,
)
ax.annotate(
    "envelope $=|B^{\\perp}|(\\sigma)$",
    (0.6, 1.45),
    fontsize=8.5,
    color="0.4",
)

# scale bar
sb_x = n - 0.2
ax.plot([sb_x, sb_x], [1.0, 1.0 + 0.5 * K], color="0.5", lw=2.5)
ax.annotate(
    r"$0.5\,\|\bar g_{\rm src}\|$",
    (sb_x + 0.18, 1.0 + 0.25 * K),
    fontsize=8.5,
    color="0.45",
    va="center",
)

ax.set_xlim(-1.0, n + 1.3)
ax.set_ylim(-0.95, 2.1)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title(
    "$1024\\to768$ — B/C legs planted on the $\\sigma$ axis "
    "(true debiased scale, angle = $\\arccos\\rho$; "
    "E14 probe-matched ledger, reenc ref)",
    fontsize=11.5,
    pad=12,
)
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print(f"wrote {OUT}")
