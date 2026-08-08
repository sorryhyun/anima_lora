"""Variant of fig_bc_comb: the resultant B+C is pinned VERTICAL at each bin,
route 1024->768, E14 probe-matched ledger (reenc ref).

The sigma axis runs obliquely (rising to the right) and each bin sits at its
TRUE sigma position. At each bin the B->C tip-to-tail chain is rotated so the
resultant stands straight up from the base: B_perp (blue) leaves the base,
C_perp (red) continues from B's tip to the resultant's tip, and B+C (black)
is the vertical base -> tip arrow at true realized-gap scale. The interior
angle between B and C stays arccos(rho). A dashed envelope traces the
resultant tips, i.e. |B+C|(sigma). Everything shares one scale (bar at
right, units of ||g_src||).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

HERE = Path(__file__).resolve().parent  # experiments/e19
LEDGER = HERE.parents[1] / "runs/20260801-2304-e14-ledger-probematched/ledger.json"
OUT = Path(__file__).parent / "bc_comb_768_vert.png"

C_B, C_C, C_NET, C_AX = "#1f77b4", "#d62728", "0.15", "#20c9ab"
E9_WINDOW = (0.56, 0.81)
K = 0.85  # ||g|| units -> axis units
SLOPE = 0.22  # oblique sigma axis: rise per x unit
X_SCALE = 15.0  # sigma in [0, 1] -> x in [0, X_SCALE]


def base_y(x: float) -> float:
    return SLOPE * x


led = json.loads(LEDGER.read_text())
centers = led["sigma_centers"]
rows = led["bc_ledger"]["768"]
n = len(rows)
xs = [s * X_SCALE for s in centers]

fig, ax = plt.subplots(figsize=(13.5, 6.4))

# oblique sigma axis (true sigma positions)
x_end = X_SCALE + 0.6
ax.annotate(
    "",
    (x_end, base_y(x_end)),
    (-0.8, base_y(-0.8)),
    arrowprops=dict(arrowstyle="-|>", color=C_AX, lw=2.4, shrinkA=0, shrinkB=0),
    annotation_clip=False,
)
ax.annotate(
    r"$\sigma$",
    (x_end - 0.05, base_y(x_end - 0.05) - 0.32),
    fontsize=13,
    color=C_AX,
    ha="center",
)

# E9 window shading: parallelogram hugging the oblique axis (true sigma span)
x0, x1 = E9_WINDOW[0] * X_SCALE, E9_WINDOW[1] * X_SCALE
band_lo, band_hi = -0.9, 1.6
ax.add_patch(
    Polygon(
        [
            (x0, base_y(x0) + band_lo),
            (x1, base_y(x1) + band_lo),
            (x1, base_y(x1) + band_hi),
            (x0, base_y(x0) + band_hi),
        ],
        closed=True,
        color="0.93",
        zorder=0,
    )
)
ax.annotate(
    "E9 window",
    ((x0 + x1) / 2, base_y((x0 + x1) / 2) + band_hi - 0.15),
    ha="center",
    fontsize=8.5,
    color="0.45",
)

r_tips = []
b_tips = []
prev_x = None
stagger = 0
for sig, x, row in zip(centers, xs, rows):
    b = math.sqrt(2 * row["S"]) * K if row["S"] > 0 else float("nan")
    c = math.sqrt(2 * row["F"]) * K if row["F"] > 0 else float("nan")
    rho = max(-1.0, min(1.0, row["rho"]))
    r = math.sqrt(max(b * b + c * c + 2 * b * c * rho, 0.0))
    base = (x, base_y(x))
    rt = (base[0], base[1] + r)  # resultant straight up
    # rotate the triangle so B+C is vertical: B at angle alpha from vertical
    if r > 1e-9:
        cos_a = max(-1.0, min(1.0, (b * b + r * r - c * c) / (2 * b * r)))
    else:
        cos_a = 0.0
    sin_a = math.sqrt(max(1.0 - cos_a * cos_a, 0.0))  # fold B toward +x
    bt = (base[0] + b * sin_a, base[1] + b * cos_a)
    ax.annotate(
        "",
        bt,
        base,
        arrowprops=dict(arrowstyle="-|>", color=C_B, lw=2.0, shrinkA=0, shrinkB=0),
        annotation_clip=False,
    )
    ax.annotate(
        "",
        rt,
        bt,
        arrowprops=dict(arrowstyle="-|>", color=C_C, lw=2.0, shrinkA=0, shrinkB=0),
        annotation_clip=False,
    )
    ax.annotate(
        "",
        rt,
        base,
        arrowprops=dict(arrowstyle="-|>", color=C_NET, lw=1.7, shrinkA=0, shrinkB=0),
        annotation_clip=False,
    )
    ax.plot(*base, "o", ms=3, color=C_NET, zorder=5)
    r_tips.append(rt)
    b_tips.append(bt)
    # cycle tick labels through three rows wherever bins crowd
    stagger = (stagger + 1) % 3 if (prev_x is not None and x - prev_x < 0.8) else 0
    prev_x = x
    y_off = -0.32 - 0.42 * stagger
    lbl = f"{sig:g}" if sig != 1.0 else "1"
    ax.annotate(
        lbl,
        (x - 0.28, base[1] + y_off),
        ha="center",
        va="top",
        fontsize=7.5,
        color="0.35",
        rotation=45,
    )

# envelope through resultant tips
ax.plot(
    [p[0] for p in r_tips],
    [p[1] for p in r_tips],
    ls=(0, (4, 3)),
    color="0.4",
    lw=1.4,
    zorder=1,
)

# labels at the bin with the widest B swing
i_max = max(range(n), key=lambda i: b_tips[i][0] - xs[i])
x_max = xs[i_max]
ax.annotate(
    r"$B^{\perp}$ (data)",
    (b_tips[i_max][0] + 0.3, b_tips[i_max][1] - 0.35),
    ha="left",
    fontsize=10,
    color=C_B,
)
ax.annotate(
    r"$C^{\perp}$ (graph)",
    (b_tips[i_max][0] + 0.35, b_tips[i_max][1] + 0.6),
    ha="left",
    fontsize=10,
    color=C_C,
)
ax.annotate(
    r"$B{+}C$ (realized gap)",
    (x_max - 0.5, r_tips[i_max][1] + 0.45),
    ha="right",
    fontsize=10,
    color=C_NET,
)
ax.annotate(
    "envelope $=|B{+}C|(\\sigma)$",
    (0.0, base_y(0.0) + 2.9),
    fontsize=8.5,
    color="0.4",
)

# scale bar
sb_x = X_SCALE - 0.2
sb_y = base_y(sb_x) + 1.0
ax.plot([sb_x, sb_x], [sb_y, sb_y + 0.5 * K], color="0.5", lw=2.5)
ax.annotate(
    r"$0.5\,\|\bar g_{\rm src}\|$",
    (sb_x + 0.18, sb_y + 0.25 * K),
    fontsize=8.5,
    color="0.45",
    va="center",
)

ax.set_xlim(-1.0, X_SCALE + 1.3)
ax.set_ylim(-0.95, base_y(X_SCALE) + 2.1)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title(
    "$1024\\to768$ — B/C legs with the resultant $B{+}C$ pinned vertical "
    "(true debiased scale, interior angle = $\\arccos\\rho$; "
    "E14 probe-matched ledger, reenc ref)",
    fontsize=11.5,
    pad=12,
)
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print(f"wrote {OUT}")
