"""Fig-4 variant with the 19.4 C-decomposition overlaid: at the PI-causal
bins (sigma 0.7 / 0.8333 / 0.9625 / 1.0) the graph leg C is drawn as the
resultant of TWO arrows, C = C_pi + Delta_phase, where

    C_pi        = g_dem,pi - g_rp   (graph leg with PI-aligned RoPE phases)
    Delta_phase = g_dem - g_dem,pi  (the phase-geometry response)

Base comb identical to fig_bc_comb.py (E14 probe-matched ledger, route
1024->768, reenc ref, true-sigma oblique axis). Overlay numbers come from
pi_194.json; at each shared bin the 19.4 triangle is rescaled by
C_e14/C_194 (~1.05) so the decomposition lands exactly on the comb's C
arrow. Drawn in the C/C_pi plane: |C_pi|, |Delta|, ang(C, C_pi) and
ang(B, C) are to scale; ang(B, C_pi) is NOT (B/C/C_pi span 3D — planar
mismatch 14-27 deg, values in pi_194.json). The sigma=1 bin is faded
(non-verdict: rel_cos_Cpi 0.58, ratio-of-small-numbers).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

HERE = Path(__file__).resolve().parent  # experiments/e19
LEDGER = HERE.parents[1] / "runs/20260801-2304-e14-ledger-probematched/ledger.json"
PI_LEDGER = HERE / "pi_194.json"
OUT = HERE / "bc_comb_768_split.png"

C_B, C_C, C_NET, C_AX = "#1f77b4", "#d62728", "0.15", "#20c9ab"
C_CPI, C_DPH = "#f08c8c", "#e8931a"  # C_pi (light red), Delta_phase (orange)
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

pi = json.loads(PI_LEDGER.read_text())
pi_rows = {
    round(s, 4): r for s, r in zip(pi["sigma_centers"], pi["pi_ledger"]["768"])
}

fig, ax = plt.subplots(figsize=(13.5, 7.2))

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

c_tips = []
b_starts = []
pi_labeled = False
prev_x = None
stagger = False
for sig, x, row in zip(centers, xs, rows):
    b = math.sqrt(2 * row["S"]) * K if row["S"] > 0 else float("nan")
    c = math.sqrt(2 * row["F"]) * K if row["F"] > 0 else float("nan")
    th = math.acos(max(-1.0, min(1.0, row["rho"])))
    base = (x, base_y(x))
    ct = (base[0], base[1] + c)  # C straight up from the base
    # B arrives at the base: tail at base - bvec, head at base
    bvec = (b * math.sin(th), b * math.cos(th))
    bstart = (base[0] - bvec[0], base[1] - bvec[1])
    ax.annotate(
        "",
        base,
        bstart,
        arrowprops=dict(arrowstyle="-|>", color=C_B, lw=2.0, shrinkA=0, shrinkB=0),
        annotation_clip=False,
    )
    ax.annotate(
        "",
        ct,
        base,
        arrowprops=dict(arrowstyle="-|>", color=C_C, lw=2.0, shrinkA=0, shrinkB=0),
        annotation_clip=False,
    )
    ax.annotate(
        "",
        ct,
        bstart,
        arrowprops=dict(arrowstyle="-|>", color=C_NET, lw=1.7, shrinkA=0, shrinkB=0),
        annotation_clip=False,
    )
    ax.plot(*base, "o", ms=3, color=C_NET, zorder=5)

    # 19.4 overlay: C = C_pi + Delta_phase, in the C/C_pi plane, tilted away
    # from B (right of vertical); rescaled so C_pi + Delta lands on C's tip
    prow = pi_rows.get(round(sig, 4))
    if prow is not None:
        alpha = 0.35 if sig == 1.0 else 1.0  # endpoint non-verdict
        c194 = math.sqrt(2 * prow["F"])
        rescale = (c / K) / c194 if c194 > 0 else 1.0
        cpi = math.sqrt(2 * prow["F_pi"]) * rescale * K
        a_ccpi = math.acos(max(-1.0, min(1.0, prow["cos_C_Cpi"])))
        cpi_tip = (base[0] + cpi * math.sin(a_ccpi), base[1] + cpi * math.cos(a_ccpi))
        ax.annotate(
            "",
            cpi_tip,
            base,
            arrowprops=dict(
                arrowstyle="-|>", color=C_CPI, lw=1.6, shrinkA=0, shrinkB=0, alpha=alpha
            ),
            annotation_clip=False,
        )
        ax.annotate(
            "",
            ct,
            cpi_tip,
            arrowprops=dict(
                arrowstyle="-|>", color=C_DPH, lw=1.6, shrinkA=0, shrinkB=0, alpha=alpha
            ),
            annotation_clip=False,
        )
        pi_labeled = True

    c_tips.append(ct)
    b_starts.append(bstart)
    # stagger tick labels into two rows wherever bins crowd
    stagger = not stagger if (prev_x is not None and x - prev_x < 0.8) else False
    prev_x = x
    y_off = -0.32 - (0.42 if stagger else 0.0)
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

# envelope through C tips
ax.plot(
    [p[0] for p in c_tips],
    [p[1] for p in c_tips],
    ls=(0, (4, 3)),
    color="0.4",
    lw=1.4,
    zorder=1,
)

# labels at the tallest bin
i_max = max(range(n), key=lambda i: c_tips[i][1] - base_y(xs[i]))
x_max = xs[i_max]
ax.annotate(
    r"$C^{\perp}$ (graph)",
    (x_max + 0.5, base_y(x_max) + (c_tips[i_max][1] - base_y(x_max)) * 0.85),
    ha="left",
    fontsize=10,
    color=C_C,
)
ax.annotate(
    r"$B^{\perp}$ (data)",
    (b_starts[i_max][0] - 0.2, b_starts[i_max][1] + 0.1),
    ha="right",
    fontsize=10,
    color=C_B,
)
ax.annotate(
    r"$B{+}C$ (realized gap)",
    (b_starts[i_max][0] + 0.5, c_tips[i_max][1] + 0.4),
    ha="right",
    fontsize=10,
    color=C_NET,
)
ax.annotate(
    "envelope $=|C^{\\perp}|(\\sigma)$",
    (0.0, base_y(0.0) + 2.9),
    fontsize=8.5,
    color="0.4",
)

# 19.4 overlay legend (arrows themselves stay unlabeled to avoid clutter)
leg_x, leg_y = 0.3, base_y(0.3) + 5.3
ax.annotate(
    "19.4 overlay ($\\sigma \\geq 0.7$):  $C = C_\\pi + \\Delta_{\\rm phase}$",
    (leg_x, leg_y),
    fontsize=10,
    color="0.25",
)
ax.annotate(
    r"$C_\pi^{\perp} = \bar g_{{\rm dem},\pi} - \bar g_{\rm rp}$ (PI-aligned graph leg)",
    (leg_x + 0.4, leg_y - 0.5),
    fontsize=9.5,
    color=C_CPI,
)
ax.annotate(
    r"$\Delta_{\rm phase} = \bar g_{\rm dem} - \bar g_{{\rm dem},\pi}$ (phase-geometry response)",
    (leg_x + 0.4, leg_y - 1.0),
    fontsize=9.5,
    color=C_DPH,
)

# scale bar
sb_x = X_SCALE - 0.2
sb_y = base_y(sb_x) + 2.2
ax.plot([sb_x, sb_x], [sb_y, sb_y + 0.5 * K], color="0.5", lw=2.5)
ax.annotate(
    r"$0.5\,\|\bar g_{\rm src}\|$",
    (sb_x + 0.18, sb_y + 0.25 * K),
    fontsize=8.5,
    color="0.45",
    va="center",
)

ax.set_xlim(-1.0, X_SCALE + 1.3)
ax.set_ylim(-0.95, base_y(X_SCALE) + 2.4)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title(
    "$1024\\to768$ — B/C legs on the $\\sigma$ axis; at the 19.4 bins the graph leg "
    "splits as $C = C_\\pi + \\Delta_{\\rm phase}$\n"
    "(E14 ledger + pi_194, reenc ref; $C/C_\\pi$ plane — "
    "$\\angle(B, C_\\pi)$ not to scale; $\\sigma{=}1$ faded = non-verdict)",
    fontsize=11,
    pad=12,
)
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print(f"wrote {OUT}")
