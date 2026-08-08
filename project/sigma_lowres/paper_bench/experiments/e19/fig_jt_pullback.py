"""The E19 L1 verdict in one picture: the same two interventions (B = data,
C = graph), BEFORE vs AFTER the pull-back through J^T.

Two mirrored combs on a shared sigma axis, route 1024->768, reenc/area:

    top    r-level legs (x-space residual contrasts, measured_192.json)
    bottom g-level legs (parameter-space gradient contrasts, E14 ledger)

Every triangle is SHAPE-NORMALIZED (|C_perp| = 1): the two levels live in
different spaces with incomparable units, so only the angle arccos(rho) and
the ratio |B|/|C| are drawn -- exactly the two quantities the L1 verdict is
about. At the r level the legs are near-perpendicular (99-103 deg mid-sigma)
with ratio ~0.4; after pull-back the SAME interventions read near-opposite
(155-164 deg) with ratio -> 1, and the resultant B+C collapses. The angle
opening AND the magnitude matching are both made in gradient formation --
"J^T-born" (19.2), with no r-level crossing anywhere (19.2 item 3).

Honesty: low-sigma bins faded (non-verdict, low-sigma caution); g-level
sigma=1 rho = -1.019 drawn clamped (out-of-domain, E14 convention); the
per-bin completeness readout 1 - |B+C|/(|B|+|C|) is computed from the drawn
triangles (geometric, not the ledger h(.) units).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent  # experiments/e19
E14_LEDGER = HERE.parents[1] / "runs/20260801-2304-e14-ledger-probematched/ledger.json"
R_LEDGER = HERE / "measured_192.json"
OUT = HERE / "jt_pullback_768.png"

C_B, C_C, C_NET = "#1f77b4", "#d62728", "0.15"
SLOT = 1.55
BASE_GAP = 0.42  # baseline offset of each comb from the central sigma axis
VERDICT_MIN = 0.3

g_led = json.loads(E14_LEDGER.read_text())
centers = g_led["sigma_centers"]
g_rows = g_led["bc_ledger"]["768"]
r_rows = json.loads(R_LEDGER.read_text())["routes"]["768"]["reenc_area"]["table"]
n = len(centers)

# per-bin (theta, ratio): r-level from rho + amp_ratio; g-level from rho + S/F
r_geom = [
    (math.acos(max(-1.0, min(1.0, r["rho"]))), r["amp_ratio_BoverC"]) for r in r_rows
]
g_geom = [
    (
        math.acos(max(-1.0, min(1.0, r["rho"]))),
        math.sqrt(r["S"] / r["F"]) if r["S"] > 0 and r["F"] > 0 else float("nan"),
    )
    for r in g_rows
]


def completeness(theta: float, ratio: float) -> float:
    res = math.hypot(ratio * math.sin(theta), 1.0 + ratio * math.cos(theta))
    return 1.0 - res / (1.0 + ratio)


fig, ax = plt.subplots(figsize=(16.0, 6.8))

for i, sig in enumerate(centers):
    x = i * SLOT
    alpha = 1.0 if sig >= VERDICT_MIN else 0.4
    for sign, (theta, ratio) in (( +1, r_geom[i]), (-1, g_geom[i])):
        base = (x, sign * BASE_GAP)
        ct = (x, base[1] + sign * 1.0)  # C, unit length, away from the axis
        bvec = (ratio * math.sin(theta), sign * ratio * math.cos(theta))
        bstart = (base[0] - bvec[0], base[1] - bvec[1])
        for tip, tail, color, lw in (
            (base, bstart, C_B, 1.9),
            (ct, base, C_C, 1.9),
            (ct, bstart, C_NET, 1.5),
        ):
            ax.annotate(
                "",
                tip,
                tail,
                arrowprops=dict(
                    arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0,
                    alpha=alpha,
                ),
                annotation_clip=False,
            )
        ax.plot(*base, "o", ms=2.5, color=C_NET, alpha=alpha, zorder=5)
        # angle readout on the verdict bins
        if sig >= VERDICT_MIN:
            ax.annotate(
                f"{math.degrees(theta):.0f}°",
                (x + 0.10, base[1] + sign * 0.13),
                fontsize=7,
                color="0.45",
                ha="left",
            )
    # sigma label on the central axis
    ax.annotate(
        f"{sig:g}" if sig != 1.0 else "1",
        (x, -0.06),
        ha="center",
        va="center",
        fontsize=7.5,
        color="0.35",
        rotation=45,
    )
    # per-bin pull-back arrow across the axis
    ax.annotate(
        "",
        (x + 0.52 * SLOT, -0.16),
        (x + 0.52 * SLOT, 0.16),
        arrowprops=dict(arrowstyle="->", color="0.75", lw=1.0),
        annotation_clip=False,
    ) if i < n - 1 else None

# central axis line
ax.axhline(0.0, color="0.85", lw=0.8, zorder=0)

x_right = (n - 1) * SLOT
mean_r = sum(
    completeness(*r_geom[i]) for i, s in enumerate(centers) if s >= VERDICT_MIN
) / sum(1 for s in centers if s >= VERDICT_MIN)
mean_g = sum(
    completeness(*g_geom[i]) for i, s in enumerate(centers) if s >= VERDICT_MIN
) / sum(1 for s in centers if s >= VERDICT_MIN)

ax.annotate(
    "r-level (x-space residual legs, 19.2)\n"
    rf"$\theta \approx$ 99–119°, $|B|/|C|$ 0.37–0.63 — completeness {mean_r:.0%}",
    (-1.1, BASE_GAP + 1.28),
    fontsize=10.5,
    color="0.25",
    va="bottom",
)
ax.annotate(
    "g-level (parameter-space gradient legs, E14)\n"
    rf"$\theta \approx$ 147–163°, $|B|/|C| \to 1$ — completeness {mean_g:.0%}",
    (-1.1, -BASE_GAP - 1.32),
    fontsize=10.5,
    color="0.25",
    va="top",
)
ax.annotate(
    r"$J^{T}$ pull-back"
    "\n"
    r"$\bar g = \mathbb{E}\,[\,J^{T} r\,]$",
    (x_right + 1.25, 0.0),
    fontsize=11.5,
    color="0.3",
    ha="left",
    va="center",
)
# one-line color key between the header and the first triangles
for label, color, kx in (
    (r"$B^{\perp}$ (data)", C_B, -1.1),
    (r"$C^{\perp}$ (graph)", C_C, 1.3),
    (r"$B{+}C$ (resultant)", C_NET, 3.8),
):
    ax.annotate(label, (kx, BASE_GAP + 1.06), fontsize=9.5, color=color)
ax.annotate(
    r"$\sigma$",
    (x_right + 0.9, -0.06),
    fontsize=12,
    color="0.35",
    va="center",
)

ax.set_xlim(-1.3, x_right + 3.6)
ax.set_ylim(-BASE_GAP - 1.85, BASE_GAP + 1.85)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title(
    "$J^{T}$-born (E19 L1): the SAME two interventions, before vs after pull-back — "
    "route $1024\\to768$, reenc/area, shape-normalized $|C^{\\perp}|=1$\n"
    "(angles+ratios only; units incomparable across rows; low-$\\sigma$ bins faded, "
    "g-level $\\sigma{=}1$ drawn clamped)",
    fontsize=11.5,
    pad=14,
)
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print(f"wrote {OUT}; completeness r={mean_r:.3f} g={mean_g:.3f}")
