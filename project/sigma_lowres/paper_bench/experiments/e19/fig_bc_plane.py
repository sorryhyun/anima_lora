"""Per-bin 2D geometry in the B/C_pi PLANE, route 1024->768, bins
sigma = 0.7 and 0.8333 (the two mid-tail 19.4 verdict bins).

Honest planar construction: B and C_pi span the drawing plane, so their
lengths and mutual angle are EXACT (the near-orthogonality read:
ang(B, C_pi) = 110/122 deg). C genuinely sticks out of this plane
(B/C/C_pi span 3D), so C is drawn as its orthogonal projection onto
span(B, C_pi) -- computed from the pi_194 Gram matrix -- carrying
92/94% of |C|; the out-of-plane share is annotated, not drawn.

The read this makes visible: in the plane, C_proj = -1.00 B + 0.08 C_pi
(sigma 0.7) -- the graph leg is "minus the data leg" plus a small slice
of the phase-aligned response, and the in-plane realized gap
B + C_proj is almost purely that small C_pi slice.

All numbers from pi_194.json (19.4 run, reenc ref); units of ||g_src||.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Arc

HERE = Path(__file__).resolve().parent  # experiments/e19
PI_LEDGER = HERE / "pi_194.json"
OUT = HERE / "bc_plane_768.png"

C_B, C_C, C_NET = "#1f77b4", "#d62728", "0.15"
C_CPI, C_DPH = "#f08c8c", "#e8931a"
BINS = (0.7, 0.8333)

pi = json.loads(PI_LEDGER.read_text())
rows = {s: r for s, r in zip(pi["sigma_centers"], pi["pi_ledger"]["768"]) if s in BINS}

fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.6))

for ax, sig in zip(axes, BINS):
    row = rows[sig]
    B = math.sqrt(2 * row["S"])
    C = math.sqrt(2 * row["F"])
    P = math.sqrt(2 * row["F_pi"])
    bc = row["rho"] * B * C
    bp = row["rho_pi"] * B * P
    cp = row["cos_C_Cpi"] * C * P

    # plane coords: B along +x, C_pi at its true angle above
    a_bp = math.acos(bp / (B * P))
    vB = (B, 0.0)
    vP = (P * math.cos(a_bp), P * math.sin(a_bp))
    # C projected onto span(B, C_pi): solve the 2x2 normal equations
    det = (B * B) * (P * P) - bp * bp
    ca = (bc * (P * P) - cp * bp) / det
    cb = (cp * (B * B) - bc * bp) / det
    vC = (ca * vB[0] + cb * vP[0], ca * vB[1] + cb * vP[1])
    Cproj = math.hypot(*vC)
    out = math.sqrt(max(0.0, C * C - Cproj * Cproj))
    vGap = (vB[0] + vC[0], vB[1] + vC[1])

    def arrow(tip, color, lw=2.2, tail=(0.0, 0.0), ls="-", alpha=1.0):
        ax.annotate(
            "",
            tip,
            tail,
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=lw,
                shrinkA=0,
                shrinkB=0,
                linestyle=ls,
                alpha=alpha,
            ),
        )

    arrow(vB, C_B)
    arrow(vP, C_CPI)
    arrow(vC, C_C)
    arrow(vC, C_DPH, lw=1.6, tail=vP, ls=(0, (5, 3)))  # Delta_phase (projected)
    arrow(vGap, C_NET, lw=1.8)

    # angle arcs at the origin
    deg_bp = math.degrees(a_bp)
    deg_bc = math.degrees(math.atan2(vC[1], vC[0]))
    ax.add_patch(
        Arc((0, 0), 0.56, 0.56, theta1=0, theta2=deg_bp, color=C_CPI, lw=1.2)
    )
    ax.add_patch(Arc((0, 0), 0.36, 0.36, theta1=0, theta2=deg_bc, color=C_C, lw=1.2))
    r = 0.34
    ax.annotate(
        f"{deg_bp:.0f}°",
        (r * math.cos(math.radians(deg_bp / 2)), r * math.sin(math.radians(deg_bp / 2)) + 0.02),
        fontsize=9,
        color=C_CPI,
        ha="center",
    )
    ax.annotate(
        f"{deg_bc:.0f}°",
        (0.26 * math.cos(math.radians(deg_bc - 12)), 0.26 * math.sin(math.radians(deg_bc - 12))),
        fontsize=9,
        color=C_C,
        ha="center",
    )

    # labels
    ax.annotate(
        rf"$B^{{\perp}}$  ({B:.2f})",
        (vB[0] + 0.03, vB[1] - 0.06),
        fontsize=10.5,
        color=C_B,
        ha="left",
    )
    ax.annotate(
        rf"$C_\pi^{{\perp}}$  ({P:.2f})",
        (vP[0] - 0.02, vP[1] + 0.04),
        fontsize=10.5,
        color=C_CPI,
        ha="center",
    )
    ax.annotate(
        rf"$C^{{\perp}}$ proj  ({Cproj:.2f} of {C:.2f})",
        (vC[0] - 0.03, vC[1] + 0.05),
        fontsize=10.5,
        color=C_C,
        ha="right",
    )
    mid = ((vP[0] + vC[0]) / 2, (vP[1] + vC[1]) / 2)
    ax.annotate(
        r"$\Delta_{\rm phase}$",
        (mid[0] - 0.05, mid[1] + 0.03),
        fontsize=10,
        color=C_DPH,
        ha="right",
    )
    ax.annotate(
        r"$B{+}C$",
        (vGap[0] + 0.02, vGap[1] + 0.04),
        fontsize=9.5,
        color=C_NET,
        ha="left",
    )

    # the in-plane story
    ax.annotate(
        rf"$C_{{\rm proj}} = {ca:+.2f}\,B {cb:+.2f}\,C_\pi$"
        + "\n"
        + rf"out-of-plane $= {out:.2f}$ ({out / C * 100:.0f}% of $|C|$, not drawn)",
        (0.03, 0.97),
        xycoords="axes fraction",
        va="top",
        fontsize=9.5,
        color="0.3",
    )

    ax.set_title(rf"$\sigma = {sig:g}$", fontsize=12)
    ax.set_xlim(-0.72, 0.62)
    ax.set_ylim(-0.18, 1.54)
    ax.set_aspect("equal")
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_color("0.8")
    ax.tick_params(colors="0.5", labelsize=8)

fig.suptitle(
    "$1024\\to768$, B/$C_\\pi$ plane (exact) with $C$ projected in — "
    "lengths in $\\|\\bar g_{\\rm src}\\|$; pi_194, reenc ref",
    fontsize=12,
)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print(f"wrote {OUT}")
