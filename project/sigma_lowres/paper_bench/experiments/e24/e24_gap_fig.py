#!/usr/bin/env python
"""sigma_lowres E24 — gap-radial sphere (illustration only; verdicts live
in e24_axis.json).

Reads ONLY the committed digest — no store access, no new quantities.

e24_bc_comb_sphere_gap.png
    bc_sphere with the GAP as the radial object: same protractor layout as
    e24_bc_comb_sphere.png (one shared origin, each pair's vertical
    half-plane at azimuth = sigma mapped onto 0..180 deg), but each pair is
    now rigidly rotated in its own plane so its resultant B+C RADIATES
    FROM THE ORIGIN (chord-along-radial, replacing chord-along-z). The
    legs are drawn tip-to-tail around it (origin -> B -> C -> gap tip), so
    R = B + C is literally the arrow leaving the origin. Emphasis is
    inverted relative to the old sphere: the gap is the thick colored
    arrow (it is the object the E25a lookup would consult and the field
    whose rotation E24/E27 measured), the legs are thin. Within-bin
    geometry exact (leg lengths sqrt(2S), sqrt(2F) in units of ||g||,
    mutual angle arccos rho, |gap| = sqrt(2R)), one true ||g|| scale —
    the faint half-shell (radius 0.5 ||g|| — a unit shell would defeat
    the zoom) is the ruler. The legs' side of the gap is a drawing
    convention and ALTERNATES per bin (even bins up, odd bins down) so
    the small triangles stay legible at true scale. sigma = 0.3 is the
    scale OUTLIER of the whole pair (gap 0.83, legs 1.5-1.9 vs a
    0.12-0.23 field): its gap AND legs are drawn faint and clipped at
    radius R_CLIP, true values in its label, so the frame can zoom to
    the sigma >= 0.4333 field. Each bin's values (theta_B, |B+C|) sit
    at the END of its dotted direction ray. Azimuth stays a sigma
    COORDINATE (the measured theta_B is annotated, not drawn as
    azimuth).

Usage::

    uv run python project/sigma_lowres/paper_bench/experiments/e24/e24_gap_fig.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
DIGEST = json.loads((HERE / "e24_axis.json").read_text())

C_NET = "0.15"
REF = "e193/768/s0.7"

COMB_BINS = [
    ("e193/768/s0.3", 0.3),
    ("e193/768/s0.4333", 0.4333),
    ("e193/768/s0.5667", 0.5667),
    ("e193/768/s0.7", 0.7),
    ("e194/768/s0.8333", 0.8333),
]


def plane_coords(leg: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Top-2 eigenplane coordinates (e24_axis_fig.py convention) — used
    only to annotate the measured theta_B per bin."""
    g = DIGEST["gram"][leg]
    ids = g["conditions"]
    M = np.array(g["matrix"])
    ev, V = np.linalg.eigh(M)
    ev = np.clip(ev, 0.0, None)
    top = np.argsort(ev)[::-1][:2]
    coords = V[:, top] * np.sqrt(ev[top])
    share = (coords**2).sum(axis=1)
    sig_by_id = {r["cond"]: r["sigma"] for r in DIGEST["conditions"]}
    if REF in ids:
        x, y = coords[ids.index(REF)]
        phi = math.atan2(y, x)
        R = np.array(
            [[math.cos(-phi), -math.sin(-phi)], [math.sin(-phi), math.cos(-phi)]]
        )
        coords = coords @ R.T
        lo = min(range(len(ids)), key=lambda i: sig_by_id[ids[i]])
        if coords[lo, 1] < 0:
            coords[:, 1] *= -1
    return ids, coords, share


def sphere_gap_fig() -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    meta = {r["cond"]: r for r in DIGEST["conditions"]}
    ids, coords, share = plane_coords("B")
    cmap = plt.get_cmap("viridis")
    sigmas = [s for _, s in COMB_BINS]
    cnorm = matplotlib.colors.Normalize(vmin=min(sigmas), vmax=max(sigmas))

    fig = plt.figure(figsize=(9.6, 7.0))
    ax = fig.add_subplot(projection="3d")
    ax.set_axis_off()

    def arrow(p0, p1, color, lw=2.0, alpha=1.0):
        d = np.asarray(p1) - np.asarray(p0)
        ln = float(np.linalg.norm(d))
        ax.quiver(
            *p0,
            *d,
            color=color,
            lw=lw,
            alpha=alpha,
            arrow_length_ratio=min(0.5, max(0.10, 0.10 / max(ln, 1e-9))),
        )

    CAM_AZ = 22.5

    def gap_azim(sig: float) -> float:
        # content azimuth: same half of the shell as the original sphere
        # (low sigma on the viewer's left, protractor 0..180)
        return math.radians(CAM_AZ + 90.0 - 180.0 * sig) + math.pi

    # faint HALF-SHELL over the protractor's 180 deg: the ||g|| ruler.
    # Radius 0.5 (not 1): the field's gaps are 0.12-0.23, and a unit
    # shell would dominate the canvas and defeat the zoom.
    SHELL = 0.5
    u = np.linspace(math.radians(CAM_AZ - 270.0), math.radians(CAM_AZ - 90.0), 13)
    v = np.linspace(0, np.pi / 2, 5)
    xs = SHELL * np.outer(np.cos(u), np.sin(v))
    ys = SHELL * np.outer(np.sin(u), np.sin(v))
    zs = SHELL * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="0.9", lw=0.3, alpha=0.45)
    eq = np.linspace(math.radians(CAM_AZ - 270.0), math.radians(CAM_AZ - 90.0), 61)
    ax.plot(SHELL * np.cos(eq), SHELL * np.sin(eq), 0 * eq, color="0.85", lw=0.8)
    ax.text(-0.18, 0.52, -0.10, "shell: 0.5 ‖g‖", fontsize=7.5, color="0.55", ha="left")
    ax.plot([0], [0], [0], "o", ms=5, color=C_NET, zorder=6)

    R_CLIP = 0.42  # sigma=0.3 is the OUTLIER bin: gap AND legs clip here
    # one dotted DIRECTION ray per bin (origin -> shell along the gap
    # direction) with the values at ITS END — no separate leaders;
    # small z-stagger keeps adjacent labels apart
    lab_dz = [0.28, -0.34, 0.20, -0.32, 0.16]
    lab_r = [0.62, 0.90, 0.60, 0.80, 0.60]
    for k, (cid, sig) in enumerate(COMB_BINS):
        m = meta[cid]
        i = ids.index(cid)
        thB = math.atan2(coords[i, 1], coords[i, 0])
        lB, lC = math.sqrt(2 * max(m["S"], 0)), math.sqrt(2 * max(m["F"], 0))
        mut = math.acos(max(-1.0, min(1.0, m["rho"])))
        col = cmap(cnorm(sig))
        # tip-to-tail triangle in the pair's own 2D plane, then rotate
        # in-plane so the resultant points along +x (the radial direction)
        B2 = np.array([lB, 0.0])
        C2 = np.array([math.cos(mut), math.sin(mut)]) * lC
        R2 = B2 + C2
        rot = -math.atan2(R2[1], R2[0])
        Rm = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
        B2, C2, R2 = Rm @ B2, Rm @ C2, Rm @ R2
        # legs' side of the gap is a drawing convention — ALTERNATED per
        # bin (even up, odd down) so the small triangles do not overlap
        want_up = k % 2 == 0
        if (B2[1] < 0) == want_up:
            B2, C2, R2 = B2 * [1, -1], C2 * [1, -1], R2 * [1, -1]
        # embed in the vertical half-plane at the protractor azimuth
        phi = gap_azim(sig)
        side = np.array([math.cos(phi), math.sin(phi), 0.0])
        up = np.array([0.0, 0.0, 1.0])

        def emb(p2):
            return side * p2[0] + up * p2[1]

        btip, rtip = emb(B2), emb(R2)
        gap = float(np.linalg.norm(rtip))
        # dotted direction ray: origin -> shell along the gap
        ax.plot(*zip((0, 0, 0), side * SHELL), color=col, lw=0.9, ls=":", alpha=0.5)
        if k == 0:
            # sigma=0.3 is the scale OUTLIER (gap 0.83, legs ~1.5-1.9 vs
            # a 0.12-0.23 field): gap AND legs are drawn faint and
            # CLIPPED at radius R_CLIP so the frame can zoom to the
            # field (true values in the label / digest / comb figs).
            tR = R_CLIP / max(gap, 1e-9)
            ax.plot(*zip((0, 0, 0), rtip * tR), color=col, lw=3.0, alpha=0.35)
            tB = R_CLIP / max(np.linalg.norm(btip), 1e-9)
            ax.plot(*zip((0, 0, 0), btip * tB), color=col, lw=1.0, alpha=0.3)
            seg = btip - rtip
            tC = R_CLIP / max(np.linalg.norm(seg), 1e-9)
            ax.plot(*zip(rtip, rtip + seg * tC), color=col, lw=1.0, alpha=0.3)
        else:
            arrow((0, 0, 0), rtip, col, lw=3.0)  # the gap: thick, from origin
            arrow((0, 0, 0), btip, col, lw=1.2, alpha=0.7)  # B leg
            arrow(btip, rtip, col, lw=1.2, alpha=0.7)  # C leg closes to gap tip
        # values at the END of the direction ray (no separate leader)
        out = side * lab_r[k] + np.array([0.0, 0.0, lab_dz[k]])
        ax.text(
            *out,
            f"σ={sig:g}\nθ_B{math.degrees(thB):+.0f}°, |B+C|={gap:.2f}",
            fontsize=8,
            color=col,
            ha="center",
        )

    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color="0.2",
                lw=2.6,
                label="B+C (gap) — radiates from the origin",
            ),
            Line2D(
                [],
                [],
                color="0.2",
                lw=1.2,
                alpha=0.7,
                label="B⊥ then C⊥ (legs, tip-to-tail)",
            ),
            Line2D(
                [],
                [],
                color="0.2",
                lw=1.0,
                alpha=0.3,
                label="σ=0.3 gap+legs (outlier, clipped)",
            ),
            Line2D(
                [],
                [],
                color="0.5",
                lw=0.9,
                ls=":",
                label="gap direction ray (values at its end)",
            ),
        ],
        fontsize=8,
        loc="lower left",
        framealpha=0.9,
    )
    ax.set_xlim(-0.84, 0.30)
    ax.set_ylim(-0.74, 0.74)
    ax.set_zlim(-0.85, 0.58)
    ax.set_box_aspect((1.14, 1.48, 1.43))
    ax.view_init(elev=34, azim=CAM_AZ)
    ax.set_title(
        "bc_sphere_gap — B+C RADIATES from the origin; azimuth = σ "
        "(protractor);\ngap thick (the E25a lookup object), legs thin "
        "tip-to-tail; true ‖g‖ scale",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.06,
        "azimuth is a σ COORDINATE, not the measured rotation (θ_B annotated); "
        "chord-along-radial + the legs' side of the gap\nare drawing "
        "conventions (side ALTERNATES per bin to separate the triangles); "
        "route 1024→768; σ≤0.7 e193, σ=0.8333 e194",
        fontsize=8,
        color="0.35",
        ha="center",
    )
    fig.savefig(HERE / "e24_bc_comb_sphere_gap.png", dpi=185, bbox_inches="tight")
    print("[fig] e24_bc_comb_sphere_gap.png")


if __name__ == "__main__":
    sphere_gap_fig()
