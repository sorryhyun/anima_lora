#!/usr/bin/env python
"""sigma_lowres E24 — axis-field figures (illustration only; verdicts live
in e24_axis.json).

Reads ONLY the committed digest — no store access, no refit. Two outputs:

e24_axis_field.png
    (a, b) every gated verdict condition's B / C axis projected into the
    top-2 eigenplane of its debiased Gram (the plane carries ~92 % of the
    Gram mass); arrow length = in-plane share of that unit axis (the
    out-of-plane remainder is annotated per arrow, not drawn — the
    fig_bc_plane honesty convention). Color = sigma, solid = 768,
    dashed = 896, square tip = e194 store.
    (c) descriptive co-rotation read: rotation of B-hat (per route) and of
    the native anchor g-hat away from their sigma = 0.7 direction, in
    degrees, from the committed pairwise cosines. No frame-relative claim
    is made (E24 Results wording) — the panel shows the two curves only.

e24_bc_comb_rot.png
    The E19 bc_comb redrawn with the MEASURED between-bin orientation:
    each bin's pair keeps its exact within-bin geometry (|B|, |C| =
    sqrt(2S), sqrt(2F) in units of ||g||; mutual angle arccos rho) but is
    now tilted to its B-axis angle in the shared top-2 plane, instead of
    the original "C always vertical" per-bin convention. Route 768;
    sigma <= 0.7 from e193, sigma = 0.8333 from e194 (the two stores agree
    to cos 0.999 at the shared 0.7 bin). C's side of B is a fixed drawing
    convention (annotated); the in-plane share per bin is printed under
    each pair.

e24_bc_comb_3d.png
    The comb lifted into 3D, in the E19 bc_comb visual style (axis-off
    diagram, teal sigma axis, direct colored labels, dashed envelope,
    scale bar): pairs planted on the sigma axis, and the E24 shared top-2
    eigenplane mapped rigidly onto the cross-plane (y, z) at each sigma.
    Shape and scale are NOT separated: legs, mutual angle, and the B->C
    chord all share one ||g|| scale, so the chord's length IS the
    realized gap |B + C| = sqrt(2R) — no separate gap axis. The frame is
    rotated so C-hat(sigma = 0.7) draws vertical (that bin reproduces the
    E19 comb exactly); every other pair twists about the sigma axis by
    its measured theta_B. Same bins, same measured theta_B / mutual
    angle / leg lengths as the 2D rot version.

e24_bc_comb_fan.png
    bc_fan: every pair planted at ONE shared origin, and sigma enters
    purely as rotation — each pair drawn at its measured theta_B in the
    shared top-2 eigenplane, true ||g|| scale (light rings as the ruler).
    B (thick, color = sigma) arrives at the origin, C (thin) leaves it,
    gray chord = realized gap. This is the raw measured configuration in
    the shared plane; the only conventions left are the plane projection
    (in-plane share in parens) and C's side of B.

e24_bc_comb_sphere.png
    bc_sphere: the fan made 3D, protractor-style. One shared origin; each
    pair rigidly rotated in its own plane so its B+C chord runs along +z
    (drawing convention, replacing E19's "C vertical"), and the pair's
    vertical half-plane placed at azimuth = sigma mapped onto 0..180 deg
    (a sigma COORDINATE — low sigma on the left, high on the right; the
    measured theta_B is annotated per bin, not drawn as azimuth).
    Within-bin geometry exact (legs, mutual angle, chord = gap), one true
    ||g|| scale — the faint unit half-shell is the ruler.

Usage::

    uv run python project/sigma_lowres/paper_bench/experiments/e24/e24_axis_fig.py
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

C_B, C_C, C_NET = "#1f77b4", "#d62728", "0.15"
REF = "e193/768/s0.7"  # +x anchor of the eigenplane drawing frame


def plane_coords(leg: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Top-2 eigenplane coordinates for every gated verdict condition of a
    leg. Returns (condition ids, coords (n,2), in-plane share (n,))."""
    g = DIGEST["gram"][leg]
    ids = g["conditions"]
    M = np.array(g["matrix"])
    ev, V = np.linalg.eigh(M)
    ev = np.clip(ev, 0.0, None)
    top = np.argsort(ev)[::-1][:2]
    coords = V[:, top] * np.sqrt(ev[top])  # row i = condition i in-plane
    share = (coords**2).sum(axis=1)  # diagonal of M is 1 by construction
    # drawing frame: REF on +x, low-sigma side rotated to +y
    if REF in ids:
        x, y = coords[ids.index(REF)]
        phi = math.atan2(y, x)
        R = np.array(
            [[math.cos(-phi), -math.sin(-phi)], [math.sin(-phi), math.cos(-phi)]]
        )
        coords = coords @ R.T
        lo = min(range(len(ids)), key=lambda i: DIGEST["conditions_by_id"][ids[i]])
        if coords[lo, 1] < 0:
            coords[:, 1] *= -1
    return ids, coords, share


def cond_meta() -> dict[str, dict]:
    return {r["cond"]: r for r in DIGEST["conditions"]}


def _style(cid: str) -> dict:
    store, route, _ = cid.split("/")
    return {
        "ls": "-" if route == "768" else "--",
        "marker": "s" if store == "e194" else "o",
    }


def axis_field_fig(meta: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), layout="constrained")
    sigmas = sorted({m["sigma"] for m in meta.values() if m["verdict_eligible"]})
    cmap = plt.get_cmap("viridis")
    cnorm = matplotlib.colors.Normalize(vmin=min(sigmas), vmax=max(sigmas))

    for ax, leg in zip(axes[:2], ("B", "C")):
        ids, coords, share = plane_coords(leg)
        g = DIGEST["gram"][leg]
        cluster = [  # sigma=0.7 arrows coincide — one group note, no labels
            (cid, sh) for cid, sh in zip(ids, share) if cid.endswith("/s0.7")
        ]
        for cid, (x, y), sh in zip(ids, coords, share):
            st = _style(cid)
            col = cmap(cnorm(meta[cid]["sigma"]))
            ax.plot([0, x], [0, y], st["ls"], color=col, lw=2)
            ax.plot([x], [y], st["marker"], color=col, ms=6)
            if not cid.endswith("/s0.7"):
                ax.annotate(
                    f"{cid.split('/', 1)[1]}  ({sh:.2f})",
                    (x, y),
                    fontsize=6.5,
                    ha="left",
                    va="bottom",
                    xytext=(3, 2),
                    textcoords="offset points",
                )
        lo, hi = min(s for _, s in cluster), max(s for _, s in cluster)
        ax.annotate(
            f"σ=0.7 ×{len(cluster)}\n(768/896 × e193/e194,\nshares {lo:.2f}–{hi:.2f})",
            (1.0, 0.02),
            fontsize=6.5,
            ha="left",
            va="top",
            xytext=(4, -4),
            textcoords="offset points",
        )
        ax.set_aspect("equal")
        ax.axhline(0, color="0.85", lw=0.6, zorder=0)
        ax.axvline(0, color="0.85", lw=0.6, zorder=0)
        ax.set_title(
            f"{leg}-axis field — top-2 eigenplane, "
            f"plane share {(g['eigvals'][0] + g['eigvals'][1]) / g['n']:.2f}\n"
            "(per-arrow in-plane share in parens)",
            fontsize=10,
        )
        ax.set_xlim(-0.15, 1.3)
        ax.set_ylim(-0.42, 1.0)
    fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=cnorm, cmap=cmap),
        ax=axes[:2],
        shrink=0.75,
        label="sigma",
    )

    # (c) co-rotation, descriptive: angle away from the sigma=0.7 direction
    ax = axes[2]
    fam = DIGEST["families"]["B"]["across_sigma"]["pairs"]

    def ang_to_07(cid_prefix: str, sigma: float) -> float | None:
        for p in fam:
            pair = {p["x"], p["y"]}
            if f"{cid_prefix}/s{sigma:g}" in pair and any(
                q.endswith("/s0.7") and q.startswith(cid_prefix) for q in pair
            ):
                return math.degrees(math.acos(min(abs(p["cos"]), 1.0)))
        return None

    for prefix, ls in (("e193/768", "-"), ("e193/896", "--")):
        xs = [s for s in (0.3, 0.4333, 0.5667, 0.7)]
        ys = [ang_to_07(prefix, s) if s != 0.7 else 0.0 for s in xs]
        ax.plot(
            xs, ys, ls, marker="o", color=C_B, label=f"B-hat {prefix.split('/')[1]}"
        )
    grows = {
        ("%s|%s" % (r["x"], r["y"])): r["cos"] for r in DIGEST["context"]["ghat_cos"]
    }

    def gang(sig: float) -> float:
        for k, v in grows.items():
            pair = set(k.split("|"))
            if {"e193/s%g" % sig, "e193/s0.7"} == pair:
                return math.degrees(math.acos(min(abs(v), 1.0)))
        return 0.0

    xs = [0.3, 0.4333, 0.5667, 0.7]
    ax.plot(
        xs,
        [gang(s) if s != 0.7 else 0.0 for s in xs],
        ":",
        marker="^",
        color="0.3",
        label="g-hat (anchor)",
    )
    ax.set_xlabel("sigma")
    ax.set_ylabel("rotation away from sigma=0.7 direction (deg)")
    ax.set_title("co-rotation (descriptive) — e193, from committed cosines")
    ax.legend(fontsize=8)
    fig.suptitle(
        "E24 cancellation-axis field — one axis per sigma across routes/stores, "
        "rotating smoothly with sigma (verdict STRUCTURED)"
    )
    fig.savefig(HERE / "e24_axis_field.png", dpi=150, bbox_inches="tight")
    print("[fig] e24_axis_field.png")


def comb_fig(meta: dict[str, dict]) -> None:
    """bc_comb with measured between-bin orientation (route 768)."""
    ids, coords, share = plane_coords("B")
    bins = [
        ("e193/768/s0.3", 0.3),
        ("e193/768/s0.4333", 0.4333),
        ("e193/768/s0.5667", 0.5667),
        ("e193/768/s0.7", 0.7),
        ("e194/768/s0.8333", 0.8333),
    ]
    K = 0.16  # ||g|| units -> sigma-axis units
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for k, (cid, sig) in enumerate(bins):
        m = meta[cid]
        i = ids.index(cid)
        thB = math.atan2(coords[i, 1], coords[i, 0])
        lB, lC = math.sqrt(2 * max(m["S"], 0)), math.sqrt(2 * max(m["F"], 0))
        mut = math.acos(max(-1.0, min(1.0, m["rho"])))
        bx, by = sig, 0.0
        Bv = np.array([math.cos(thB), math.sin(thB)]) * lB * K
        Cv = np.array([math.cos(thB + mut), math.sin(thB + mut)]) * lC * K
        # tip-to-tail (original comb semantics): B arrives at the base,
        # C leaves the base, resultant B+C from B's start to C's tip
        ax.annotate(
            "",
            xy=(bx, by),
            xytext=(bx - Bv[0], by - Bv[1]),
            arrowprops=dict(arrowstyle="->", color=C_B, lw=2),
        )
        ax.annotate(
            "",
            xy=(bx + Cv[0], by + Cv[1]),
            xytext=(bx, by),
            arrowprops=dict(arrowstyle="->", color=C_C, lw=2),
        )
        ax.plot(
            [bx - Bv[0], bx + Cv[0]],
            [by - Bv[1], by + Cv[1]],
            color=C_NET,
            lw=1.6,
        )
        ax.plot([bx], [by], "k.", ms=3)
        ax.annotate(
            f"σ={sig:g}\nθ_B={math.degrees(thB):+.0f}°  ({share[i]:.2f})",
            (bx, -0.36 if k % 2 == 0 else -0.46),
            fontsize=7,
            ha="center",
        )
    ax.axhline(0, color="0.9", lw=0.8, zorder=0)
    ax.set_aspect("equal")
    ax.set_xlim(0.02, 1.0)
    ax.set_ylim(-0.5, 0.42)
    ax.set_yticks([])
    ax.set_xlabel("sigma (true positions; route 1024→768; σ≤0.7 e193, σ=0.8333 e194)")
    ax.set_title(
        "bc_comb, between-bin orientation now MEASURED (E24 top-2 plane; "
        "in-plane share in parens)\nwithin-bin lengths/angle exact as before, one "
        "true scale (σ=0.3's legs really are ~4× larger); C's side of B remains a "
        "drawing convention",
        fontsize=10,
    )
    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D([], [], color=C_B, lw=2, label="B⊥ (data leg, arrives at base)"),
            Line2D([], [], color=C_C, lw=2, label="C⊥ (graph leg, leaves base)"),
            Line2D([], [], color=C_NET, lw=1.6, label="resultant B+C"),
        ],
        fontsize=8,
        loc="upper right",
    )
    fig.savefig(HERE / "e24_bc_comb_rot.png", dpi=150, bbox_inches="tight")
    print("[fig] e24_bc_comb_rot.png")


COMB_BINS = [
    ("e193/768/s0.3", 0.3),
    ("e193/768/s0.4333", 0.4333),
    ("e193/768/s0.5667", 0.5667),
    ("e193/768/s0.7", 0.7),
    ("e194/768/s0.8333", 0.8333),
]


def comb3d_fig(meta: dict[str, dict]) -> None:
    """bc_comb in 3D, E19 style: pairs planted on the sigma axis, and the
    shared top-2 eigenplane mapped rigidly onto the cross-plane (y, z) at
    each sigma — one ||g|| scale for everything, so the B->C chord's
    length IS the realized gap (no separate gap axis). The frame is
    rotated so C-hat(sigma = 0.7) draws vertical (that bin reproduces the
    E19 comb exactly); every other pair twists about the sigma axis by
    its measured theta_B."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    C_AX = "#20c9ab"
    ids, coords, share = plane_coords("B")
    fig = plt.figure(figsize=(11.5, 6.2))
    ax = fig.add_subplot(projection="3d")
    ax.set_axis_off()

    def arrow(p0, p1, color, lw=2.0):
        d = np.asarray(p1) - np.asarray(p0)
        ln = float(np.linalg.norm(d))
        ax.quiver(
            *p0,
            *d,
            color=color,
            lw=lw,
            arrow_length_ratio=min(0.5, max(0.10, 0.09 / max(ln, 1e-9))),
        )

    # teal sigma axis (true sigma positions)
    arrow((0.2, 0, 0), (0.95, 0, 0), C_AX, lw=2.4)
    ax.text(0.965, 0, 0, "σ", color=C_AX, fontsize=13, ha="left", va="center")

    # rigid rotation of the shared eigenplane: C-hat(sigma=0.7) -> vertical
    phi0 = math.acos(max(-1.0, min(1.0, meta["e193/768/s0.7"]["rho"])))

    def yz(phi: float, ln: float) -> np.ndarray:
        return np.array([ln * math.sin(phi - phi0), ln * math.cos(phi - phi0)])

    ctips, ends = [], {}
    for k, (cid, sig) in enumerate(COMB_BINS):
        m = meta[cid]
        i = ids.index(cid)
        thB = math.atan2(coords[i, 1], coords[i, 0])
        lB, lC = math.sqrt(2 * max(m["S"], 0)), math.sqrt(2 * max(m["F"], 0))
        mut = math.acos(max(-1.0, min(1.0, m["rho"])))
        Bv, Cv = yz(thB, lB), yz(thB + mut, lC)
        base = np.array([sig, 0.0, 0.0])
        btail = np.array([sig, -Bv[0], -Bv[1]])
        ctip = np.array([sig, Cv[0], Cv[1]])
        ctips.append(ctip)
        ends[cid] = (btail, ctip)
        arrow(btail, base, C_B)
        arrow(base, ctip, C_C)
        ax.plot(*zip(btail, ctip), color=C_NET, lw=1.7)
        ax.plot([sig], [0], [0], "o", ms=3, color=C_NET)
        gap = float(np.linalg.norm(ctip - btail))
        mid = (btail + ctip) / 2
        if k > 0:  # sigma=0.3's value rides its B+C direct label instead
            ax.text(
                mid[0] + 0.02,
                mid[1] - 0.22,
                mid[2] + 0.02,
                f"{gap:.2f}",
                fontsize=7,
                color="0.25",
                ha="left",
            )
        ax.text(
            sig,
            0,
            -0.14 if k % 2 == 0 else -0.30,
            f"{sig:g}\nθ_B={math.degrees(thB):+.0f}°",
            fontsize=7.5,
            color="0.35",
            ha="center",
            va="top",
        )

    # envelope through the C tips (E19 convention), now a 3D curve
    ctips = np.array(ctips)
    ax.plot(ctips[:, 0], ctips[:, 1], ctips[:, 2], ls=(0, (4, 3)), color="0.4", lw=1.4)
    ax.text(
        0.64,
        0.5,
        0.74,
        "envelope through C⊥ tips",
        fontsize=8.5,
        color="0.4",
        ha="center",
    )

    # direct labels at the tallest bin (sigma = 0.3), E19-style
    btail, ctip = ends[COMB_BINS[0][0]]
    ax.text(
        *(btail + [0.0, 0.10, -0.16]),
        "B⊥ (data)",
        color=C_B,
        fontsize=10,
        ha="center",
    )
    ax.text(
        *(ctip + [0.0, 0.10, 0.10]),
        "C⊥ (graph)",
        color=C_C,
        fontsize=10,
        ha="center",
    )
    gap0 = float(np.linalg.norm(ctip - btail))
    ax.text(
        *((btail + ctip) / 2 + [0.0, 0.52, -0.02]),
        f"B+C = {gap0:.2f}\n(realized gap)",
        color=C_NET,
        fontsize=10,
        ha="center",
    )

    # scale bar (one ||g|| scale for legs AND chord)
    ax.plot([0.93, 0.93], [0, 0], [0.40, 0.90], color="0.5", lw=2.5)
    ax.text(0.945, 0, 0.65, "0.5 ‖ḡ‖", fontsize=8.5, color="0.45", va="center")

    fig.text(
        0.5,
        0.08,
        "cross-plane (y, z) at each σ = the E24 shared top-2 eigenplane, one rigid "
        "rotation: Ĉ(σ=0.7) drawn vertical — twist about the σ axis = measured θ_B\n"
        "legs, mutual angle and chord share one ‖g‖ scale, so the B+C chord length "
        "IS the realized gap; route 1024→768; σ≤0.7 e193, σ=0.8333 e194",
        fontsize=8,
        color="0.35",
        ha="center",
    )
    ax.set_xlim(0.18, 1.0)
    ax.set_ylim(-0.4, 2.0)
    ax.set_zlim(-0.35, 1.05)
    ax.set_box_aspect((1.6, 2.4, 1.4))
    ax.view_init(elev=18, azim=-110)
    ax.set_title(
        "bc_comb in 3D — pairs planted on the σ axis, shared eigenplane as the "
        "cross-plane (one ‖g‖ scale: chord = gap)\nσ=0.7 reproduces the E19 comb; "
        "the pair twists about the σ axis by the measured θ_B",
        fontsize=10,
    )
    fig.savefig(HERE / "e24_bc_comb_3d.png", dpi=150, bbox_inches="tight")
    print("[fig] e24_bc_comb_3d.png")


def fan_fig(meta: dict[str, dict]) -> None:
    """bc_fan: all pairs planted at ONE shared origin; sigma enters purely
    as rotation (the measured theta_B in the shared top-2 eigenplane), at
    true ||g|| scale. B arrives at the origin, C leaves it, chord = the
    realized gap. This is the raw measured configuration in the shared
    plane — the only conventions left are the plane projection (in-plane
    share in parens) and C's side of B."""
    ids, coords, share = plane_coords("B")
    fig, ax = plt.subplots(figsize=(9.5, 7.6))
    cmap = plt.get_cmap("viridis")
    sigmas = [s for _, s in COMB_BINS]
    cnorm = matplotlib.colors.Normalize(vmin=min(sigmas), vmax=max(sigmas))

    # light polar rings: the ruler for the true ||g|| scale
    for r in (0.5, 1.0, 1.5, 2.0):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color="0.92", lw=0.9))
        ax.annotate(
            f"{r:g}",
            (r * math.cos(math.radians(115)), r * math.sin(math.radians(115))),
            fontsize=7,
            color="0.6",
            ha="center",
            va="center",
        )
    ax.annotate(
        "rings: ‖g‖ units",
        (2.0 * math.cos(math.radians(115)), 2.0 * math.sin(math.radians(115)) + 0.14),
        fontsize=7.5,
        color="0.6",
        ha="center",
    )
    ax.plot(0, 0, "o", ms=5, color=C_NET, zorder=6)

    for cid, sig in COMB_BINS:
        m = meta[cid]
        i = ids.index(cid)
        thB = math.atan2(coords[i, 1], coords[i, 0])
        lB, lC = math.sqrt(2 * max(m["S"], 0)), math.sqrt(2 * max(m["F"], 0))
        mut = math.acos(max(-1.0, min(1.0, m["rho"])))
        col = cmap(cnorm(sig))
        btail = -np.array([math.cos(thB), math.sin(thB)]) * lB
        ctip = np.array([math.cos(thB + mut), math.sin(thB + mut)]) * lC
        ax.annotate(
            "",
            (0, 0),
            btail,
            arrowprops=dict(arrowstyle="-|>", color=col, lw=2.4, shrinkA=0, shrinkB=0),
        )
        ax.annotate(
            "",
            ctip,
            (0, 0),
            arrowprops=dict(
                arrowstyle="-|>", color=col, lw=1.5, alpha=0.8, shrinkA=0, shrinkB=0
            ),
        )
        ax.plot(*zip(btail, ctip), color="0.55", lw=1.1, zorder=1)
        gap = float(np.linalg.norm(ctip - btail))
        # sigma label just beyond the B tail (the outer end of each pair)
        out = btail * (1 + 0.14 / max(np.linalg.norm(btail), 1e-9))
        if cid == COMB_BINS[0][0]:
            lbl = f"σ={sig:g}\nθ_B={math.degrees(thB):+.0f}°  ({share[i]:.2f})"
        else:
            lbl = f"σ={sig:g} ({math.degrees(thB):+.0f}°, {share[i]:.2f})"
        ax.annotate(lbl, out, fontsize=7.5, color=col, ha="right", va="center")
        if cid == COMB_BINS[0][0]:
            mid = (btail + ctip) / 2
            ax.annotate(
                f"|B+C|={gap:.2f}",
                mid + [0.05, -0.05],
                fontsize=7,
                color="0.35",
                ha="left",
            )
        elif cid == COMB_BINS[-1][0]:
            mid = (btail + ctip) / 2
            ax.annotate(
                f"|B+C|={gap:.2f}",
                mid + [-0.08, 0.02],
                fontsize=7,
                color="0.35",
                ha="right",
            )

    # direct role labels on the sigma=0.3 pair (the big one)
    m = meta[COMB_BINS[0][0]]
    i = ids.index(COMB_BINS[0][0])
    thB = math.atan2(coords[i, 1], coords[i, 0])
    lB, lC = math.sqrt(2 * m["S"]), math.sqrt(2 * m["F"])
    mut = math.acos(max(-1.0, min(1.0, m["rho"])))
    bmid = -np.array([math.cos(thB), math.sin(thB)]) * lB * 0.55
    cmid = np.array([math.cos(thB + mut), math.sin(thB + mut)]) * lC * 0.72
    ax.annotate(
        "B⊥ (data, arrives at origin)",
        bmid + [0.08, 0.02],
        fontsize=9,
        color="0.2",
        ha="left",
    )
    ax.annotate(
        "C⊥ (graph, leaves origin)",
        cmid + [-0.05, 0.09],
        fontsize=9,
        color="0.2",
        ha="right",
    )
    ax.annotate(
        "B+C (realized gap)",
        ((bmid / 0.55 + cmid / 0.72) / 2) + [-0.08, -0.12],
        fontsize=9,
        color="0.35",
        ha="right",
    )

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-2.35, 0.9)
    ax.set_ylim(-1.85, 1.15)
    ax.set_title(
        "bc_fan — every pair planted at one origin; σ enters as ROTATION "
        "(measured θ_B, shared top-2 plane), true ‖g‖ scale\n"
        "arrow color = σ; B (thick) arrives at the origin, C (thin) leaves it, "
        "gray chord = realized gap; C's side of B is a drawing convention",
        fontsize=10,
    )
    fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=cnorm, cmap=cmap),
        ax=ax,
        shrink=0.6,
        label="sigma",
    )
    fig.savefig(HERE / "e24_bc_comb_fan.png", dpi=150, bbox_inches="tight")
    print("[fig] e24_bc_comb_fan.png")


def sphere_fig(meta: dict[str, dict]) -> None:
    """bc_sphere: the fan made spherical. All pairs share one origin (the
    sphere center); each pair is rigidly rotated in its own plane so its
    B+C chord runs along +z (drawing convention, replacing E19's
    "C vertical"); sigma's measured rotation theta_B is carried by the
    AZIMUTH of the pair's vertical plane about the z axis. Within-bin
    geometry (leg lengths, mutual angle, chord = gap) exact, one true
    ||g|| scale — the faint unit sphere is the ruler."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    ids, coords, share = plane_coords("B")
    cmap = plt.get_cmap("viridis")
    sigmas = [s for _, s in COMB_BINS]
    cnorm = matplotlib.colors.Normalize(vmin=min(sigmas), vmax=max(sigmas))

    fig = plt.figure(figsize=(10, 6.4))
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

    # protractor mapping: sigma 0..1 -> 0..180 deg of azimuth, low sigma on
    # the viewer's LEFT, mid sigma pointing away (camera azim below). The
    # pair content lives on the NEGATIVE side of its plane axis, so the
    # plane azimuth is the content direction + 180.
    CAM_AZ = 22.5

    def sig_azim(sig: float) -> float:
        return math.radians(CAM_AZ + 90.0 - 180.0 * sig)  # plane axis

    # faint unit HALF-SHELL spanning the protractor's 180 deg: ||g|| ruler
    u = np.linspace(math.radians(CAM_AZ - 270.0), math.radians(CAM_AZ - 90.0), 13)
    v = np.linspace(0, np.pi / 2, 5)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="0.9", lw=0.3, alpha=0.45)
    eq = np.linspace(math.radians(CAM_AZ - 270.0), math.radians(CAM_AZ - 90.0), 61)
    ax.plot(np.cos(eq), np.sin(eq), 0 * eq, color="0.85", lw=0.8)
    ax.text(
        -0.42, 1.15, -0.05, "unit shell: 1 ‖g‖", fontsize=7.5, color="0.55", ha="left"
    )

    # the shared chord direction: +z through the origin (bottom center)
    ax.plot([0, 0], [0, 0], [-0.25, 1.35], color="0.6", lw=1.0, ls=(0, (4, 3)))
    ax.text(
        0, 0, -0.47, "ẑ ∥ B+C (every chord)", fontsize=8.5, color="0.4", ha="center"
    )
    ax.plot([0], [0], [0], "o", ms=5, color=C_NET, zorder=6)

    for k, (cid, sig) in enumerate(COMB_BINS):
        m = meta[cid]
        i = ids.index(cid)
        thB = math.atan2(coords[i, 1], coords[i, 0])
        lB, lC = math.sqrt(2 * max(m["S"], 0)), math.sqrt(2 * max(m["F"], 0))
        mut = math.acos(max(-1.0, min(1.0, m["rho"])))
        col = cmap(cnorm(sig))
        # pair in its local plane, then rotate in-plane so chord points up
        btail2 = -np.array([math.cos(thB), math.sin(thB)]) * lB
        ctip2 = np.array([math.cos(thB + mut), math.sin(thB + mut)]) * lC
        chord = ctip2 - btail2
        rot = math.pi / 2 - math.atan2(chord[1], chord[0])
        R = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
        btail2, ctip2 = R @ btail2, R @ ctip2
        # embed in the vertical half-plane at the protractor azimuth
        phi = sig_azim(sig)
        side = np.array([math.cos(phi), math.sin(phi), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        btail = side * btail2[0] + up * btail2[1]
        ctip = side * ctip2[0] + up * ctip2[1]
        arrow(btail, (0, 0, 0), col, lw=2.4)
        arrow((0, 0, 0), ctip, col, lw=1.5, alpha=0.85)
        ax.plot(*zip(btail, ctip), color="0.45", lw=1.2)
        gap = float(np.linalg.norm(ctip - btail))
        # radial label beyond the B tail; the left-right camera spread
        # (sigma=0.3 far left ... 0.8333 right) separates them
        label_dz = [0.55, -0.95, 0.12, 0.0, 0.0]
        label_rr = [2.2, 2.2, 2.2, 2.2, 2.2]
        out = btail * (label_rr[k] / max(np.linalg.norm(btail), 1e-9))
        out = out + np.array([0.0, 0.0, label_dz[k]])
        ax.plot(*zip(btail, out * 0.94), color=col, lw=0.6, ls=":", alpha=0.5)
        ax.text(
            *out,
            f"σ={sig:g}\nθ_B{math.degrees(thB):+.0f}°, gap {gap:.2f}",
            fontsize=7.5,
            color=col,
            ha="right" if k == 1 else "center",
        )

    ax.set_xlim(-2.4, 0.6)
    ax.set_ylim(-2.05, 2.05)
    ax.set_zlim(-0.5, 1.35)
    ax.set_box_aspect((3.0, 4.1, 1.85))
    ax.view_init(elev=14, azim=CAM_AZ)
    ax.set_title(
        "bc_sphere — one origin, every pair rotated so its B+C chord runs along ẑ; "
        "azimuth = σ (0–1 ↦ 0–180°, protractor)\narrow color = σ; B thick into the "
        "origin, C thin out; within-bin geometry exact, true ‖g‖ scale",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.06,
        "azimuth is a σ COORDINATE (protractor), not the measured rotation — "
        "θ_B is annotated per bin; chord-along-ẑ is a drawing convention "
        '(replacing E19\'s "C vertical")\n'
        "route 1024→768; σ≤0.7 e193, σ=0.8333 e194; C's side of B is a drawing "
        "convention",
        fontsize=8,
        color="0.35",
        ha="center",
    )
    fig.savefig(HERE / "e24_bc_comb_sphere.png", dpi=150, bbox_inches="tight")
    print("[fig] e24_bc_comb_sphere.png")


def main() -> None:
    meta = cond_meta()
    DIGEST["conditions_by_id"] = {r["cond"]: r["sigma"] for r in DIGEST["conditions"]}
    axis_field_fig(meta)
    comb_fig(meta)
    comb3d_fig(meta)
    fan_fig(meta)
    sphere_fig(meta)


if __name__ == "__main__":
    main()
