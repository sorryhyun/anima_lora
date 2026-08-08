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


def main() -> None:
    meta = cond_meta()
    DIGEST["conditions_by_id"] = {r["cond"]: r["sigma"] for r in DIGEST["conditions"]}
    axis_field_fig(meta)
    comb_fig(meta)


if __name__ == "__main__":
    main()
