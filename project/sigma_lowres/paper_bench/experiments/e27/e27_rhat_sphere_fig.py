#!/usr/bin/env python
"""sigma_lowres E27 — R-hat sphere (illustration only; verdicts live in
e27_rotlaw.json / e250_read.json).

Reads ONLY committed digests (e25/e250_read.json pairwise R-hat cosines,
e27_rotlaw.json LOO rows) — no store access, no new quantities beyond
arithmetic on committed numbers.

e27_rhat_sphere.png
    The pooled residual direction R-hat = (B+C)-hat per sigma bin drawn as
    a point on the unit sphere, one panel per route, with NEIGHBOR SLERP
    great-circle arcs between adjacent bins — i.e. the figure draws
    exactly the settled model (27.3: measure-every-bin + neighbor
    interpolation; parametric theta(sigma) laws are LAW-WORSE) and nothing
    more. No global plane is drawn anywhere: consecutive arcs visibly
    bend (the bend angle at each interior vertex is computed from the
    committed cosine triple via the spherical law of cosines and
    annotated) — the "rolling plane" of 27.1 made visible; 0 deg would be
    one great circle.

    Embedding honesty: the 896 chain {0.3, 0.4333, 0.5667, 0.7} has ALL
    six pairwise committed cosines (e250 r_descriptive) — its Gram is
    embedded exactly (top-3; captured share per point annotated when it
    rounds below 1). The 768 chain {0.3, 0.5667, 0.7} is an exact
    triangle. sigma = 0.8333 (e194) has ONE committed cosine (to 0.7):
    it is attached at the correct angle in the continuation plane of the
    last exact arc — a drawing convention, drawn dashed with a hollow
    marker. The sigma = 0.7 cross-store duplicate is bridged
    (e193 == e194, committed cos ~ 1.05 ~ 1, clamped). 768/sigma=0.4333
    is the 25.0-1 reliability hole (rel_cos_R 0.37 < 0.5): no point is
    invented for it; the annotation marks the unread bin under its
    committed flanking arc.

    Per-point annotations: |R| = sqrt(2*R_quad) in ||g|| units (committed
    read1 rows) and, at interior verdict bins, the held-out LOO SLERP
    |cos| from the e27 fold rows — the certified interpolation quality at
    that bin. Debiased cosines are clamped to [-1, 1] for drawing only.

Usage::

    uv run python project/sigma_lowres/paper_bench/experiments/e27/e27_rhat_sphere_fig.py
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
E250 = json.loads((HERE.parent / "e25" / "e250_read.json").read_text())
E27 = json.loads((HERE / "e27_rotlaw.json").read_text())

C_ARC = "0.25"


def clamp(c: float) -> float:
    return max(-1.0, min(1.0, c))


# ---- committed pairwise R-hat cosines --------------------------------------
PAIR_COS: dict[frozenset, float] = {}
for fam in ("across_sigma", "across_route", "across_store"):
    for p in E250["r_descriptive"][fam]["pairs"]:
        PAIR_COS[frozenset((p["x"], p["y"]))] = p["cos_raw"]


def pcos(a: str, b: str) -> float:
    return PAIR_COS[frozenset((a, b))]


R_QUAD = {r["cond"]: r["R_quad"] for r in E250["read1"]["conditions"]}
LOO_SLERP = {
    t["cond"]: t["cos_slerp"] for f in E27["loo"]["R"]["folds"] for t in f["targets"]
}


def embed_chain(conds: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-3 embedding of a fully-committed chain Gram. Returns
    (unit coords (n,3), captured share per point)."""
    n = len(conds)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = clamp(pcos(conds[i], conds[j]))
    ev, V = np.linalg.eigh(M)
    ev = np.clip(ev, 0.0, None)
    top = np.argsort(ev)[::-1][:3]
    X = V[:, top] * np.sqrt(ev[top])
    share = (X**2).sum(axis=1)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X, share


def attach_by_convention(X: np.ndarray, cos_to_last: float) -> np.ndarray:
    """Attach one more unit vector past X[-1] at angle arccos(cos) in the
    continuation plane of the last arc (drawing convention)."""
    d, c = X[-1], X[-2]
    t = c - float(np.dot(c, d)) * d  # tangent at d toward c
    t = t / max(np.linalg.norm(t), 1e-12)
    th = math.acos(clamp(cos_to_last))
    return d * math.cos(th) - t * math.sin(th)


def orient(X: np.ndarray) -> np.ndarray:
    """Rotate the chain into a shared drawing frame: chain mean toward the
    camera-ish pole, chain spread horizontal (pure rigid rotation)."""
    m = X.mean(axis=0)
    m = m / max(np.linalg.norm(m), 1e-12)
    s = X[-1] - X[0]
    s = s - float(np.dot(s, m)) * m
    s = s / max(np.linalg.norm(s), 1e-12)
    frame_from = np.stack([m, s, np.cross(m, s)])
    el, azc = math.radians(38.0), math.radians(-60.0)
    mt = np.array(
        [math.cos(el) * math.cos(azc), math.cos(el) * math.sin(azc), math.sin(el)]
    )
    st = np.array([math.sin(azc), -math.cos(azc), 0.0])  # horizontal, lo->hi sigma
    frame_to = np.stack([mt, st, np.cross(mt, st)])
    return X @ frame_from.T @ frame_to


def slerp_arc(u: np.ndarray, v: np.ndarray, n: int = 40) -> np.ndarray:
    om = math.acos(clamp(float(np.dot(u, v))))
    if om < 1e-9:
        return np.stack([u, v])
    ts = np.linspace(0, 1, n)
    return np.stack(
        [(math.sin((1 - t) * om) * u + math.sin(t * om) * v) / math.sin(om) for t in ts]
    )


def bend_deg(ca_b: float, cb_c: float, ca_c: float) -> float:
    """180 deg minus the spherical-triangle vertex angle at b — 0 means
    a, b, c share one great circle."""
    tab, tbc = math.acos(clamp(ca_b)), math.acos(clamp(cb_c))
    num = clamp(ca_c) - math.cos(tab) * math.cos(tbc)
    den = max(math.sin(tab) * math.sin(tbc), 1e-12)
    return 180.0 - math.degrees(math.acos(clamp(num / den)))


ROUTES = {
    "896": {
        "exact": [
            ("e193/896/s0.3", 0.3),
            ("e193/896/s0.4333", 0.4333),
            ("e193/896/s0.5667", 0.5667),
            ("e193/896/s0.7", 0.7),
        ],
        "attach": ("e194/896/s0.8333", 0.8333, ("e194/896/s0.7", "e194/896/s0.8333")),
        "bends": [
            ("e193/896/s0.3", "e193/896/s0.4333", "e193/896/s0.5667"),
            ("e193/896/s0.4333", "e193/896/s0.5667", "e193/896/s0.7"),
        ],
        "hole": None,
        "nudge": [0.14, -0.12, -0.28, -0.20, 0.32],
        "shift": [
            (0, 0, 0),
            (0.30, 0.17, -0.06),
            (0, 0, 0),
            (-0.30, -0.17, -0.06),
            (0, 0, 0),
        ],
    },
    "768": {
        "exact": [
            ("e193/768/s0.3", 0.3),
            ("e193/768/s0.5667", 0.5667),
            ("e193/768/s0.7", 0.7),
        ],
        "attach": ("e194/768/s0.8333", 0.8333, ("e194/768/s0.7", "e194/768/s0.8333")),
        "bends": [("e193/768/s0.3", "e193/768/s0.5667", "e193/768/s0.7")],
        "hole": (0, 1, "σ=0.4333: rel_cos_R 0.37 < 0.5\n(25.0-1 hole — bin unread)"),
        "nudge": [0.14, -0.28, -0.20, 0.32],
        "shift": [(0.22, 0.13, 0.04), (0, 0, 0), (-0.42, -0.24, -0.04), (0, 0, 0)],
    },
}


def main() -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    cmap = plt.get_cmap("viridis")
    cnorm = matplotlib.colors.Normalize(vmin=0.3, vmax=0.8333)
    fig = plt.figure(figsize=(13.6, 7.0))

    for pi, (route, spec) in enumerate(ROUTES.items()):
        ax = fig.add_subplot(1, 2, pi + 1, projection="3d")
        ax.set_axis_off()
        conds = [c for c, _ in spec["exact"]]
        sigs = [s for _, s in spec["exact"]]
        X, share = embed_chain(conds)
        att_cid, att_sig, (bridge_a, bridge_b) = spec["attach"]
        # sigma=0.7 duplicate bridged e193==e194 (committed cross-store
        # cos ~ 1.05 ~ 1): the 0.7<->0.8333 cosine is applied at the chain's
        # 0.7 point.
        Xa = attach_by_convention(X, pcos(bridge_a, bridge_b))
        pts = np.vstack([X, Xa[None]])
        pts = orient(pts)
        allc = conds + [att_cid]
        alls = sigs + [att_sig]

        # faint unit sphere
        u = np.linspace(0, 2 * np.pi, 25)
        v = np.linspace(0, np.pi, 13)
        ax.plot_wireframe(
            np.outer(np.cos(u), np.sin(v)),
            np.outer(np.sin(u), np.sin(v)),
            np.outer(np.ones_like(u), np.cos(v)),
            color="0.92",
            lw=0.3,
            alpha=0.5,
        )
        ax.plot([0], [0], [0], "o", ms=4, color="0.2")

        # arcs (neighbor SLERP = the certified interpolation)
        for i in range(len(pts) - 1):
            arc = slerp_arc(pts[i], pts[i + 1])
            dashed = i == len(pts) - 2  # the convention-attached 0.8333 arc
            ax.plot(
                *arc.T,
                color=C_ARC,
                lw=2.0,
                ls=(0, (4, 3)) if dashed else "-",
                alpha=0.85,
            )
        if spec["hole"]:
            i, j, txt = spec["hole"]
            mid = slerp_arc(pts[i], pts[j], 3)[1] * 1.32 + np.array([0.30, 0.17, -0.34])
            ax.text(*mid, txt, fontsize=7, color="#b05050", ha="center")

        # radial arrows + points + labels
        for k, (cid, sig) in enumerate(zip(allc, alls)):
            col = cmap(cnorm(sig))
            p = pts[k]
            ax.quiver(0, 0, 0, *p, color=col, lw=1.4, alpha=0.7, arrow_length_ratio=0.0)
            hollow = cid == att_cid
            ax.plot(
                [p[0]],
                [p[1]],
                [p[2]],
                "o",
                ms=7,
                mfc="white" if hollow else col,
                mec=col,
                mew=1.6,
                zorder=6,
            )
            lbl = f"σ={sig:g}"
            rq = R_QUAD.get(cid)
            if rq is not None:
                lbl += f"  |R|={math.sqrt(2 * max(rq, 0)):.2f}"
            if k < len(conds) and share[k] < 0.995:
                lbl += f"  ({share[k]:.2f})"
            if cid in LOO_SLERP:
                lbl += f"\nLOO SLERP {abs(LOO_SLERP[cid]):.2f}"
            if hollow:
                lbl += "\n(attached: 1 committed cos)"
            ax.text(
                *(p * 1.36 + np.array(spec["shift"][k]) + [0, 0, spec["nudge"][k]]),
                lbl,
                fontsize=7.5,
                color=col,
                ha="center",
            )

        # bend annotations at interior vertices (committed cosine triples)
        for a, b, c in spec["bends"]:
            bd = bend_deg(pcos(a, b), pcos(b, c), pcos(a, c))
            p = pts[conds.index(b)]
            ax.text(
                *(p * 0.56 - [0, 0, 0.05]),
                f"bend {bd:.0f}°",
                fontsize=7,
                color="0.35",
                ha="center",
                style="italic",
            )

        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_zlim(-0.9, 1.15)
        ax.set_box_aspect((1, 1, 0.98))
        ax.view_init(elev=22, azim=-60)
        ax.set_title(
            f"route 1024→{route} — R̂(σ) on the unit sphere, neighbor SLERP arcs",
            fontsize=10,
        )

    fig.suptitle(
        "R̂ = (B+C)̂ axis field — measure-every-bin + neighbor SLERP "
        "(the settled lookup shape; 27.3: parametric θ(σ) laws LAW-WORSE)\n"
        "no plane is drawn: consecutive arcs bend (annotated; 0° = one great "
        "circle) — the 27.1 rolling plane made visible",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.045,
        "committed digests only: R̂ pairwise cosines e250_read.json "
        "(896 chain: all 6 pairs exact; 768: exact triangle), LOO SLERP rows "
        "e27_rotlaw.json; |R| = √(2·R_quad) in ‖g‖ units\n"
        "σ=0.8333 attached from its single committed cosine (dashed, hollow — "
        "drawing convention); σ=0.7 duplicate bridged e193≡e194 (cos ≈ 1.05 ≈ 1); "
        "debiased cosines clamped to [−1, 1] for drawing",
        fontsize=8,
        color="0.35",
        ha="center",
    )
    fig.savefig(HERE / "e27_rhat_sphere.png", dpi=150, bbox_inches="tight")
    print("[fig] e27_rhat_sphere.png")


if __name__ == "__main__":
    main()
