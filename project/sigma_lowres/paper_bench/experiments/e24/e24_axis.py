#!/usr/bin/env python
"""sigma_lowres E24.1 — cancellation-axis geometry (CPU).

Pre-registered in README.md (committed before this file existed). Single
pass over the three surviving arm_sums/ stores (e193 depth-ledger, e194
pi-causal, e221 per-image-ledger), asking whether the near-1D cancelled
direction per (store, route, sigma) condition is ONE shared axis across
conditions, plus the residual knob decomposition (angle vs amplitude).

Frozen conventions (no tuning on outputs):
 * legs verbatim vector_ledger.bc_ledger, data_ref=reenc: B = g_rp -
   g_reenc, C = g_dem - g_rp, each perp'd against the condition's own
   ghat. pi arms not read (G11).
 * within-condition scalars must reproduce `vector_ledger.py --data_ref
   reenc` exactly at output rounding on e221 (validation gate, --validate).
 * cross-condition cosines debiased: numerator <Bx_mean, By_mean> (draw
   noise independent across conditions), EXCEPT same-store same-bin
   cross-route B pairs which share the reenc reference -> subtract the
   shared ref-noise power (|d_perp|^2/4). Denominators are the
   within-condition cross-set debiased norms sqrt(2 G^2 S) / sqrt(2 G^2 F).
 * gate rel_cos >= 0.5 per leg; verdict sigma in-window {0.3, 0.4333,
   0.5667, 0.7, 0.8333}; endpoints detached; e221 rows descriptive only.

Outputs (this dir): e24_axis.json (the record), e24_axis_cos.png,
e24_knobs.png. Wording: pooled cancellation axis at this operating
point — never per-sample.

Usage::

    uv run python project/sigma_lowres/paper_bench/experiments/e24/e24_axis.py
    ... --validate     # synthetic + e221-ledger validation gates only
    ... --figs_only    # redraw figures from the committed digest
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from project.sigma_lowres.paper_bench.vector_ledger import (  # noqa: E402
    Sums,
    bc_ledger,
)

SL = REPO_ROOT / "project/sigma_lowres"
STORES = {
    "e193": SL / "bench/results/20260807-0745-e193-depth-ledger/arm_sums",
    "e194": SL / "bench/results/20260807-1400-e194-pi-causal/arm_sums",
    "e221": SL / "paper_bench/runs/20260808-1633-e221-per-image-ledger/arm_sums",
}
VERDICT_SIGMA = {0.3, 0.4333, 0.5667, 0.7, 0.8333}
REL_GATE = 0.5
STANDARD = ("e193", "e194")  # standard-corpus stores (verdict-eligible)


def _routes(man: dict) -> list[str]:
    return [e.strip() for e in str(man["demote_edges"]).split(",")]


class Cond:
    """One (store, route, bin) condition: fp32 perp'd leg means + scalars."""

    __slots__ = (
        "store",
        "route",
        "sigma",
        "G",
        "ghat32",
        "B32",
        "C32",
        "nB2",
        "nC2",
        "ref_noise",
        "rel_B",
        "rel_C",
        "S",
        "F",
        "I",
        "rho",
        "verdict",
        "iB",
        "iC",
        "ighat",
    )

    def __init__(self, store: str, route: str, sigma: float):
        self.store, self.route, self.sigma = store, route, sigma
        self.verdict = store in STANDARD and sigma in VERDICT_SIGMA


def build_cond(s: Sums, store: str, route: str, bi: int) -> Cond:
    """Legs mirroring bc_ledger(data_ref='reenc'), keeping the vectors."""
    c = Cond(store, route, float(s.man["sigma_centers"][bi]))
    a, b = s.mean("a", bi), s.mean("b", bi)
    g0 = 0.5 * (a + b)
    G = float(np.linalg.norm(g0))
    ghat = g0 / G
    re1, re2 = s.mean("reenc", bi), s.mean("reenc__2", bi)
    ref, ref_diff = 0.5 * (re1 + re2), re1 - re2
    rp1, rp2 = s.mean(f"{route}rp", bi), s.mean(f"{route}rp__2", bi)
    dem1, dem2 = s.mean(route, bi), s.mean(f"{route}__2", bi)

    def perp(v):
        return v - float(ghat @ v) * ghat

    B1p, B2p = perp(rp1 - ref), perp(rp2 - ref)
    C1p, C2p = perp(dem1 - rp1), perp(dem2 - rp2)
    rdp = perp(ref_diff)
    G2 = G * G
    ref_noise = float(rdp @ rdp) / 4.0
    c.G = G
    c.ref_noise = ref_noise
    c.S = (float(B1p @ B2p) - ref_noise) / (2 * G2)
    c.F = float(C1p @ C2p) / (2 * G2)
    c.I = (float(B1p @ C2p) + float(B2p @ C1p)) / (2 * G2)
    denom = 2.0 * math.sqrt(max(c.S * c.F, 0.0))
    c.rho = c.I / denom if denom > 0 else float("nan")
    c.rel_B = float((B1p @ B2p) / (np.linalg.norm(B1p) * np.linalg.norm(B2p)))
    c.rel_C = float((C1p @ C2p) / (np.linalg.norm(C1p) * np.linalg.norm(C2p)))
    # debiased squared norms of the TRUE mean-leg (per bc_ledger S/F scale)
    c.nB2 = max(c.S * 2 * G2, 0.0)
    c.nC2 = max(c.F * 2 * G2, 0.0)
    c.B32 = (0.5 * (B1p + B2p)).astype(np.float32)
    c.C32 = (0.5 * (C1p + C2p)).astype(np.float32)
    c.ghat32 = ghat.astype(np.float32)
    return c


def _dot64(a32: np.ndarray, b32: np.ndarray) -> float:
    return float(a32.astype(np.float64) @ b32.astype(np.float64))


GRAM: np.ndarray | None = None  # all-legs Gram (fp64), built once in main


def build_gram(vecs: list[np.ndarray], chunk: int = 4_000_000) -> np.ndarray:
    """Chunked fp64 Gram over fp32 vectors — one conversion pass total
    (the naive per-pair _dot64 re-converts 1.2 GB per pair)."""
    n, dim = len(vecs), vecs[0].shape[0]
    G = np.zeros((n, n))
    for lo in range(0, dim, chunk):
        X = np.stack([v[lo : lo + chunk] for v in vecs]).astype(np.float64)
        G += X @ X.T
    return G


def _leg_dot(x: Cond, y: Cond, lx: str, ly: str) -> float:
    if GRAM is not None:
        ix = x.iB if lx == "B" else x.iC
        iy = y.iB if ly == "B" else y.iC
        return float(GRAM[ix, iy])
    vx = x.B32 if lx == "B" else x.C32
    vy = y.B32 if ly == "B" else y.C32
    return _dot64(vx, vy)


def cross_cos(x: Cond, y: Cond, leg: str) -> float:
    """Debiased cross-condition cosine of a leg (see README conventions)."""
    num = _leg_dot(x, y, leg, leg)
    if leg == "B" and x.store == y.store and x.sigma == y.sigma and x.route != y.route:
        num -= x.ref_noise  # shared reenc reference noise survives the mean
    nx2, ny2 = (x.nB2, y.nB2) if leg == "B" else (x.nC2, y.nC2)
    den = math.sqrt(nx2 * ny2)
    return num / den if den > 0 else float("nan")


def pair_family(x: Cond, y: Cond) -> str | None:
    """Pre-named verdict families (standard-corpus, verdict sigma only)."""
    if not (x.verdict and y.verdict):
        return None
    if x.store == y.store and x.route == y.route and x.sigma != y.sigma:
        return "across_sigma"
    if x.store == y.store and x.sigma == y.sigma and x.route != y.route:
        return "across_route"
    if x.store != y.store and x.sigma == y.sigma == 0.7 and x.route == y.route:
        return "across_store"
    return None


def gated(c: Cond, leg: str) -> bool:
    return (c.rel_B if leg == "B" else c.rel_C) >= REL_GATE


# ---------------------------------------------------------------- verdict


def family_tables(conds: list[Cond]) -> dict:
    out: dict = {}
    for leg in ("B", "C"):
        fams: dict[str, list] = {
            "across_sigma": [],
            "across_route": [],
            "across_store": [],
        }
        for i, x in enumerate(conds):
            for y in conds[i + 1 :]:
                fam = pair_family(x, y)
                if fam is None or not (gated(x, leg) and gated(y, leg)):
                    continue
                fams[fam].append(
                    {
                        "x": cond_id(x),
                        "y": cond_id(y),
                        "cos": round(cross_cos(x, y, leg), 5),
                    }
                )
        stats = {}
        for fam, rows in fams.items():
            av = [abs(r["cos"]) for r in rows]
            stats[fam] = {
                "n_pairs": len(rows),
                "median_abs": round(float(np.median(av)), 5) if av else None,
                "min_abs": round(min(av), 5) if av else None,
                "pairs": rows,
            }
        out[leg] = stats
    return out


def verdict_from(tables: dict) -> str:
    b = tables["B"]
    meds = [
        b[f]["median_abs"] for f in ("across_sigma", "across_route", "across_store")
    ]
    mins = [b[f]["min_abs"] for f in ("across_sigma", "across_route", "across_store")]
    if any(m is None for m in meds):
        return "STRUCTURED (family empty after gates — record, no verdict weight)"
    if all(m >= 0.7 for m in meds) and all(m >= 0.5 for m in mins):
        return "SHARED-AXIS"
    if b["across_sigma"]["median_abs"] <= 0.3 or b["across_route"]["median_abs"] <= 0.3:
        return "LOCAL-AXIS"
    return "STRUCTURED"


# ------------------------------------------------------------ descriptive


def cond_id(c: Cond) -> str:
    return f"{c.store}/{c.route}/s{c.sigma:g}"


def gram_rank(conds: list[Cond], leg: str) -> dict:
    """Participation ratio + top-1 share of the debiased Gram (gated verdict
    conditions, standard corpus). Gram entries = debiased <x, y> in leg-norm
    units: diagonal = 1 by construction, off-diagonal = cross_cos."""
    sel = [c for c in conds if c.verdict and gated(c, leg)]
    if len(sel) < 2:
        return {"n": len(sel)}
    n = len(sel)
    Gm = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            Gm[i, j] = Gm[j, i] = cross_cos(sel[i], sel[j], leg)
    ev = np.linalg.eigvalsh(Gm)
    ev = np.clip(ev, 0.0, None)
    tot = float(ev.sum())
    pr = tot**2 / float((ev**2).sum()) if tot > 0 else float("nan")
    return {
        "n": n,
        "conditions": [cond_id(c) for c in sel],
        "participation_ratio": round(pr, 4),
        "top1_share": round(float(ev[-1]) / tot, 4) if tot > 0 else None,
        "eigvals": [round(float(v), 4) for v in ev[::-1]],
        "matrix": [[round(float(v), 5) for v in row] for row in Gm],
    }


def knob_rows(conds: list[Cond]) -> list[dict]:
    """24.2 residual knob decomposition from S/F/rho (gated on BOTH legs)."""
    rows = []
    for c in conds:
        if not (c.verdict and gated(c, "B") and gated(c, "C")):
            continue
        R = c.S + c.F + c.I
        r_angle = (math.sqrt(max(c.S, 0)) - math.sqrt(max(c.F, 0))) ** 2
        r_amp = 2 * c.S * (1 + c.rho)
        rows.append(
            {
                "cond": cond_id(c),
                "S": round(c.S, 5),
                "F": round(c.F, 5),
                "rho": round(c.rho, 4),
                "R": round(R, 5),
                "R_angle": round(r_angle, 5),
                "R_amp": round(r_amp, 5),
                "share_angle": round(r_angle / R, 4) if R > 0 else None,
                "share_amp": round(r_amp / R, 4) if R > 0 else None,
                "winner": "amp" if r_amp < r_angle else "angle",
            }
        )
    return rows


# ------------------------------------------------------------- validation


def validate_synthetic() -> None:
    """Synthetic mini-store: build_cond must agree with bc_ledger exactly,
    and cross_cos must recover planted cross-condition cosines."""
    rng = np.random.default_rng(0)
    dim, draws = 4096, 6

    class FakeSums:
        def __init__(self, arms: dict[str, np.ndarray]):
            self.man = {
                "sigma_centers": [0.5],
                "demote_edges": "768",
                "keys": {k: {"n_images": 1} for k in arms},
            }
            self.draws = draws
            self._arms = arms

        def mean(self, key: str, bi: int):
            return self._arms[key].astype(np.float64) / (1 * draws)

        def has(self, key: str) -> bool:
            return key in self._arms

    def mk_store(axis: np.ndarray, noise: float):
        g = rng.normal(size=dim)
        ref = g + rng.normal(size=dim) * 0.01
        Bt = 0.9 * axis + 0.1 * rng.normal(size=dim) / math.sqrt(dim) * 3
        Ct = -0.85 * Bt + 0.2 * rng.normal(size=dim) / math.sqrt(dim) * 3
        arms = {}
        for sfx in ("", "__2"):
            n = lambda: rng.normal(size=dim) * noise  # noqa: E731
            # each arm's draw noise is INDEPENDENT (as in the real probe —
            # bc_ledger's ref-noise convention assumes exactly this)
            arms[f"reenc{sfx}"] = (ref + n()) * draws
            arms[f"768rp{sfx}"] = (ref + Bt + n()) * draws
            arms[f"768{sfx}"] = (ref + Bt + Ct + n()) * draws
        arms["a"] = (g + rng.normal(size=dim) * 0.01) * draws
        arms["b"] = (g - rng.normal(size=dim) * 0.01) * draws
        return FakeSums(arms), Bt

    axis = rng.normal(size=dim)
    axis /= np.linalg.norm(axis)
    s1, bt1 = mk_store(axis, 0.02)
    s2, bt2 = mk_store(axis, 0.02)

    led = bc_ledger(s1, "768", 0, "reenc")
    c1 = build_cond(s1, "syn1", "768", 0)
    for k, v in (("S", c1.S), ("F", c1.F), ("I", c1.I), ("rho", c1.rho)):
        assert abs(round(v, 5) - led[k]) <= 1e-5, (k, v, led[k])
    assert round(c1.rel_B, 5) == led["rel_cos_B"], (c1.rel_B, led["rel_cos_B"])
    assert round(c1.rel_C, 5) == led["rel_cos_C"], (c1.rel_C, led["rel_cos_C"])

    c2 = build_cond(s2, "syn2", "768", 0)
    cb = cross_cos(c1, c2, "B")
    true_cb = float(bt1 @ bt2) / (np.linalg.norm(bt1) * np.linalg.norm(bt2))
    assert abs(cb - true_cb) < 0.05, (cb, true_cb)  # planted cosine recovered
    # orthogonal-axis control
    axis2 = rng.normal(size=dim)
    axis2 -= float(axis2 @ axis) * axis
    axis2 /= np.linalg.norm(axis2)
    s3, _ = mk_store(axis2, 0.02)
    c3 = build_cond(s3, "syn3", "768", 0)
    assert abs(cross_cos(c1, c3, "B")) < 0.2, cross_cos(c1, c3, "B")
    print("[validate] synthetic: build_cond == bc_ledger; planted axis recovered")


def validate_e221(conds: list[Cond]) -> None:
    """Within-condition scalars must reproduce e221's committed ledger.json."""
    ref = json.loads((STORES["e221"].parent / "ledger.json").read_text())
    rows = ref["bc_ledger"]["768"]  # one row per bin, run order
    centers = [round(float(s), 4) for s in ref["sigma_centers"]]
    if any(r.get("data_ref") != "reenc" for r in rows):
        raise SystemExit("[validate] e221 ledger.json rows are not reenc-ref")
    by_sigma = {}
    for c in conds:
        if c.store == "e221":
            by_sigma[round(c.sigma, 4)] = c
    n_ok = 0
    for sigma, r in zip(centers, rows):
        c = by_sigma[sigma]
        for k, v in (
            ("S", c.S),
            ("F", c.F),
            ("I", c.I),
            ("rho", c.rho),
            ("rel_cos_B", c.rel_B),
            ("rel_cos_C", c.rel_C),
        ):
            assert abs(round(v, 5) - r[k]) <= 1e-5, (sigma, k, v, r[k])
        n_ok += 1
    print(f"[validate] e221 ledger.json reproduced exactly ({n_ok} bins)")


# ------------------------------------------------------------------ main


def figures(digest: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ids = digest["gram"]["B"].get("conditions", [])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, leg in zip(axes, ("B", "C")):
        g = digest["gram"][leg]
        sel = g.get("conditions", [])
        n = g.get("n", 0)
        if n < 2:
            ax.set_axis_off()
            ax.set_title(f"{leg}: <2 gated conditions")
            continue
        M = np.array(g["matrix"])
        im = ax.imshow(np.clip(M, -1, 1), vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(n), sel, rotation=90, fontsize=7)
        ax.set_yticks(range(n), sel, fontsize=7)
        ax.set_title(
            f"{leg}-axis debiased cos (gated verdict conds)\n"
            f"top-1 share {g['top1_share']}, PR {g['participation_ratio']}"
        )
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"E24 cancellation-axis cosines — verdict {digest['verdict']}")
    fig.tight_layout()
    fig.savefig(HERE / "e24_axis_cos.png", dpi=150)
    print(f"[fig] e24_axis_cos.png ({len(ids)} B-conditions)")

    rows = digest["knobs"]
    if rows:
        fig2, ax = plt.subplots(figsize=(max(6, 0.6 * len(rows)), 4.5))
        x = np.arange(len(rows))
        ax.bar(
            x - 0.2, [r["share_angle"] for r in rows], 0.4, label="R_angle/R (rho→−1)"
        )
        ax.bar(x + 0.2, [r["share_amp"] for r in rows], 0.4, label="R_amp/R (|C|→|B|)")
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_xticks(x, [r["cond"] for r in rows], rotation=90, fontsize=7)
        ax.set_ylabel("residual share left by the counterfactual")
        ax.set_title("E24.2 knob decomposition (lower bar = better knob)")
        ax.legend()
        fig2.tight_layout()
        fig2.savefig(HERE / "e24_knobs.png", dpi=150)
        print("[fig] e24_knobs.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--figs_only", action="store_true")
    args = ap.parse_args()

    if args.figs_only:
        figures(json.loads((HERE / "e24_axis.json").read_text()))
        return

    validate_synthetic()
    if args.validate and not (STORES["e221"].exists()):
        raise SystemExit("e221 store missing")

    t0 = time.time()
    conds: list[Cond] = []
    for store, root in STORES.items():
        s = Sums(root)
        ghat_shared: dict[int, np.ndarray] = {}
        for bi in range(s.n_bins):
            for route in _routes(s.man):
                c = build_cond(s, store, route, bi)
                # routes of one (store, bin) share ghat — keep one copy
                if bi in ghat_shared:
                    c.ghat32 = ghat_shared[bi]
                else:
                    ghat_shared[bi] = c.ghat32
                conds.append(c)
                print(
                    f"[cond] {cond_id(c):22s} rho={c.rho:+.3f} "
                    f"relB={c.rel_B:.2f} relC={c.rel_C:.2f} "
                    f"{'verdict' if c.verdict else 'detached'}"
                )
            s.drop_bin(bi)  # free this bin's fp64 arm cache (~7.5 GB)
        del s

    validate_e221(conds)
    if args.validate:
        print("[validate] gates passed — stopping (--validate)")
        return

    # one Gram over every leg + unique ghat: all later reads are lookups
    global GRAM
    vecs: list[np.ndarray] = []
    ghat_idx: dict[int, int] = {}
    for c in conds:
        c.iB = len(vecs)
        vecs.append(c.B32)
        c.iC = len(vecs)
        vecs.append(c.C32)
        gid = id(c.ghat32)
        if gid not in ghat_idx:
            ghat_idx[gid] = len(vecs)
            vecs.append(c.ghat32)
        c.ighat = ghat_idx[gid]
    print(f"[gram] {len(vecs)} vectors x {vecs[0].shape[0]} dims ...")
    GRAM = build_gram(vecs)
    print(f"[gram] done ({time.time() - t0:.0f}s elapsed)")

    tables = family_tables(conds)
    digest = {
        "generated_by": "e24_axis.py",
        "rel_gate": REL_GATE,
        "verdict_sigma": sorted(VERDICT_SIGMA),
        "conditions": [
            {
                "cond": cond_id(c),
                "sigma": c.sigma,
                "G": round(c.G, 4),
                "S": round(c.S, 5),
                "F": round(c.F, 5),
                "I": round(c.I, 5),
                "rho": round(c.rho, 4),
                "rel_cos_B": round(c.rel_B, 4),
                "rel_cos_C": round(c.rel_C, 4),
                "verdict_eligible": c.verdict,
                "gated_B": gated(c, "B"),
                "gated_C": gated(c, "C"),
            }
            for c in conds
        ],
        "families": tables,
        "gram": {"B": gram_rank(conds, "B"), "C": gram_rank(conds, "C")},
        "knobs": knob_rows(conds),
        "context": {
            "ghat_cos": ghat_matrix(conds),
            "cross_leg_BC": cross_leg(conds),
            "e221_rows": e221_rows(conds),
        },
    }
    digest["verdict"] = verdict_from(tables)

    def _np_default(o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(f"not serializable: {type(o)}")

    (HERE / "e24_axis.json").write_text(
        json.dumps(digest, indent=1, default=_np_default)
    )
    print(f"[done] verdict {digest['verdict']}  ({time.time() - t0:.0f}s)")
    figures(digest)


def ghat_matrix(conds: list[Cond]) -> list[dict]:
    """Context: anchor motion. One row per unique (store, bin) pair."""
    seen: dict[str, Cond] = {}
    for c in conds:  # ghat identical across routes of one (store, bin)
        seen.setdefault(f"{c.store}/s{c.sigma:g}", c)
    keys = list(seen)
    rows = []
    for i, ki in enumerate(keys):
        for kj in keys[i + 1 :]:
            a, b = seen[ki], seen[kj]
            if GRAM is not None:
                cos = float(GRAM[a.ighat, b.ighat])
            else:
                cos = _dot64(a.ghat32, b.ghat32)
            rows.append({"x": ki, "y": kj, "cos": round(cos, 4)})
    return rows


def cross_leg(conds: list[Cond]) -> list[dict]:
    """Context: does span(B, C) form one global plane — cos(B_x, C_y) for
    gated verdict pairs across sigma within store/route (descriptive)."""
    rows = []
    for i, x in enumerate(conds):
        for y in conds[i + 1 :]:
            if pair_family(x, y) != "across_sigma":
                continue
            if not (gated(x, "B") and gated(y, "C")):
                continue
            num = _leg_dot(x, y, "B", "C")
            den = math.sqrt(x.nB2 * y.nC2)
            rows.append(
                {
                    "B_of": cond_id(x),
                    "C_of": cond_id(y),
                    "cos": round(num / den, 5) if den > 0 else None,
                }
            )
    return rows


def e221_rows(conds: list[Cond]) -> dict:
    """Consistency: e221 across-sigma cosines vs the e193/768 equivalents."""
    out: dict[str, list] = {"e221": [], "e193_768": []}
    for i, x in enumerate(conds):
        for y in conds[i + 1 :]:
            if x.store != y.store or x.route != y.route or x.sigma == y.sigma:
                continue
            if not (gated(x, "B") and gated(y, "B")):
                continue
            pair = {
                "x": cond_id(x),
                "y": cond_id(y),
                "cos_B": round(cross_cos(x, y, "B"), 5),
            }
            if x.store == "e221":
                out["e221"].append(pair)
            elif (
                x.store == "e193"
                and x.route == "768"
                and x.sigma in VERDICT_SIGMA
                and y.sigma in VERDICT_SIGMA
            ):
                out["e193_768"].append(pair)
    return out


if __name__ == "__main__":
    main()
