#!/usr/bin/env python
"""sigma_lowres E25.0 — prerequisite read (CPU).

Pre-registered in README.md (committed before this file existed). Two
questions on the three surviving arm_sums stores, E24 machinery verbatim
(conditions, legs, debias, gates imported from e24_axis.py):

(1) residual-direction reliability: is the pooled (B+C)-hat direction
    reproducible across draw sets per (route, sigma)?  rel_cos_R =
    raw cosine(R1_perp, R2_perp), the exact mirror of rel_cos_B/C;
    debiased residual power nR^2 = 2 G^2 (S+F+I) (identity gate).
(2) frame-relative stationarity: transported cosine cos_fr after the
    minimal rotation in span(ghat_x, ghat_y) taking ghat_x -> ghat_y is
    applied to the pooled leg mean; compared with the raw E24 cosine on
    the across-sigma family. Distinguishes literal co-rotation in
    ghat's motion plane (cos_fr ~ 1) from matched-angle rotation in an
    unrelated plane (cos_fr ~ cos_raw).

Verdicts frozen in README.md: 25.0-1 RELIABLE / UNRELIABLE / PARTIAL;
25.0-2 FRAME-STATIONARY / NO-GAIN / PARTIAL (precedence in that order).
Outputs (this dir): e250_read.json, e250_rel.png, e250_frame.png.

Usage::

    uv run python project/sigma_lowres/paper_bench/experiments/e25/e250_read.py
    ... --validate     # synthetic gates only, no store access
    ... --figs_only    # redraw figures from the committed record
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
E24_DIR = HERE.parent / "e24"
sys.path.insert(0, str(E24_DIR))

import numpy as np  # noqa: E402

import e24_axis as e24  # noqa: E402  (inserts REPO_ROOT for vector_ledger)
from e24_axis import (  # noqa: E402
    Cond,
    STORES,
    _routes,
    build_cond,
    build_gram,
    cond_id,
    cross_cos,
    gated,
    pair_family,
    validate_e221,
    validate_synthetic,
)

REL_GATE = e24.REL_GATE  # 0.5, inherited


# ------------------------------------------------------- residual extras


def residual_extras(s, route: str, bi: int, c: Cond) -> dict:
    """R = B + C per half-set (= perp'd dem - ref; rp cancels by linearity
    of perp). Same arithmetic recipe as build_cond, same Sums cache."""
    a, b = s.mean("a", bi), s.mean("b", bi)
    g0 = 0.5 * (a + b)
    G = float(np.linalg.norm(g0))
    ghat = g0 / G
    re1, re2 = s.mean("reenc", bi), s.mean("reenc__2", bi)
    ref, ref_diff = 0.5 * (re1 + re2), re1 - re2
    dem1, dem2 = s.mean(route, bi), s.mean(f"{route}__2", bi)

    def perp(v):
        return v - float(ghat @ v) * ghat

    R1p, R2p = perp(dem1 - ref), perp(dem2 - ref)
    rdp = perp(ref_diff)
    ref_noise = float(rdp @ rdp) / 4.0
    G2 = G * G
    rel_R = float((R1p @ R2p) / (np.linalg.norm(R1p) * np.linalg.norm(R2p)))
    R_quad = (float(R1p @ R2p) - ref_noise) / (2 * G2)
    identity_dev = abs(R_quad - (c.S + c.F + c.I))
    return {
        "rel_cos_R": rel_R,
        "R_quad": R_quad,
        "identity_dev": identity_dev,
        "nR2": max(R_quad * 2 * G2, 0.0),
        "pass1": bool(rel_R >= REL_GATE and R_quad > 0),
    }


# ---------------------------------------------------- Gram-entry helpers
# R-leg Gram entries via linearity: R32 = B32 + C32 exactly (fp32 sums are
# not re-materialized; entries are summed in fp64 from the leg Gram).


def _idx(c: Cond, leg: str) -> list[int]:
    if leg == "B":
        return [c.iB]
    if leg == "C":
        return [c.iC]
    if leg == "R":
        return [c.iB, c.iC]
    if leg == "g":
        return [c.ighat]
    raise ValueError(leg)


def gdot(x: Cond, lx: str, y: Cond, ly: str) -> float:
    return float(sum(e24.GRAM[i, j] for i in _idx(x, lx) for j in _idx(y, ly)))


def leg_n2(c: Cond, leg: str, rext: dict) -> float:
    if leg == "B":
        return c.nB2
    if leg == "C":
        return c.nC2
    return rext[cond_id(c)]["nR2"]


def r_cross_cos(x: Cond, y: Cond, rext: dict) -> float:
    """Debiased cross-condition cosine of the R leg. Same-store same-bin
    cross-route pairs share the reenc reference (R = dem - ref), so the
    shared ref-noise power is subtracted — the B-pair rule, mirrored."""
    num = gdot(x, "R", y, "R")
    if x.store == y.store and x.sigma == y.sigma and x.route != y.route:
        num -= x.ref_noise
    den = math.sqrt(leg_n2(x, "R", rext) * leg_n2(y, "R", rext))
    return num / den if den > 0 else float("nan")


def frame_cos(x: Cond, y: Cond, leg: str, rext: dict) -> float:
    """Debiased cosine after transporting x's pooled leg mean by the
    minimal rotation in span(ghat_x, ghat_y) taking ghat_x -> ghat_y.
    Exact in the Gram entries — no orthogonality shortcut. Rotation
    preserves norms, so denominators are the usual debiased norms."""
    cg = gdot(x, "g", y, "g")
    s2 = max(1.0 - cg * cg, 0.0)
    if s2 < 1e-12:  # frames coincide — transport is the identity
        raw = gdot(x, leg, y, leg)
        den = math.sqrt(leg_n2(x, leg, rext) * leg_n2(y, leg, rext))
        return raw / den if den > 0 else float("nan")
    s = math.sqrt(s2)
    w_u = gdot(x, leg, x, "g")  # <w, ghat_x>
    w_v = gdot(x, leg, y, "g")  # <w, ghat_y>
    y_u = gdot(y, leg, x, "g")  # <leg_y, ghat_x>
    y_v = gdot(y, leg, y, "g")  # <leg_y, ghat_y>
    a = w_u
    bcoef = (w_v - cg * a) / s
    p_dot_y = (y_v - cg * y_u) / s
    num = (
        gdot(x, leg, y, leg)
        + (a * (cg - 1.0) - bcoef * s) * y_u
        + (a * s + bcoef * (cg - 1.0)) * p_dot_y
    )
    den = math.sqrt(leg_n2(x, leg, rext) * leg_n2(y, leg, rext))
    return num / den if den > 0 else float("nan")


# ------------------------------------------------------------- validation


def validate_transport() -> None:
    """Planted rigid co-rotation in ghat's motion plane must yield
    cos_fr ~ 1 where raw ~ cos(ghat, ghat); an out-of-plane control
    (axis motion in a plane not containing ghat's motion) must NOT
    inflate. Residual identity + planted-residual recovery checked on
    the same synthetic stores."""
    rng = np.random.default_rng(7)
    dim, draws = 4096, 6
    theta = math.radians(60.0)
    ct, st = math.cos(theta), math.sin(theta)

    def unit(v):
        return v / np.linalg.norm(v)

    u = unit(rng.normal(size=dim))
    p = rng.normal(size=dim)
    p = unit(p - float(p @ u) * u)
    q1 = rng.normal(size=dim)
    q1 = unit(q1 - float(q1 @ u) * u - float(q1 @ p) * p)
    q2 = rng.normal(size=dim)
    q2 = unit(
        q2 - float(q2 @ u) * u - float(q2 @ p) * p - float(q2 @ q1) * q1
    )
    v = ct * u + st * p  # ghat_y: ghat rotated by theta in span(u, p)

    class FakeSums:
        def __init__(self, arms):
            self.man = {
                "sigma_centers": [0.5],
                "demote_edges": "768",
                "keys": {k: {"n_images": 1} for k in arms},
            }
            self.draws = draws
            self._arms = arms

        def mean(self, key, bi):
            return self._arms[key].astype(np.float64) / (1 * draws)

        def has(self, key):
            return key in self._arms

    def mk(g_dir, b_dir, c_scale=-0.85, noise=0.005):
        g = g_dir * 30.0
        Bt = b_dir * 3.0
        Ct = c_scale * Bt + 0.02 * rng.normal(size=dim)
        ref = g + rng.normal(size=dim) * 0.01
        arms = {}
        for sfx in ("", "__2"):
            n = lambda: rng.normal(size=dim) * noise  # noqa: E731
            arms[f"reenc{sfx}"] = (ref + n()) * draws
            arms[f"768rp{sfx}"] = (ref + Bt + n()) * draws
            arms[f"768{sfx}"] = (ref + Bt + Ct + n()) * draws
        arms["a"] = (g + rng.normal(size=dim) * 0.01) * draws
        arms["b"] = (g - rng.normal(size=dim) * 0.01) * draws
        return FakeSums(arms), Bt + Ct

    # in-plane co-rotation: B_x = p (perp to u), B_y = Rot(theta) p
    sx, res_x = mk(u, p)
    sy, _ = mk(v, -st * u + ct * p)
    # out-of-plane control: B rotates by theta in span(q1, q2) — a plane
    # disjoint from ghat's motion plane span(u, p)
    so, _ = mk(v, ct * q1 + st * q2)
    sq, _ = mk(u, q1)

    conds, rext = [], {}
    for store, s in (("sx", sx), ("sy", sy), ("sq", sq), ("so", so)):
        c = build_cond(s, store, "768", 0)
        conds.append(c)
        rext[cond_id(c)] = residual_extras(s, "768", 0, c)

    # residual identity + planted-residual recovery on the synthetic store
    for c in conds:
        r = rext[cond_id(c)]
        assert r["identity_dev"] <= 1e-8, (cond_id(c), r["identity_dev"])
    r0 = rext[cond_id(conds[0])]
    assert r0["rel_cos_R"] > 0.9, r0["rel_cos_R"]  # low noise -> reliable
    # noisy store: residual direction must be destroyed
    sn, _ = mk(u, p, noise=3.0)
    cn = build_cond(sn, "sn", "768", 0)
    rn = residual_extras(sn, "768", 0, cn)
    assert abs(rn["rel_cos_R"]) < 0.5, rn["rel_cos_R"]

    vecs, ghat_idx = [], {}
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
    e24.GRAM = build_gram(vecs, chunk=dim)

    cx, cy, cq, co = conds
    raw_in = cross_cos(cx, cy, "B")
    fr_in = frame_cos(cx, cy, "B", rext)
    assert abs(raw_in - ct) < 0.05, (raw_in, ct)  # raw ~ cos(ghat, ghat)
    assert fr_in > 0.95, fr_in  # transport recovers the co-rotation
    raw_out = cross_cos(cq, co, "B")
    fr_out = frame_cos(cq, co, "B", rext)
    assert abs(raw_out - ct) < 0.05, (raw_out, ct)
    assert abs(fr_out - raw_out) < 0.05, (fr_out, raw_out)  # no inflation
    assert fr_out < 0.7, fr_out
    e24.GRAM = None
    print(
        "[validate] transport: in-plane raw "
        f"{raw_in:.3f} -> fr {fr_in:.3f}; out-of-plane raw {raw_out:.3f} "
        f"-> fr {fr_out:.3f} (no inflation); residual identity + "
        "planted rel_cos_R recovered"
    )


# ------------------------------------------------------------------ main


def read1(conds: list[Cond], rext: dict) -> dict:
    rows = []
    for c in conds:
        r = rext[cond_id(c)]
        rows.append(
            {
                "cond": cond_id(c),
                "sigma": c.sigma,
                "rel_cos_R": round(r["rel_cos_R"], 4),
                "rel_cos_B": round(c.rel_B, 4),
                "rel_cos_C": round(c.rel_C, 4),
                "R_quad": round(r["R_quad"], 5),
                "nR2_pos": bool(r["nR2"] > 0),
                "pass": r["pass1"],
                "verdict_eligible": c.verdict,
            }
        )
    vrows = [r for r in rows if r["verdict_eligible"]]
    rels = [r["rel_cos_R"] for r in vrows]
    if all(r["pass"] for r in vrows):
        verdict = "RELIABLE"
    elif float(np.median(rels)) < REL_GATE:
        verdict = "UNRELIABLE"
    else:
        verdict = "PARTIAL"
    return {
        "verdict": verdict,
        "n_verdict_conditions": len(vrows),
        "n_pass": sum(r["pass"] for r in vrows),
        "median_rel_cos_R": round(float(np.median(rels)), 4),
        "min_rel_cos_R": round(min(rels), 4),
        "conditions": rows,
    }


def read2(conds: list[Cond], rext: dict) -> dict:
    out: dict = {}
    for leg in ("B", "C"):
        pairs = []
        for i, x in enumerate(conds):
            for y in conds[i + 1 :]:
                if pair_family(x, y) != "across_sigma":
                    continue
                if not (gated(x, leg) and gated(y, leg)):
                    continue
                raw = cross_cos(x, y, leg)
                fr = frame_cos(x, y, leg, rext)
                pairs.append(
                    {
                        "x": cond_id(x),
                        "y": cond_id(y),
                        "span": round(abs(x.sigma - y.sigma), 4),
                        "cos_ghat": round(gdot(x, "g", y, "g"), 4),
                        "cos_raw": round(raw, 5),
                        "cos_fr": round(fr, 5),
                        "delta": round(abs(fr) - abs(raw), 5),
                    }
                )
        afr = [abs(p["cos_fr"]) for p in pairs]
        deltas = [p["delta"] for p in pairs]
        stats = {
            "n_pairs": len(pairs),
            "median_abs_fr": round(float(np.median(afr)), 5) if afr else None,
            "min_abs_fr": round(min(afr), 5) if afr else None,
            "median_abs_raw": round(float(np.median([abs(p["cos_raw"]) for p in pairs])), 5)
            if pairs
            else None,
            "min_abs_raw": round(min(abs(p["cos_raw"]) for p in pairs), 5)
            if pairs
            else None,
            "median_delta": round(float(np.median(deltas)), 5) if deltas else None,
            "pairs": pairs,
        }
        out[leg] = stats
    b = out["B"]
    if b["median_abs_fr"] is None:
        verdict = "PARTIAL (no gated pairs — record only)"
    elif b["median_abs_fr"] >= 0.7 and b["min_abs_fr"] >= 0.5:
        verdict = "FRAME-STATIONARY"
    elif b["median_delta"] <= 0.02:
        verdict = "NO-GAIN"
    else:
        verdict = "PARTIAL"
    out["verdict"] = verdict
    return out


def r_descriptive(conds: list[Cond], rext: dict) -> dict:
    """Raw + transported R-hat cosines over read-1-passing verdict
    conditions, in the three E24 families (descriptive, no verdict)."""
    passing = [c for c in conds if c.verdict and rext[cond_id(c)]["pass1"]]
    fams: dict[str, list] = {
        "across_sigma": [],
        "across_route": [],
        "across_store": [],
    }
    for i, x in enumerate(passing):
        for y in passing[i + 1 :]:
            fam = pair_family(x, y)
            if fam is None:
                continue
            row = {
                "x": cond_id(x),
                "y": cond_id(y),
                "cos_raw": round(r_cross_cos(x, y, rext), 5),
            }
            if fam == "across_sigma":
                row["cos_fr"] = round(frame_cos(x, y, "R", rext), 5)
            fams[fam].append(row)
    stats = {}
    for fam, rows in fams.items():
        av = [abs(r["cos_raw"]) for r in rows]
        stats[fam] = {
            "n_pairs": len(rows),
            "median_abs_raw": round(float(np.median(av)), 5) if av else None,
            "min_abs_raw": round(min(av), 5) if av else None,
            "pairs": rows,
        }
        if fam == "across_sigma" and rows:
            afr = [abs(r["cos_fr"]) for r in rows]
            stats[fam]["median_abs_fr"] = round(float(np.median(afr)), 5)
            stats[fam]["min_abs_fr"] = round(min(afr), 5)
    # e221 descriptive rows (never verdict): reliability only
    stats["e221_rel"] = [
        {
            "cond": cond_id(c),
            "rel_cos_R": round(rext[cond_id(c)]["rel_cos_R"], 4),
            "pass": rext[cond_id(c)]["pass1"],
        }
        for c in conds
        if c.store == "e221"
    ]
    return stats


def figures(digest: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = digest["read1"]["conditions"]
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(rows)), 4.5))
    x = np.arange(len(rows))
    colors = [
        ("tab:blue" if r["pass"] else "tab:red")
        if r["verdict_eligible"]
        else "0.7"
        for r in rows
    ]
    ax.bar(x, [r["rel_cos_R"] for r in rows], 0.62, color=colors)
    ax.plot(x, [r["rel_cos_B"] for r in rows], "k^", ms=4, label="rel_cos_B")
    ax.plot(x, [r["rel_cos_C"] for r in rows], "kv", ms=4, label="rel_cos_C")
    ax.axhline(REL_GATE, color="k", lw=0.8, ls="--", label=f"gate {REL_GATE}")
    ax.set_xticks(x, [r["cond"] for r in rows], rotation=90, fontsize=7)
    ax.set_ylabel("cross-set direction cosine")
    ax.set_title(
        "E25.0-1 residual-direction reliability (bars = rel_cos_R; gray = "
        f"descriptive) — {digest['read1']['verdict']}"
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "e250_rel.png", dpi=150)
    print("[fig] e250_rel.png")

    fig2, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, leg in zip(axes, ("B", "C")):
        pairs = sorted(digest["read2"][leg]["pairs"], key=lambda p: p["span"])
        xs = np.arange(len(pairs))
        ax.plot(xs, [abs(p["cos_raw"]) for p in pairs], "o-", label="|cos| raw")
        ax.plot(
            xs,
            [abs(p["cos_fr"]) for p in pairs],
            "s-",
            label="|cos| transported",
        )
        ax.plot(
            xs, [abs(p["cos_ghat"]) for p in pairs], ":", color="0.5", label="|cos(ĝ,ĝ)|"
        )
        ax.axhline(0.5, color="k", lw=0.8, ls="--")
        ax.set_xticks(
            xs,
            [f"{p['x'].split('/', 1)[1]}↔{p['y'].rsplit('s', 1)[1]}" for p in pairs],
            rotation=90,
            fontsize=6,
        )
        ax.set_title(f"{leg} legs (pairs sorted by σ span)")
        ax.legend(fontsize=7)
    fig2.suptitle(
        f"E25.0-2 frame-relative stationarity — {digest['read2']['verdict']}"
    )
    fig2.tight_layout()
    fig2.savefig(HERE / "e250_frame.png", dpi=150)
    print("[fig] e250_frame.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--figs_only", action="store_true")
    args = ap.parse_args()

    if args.figs_only:
        figures(json.loads((HERE / "e250_read.json").read_text()))
        return

    validate_synthetic()  # E24 gate: build_cond == bc_ledger, axis recovery
    validate_transport()  # E25.0 gates: transport + residual identity
    if args.validate:
        print("[validate] synthetic gates passed — stopping (--validate)")
        return

    t0 = time.time()
    conds: list[Cond] = []
    rext: dict[str, dict] = {}
    for store, root in STORES.items():
        s = e24.Sums(root)
        ghat_shared: dict[int, np.ndarray] = {}
        for bi in range(s.n_bins):
            for route in _routes(s.man):
                c = build_cond(s, store, route, bi)
                if bi in ghat_shared:
                    c.ghat32 = ghat_shared[bi]
                else:
                    ghat_shared[bi] = c.ghat32
                r = residual_extras(s, route, bi, c)
                conds.append(c)
                rext[cond_id(c)] = r
                print(
                    f"[cond] {cond_id(c):22s} relR={r['rel_cos_R']:+.3f} "
                    f"(relB={c.rel_B:.2f} relC={c.rel_C:.2f}) "
                    f"idev={r['identity_dev']:.1e} "
                    f"{'pass' if r['pass1'] else 'FAIL'}"
                    f"{'' if c.verdict else ' (descriptive)'}"
                )
            s.drop_bin(bi)
        del s

    validate_e221(conds)  # E24 gate: committed ledger reproduced exactly
    max_idev = max(r["identity_dev"] for r in rext.values())
    assert max_idev <= 1e-6, f"nR2 identity violated: {max_idev}"
    print(f"[validate] nR2 identity on real stores: max dev {max_idev:.2e}")

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
    e24.GRAM = build_gram(vecs)
    print(f"[gram] done ({time.time() - t0:.0f}s elapsed)")

    digest = {
        "generated_by": "e250_read.py",
        "rel_gate": REL_GATE,
        "verdict_sigma": sorted(e24.VERDICT_SIGMA),
        "validation": {"nR2_identity_max_dev": max_idev},
        "read1": read1(conds, rext),
        "read2": read2(conds, rext),
        "r_descriptive": r_descriptive(conds, rext),
    }

    def _np_default(o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(f"not serializable: {type(o)}")

    (HERE / "e250_read.json").write_text(
        json.dumps(digest, indent=1, default=_np_default)
    )
    print(
        f"[done] 25.0-1 {digest['read1']['verdict']}  "
        f"25.0-2 {digest['read2']['verdict']}  ({time.time() - t0:.0f}s)"
    )
    figures(digest)


if __name__ == "__main__":
    main()
