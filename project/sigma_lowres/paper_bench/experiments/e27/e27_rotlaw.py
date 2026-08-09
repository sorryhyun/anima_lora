#!/usr/bin/env python
"""sigma_lowres E27 — σ-rotation law read (CPU).

Pre-registered in README.md (committed before this file existed). One
pass over the three surviving arm_sums stores, asking whether the E24
axis field's σ-rotation upgrades from description to model:

 * 27.1 plane σ-stability: LOO in-plane share of held-out directions
   in the top-2 eigenplane fitted on the other σ bins.
 * 27.2 law vs SLERP (the Q7 verdict, B legs): fixed-plane geodesic
   θ(σ) = θ0 + ω·ln σ (L-log, primary) / θ0 + ω·σ (L-lin, secondary),
   scored as debiased |cos| at held-out bins against nearest-copy (a)
   and flanking-SLERP (b). LAW-BEATS-SLERP / NO-GAIN / LAW-WORSE with
   the frozen Δ = 0.02 margin.
 * 27.3 R̂ transfer (descriptive, E25a-facing): same scoring on R legs.
 * 27.4 phase generator anchor: V_phase = (C − C_pi)⊥ from e194's
   stored π arms vs the C-leg in-plane tangent t̂_C(σ). TANGENT-ALIGNED
   / AXIS-ALIGNED / NULL / MIXED.

All cross-condition arithmetic lives in per-leg unit Grams (diag = 1,
off-diagonal = the E24/E25 debiased cosines); derived directions
(plane bases, SLERP, predictions, tangents) are coefficient vectors
over condition units, so every score is Gram algebra.

Usage::

    uv run python project/sigma_lowres/paper_bench/experiments/e27/e27_rotlaw.py
    ... --validate     # synthetic + e221 gates only, no Gram build
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
sys.path.insert(0, str(HERE.parent / "e24"))
sys.path.insert(0, str(HERE.parent / "e25"))

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
    validate_e221,
    validate_synthetic,
)
from e250_read import r_cross_cos, residual_extras  # noqa: E402

REL_GATE = e24.REL_GATE  # 0.5, inherited
TARGET_SIGMA = (0.4333, 0.5667, 0.7, 0.8333, 0.9625)  # interior bins only
ANCHOR_SIGMA = (0.7, 0.8333, 0.9625)
DELTA_MARGIN = 0.02  # mirrors E25.0-2's NO-GAIN threshold
VARIANTS = {"L-log": lambda s: math.log(s), "L-lin": lambda s: s}
PRIMARY = "L-log"


# ------------------------------------------------------------ V_phase pass


def vphase_extras(s, route: str, bi: int, c: Cond) -> dict:
    """V_phase = (C − C_pi)⊥ = perp(dem − dem_pi); rp cancels by
    linearity. Same arithmetic recipe / Sums cache as build_cond. The
    same-bin ⟨V̂p, Ĉ⟩ shares the dem arm's draw noise within a half-set,
    so it is cross-half debiased here (the only place it is needed)."""
    a, b = s.mean("a", bi), s.mean("b", bi)
    g0 = 0.5 * (a + b)
    ghat = g0 / float(np.linalg.norm(g0))

    def perp(v):
        return v - float(ghat @ v) * ghat

    dem1, dem2 = s.mean(route, bi), s.mean(f"{route}__2", bi)
    pi1, pi2 = s.mean(f"{route}pi", bi), s.mean(f"{route}pi__2", bi)
    rp1, rp2 = s.mean(f"{route}rp", bi), s.mean(f"{route}rp__2", bi)
    Vp1, Vp2 = perp(dem1 - pi1), perp(dem2 - pi2)
    C1p, C2p = perp(dem1 - rp1), perp(dem2 - rp2)
    # linearity identity: perp(dem − dem_pi) == C − C_pi, elementwise
    Cpi1 = perp(pi1 - rp1)
    dev = float(np.max(np.abs(Vp1 - (C1p - Cpi1))))
    scale = float(np.max(np.abs(Vp1))) or 1.0
    nVp2 = float(Vp1 @ Vp2)
    rel = float(
        (Vp1 @ Vp2) / (np.linalg.norm(Vp1) * np.linalg.norm(Vp2))
    )
    vc_num = 0.5 * (float(Vp1 @ C2p) + float(Vp2 @ C1p))
    return {
        "rel_cos_Vp": rel,
        "nVp2": nVp2,
        "vc_same_num": vc_num,  # cross-half ⟨V̄p, C̄⟩, same condition
        "identity_rel_dev": dev / scale,
        "readable": bool(rel >= REL_GATE and nVp2 > 0),
        "Vp32": (0.5 * (Vp1 + Vp2)).astype(np.float32),
    }


# ------------------------------------------------------- unit-Gram machinery


class Row:
    """One condition's seat in a per-leg unit Gram."""

    __slots__ = ("cid", "sigma", "route", "store", "cond")

    def __init__(self, cond: Cond):
        self.cid = cond_id(cond)
        self.sigma = round(cond.sigma, 4)
        self.route = cond.route
        self.store = cond.store
        self.cond = cond


def sort_rows(rows: list[Row]) -> list[Row]:
    """Frozen order: ascending σ; ties 896 before 768, e193 before e194."""
    return sorted(
        rows, key=lambda r: (r.sigma, 0 if r.route == "896" else 1, r.store)
    )


def unit_gram(rows: list[Row], pairfn) -> np.ndarray:
    n = len(rows)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = pairfn(rows[i].cond, rows[j].cond)
    return M


def orient(rows: list[Row], M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Chain orientation along the frozen order: flip each unit to keep a
    non-negative cosine with the previously oriented condition."""
    n = len(rows)
    signs = np.ones(n)
    for i in range(1, n):
        if signs[i - 1] * M[i, i - 1] < 0:
            signs[i] = -1.0
    Mo = M * np.outer(signs, signs)
    return signs, Mo


def plane_fit(Mo_fit: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Top-2 eigenplane of the (oriented) fit Gram. Returns coefficient
    vectors a1, a2 over the fit conditions (kernel-PCA relation: the
    ambient basis e_k = Σ_i a_k[i]·û_i is orthonormal)."""
    lam, W = np.linalg.eigh(Mo_fit)
    l1, l2 = float(lam[-1]), float(lam[-2])
    a1 = W[:, -1] / math.sqrt(l1)
    a2 = W[:, -2] / math.sqrt(l2)
    tot = float(np.clip(lam, 0.0, None).sum())
    info = {
        "lam_top2": [round(l1, 4), round(l2, 4)],
        "top2_share_fit": round((l1 + l2) / tot, 4) if tot > 0 else None,
    }
    return a1, a2, info


def pooled_unit(fit_idx: list[int], Mo: np.ndarray, members: list[int]) -> np.ndarray:
    """Sign-aligned unit-mean of `members` (already oriented), as a
    coefficient vector over fit_idx, renormalized in the Gram metric."""
    v = np.zeros(len(fit_idx))
    pos = {g: k for k, g in enumerate(fit_idx)}
    for m in members:
        v[pos[m]] += 1.0 / len(members)
    Mf = Mo[np.ix_(fit_idx, fit_idx)]
    n2 = float(v @ Mf @ v)
    return v / math.sqrt(n2) if n2 > 0 else v


def loo_read(rows: list[Row], Mo: np.ndarray) -> dict:
    """The 27.1/27.2 machinery for one leg's oriented unit Gram."""
    sigmas = np.array([r.sigma for r in rows])
    folds, cond_rows = [], []
    for st in TARGET_SIGMA:
        tgt = [i for i in range(len(rows)) if sigmas[i] == round(st, 4)]
        fit = [i for i in range(len(rows)) if sigmas[i] != round(st, 4)]
        if not tgt or len(fit) < 3:
            folds.append({"sigma_t": st, "skipped": True})
            continue
        Mf = Mo[np.ix_(fit, fit)]
        a1, a2, pinfo = plane_fit(Mf)
        sig_fit = sigmas[fit]
        # in-plane angles of the fit conditions, unwrapped along σ
        comp1 = Mf @ a1
        comp2 = Mf @ a2
        order = np.argsort(sig_fit, kind="stable")
        th_sorted = np.unwrap(np.arctan2(comp2[order], comp1[order]))
        theta = np.empty_like(th_sorted)
        theta[order] = th_sorted
        fits = {}
        for name, f in VARIANTS.items():
            X = np.stack([np.ones(len(fit)), np.array([f(s) for s in sig_fit])]).T
            coef, *_ = np.linalg.lstsq(X, theta, rcond=None)
            fits[name] = coef  # (theta0, omega)
        # baselines: pooled flank / nearest-copy directions
        fit_sigmas = sorted(set(sig_fit))
        lows = [s for s in fit_sigmas if s < st]
        highs = [s for s in fit_sigmas if s > st]
        s_lo, s_hi = max(lows), min(highs)
        mem = lambda s: [g for g in fit if sigmas[g] == s]  # noqa: E731
        v_lo = pooled_unit(fit, Mo, mem(s_lo))
        v_hi = pooled_unit(fit, Mo, mem(s_hi))
        cosom = float(np.clip(v_lo @ Mf @ v_hi, -1.0, 1.0))
        if cosom < 0:  # sign-align (should not trigger post-orientation)
            v_hi, cosom = -v_hi, -cosom
        om = math.acos(cosom)
        t = (st - s_lo) / (s_hi - s_lo)
        if om > 1e-6:
            v_slerp = (
                math.sin((1 - t) * om) * v_lo + math.sin(t * om) * v_hi
            ) / math.sin(om)
        else:
            v = (1 - t) * v_lo + t * v_hi
            v_slerp = v / math.sqrt(float(v @ Mf @ v))
        s_near = min(fit_sigmas, key=lambda s: (abs(st - s), s))
        v_copy = pooled_unit(fit, Mo, mem(s_near))
        fold = {
            "sigma_t": st,
            "fit_n": len(fit),
            "flanks": [s_lo, s_hi],
            "copy_from": s_near,
            "plane": pinfo,
            "theta_fit": {
                k: [round(float(c), 4) for c in v] for k, v in fits.items()
            },
            "targets": [],
        }
        for j in tgt:
            col = Mo[fit, j]
            share = float((a1 @ col) ** 2 + (a2 @ col) ** 2)
            row = {
                "cond": rows[j].cid,
                "plane_share": round(share, 4),
                "cos_slerp": round(float(v_slerp @ col), 5),
                "cos_copy": round(float(v_copy @ col), 5),
            }
            for name, (th0, omega) in fits.items():
                th_t = th0 + omega * VARIANTS[name](st)
                p = math.cos(th_t) * a1 + math.sin(th_t) * a2
                row[f"cos_{name}"] = round(float(p @ col), 5)
            row["delta_primary"] = round(
                abs(row[f"cos_{PRIMARY}"]) - abs(row["cos_slerp"]), 5
            )
            fold["targets"].append(row)
            cond_rows.append(row)
        folds.append(fold)

    shares = [t["plane_share"] for t in cond_rows]
    if shares:
        med_s, min_s = float(np.median(shares)), min(shares)
        if med_s >= 0.8 and min_s >= 0.6:
            v271 = "PLANE-STABLE"
        elif med_s < 0.6:
            v271 = "PLANE-UNSTABLE"
        else:
            v271 = "PLANE-MIXED"
    else:
        v271, med_s, min_s = "NO-TARGETS", None, None

    deltas = [t["delta_primary"] for t in cond_rows]
    if deltas:
        med_d = float(np.median(deltas))
        wins = sum(d > 0 for d in deltas) / len(deltas)
        losses = sum(d < 0 for d in deltas) / len(deltas)
        if med_d >= DELTA_MARGIN and wins >= 2 / 3:
            v272 = "LAW-BEATS-SLERP"
        elif med_d <= -DELTA_MARGIN and losses >= 2 / 3:
            v272 = "LAW-WORSE"
        else:
            v272 = "NO-GAIN"
    else:
        v272, med_d, wins = "NO-TARGETS", None, None

    def med(key):
        vals = [abs(t[key]) for t in cond_rows]
        return round(float(np.median(vals)), 5) if vals else None

    return {
        "folds": folds,
        "n_target_conditions": len(cond_rows),
        "verdict_27_1": v271,
        "plane_share_median": round(med_s, 4) if med_s is not None else None,
        "plane_share_min": round(min_s, 4) if min_s is not None else None,
        "verdict_27_2": v272,
        "delta_primary_median": round(med_d, 5) if med_d is not None else None,
        "delta_primary_win_share": round(wins, 3) if wins is not None else None,
        "median_abs_cos": {
            "slerp": med("cos_slerp"),
            "copy": med("cos_copy"),
            **{k: med(f"cos_{k}") for k in VARIANTS},
        },
        "secondary_deltas": [
            round(abs(t["cos_L-lin"]) - abs(t["cos_slerp"]), 5)
            for t in cond_rows
        ],
    }


# ----------------------------------------------------------- anchor (27.4)


def tangent_coeff(
    rows: list[Row], Mo: np.ndarray, j: int, lo: int, hi: int
) -> np.ndarray | None:
    """Central-difference in-plane tangent at rows[j]: û_hi − û_lo
    (oriented), projected ⊥ û_j, normalized — as a coefficient vector
    over the full row set."""
    n = len(rows)
    tvec = np.zeros(n)
    tvec[hi], tvec[lo] = 1.0, -1.0
    proj = float(tvec @ Mo[:, j])
    tvec[j] -= proj
    n2 = float(tvec @ Mo @ tvec)
    if n2 <= 0:
        return None
    return tvec / math.sqrt(n2)


def pick_neighbor(
    rows: list[Row], route: str, store: str, sigma: float
) -> int | None:
    """Gated same-route condition at σ: same store preferred, else the
    other standard store."""
    cands = [
        i
        for i, r in enumerate(rows)
        if r.route == route and r.sigma == round(sigma, 4)
    ]
    if not cands:
        return None
    same = [i for i in cands if rows[i].store == store]
    return same[0] if same else cands[0]


def anchor_read(
    c_rows: list[Row],
    Mo_C: np.ndarray,
    signs_C: np.ndarray,
    vext: dict,
    vp_gram_col,
    grid: list[float],
) -> dict:
    """27.4: V̂_phase vs the C-leg tangent and the C axis itself."""
    a1, a2, pinfo = plane_fit(Mo_C)  # full-data C plane (descriptive)
    out_rows = []
    for route in ("768", "896"):
        for st in ANCHOR_SIGMA:
            cid = f"e194/{route}/s{st:g}"
            v = vext.get(cid)
            row = {"cond": cid, "sigma": st, "route": route}
            if v is None:
                continue
            row["rel_cos_Vp"] = round(v["rel_cos_Vp"], 4)
            row["readable_Vp"] = v["readable"]
            j = pick_neighbor(c_rows, route, "e194", st)
            gi = grid.index(round(st, 4))
            lo = pick_neighbor(c_rows, route, "e194", grid[gi - 1])
            hi = (
                pick_neighbor(c_rows, route, "e194", grid[gi + 1])
                if gi + 1 < len(grid)
                else None
            )
            if not v["readable"] or j is None or lo is None or hi is None:
                row["readable"] = False
                out_rows.append(row)
                continue
            # debiased cos(V̂p, û_C,k) for every C row (oriented)
            vc = np.empty(len(c_rows))
            for k, r in enumerate(c_rows):
                if r.cid == cid:
                    num = v["vc_same_num"]  # cross-half (shared dem arm)
                else:
                    num = vp_gram_col(cid, r.cond)
                vc[k] = (
                    signs_C[k]
                    * num
                    / math.sqrt(v["nVp2"] * r.cond.nC2)
                )
            t_hat = tangent_coeff(c_rows, Mo_C, j, lo, hi)
            row["readable"] = t_hat is not None
            if t_hat is None:
                out_rows.append(row)
                continue
            row["cos_tangent"] = round(float(t_hat @ vc), 5)
            row["cos_axis"] = round(float(vc[j]), 5)
            row["plane_share_Vp"] = round(
                float((a1 @ vc) ** 2 + (a2 @ vc) ** 2), 4
            )
            row["tangent_flanks"] = [c_rows[lo].cid, c_rows[hi].cid]
            out_rows.append(row)

    readable = [
        r for r in out_rows if r.get("readable") and r["route"] == "768"
    ]
    if readable:
        mt = float(np.median([abs(r["cos_tangent"]) for r in readable]))
        ma = float(np.median([abs(r["cos_axis"]) for r in readable]))
        t_dom = all(
            abs(r["cos_tangent"]) > abs(r["cos_axis"]) for r in readable
        )
        a_dom = all(
            abs(r["cos_axis"]) > abs(r["cos_tangent"]) for r in readable
        )
        if mt >= 0.5 and t_dom:
            verdict = "TANGENT-ALIGNED"
        elif ma >= 0.5 and a_dom:
            verdict = "AXIS-ALIGNED"
        elif mt < 0.3 and ma < 0.3:
            verdict = "NULL"
        else:
            verdict = "MIXED"
    else:
        verdict, mt, ma = "NO-READABLE-BINS", None, None
    return {
        "verdict_27_4": verdict,
        "median_abs_cos_tangent_768": round(mt, 4) if mt is not None else None,
        "median_abs_cos_axis_768": round(ma, 4) if ma is not None else None,
        "full_plane": pinfo,
        "rows": out_rows,
    }


# ------------------------------------------------------------- validation


def _synth_ug(vec_pairs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Debiased unit Gram from synthetic half-set vectors, mirroring the
    real conventions (cross-condition numerators = pooled-mean products;
    denominators = cross-half debiased norms)."""
    n = len(vec_pairs)
    pooled = [0.5 * (v1 + v2) for v1, v2 in vec_pairs]
    den = [math.sqrt(float(v1 @ v2)) for v1, v2 in vec_pairs]
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = float(pooled[i] @ pooled[j]) / (den[i] * den[j])
    return M


class _FakeRow:
    __slots__ = ("cid", "sigma", "route", "store", "cond")

    def __init__(self, cid, sigma, route, store):
        self.cid, self.sigma = cid, round(sigma, 4)
        self.route, self.store = route, store
        self.cond = None


_GRID = [0.3, 0.4333, 0.5667, 0.7, 0.8333, 0.9625, 1.0]


def _mk_field(rng, dim, planar: bool, eps: float):
    """Synthetic axis field over the real σ grid × 2 routes: fixed-plane
    geodesic θ(σ) = 0.3 + 1.2·ln σ, optionally lifted out of plane."""
    basis = np.linalg.qr(rng.normal(size=(dim, 3)))[0]
    e1, e2, e3 = basis.T
    rows, pairs, trues = [], [], []
    for s in _GRID:
        th = 0.3 + 1.2 * math.log(s)
        u = math.cos(th) * e1 + math.sin(th) * e2
        if not planar:
            ph = 1.6 * (math.log(s) + 1.204) / 1.204 - 0.8
            u = math.cos(ph) * u + math.sin(ph) * e3
        for route in ("896", "768"):
            pert = rng.normal(size=dim)
            d = u + 0.02 * pert / np.linalg.norm(pert)
            d /= np.linalg.norm(d)
            v1 = d + eps * rng.normal(size=dim) / math.sqrt(dim)
            v2 = d + eps * rng.normal(size=dim) / math.sqrt(dim)
            rows.append(_FakeRow(f"syn/{route}/s{s:g}", s, route, "e193"))
            pairs.append((v1, v2))
            trues.append(d)
    return rows, pairs, trues, (e1, e2, e3)


def validate_geodesic() -> None:
    """Planted planar geodesic: the plane and ω must be recovered, the
    law must beat nearest-copy (which has a true geometry error) and
    hold parity with SLERP (a planar geodesic is SLERP-representable —
    the amended gate; see the pre-registration-defect note in
    README.md). The non-planar control must degrade the plane and must
    not produce a spurious LAW-BEATS-SLERP."""
    rng = np.random.default_rng(11)
    dim = 4096
    rows, pairs, _, _ = _mk_field(rng, dim, planar=True, eps=0.7)
    M = _synth_ug(pairs)
    _, Mo = orient(rows, M)
    res = loo_read(rows, Mo)
    assert res["verdict_27_1"] == "PLANE-STABLE", res["verdict_27_1"]
    assert res["plane_share_median"] > 0.9, res["plane_share_median"]
    assert res["delta_primary_median"] > -DELTA_MARGIN, res[
        "delta_primary_median"
    ]
    tgts = [t for f in res["folds"] if "targets" in f for t in f["targets"]]
    d_copy = float(
        np.median([abs(t[f"cos_{PRIMARY}"]) - abs(t["cos_copy"]) for t in tgts])
    )
    assert d_copy > DELTA_MARGIN, d_copy
    omegas = [
        f["theta_fit"][PRIMARY][1] for f in res["folds"] if "theta_fit" in f
    ]
    for w in omegas:
        assert abs(w - 1.2) / 1.2 < 0.2, omegas
    # non-planar control: must NOT pass, and the plane must degrade
    rows_n, pairs_n, _, _ = _mk_field(rng, dim, planar=False, eps=0.7)
    Mn = _synth_ug(pairs_n)
    _, Mon = orient(rows_n, Mn)
    res_n = loo_read(rows_n, Mon)
    assert res_n["verdict_27_2"] != "LAW-BEATS-SLERP", res_n["verdict_27_2"]
    assert res_n["plane_share_median"] < 0.85, res_n["plane_share_median"]
    print(
        f"[validate] geodesic: plane+ω recovered (ω {omegas[0]:.2f}), "
        f"law>copy {d_copy:+.3f}, law~slerp "
        f"{res['delta_primary_median']:+.3f}; non-planar control: share "
        f"{res_n['plane_share_median']:.2f}, {res_n['verdict_27_2']}"
    )


def validate_vphase_tangent() -> None:
    """Planted phase component along the known tangent must be recovered
    through the coefficient-space tangent read; an orthogonal control
    must not inflate."""
    rng = np.random.default_rng(23)
    dim = 4096
    rows, pairs, trues, (e1, e2, e3) = _mk_field(rng, dim, planar=True, eps=0.3)
    M = _synth_ug(pairs)
    signs, Mo = orient(rows, M)
    # anchor at σ = 0.8333, route 768 (index lookup in the frozen order)
    rs = sort_rows(rows)
    order = [rows.index(r) for r in rs]
    Mo_s = Mo[np.ix_(order, order)]
    signs_s = signs[order]
    rows_s = [rows[i] for i in order]
    j = next(
        i for i, r in enumerate(rows_s) if r.sigma == 0.8333 and r.route == "768"
    )
    lo = next(
        i for i, r in enumerate(rows_s) if r.sigma == 0.7 and r.route == "768"
    )
    hi = next(
        i for i, r in enumerate(rows_s) if r.sigma == 0.9625 and r.route == "768"
    )
    th = 0.3 + 1.2 * math.log(0.8333)
    tan_true = -math.sin(th) * e1 + math.cos(th) * e2
    for planted, want_hi in ((tan_true, True), (e3, False)):
        vp1 = planted + 0.1 * rng.normal(size=dim) / math.sqrt(dim)
        vp2 = planted + 0.1 * rng.normal(size=dim) / math.sqrt(dim)
        vp_pool, nvp2 = 0.5 * (vp1 + vp2), float(vp1 @ vp2)
        vc = np.empty(len(rows_s))
        for k, r in enumerate(rows_s):
            i0 = order[k]
            pool_k = 0.5 * (pairs[i0][0] + pairs[i0][1])
            den_k = math.sqrt(float(pairs[i0][0] @ pairs[i0][1]))
            vc[k] = signs_s[k] * float(vp_pool @ pool_k) / (
                math.sqrt(nvp2) * den_k
            )
        t_hat = tangent_coeff(rows_s, Mo_s, j, lo, hi)
        got = abs(float(t_hat @ vc))
        if want_hi:
            assert got > 0.9, got
        else:
            assert got < 0.3, got
    print("[validate] V_phase: planted tangent recovered; orthogonal control flat")


# ------------------------------------------------------------------ main


def figures(digest: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, leg in zip(axes, ("B", "C", "R")):
        d = digest["loo"].get(leg)
        if not d or not d.get("folds"):
            ax.set_axis_off()
            continue
        rows = [t for f in d["folds"] if "targets" in f for t in f["targets"]]
        x = np.arange(len(rows))
        ax.plot(x, [abs(r[f"cos_{PRIMARY}"]) for r in rows], "o-", label="law (L-log)")
        ax.plot(x, [abs(r["cos_slerp"]) for r in rows], "s-", label="SLERP (b)")
        ax.plot(x, [abs(r["cos_copy"]) for r in rows], "^:", label="copy (a)")
        ax.plot(
            x,
            [r["plane_share"] for r in rows],
            "x--",
            color="0.5",
            label="plane share",
        )
        ax.set_xticks(x, [r["cond"] for r in rows], rotation=90, fontsize=6)
        ax.set_ylim(0, 1.15)
        ax.set_title(
            f"{leg}: {d['verdict_27_2']} (Δ̃ {d['delta_primary_median']}); "
            f"{d['verdict_27_1']}"
        )
        ax.legend(fontsize=7)
    fig.suptitle("E27 LOO direction cosines at held-out σ (debiased, |cos| clamped in view)")
    fig.tight_layout()
    fig.savefig(HERE / "e27_loo.png", dpi=150)
    print("[fig] e27_loo.png")

    a = digest["anchor"]
    rows = [r for r in a["rows"] if r.get("readable")]
    if rows:
        fig2, ax = plt.subplots(figsize=(7, 4.5))
        x = np.arange(len(rows))
        ax.bar(x - 0.2, [abs(r["cos_tangent"]) for r in rows], 0.4, label="|cos(V̂p, t̂_C)|")
        ax.bar(x + 0.2, [abs(r["cos_axis"]) for r in rows], 0.4, label="|cos(V̂p, Ĉ)|")
        ax.plot(x, [r["plane_share_Vp"] for r in rows], "kx--", label="V̂p plane share")
        ax.axhline(0.5, color="k", lw=0.8, ls="--")
        ax.set_xticks(x, [r["cond"] for r in rows], rotation=45, fontsize=8)
        ax.set_title(f"E27.4 phase generator anchor — {a['verdict_27_4']}")
        ax.legend(fontsize=8)
        fig2.tight_layout()
        fig2.savefig(HERE / "e27_phase.png", dpi=150)
        print("[fig] e27_phase.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--figs_only", action="store_true")
    args = ap.parse_args()

    if args.figs_only:
        figures(json.loads((HERE / "e27_rotlaw.json").read_text()))
        return

    validate_synthetic()  # E24 gate: build_cond == bc_ledger
    validate_geodesic()  # E27 gates: planted law + non-planar control
    validate_vphase_tangent()
    if args.validate:
        print("[validate] synthetic gates passed — stopping (--validate)")
        return

    t0 = time.time()
    conds: list[Cond] = []
    rext: dict[str, dict] = {}
    vext: dict[str, dict] = {}
    for store, root in STORES.items():
        s = e24.Sums(root)
        has_pi = any(k.endswith("pi") for k in s.man["keys"])
        ghat_shared: dict[int, np.ndarray] = {}
        for bi in range(s.n_bins):
            for route in _routes(s.man):
                c = build_cond(s, store, route, bi)
                if bi in ghat_shared:
                    c.ghat32 = ghat_shared[bi]
                else:
                    ghat_shared[bi] = c.ghat32
                conds.append(c)
                if store in ("e193", "e194"):
                    rext[cond_id(c)] = residual_extras(s, route, bi, c)
                msg = (
                    f"[cond] {cond_id(c):22s} relB={c.rel_B:.2f} "
                    f"relC={c.rel_C:.2f}"
                )
                if has_pi:
                    v = vphase_extras(s, route, bi, c)
                    assert v["identity_rel_dev"] < 1e-9, (
                        cond_id(c),
                        v["identity_rel_dev"],
                    )
                    vext[cond_id(c)] = v
                    msg += (
                        f" relVp={v['rel_cos_Vp']:+.2f}"
                        f"{' (Vp ok)' if v['readable'] else ' (Vp FAIL)'}"
                    )
                print(msg)
            s.drop_bin(bi)
        del s

    validate_e221(conds)
    max_idev = max(r["identity_dev"] for r in rext.values())
    assert max_idev <= 1e-6, f"nR2 identity violated: {max_idev}"
    print(f"[validate] nR2 identity max dev {max_idev:.2e}; V_phase linearity ok")

    std = [c for c in conds if c.store in ("e193", "e194")]
    vecs: list[np.ndarray] = []
    for c in std:
        c.iB = len(vecs)
        vecs.append(c.B32)
        c.iC = len(vecs)
        vecs.append(c.C32)
    vp_col: dict[str, int] = {}
    for cid, v in vext.items():
        vp_col[cid] = len(vecs)
        vecs.append(v.pop("Vp32"))
    print(f"[gram] {len(vecs)} vectors x {vecs[0].shape[0]} dims ...")
    e24.GRAM = build_gram(vecs)
    print(f"[gram] done ({time.time() - t0:.0f}s elapsed)")

    def vp_gram_col(cid: str, cond: Cond) -> float:
        return float(e24.GRAM[vp_col[cid], cond.iC])

    grid = sorted({round(c.sigma, 4) for c in std})
    loo: dict[str, dict] = {}
    anchor = None
    for leg in ("B", "C", "R"):
        if leg == "R":
            sel = [c for c in std if rext[cond_id(c)]["pass1"]]
            pairfn = lambda x, y: r_cross_cos(x, y, rext)  # noqa: E731
        else:
            sel = [c for c in std if gated(c, leg)]
            pairfn = lambda x, y, _l=leg: cross_cos(x, y, _l)  # noqa: E731
        rows = sort_rows([Row(c) for c in sel])
        M = unit_gram(rows, pairfn)
        signs, Mo = orient(rows, M)
        loo[leg] = loo_read(rows, Mo)
        print(
            f"[loo:{leg}] {loo[leg]['verdict_27_2']} "
            f"(Δ̃ {loo[leg]['delta_primary_median']}, "
            f"wins {loo[leg]['delta_primary_win_share']}); "
            f"{loo[leg]['verdict_27_1']} "
            f"(share med {loo[leg]['plane_share_median']})"
        )
        if leg == "C":
            anchor = anchor_read(rows, Mo, signs, vext, vp_gram_col, grid)
            print(f"[anchor] {anchor['verdict_27_4']}")

    digest = {
        "generated_by": "e27_rotlaw.py",
        "rel_gate": REL_GATE,
        "delta_margin": DELTA_MARGIN,
        "primary_variant": PRIMARY,
        "target_sigma": list(TARGET_SIGMA),
        "conditions": [
            {
                "cond": cond_id(c),
                "sigma": c.sigma,
                "rel_cos_B": round(c.rel_B, 4),
                "rel_cos_C": round(c.rel_C, 4),
                "rel_cos_R": round(rext[cond_id(c)]["rel_cos_R"], 4),
                "pass_R": rext[cond_id(c)]["pass1"],
                **(
                    {
                        "rel_cos_Vp": round(vext[cond_id(c)]["rel_cos_Vp"], 4),
                        "readable_Vp": vext[cond_id(c)]["readable"],
                    }
                    if cond_id(c) in vext
                    else {}
                ),
            }
            for c in std
        ],
        "loo": loo,
        "anchor": anchor,
        "verdicts": {
            "27.1": loo["B"]["verdict_27_1"],
            "27.2": loo["B"]["verdict_27_2"],
            "27.3_R": loo["R"]["verdict_27_2"],
            "27.4": anchor["verdict_27_4"],
        },
    }

    def _np_default(o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(f"not serializable: {type(o)}")

    (HERE / "e27_rotlaw.json").write_text(
        json.dumps(digest, indent=1, default=_np_default)
    )
    print(
        f"[done] 27.1 {digest['verdicts']['27.1']}  27.2 "
        f"{digest['verdicts']['27.2']}  27.3(R) {digest['verdicts']['27.3_R']}  "
        f"27.4 {digest['verdicts']['27.4']}  ({time.time() - t0:.0f}s)"
    )
    figures(digest)


if __name__ == "__main__":
    main()
