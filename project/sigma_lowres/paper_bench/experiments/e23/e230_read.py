#!/usr/bin/env python
"""sigma_lowres E23.0 — one-step counterfactual lever read (CPU).

Pre-registered in README.md (committed before this file existed). Two
implementable demoted-step levers simulated exactly, pooled, from the
four committed arm_sums stores:

  lever 1 (E23a form): adaln-band damping  M(lambda) . dem
  lever 2 (E25a form): residual projection (I - u u^T) . dem

Everything is algebra on per-band Gram entries: one streaming fp64 pass
computes a (10 core-type bands) x N x N Gram over the loaded arm
vectors; every derived vector (band-scaled dem, perp'd residual
directions, projections, spans) is a per-band coefficient combination,
so each read is exact small-matrix arithmetic.

Estimand: debiased gap to native  d = 1 - num/den with the numerator
symmetrized over independent draw sets and both norms cross-set
debiased; closure = (d_dem - d_lever)/(d_dem - d_reenc). Readings
23.0-A/B/C + the Stage-2 map are frozen in README.md.

Usage::

    uv run python project/sigma_lowres/paper_bench/experiments/e23/e230_read.py
    ... --validate     # synthetic gates only, no store access
    ... --figs_only    # redraw figures from the committed record
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
PB = REPO_ROOT / "project/sigma_lowres/paper_bench"

# ---------------------------------------------------------------- frozen
STORES = {
    "e193": {
        "root": REPO_ROOT
        / "project/sigma_lowres/bench/results/20260807-0745-e193-depth-ledger",
        "sigmas": [0.3, 0.4333, 0.5667, 0.7],
        "verdict": True,
        "rp": False,
    },
    "e194": {
        "root": REPO_ROOT
        / "project/sigma_lowres/bench/results/20260807-1400-e194-pi-causal",
        "sigmas": [0.7, 0.8333],
        "verdict": True,
        "rp": False,
    },
    "e221": {
        "root": PB / "runs/20260808-1633-e221-per-image-ledger",
        "sigmas": [0.4333, 0.5667, 0.7],
        "verdict": False,
        "rp": True,  # loaded for validation gate 1
    },
    "e224": {
        "root": PB / "runs/20260809-0031-e224-per-image-d96",
        "sigmas": [0.7],
        "verdict": False,
        "rp": True,
    },
}
CORE_TYPES = [  # e21's verified disjoint cover, band order fixed
    "adaln_up_self_attn",
    "adaln_up_cross_attn",
    "adaln_up_mlp",
    "self_attn_qkv_proj",
    "self_attn_output_proj",
    "cross_attn_q_proj",
    "cross_attn_kv_proj",
    "cross_attn_output_proj",
    "mlp_layer1",
    "mlp_layer2",
]
BANDS = {
    "adaln": (0, 1, 2),
    "self_attn": (3, 4),
    "cross_attn": (5, 6, 7),
    "mlp": (8, 9),
}
NB = len(CORE_TYPES)
WINDOW = (0.5667, 0.7, 0.8333)  # shipped sigma>0.5 demote gate, probe bins
LAM_FINE = np.round(np.arange(0.0, 1.0001, 0.01), 2)
LAM_SHOW = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
EXCESS_ABS, EXCESS_NSIG = 0.005, 3.0
REL_GATE = 0.5
CHUNK = 4_000_000


# ------------------------------------------------------- Gram machinery


class Ctx:
    """Per-band Grams over a vector registry; all reads happen here."""

    def __init__(self, grams: np.ndarray) -> None:
        self.G = grams  # (NB, N, N) fp64
        self.full = grams.sum(axis=0)

    def dot(self, u: "LV", v: "LV") -> float:
        return float(np.einsum("bi,bij,bj->", u.C, self.G, v.C))


@dataclass
class LV:
    """A vector represented by per-band coefficients over the registry."""

    C: np.ndarray  # (NB, N)

    @classmethod
    def unit(cls, n: int, i: int) -> "LV":
        C = np.zeros((NB, n))
        C[:, i] = 1.0
        return cls(C)

    def __add__(self, o: "LV") -> "LV":
        return LV(self.C + o.C)

    def __sub__(self, o: "LV") -> "LV":
        return LV(self.C - o.C)

    def __mul__(self, s: float) -> "LV":
        return LV(self.C * s)

    __rmul__ = __mul__

    def band_scale(self, scales: np.ndarray) -> "LV":
        return LV(self.C * scales[:, None])


def norm(ctx: Ctx, v: LV) -> float:
    return math.sqrt(max(ctx.dot(v, v), 0.0))


def unit(ctx: Ctx, v: LV) -> LV:
    return v * (1.0 / norm(ctx, v))


def cosine(ctx: Ctx, u: LV, v: LV) -> float:
    nu, nv = norm(ctx, u), norm(ctx, v)
    if nu == 0.0 or nv == 0.0:
        return float("nan")
    return ctx.dot(u, v) / (nu * nv)


def perp(ctx: Ctx, v: LV, ghat: LV) -> LV:
    return v - ctx.dot(ghat, v) * ghat


def project_out(ctx: Ctx, v: LV, dirs: list[LV]) -> LV:
    """v minus its component in span(dirs) (small-Gram solve)."""
    if not dirs:
        return v
    M = np.array([[ctx.dot(a, b) for b in dirs] for a in dirs])
    r = np.array([ctx.dot(a, v) for a in dirs])
    coef = np.linalg.solve(M, r)
    out = v
    for c, d in zip(coef, dirs):
        out = out - float(c) * d
    return out


# --------------------------------------------------------------- loading


def band_segments(groups: dict) -> tuple[list[tuple[int, int, int]], int]:
    segs: list[tuple[int, int, int]] = []
    for bi, t in enumerate(CORE_TYPES):
        for s, e in groups[f"type:{t}"]:
            while s < e:
                le = min(s + CHUNK, e)
                segs.append((s, le, bi))
                s = le
    segs.sort()
    total = 0
    prev = 0
    for s, e, _ in segs:
        assert s == prev, f"core cover has a hole/overlap at {s} (prev {prev})"
        prev = e
        total += e - s
    return segs, total


@dataclass
class Cond:
    store: str
    route: str
    sigma: float
    verdict: bool
    idx: dict[str, int]  # arm key -> registry index
    rel_B: float = float("nan")
    rel_C: float = float("nan")
    read1_pass: bool = False
    # filled after Gram
    g0: LV = None
    ghat: LV = None
    G: float = 0.0
    nab: float = 0.0
    d_dem: float = 0.0
    d_reenc: float = 0.0
    excess: float = 0.0
    sig_twin: float = 0.0
    excess_pass: bool = False
    r_set: list[LV] = field(default_factory=list)  # set-pure unit dirs
    r_mean: LV = None  # condition-mean unit dir
    rel_R: float = float("nan")

    def cid(self) -> str:
        return f"{self.store}/{self.route}/s{self.sigma:g}"


def load_registry() -> tuple[list[tuple[Path, float]], list[Cond], list, int]:
    reg: list[tuple[Path, float]] = []
    conds: list[Cond] = []
    groups_ref = None
    segs = total = None
    for store, cfg in STORES.items():
        man = json.loads((cfg["root"] / "arm_sums/manifest.json").read_text())
        groups = json.loads((cfg["root"] / "arm_sums/groups.json").read_text())
        tsel = {f"type:{t}": groups[f"type:{t}"] for t in CORE_TYPES}
        if groups_ref is None:
            groups_ref = tsel
            segs, total = band_segments(groups)
        else:
            assert tsel == groups_ref, f"{store}: type ranges differ from e193"
        scale = 1.0 / (man["n_images"] * man["draws_per_bin"])
        routes = [e for e in str(man["demote_edges"]).split(",") if e]
        centers = man["sigma_centers"]

        def add(key: str, bi: int) -> int:
            fname = f"{key}__b{bi}.npy"
            reg.append((cfg["root"] / "arm_sums" / fname, scale))
            return len(reg) - 1

        for sig in cfg["sigmas"]:
            bi = centers.index(sig)
            shared = {k: add(k, bi) for k in ("a", "b", "reenc", "reenc__2")}
            for route in routes:
                idx = dict(shared)
                idx["dem1"], idx["dem2"] = add(route, bi), add(f"{route}__2", bi)
                if cfg["rp"]:
                    idx["rp1"] = add(f"{route}rp", bi)
                    idx["rp2"] = add(f"{route}rp__2", bi)
                conds.append(Cond(store, route, sig, cfg["verdict"], idx))
    return reg, conds, segs, total


def build_grams(reg: list[tuple[Path, float]], segs: list) -> np.ndarray:
    n = len(reg)
    mms = [np.load(p, mmap_mode="r") for p, _ in reg]
    scales = np.array([s for _, s in reg])
    G = np.zeros((NB, n, n))
    t0 = time.time()
    done = 0
    total = sum(e - s for s, e, _ in segs)
    for k, (s, e, b) in enumerate(segs):
        X = np.empty((n, e - s))
        for i, mm in enumerate(mms):
            X[i] = mm[s:e]
        G[b] += X @ X.T
        done += e - s
        if k % 25 == 0 or done == total:
            print(
                f"[gram] {done / total * 100:5.1f}%  ({time.time() - t0:.0f}s)",
                flush=True,
            )
    G *= scales[None, :, None] * scales[None, None, :]
    return G


# --------------------------------------------------------- core estimands


def d_gap(ctx: Ctx, c: Cond, x1: LV, x2: LV) -> tuple[float, float]:
    """Debiased gap of the (x1, x2) half-set pair vs native + twin sigma."""
    n1, n2 = ctx.dot(x1, c.g0), ctx.dot(x2, c.g0)
    den = math.sqrt(max(ctx.dot(x1, x2) * c.nab, 0.0))
    d = 1.0 - 0.5 * (n1 + n2) / den
    return d, abs((1.0 - n1 / den) - (1.0 - n2 / den)) / 2.0


def finish_cond(ctx: Ctx, c: Cond, n: int) -> None:
    v = {k: LV.unit(n, i) for k, i in c.idx.items()}
    c.g0 = 0.5 * (v["a"] + v["b"])
    c.G = norm(ctx, c.g0)
    c.ghat = c.g0 * (1.0 / c.G)
    c.nab = ctx.dot(v["a"], v["b"])
    dem1, dem2 = v["dem1"], v["dem2"]
    re1, re2 = v["reenc"], v["reenc__2"]
    c.d_dem, s_dem = d_gap(ctx, c, dem1, dem2)
    c.d_reenc, s_re = d_gap(ctx, c, re1, re2)
    c.excess = c.d_dem - c.d_reenc
    c.sig_twin = math.sqrt(s_dem**2 + s_re**2)
    c.excess_pass = c.excess >= max(EXCESS_ABS, EXCESS_NSIG * c.sig_twin)
    # residual directions
    c.r_set = [
        unit(ctx, perp(ctx, dem1 - re1, c.ghat)),
        unit(ctx, perp(ctx, dem2 - re2, c.ghat)),
    ]
    rem = 0.5 * (re1 + re2)
    c.r_mean = unit(ctx, perp(ctx, 0.5 * (dem1 + dem2) - rem, c.ghat))
    # e250-convention rel_cos_R (validation gate 3)
    R1, R2 = perp(ctx, dem1 - rem, c.ghat), perp(ctx, dem2 - rem, c.ghat)
    c.rel_R = cosine(ctx, R1, R2)


def damp_curve(ctx: Ctx, c: Cond, n: int, band: str = "adaln") -> dict:
    dem1 = LV.unit(n, c.idx["dem1"])
    dem2 = LV.unit(n, c.idx["dem2"])
    rows = []
    num1 = None
    for lam in LAM_FINE:
        s = np.ones(NB)
        for bi in BANDS[band]:
            s[bi] = lam
        m1, m2 = dem1.band_scale(s), dem2.band_scale(s)
        num = 0.5 * (ctx.dot(m1, c.g0) + ctx.dot(m2, c.g0))
        den = math.sqrt(max(ctx.dot(m1, m2) * c.nab, 0.0))
        d = 1.0 - num / den
        rows.append((float(lam), d, num))
    num1 = rows[-1][2]  # lambda = 1
    assert abs(rows[-1][1] - c.d_dem) < 1e-12, "lambda=1 identity gate"
    best = min(rows, key=lambda r: r[1])
    exc = c.excess if c.excess != 0 else float("nan")
    return {
        "lambda_star": best[0],
        "d_star": best[1],
        "closure_star": (c.d_dem - best[1]) / exc,
        "SR_star": best[2] / num1,
        "curve": {
            f"{lam:g}": {
                "d": round(d, 6),
                "closure": round((c.d_dem - d) / exc, 4),
                "SR": round(num / num1, 5),
            }
            for lam, d, num in rows
            if lam in LAM_SHOW
        },
        "curve_fine": [[lam, (c.d_dem - d) / exc, num / num1] for lam, d, num in rows],
    }


def proj_read(ctx: Ctx, c: Cond, n: int, dirs1: list[LV], dirs2: list[LV]) -> dict:
    """Projected debiased gap; dirs1 applied to dem1, dirs2 to dem2
    (identical fixed lists for transfers, swapped set-pure for own-bin)."""
    dem1 = LV.unit(n, c.idx["dem1"])
    dem2 = LV.unit(n, c.idx["dem2"])
    p1 = project_out(ctx, dem1, dirs1)
    p2 = project_out(ctx, dem2, dirs2)
    n1, n2 = ctx.dot(p1, c.g0), ctx.dot(p2, c.g0)
    den = math.sqrt(max(ctx.dot(p1, p2) * c.nab, 0.0))
    d = 1.0 - 0.5 * (n1 + n2) / den
    num0 = 0.5 * (ctx.dot(dem1, c.g0) + ctx.dot(dem2, c.g0))
    exc = c.excess if c.excess != 0 else float("nan")
    return {
        "d": round(d, 6),
        "closure": round((c.d_dem - d) / exc, 4),
        "SR": round(0.5 * (n1 + n2) / num0, 5),
    }


# ------------------------------------------------------------ validation


def validate_gate1(ctx: Ctx, conds: list[Cond], n: int) -> dict:
    """Reproduce the committed e221/e224 ledger.json exactly (round-5)."""
    out = {}
    for store in ("e221", "e224"):
        led = json.loads((STORES[store]["root"] / "ledger.json").read_text())
        worst: dict[str, float] = {}
        n_rows = 0
        for c in [c for c in conds if c.store == store]:
            bins = [x.sigma for x in conds if x.store == store]
            bi = bins.index(c.sigma)
            ref_row = led["bc_ledger"][c.route][bi]
            assert ref_row["data_ref"] == "reenc"
            v = {k: LV.unit(n, i) for k, i in c.idx.items()}
            rem = 0.5 * (v["reenc"] + v["reenc__2"])
            rd = v["reenc"] - v["reenc__2"]
            B1, B2 = v["rp1"] - rem, v["rp2"] - rem
            C1, C2 = v["dem1"] - v["rp1"], v["dem2"] - v["rp2"]
            B1p, B2p = perp(ctx, B1, c.ghat), perp(ctx, B2, c.ghat)
            C1p, C2p = perp(ctx, C1, c.ghat), perp(ctx, C2, c.ghat)
            Bp, Cp = 0.5 * (B1p + B2p), 0.5 * (C1p + C2p)
            Bm, Cm = 0.5 * (B1 + B2), 0.5 * (C1 + C2)
            G2 = c.G * c.G
            ref_noise = norm(ctx, perp(ctx, rd, c.ghat)) ** 2 / 4.0
            S = (ctx.dot(B1p, B2p) - ref_noise) / (2 * G2)
            F = ctx.dot(C1p, C2p) / (2 * G2)
            I_x = (ctx.dot(B1p, C2p) + ctx.dot(B2p, C1p)) / (2 * G2)
            I_same = (ctx.dot(B1p, C1p) + ctx.dot(B2p, C2p)) / (2 * G2)
            denom = 2.0 * math.sqrt(max(S * F, 0.0))
            mine = {
                "G": round(c.G, 4),
                "b_perp_rawnorm": round(norm(ctx, Bp) / c.G, 5),
                "c_perp_rawnorm": round(norm(ctx, Cp) / c.G, 5),
                "kappa_par_B": round(ctx.dot(c.ghat, Bm) / c.G, 5),
                "kappa_par_C": round(ctx.dot(c.ghat, Cm) / c.G, 5),
                "rel_cos_B": round(cosine(ctx, B1p, B2p), 5),
                "rel_cos_C": round(cosine(ctx, C1p, C2p), 5),
                "S": round(S, 5),
                "F": round(F, 5),
                "I": round(I_x, 5),
                "I_sameset_biascheck": round(I_same, 5),
                "rho": round(I_x / denom, 5),
                "quad_pred_gap": round(S + F + I_x, 5),
                "h_B": round(1.0 - cosine(ctx, c.g0, c.g0 + Bm), 5),
                "h_C": round(1.0 - cosine(ctx, c.g0, c.g0 + Cm), 5),
                "h_B_plus_C": round(1.0 - cosine(ctx, c.g0, c.g0 + Bm + Cm), 5),
            }
            for k, mv in mine.items():
                dv = abs(mv - ref_row[k])
                worst[k] = max(worst.get(k, 0.0), dv)
                assert dv == 0.0, f"gate1 {store} {c.cid()} {k}: {mv} != {ref_row[k]}"
            n_rows += 1
        out[store] = {"rows": n_rows, "max_abs_diff": max(worst.values())}
        print(f"[gate1] {store}: {n_rows} rows reproduced exactly")
    return out


def validate_gate2(ctx: Ctx, reg: list, conds: list[Cond]) -> float:
    """Band Grams resum to direct full-vector dots (spot check per store)."""
    worst = 0.0
    for store in STORES:
        c = next(c for c in conds if c.store == store)
        for k1, k2 in (("a", "b"), ("dem1", "dem2"), ("a", "dem1")):
            i, j = c.idx[k1], c.idx[k2]
            (p1, s1), (p2, s2) = reg[i], reg[j]
            v1, v2 = np.load(p1, mmap_mode="r"), np.load(p2, mmap_mode="r")
            direct = 0.0
            for s in range(0, v1.shape[0], CHUNK):
                e = min(s + CHUNK, v1.shape[0])
                direct += float(
                    np.asarray(v1[s:e], dtype=np.float64)
                    @ np.asarray(v2[s:e], dtype=np.float64)
                )
            direct *= s1 * s2
            dev = abs(ctx.full[i, j] - direct) / max(abs(direct), 1e-30)
            worst = max(worst, dev)
    assert worst <= 1e-10, f"gate2 partition dev {worst}"
    print(f"[gate2] band-Gram resum vs direct dots: max rel dev {worst:.2e}")
    return worst


def validate_gate3(conds: list[Cond]) -> float:
    committed = {
        r["cond"]: r["rel_cos_R"]
        for r in json.loads((PB / "experiments/e25/e250_read.json").read_text())[
            "read1"
        ]["conditions"]
    }
    worst = 0.0
    n = 0
    for c in conds:
        if c.cid() not in committed:
            continue
        dev = abs(c.rel_R - committed[c.cid()])
        worst = max(worst, dev)
        n += 1
    assert n >= 15 and worst <= 1e-4, f"gate3: n={n} dev={worst}"
    print(
        f"[gate3] rel_cos_R vs committed e250_read.json: {n} conds, max dev {worst:.1e}"
    )
    return worst


def validate_synthetic() -> None:
    """Gate 4: analytic lambda*, projection closures, debias control."""
    rng = np.random.default_rng(11)
    dpb = 4000  # dims per band
    n_reg = 8

    def mk_ctx(vecs: dict[str, np.ndarray]) -> tuple[Ctx, dict[str, int]]:
        keys = list(vecs)
        V = np.stack([vecs[k] for k in keys])  # (n, NB*dpb)
        G = np.zeros((NB, len(keys), len(keys)))
        for b in range(NB):
            X = V[:, b * dpb : (b + 1) * dpb]
            G[b] = X @ X.T
        return Ctx(G), {k: i for i, k in enumerate(keys)}

    def basis(band: int, seed_vec: np.ndarray | None = None) -> np.ndarray:
        v = np.zeros(NB * dpb)
        r = rng.normal(size=dpb)
        if seed_vec is not None:
            sl = seed_vec[band * dpb : (band + 1) * dpb]
            r -= (r @ sl) / max(sl @ sl, 1e-30) * sl
        v[band * dpb : (band + 1) * dpb] = r / np.linalg.norm(r)
        return v

    # signal: adaln component ga (band 0) + outside component go (band 8)
    ga, go = basis(0), basis(8)
    A, Ob = 1.0, 1.0
    g0 = math.sqrt(A) * ga + math.sqrt(Ob) * go
    weps = 0.35
    w = weps * basis(0, seed_vec=ga)  # adaln-borne excess, perp to g0

    def cond_of(ctx, idx, verdict=True):
        c = Cond("syn", "768", 0.7, verdict, idx)
        finish_cond(ctx, c, len(idx))
        return c

    # (a) analytic lambda* = A / (A + |w|^2); control lambda* = 1
    lam_true = A / (A + weps**2)
    for w_used, expect in ((w, lam_true), (weps * basis(8, seed_vec=go), 1.0)):
        dem = g0 + w_used
        ctx, idx = mk_ctx(
            {"a": g0, "b": g0, "reenc": g0, "reenc__2": g0, "dem1": dem, "dem2": dem}
        )
        c = cond_of(ctx, idx)
        dc = damp_curve(ctx, c, len(idx))
        assert abs(dc["lambda_star"] - expect) <= 0.011, (dc["lambda_star"], expect)
    print(f"[gate4a] lambda* analytic {lam_true:.3f} + control 1.0 recovered")

    # (b) projection: own-bin closure ~ 1; transfer at 60 deg ~ cos^2 = 0.25
    dem = g0 + w
    ctx, idx = mk_ctx(
        {"a": g0, "b": g0, "reenc": g0, "reenc__2": g0, "dem1": dem, "dem2": dem}
    )
    c = cond_of(ctx, idx)
    own = proj_read(ctx, c, len(idx), [c.r_set[1]], [c.r_set[0]])
    assert abs(own["closure"] - 1.0) <= 1e-6, own
    q = basis(0, seed_vec=ga)
    q -= (q @ (w / np.linalg.norm(w))) * (w / np.linalg.norm(w))
    q /= np.linalg.norm(q)
    phi = math.radians(60.0)
    u_vec = math.cos(phi) * w / np.linalg.norm(w) + math.sin(phi) * q
    ctx2, idx2 = mk_ctx(
        {
            "a": g0,
            "b": g0,
            "reenc": g0,
            "reenc__2": g0,
            "dem1": dem,
            "dem2": dem,
            "u": u_vec,
        }
    )
    c2 = cond_of(ctx2, idx2)
    u_lv = LV.unit(len(idx2), idx2["u"])
    tr = proj_read(ctx2, c2, len(idx2), [u_lv], [u_lv])
    assert abs(tr["closure"] - math.cos(phi) ** 2) <= 0.01, tr
    # empty projector identity (proj_read rounds outputs to 6 decimals;
    # the unrounded computation is the identical expression)
    ident = proj_read(ctx2, c2, len(idx2), [], [])
    assert ident["d"] == round(c2.d_dem, 6), (ident["d"], c2.d_dem)
    print("[gate4b] projection: own-bin closure 1.0, 60deg transfer 0.25, identity ok")

    # (c) debias negative control: noise-only adaln excess
    noise = 1.0
    n1 = noise * basis(0, seed_vec=ga)
    n2 = noise * basis(0, seed_vec=ga)
    a_v = g0 + 0.02 * basis(8)
    b_v = g0 - 0.02 * basis(8)
    ctx3, idx3 = mk_ctx(
        {
            "a": a_v,
            "b": b_v,
            "reenc": g0,
            "reenc__2": g0,
            "dem1": g0 + n1,
            "dem2": g0 + n2,
        }
    )
    c3 = cond_of(ctx3, idx3)
    deb = damp_curve(ctx3, c3, len(idx3))
    dem1 = LV.unit(len(idx3), idx3["dem1"])
    dem2 = LV.unit(len(idx3), idx3["dem2"])
    raw_rows = []
    for lam in LAM_FINE:
        s = np.ones(NB)
        for bi in BANDS["adaln"]:
            s[bi] = lam
        mm = (0.5 * (dem1 + dem2)).band_scale(s)
        raw_rows.append((float(lam), 1.0 - cosine(ctx3, c3.g0, mm)))
    raw_star = min(raw_rows, key=lambda r: r[1])[0]
    assert raw_star <= 0.9, raw_star  # raw sweep chases the noise
    assert deb["lambda_star"] >= 0.95, deb["lambda_star"]  # debiased does not
    print(
        f"[gate4c] debias control: raw lambda* {raw_star:.2f} (noise-chasing), "
        f"debiased {deb['lambda_star']:.2f}"
    )
    _ = n_reg


# ---------------------------------------------------------------- reads


def median(xs: list[float]) -> float:
    return float(np.median(xs)) if xs else float("nan")


def read_A(ctx: Ctx, conds: list[Cond], n: int) -> dict:
    rows = []
    for c in conds:
        dc = damp_curve(ctx, c, n)
        rows.append(
            {
                "cond": c.cid(),
                "verdict_eligible": c.verdict,
                "excess_pass": c.excess_pass,
                "lambda_star": dc["lambda_star"],
                "closure_star": round(dc["closure_star"], 4),
                "SR_star": round(dc["SR_star"], 5),
                "curve": dc["curve"],
                "curve_fine": dc["curve_fine"],
            }
        )
    sel = [r for r in rows if r["verdict_eligible"] and r["excess_pass"]]
    med_cl = median([r["closure_star"] for r in sel])
    med_sr = median([r["SR_star"] for r in sel])
    med_ls = median([r["lambda_star"] for r in sel])
    frac_lt1 = (
        float(np.mean([r["lambda_star"] <= 0.99 for r in sel])) if sel else float("nan")
    )
    # DEAD clause 3: every lambda whose median closure reaches 0.25 has
    # median SR < 0.9 (evaluated on the fine curves)
    reach_sr: list[float] = []
    for li in range(len(LAM_FINE)):
        cls = [r["curve_fine"][li][1] for r in sel]
        if sel and median(cls) >= 0.25:
            reach_sr.append(median([r["curve_fine"][li][2] for r in sel]))
    if math.isnan(med_cl):
        verdict = "DAMP-PARTIAL (no eligible conditions)"
    elif med_cl >= 0.25 and med_sr >= 0.95 and frac_lt1 >= 0.75:
        verdict = "DAMP-VIABLE"
    elif med_cl <= 0.05 or med_ls >= 0.99 or (reach_sr and max(reach_sr) < 0.9):
        verdict = "DAMP-DEAD"
    else:
        verdict = "DAMP-PARTIAL"
    for r in rows:
        r.pop("curve_fine")
    return {
        "verdict": verdict,
        "n_eligible": len(sel),
        "median_closure_star": round(med_cl, 4),
        "median_SR_star": round(med_sr, 5) if sel else None,
        "median_lambda_star": med_ls,
        "frac_lambda_le_0p99": round(frac_lt1, 3) if sel else None,
        "rows": rows,
    }


def read_B(ctx: Ctx, conds: list[Cond], n: int) -> dict:
    own_rows, tr_rows = [], []
    for c in conds:
        r = proj_read(ctx, c, n, [c.r_set[1]], [c.r_set[0]])
        own_rows.append(
            {
                "cond": c.cid(),
                "verdict_eligible": c.verdict,
                "excess_pass": c.excess_pass,
                "read1_pass": c.read1_pass,
                **r,
            }
        )
    verd = [c for c in conds if c.verdict]
    for src in verd:
        if not src.read1_pass:
            continue
        for tgt in verd:
            if tgt is src:
                continue
            fam = None
            if src.sigma == tgt.sigma == 0.7 and src.route == tgt.route:
                if src.store != tgt.store:
                    fam = "cross_store"
            if src.store == tgt.store and src.sigma == tgt.sigma:
                if src.route != tgt.route:
                    fam = "cross_route"
            if fam is None:
                continue
            r = proj_read(ctx, tgt, n, [src.r_mean], [src.r_mean])
            tr_rows.append(
                {
                    "family": fam,
                    "src": src.cid(),
                    "tgt": tgt.cid(),
                    "tgt_excess_pass": tgt.excess_pass,
                    **r,
                }
            )
    sel = [r for r in tr_rows if r["tgt_excess_pass"]]
    med_cl = median([r["closure"] for r in sel])
    min_cl = min([r["closure"] for r in sel], default=float("nan"))
    med_sr = median([r["SR"] for r in sel])
    if math.isnan(med_cl):
        verdict = "PROJ-PARTIAL (no eligible transfers)"
    elif med_cl >= 0.5 and min_cl >= 0.25 and med_sr >= 0.98:
        verdict = "PROJ-TRANSFERS"
    elif med_cl < 0.25:
        verdict = "PROJ-WEAK"
    else:
        verdict = "PROJ-PARTIAL"
    return {
        "verdict": verdict,
        "n_transfers": len(sel),
        "median_closure": round(med_cl, 4),
        "min_closure": round(min_cl, 4) if sel else None,
        "median_SR": round(med_sr, 5) if sel else None,
        "own_bin_context": own_rows,
        "transfers": tr_rows,
    }


def read_C(ctx: Ctx, conds: list[Cond], n: int) -> dict:
    verd = [c for c in conds if c.verdict]
    pools = {s: [c for c in verd if c.sigma == s and c.read1_pass] for s in WINDOW}

    def lookup(sig: float, exclude: Cond | None) -> LV | None:
        pool = [c for c in pools[sig] if c is not exclude]
        if not pool:
            return None
        acc = pool[0].r_mean
        for c in pool[1:]:
            acc = acc + c.r_mean
        return unit(ctx, acc)

    rows = []
    targets = [c for c in conds if c.sigma in WINDOW]
    for c in targets:
        excl = c if c.verdict else None
        own_dirs = lookup(c.sigma, excl)
        row = {
            "cond": c.cid(),
            "verdict_eligible": c.verdict,
            "excess_pass": c.excess_pass,
            "own_bin": proj_read(ctx, c, n, [c.r_set[1]], [c.r_set[0]]),
            "lookup_own_sigma": proj_read(ctx, c, n, [own_dirs], [own_dirs])
            if own_dirs
            else None,
        }
        for s2 in WINDOW:
            if s2 == c.sigma:
                continue
            d2 = lookup(s2, excl)
            row[f"lookup_s{s2:g}"] = proj_read(ctx, c, n, [d2], [d2]) if d2 else None
        span = [d for d in (lookup(s, excl) for s in WINDOW) if d is not None]
        row["span"] = proj_read(ctx, c, n, span, span)
        rows.append(row)

    gated_v = [r for r in rows if r["verdict_eligible"]]
    span_ok = all(
        r["span"]["SR"] >= 0.98
        and (
            not r["excess_pass"]
            or r["span"]["closure"] >= 0.8 * r["own_bin"]["closure"]
        )
        for r in gated_v
    )
    per_bin = any(
        r["span"]["SR"] < 0.98
        or (r["excess_pass"] and r["span"]["closure"] < 0.5 * r["own_bin"]["closure"])
        for r in gated_v
    )
    verdict = "SPAN-OK" if span_ok else ("PER-BIN-ONLY" if per_bin else "SHAPE-PARTIAL")

    # leakage matrix: full-pool lookup directions vs each window ghat
    leak = {}
    full_dirs = {s: lookup(s, None) for s in WINDOW}
    for s, d in full_dirs.items():
        if d is None:
            continue
        leak[f"s{s:g}"] = {c.cid(): round(ctx.dot(d, c.ghat), 4) for c in targets}
    return {"verdict": verdict, "rows": rows, "leakage_ghat": leak}


def stage2_map(a: str, b: str, read_a: dict, read_b: dict) -> str:
    a0 = a.split(" ")[0]
    b0 = b.split(" ")[0]
    if a0 == "DAMP-PARTIAL":
        a0 = "DAMP-DEAD"  # frozen conservative default
    if b0 == "PROJ-PARTIAL":
        b0 = "PROJ-WEAK"
    if a0 == "DAMP-DEAD" and b0 == "PROJ-TRANSFERS":
        return "Stage 2 = E25a (projection guard); E23a damping closed at probe level"
    if a0 == "DAMP-VIABLE" and b0 == "PROJ-WEAK":
        return "Stage 2 = E23a (damping bench); E25a not frozen"
    if a0 == "DAMP-VIABLE" and b0 == "PROJ-TRANSFERS":
        win = (
            "E25a"
            if read_b["median_closure"] >= read_a["median_closure_star"]
            else "E23a"
        )
        return f"both viable — higher median closure at matched SR: Stage 2 = {win}"
    return (
        "no gradient-surgery Stage 2 (both dead/weak); E25b remains the only "
        "candidate arm; scheduler-side recipe reconfirmed"
    )


# ---------------------------------------------------------------- figures


def figures(digest: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in digest["read_A"]["rows"] if r["verdict_eligible"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    lam = [float(k) for k in rows[0]["curve"]]
    for r in rows:
        cl = [r["curve"][f"{v:g}"]["closure"] for v in lam]
        sr = [r["curve"][f"{v:g}"]["SR"] for v in lam]
        style = "-" if "/768/" in r["cond"] else "--"
        alpha = 1.0 if r["excess_pass"] else 0.3
        axes[0].plot(lam, cl, style, alpha=alpha, label=r["cond"])
        axes[1].plot(lam, sr, style, alpha=alpha)
    axes[0].axhline(0.25, color="k", ls=":", lw=0.8)
    axes[0].axhline(0.0, color="0.6", lw=0.8)
    axes[0].set_xlabel("lambda (adaln damping)")
    axes[0].set_ylabel("closure of demotion excess")
    axes[0].set_title(f"23.0-A damping — {digest['read_A']['verdict']}")
    axes[0].legend(fontsize=5)
    axes[1].axhline(0.95, color="k", ls=":", lw=0.8)
    axes[1].set_xlabel("lambda")
    axes[1].set_ylabel("signal retention SR")
    axes[1].set_title("retained native-parallel component")
    fig.tight_layout()
    fig.savefig(HERE / "e230_damp.png", dpi=150)
    print("[fig] e230_damp.png")

    tr = digest["read_B"]["transfers"]
    own = {r["cond"]: r for r in digest["read_B"]["own_bin_context"]}
    fig2, ax = plt.subplots(figsize=(max(8, 0.5 * len(tr)), 4.6))
    x = np.arange(len(tr))
    from matplotlib.colors import to_rgba

    cols = [
        to_rgba(
            "tab:blue" if t["family"] == "cross_store" else "tab:green",
            1.0 if t["tgt_excess_pass"] else 0.35,
        )
        for t in tr
    ]
    ax.bar(x, [t["closure"] for t in tr], 0.62, color=cols)
    for xi, t in zip(x, tr):
        o = own.get(t["tgt"])
        if o:
            ax.plot([xi - 0.31, xi + 0.31], [o["closure"]] * 2, "k-", lw=1)
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.axhline(0.25, color="k", ls=":", lw=0.8)
    ax.set_xticks(x, [f"{t['src']}→{t['tgt']}" for t in tr], rotation=90, fontsize=6)
    ax.set_ylabel("transfer closure (black tick = own-bin)")
    ax.set_title(
        f"23.0-B projection transfers — {digest['read_B']['verdict']} "
        "(blue cross-store, green cross-route; faint = excess-filtered)"
    )
    fig2.tight_layout()
    fig2.savefig(HERE / "e230_proj.png", dpi=150)
    print("[fig] e230_proj.png")

    rowsC = digest["read_C"]["rows"]
    keys = ["own_bin", "lookup_own_sigma", "span"]
    fig3, axes3 = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(rowsC))
    w = 0.25
    for ki, k in enumerate(keys):
        vals = [r[k]["closure"] if r[k] else np.nan for r in rowsC]
        srs = [r[k]["SR"] if r[k] else np.nan for r in rowsC]
        axes3[0].bar(x + (ki - 1) * w, vals, w, label=k)
        axes3[1].bar(x + (ki - 1) * w, srs, w, label=k)
    for ax, ylab, hl in ((axes3[0], "closure", None), (axes3[1], "SR", 0.98)):
        ax.set_xticks(x, [r["cond"] for r in rowsC], rotation=90, fontsize=6)
        ax.set_ylabel(ylab)
        if hl:
            ax.axhline(hl, color="k", ls=":", lw=0.8)
            ax.set_ylim(0.9, 1.01)
        ax.legend(fontsize=7)
    axes3[0].set_title(f"23.0-C lookup shape — {digest['read_C']['verdict']}")
    axes3[1].set_title("signal retention")
    fig3.tight_layout()
    fig3.savefig(HERE / "e230_leak.png", dpi=150)
    print("[fig] e230_leak.png")


# ------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--figs_only", action="store_true")
    args = ap.parse_args()

    if args.figs_only:
        figures(json.loads((HERE / "e230_read.json").read_text()))
        return

    validate_synthetic()
    if args.validate:
        print("[validate] synthetic gates passed — stopping (--validate)")
        return

    t0 = time.time()
    reg, conds, segs, total_dims = load_registry()
    print(f"[load] {len(reg)} vectors, {total_dims} dims, {len(segs)} segments")
    ctx = Ctx(build_grams(reg, segs))
    n = len(reg)

    # inherited gates
    e24_rows = {
        r["cond"]: r
        for r in json.loads((PB / "experiments/e24/e24_axis.json").read_text())[
            "conditions"
        ]
    }
    e250_rows = {
        r["cond"]: r
        for r in json.loads((PB / "experiments/e25/e250_read.json").read_text())[
            "read1"
        ]["conditions"]
    }
    for c in conds:
        finish_cond(ctx, c, n)
        cid = c.cid()
        if cid in e24_rows:
            c.rel_B = e24_rows[cid]["rel_cos_B"]
            c.rel_C = e24_rows[cid]["rel_cos_C"]
        if cid in e250_rows:
            c.read1_pass = bool(e250_rows[cid]["pass"])
        elif c.store == "e224":  # not covered by e250 — same rule, descriptive
            c.read1_pass = c.rel_R >= REL_GATE
        print(
            f"[cond] {cid:18s} d_dem={c.d_dem:.5f} d_reenc={c.d_reenc:.5f} "
            f"excess={c.excess:+.5f} (3sig={3 * c.sig_twin:.5f}) "
            f"{'PASS' if c.excess_pass else 'excess-filtered'}"
            f"{'' if c.verdict else ' (descriptive)'}"
        )

    g1 = validate_gate1(ctx, conds, n)
    g2 = validate_gate2(ctx, reg, conds)
    g3 = validate_gate3(conds)

    verdict_conds = [c for c in conds if c.verdict]
    A = read_A(ctx, verdict_conds, n)
    B = read_B(ctx, conds, n)
    C = read_C(ctx, conds, n)
    # descriptive: other-band damping (verdict conds, medians only)
    other_bands = {}
    for band in ("self_attn", "mlp", "cross_attn"):
        cls = []
        for c in verdict_conds:
            if not c.excess_pass:
                continue
            dc = damp_curve(ctx, c, n, band=band)
            cls.append(
                {
                    "cond": c.cid(),
                    "lambda_star": dc["lambda_star"],
                    "closure_star": round(dc["closure_star"], 4),
                    "SR_star": round(dc["SR_star"], 5),
                }
            )
        other_bands[band] = cls
    # descriptive: e221/e224 damping + own-bin rows
    desc_rows = []
    for c in [c for c in conds if not c.verdict]:
        dc = damp_curve(ctx, c, n)
        desc_rows.append(
            {
                "cond": c.cid(),
                "excess_pass": c.excess_pass,
                "lambda_star": dc["lambda_star"],
                "closure_star": round(dc["closure_star"], 4),
                "own_bin": proj_read(ctx, c, n, [c.r_set[1]], [c.r_set[0]]),
            }
        )

    stage2 = stage2_map(A["verdict"], B["verdict"], A, B)
    digest = {
        "generated_by": "e230_read.py",
        "frozen": {
            "window": list(WINDOW),
            "excess_filter": [EXCESS_ABS, EXCESS_NSIG],
            "rel_gate": REL_GATE,
        },
        "validation": {
            "gate1": g1,
            "gate2_max_rel_dev": g2,
            "gate3_max_dev": g3,
            "gate4": "passed (synthetic)",
        },
        "conditions": [
            {
                "cond": c.cid(),
                "verdict_eligible": c.verdict,
                "d_dem": round(c.d_dem, 6),
                "d_reenc": round(c.d_reenc, 6),
                "excess": round(c.excess, 6),
                "sig_twin": round(c.sig_twin, 6),
                "excess_pass": c.excess_pass,
                "rel_B": c.rel_B,
                "rel_C": c.rel_C,
                "rel_R": round(c.rel_R, 4),
                "read1_pass": c.read1_pass,
            }
            for c in conds
        ],
        "read_A": A,
        "read_B": B,
        "read_C": C,
        "descriptive": {"other_band_damping": other_bands, "e221_e224": desc_rows},
        "stage2": stage2,
    }

    def _np_default(o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(f"not serializable: {type(o)}")

    (HERE / "e230_read.json").write_text(
        json.dumps(digest, indent=1, default=_np_default)
    )
    print(
        f"\n[done] 23.0-A {A['verdict']}  23.0-B {B['verdict']}  "
        f"23.0-C {C['verdict']}\n[stage2] {stage2}\n({time.time() - t0:.0f}s)"
    )
    figures(digest)


if __name__ == "__main__":
    main()
