#!/usr/bin/env python
"""sigma_lowres E30.1 read — validation gates + 30-A/30-B/30-C.

Consumes the 20260811-0821-e30-basesketch-768 run (arm_sums + base_sketch
store) per the pre-registered README (frozen 2026-08-10, instrument
amendment 2026-08-11). Every read is within-run; every estimand is
Gram-based (count-sketch preserves inner products in expectation, so the
E24/E26 machinery applies verbatim on sketched vectors via a coefficient
engine over per-family 40x40 condition Grams).

Gates (frozen):
  1. param-frame ledger passes the E26 per-bin criteria (readable:
     rel_B ^ rel_C >= 0.5; pass: I < 0, rho <= -0.5, h(B+C) < min(h_B,
     h_C)) on >= 4/5 bins. Applied as frozen; the sincos twin's own
     columns (published in e26_grid_read.json BEFORE the E30
     registration) are compared alongside — they showed 1/5 passing, so
     the gate's pass condition was never attainable at the sincos
     operating point (registration defect, recorded in the README
     amendment). The gate's diagnostic intent (instrument does not
     perturb the param-frame geometry) is measured as max |delta| vs the
     twin over the shared scalar columns.
  2. sketched-vs-exact cosine over all ledger-used pairs (the 28
     within-bin condition pairs x 5 bins), param family vs the exact
     arm store: max |delta cos| <= 0.02. Plus the streaming-path
     certification: cos(sketch_cpu(exact store vector), in-run param
     sketch) per condition (smoke precedent 1.000006).

Readings:
  30-A  bc_ledger machinery (data_ref=reenc) on the full-base sketch;
        gate-1 criteria per bin; verdict per the frozen table.
  30-B  (i) adaln norm share s = |leg_adaln|^2/|leg_full|^2 per leg per
        bin over 30-A-readable bins — numerator EXACT (fp64 Gram of the
        streamed adaln slice), denominator from the full sketch (the
        registration's "streamed exact norms" exist for the slice only;
        sketch norm error at k=2^18 is ~0.3 %, negligible vs the 0.5
        threshold — recorded deviation). Cross-set debiased raw legs
        (share is a magnitude question; the perp-leg variant is
        reported alongside from sketch Grams).
        (ii) across-sigma rotation profile (R-hat primary, B/C-hat
        alongside, E24/e28 estimand) of the adaln sketch vs the full
        sketch: median matched-pair |delta cos| <= 0.1.
        SUFFICIENT iff median s >= 0.5 (both legs) AND profile
        congruent.
  30-C  descriptive: base vs param rho(sigma); base vs param vs
        committed-twin R-hat profiles (scalar comparison only —
        cross-boot caveat); complement shares; E29 k* classification of
        the base-frame R-hat profile.

Usage::

    uv run python project/sigma_lowres/paper_bench/experiments/e30/e30_read.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from project.sigma_lowres.paper_bench.experiments.e29.e29_cluster import (  # noqa: E402
    BINS as E29_BINS,
    classify,
    to_mat,
)

SL = REPO_ROOT / "project/sigma_lowres"
RUN = SL / "bench/results/20260811-0821-e30-basesketch-768"
STORE = RUN / "arm_sums"
SKETCH = STORE / "base_sketch"
ARMS = ["a", "b", "reenc", "reenc__2", "768", "768__2", "768rp", "768rp__2"]
ROUTE = "768"
REL_GATE = 0.5
GATE2_TOL = 0.02
S_THRESH, PROFILE_TOL = 0.5, 0.1  # 30-B design constants (frozen judgment values)
CHUNK = 4_000_000


# ---------------------------------------------------------------- Gram engine


class Frame:
    """Quadratic-form engine over one family's condition Gram.

    Vectors are coefficient arrays over the 40 (arm, bin) conditions;
    <u, v> = u^T G v. Uniform store scale (sum over 40 images x 12
    draws, identical for every condition) cancels in every ratio read.
    """

    def __init__(self, gram: np.ndarray, conds: list[str]) -> None:
        self.g = gram
        self.ix = {c: i for i, c in enumerate(conds)}
        self.n = len(conds)

    def vec(self, terms: dict[str, float]) -> np.ndarray:
        v = np.zeros(self.n)
        for name, coef in terms.items():
            v[self.ix[name]] += coef
        return v

    def dot(self, u: np.ndarray, v: np.ndarray) -> float:
        return float(u @ self.g @ v)

    def norm(self, u: np.ndarray) -> float:
        return math.sqrt(max(self.dot(u, u), 0.0))

    def cos(self, u: np.ndarray, v: np.ndarray) -> float:
        d = self.norm(u) * self.norm(v)
        return self.dot(u, v) / d if d > 0 else float("nan")

    def pdot(self, u, v, ghat) -> float:
        """<u_perp, v_perp> against unit-coefficient direction ghat."""
        return self.dot(u, v) - self.dot(u, ghat) * self.dot(ghat, v)


def bin_conds(bi: int) -> dict[str, str]:
    return {a: f"{a}__b{bi}" for a in ARMS}


def frame_cond(f: Frame, bi: int) -> dict:
    """bc_ledger(data_ref='reenc') scalars + leg coefficient vectors."""
    c = bin_conds(bi)
    g0 = f.vec({c["a"]: 0.5, c["b"]: 0.5})
    G = f.norm(g0)
    ghat = g0 / G
    ref = f.vec({c["reenc"]: 0.5, c["reenc__2"]: 0.5})
    ref_diff = f.vec({c["reenc"]: 1.0, c["reenc__2"]: -1.0})
    B1 = f.vec({c["768rp"]: 1.0}) - ref
    B2 = f.vec({c["768rp__2"]: 1.0}) - ref
    C1 = f.vec({c["768"]: 1.0, c["768rp"]: -1.0})
    C2 = f.vec({c["768__2"]: 1.0, c["768rp__2"]: -1.0})
    Bm, Cm = 0.5 * (B1 + B2), 0.5 * (C1 + C2)

    G2 = G * G
    pd = lambda u, v: f.pdot(u, v, ghat)  # noqa: E731
    ref_noise = pd(ref_diff, ref_diff) / 4.0
    S = (pd(B1, B2) - ref_noise) / (2 * G2)
    F = pd(C1, C2) / (2 * G2)
    I = (pd(B1, C2) + pd(B2, C1)) / (2 * G2)  # noqa: E741
    denom = 2.0 * math.sqrt(max(S * F, 0.0))
    pn = lambda u: math.sqrt(max(pd(u, u), 0.0))  # noqa: E731
    h = lambda u: 1.0 - f.cos(g0, g0 + u)  # noqa: E731
    return {
        "G": G,
        "S": S,
        "F": F,
        "I": I,
        "rho": I / denom if denom > 0 else float("nan"),
        "rel_cos_B": pd(B1, B2) / (pn(B1) * pn(B2)),
        "rel_cos_C": pd(C1, C2) / (pn(C1) * pn(C2)),
        "h_B": h(Bm),
        "h_C": h(Cm),
        "h_B_plus_C": h(Bm + Cm),
        # debiased squared mean-leg norms (bc_ledger S/F scale), raw + perp
        "nB2_raw": max(f.dot(B1, B2) - f.dot(ref_diff, ref_diff) / 4.0, 0.0),
        "nC2_raw": max(f.dot(C1, C2), 0.0),
        "nB2_perp": max(S * 2 * G2, 0.0),
        "nC2_perp": max(F * 2 * G2, 0.0),
        "nR2_perp": max((S + F + I) * 2 * G2, 0.0),
        "_ghat": ghat,
        "_Bm": Bm,
        "_Cm": Cm,
    }


def criteria(row: dict) -> tuple[bool, bool]:
    readable = row["rel_cos_B"] >= REL_GATE and row["rel_cos_C"] >= REL_GATE
    ok = (
        row["I"] < 0
        and row["rho"] <= -0.5
        and row["h_B_plus_C"] < min(row["h_B"], row["h_C"])
    )
    return readable, readable and ok


def across_sigma(f: Frame, conds: list[dict], leg: str) -> list[dict]:
    """E24/e28 estimand on Frame conds: perp mean legs, debiased norms."""
    rows = []
    for (i, x), (j, y) in combinations(enumerate(conds), 2):
        if leg == "R":
            if (
                min(x["rel_cos_B"], x["rel_cos_C"], y["rel_cos_B"], y["rel_cos_C"])
                < REL_GATE
            ):
                continue
            ux, uy = x["_Bm"] + x["_Cm"], y["_Bm"] + y["_Cm"]
            nx2, ny2 = x["nR2_perp"], y["nR2_perp"]
        else:
            rk = f"rel_cos_{leg}"
            if x[rk] < REL_GATE or y[rk] < REL_GATE:
                continue
            ux, uy = (x["_Bm"], y["_Bm"]) if leg == "B" else (x["_Cm"], y["_Cm"])
            nx2, ny2 = (
                (x["nB2_perp"], y["nB2_perp"])
                if leg == "B"
                else (x["nC2_perp"], y["nC2_perp"])
            )
        # perp against each condition's own ghat (E24 dots the stored
        # per-condition perp legs); emulate by projecting each side.
        vx = ux - f.dot(ux, x["_ghat"]) * x["_ghat"]
        vy = uy - f.dot(uy, y["_ghat"]) * y["_ghat"]
        num = f.dot(vx, vy)
        den = math.sqrt(max(nx2 * ny2, 0.0))
        rows.append(
            {
                "pair": f"s{SIGMAS[i]:g}~s{SIGMAS[j]:g}",
                "cos": round(num / den, 5) if den > 0 else float("nan"),
            }
        )
    return rows


def match_delta(t1: list[dict], t2: list[dict]) -> dict:
    m1 = {r["pair"]: r["cos"] for r in t1}
    m2 = {r["pair"]: r["cos"] for r in t2}
    shared = sorted(set(m1) & set(m2))
    dv = [abs(m1[p] - m2[p]) for p in shared]
    return {
        "n_pairs": len(shared),
        "median_abs_delta": round(float(np.median(dv)), 4) if dv else None,
        "max_abs_delta": round(max(dv), 4) if dv else None,
    }


# ---------------------------------------------------------------- CPU hashing


def hash_constants(seed: int, name: str) -> tuple[int, int, int, int]:
    h = hashlib.blake2b(f"{seed}:{name}".encode(), digest_size=32).digest()

    def signed64(b: bytes, make_odd: bool = False) -> int:
        v = int.from_bytes(b, "little")
        if make_odd:
            v |= 1
        return v - (1 << 64) if v >= 1 << 63 else v

    return (
        signed64(h[0:8], make_odd=True),
        signed64(h[8:16]),
        signed64(h[16:24], make_odd=True),
        signed64(h[24:32]),
    )


def sketch_exact_store(man: dict, k: int, seed: int) -> dict[str, np.ndarray]:
    """CPU replica of the in-run 'lora_flat' sketch over the exact arm
    store (numpy int64 wraparound == torch; arithmetic-vs-logical shift
    is masked away identically)."""
    a1, b1, a2, b2 = hash_constants(seed, "lora_flat")
    kbits = k.bit_length() - 1
    dim = None
    out = {}
    bucket = sign = None
    for key in man["keys"]:
        for bi in range(len(man["sigma_centers"])):
            fname = f"{key.replace('@', '~')}__b{bi}.npy"
            v = np.load(STORE / fname, mmap_mode="r")
            if bucket is None:
                dim = v.shape[0]
                idx = np.arange(dim, dtype=np.int64)
                bucket = (((idx * a1 + b1) >> (64 - kbits)) & (k - 1)).astype(np.int64)
                sign = (1 - 2 * (((idx * a2 + b2) >> 62) & 1)).astype(np.float32)
                del idx
            sk = np.zeros(k, dtype=np.float64)
            for lo in range(0, dim, CHUNK):
                w = v[lo : lo + CHUNK].astype(np.float64) * sign[lo : lo + CHUNK]
                sk += np.bincount(bucket[lo : lo + CHUNK], weights=w, minlength=k)
            out[f"{key.replace('@', '~')}__b{bi}"] = sk.astype(np.float32)
    return out


def exact_param_gram(man: dict, conds: list[str]) -> np.ndarray:
    """Chunked fp64 Gram of the exact arm-store vectors (one pass)."""
    mms = [
        np.load(
            STORE / f"{c.rsplit('__b', 1)[0]}__b{c.rsplit('__b', 1)[1]}.npy",
            mmap_mode="r",
        )
        for c in conds
    ]
    dim = mms[0].shape[0]
    g = np.zeros((len(conds), len(conds)))
    for lo in range(0, dim, CHUNK):
        X = np.stack([m[lo : lo + CHUNK] for m in mms]).astype(np.float64)
        g += X @ X.T
    return g


def npz_gram(path: Path, conds: list[str]) -> np.ndarray:
    z = np.load(path)
    M = np.stack([z[c].astype(np.float64) for c in conds])
    return M @ M.T


# ---------------------------------------------------------------- main


def main() -> None:
    man = json.loads((STORE / "manifest.json").read_text())
    meta = json.loads((SKETCH / "meta.json").read_text())
    global SIGMAS
    SIGMAS = [round(s, 4) for s in man["sigma_centers"]]
    n_bins = len(SIGMAS)
    conds = [f"{a}__b{bi}" for bi in range(n_bins) for a in ARMS]

    out: dict = {
        "doc": "E30.1 read — gates + 30-A/30-B/30-C per the frozen registration",
        "run": str(RUN.relative_to(SL)),
        "store_meta": {
            k: meta[k]
            for k in ("k", "seed", "adaln_numel", "base_numel", "n_base_tensors")
        },
    }

    # ---- gate 1: param-frame ledger (as frozen) + twin comparison
    led = json.loads((RUN / "ledger.json").read_text())["bc_ledger"][ROUTE]
    g1_bins = []
    for s, r in zip(SIGMAS, led):
        readable, ok = criteria(r)
        g1_bins.append(
            {
                "sigma": s,
                "readable": readable,
                "criteria_pass": ok,
                **{
                    k: r[k]
                    for k in (
                        "h_B",
                        "h_C",
                        "h_B_plus_C",
                        "rho",
                        "I",
                        "rel_cos_B",
                        "rel_cos_C",
                    )
                },
            }
        )
    n_pass = sum(b["criteria_pass"] for b in g1_bins)
    twin = json.loads((HERE.parent / "e26" / "e26_grid_read.json").read_text())[
        "adapters"
    ]["sincos_twin"]["bins"]
    cols = ("h_B", "h_C", "h_B_plus_C", "rho", "I", "rel_cos_B", "rel_cos_C")
    twin_delta = max(abs(b[c] - t[c]) for b, t in zip(g1_bins, twin) for c in cols)
    twin_pass = sum(t["criteria_pass"] for t in twin)
    out["gate1"] = {
        "criteria": "E26 frozen: readable rel_B^rel_C>=0.5; pass I<0, "
        "rho<=-0.5, h(B+C)<min(h_B,h_C); gate: pass on >=4/5 bins",
        "bins": g1_bins,
        "n_readable": sum(b["readable"] for b in g1_bins),
        "n_criteria_pass": n_pass,
        "pass_as_frozen": n_pass >= 4,
        "twin_reference": {
            "n_criteria_pass": twin_pass,
            "max_abs_delta_vs_twin": round(twin_delta, 5),
            "note": "e26_grid_read.json sincos_twin columns (published "
            "2026-08-10, before this registration) show the identical "
            "1/5 pattern — the frozen pass condition was unattainable at "
            "the sincos operating point (registration defect; README "
            "amendment). Instrument-perturbation read: this run "
            "reproduces the un-instrumented twin's ledger to "
            "max|delta| above.",
        },
    }
    print(
        f"[gate1] pass_as_frozen={n_pass >= 4} ({n_pass}/5 bins; twin "
        f"itself {twin_pass}/5; max|Δ| vs twin {twin_delta:.5f})"
    )

    # ---- Grams
    print("[gram] exact param store (one chunked pass over 12 GB)…")
    G_exact = exact_param_gram(man, conds)
    fams = {
        fam: Frame(npz_gram(SKETCH / f"sketch_{fam}.npz", conds), conds)
        for fam in ("full", "adaln", "complement", "param")
    }
    z = np.load(SKETCH / "adaln_exact.npz")
    order = [list(z["conditions"]).index(c) for c in conds]
    f_adaln_exact = Frame(z["gram"][np.ix_(order, order)], conds)
    f_exact = Frame(G_exact, conds)

    # ---- gate 2: pairwise sketched-vs-exact cosines + streaming cert
    deltas = []
    for bi in range(n_bins):
        names = [f"{a}__b{bi}" for a in ARMS]
        for x, y in combinations(names, 2):
            ce = f_exact.cos(f_exact.vec({x: 1.0}), f_exact.vec({y: 1.0}))
            cs = fams["param"].cos(
                fams["param"].vec({x: 1.0}), fams["param"].vec({y: 1.0})
            )
            deltas.append(
                {"bin": bi, "pair": f"{x}~{y}", "delta": round(abs(ce - cs), 6)}
            )
    max_d = max(d["delta"] for d in deltas)
    print(
        f"[gate2] pairwise max|Δcos| = {max_d:.6f} (tol {GATE2_TOL}) over "
        f"{len(deltas)} pairs"
    )

    print("[gate2] streaming-path certification (CPU re-sketch of the exact store)…")
    cpu_sk = sketch_exact_store(man, meta["k"], meta["seed"])
    z_param = np.load(SKETCH / "sketch_param.npz")
    cert = {}
    for c in conds:
        u, v = cpu_sk[c].astype(np.float64), z_param[c].astype(np.float64)
        cert[c] = round(float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v))), 6)
    cert_min = min(cert.values())
    print(f"[gate2] streaming cert min cos = {cert_min:.6f}")
    out["gate2"] = {
        "pairwise_max_abs_delta_cos": max_d,
        "tolerance": GATE2_TOL,
        "n_pairs": len(deltas),
        "pass": max_d <= GATE2_TOL,
        "worst_pairs": sorted(deltas, key=lambda d: -d["delta"])[:5],
        "streaming_cert_min_cos": cert_min,
        "streaming_cert": cert,
    }

    # ---- 30-A: base-frame expression gate on the full sketch
    conds_by_frame = {}
    for fam in ("full", "adaln", "complement", "param"):
        conds_by_frame[fam] = [frame_cond(fams[fam], bi) for bi in range(n_bins)]
    conds_exact = [frame_cond(f_exact, bi) for bi in range(n_bins)]

    a_bins = []
    for s, row in zip(SIGMAS, conds_by_frame["full"]):
        readable, ok = criteria(row)
        a_bins.append(
            {
                "sigma": s,
                "readable": readable,
                "criteria_pass": ok,
                **{
                    k: round(row[k], 5)
                    for k in (
                        "S",
                        "F",
                        "I",
                        "rho",
                        "rel_cos_B",
                        "rel_cos_C",
                        "h_B",
                        "h_C",
                        "h_B_plus_C",
                    )
                },
            }
        )
    n_read = sum(b["readable"] for b in a_bins)
    n_ok = sum(b["criteria_pass"] for b in a_bins)
    if n_read >= 4 and all(b["criteria_pass"] for b in a_bins if b["readable"]):
        verdict_a = "EXPRESSES"
    elif n_read < math.ceil(n_bins / 2):
        verdict_a = "NOT-EXPRESSED"
    else:
        verdict_a = "PARTIAL"
    out["reading_30A"] = {
        "frame": "full-base count-sketch (k=2^18), bc_ledger machinery, data_ref=reenc",
        "bins": a_bins,
        "n_readable": n_read,
        "n_criteria_pass": n_ok,
        "verdict": verdict_a,
        "gate1_caveat": "gate 1 failed as frozen (see gate1.twin_reference)",
    }
    print(f"\n[30-A] verdict {verdict_a} (readable {n_read}/5, pass {n_ok}/5)")
    print(
        "  σ       S        F        I        rho     relB   relC   hB      hC      hBC"
    )
    for b in a_bins:
        print(
            f"  {b['sigma']:<7} {b['S']:+.4f}  {b['F']:+.4f}  {b['I']:+.4f}  "
            f"{b['rho']:+.3f}  {b['rel_cos_B']:.3f}  {b['rel_cos_C']:.3f}  "
            f"{b['h_B']:.4f}  {b['h_C']:.4f}  {b['h_B_plus_C']:.4f}"
            f"  {'PASS' if b['criteria_pass'] else ('read' if b['readable'] else 'UNREADABLE')}"
        )

    readable_idx = [i for i, b in enumerate(a_bins) if b["readable"]]

    # ---- 30-B(i): adaln norm shares (exact numerator / sketch denominator)
    shares = []
    for bi in readable_idx:
        ex, fu = frame_cond(f_adaln_exact, bi), conds_by_frame["full"][bi]
        sk_ad = conds_by_frame["adaln"][bi]
        shares.append(
            {
                "sigma": SIGMAS[bi],
                "s_B_exact_raw": round(ex["nB2_raw"] / fu["nB2_raw"], 4),
                "s_C_exact_raw": round(ex["nC2_raw"] / fu["nC2_raw"], 4),
                "s_B_sketch_perp": round(sk_ad["nB2_perp"] / fu["nB2_perp"], 4),
                "s_C_sketch_perp": round(sk_ad["nC2_perp"] / fu["nC2_perp"], 4),
                "adaln_norm_calib_sketch_over_exact_B": round(
                    math.sqrt(sk_ad["nB2_raw"] / ex["nB2_raw"]), 4
                ),
            }
        )
    med = lambda k: round(float(np.median([r[k] for r in shares])), 4)  # noqa: E731
    med_sB, med_sC = med("s_B_exact_raw"), med("s_C_exact_raw")

    # ---- 30-B(ii): slice-vs-full across-sigma profiles
    prof = {}
    for fam in ("full", "adaln", "complement"):
        prof[fam] = {
            leg: across_sigma(fams[fam], conds_by_frame[fam], leg)
            for leg in ("B", "C", "R")
        }
    prof["param_exact"] = {
        leg: across_sigma(f_exact, conds_exact, leg) for leg in ("B", "C", "R")
    }
    slice_vs_full = {
        leg: match_delta(prof["adaln"][leg], prof["full"][leg])
        for leg in ("B", "C", "R")
    }
    prof_ok = (
        slice_vs_full["R"]["median_abs_delta"] is not None
        and slice_vs_full["R"]["median_abs_delta"] <= PROFILE_TOL
    )
    verdict_b = (
        "SUFFICIENT"
        if med_sB >= S_THRESH and med_sC >= S_THRESH and prof_ok
        else "INSUFFICIENT"
    )
    out["reading_30B"] = {
        "shares": shares,
        "median_s_B": med_sB,
        "median_s_C": med_sC,
        "denominator_note": "exact full norms were not stored (registration "
        "wrote 'streamed exact norms'; exact streaming covered the adaln "
        "slice only) — denominator from the full sketch, ~0.3 % norm error "
        "at k=2^18; recorded deviation",
        "profile_slice_vs_full": slice_vs_full,
        "profile_gate_leg": "R (rotation profile primary; B/C alongside)",
        "thresholds": {"s": S_THRESH, "profile": PROFILE_TOL},
        "verdict": verdict_b,
        "gate1_caveat": "gate 1 failed as frozen (see gate1.twin_reference)",
    }
    print(
        f"\n[30-B] verdict {verdict_b}: median s_B={med_sB} s_C={med_sC} "
        f"(>= {S_THRESH} both), slice-vs-full R median|Δ|="
        f"{slice_vs_full['R']['median_abs_delta']} (<= {PROFILE_TOL})"
    )
    for r in shares:
        print(
            f"  σ={r['sigma']:<7} sB={r['s_B_exact_raw']:.3f} "
            f"sC={r['s_C_exact_raw']:.3f}  (perp-sketch "
            f"{r['s_B_sketch_perp']:.3f}/{r['s_C_sketch_perp']:.3f})"
        )

    # ---- 30-C descriptives
    rho_tbl = [
        {
            "sigma": SIGMAS[i],
            "rho_base": round(conds_by_frame["full"][i]["rho"], 4),
            "rho_param": round(led[i]["rho"], 4),
            "rho_adaln": round(conds_by_frame["adaln"][i]["rho"], 4),
            "rho_complement": round(conds_by_frame["complement"][i]["rho"], 4),
        }
        for i in range(n_bins)
    ]
    comp_shares = [
        {
            "sigma": SIGMAS[bi],
            "s_B": round(
                conds_by_frame["complement"][bi]["nB2_perp"]
                / conds_by_frame["full"][bi]["nB2_perp"],
                4,
            ),
            "s_C": round(
                conds_by_frame["complement"][bi]["nC2_perp"]
                / conds_by_frame["full"][bi]["nC2_perp"],
                4,
            ),
        }
        for bi in readable_idx
    ]
    base_vs_param = {
        leg: match_delta(prof["full"][leg], prof["param_exact"][leg])
        for leg in ("B", "C", "R")
    }
    twin_R = json.loads(
        (HERE.parent / "e26" / "e26_grid_across_sigma.json").read_text()
    )["adapters"]["sincos_twin"]["across_sigma_R"]["pairs"]
    base_vs_twin_R = match_delta(prof["full"]["R"], twin_R)

    e29_cls = None
    r_pairs = prof["full"]["R"]
    if len(r_pairs) == len(E29_BINS) * (len(E29_BINS) - 1) // 2:
        e29_cls = classify(to_mat(r_pairs))
    out["reading_30C"] = {
        "rho_by_frame": rho_tbl,
        "complement_shares_perp": comp_shares,
        "profiles": prof,
        "base_vs_param_profile": base_vs_param,
        "base_vs_committed_twin_R": {
            **base_vs_twin_R,
            "caveat": "scalar comparison only — cross-boot Δρ-band caveat "
            "(project_crossboot_arm_store_break); twin table from "
            "e26_grid_across_sigma.json",
        },
        "e29_classification_base_R": e29_cls,
    }
    print("\n[30-C] rho by frame (base / param / adaln / complement):")
    for r in rho_tbl:
        print(
            f"  σ={r['sigma']:<7} {r['rho_base']:+.3f} {r['rho_param']:+.3f} "
            f"{r['rho_adaln']:+.3f} {r['rho_complement']:+.3f}"
        )
    print(f"[30-C] base-vs-param R profile: {base_vs_param['R']}")
    print(f"[30-C] base-vs-twin R profile: {base_vs_twin_R}")
    if e29_cls:
        print(
            f"[30-C] E29 k* of base R̂: {e29_cls['k_star']} "
            f"partition {e29_cls['partition']}"
        )

    out_path = HERE / "e30_read.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
