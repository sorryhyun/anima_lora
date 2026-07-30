#!/usr/bin/env python
"""sigma_lowres — interventional vector ledger over a probe run's arm_sums/.

CPU reanalysis of a ``run_sigma_probe.py --keep_arm_sums`` run. Replaces the
scalar read Δ_demote − Δ_repromote − Floor_e (which conflates a negative
interaction with a graph share that collapses below its endpoint value) with
the direct decomposition in shared adapter-parameter space:

    B(σ) = ḡ_repromote − ḡ_native     (data intervention, native graph)
    C(σ) = ḡ_demote    − ḡ_repromote  (graph intervention, demoted data)
    ḡ_demote − ḡ_native = B + C        (exact)

Projected against the native direction (P⊥ = I − ĝĝᵀ):

    S(σ) = |B⊥|²/2G²   F(σ) = |C⊥|²/2G²   I(σ) = ⟨B⊥, C⊥⟩/G²

**Noise handling**: every arm was run with two independent draw sets
(--self_floor), so second moments are debiased by CROSS-SET inner products —
⟨B₁⊥, B₂⊥⟩ is an unbiased |B⊥|² (independent draw noise cancels in
expectation), and I uses the set-crossed pairs (B₁,C₂)/(B₂,C₁) only, because
B₁ and C₁ share the repromote set-1 vector with opposite signs (shared noise
would fake anti-correlation — exactly the artifact that would manufacture a
spurious negative interaction). Same-set values are reported as a bias check.

Also reports the exact counterfactual angles h(u) = 1 − cos(ḡ, ḡ+u) for
u ∈ {B, C, B+C} (the quadratic expansion is marginal at 768 and poor at 512),
κ∥ components (the gap is blind to parallel perturbations), and — when the
run carried --target_alpha 0,1 — the aggregate target-content ledger
t = ḡ(α=1) − ḡ(α=0) with κ∥/κ⊥ of δt = t_arm − t_src.

Usage::

    uv run python project/sigma_lowres/paper_bench/vector_ledger.py \
        --run project/sigma_lowres/bench/results/<run>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--run", required=True, help="probe run dir holding arm_sums/")
    p.add_argument(
        "--out", default=None, help="output json (default <run>/ledger.json)"
    )
    p.add_argument(
        "--data_ref",
        choices=["native", "reenc"],
        default="native",
        help="reference arm for the data intervention B: 'native' (raw "
        "B = rp − ḡ0, carries the full down+up+encode pipeline cost) or "
        "'reenc' (B = rp − reenc̄, strips the shared encode-chain cost). "
        "Both are reported; this picks which one feeds S/I/h.",
    )
    return p.parse_args()


class Sums:
    """Lazy loader for the fp32 arm-sum memmaps (read-only)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.man = json.loads((root / "manifest.json").read_text())
        self.n_bins = len(self.man["sigma_centers"])
        self.draws = self.man["draws_per_bin"]

    def has(self, key: str) -> bool:
        return key in self.man["keys"]

    def mean(self, key: str, bi: int) -> np.ndarray:
        """Per-draw-mean gradient vector for (arm key, bin), float64."""
        info = self.man["keys"][key]
        fname = f"{key.replace('@', '~')}__b{bi}.npy"
        v = np.load(self.root / fname, mmap_mode="r")
        return np.asarray(v, dtype=np.float64) / (info["n_images"] * self.draws)


def perp(v: np.ndarray, ghat: np.ndarray) -> np.ndarray:
    return v - float(ghat @ v) * ghat


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b) / (na * nb)


def bc_ledger(s: Sums, edge: str, bi: int, data_ref: str) -> dict:
    """B/C split for one (route, bin). Requires a,b, <e>, <e>rp (+ __2 sets)."""
    a, b = s.mean("a", bi), s.mean("b", bi)
    g0 = 0.5 * (a + b)
    G = float(np.linalg.norm(g0))
    ghat = g0 / G
    dem1, dem2 = s.mean(edge, bi), s.mean(f"{edge}__2", bi)
    rp1, rp2 = s.mean(f"{edge}rp", bi), s.mean(f"{edge}rp__2", bi)

    refs = {"native": (g0, a - b)}
    if s.has("reenc") and s.has("reenc__2"):
        re1, re2 = s.mean("reenc", bi), s.mean("reenc__2", bi)
        refs["reenc"] = (0.5 * (re1 + re2), re1 - re2)
    ref, ref_diff = refs.get(data_ref, refs["native"])

    B1, B2 = rp1 - ref, rp2 - ref
    C1, C2 = dem1 - rp1, dem2 - rp2
    B1p, B2p = perp(B1, ghat), perp(B2, ghat)
    C1p, C2p = perp(C1, ghat), perp(C2, ghat)
    Bp, Cp = 0.5 * (B1p + B2p), 0.5 * (C1p + C2p)
    Bm, Cm = 0.5 * (B1 + B2), 0.5 * (C1 + C2)

    G2 = G * G
    # cross-set debiased second moments (draw noise independent across sets).
    # B1 and B2 share the REFERENCE arm's noise (ref averages its two draw
    # sets), which survives the cross-set product: E⟨B1⊥,B2⊥⟩ = |B⊥|² +
    # (ref noise power)/2 with the power estimated from the ref set diff
    # (E|d⊥|² = 2·per-set power ⇒ shared power = |d⊥|²/4).
    ref_noise = float(np.linalg.norm(perp(ref_diff, ghat)) ** 2) / 4.0
    S_deb = (float(B1p @ B2p) - ref_noise) / (2 * G2)
    F_deb = float(C1p @ C2p) / (2 * G2)
    I_deb = (float(B1p @ C2p) + float(B2p @ C1p)) / (2 * G2)
    I_same = (float(B1p @ C1p) + float(B2p @ C2p)) / (2 * G2)  # bias check
    denom = 2.0 * math.sqrt(max(S_deb * F_deb, 0.0)) or float("nan")

    def h(u: np.ndarray) -> float:
        return 1.0 - cosine(g0, g0 + u)

    out = {
        "G": round(G, 4),
        # raw (noise-inflated) magnitudes + parallel components, both refs
        "b_perp_rawnorm": round(float(np.linalg.norm(Bp)) / G, 5),
        "c_perp_rawnorm": round(float(np.linalg.norm(Cp)) / G, 5),
        "kappa_par_B": round(float(ghat @ Bm) / G, 5),
        "kappa_par_C": round(float(ghat @ Cm) / G, 5),
        # reliability: direction reproducibility across independent draw sets
        "rel_cos_B": round(cosine(B1p, B2p), 5),
        "rel_cos_C": round(cosine(C1p, C2p), 5),
        # the ledger proper (cross-set debiased)
        "S": round(S_deb, 5),
        "F": round(F_deb, 5),
        "I": round(I_deb, 5),
        "I_sameset_biascheck": round(I_same, 5),
        "rho": round(I_deb / denom, 5) if denom == denom else float("nan"),
        "quad_pred_gap": round(S_deb + F_deb + I_deb, 5),
        # exact counterfactual angles (no quadratic assumption)
        "h_B": round(h(Bm), 5),
        "h_C": round(h(Cm), 5),
        "h_B_plus_C": round(h(Bm + Cm), 5),
        "data_ref": data_ref,
    }
    if "reenc" in refs and data_ref == "native":
        re_mean = refs["reenc"][0]
        Br = 0.5 * ((rp1 - re_mean) + (rp2 - re_mean))
        out["b_perp_rawnorm_reencref"] = round(
            float(np.linalg.norm(perp(Br, ghat))) / G, 5
        )
    return out


def kappa_ledger(s: Sums, arm: str, bi: int) -> dict:
    """Aggregate target-content read for one (arm, bin): t = ḡ(1) − ḡ(0)."""
    a1, b1 = s.mean("a", bi), s.mean("b", bi)
    a0, b0 = s.mean("a@a0", bi), s.mean("b@a0", bi)
    g0 = 0.5 * (a1 + b1)
    G = float(np.linalg.norm(g0))
    ghat = g0 / G
    t_src = 0.5 * ((a1 - a0) + (b1 - b0))
    k1, k0 = s.mean(arm, bi), s.mean(f"{arm}@a0", bi)
    t_arm = k1 - k0
    dt = t_arm - t_src
    par = float(ghat @ dt)
    perp_n = math.sqrt(max(0.0, float(dt @ dt) - par * par))
    out = {
        "tnorm_src": round(float(np.linalg.norm(t_src)) / G, 5),
        f"tnorm_{arm}": round(float(np.linalg.norm(t_arm)) / G, 5),
        "kappa_par": round(par / G, 5),
        "kappa_perp": round(perp_n / G, 5),
        "cos_t_arm_vs_src": round(cosine(t_arm, t_src), 5),
    }
    if s.has(f"{arm}__2") and s.has(f"{arm}__2@a0"):
        dt2 = (s.mean(f"{arm}__2", bi) - s.mean(f"{arm}__2@a0", bi)) - t_src
        out["rel_cos_dt"] = round(cosine(dt, dt2), 5)
    return out


def main() -> None:
    args = parse_args()
    run = Path(args.run).resolve()
    s = Sums(run / "arm_sums")
    man = s.man
    centers = man["sigma_centers"]
    edges = [e for e in man["demote_edges"].split(",") if e]
    result: dict = {"run": str(run), "sigma_centers": centers, "manifest": man}

    if man.get("repromote"):
        bc: dict = {}
        for e in edges:
            if not (s.has(e) and s.has(f"{e}rp")):
                continue
            bc[e] = [bc_ledger(s, e, bi, args.data_ref) for bi in range(s.n_bins)]
            print(f"\n== route {e} (data_ref={args.data_ref}) ==")
            hdr = "σ       S        F        I        rho     relB    relC   quad   h(B+C)"
            print(hdr)
            for bi, row in enumerate(bc[e]):
                print(
                    f"{centers[bi]:<7} {row['S']:+.4f}  {row['F']:+.4f}  "
                    f"{row['I']:+.4f}  {row['rho']:+.3f}  {row['rel_cos_B']:.3f}  "
                    f"{row['rel_cos_C']:.3f}  {row['quad_pred_gap']:+.4f} "
                    f"{row['h_B_plus_C']:+.4f}"
                )
        result["bc_ledger"] = bc

    if len(man.get("target_alphas", [1.0])) > 1:
        kap: dict = {}
        arms = [k for k in man["keys"] if k not in ("a", "b") and "@" not in k]
        for arm in arms:
            if not s.has(f"{arm}@a0"):
                continue
            kap[arm] = [kappa_ledger(s, arm, bi) for bi in range(s.n_bins)]
            last = kap[arm][-1]
            print(
                f"[kappa-agg] {arm}: par={last['kappa_par']:+.5f} "
                f"perp={last['kappa_perp']:.5f} |t|/G={last[f'tnorm_{arm}']:.4f} "
                f"(src {last['tnorm_src']:.4f})"
            )
        result["kappa_ledger"] = kap

    out = Path(args.out) if args.out else run / "ledger.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nledger → {out}")


if __name__ == "__main__":
    main()
