#!/usr/bin/env python
"""sigma_lowres E19.4 — PI-align causal ledger over a probe run's arm_sums/.

CPU reanalysis of a ``run_sigma_probe.py --repromote --pi_align
--keep_arm_sums --self_floor`` run. The ``<e>pi`` arm re-runs the
demoted-graph forward with PI-stretched RoPE phases (same latent, phase
geometry matched to the native grid), so alongside E14's

    B    = ḡ_rp   − ḡ_ref       (data intervention)
    C    = ḡ_dem  − ḡ_rp        (graph intervention)

it defines

    C_pi = ḡ_dem,pi − ḡ_rp      (graph intervention MINUS its phase geometry)

If the graph branch's anti-aligned component is RoPE-mediated, C must
**rotate** when the phase geometry is restored: ρ_pi = corr(B⊥, C_pi⊥)
moves toward 0 and the amplitude ratio √(S/F) moves, not merely shrinks.
The rotation is read directly as the debiased cos(C⊥, C_pi⊥).

Noise handling mirrors ``vector_ledger.bc_ledger`` exactly: two independent
draw sets per arm, second moments from CROSS-SET products only (C₁ and
C_pi,₁ share the rp set-1 vector, so same-set products would fake
correlation — the set-crossed pairs are unbiased), ref-noise subtraction on
S, same-set values reported as bias checks.

Regime restriction (G11, pre-registered): PI is off-manifold with content at
mid σ, so the clean read is noise-dominated bins only — the run's grid is
already restricted to the window top + tail + endpoint.

Usage::

    uv run python project/sigma_lowres/paper_bench/ledger_pi.py \
        --run project/sigma_lowres/bench/results/<run> --data_ref reenc
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

from project.sigma_lowres.paper_bench.vector_ledger import (  # noqa: E402
    Sums,
    bc_ledger,
    cosine,
    perp,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--run", required=True, help="probe run dir holding arm_sums/")
    p.add_argument(
        "--out",
        default=None,
        help="output json (default <run>/ledger_pi[_ref].json)",
    )
    p.add_argument(
        "--data_ref",
        choices=["native", "reenc"],
        default="reenc",
        help="reference arm for B (reenc = E19's primary)",
    )
    return p.parse_args()


def pi_ledger(s: Sums, edge: str, bi: int, data_ref: str) -> dict:
    """B/C/C_pi ledger for one (route, bin)."""
    a, b = s.mean("a", bi), s.mean("b", bi)
    g0 = 0.5 * (a + b)
    G = float(np.linalg.norm(g0))
    ghat = g0 / G
    if data_ref == "reenc" and s.has("reenc") and s.has("reenc__2"):
        re1, re2 = s.mean("reenc", bi), s.mean("reenc__2", bi)
        ref, ref_diff = 0.5 * (re1 + re2), re1 - re2
    else:
        data_ref = "native"
        ref, ref_diff = g0, a - b
    rp1, rp2 = s.mean(f"{edge}rp", bi), s.mean(f"{edge}rp__2", bi)
    dem1, dem2 = s.mean(edge, bi), s.mean(f"{edge}__2", bi)
    pi1, pi2 = s.mean(f"{edge}pi", bi), s.mean(f"{edge}pi__2", bi)

    B1, B2 = rp1 - ref, rp2 - ref
    C1, C2 = dem1 - rp1, dem2 - rp2
    P1, P2 = pi1 - rp1, pi2 - rp2
    B1p, B2p = perp(B1, ghat), perp(B2, ghat)
    C1p, C2p = perp(C1, ghat), perp(C2, ghat)
    P1p, P2p = perp(P1, ghat), perp(P2, ghat)

    G2 = 2.0 * G * G
    ref_noise = (float(np.linalg.norm(perp(ref_diff, ghat))) ** 2) / 4.0
    S = (float(B1p @ B2p) - ref_noise) / G2
    F = float(C1p @ C2p) / G2
    F_pi = float(P1p @ P2p) / G2
    I_c = (float(B1p @ C2p) + float(B2p @ C1p)) / G2
    I_pi = (float(B1p @ P2p) + float(B2p @ P1p)) / G2
    I_pi_same = (float(B1p @ P1p) + float(B2p @ P2p)) / G2
    # rotation: debiased <C⊥, C_pi⊥> via set-crossed pairs (C₁/P₁ share rp₁)
    R = (float(C1p @ P2p) + float(C2p @ P1p)) / G2
    R_same = (float(C1p @ P1p) + float(C2p @ P2p)) / G2

    def rho(i_val: float, f_val: float) -> float:
        d = 2.0 * math.sqrt(max(S * f_val, 0.0))
        return round(i_val / d, 5) if d > 0.0 else float("nan")

    def h(u: np.ndarray) -> float:
        return 1.0 - cosine(g0, g0 + u)

    Bm, Cm, Pm = 0.5 * (B1 + B2), 0.5 * (C1 + C2), 0.5 * (P1 + P2)
    cos_c_cpi = (
        R / (2.0 * math.sqrt(max(F * F_pi, 0.0)))
        if F > 0 and F_pi > 0
        else float("nan")
    )
    return {
        "G": round(G, 4),
        "S": round(S, 5),
        "F": round(F, 5),
        "F_pi": round(F_pi, 5),
        "I": round(I_c, 5),
        "I_pi": round(I_pi, 5),
        "I_pi_sameset_biascheck": round(I_pi_same, 5),
        "rho": rho(I_c, F),
        "rho_pi": rho(I_pi, F_pi),
        "cos_C_Cpi": round(cos_c_cpi, 5),
        "cos_C_Cpi_sameset_biascheck": round(
            R_same / (2.0 * math.sqrt(max(F * F_pi, 0.0)))
            if F > 0 and F_pi > 0
            else float("nan"),
            5,
        ),
        "amp_ratio": round(math.sqrt(S / F), 5) if S > 0 and F > 0 else float("nan"),
        "amp_ratio_pi": round(math.sqrt(S / F_pi), 5)
        if S > 0 and F_pi > 0
        else float("nan"),
        "rel_cos_B": round(cosine(B1p, B2p), 5),
        "rel_cos_C": round(cosine(C1p, C2p), 5),
        "rel_cos_Cpi": round(cosine(P1p, P2p), 5),
        "h_C": round(h(Cm), 5),
        "h_Cpi": round(h(Pm), 5),
        "h_B_plus_C": round(h(Bm + Cm), 5),
        "h_B_plus_Cpi": round(h(Bm + Pm), 5),
        "data_ref": data_ref,
    }


def main() -> None:
    args = parse_args()
    run = Path(args.run).resolve()
    s = Sums(run / "arm_sums", device="cpu")
    man = s.man
    centers = man["sigma_centers"]
    edges = [
        e
        for e in man["demote_edges"].split(",")
        if e and s.has(f"{e}pi") and s.has(f"{e}rp")
    ]
    if not edges:
        raise SystemExit("no <e>pi + <e>rp arm pairs in the store")

    result: dict = {
        "run": str(run),
        "sigma_centers": centers,
        "data_ref": args.data_ref,
        "manifest": man,
        "pi_ledger": {},
        "bc_ledger": {},
    }
    for bi in range(s.n_bins):
        for e in edges:
            result["pi_ledger"].setdefault(e, [None] * s.n_bins)[bi] = pi_ledger(
                s, e, bi, args.data_ref
            )
            result["bc_ledger"].setdefault(e, [None] * s.n_bins)[bi] = bc_ledger(
                s, e, bi, args.data_ref
            )
        s.drop_bin(bi)

    for e in edges:
        print(f"\n== route {e} (data_ref={args.data_ref}) ==")
        print(
            "σ       rho     rho_pi  cos(C,Cpi)  |B|/|C|  |B|/|Cpi|  "
            "h(C)     h(Cpi)   relB  relC  relCpi"
        )
        for bi, row in enumerate(result["pi_ledger"][e]):
            print(
                f"{centers[bi]:<7} {row['rho']:+.3f}  {row['rho_pi']:+.3f}  "
                f"{row['cos_C_Cpi']:+.3f}      {row['amp_ratio']:.3f}    "
                f"{row['amp_ratio_pi']:.3f}      {row['h_C']:+.4f}  "
                f"{row['h_Cpi']:+.4f}  {row['rel_cos_B']:.2f}  "
                f"{row['rel_cos_C']:.2f}  {row['rel_cos_Cpi']:.2f}"
            )

    suffix = "" if args.data_ref == "reenc" else f"_{args.data_ref}"
    out = Path(args.out) if args.out else run / f"ledger_pi{suffix}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nledger → {out}")


if __name__ == "__main__":
    main()
