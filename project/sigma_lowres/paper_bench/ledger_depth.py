#!/usr/bin/env python
"""sigma_lowres E19.3 — depth-resolved interventional ledger (ρ_ℓ per slice).

CPU reanalysis of a ``run_sigma_probe.py --repromote --keep_arm_sums
--self_floor`` run whose ``arm_sums/`` carries ``groups.json`` (the
``build_groups`` flat-vector layout: ``type:<module>`` + ``block:<idx>`` +
the fused-qkv row splits). ``--per_group`` is bookkeeping over the same flat
gradient vectors, so the full-vector store already contains the depth ledger —
this script slices each stored arm sum by the parameter layout and runs the
E14 ``bc_ledger`` estimator per slice.

Two estimands per (group, route, bin), both cross-set debiased exactly as the
global ledger (B₁/C₂ + B₂/C₁ crossing, ref-noise subtraction):

* **slice-local ledger** — the slice treated as its own space (ĝ_ℓ from the
  slice's own native gradient): S_ℓ/F_ℓ/I_ℓ/ρ_ℓ + rel_cos gates. Answers
  "within this slice's parameters, are B and C anti-aligned?" — ρ_ℓ is the
  19.3 verdict statistic (depth-concentrated vs depth-uniform).
* **global-perp partition** — ⟂ taken against the *global* ĝ, restricted to
  the slice's coordinates: S/F/I_part sum over the block partition to the
  global S/F/I **exactly**, so ``I_part`` is an additive answer to "which
  depth owns the global interaction term".

Usage::

    uv run python project/sigma_lowres/paper_bench/ledger_depth.py \
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
)

REL_GATE = 0.5  # E14's reproducibility floor — cells below do not read


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--run", required=True, help="probe run dir holding arm_sums/")
    p.add_argument(
        "--out",
        default=None,
        help="output json (default <run>/ledger_depth[_ref].json)",
    )
    p.add_argument(
        "--data_ref",
        choices=["native", "reenc"],
        default="reenc",
        help="reference arm for B (reenc = E19's primary, matching ledger.json)",
    )
    return p.parse_args()


def _gather(v: np.ndarray, ranges: list[list[int]]) -> np.ndarray:
    if len(ranges) == 1:
        s, e = ranges[0]
        return np.asarray(v[s:e], dtype=np.float64)
    return np.concatenate([v[s:e] for s, e in ranges]).astype(np.float64, copy=False)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(a @ b) / (na * nb)


def slice_ledger(
    g0l: np.ndarray,
    B1l: np.ndarray,
    B2l: np.ndarray,
    C1l: np.ndarray,
    C2l: np.ndarray,
    rdl: np.ndarray,
    scal: dict[str, float],
) -> dict:
    """Both estimands for one slice. ``scal`` carries the global scalars
    (G, ĝ·B₁, ĝ·B₂, ĝ·C₁, ĝ·C₂, ĝ·ref_diff)."""
    # --- global-perp partition: (B⊥)[R] = B[R] − (ĝᵀB) ĝ[R], ĝ[R] = g0[R]/G
    gh_g = g0l / scal["G"]
    B1pg, B2pg = B1l - scal["cB1"] * gh_g, B2l - scal["cB2"] * gh_g
    C1pg, C2pg = C1l - scal["cC1"] * gh_g, C2l - scal["cC2"] * gh_g
    rdpg = rdl - scal["cRd"] * gh_g
    G2g = 2.0 * scal["G"] ** 2
    S_part = (float(B1pg @ B2pg) - float(rdpg @ rdpg) / 4.0) / G2g
    F_part = float(C1pg @ C2pg) / G2g
    I_part = (float(B1pg @ C2pg) + float(B2pg @ C1pg)) / G2g

    # --- slice-local ledger: the slice as its own space
    Gl = float(np.linalg.norm(g0l))
    ghl = g0l / Gl
    B1p, B2p = B1l - float(ghl @ B1l) * ghl, B2l - float(ghl @ B2l) * ghl
    C1p, C2p = C1l - float(ghl @ C1l) * ghl, C2l - float(ghl @ C2l) * ghl
    rdp = rdl - float(ghl @ rdl) * ghl
    G2 = 2.0 * Gl * Gl
    ref_noise = float(rdp @ rdp) / 4.0
    S = (float(B1p @ B2p) - ref_noise) / G2
    F = float(C1p @ C2p) / G2
    I_x = (float(B1p @ C2p) + float(B2p @ C1p)) / G2
    I_same = (float(B1p @ C1p) + float(B2p @ C2p)) / G2
    denom = 2.0 * math.sqrt(max(S * F, 0.0))
    return {
        "G_l": round(Gl, 4),
        "S": round(S, 6),
        "F": round(F, 6),
        "I": round(I_x, 6),
        "I_sameset_biascheck": round(I_same, 6),
        "rho": round(I_x / denom, 5) if denom > 0.0 else float("nan"),
        "rel_cos_B": round(_cos(B1p, B2p), 5),
        "rel_cos_C": round(_cos(C1p, C2p), 5),
        "S_part": round(S_part, 6),
        "F_part": round(F_part, 6),
        "I_part": round(I_part, 6),
    }


def main() -> None:
    args = parse_args()
    run = Path(args.run).resolve()
    s = Sums(run / "arm_sums", device="cpu")
    man = s.man
    groups: dict[str, list[list[int]]] = json.loads(
        (run / "arm_sums" / "groups.json").read_text()
    )
    centers = man["sigma_centers"]
    edges = [
        e for e in man["demote_edges"].split(",") if e and s.has(e) and s.has(f"{e}rp")
    ]
    if not edges:
        raise SystemExit("no repromote route pairs in the store")
    if not (s.has("a") and s.has("b")):
        raise SystemExit("store lacks the native a/b draw sets")
    if args.data_ref == "reenc" and not s.has("reenc__2"):
        raise SystemExit("--data_ref reenc needs reenc + reenc__2 in the store")

    block_keys = sorted(
        g for g in groups if g.startswith("block:") and g != "block:other"
    )
    result: dict = {
        "run": str(run),
        "sigma_centers": centers,
        "data_ref": args.data_ref,
        "rel_gate": REL_GATE,
        "manifest": man,
        "global": {e: [None] * s.n_bins for e in edges},
        "groups": {e: {g: [None] * s.n_bins for g in groups} for e in edges},
        "partition_check": {e: [None] * s.n_bins for e in edges},
    }

    for bi in range(s.n_bins):
        a, b = s.mean("a", bi), s.mean("b", bi)
        g0 = 0.5 * (a + b)
        G = float(np.linalg.norm(g0))
        ghat = g0 / G
        if args.data_ref == "reenc":
            re1, re2 = s.mean("reenc", bi), s.mean("reenc__2", bi)
            ref, ref_diff = 0.5 * (re1 + re2), re1 - re2
        else:
            ref, ref_diff = g0, a - b
        for e in edges:
            result["global"][e][bi] = bc_ledger(s, e, bi, args.data_ref)
            dem1, dem2 = s.mean(e, bi), s.mean(f"{e}__2", bi)
            rp1, rp2 = s.mean(f"{e}rp", bi), s.mean(f"{e}rp__2", bi)
            B1, B2 = rp1 - ref, rp2 - ref
            C1, C2 = dem1 - rp1, dem2 - rp2
            scal = {
                "G": G,
                "cB1": float(ghat @ B1),
                "cB2": float(ghat @ B2),
                "cC1": float(ghat @ C1),
                "cC2": float(ghat @ C2),
                "cRd": float(ghat @ ref_diff),
            }
            for g, ranges in groups.items():
                result["groups"][e][g][bi] = slice_ledger(
                    _gather(g0, ranges),
                    _gather(B1, ranges),
                    _gather(B2, ranges),
                    _gather(C1, ranges),
                    _gather(C2, ranges),
                    _gather(ref_diff, ranges),
                    scal,
                )
            # exactness check: block partition (+ block:other) must resum to
            # the global cross-set S/F/I
            part_keys = block_keys + (
                ["block:other"] if "block:other" in groups else []
            )
            sums = {
                k: round(
                    sum(result["groups"][e][g][bi][f"{k}_part"] for g in part_keys), 5
                )
                for k in ("S", "F", "I")
            }
            glob = result["global"][e][bi]
            sums["global"] = {k: glob[k] for k in ("S", "F", "I")}
            result["partition_check"][e][bi] = sums
        s.drop_bin(bi)

    # ---- console: the 19.3 read. ρ_ℓ per block × bin + I_part depth shares
    for e in edges:
        print(f"\n== route {e} (data_ref={args.data_ref}) ==")
        hdr = "block   " + "".join(f"σ={c:<7}" for c in centers) + "  I_part_share(mid)"
        print("ρ_ℓ per block ('()' = rel-gated out, rel_cos_B/C < 0.5):")
        print(hdr)
        mids = [bi for bi, c in enumerate(centers) if 0.29 <= c <= 0.84]
        I_mid_glob = sum(result["global"][e][bi]["I"] for bi in mids)
        for g in block_keys:
            cells = []
            for bi in range(s.n_bins):
                r = result["groups"][e][g][bi]
                gated = r["rel_cos_B"] < REL_GATE or r["rel_cos_C"] < REL_GATE
                cells.append(f"({r['rho']:+.2f}) " if gated else f" {r['rho']:+.2f}  ")
            I_share = (
                sum(result["groups"][e][g][bi]["I_part"] for bi in mids) / I_mid_glob
                if I_mid_glob
                else float("nan")
            )
            print(f"{g:7s} " + "".join(cells) + f"   {I_share:+.3f}")
        print("type groups (ρ_ℓ, same gating):")
        for g in sorted(k for k in groups if k.startswith("type:")):
            cells = []
            for bi in range(s.n_bins):
                r = result["groups"][e][g][bi]
                gated = r["rel_cos_B"] < REL_GATE or r["rel_cos_C"] < REL_GATE
                cells.append(f"({r['rho']:+.2f}) " if gated else f" {r['rho']:+.2f}  ")
            print(f"{g:28s} " + "".join(cells))
        pc = result["partition_check"][e]
        worst = max(
            abs(pc[bi][k] - pc[bi]["global"][k])
            for bi in range(s.n_bins)
            for k in ("S", "F", "I")
        )
        print(f"partition exactness: max |Σ_block − global| = {worst:.5f}")

    suffix = "" if args.data_ref == "reenc" else f"_{args.data_ref}"
    out = Path(args.out) if args.out else run / f"ledger_depth{suffix}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nledger → {out}")


if __name__ == "__main__":
    main()
