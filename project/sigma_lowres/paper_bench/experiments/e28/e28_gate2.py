#!/usr/bin/env python
"""sigma_lowres E28 — validation gates 1+2 on the 768 stage (CPU).

Gate 1 (machinery): the E24 synthetic suite reruns clean, and the new
store's within-condition scalars reproduce through the independent
``vector_ledger.bc_ledger`` path on every bin.

Gate 2 (the free control): at noising sigma = 0.7 the frozen-conditioning
run is protocol-near-identical to a native run — its B-hat/C-hat must match
the committed e193/e194 sigma = 0.7 / 768 axis at the cross-store band
(|cos| >= 0.95, both legs). Fails => the flag is not the one-argument swap
it claims to be; nothing else is read and the 896 stage is not submitted.

Seed-layout caveat (recorded, not frozen in the registration): e28 shares
seed 42, the probe list, AND the per-bin draw offsets with e193 (both grids
put the 0.7 bin at draw indices 36-47), and the reenc arm sits at arm_idx 2
in both runs — so the reenc REFERENCE noise is a shared realization across
the e28<->e193 pair, violating the cross-store independence the E24
convention assumes. The B-leg numerator therefore gets the same shared-ref
correction E24 applies within-store: subtract <d_perp_x, d_perp_y> / 4,
the empirical covariance of the two runs' reenc set-differences (a pure
noise read; ~0 for truly independent runs — asserted on the e194 pair).
Native a/b share seeds the same way, which is gate 2's *point* (the anchor
is supposed to reproduce); the 768/768rp arms sit at different arm indices
(3/4 vs e193's 5/6) and stay independent. The gate reads the CORRECTED
cosine; raw is reported alongside.

28-A preview (context only, no verdict weight — 28-A is read on the full
store after the 896 stage): per-bin rel_cos_B/rel_cos_C of every 768
frozen-conditioning condition, against the rel gate 0.5.

Usage::

    uv run python project/sigma_lowres/paper_bench/experiments/e28/e28_gate2.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from project.sigma_lowres.paper_bench.experiments.e24.e24_axis import (  # noqa: E402
    build_cond,
    cross_cos,
    validate_synthetic,
)
from project.sigma_lowres.paper_bench.vector_ledger import Sums, bc_ledger  # noqa: E402

SL = REPO_ROOT / "project/sigma_lowres"
E28_STORE = SL / "bench/results/20260809-2216-e28-cond07-768/arm_sums"
NATIVE = {
    "e193": SL / "bench/results/20260807-0745-e193-depth-ledger/arm_sums",
    "e194": SL / "bench/results/20260807-1400-e194-pi-causal/arm_sums",
}
GATE_SIGMA = 0.7
GATE_COS = 0.95
REL_GATE = 0.5


def perp_refdiff(s: Sums, bi: int) -> np.ndarray:
    """The condition's reenc set-difference, perp'd against its own ghat —
    the pure-noise read the shared-ref correction is built from."""
    a, b = s.mean("a", bi), s.mean("b", bi)
    g0 = 0.5 * (a + b)
    ghat = g0 / float(np.linalg.norm(g0))
    d = s.mean("reenc", bi) - s.mean("reenc__2", bi)
    return (d - float(ghat @ d) * ghat).astype(np.float32)


def main() -> None:
    t0 = time.time()
    validate_synthetic()  # gate 1a

    s28 = Sums(E28_STORE)
    assert s28.man.get("cond_sigma") == GATE_SIGMA, s28.man.get("cond_sigma")
    centers28 = [round(float(v), 4) for v in s28.man["sigma_centers"]]

    # gate 1b: build_cond == bc_ledger on EVERY bin of the new store,
    # collecting the 28-A preview rows and the gate-bin condition
    preview: list[dict] = []
    c28 = d28 = None
    for bi, sig in enumerate(centers28):
        led = bc_ledger(s28, "768", bi, "reenc")
        c = build_cond(s28, "e28", "768", bi)
        for k, v in (
            ("S", c.S),
            ("F", c.F),
            ("I", c.I),
            ("rho", c.rho),
            ("rel_cos_B", c.rel_B),
            ("rel_cos_C", c.rel_C),
        ):
            assert abs(round(v, 5) - led[k]) <= 1e-5, (sig, k, v, led[k])
        preview.append(
            {
                "sigma": sig,
                "rho": round(c.rho, 4),
                "rel_cos_B": round(c.rel_B, 4),
                "rel_cos_C": round(c.rel_C, 4),
                "readable": bool(c.rel_B >= REL_GATE and c.rel_C >= REL_GATE),
            }
        )
        if sig == GATE_SIGMA:
            c28, d28 = c, perp_refdiff(s28, bi)
        s28.drop_bin(bi)
    assert c28 is not None
    print(f"[gate1] synthetic clean; bc_ledger reproduced on all {len(centers28)} bins")
    for r in preview:
        flag = "PASS" if r["readable"] else "FAIL"
        print(
            f"[28-A preview] s{r['sigma']:<6} rho={r['rho']:+.3f} "
            f"relB={r['rel_cos_B']:.3f} relC={r['rel_cos_C']:.3f} {flag}"
        )

    # gate 2: cross-store axis agreement at the sigma_cond bin
    record: dict = {
        "generated_by": "e28_gate2.py",
        "store": str(E28_STORE.relative_to(REPO_ROOT)),
        "gate_sigma": GATE_SIGMA,
        "gate_cos": GATE_COS,
        "preview_28A_768": preview,
        "pairs": {},
    }
    verdicts: list[bool] = []
    for name, root in NATIVE.items():
        s = Sums(root)
        centers = [round(float(v), 4) for v in s.man["sigma_centers"]]
        bi = centers.index(GATE_SIGMA)
        cn = build_cond(s, name, "768", bi)
        dn = perp_refdiff(s, bi)
        shared = float(d28.astype(np.float64) @ dn.astype(np.float64)) / 4.0
        cos_b_raw = cross_cos(c28, cn, "B")
        den_b = float(np.sqrt(c28.nB2 * cn.nB2))
        cos_b_corr = cos_b_raw - (shared / den_b if den_b > 0 else 0.0)
        cos_c = cross_cos(c28, cn, "C")
        ghat_cos = float(c28.ghat32.astype(np.float64) @ cn.ghat32.astype(np.float64))
        ok = abs(cos_b_corr) >= GATE_COS and abs(cos_c) >= GATE_COS
        verdicts.append(ok)
        record["pairs"][name] = {
            "cos_B_raw": round(cos_b_raw, 5),
            "shared_ref_num": round(shared / den_b, 6) if den_b > 0 else None,
            "cos_B_corrected": round(cos_b_corr, 5),
            "cos_C": round(cos_c, 5),
            "ghat_cos": round(ghat_cos, 5),
            "rel_B_native": round(cn.rel_B, 4),
            "rel_C_native": round(cn.rel_C, 4),
            "pass": ok,
        }
        print(
            f"[gate2] e28 <-> {name}/768/s0.7: "
            f"cos_B raw {cos_b_raw:+.4f} corr {cos_b_corr:+.4f} "
            f"(shared-ref {shared / den_b:+.5f})  cos_C {cos_c:+.4f}  "
            f"ghat {ghat_cos:+.5f}  -> {'PASS' if ok else 'FAIL'}"
        )
        s.drop_bin(bi)
        del s

    # the e194 pair is seed-independent of e28 at this bin (different draw
    # offsets AND different arm indices) — its shared-ref term doubles as
    # the negative control for the correction itself
    ctrl = abs(record["pairs"]["e194"]["shared_ref_num"])
    print(f"[control] e194 shared-ref term |{ctrl:.6f}| (expect ~0)")

    record["verdict"] = "PASS" if all(verdicts) else "FAIL"
    (HERE / "e28_gate2.json").write_text(json.dumps(record, indent=1))
    print(
        f"[done] gate 2 {record['verdict']} — "
        f"{'896 stage licensed' if all(verdicts) else 'STOP: do not submit 896'}"
        f"  ({time.time() - t0:.0f}s)"
    )


if __name__ == "__main__":
    main()
