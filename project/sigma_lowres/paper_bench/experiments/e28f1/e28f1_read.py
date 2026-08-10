#!/usr/bin/env python
"""E28-F1-ii — the sigma_cond-relocation discriminator read (CPU).

Frozen protocol per README: gates -> F1-A (readability) -> F1-B (the
E29 classifier on the frozen-conditioning store's own across-sigma
R-hat table) -> F1-C descriptive rows. Machinery is e28/e29 verbatim:
build_cond/bc_ledger/cross_cos (E24), set-pure cross-set cosines
(e28_gate2_twin, the seed-twin estimand), e29_cluster (tau = 0.30).

Usage::

    uv run python project/sigma_lowres/paper_bench/experiments/e28f1/e28f1_read.py
"""

from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE.parent / "e28"))
sys.path.insert(0, str(HERE.parent / "e29"))

import numpy as np  # noqa: E402

import e29_cluster as e29  # noqa: E402
from e28_gate2_twin import cross, set_pure_legs  # noqa: E402
from e28_read768 import REL_GATE, across_sigma_table, r_cos, stats  # noqa: E402
from project.sigma_lowres.paper_bench.experiments.e24.e24_axis import (  # noqa: E402
    build_cond,
    validate_synthetic,
)
from project.sigma_lowres.paper_bench.vector_ledger import (  # noqa: E402
    Sums,
    assert_same_family,
    bc_ledger,
)

SL = REPO_ROOT / "project/sigma_lowres"
FROZEN = SL / "paper_bench/arm_sums/20260810-2301-e28f1-cond0433-768"
TWIN = SL / "paper_bench/arm_sums/20260811-0247-e28f1-native-twin-768"
PIN = 0.4333  # rounded center; manifest literal asserted below
PIN_LITERAL = 0.43333333333333335
GATE_COS = 0.95
TWIN_REPRO_TOL = 0.05
FAMILY_B_EPOCH = 1786271541


def verdict_from(res: dict) -> str:
    if res["k_star"] == 1:
        return "NOT-REPLICATED"
    if res["k_star"] != 2:
        return "OTHER-STRUCTURE"
    first = len(res["partition"][0])
    if first in (1, 2):  # cut s0.3|s0.4333 or s0.4333|s0.5667 — adjacent to the pin
        return "MISMATCH-CARRIED"
    if first == 3:  # cut s0.5667|s0.7 — the e28 boundary, pin moved 2 bins away
        return "INTRINSIC-BLOCKS"
    return "OTHER-STRUCTURE"


def main() -> None:
    t0 = time.time()
    e28_json = json.loads((HERE.parent / "e28" / "e28_read768.json").read_text())
    e26_json = json.loads(e29.E26_JSON.read_text())

    # ---- gate 1: E29 instrument gates rerun
    gates, _ = e29.run_gates(e28_json)
    g5, g5_dev = e29.gate5_provenance(e26_json, e28_json)
    gates["gate5_provenance"] = g5
    fails = [k for k, v in gates.items() if v == "FAIL"]
    assert not fails, f"E29 instrument gates failed: {fails}"
    print("[gate1] E29 instrument gates PASS (incl. provenance)")

    # ---- gate 2a: E24 synthetic suite
    validate_synthetic()
    print("[gate2a] E24 synthetic suite clean")

    sf, st = Sums(FROZEN), Sums(TWIN)
    assert sf.man.get("cond_sigma") == PIN_LITERAL, sf.man.get("cond_sigma")
    assert st.man.get("cond_sigma") is None
    assert sf.man["seed"] == st.man["seed"] == 42
    centers = [round(float(v), 4) for v in sf.man["sigma_centers"]]
    assert centers == [round(float(v), 4) for v in st.man["sigma_centers"]]
    assert PIN in centers

    # ---- gate 3: same boot family (T0), family-B identity recorded
    assert_same_family(sf, st)
    same_family_b = all(
        s.fingerprint["boot_epoch"] == FAMILY_B_EPOCH for s in (sf, st)
    )
    print(f"[gate3] same-family PASS (family B: {same_family_b})")

    # ---- one pass per bin: gate 2b (bc_ledger reproduction, both stores),
    # conds for the across-sigma tables, set-pure cross-store rows
    conds_f, conds_t, scal_f, scal_t, cross_rows = [], [], [], [], []
    for bi, sig in enumerate(centers):
        for s, name, conds, scal in (
            (sf, "f1frozen", conds_f, scal_f),
            (st, "f1twin", conds_t, scal_t),
        ):
            led = bc_ledger(s, "768", bi, "reenc")
            c = build_cond(s, name, "768", bi)
            for k, v in (
                ("S", c.S),
                ("F", c.F),
                ("I", c.I),
                ("rho", c.rho),
                ("rel_cos_B", c.rel_B),
                ("rel_cos_C", c.rel_C),
            ):
                assert abs(round(v, 5) - led[k]) <= 1e-5, (name, sig, k, v, led[k])
            c.ghat32 = None
            conds.append(c)
            scal.append(
                {
                    "sigma": sig,
                    "rho": round(c.rho, 4),
                    "rel_B": round(c.rel_B, 4),
                    "rel_C": round(c.rel_C, 4),
                }
            )
        lf, lt = set_pure_legs(sf, bi), set_pure_legs(st, bi)
        cross_rows.append(
            {
                "sigma": sig,
                "cos_B": round(cross(lf, lt, "B1", "B2"), 5),
                "cos_C": round(cross(lf, lt, "C1", "C2"), 5),
                "ghat_cos": round(cross(lf, lt, "a", "b"), 5),
            }
        )
        del lf, lt
        sf.drop_bin(bi)
        st.drop_bin(bi)
    print("[gate2b] bc_ledger reproduced on all 5 bins of both stores")

    # ---- gate 4: pin-bin consistency (the relocated gate-2 analog)
    pin_row = next(r for r in cross_rows if r["sigma"] == PIN)
    g4 = abs(pin_row["cos_B"]) >= GATE_COS and abs(pin_row["cos_C"]) >= GATE_COS
    print(
        f"[gate4] pin bin s{PIN}: B {pin_row['cos_B']:+.4f} C {pin_row['cos_C']:+.4f} "
        f"ghat {pin_row['ghat_cos']:+.4f} vs >= {GATE_COS} -> {'PASS' if g4 else 'FAIL'}"
    )
    assert g4, "gate 4 FAIL — instrument break; nothing else is read"

    # ---- gate 5: new twin reproduces the committed twin B table (family B)
    twin_b = across_sigma_table(conds_t, "B")
    committed = {r["pair"]: r["cos"] for r in e28_json["twin_across_sigma_B"]["pairs"]}
    devs = {
        r["pair"]: round(abs(r["cos"] - committed[r["pair"]]), 4)
        for r in twin_b
        if r["pair"] in committed
    }
    max_dev = max(devs.values()) if devs else None
    g5b = same_family_b and max_dev is not None and max_dev <= TWIN_REPRO_TOL
    print(f"[gate5] twin-table reproduction max |Δ| = {max_dev} -> "
          f"{'PASS' if g5b else ('CONTEXT (not family B)' if not same_family_b else 'FAIL')}")
    if same_family_b:
        assert g5b, "gate 5 FAIL"

    # ---- F1-A: readability (off-pin bins, both legs)
    off = [c for c in conds_f if c.sigma != PIN]
    a_fails = [c.sigma for c in off if c.rel_B < REL_GATE or c.rel_C < REL_GATE]
    a_verdict = (
        "INCONCLUSIVE-OFF-MANIFOLD" if len(a_fails) * 2 > len(off) else "READABLE"
    )
    print(f"[F1-A] off-pin rel-gate failures: {len(a_fails)}/{len(off)} "
          f"{a_fails} -> {a_verdict}")

    # ---- F1-B: the discriminator (frozen store's own R-hat table)
    r_frozen = [
        {"pair": f"s{x.sigma:g}~s{y.sigma:g}", "cos": round(r_cos(x, y), 4)}
        for x, y in combinations(conds_f, 2)
        if min(x.rel_B, x.rel_C, y.rel_B, y.rel_C) >= REL_GATE
    ]
    if a_verdict != "READABLE":
        b_verdict, b_res = "INCONCLUSIVE-OFF-MANIFOLD", None
    elif len(r_frozen) < 10:
        b_verdict, b_res = "TABLE-INCOMPLETE", None
    else:
        b_res = e29.classify(e29.to_mat(r_frozen))
        b_verdict = verdict_from(b_res)
    if b_res:
        print(f"[F1-B] R k*={b_res['k_star']} partition={b_res['partition']} "
              f"gap12={b_res['gap12']} margin={b_res['separation_margin']}")
    print(f"[F1-B] verdict: {b_verdict}")

    # ---- F1-C descriptive rows
    fb = across_sigma_table(conds_f, "B")
    fc = across_sigma_table(conds_f, "C")
    c_cls = {}
    for name, pairs in (("B_frozen", fb), ("C_frozen", fc)):
        c_cls[name] = e29.classify(e29.to_mat(pairs)) if len(pairs) == 10 else None
    tmap = {r["pair"]: r["cos"] for r in twin_b}
    deltas = [
        {"pair": r["pair"], "d": round(abs(r["cos"]) - abs(tmap[r["pair"]]), 4)}
        for r in fb
        if r["pair"] in tmap
    ]
    med_absdelta = (
        round(float(np.median([abs(d["d"]) for d in deltas])), 4) if deltas else None
    )
    for r in cross_rows:
        tag = " <-- pin" if r["sigma"] == PIN else ""
        print(f"[F1-C(ii)] s{r['sigma']:<6} cos_B {r['cos_B']:+.4f} "
              f"cos_C {r['cos_C']:+.4f} ghat {r['ghat_cos']:+.4f}{tag}")
    print(f"[F1-C(iii)] matched-pair median |Δcos| (B vs twin): {med_absdelta}")

    record = {
        "generated_by": "e28f1_read.py",
        "frozen_store": FROZEN.name,
        "twin_store": TWIN.name,
        "pin": PIN_LITERAL,
        "boot_family_B": same_family_b,
        "gates": {
            "g1_e29_instrument": "PASS",
            "g2_synthetic_and_ledger": "PASS",
            "g3_same_family": "PASS",
            "g4_pin_bin": pin_row,
            "g5_twin_reproduction": {"max_dev": max_dev, "pairs": devs},
        },
        "verdict_F1A": a_verdict,
        "verdict_F1B": b_verdict,
        "F1B_cluster": b_res,
        "frozen_across_sigma_R": {"stats": stats(r_frozen), "pairs": r_frozen},
        "frozen_across_sigma_B": {"stats": stats(fb), "pairs": fb},
        "frozen_across_sigma_C": {"stats": stats(fc), "pairs": fc},
        "twin_across_sigma_B": {"stats": stats(twin_b), "pairs": twin_b},
        "c_rows": {
            "cluster_B_frozen": c_cls["B_frozen"],
            "cluster_C_frozen": c_cls["C_frozen"],
            "matched_sigma_cross": cross_rows,
            "matched_pair_deltas": {"median_absdelta": med_absdelta, "pairs": deltas},
            "scalars_frozen": scal_f,
            "scalars_twin": scal_t,
        },
    }
    (HERE / "e28f1_read.json").write_text(json.dumps(record, indent=1))
    print(f"[done] F1-A {a_verdict}; F1-B {b_verdict}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
