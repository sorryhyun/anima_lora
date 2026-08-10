#!/usr/bin/env python
"""E28-F1-i — formal re-read of the committed E28 frozen-frame tables.

Applies the E29 frozen classifier (signed-cos contiguity clustering,
tau = 0.30, sequential elbow) to the committed `e28_read768.json`
tables, replacing the eyeballed pair-sign read as the formal statement
of the two-block object. Zero GPU; the E29 gates rerun first.

Registered expectation (public via E29 gate 3): R-hat k* = 2 at
{0.3, 0.4333, 0.5667} | {0.7, 0.8333}. Formalization weight only.

Usage::

    uv run python project/sigma_lowres/paper_bench/experiments/e28f1/e28f1_reread.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e29"))

import e29_cluster as e29  # noqa: E402

EXPECT_PARTITION = [e29.BINS[:3], e29.BINS[3:]]  # {0.3,0.4333,0.5667} | {0.7,0.8333}


def main() -> None:
    e28 = json.loads((HERE.parent / "e29").parent.joinpath("e28", "e28_read768.json").read_text())
    e26 = json.loads(e29.E26_JSON.read_text())

    gates, _ = e29.run_gates(e28)
    g5, g5_dev = e29.gate5_provenance(e26, e28)
    gates["gate5_provenance"] = g5
    gates["gate5_max_dev"] = g5_dev
    fails = [k for k, v in gates.items() if isinstance(v, str) and v == "FAIL"]
    assert not fails, f"E29 instrument gates failed: {fails}"
    print("[gates] E29 instrument gates 1-5 all PASS")

    tables = {
        "R_frozen": e28["c_rows"]["R_across_sigma_frozen"],
        "B_frozen": e28["frozen_across_sigma_B"]["pairs"],
        "C_frozen": e28["frozen_across_sigma_C"]["pairs"],
    }
    out = {}
    for name, pairs in tables.items():
        res = e29.classify(e29.to_mat(pairs))
        out[name] = res
        print(
            f"[{name}] k*={res['k_star']} partition={res['partition']} "
            f"gap12={res['gap12']} margin={res['separation_margin']}"
        )

    primary = out["R_frozen"]
    formal = primary["k_star"] == 2 and primary["partition"] == EXPECT_PARTITION
    verdict = (
        "TWO-BLOCK-FORMAL (boundary s0.5667|s0.7)" if formal else "MISMATCH-VS-E29-GATE3"
    )
    print(f"[F1-i] {verdict}")

    record = {
        "generated_by": "e28f1_reread.py",
        "role": "F1-i formalization (outcome public via E29 gate 3 — no discovery weight)",
        "instrument": "e29_cluster.py frozen classifier, tau=0.30, signed cos",
        "gates": {k: v for k, v in gates.items() if isinstance(v, (str, float, type(None)))},
        "verdict_F1i": verdict,
        "primary_R_frozen": out["R_frozen"],
        "descriptive": {k: v for k, v in out.items() if k != "R_frozen"},
    }
    (HERE / "e28f1_reread.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
