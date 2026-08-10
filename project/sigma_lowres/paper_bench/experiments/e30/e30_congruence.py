"""E30.0 — rotation-profile congruence (zero GPU, prior tier, descriptive only).

Matched-pair Δcos between adapters' across-σ profiles (frame-safe
scalars) from the committed e26 tables. No verdict weight — inputs were
public (e29_read.json) before the E30 registration; see README honesty
note. Output: e30_congruence.json.
"""

import itertools
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
E26_JSON = HERE.parent / "e26" / "e26_grid_across_sigma.json"

ADAPTERS = ["sincos_twin", "flat", "dirty"]
LEGS = ["R", "B", "C"]


def table(e26, adapter, leg):
    return {
        p["pair"]: p["cos"]
        for p in e26["adapters"][adapter][f"across_sigma_{leg}"]["pairs"]
    }


def main():
    e26 = json.loads(E26_JSON.read_text())
    out = {
        "generated_by": "e30_congruence.py",
        "status": "descriptive only — prior tier, no verdict weight (README)",
        "truncation_cell_rule": "pairs with |cos| > 1 in either adapter excluded from medians, listed",
        "legs": {},
    }
    for leg in LEGS:
        tabs = {a: table(e26, a, leg) for a in ADAPTERS}
        trunc = {
            a: sorted(k for k, v in tabs[a].items() if abs(v) > 1.0)
            for a in ADAPTERS
        }
        rows = {}
        for a, b in itertools.combinations(ADAPTERS, 2):
            excluded = sorted(set(trunc[a]) | set(trunc[b]))
            deltas = {
                k: round(tabs[a][k] - tabs[b][k], 5)
                for k in tabs[a]
                if k not in excluded
            }
            absd = [abs(v) for v in deltas.values()]
            rows[f"{a}~{b}"] = {
                "n_pairs_used": len(deltas),
                "excluded_truncation_cells": excluded,
                "median_absdelta": round(statistics.median(absd), 5),
                "max_absdelta": round(max(absd), 5),
                "deltas": deltas,
            }
        out["legs"][leg] = {
            "truncation_cells_per_adapter": {
                a: v for a, v in trunc.items() if v
            },
            "adapter_pairs": rows,
        }
    (HERE / "e30_congruence.json").write_text(json.dumps(out, indent=1))
    for leg in LEGS:
        print(f"-- {leg} --")
        for pair, r in out["legs"][leg]["adapter_pairs"].items():
            print(
                f"  {pair}: median|Δ|={r['median_absdelta']} "
                f"max|Δ|={r['max_absdelta']} n={r['n_pairs_used']} "
                f"excl={r['excluded_truncation_cells']}"
            )


if __name__ == "__main__":
    main()
