#!/usr/bin/env python
"""sigma_lowres E26 — across-sigma table extraction (pre-reclamation commit).

2026-08-10: the raw arm_sums stores are being reclaimed for disk (roadmap
§3 storage economics, user decision). This script commits every vector
table the pending CPU analyses (roadmap §1(c) native-block clustering)
could need from the E26 grid family, so the reclamation loses nothing:

  per adapter (flat, dirty, sincos_twin), route 768, 5 verdict bins:
    across_sigma_B / across_sigma_C  — E24 estimand verbatim
      (build_cond legs, cross_cos debias, rel gate 0.5 per leg)
    across_sigma_R  — residual-leg R = B+C cosine, e28_read768 r_cos
      verbatim (all-legs rel gate 0.5)

NO verdict is applied and NO table values are printed: item (c)'s
clustering criterion is not yet frozen, so this run is a mechanical
commit only (stats + pair counts to stdout, values to JSON). The twin's
B-leg recomputation is checked programmatically against the table
already committed in e28_read768.json (PASS/FAIL printed, values not).

Usage::

    uv run python project/sigma_lowres/paper_bench/experiments/e26/e26_grid_across_sigma.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from project.sigma_lowres.paper_bench.experiments.e24.e24_axis import (  # noqa: E402
    build_cond,
    cross_cos,
)
from project.sigma_lowres.paper_bench.vector_ledger import Sums  # noqa: E402

SL = REPO_ROOT / "project/sigma_lowres"
STORES = {
    "flat": SL / "bench/results/20260810-1114-e26-grid-flat/arm_sums",
    "dirty": SL / "bench/results/20260810-1519-e26-grid-dirty/arm_sums",
    "sincos_twin": SL / "bench/results/20260810-0658-e28-native-twin-768/arm_sums",
}
ROUTE = "768"
REL_GATE = 0.5


def load_conds(root: Path, name: str) -> list:
    s = Sums(root)
    conds = []
    for bi in range(len(s.man["sigma_centers"])):
        c = build_cond(s, name, ROUTE, bi)
        c.ghat32 = None  # not needed downstream; free ~300 MB/bin
        conds.append(c)
        s.drop_bin(bi)
    del s
    return conds


def r_cos(x, y) -> float:
    # e28_read768 verbatim: residual leg R = B+C, debiased norm from the
    # E25.0 identity nR^2 = 2 G^2 (S + F + I)
    num = float((x.B32 + x.C32).astype(np.float64) @ (y.B32 + y.C32).astype(np.float64))
    nx2 = max(2.0 * x.G * x.G * (x.S + x.F + x.I), 0.0)
    ny2 = max(2.0 * y.G * y.G * (y.S + y.F + y.I), 0.0)
    den = math.sqrt(nx2 * ny2)
    return num / den if den > 0 else float("nan")


def leg_table(conds: list, leg: str) -> list[dict]:
    rows = []
    for x, y in combinations(conds, 2):
        gx = (x.rel_B if leg == "B" else x.rel_C) >= REL_GATE
        gy = (y.rel_B if leg == "B" else y.rel_C) >= REL_GATE
        if not (gx and gy):
            continue
        rows.append(
            {
                "pair": f"s{x.sigma:g}~s{y.sigma:g}",
                "cos": round(cross_cos(x, y, leg), 5),
            }
        )
    return rows


def r_table(conds: list) -> list[dict]:
    return [
        {"pair": f"s{x.sigma:g}~s{y.sigma:g}", "cos": round(r_cos(x, y), 5)}
        for x, y in combinations(conds, 2)
        if min(x.rel_B, x.rel_C, y.rel_B, y.rel_C) >= REL_GATE
    ]


def stats(rows: list[dict]) -> dict:
    av = [abs(r["cos"]) for r in rows]
    return {
        "n_pairs": len(rows),
        "median_abs": round(float(np.median(av)), 4) if av else None,
        "min_abs": round(min(av), 4) if av else None,
    }


def main() -> None:
    t0 = time.time()
    out = {
        "doc": "E26 grid-family across-sigma tables, committed before arm_sums "
        "reclamation (2026-08-10). Estimand: E24 build_cond/cross_cos verbatim; "
        "R leg per e28_read768 r_cos. No verdict applied — roadmap 1(c) input only.",
        "route": ROUTE,
        "rel_gate": REL_GATE,
        "stores": {k: str(v.parent.relative_to(SL)) for k, v in STORES.items()},
        "adapters": {},
    }
    for name, root in STORES.items():
        conds = load_conds(root, name)
        tables = {
            "across_sigma_B": leg_table(conds, "B"),
            "across_sigma_C": leg_table(conds, "C"),
            "across_sigma_R": r_table(conds),
        }
        out["adapters"][name] = {
            k: {"stats": stats(rows), "pairs": rows} for k, rows in tables.items()
        }
        counts = {k: len(rows) for k, rows in tables.items()}
        print(f"[{name}] pair counts {counts}  ({time.time() - t0:.0f}s)")
        del conds

    # consistency check: recomputed twin B-leg vs the committed e28 table
    committed = json.loads((HERE.parent / "e28" / "e28_read768.json").read_text())
    ref = {r["pair"]: r["cos"] for r in committed["twin_across_sigma_B"]["pairs"]}
    got = {
        r["pair"]: r["cos"]
        for r in out["adapters"]["sincos_twin"]["across_sigma_B"]["pairs"]
    }
    ok = set(ref) == set(got) and all(abs(ref[p] - got[p]) < 1e-4 for p in ref)
    print(
        f"[check] twin B-leg vs e28_read768.json committed table: "
        f"{'PASS' if ok else 'FAIL'} ({len(ref)} pairs)"
    )
    if not ok:
        raise SystemExit("consistency check FAILED — do not reclaim stores")

    out_path = HERE / "e26_grid_across_sigma.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
