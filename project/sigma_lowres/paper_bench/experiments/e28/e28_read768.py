#!/usr/bin/env python
"""sigma_lowres E28 — the pre-registered read, RESTRICTED TO THE 768 ROUTE
(recorded scope amendment: the 896 stage is unrun; the verdict below is
scoped "768 route, one operating point" and the frozen thresholds are
applied unchanged).

Precedence per the registration: 28-A (readability) -> 28-B
(CONDITIONING-CARRIED vs STATISTICS-CARRIED vs PARTIAL) -> 28-C
descriptive rows.

Native reference: per the 2026-08-10 environment amendment, the
same-environment seed-twin store (`e28-native-twin-768`) supplies the
native across-sigma table; the committed E24 family stats (median 0.791,
min 0.442, both routes) are quoted as context only. Within-store
across-sigma pairs use the E24 estimand verbatim (build_cond legs,
cross_cos debias, rel gate 0.5 per leg); bins occupy disjoint draw
blocks, so within-store cross-bin draw noise is independent in both
stores.

Usage::

    uv run python project/sigma_lowres/paper_bench/experiments/e28/e28_read768.py
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
from project.sigma_lowres.paper_bench.vector_ledger import Sums, bc_ledger  # noqa: E402

SL = REPO_ROOT / "project/sigma_lowres"
FROZEN = SL / "bench/results/20260809-2216-e28-cond07-768/arm_sums"
TWIN = SL / "bench/results/20260810-0658-e28-native-twin-768/arm_sums"
SIGMA_COND = 0.7
REL_GATE = 0.5
E24_NATIVE_CONTEXT = {"median_abs": 0.791, "min_abs": 0.442}  # committed, both routes


def load_store(root: Path, name: str) -> tuple[list, list[dict]]:
    """build_cond per bin (route 768) + bc_ledger scalar rows."""
    s = Sums(root)
    centers = [round(float(v), 4) for v in s.man["sigma_centers"]]
    conds, scalars = [], []
    for bi, sig in enumerate(centers):
        led = bc_ledger(s, "768", bi, "reenc")
        c = build_cond(s, name, "768", bi)
        c.ghat32 = None  # not needed downstream; free ~300 MB/bin
        conds.append(c)
        scalars.append(
            {
                "sigma": sig,
                "rho": led["rho"],
                "h_B": led.get("h_B"),
                "h_C": led.get("h_C"),
                "h_BC": led.get("h_B_plus_C"),
                "rel_B": led["rel_cos_B"],
                "rel_C": led["rel_cos_C"],
            }
        )
        s.drop_bin(bi)
    del s
    return conds, scalars


def r_leg(c) -> np.ndarray:
    return c.B32 + c.C32


def r_norm2(c) -> float:
    # debiased residual power nR^2 = 2 G^2 (S + F + I)  (E25.0 identity)
    return max(2.0 * c.G * c.G * (c.S + c.F + c.I), 0.0)


def r_cos(x, y) -> float:
    num = float(r_leg(x).astype(np.float64) @ r_leg(y).astype(np.float64))
    den = math.sqrt(r_norm2(x) * r_norm2(y))
    return num / den if den > 0 else float("nan")


def across_sigma_table(conds: list, leg: str) -> list[dict]:
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


def stats(rows: list[dict]) -> dict:
    av = [abs(r["cos"]) for r in rows]
    return {
        "n_pairs": len(rows),
        "median_abs": round(float(np.median(av)), 4) if av else None,
        "min_abs": round(min(av), 4) if av else None,
    }


def main() -> None:
    t0 = time.time()
    frozen, f_scal = load_store(FROZEN, "e28")
    twin, t_scal = load_store(TWIN, "twin")

    # ---- 28-A: readability gate (off-sigma_cond conditions, both legs)
    off = [c for c in frozen if c.sigma != SIGMA_COND]
    fails = [c for c in off if c.rel_B < REL_GATE or c.rel_C < REL_GATE]
    a_verdict = "INCONCLUSIVE-OFF-MANIFOLD" if len(fails) * 2 > len(off) else "READABLE"
    print(f"[28-A] off-cond rel-gate failures: {len(fails)}/{len(off)} -> {a_verdict}")

    # ---- 28-B: the discriminator (B table; C reported)
    fb = across_sigma_table(frozen, "B")
    fc = across_sigma_table(frozen, "C")
    tb = across_sigma_table(twin, "B")
    fb_stats, tb_stats = stats(fb), stats(tb)
    shared_axis = (
        fb_stats["median_abs"] is not None
        and fb_stats["median_abs"] >= 0.7
        and fb_stats["min_abs"] >= 0.5
    )
    tmap = {r["pair"]: r["cos"] for r in tb}
    deltas = [
        {"pair": r["pair"], "d": round(abs(r["cos"]) - abs(tmap[r["pair"]]), 4)}
        for r in fb
        if r["pair"] in tmap
    ]
    med_absdelta = (
        round(float(np.median([abs(d["d"]) for d in deltas])), 4) if deltas else None
    )
    if a_verdict != "READABLE":
        b_verdict = "INCONCLUSIVE-OFF-MANIFOLD"
    elif shared_axis:
        b_verdict = "CONDITIONING-CARRIED"
    elif (
        med_absdelta is not None and med_absdelta <= 0.05 and fb_stats["min_abs"] < 0.5
    ):
        b_verdict = "STATISTICS-CARRIED"
    else:
        b_verdict = "PARTIAL"

    print(
        f"[28-B] frozen across-sigma B: {fb_stats} (twin native: {tb_stats}; "
        f"E24 committed context: {E24_NATIVE_CONTEXT})"
    )
    for r in fb:
        print(
            f"        {r['pair']:16s} frozen {r['cos']:+.4f}   "
            f"native(twin) {tmap.get(r['pair'], float('nan')):+.4f}"
        )
    print(f"[28-B] median |Δcos| vs twin table: {med_absdelta} -> {b_verdict}")

    # ---- 28-C descriptive rows
    r_frozen = [
        {"pair": f"s{x.sigma:g}~s{y.sigma:g}", "cos": round(r_cos(x, y), 4)}
        for x, y in combinations(frozen, 2)
        if min(x.rel_B, x.rel_C, y.rel_B, y.rel_C) >= REL_GATE
    ]
    print("[28-C(i)] per-bin scalars (frozen | twin):")
    for f, t in zip(f_scal, t_scal):
        print(
            f"        s{f['sigma']:<7} rho {f['rho']:+.3f}|{t['rho']:+.3f}  "
            f"relB {f['rel_B']:.2f}|{t['rel_B']:.2f}  "
            f"relC {f['rel_C']:.2f}|{t['rel_C']:.2f}"
        )

    record = {
        "generated_by": "e28_read768.py",
        "scope": "768 route only (896 unrun) — recorded restriction",
        "native_reference": "same-environment seed-twin (2026-08-10 amendment)",
        "verdict_28A": a_verdict,
        "verdict_28B": b_verdict,
        "frozen_across_sigma_B": {"stats": fb_stats, "pairs": fb},
        "frozen_across_sigma_C": {"stats": stats(fc), "pairs": fc},
        "twin_across_sigma_B": {"stats": tb_stats, "pairs": tb},
        "matched_pair_deltas": {"median_absdelta": med_absdelta, "pairs": deltas},
        "e24_native_context": E24_NATIVE_CONTEXT,
        "c_rows": {
            "scalars_frozen": f_scal,
            "scalars_twin": t_scal,
            "R_across_sigma_frozen": r_frozen,
        },
    }
    (HERE / "e28_read768.json").write_text(json.dumps(record, indent=1))
    print(f"[done] 28-A {a_verdict}; 28-B {b_verdict}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
