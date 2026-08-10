#!/usr/bin/env python
"""sigma_lowres E28 — gate-2 re-evaluation against the seed-twin native
reference (CPU), per the amendment recorded in README (768-stage record).

Reference: `20260810-0658-e28-native-twin-768` — identical grid, seed,
probe list, arm layout, and boot as the frozen-conditioning store; only
`--cond_sigma` differs. Same boot verified (uptime 2026-08-09 19:32
spans both runs), so the kernel path is held fixed.

Estimand — cross-set debiased cosine. Seed-twinning shares draw noise
between same-numbered draw sets across the runs, so the E24 mean-leg
estimand would be noise-inflated. Instead, legs are SET-PURE (the E23.0
convention: B_s = P_perp_ghat(rp_s - re_s), C_s = P_perp_ghat(dem_s -
rp_s), each set's own reenc reference), and every cross-run product
pairs set 1 with set 2 (primary and alt seed blocks are disjoint by
construction), restoring exact noise independence:

    cos_leg(x, y) = [<L1_x, L2_y> + <L2_x, L1_y>] / 2
                    / sqrt(<L1_x, L2_x> * <L1_y, L2_y>)

with each condition's legs perp'd against its own run's ghat (E24
convention). The anchor read is the same shape on the native a/b sets:
ghat_cos = [<a_x, b_y> + <b_x, a_y>]/2 / sqrt(<a_x, b_x><a_y, b_y>).

Gate (frozen threshold, amended reference): at noising sigma = 0.7,
|cos_B| >= 0.95 and |cos_C| >= 0.95. Other bins are reported as the
pre-registered 28-C(ii) descriptive rows (matched-sigma distance of the
frozen field from native); the 28-B across-sigma verdict table is NOT
computed here (protocol order: staging first).

Usage::

    uv run python project/sigma_lowres/paper_bench/experiments/e28/e28_gate2_twin.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from project.sigma_lowres.paper_bench.vector_ledger import Sums  # noqa: E402

SL = REPO_ROOT / "project/sigma_lowres"
FROZEN = SL / "bench/results/20260809-2216-e28-cond07-768/arm_sums"
TWIN = SL / "bench/results/20260810-0658-e28-native-twin-768/arm_sums"
GATE_SIGMA = 0.7
GATE_COS = 0.95


def dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(a.astype(np.float64) @ b.astype(np.float64))


def set_pure_legs(s: Sums, bi: int) -> dict:
    """Per-set perp'd legs + native sets for one (run, bin), fp32."""
    a, b = s.mean("a", bi), s.mean("b", bi)
    g0 = 0.5 * (a + b)
    ghat = g0 / float(np.linalg.norm(g0))

    def perp(v):
        return (v - float(ghat @ v) * ghat).astype(np.float32)

    out = {"a": a.astype(np.float32), "b": b.astype(np.float32)}
    for si, sfx in ((1, ""), (2, "__2")):
        re = s.mean(f"reenc{sfx}", bi)
        rp = s.mean(f"768rp{sfx}", bi)
        dem = s.mean(f"768{sfx}", bi)
        out[f"B{si}"] = perp(rp - re)
        out[f"C{si}"] = perp(dem - rp)
    return out


def cross(x: dict, y: dict, k1: str, k2: str) -> float:
    """Cross-set debiased cosine between runs for a leg pair (k1, k2)."""
    num = 0.5 * (dot(x[k1], y[k2]) + dot(x[k2], y[k1]))
    nx = dot(x[k1], x[k2])
    ny = dot(y[k1], y[k2])
    den = math.sqrt(max(nx, 0.0) * max(ny, 0.0))
    return num / den if den > 0 else float("nan")


def main() -> None:
    t0 = time.time()
    sf, st = Sums(FROZEN), Sums(TWIN)
    assert sf.man.get("cond_sigma") == GATE_SIGMA
    assert st.man.get("cond_sigma") is None
    centers = [round(float(v), 4) for v in sf.man["sigma_centers"]]
    assert centers == [round(float(v), 4) for v in st.man["sigma_centers"]]
    assert sf.man["seed"] == st.man["seed"] == 42

    record: dict = {
        "generated_by": "e28_gate2_twin.py",
        "frozen_store": str(FROZEN.relative_to(REPO_ROOT)),
        "twin_store": str(TWIN.relative_to(REPO_ROOT)),
        "estimand": "cross-set debiased cosine, set-pure perp'd legs",
        "gate_sigma": GATE_SIGMA,
        "gate_cos": GATE_COS,
        "rows": [],
    }
    gate_row = None
    for bi, sig in enumerate(centers):
        lf, lt = set_pure_legs(sf, bi), set_pure_legs(st, bi)
        row = {
            "sigma": sig,
            "cos_B": round(cross(lf, lt, "B1", "B2"), 5),
            "cos_C": round(cross(lf, lt, "C1", "C2"), 5),
            "ghat_cos": round(cross(lf, lt, "a", "b"), 5),
            # within-run set-pure reliabilities (context, raw cosines)
            "relB_frozen": round(
                float(dot(lf["B1"], lf["B2"]))
                / float(np.linalg.norm(lf["B1"]) * np.linalg.norm(lf["B2"])),
                4,
            ),
            "relB_twin": round(
                float(dot(lt["B1"], lt["B2"]))
                / float(np.linalg.norm(lt["B1"]) * np.linalg.norm(lt["B2"])),
                4,
            ),
        }
        record["rows"].append(row)
        if sig == GATE_SIGMA:
            gate_row = row
        sf.drop_bin(bi)
        st.drop_bin(bi)
        tag = " <-- gate bin" if sig == GATE_SIGMA else ""
        print(
            f"[28-C(ii)] s{sig:<6} cos_B {row['cos_B']:+.4f} "
            f"cos_C {row['cos_C']:+.4f} ghat {row['ghat_cos']:+.4f}{tag}"
        )

    assert gate_row is not None
    ok = abs(gate_row["cos_B"]) >= GATE_COS and abs(gate_row["cos_C"]) >= GATE_COS
    record["verdict"] = "PASS" if ok else "FAIL"
    (HERE / "e28_gate2_twin.json").write_text(json.dumps(record, indent=1))
    print(
        f"[done] gate 2 (twin reference) {record['verdict']} — "
        f"B {gate_row['cos_B']:+.4f} C {gate_row['cos_C']:+.4f} vs >= {GATE_COS}"
        f"  ({time.time() - t0:.0f}s)"
    )


if __name__ == "__main__":
    main()
