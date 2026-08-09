#!/usr/bin/env python
"""sigma_lowres E28 — gate-2 failure diagnosis (CPU).

The gate-2 FAIL (e28_gate2.json) has two candidate causes; a same-boot
single-bin NATIVE run at the 0.7 bin (20260810-0214-e28-g2diag-native07,
same kernel cache as the e28 store, no --cond_sigma) separates them:

 * diag <-> e193/e194  isolates the KERNEL-PATH effect (native vs native,
   different boots; the deterministic flag does not pin the kernel set).
 * diag <-> e28@0.7    isolates the CONDITIONING effect (same boot, same
   kernels; frozen vs native conditioning; draw seeds independent — e28's
   0.7 bin sits at draw offsets 36-47, the diag bin at 0-11).

Shared-noise bookkeeping per pair (seed 42, probe list, arm layouts):
 * diag <-> e193: reenc both at arm_idx 2 but different draw offsets ->
   independent; a/b likewise -> independent. No corrections.
 * diag <-> e194: a/b AND reenc share seeds (draw offsets 0-11, same
   arm_idx) -> ghat cosine read as inflated context; B-leg gets the
   shared-ref correction <d_perp_x, d_perp_y>/4.
 * diag <-> e28: all arms draw-independent -> no corrections.

Usage::

    uv run python project/sigma_lowres/paper_bench/experiments/e28/e28_gate2_diag.py
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
)
from project.sigma_lowres.paper_bench.experiments.e28.e28_gate2 import (  # noqa: E402
    perp_refdiff,
)
from project.sigma_lowres.paper_bench.vector_ledger import Sums  # noqa: E402

SL = REPO_ROOT / "project/sigma_lowres"
STORES = {
    "diag": SL / "bench/results/20260810-0214-e28-g2diag-native07/arm_sums",
    "e28": SL / "bench/results/20260809-2216-e28-cond07-768/arm_sums",
    "e193": SL / "bench/results/20260807-0745-e193-depth-ledger/arm_sums",
    "e194": SL / "bench/results/20260807-1400-e194-pi-causal/arm_sums",
}
REF_SHARED = {("diag", "e194"), ("e28", "e193")}  # pairs sharing reenc noise


def main() -> None:
    t0 = time.time()
    conds: dict[str, object] = {}
    refd: dict[str, np.ndarray] = {}
    for name, root in STORES.items():
        s = Sums(root)
        centers = [round(float(v), 4) for v in s.man["sigma_centers"]]
        bi = centers.index(0.7)
        conds[name] = build_cond(s, name, "768", bi)
        refd[name] = perp_refdiff(s, bi)
        s.drop_bin(bi)
        del s
        c = conds[name]
        print(
            f"[cond] {name:5s} rho={c.rho:+.3f} relB={c.rel_B:.3f} "
            f"relC={c.rel_C:.3f} G={c.G:.3f}"
        )

    record: dict = {"generated_by": "e28_gate2_diag.py", "pairs": {}}
    for x, y in (
        ("diag", "e193"),
        ("diag", "e194"),
        ("diag", "e28"),
        ("e28", "e193"),
        ("e28", "e194"),
    ):
        cx, cy = conds[x], conds[y]
        cos_b = cross_cos(cx, cy, "B")
        shared = 0.0
        if (x, y) in REF_SHARED or (y, x) in REF_SHARED:
            num = float(refd[x].astype(np.float64) @ refd[y].astype(np.float64)) / 4.0
            den = float(np.sqrt(cx.nB2 * cy.nB2))
            shared = num / den if den > 0 else 0.0
        cos_c = cross_cos(cx, cy, "C")
        ghat = float(cx.ghat32.astype(np.float64) @ cy.ghat32.astype(np.float64))
        record["pairs"][f"{x}~{y}"] = {
            "cos_B": round(cos_b - shared, 5),
            "shared_ref_num": round(shared, 6),
            "cos_C": round(cos_c, 5),
            "ghat_cos": round(ghat, 5),
        }
        print(
            f"[pair] {x:5s} <-> {y:5s}  B {cos_b - shared:+.4f} "
            f"(shared {shared:+.5f})  C {cos_c:+.4f}  ghat {ghat:+.5f}"
        )

    (HERE / "e28_gate2_diag.json").write_text(json.dumps(record, indent=1))
    print(f"[done] ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
