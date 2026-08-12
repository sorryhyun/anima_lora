#!/usr/bin/env python
"""sigma_lowres E25b Stage-1 read — gates + the 25b-1 discriminator.

Frozen registration (README 2026-08-11): two E16-pattern deterministic
seed-twins (ctrl = stock combo, rescond = combo + --sigma_lowres_res_cond,
seed 42) probed on the 768 route over the E26/E28 verdict grid.

Gates (all before the verdict row is read):
  1. Twin-start identity — recorded in-vivo evidence from the gate-1
     micro-runs (2026-08-12, jobs 20260812-005222/005223 + noclip pair +
     ctrl2 determinism control): first-forward loss bit-identical across
     ctrl / ctrl2 / rescond / {clip,noclip} variants (0.421300232410),
     ctrl vs ctrl2 full-run checkpoints 0/1092 keys mismatched. The
     zero-init invariant holds in vivo; later divergence is
     gradient-borne (the projection trains from step 1).
  2. Ledger reproduction — the independent bc_ledger path (vector_ledger,
     data_ref=reenc) on all 5 bins of both stores.
  3. Readability — E26 frozen readable criterion (rel_B ^ rel_C >= 0.5)
     on >= 4/5 verdict bins in BOTH stores; fewer =>
     INCONCLUSIVE-UNRELIABLE, nothing else is read.
  4. Same-family — assert_same_family on the probe pair (T0).

25b-1 (frozen constants 0.9 / 1.1, judgment values): per-bin paired
r_h = h(B+C)_rescond / h(B+C)_ctrl over co-readable bins, median;
Δρ = ρ_rescond − ρ_ctrl alongside.
  median r_h <= 0.9 AND median Δρ <= +0.02  -> IMPROVES
  median r_h >= 1.1 OR  median Δρ >= +0.05  -> WORSENS
  otherwise                                  -> NULL (scale-qualified)

Descriptives (no verdict weight): per-bin r_h / Δρ profiles; rel_R of
both stores (E25.0 hole context at σ = 0.4333); h(B)/h(C) per arm;
ctrl-vs-sincos scalar context (e30 run ledger; cross-boot Δρ caveat).

Usage::

    uv run python project/sigma_lowres/paper_bench/experiments/e25b/e25b_read.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from project.sigma_lowres.paper_bench.vector_ledger import (  # noqa: E402
    Sums,
    assert_same_family,
    bc_ledger,
    cosine,
    perp,
)

SL = REPO_ROOT / "project/sigma_lowres"
RESULTS = SL / "bench" / "results"
ROUTE = "768"
REL_GATE = 0.5
R_H_IMPROVE, R_H_WORSEN = 0.9, 1.1
DRHO_IMPROVE, DRHO_WORSEN = 0.02, 0.05

GATE1_EVIDENCE = {
    "first_forward_loss": "0.421300232410 bit-identical across ctrl, ctrl2, "
    "rescond, ctrl_noclip, rescond_noclip (TB loss/current step 1)",
    "determinism_control": "ctrl vs ctrl2 (identical argv): 0/1092 checkpoint "
    "keys mismatched, full run",
    "micro_jobs": [
        "20260812-005222-960b5d",
        "20260812-005223-093b78",
        "20260812-010513-b1afd3",
        "20260812-010513-62920b",
        "20260812-011749-3b1f7d",
    ],
    "note": "later checkpoint divergence (728/1092 keys) is the lever "
    "training — the projection receives nonzero gradient from step 1 "
    "(zero value, nonzero grad), so arms diverge from step 2 on; "
    "gradient-borne per the registration's premise.",
}


def newest(label: str) -> Path:
    hits = sorted(RESULTS.glob(f"*-{label}"))
    if not hits:
        raise SystemExit(f"no run dir matching *-{label} under {RESULTS}")
    return hits[-1]


def rel_R(s: Sums, bi: int) -> float:
    """Reliability of the residual leg R = dem − reenc̄ across draw sets,
    perp against the native ĝ (E25.0 hole context row)."""
    a, b = s.mean("a", bi), s.mean("b", bi)
    g0 = 0.5 * (a + b)
    ghat = g0 / np.linalg.norm(g0)
    ref = 0.5 * (s.mean("reenc", bi) + s.mean("reenc__2", bi))
    R1 = perp(s.mean(ROUTE, bi) - ref, ghat)
    R2 = perp(s.mean(f"{ROUTE}__2", bi) - ref, ghat)
    return round(cosine(R1, R2), 5)


def read_store(run_dir: Path) -> dict:
    s = Sums(run_dir / "arm_sums")
    n_bins = len(s.man["sigma_centers"])
    rows, relr = [], []
    for bi in range(n_bins):
        rows.append(bc_ledger(s, ROUTE, bi, "reenc"))
        relr.append(rel_R(s, bi))
        s.drop_bin(bi)
    return {
        "run": str(run_dir.relative_to(SL)),
        "sigmas": [round(x, 4) for x in s.man["sigma_centers"]],
        "res_cond": bool(s.man.get("res_cond")),
        "rows": rows,
        "rel_R": relr,
        "_sums": s,
    }


def main() -> None:
    ctrl = read_store(newest("e25b-ctrl-768"))
    resc = read_store(newest("e25b-rescond-768"))
    assert resc["res_cond"] and not ctrl["res_cond"], (
        "store/arm mismatch: rescond store must carry res_cond=true, ctrl must not"
    )
    assert_same_family(ctrl["_sums"], resc["_sums"])  # gate 4 (raises)

    out: dict = {
        "doc": "E25b Stage-1 read — frozen 2026-08-11 registration",
        "route": ROUTE,
        "stores": {"ctrl": ctrl["run"], "rescond": resc["run"]},
        "gate1_twin_start_identity": {"pass": True, **GATE1_EVIDENCE},
        "gate4_same_family": "pass (assert_same_family)",
    }

    # gate 2/3: independent ledger on all bins + readability, both stores
    sig = ctrl["sigmas"]
    readable = {}
    for name, st in (("ctrl", ctrl), ("rescond", resc)):
        rd = [
            r["rel_cos_B"] >= REL_GATE and r["rel_cos_C"] >= REL_GATE
            for r in st["rows"]
        ]
        readable[name] = rd
        out[f"ledger_{name}"] = [
            {
                "sigma": sig[i],
                "readable": rd[i],
                "rel_R": st["rel_R"][i],
                **{
                    k: r[k]
                    for k in (
                        "G",
                        "S",
                        "F",
                        "I",
                        "rho",
                        "rel_cos_B",
                        "rel_cos_C",
                        "h_B",
                        "h_C",
                        "h_B_plus_C",
                    )
                },
            }
            for i, r in enumerate(st["rows"])
        ]
    n_read = {k: sum(v) for k, v in readable.items()}
    out["gate3_readability"] = {
        **n_read,
        "pass": all(n >= 4 for n in n_read.values()),
    }
    if not out["gate3_readability"]["pass"]:
        out["verdict"] = "INCONCLUSIVE-UNRELIABLE"
        _finish(out)
        return

    # 25b-1 — the discriminator
    co = [i for i in range(len(sig)) if readable["ctrl"][i] and readable["rescond"][i]]
    per_bin = [
        {
            "sigma": sig[i],
            "r_h": round(
                resc["rows"][i]["h_B_plus_C"] / ctrl["rows"][i]["h_B_plus_C"], 4
            ),
            "d_rho": round(resc["rows"][i]["rho"] - ctrl["rows"][i]["rho"], 4),
            "h_ctrl": ctrl["rows"][i]["h_B_plus_C"],
            "h_rescond": resc["rows"][i]["h_B_plus_C"],
        }
        for i in co
    ]
    med_rh = float(np.median([b["r_h"] for b in per_bin]))
    med_drho = float(np.median([b["d_rho"] for b in per_bin]))
    if med_rh <= R_H_IMPROVE and med_drho <= DRHO_IMPROVE:
        verdict = "IMPROVES"
    elif med_rh >= R_H_WORSEN or med_drho >= DRHO_WORSEN:
        verdict = "WORSENS"
    else:
        verdict = "NULL"
    out["reading_25b1"] = {
        "co_readable_bins": [sig[i] for i in co],
        "per_bin": per_bin,
        "median_r_h": round(med_rh, 4),
        "median_d_rho": round(med_drho, 4),
        "thresholds": {
            "improve": f"r_h <= {R_H_IMPROVE} AND d_rho <= +{DRHO_IMPROVE}",
            "worsen": f"r_h >= {R_H_WORSEN} OR d_rho >= +{DRHO_WORSEN}",
        },
        "verdict": verdict,
    }

    # descriptives (no verdict weight)
    e30_ledger = SL / "bench/results/20260811-0821-e30-basesketch-768/ledger.json"
    sincos_rows = None
    if e30_ledger.exists():
        led = json.loads(e30_ledger.read_text())
        sincos_rows = [
            {"sigma": round(sg, 4), "rho": r["rho"], "h_B_plus_C": r["h_B_plus_C"]}
            for sg, r in zip(led["sigma_centers"], led["bc_ledger"][ROUTE])
        ]
    out["descriptive"] = {
        "leg_ratios": [
            {
                "sigma": sig[i],
                "r_h_B": round(resc["rows"][i]["h_B"] / ctrl["rows"][i]["h_B"], 4),
                "r_h_C": round(resc["rows"][i]["h_C"] / ctrl["rows"][i]["h_C"], 4),
            }
            for i in co
        ],
        "rel_R_hole_context": {
            "note": "E25.0 hole was 768/sigma=0.4333; rel_R per bin above",
            "ctrl": ctrl["rel_R"],
            "rescond": resc["rel_R"],
        },
        "ctrl_vs_sincos_context": {
            "caveat": "cross-boot scalar comparison only (Δρ band 0.03–0.05); "
            "sincos = full-scale standing operating point (e30 run ledger), "
            "ctrl = tenth-scale combo — context for where tenth-scale sits",
            "sincos": sincos_rows,
        },
    }
    _finish(out)


def _finish(out: dict) -> None:
    for st in ("ctrl", "rescond"):
        key = f"ledger_{st}"
        if key not in out:
            continue
        print(f"\n== {st}")
        print("  σ       rho     relB   relC   relR   h(B)    h(C)    h(B+C)")
        for r in out[key]:
            print(
                f"  {r['sigma']:<7} {r['rho']:+.3f}  {r['rel_cos_B']:.3f}  "
                f"{r['rel_cos_C']:.3f}  {r['rel_R']:.3f}  {r['h_B']:.4f}  "
                f"{r['h_C']:.4f}  {r['h_B_plus_C']:.4f}"
                f"  {'read' if r['readable'] else 'UNREADABLE'}"
            )
    if "reading_25b1" in out:
        v = out["reading_25b1"]
        print(
            f"\n[25b-1] verdict {v['verdict']}: median r_h={v['median_r_h']} "
            f"median Δρ={v['median_d_rho']:+.4f}"
        )
        for b in v["per_bin"]:
            print(f"  σ={b['sigma']:<7} r_h={b['r_h']:.3f} Δρ={b['d_rho']:+.4f}")
    else:
        print(f"\nverdict: {out.get('verdict')}")
    out_path = HERE / "e25b_read.json"
    out_path.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
