"""E26 full-grid read — frozen verdict per the 2026-08-10 amendment.

Loads the two grid stores' ledger.json (produced by vector_ledger.py
unmodified, --data_ref reenc) plus the in-session sincos twin reference,
applies the frozen per-bin criteria, and writes e26_grid_read.json.

Frozen criteria (README "Full-grid amendment", frozen 2026-08-10 before
any grid gradient existed):
  readable(bin) : rel_cos_B >= 0.5 and rel_cos_C >= 0.5
  criteria(bin) : I < 0 and rho <= -0.5 and h(B+C) < min(h(B), h(C))
  REPLICATES    : >= 4/5 readable AND criteria at every readable bin
  PARTIAL       : criteria at >= 3 readable bins including sigma = 0.7
  FAILS         : otherwise
  INCONCLUSIVE  : < 3 readable bins (remedy: single D=24 top-up rerun)

Identity-consistency column (pre-registered): per bin,
  rho_implied = (h(B+C)^2 - h(B)^2 - h(C)^2) / (2 h(B) h(C))
reported next to measured rho; the depth-scaling upgrade is claimed only
where measured cross-adapter rho deepening exceeds what rho_implied
already forces from the magnitudes.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # project/sigma_lowres
RESULTS = ROOT / "bench" / "results"

STORES = {
    "flat": RESULTS / "20260810-1114-e26-grid-flat",
    "dirty": RESULTS / "20260810-1519-e26-grid-dirty",
    "sincos_twin": RESULTS / "20260810-0658-e28-native-twin-768",
}
ROUTE = "768"
SIGMA_070 = 0.7


def load_rows(run_dir):
    led = json.loads((run_dir / "ledger.json").read_text())
    sigmas = led["sigma_centers"]
    rows = led["bc_ledger"][ROUTE]
    assert len(rows) == len(sigmas), (run_dir, len(rows), len(sigmas))
    return sigmas, rows


def read_adapter(run_dir):
    sigmas, rows = load_rows(run_dir)
    bins = []
    for s, r in zip(sigmas, rows):
        hB, hC, hBC = r["h_B"], r["h_C"], r["h_B_plus_C"]
        readable = r["rel_cos_B"] >= 0.5 and r["rel_cos_C"] >= 0.5
        crit = r["I"] < 0 and r["rho"] <= -0.5 and hBC < min(hB, hC)
        rho_implied = (hBC**2 - hB**2 - hC**2) / (2 * hB * hC)
        bins.append(
            {
                "sigma": round(s, 4),
                "G": r["G"],
                "h_B": hB,
                "h_C": hC,
                "h_B_plus_C": hBC,
                "rho": r["rho"],
                "rho_implied": round(rho_implied, 4),
                "I": r["I"],
                "rel_cos_B": r["rel_cos_B"],
                "rel_cos_C": r["rel_cos_C"],
                "readable": readable,
                "criteria_pass": readable and crit,
            }
        )
    readable_bins = [b for b in bins if b["readable"]]
    passing = [b for b in bins if b["criteria_pass"]]
    pass_at_07 = any(abs(b["sigma"] - SIGMA_070) < 1e-3 for b in passing)
    if len(readable_bins) < 3:
        verdict = "INCONCLUSIVE"
    elif len(readable_bins) >= 4 and all(b["criteria_pass"] for b in readable_bins):
        verdict = "REPLICATES"
    elif len(passing) >= 3 and pass_at_07:
        verdict = "PARTIAL"
    else:
        verdict = "FAILS"
    return {
        "bins": bins,
        "n_readable": len(readable_bins),
        "n_criteria_pass": len(passing),
        "pass_includes_s07": pass_at_07,
        "verdict": verdict,
    }


def main():
    out = {
        "doc": "E26 full-grid read — frozen 2026-08-10 amendment criteria",
        "instrument": "vector_ledger.py unmodified, --data_ref reenc",
        "route": "1024->768",
        "reference": "in-session sincos = e28 native twin (same boot family 2026-08-09 19:32)",
        "stores": {k: str(v.relative_to(ROOT)) for k, v in STORES.items()},
        "adapters": {},
    }
    for name in ("flat", "dirty", "sincos_twin"):
        res = read_adapter(STORES[name])
        if name == "sincos_twin":
            res["verdict"] = "(reference — criteria columns reported for context only)"
        out["adapters"][name] = res

    out_path = Path(__file__).with_name("e26_grid_read.json")
    out_path.write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path}")

    for name, res in out["adapters"].items():
        print(f"\n== {name}  verdict: {res['verdict']}"
              f"  (readable {res['n_readable']}/5, pass {res['n_criteria_pass']}/5)")
        print("  sigma    G       h(B)    h(C)    h(B+C)  rho     rho_impl  I        relB   relC   ok")
        for b in res["bins"]:
            print(
                f"  {b['sigma']:<7} {b['G']:<7.4f} {b['h_B']:<7.4f} {b['h_C']:<7.4f} "
                f"{b['h_B_plus_C']:<7.4f} {b['rho']:<7.3f} {b['rho_implied']:<8.3f} "
                f"{b['I']:<8.4f} {b['rel_cos_B']:<6.3f} {b['rel_cos_C']:<6.3f} "
                f"{'PASS' if b['criteria_pass'] else ('read' if b['readable'] else 'UNREADABLE')}"
            )


if __name__ == "__main__":
    main()
