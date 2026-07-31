#!/usr/bin/env python3
"""E14 14.0 — pricing the two-level unbiased demotion estimator. No GPU.

Prices candidates A (768 @ sigma>0.5), B (896 all-sigma), C (768 all-sigma)
from already-committed run data:

- PRIMARY: E1b per_image.jsonl (runs/20260729-0014-e1b-debiased-map) —
  per-image cross-grid cosine + both gradient norms per sigma bin, full
  [0,1] range, D=8, N=40. The per-image mean-difference norm is exact by
  law of cosines: ||d||^2/||g_nat||^2 = 1 + r^2 - 2 r cos, r = g_e/g_nat.
- CROSS-CHECK: E9 arm-sum vectors (bench/results/20260731-0721/arm_sums)
  — aggregate ||g_dem - g_nat||/||g_nat|| per in-window bin, exact, with
  the a/b native-twin floor. Different estimand (aggregate vs per-image);
  the gap between the two columns is the cross-image heterogeneity share.
- REFERENCE: E5 X-form fit (runs/20260729-1322-e5-refit) — carried as
  model_reference; the 1024-tier price is DATA-priced here (design
  deviation from the proposal's model-priced q1, recorded in result.json:
  E1b measures the full sigma range, so the model is only needed for
  unmeasured tiers).

Cost constants are E4's measured per-step FLOPs ratios (e4 README table,
n=3): c_896 = 1-0.308, c_768 = 1-0.522. A correction step adds one
native fwd+bwd (relative cost 1).

Estimator variance added per eligible step at bin j: (1/q_j - 1) * E2_j,
E2_j = mean_i ||d_i||^2/||g_nat,i||^2  (LOWER BOUND on the per-draw
second moment: per-image D=8 means; per-draw dispersion not in stored
data — proposal hazard h4). Baseline V0_j = per-draw within-image noise
from the split-half floor: nu_i = (D/2)*(1/cos_floor_i - 1), median over
images (CONSERVATIVE: omits between-image dispersion, so inflation is
overstated on both counts... numerator under}stated, denominator
understated — the ratio's net direction is dominated by the missing
between-image term, so the gate read is conservative).

Gate (pre-registered): some candidate reaches net >= -20% FLOPs at
inflation <= 1.5x.  q floor 0.05 (hazard h2).
"""

import argparse
import datetime as _dt
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAPER_BENCH = HERE.parent.parent
SIGMA = PAPER_BENCH.parent

E1B_DIR = PAPER_BENCH / "runs" / "20260729-0014-e1b-debiased-map"
E9_DIR = SIGMA / "bench" / "results" / "20260731-0721"
E5_RESULT = PAPER_BENCH / "runs" / "20260729-1322-e5-refit" / "result.json"

E4_COST = {896: 1.0 - 0.308, 768: 1.0 - 0.522}  # measured, e4 README n=3
Q_FLOOR = 0.05
INFL_GATE = 1.5
NET_GATE = 0.20
D_DRAWS = 8  # e1b draws_per_bin

CANDIDATES = [
    {"key": "A", "route": 768, "gate": "window", "desc": "768 @ sigma>0.5 + MLMC"},
    {"key": "B", "route": 896, "gate": "all", "desc": "896 all-sigma + MLMC"},
    {"key": "C", "route": 768, "gate": "all", "desc": "768 all-sigma + MLMC"},
]


def load_e1b(path):
    rows = [json.loads(l) for l in open(path)]
    centers = rows[0]["sigma_centers"]  # 9: 8 interior + endpoint 1.0
    return rows, centers


def bin_stats(rows, route, j):
    """Per-image relative difference^2 at bin j for one route."""
    rel2 = []
    for r in rows:
        gn = r["gnorm_native"][j]
        ge = r[f"gnorm_{route}"][j]
        c = r[f"cos_{route}"][j]
        if not (gn > 0 and ge > 0 and math.isfinite(c)):
            continue
        rr = ge / gn
        rel2.append(max(0.0, 1.0 + rr * rr - 2.0 * rr * c))
    a = np.asarray(rel2)
    n = len(a)
    trim = np.sort(a)[: max(1, int(round(n * 0.9)))]  # top-10% trimmed
    return {
        "n": n,
        "mean": float(a.mean()),
        "sem": float(a.std(ddof=1) / math.sqrt(n)),
        "median": float(np.median(a)),
        "trim10_mean": float(trim.mean()),
    }


def v0_stats(rows, j):
    """Per-draw within-image relative noise from the split-half floor."""
    nus = []
    for r in rows:
        cf = r["cos_floor"][j]
        if cf > 0.05:
            nus.append((D_DRAWS / 2.0) * (1.0 / cf - 1.0))
    return float(np.median(np.asarray(nus))), len(nus)


def allocate(m2, elig, qbar):
    """q_j proportional to sqrt(m2_j) on eligible bins, mean = qbar, clipped."""
    q = np.zeros(len(m2))
    w = np.sqrt(np.maximum(m2, 1e-12))
    q[elig] = w[elig]
    for _ in range(20):
        scale = qbar / max(q[elig].mean(), 1e-12)
        q[elig] = np.clip(q[elig] * scale, Q_FLOOR, 1.0)
        if abs(q[elig].mean() - qbar) < 1e-6:
            break
    return q


def price(cand, e2, m2, v0, centers):
    interior = list(range(8))  # endpoint (j=8) is measure-zero under the flat draw
    if cand["gate"] == "window":
        elig = [j for j in interior if centers[j] > 0.5]
    else:
        elig = interior
    mass = len(elig) / 8.0
    c_e = E4_COST[cand["route"]]
    base = sum(v0[j] for j in interior) / 8.0

    best = None
    for qbar in np.arange(0.02, 0.61, 0.01):
        q = allocate(m2, elig, qbar)
        qbar_real = float(q[elig].mean())
        added = sum((1.0 / q[j] - 1.0) * e2[j] for j in elig) / 8.0
        infl = 1.0 + added / base
        net = mass * (1.0 - c_e) - mass * qbar_real
        spike = max(math.sqrt(e2[j]) / q[j] for j in elig)
        row = {
            "qbar": round(qbar_real, 4),
            "net_flops": round(net, 4),
            "inflation_lb": round(infl, 3),
            "spike_ratio": round(spike, 2),
            "q_per_bin": {round(centers[j], 4): round(float(q[j]), 3) for j in elig},
        }
        if infl <= INFL_GATE and (best is None or net > best["net_flops"]):
            best = row
    return {
        "candidate": cand["key"],
        "desc": cand["desc"],
        "route": cand["route"],
        "gross_flops": round(mass * (1.0 - c_e), 4),
        "mass": mass,
        "best": best,
        "feasible": best is not None,
        "gate_pass": bool(best and best["net_flops"] >= NET_GATE),
    }


def e9_oneprocess_pricing():
    """Candidate-A pricing entirely inside E9's process (per_image + arm sums).

    All quantities relative to ||mu_bar||^2 (the grand-mean native gradient —
    the thing SGD is estimating) per bin:
      between  = mean_i(gnorm_i^2)/||mu_bar||^2 - 1     (between-image dispersion)
      within   = mean_i(nu_i * gnorm_i^2)/||mu_bar||^2   (per-draw within-image)
      V_total  = between + within                        (per-step variance, bs=1)
      E2_abs   = mean_i(rel2_i * gnorm_i^2)/||mu_bar||^2 (correction 2nd moment,
                 LOWER bound: per-image D-draw means; per-draw UB = D * LB)
    Gate is evaluated at the OPTIMISTIC accounting (LB numerator, full
    baseline): if A fails even here, the kill is airtight.
    """
    d = E9_DIR / "arm_sums"
    rows = [json.loads(l) for l in open(E9_DIR / "per_image.jsonl")]
    n_img, d_draws = 24, 8
    centers = rows[0]["sigma_centers"]  # [0.5625, 0.6875, 0.8125, 0.9375, 1.0]
    bins = {}
    for b in range(5):
        a = np.load(d / f"a__b{b}.npy")
        bb = np.load(d / f"b__b{b}.npy")
        mu_bar = (a + bb) / (2.0 * n_img * d_draws)
        mu2 = float(np.linalg.norm(mu_bar)) ** 2
        g2, nu_abs, e2_abs = [], [], []
        for r in rows:
            gn = r["gnorm_native"][b]
            cf = r["cos_floor"][b]
            g2.append(gn * gn)
            if cf > 0.05:
                nu_abs.append((d_draws / 2.0) * (1.0 / cf - 1.0) * gn * gn)
        entry = {
            "mu_bar_norm": round(math.sqrt(mu2), 5),
            "between": round(float(np.mean(g2)) / mu2 - 1.0, 2),
            "within": round(float(np.mean(nu_abs)) / mu2, 2),
        }
        entry["v_total"] = round(entry["between"] + entry["within"], 2)
        for route in (896, 768, 512):
            vals = []
            for r in rows:
                gn = r["gnorm_native"][b]
                ge = r[f"gnorm_{route}"][b]
                c = r[f"cos_{route}"][b]
                if gn > 0 and ge > 0 and math.isfinite(c):
                    rr = ge / gn
                    rel2 = max(0.0, 1.0 + rr * rr - 2.0 * rr * c)
                    vals.append(rel2 * gn * gn)
            entry[f"e2_abs_{route}"] = round(float(np.mean(vals)) / mu2, 2)
        bins[b] = entry
    # candidate A: route 768, eligible = interior in-window bins 0..3 (endpoint excluded)
    elig = [0, 1, 2, 3]
    mass = 0.5  # sigma>0.5 under the flat draw
    c_e = E4_COST[768]
    base = float(np.mean([bins[b]["v_total"] for b in elig]))
    e2 = np.array([bins[b]["e2_abs_768"] for b in range(5)])
    sweep = []
    best = None
    for qbar in np.arange(0.02, 0.71, 0.01):
        q = allocate(e2[:4], list(range(4)), qbar)
        qbar_real = float(q.mean())
        added = float(np.mean([(1.0 / q[b] - 1.0) * e2[b] for b in elig]))
        infl = 1.0 + added / base
        net = mass * (1.0 - c_e) - mass * qbar_real
        row = {"qbar": round(qbar_real, 3), "net_flops": round(net, 4), "inflation_lb": round(infl, 3)}
        sweep.append(row)
        if infl <= INFL_GATE and (best is None or net > best["net_flops"]):
            best = dict(row, q_per_bin={round(centers[b], 4): round(float(q[b]), 3) for b in elig})
    return {
        "sigma_centers": centers,
        "bins": bins,
        "candidate_A": {
            "best_optimistic": best,
            "gate_pass": bool(best and best["net_flops"] >= NET_GATE),
            "sweep_head": sweep[:8],
            "note": "LB numerator (per-draw UB is D=8x larger); full V_total baseline — most favorable read",
        },
    }


def e9_cross_check():
    """Aggregate rel-diff from E9 arm-sum vectors, with the a/b twin floor."""
    d = E9_DIR / "arm_sums"
    if not d.exists():
        return {"available": False}
    out = {"available": True, "sigma_centers": [0.5625, 0.6875, 0.8125, 0.9375, 1.0]}
    for route in (896, 768, 512):
        rels, floors = [], []
        for b in range(5):
            a = np.load(d / f"a__b{b}.npy") if (d / f"a__b{b}.npy").exists() else None
            bb = np.load(d / f"b__b{b}.npy") if (d / f"b__b{b}.npy").exists() else None
            if a is None or bb is None:
                return {"available": False, "note": "native a/b sums not found"}
            nat = (a + bb) / 2.0
            sets = []
            for suffix in ("", "__2"):
                f = d / f"{route}{suffix}__b{b}.npy"
                if f.exists():
                    sets.append(np.load(f))
            dem = np.mean(sets, axis=0)
            gn = float(np.linalg.norm(nat))
            rels.append(round(float(np.linalg.norm(dem - nat)) / gn, 4))
            floors.append(round(float(np.linalg.norm(a - bb)) / gn, 4))
        out[str(route)] = {"rel_diff_aggregate": rels, "ab_twin_floor": floors}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="run dir (default: runs/<ts>-e14-pricing)")
    args = ap.parse_args()

    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M")
    out = Path(args.out) if args.out else PAPER_BENCH / "runs" / f"{ts}-e14-pricing"
    out.mkdir(parents=True, exist_ok=True)

    rows, centers = load_e1b(E1B_DIR / "per_image.jsonl")

    v0, v0_n = zip(*(v0_stats(rows, j) for j in range(9)))
    map_stats = {}
    for route in (896, 768, 512):
        map_stats[route] = [bin_stats(rows, route, j) for j in range(9)]

    results = []
    for cand in CANDIDATES:
        st = map_stats[cand["route"]]
        e2 = np.array([s["mean"] for s in st])
        m2 = e2 + np.array([s["sem"] for s in st])  # uncertainty-inflated pricing moment
        results.append(price(cand, e2, m2, np.array(v0), centers))

    xcheck = e9_cross_check()
    oneproc = e9_oneprocess_pricing()

    e5 = json.load(open(E5_RESULT))
    model_ref = {
        "form": "X",
        "F0": e5["forms"]["X"]["F0"],
        "tau_tokens": e5["forms"]["X"]["tau_tokens"],
        "f768_predicted": e5["forms"]["X"]["f768_predicted"],
        "held_out_mean_rmse": e5["verdict"]["held_out_mean_rmse"]["X"],
    }

    winner = max((r for r in results if r["feasible"]), key=lambda r: r["best"]["net_flops"], default=None)
    gate_pass = bool(winner and winner["gate_pass"])

    result = {
        "schema_version": 1,
        "script": "project/sigma_lowres/paper_bench/experiments/e14/e14_price.py",
        "label": "e14-pricing",
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": {"e1b": str(E1B_DIR), "e9": str(E9_DIR), "e5": str(E5_RESULT)},
        "design_note": (
            "q1 is DATA-priced from E1b per-image vectors (full sigma range measured), "
            "not model-priced as the proposal sketched; E5-X carried as model_reference "
            "for unmeasured tiers."
        ),
        "caveats": [
            "E2 is a LOWER BOUND on per-draw correction second moment (per-image D=8 means; h4)",
            "V0 baseline = within-image draw noise only (no between-image term): inflation overstated -> gate conservative",
            "raw (undebiased) cross-grid cosines used: attenuation overstates rel-diff -> conservative",
            "E1b arms are plain demote; trainer g_dem includes yarnsig (h3) — 14.1 measures the wired estimand",
            "E9 cross-check is a different process AND different estimand (aggregate vs per-image); norms only, no cross-process cosines (E1 rule)",
        ],
        "constants": {"E4_cost": E4_COST, "q_floor": Q_FLOOR, "infl_gate": INFL_GATE, "net_gate": NET_GATE},
        "metrics": {
            "sigma_centers": centers,
            "rel2_map": {str(r): map_stats[r] for r in map_stats},
            "v0_per_bin": list(v0),
            "v0_n_used": list(v0_n),
            "e9_cross_check": xcheck,
            "e9_oneprocess": oneproc,
            "model_reference": model_ref,
        },
        "candidates": results,
        "verdict": {
            "gate": f"net >= -{int(NET_GATE*100)}% FLOPs at inflation <= {INFL_GATE}x",
            "gate_pass_e1b_conservative": gate_pass,
            "gate_pass_e9_optimistic": oneproc["candidate_A"]["gate_pass"],
            "gate_pass": bool(gate_pass or oneproc["candidate_A"]["gate_pass"]),
            "winner": winner["candidate"] if winner else None,
            "winner_best": winner["best"] if winner else None,
            "A_best_optimistic": oneproc["candidate_A"]["best_optimistic"],
        },
    }
    (out / "result.json").write_text(json.dumps(result, indent=1))

    q_table = {
        "source_run": str(out.name),
        "estimand": "q per sigma bin, flat-sigma interior bins; endpoint measure-zero (train never lands there)",
        "candidates": {
            r["candidate"]: {"route": r["route"], "desc": r["desc"], "best": r["best"]} for r in results
        },
        "winner": winner["candidate"] if winner else None,
    }
    (out / "q_table.json").write_text(json.dumps(q_table, indent=1))

    # readable report
    print(f"== E14 14.0 pricing -> {out}")
    print("\nrel-diff map  sqrt(E2) = mean ||g_dem-g_nat||/||g_nat|| per image (E1b, N=40, D=8)")
    hdr = "route " + " ".join(f"{c:>7.3f}" for c in centers)
    print(hdr)
    for route in (896, 768, 512):
        print(f"{route:>5} " + " ".join(f"{math.sqrt(s['mean']):>7.3f}" for s in map_stats[route]))
    print("V0    " + " ".join(f"{v:>7.2f}" for v in v0) + "   (per-draw within-image rel var, median)")
    if xcheck.get("available"):
        print("\nE9 aggregate cross-check (in-window bins, exact vectors; a/b twin floor in parens)")
        for route in ("896", "768", "512"):
            r = xcheck[route]
            print(f"{route:>5} " + " ".join(f"{v:.3f}({f_:.3f})" for v, f_ in zip(r["rel_diff_aggregate"], r["ab_twin_floor"])))
    print("\ncandidates:")
    for r in results:
        b = r["best"]
        if b:
            print(
                f"  {r['candidate']} {r['desc']:<28} gross {r['gross_flops']*100:5.1f}%  "
                f"best: net {b['net_flops']*100:5.1f}% @ qbar {b['qbar']:.2f}  "
                f"infl(LB) {b['inflation_lb']:.2f}x  spike {b['spike_ratio']:.1f}  "
                f"{'GATE PASS' if r['gate_pass'] else 'below net gate'}"
            )
        else:
            print(f"  {r['candidate']} {r['desc']:<28} gross {r['gross_flops']*100:5.1f}%  INFEASIBLE at infl <= {INFL_GATE}")
    print("\nE9 one-process pricing (candidate A, optimistic: LB numerator / full V_total baseline)")
    print("  bin      mu_bar   between   within   V_total   E2abs(768)")
    for b in range(4):
        e = oneproc["bins"][b]
        print(
            f"  {oneproc['sigma_centers'][b]:.4f}  {e['mu_bar_norm']:.4f}  {e['between']:8.1f} {e['within']:8.1f}"
            f" {e['v_total']:9.1f}  {e['e2_abs_768']:9.1f}"
        )
    ba = oneproc["candidate_A"]["best_optimistic"]
    if ba:
        print(f"  A best: net {ba['net_flops']*100:5.1f}% @ qbar {ba['qbar']:.2f}  infl(LB) {ba['inflation_lb']:.2f}x"
              f"  {'GATE PASS' if oneproc['candidate_A']['gate_pass'] else 'below net gate'}")
    else:
        print(f"  A: INFEASIBLE at infl <= {INFL_GATE} even at qbar 0.70")
    final = bool(gate_pass or oneproc["candidate_A"]["gate_pass"])
    print(f"\nVERDICT: gate_pass={final}"
          f" (e1b-conservative={gate_pass}, e9-optimistic={oneproc['candidate_A']['gate_pass']})"
          + (f", best-net candidate={winner['candidate']}" if winner else ""))


if __name__ == "__main__":
    main()
