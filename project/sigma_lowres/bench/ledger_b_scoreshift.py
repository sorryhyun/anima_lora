#!/usr/bin/env python
"""E19.5 — is the data leg literally the derivable score-shift? Direction-level
correlation of the MEASURED r-level legs (19.2 residual store) against the
CLOSURE-PREDICTED legs (19.1 machinery), per (route, σ, closure).

19.1/19.2 compared the two ledgers only through summary statistics (ρ_r sign/
shape/ordering, amplitude ratios). This read asks the sharper question the
Yang-Song picture suggests: does the Gaussian closure predict the *direction*
of each leg — i.e. is B_r (data branch) the analytically-derivable score-shift
of demotion s_lo − s_hi, and how much of C_r (graph branch) the same machinery
explains. Estimand per (route e, σ, closure):

    dircos(L) = Σ_i⟨L̄_meas⊥, L̂_pred⊥⟩ / √(Σ_i debias|L_meas⊥|² · Σ_i|L̂_pred⊥|²)

for L ∈ {B, C, net=B+C}, on the demoted grid, area downsample, ⊥ to the
MEASURED native residual direction (one common subspace for both sides;
predicted-native / no-projection as controls), reenc ref primary. The
numerator is unbiased under measurement noise (closure fields deterministic,
fit split disjoint from the probe holdout); the measured second moment is the
cross-half debiased one from ``ledger_rho_r.py`` (shared-ref term subtracted
for B and net). The naive (attenuated) cosine and per-image cosines are
emitted as secondary; reliability gates (rel_cos ≥ 0.5) carry over unchanged.

**Pre-registered reading (verdict bins σ ≥ 0.3, reenc/area, gated):**
dircos_B − dircos_C ≥ +0.2 consistently → the data leg is the derivable
score-shift and beyond-Gaussian structure concentrates in the graph leg;
both ≥ 0.7 → the closure owns both directions and its 19.2 magnitude miss is
a pure scale story; both ≤ 0.3 → r-level directions are beyond-Gaussian and
the 19.1/19.2 ρ agreement was angle-statistics only.

CPU-only when the 19.1 closure-latent cache is warm.

Usage::

    python project/sigma_lowres/bench/ledger_b_scoreshift.py \
        --run project/sigma_lowres/bench/results/20260806-2342-e192-rho-r-measured \
        [--smoke] [--write_record]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402
from project.sigma_lowres.bench.run_closure_rho import ensure_latents  # noqa: E402
from project.sigma_lowres.bench.run_posterior_closure import (  # noqa: E402
    ArmModel,
    dct2,
    idct2,
    predict_residuals,
)
from project.sigma_lowres.bench.tier_routing.redundancy import score_corpus  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

SIGMA = Path(__file__).resolve().parents[1]
E19 = SIGMA / "paper_bench" / "experiments" / "e19"
RUN_191 = SIGMA / "bench" / "results" / "20260806-2223-e19-closure-rho"

CLOSURES = ("diag", "block", "octave")
CONTROL_CLOSURE = "block"
REL_GATE = 0.5
VERDICT_SIGMA_MIN = 0.3
LEGS = ("B", "C", "net")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--run", required=True, help="19.2 run dir holding residuals/")
    p.add_argument("--tier", type=int, default=1024)
    p.add_argument("--edges", default="896,768,512")
    p.add_argument("--vae", default=None)
    p.add_argument(
        "--data_root",
        default=str(Path(__file__).resolve().parent / "probe1024_closure"),
    )
    p.add_argument("--n_bands", type=int, default=10)
    p.add_argument("--shrink_R", type=float, default=0.2)
    p.add_argument("--clip_pc", type=float, default=0.6)
    p.add_argument("--quant_k", type=int, default=4)
    p.add_argument("--label", default=None)
    p.add_argument("--smoke", action="store_true", help="2 images, 1 route, block only")
    p.add_argument(
        "--write_record",
        action="store_true",
        help="also write dircos_195.json into experiments/e19/",
    )
    return p.parse_args()


def down(x: torch.Tensor, hw: tuple[int, int], op: str) -> torch.Tensor:
    if x.shape[-2:] == tuple(hw):
        return x
    kw = {} if op == "area" else {"align_corners": False}
    return F.interpolate(x.unsqueeze(0), size=hw, mode=op, **kw).squeeze(0)


def load_halves(f: Path) -> tuple[torch.Tensor, torch.Tensor]:
    z = np.load(f)
    return (
        torch.from_numpy(z["m1"].astype(np.float32)),
        torch.from_numpy(z["m2"].astype(np.float32)),
    )


def pooled_row(s: dict[str, float], pic: dict[str, list[float]]) -> dict:
    out: dict = {"n_images": int(s["n"])}
    rel = {}
    for leg in LEGS:
        mm = s[f"{leg}_mm_debias"]
        pp = s[f"{leg}_pp"]
        den = math.sqrt(max(mm, 0.0) * pp)
        den_naive = math.sqrt(s[f"{leg}_mm_naive"] * pp)
        rel[leg] = (
            s[f"{leg}_mm12"] / math.sqrt(s[f"{leg}_mm1"] * s[f"{leg}_mm2"])
            if s[f"{leg}_mm1"] > 0 and s[f"{leg}_mm2"] > 0
            else float("nan")
        )
        pa = np.array(pic[leg])
        out[f"dircos_{leg}"] = (
            round(s[f"{leg}_mp"] / den, 5) if den > 0 else float("nan")
        )
        out[f"dircos_{leg}_naive"] = (
            round(s[f"{leg}_mp"] / den_naive, 5) if den_naive > 0 else float("nan")
        )
        out[f"amp_pred_over_meas_{leg}"] = (
            round(math.sqrt(pp / mm), 5) if mm > 0 else float("nan")
        )
        out[f"rel_cos_{leg}"] = round(rel[leg], 5)
        out[f"per_image_cos_{leg}_mean"] = (
            round(float(pa.mean()), 5) if len(pa) else None
        )
    out["rel_ok"] = bool(
        rel["B"] == rel["B"]
        and rel["C"] == rel["C"]
        and rel["B"] >= REL_GATE
        and rel["C"] >= REL_GATE
    )
    return out


def main() -> None:
    args = parse_args()
    start_heartbeat()
    run = Path(args.run).resolve()
    res_dir = run / "residuals"
    if not res_dir.is_dir():
        raise SystemExit(f"no residuals/ under {run}")
    native = str(args.tier)
    edges = [e for e in args.edges.split(",") if e]
    closures = CLOSURES

    # measured store index
    index: dict[tuple[str, str, int], Path] = {}
    stems, sis = set(), set()
    for f in sorted(res_dir.glob("*.npz")):
        stem, grid, s = f.stem.rsplit("__", 2)
        index[(stem, grid, int(s[1:]))] = f
        stems.add(stem)
        sis.add(int(s[1:]))
    stems_l = sorted(stems)
    n_sig = max(sis) + 1
    sigmas = [
        round(float(np.load(index[(stems_l[0], native, si)])["sigma"]), 6)
        for si in range(n_sig)
    ]

    # 19.1's exact fit split (fit models must match the frozen prediction's)
    rec_191 = json.loads((RUN_191 / "result.json").read_text())
    fit_stems = set(rec_191["extra"]["fit"])
    hold_stems_191 = set(rec_191["extra"]["holdout"])
    if set(stems_l) != hold_stems_191:
        raise SystemExit("19.2 store stems != 19.1 holdout — refusing to mix splits")

    if args.smoke:
        stems_l = stems_l[:2]
        edges = edges[:1]
        closures = (CONTROL_CLOSURE,)

    records = [
        r
        for r in score_corpus(k=args.quant_k)
        if r.complete() and r.tier == args.tier and r.stem in (fit_stems | set(stems_l))
    ]
    by_stem = {r.stem: r for r in records}
    fit_sorted = sorted((by_stem[s] for s in fit_stems), key=lambda r: r.stem)
    holdout = [by_stem[s] for s in stems_l]
    log.info(f"{len(holdout)} probe images, fit {len(fit_sorted)}, σ={sigmas}")

    lat = ensure_latents(
        holdout + fit_sorted, [int(e) for e in edges], args.vae, Path(args.data_root)
    )
    arms_cl = (
        ["native", "reenc"]
        + [f"demote{e}" for e in edges]
        + [f"repromote{e}" for e in edges]
    )
    log.info("fitting arm models (full fit, 19.1 split)")
    models: dict[str, ArmModel] = {}
    for a in arms_cl:
        coeffs = [dct2(lat[(r.stem, a)]) for r in fit_sorted]
        models[a] = ArmModel.fit(coeffs, args.n_bands, args.shrink_R, args.clip_pc)

    # variant grid: (closure, ref, op, projector)
    variants = [(cl, "reenc", "area", "meas") for cl in closures]
    if CONTROL_CLOSURE in closures and not args.smoke:
        variants += [
            (CONTROL_CLOSURE, "reenc", "bicubic", "meas"),
            (CONTROL_CLOSURE, "native", "area", "meas"),
            (CONTROL_CLOSURE, "reenc", "area", "pred"),
            (CONTROL_CLOSURE, "reenc", "area", "none"),
        ]

    chain_cache: dict = {}
    pools: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    percos: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for st in stems_l:
        coeff = {a: dct2(lat[(st, a)]) for a in arms_cl}
        pred = {
            cl: {
                a: [
                    torch.from_numpy(idct2(x).astype(np.float32))
                    for x in predict_residuals(
                        coeff[a], models[a], sigmas, args.n_bands, cl, chain_cache
                    )
                ]
                for a in arms_cl
            }
            for cl in closures
        }
        for si in range(n_sig):
            nat1f, nat2f = load_halves(index[(st, native, si)])
            re1f, re2f = load_halves(index[(st, "reenc", si)])
            for e in edges:
                dem1, dem2 = load_halves(index[(st, e, si)])
                rp1f, rp2f = load_halves(index[(st, f"repromote{e}", si)])
                hw = dem1.shape[-2:]
                d1 = dem1.double().flatten()
                d2 = dem2.double().flatten()
                for cl, ref_name, op, proj in variants:
                    nat = down(0.5 * (nat1f + nat2f), hw, op)
                    nhat_m = (nat / nat.norm()).double().flatten()
                    rp1 = down(rp1f, hw, op).double().flatten()
                    rp2 = down(rp2f, hw, op).double().flatten()
                    if ref_name == "reenc":
                        ref = down(0.5 * (re1f + re2f), hw, op).double().flatten()
                        rd = down(re1f - re2f, hw, op).double().flatten()
                    else:
                        ref = nat.double().flatten()
                        rd = down(nat1f - nat2f, hw, op).double().flatten()
                    # closure-predicted legs on the same grid
                    p_nat = down(pred[cl]["native"][si], hw, op)
                    p_rp = (
                        down(pred[cl][f"repromote{e}"][si], hw, op).double().flatten()
                    )
                    p_dem = pred[cl][f"demote{e}"][si].double().flatten()
                    p_ref = (
                        down(pred[cl]["reenc"][si], hw, op).double().flatten()
                        if ref_name == "reenc"
                        else p_nat.double().flatten()
                    )
                    if proj == "meas":
                        nhat = nhat_m
                    elif proj == "pred":
                        pn = p_nat.double().flatten()
                        nhat = pn / pn.norm()
                    else:
                        nhat = None

                    def pp(v: torch.Tensor) -> torch.Tensor:
                        return v if nhat is None else v - (v @ nhat) * nhat

                    halves = {
                        "B": (pp(rp1 - ref), pp(rp2 - ref)),
                        "C": (pp(d1 - rp1), pp(d2 - rp2)),
                        "net": (pp(d1 - ref), pp(d2 - ref)),
                    }
                    preds_l = {
                        "B": pp(p_rp - p_ref),
                        "C": pp(p_dem - p_rp),
                        "net": pp(p_dem - p_ref),
                    }
                    rdp = pp(rd)
                    shared_ref = {"B": True, "C": False, "net": True}
                    s = pools[(e, cl, ref_name, op, proj, si)]
                    pc = percos[(e, cl, ref_name, op, proj, si)]
                    for leg in LEGS:
                        h1, h2 = halves[leg]
                        pv = preds_l[leg]
                        m = 0.5 * (h1 + h2)
                        mm12 = float(h1 @ h2)
                        s[f"{leg}_mp"] += float(m @ pv)
                        s[f"{leg}_pp"] += float(pv @ pv)
                        s[f"{leg}_mm12"] += mm12
                        s[f"{leg}_mm1"] += float(h1 @ h1)
                        s[f"{leg}_mm2"] += float(h2 @ h2)
                        s[f"{leg}_mm_debias"] += mm12 - (
                            float(rdp @ rdp) / 4.0 if shared_ref[leg] else 0.0
                        )
                        s[f"{leg}_mm_naive"] += float(m @ m)
                        nm, npv = float(m.norm()), float(pv.norm())
                        if nm > 0 and npv > 0:
                            pc[leg].append(float(m @ pv) / (nm * npv))
                    s["n"] += 1
        log.info(f"  pooled {st}")

    run_dir = make_run_dir(
        "sigma_lowres", args.label, root=Path(__file__).resolve().parent / "results"
    )
    result: dict = {
        "run_dir_measured": str(run),
        "run_dir_closure_fit": str(RUN_191),
        "sigma_centers": sigmas,
        "n_images": len(stems_l),
        "estimand": "pooled direction cosine measured-vs-closure per leg on the "
        "demoted grid, area downsample, ⊥ to D_e·r̄_native (measured) — cross-half "
        "debiased measured second moments, reenc ref primary",
        "rel_gate": REL_GATE,
        "verdict_sigma_min": VERDICT_SIGMA_MIN,
        "routes": {},
    }
    for e in edges:
        blk: dict = {}
        for cl, ref_name, op, proj in variants:
            tab = [
                pooled_row(
                    pools[(e, cl, ref_name, op, proj, si)],
                    percos[(e, cl, ref_name, op, proj, si)],
                )
                for si in range(n_sig)
            ]
            blk[f"{cl}_{ref_name}_{op}_{proj}"] = {"table": tab}
        result["routes"][e] = blk

    # verdict: pre-registered contrast on gated verdict bins (reenc, area, meas)
    for e in edges:
        rows = []
        for si, s_ in enumerate(sigmas):
            if s_ < VERDICT_SIGMA_MIN:
                continue
            for cl in closures:
                t = result["routes"][e][f"{cl}_reenc_area_meas"]["table"][si]
                rows.append(
                    {
                        "sigma": s_,
                        "closure": cl,
                        "dircos_B": t["dircos_B"],
                        "dircos_C": t["dircos_C"],
                        "gap_BminusC": round(t["dircos_B"] - t["dircos_C"], 5)
                        if t["dircos_B"] == t["dircos_B"]
                        and t["dircos_C"] == t["dircos_C"]
                        else float("nan"),
                        "rel_ok": t["rel_ok"],
                    }
                )
        gated = [r for r in rows if r["rel_ok"]]
        result["routes"][e]["verdict"] = {
            "bins": rows,
            "n_gated": len(gated),
            "median_gap_gated": round(
                float(np.median([r["gap_BminusC"] for r in gated])), 5
            )
            if gated
            else None,
        }

    write_result(run_dir, script=__file__, args=args, metrics=result, label=args.label)
    if args.write_record and not args.smoke:
        (E19 / "dircos_195.json").write_text(json.dumps(result, indent=2))

    for e in edges:
        log.info(f"\n== route {e} (reenc ref, area, ⊥ measured-native) ==")
        hdr = "σ       "
        for cl in closures:
            hdr += f"  {cl[:5]}:B     C      net "
        log.info(hdr)
        for si, s_ in enumerate(sigmas):
            line = f"{s_:<7}"
            for cl in closures:
                t = result["routes"][e][f"{cl}_reenc_area_meas"]["table"][si]
                line += (
                    f"  {t['dircos_B']:+.3f} {t['dircos_C']:+.3f} "
                    f"{t['dircos_net']:+.3f}"
                )
            t0 = result["routes"][e][f"{closures[0]}_reenc_area_meas"]["table"][si]
            line += "" if t0["rel_ok"] else "  GATE-FAIL"
            log.info(line)
        v = result["routes"][e]["verdict"]
        log.info(f"median gated gap B−C: {v['median_gap_gated']}")
    log.info(f"\nresult → {run_dir}")


if __name__ == "__main__":
    main()
