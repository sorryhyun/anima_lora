#!/usr/bin/env python
"""E19.2 — measured r-level paired branch ledger ρ_r (CPU reanalysis).

Reads the residual half-mean store dumped by ``run_prior_distance.py
--repromote --save_residuals`` (per image × arm × σ parity half-means of the
K-draw mean residual) and computes the MEASURED paired branch ledger on the
trained model, on exactly the estimand pinned by ``run_closure_rho.py`` (the
frozen 19.1 prediction):

    B_r = D_e r̄_rp − D_e r̄_ref     (data branch;  ref ∈ {reenc, native})
    C_r = r̄_dem − D_e r̄_rp         (graph branch)
    B_r + C_r = r̄_dem − D_e r̄_ref  (exact)

on the demoted grid (D_e = area downsample; bicubic as the operator-sensitivity
control), ⊥ to D_e r̄_native per image, pooled by summed inner products across
images (per-image cosines secondary — E11 norm-only).

**Split-half correction** (the r-level mirror of ``vector_ledger.bc_ledger``):
B and C share the repromote arm's draw noise with opposite signs, biasing any
same-half product negative. The ledger therefore uses cross-half products —
I = (⟨B₁⊥,C₂⊥⟩ + ⟨B₂⊥,C₁⊥⟩)/2, second moments ⟨B₁⊥,B₂⊥⟩ / ⟨C₁⊥,C₂⊥⟩ — and
subtracts the shared-ref-noise term from Σ|B⊥|² via the ref half-difference
(B₁, B₂ share the averaged ref arm's noise: E⟨B₁⊥,B₂⊥⟩ = |B⊥|² + |d⊥|²/4
with d the ref parity difference). The naive same-half read is emitted as
``rho_sameset_biascheck``. Reliability is the pooled cross-half direction
cosine per branch (``rel_cos_B/C``), gated at the E14 floor (0.5).

Scores the measured table against the frozen 19.1 prediction band
(``experiments/e19/prediction_191.json``) per the pre-registered reading rule:
verdict bins σ ≥ 0.3, reenc ref, area operator, gated cells only —
|ρ_r| ≤ 0.35 → weak/derivable seed (Jᵀ owns the depth); |ρ_r| ≥ 0.5 with the
g-profile → the residual level owns the cancellation; ρ_r ≥ 0 → the closure's
sign prediction is falsified at the trained model.

Usage::

    python project/sigma_lowres/bench/ledger_rho_r.py \
        --run project/sigma_lowres/bench/results/<run> [--write_record]
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

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

E19 = Path(__file__).resolve().parents[1] / "paper_bench" / "experiments" / "e19"

REL_GATE = 0.5
VERDICT_SIGMA_MIN = 0.3
WEAK_MAX = 0.35  # pre-registered: within/below the closure band
DEEP_MIN = 0.5  # pre-registered: g-level-deep


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--run", required=True, help="run dir holding residuals/")
    p.add_argument("--tier", type=int, default=1024)
    p.add_argument("--edges", default="896,768,512")
    p.add_argument(
        "--write_record",
        action="store_true",
        help="also write measured_192.json into experiments/e19/ (the "
        "committed record); otherwise only into the run dir",
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


def crossings(sigmas: list[float], ratios: list[float]) -> list[float]:
    pts = [(s, r) for s, r in zip(sigmas, ratios) if r == r]
    out = []
    for (s0, r0), (s1, r1) in zip(pts[:-1], pts[1:]):
        if (r0 - 1.0) * (r1 - 1.0) < 0:
            out.append(round(s0 + (1.0 - r0) / (r1 - r0) * (s1 - s0), 4))
    return out


def pooled_row(s: dict[str, float], pic: list[float]) -> dict:
    bb, cc = s["bb_cross"], s["cc_cross"]
    den = math.sqrt(max(bb, 0.0) * max(cc, 0.0))
    rel_b = (
        s["bb12"] / math.sqrt(s["bb1"] * s["bb2"])
        if s["bb1"] > 0 and s["bb2"] > 0
        else float("nan")
    )
    rel_c = (
        s["cc12"] / math.sqrt(s["cc1"] * s["cc2"])
        if s["cc1"] > 0 and s["cc2"] > 0
        else float("nan")
    )
    norm_sum2 = bb + cc + 2 * s["bc_cross"]
    cancel = (
        1.0 - math.sqrt(max(norm_sum2, 0.0)) / (math.sqrt(bb) + math.sqrt(cc))
        if bb > 0 and cc > 0
        else float("nan")
    )
    pa = np.array(pic)
    return {
        "rho": round(s["bc_cross"] / den, 5) if den > 0 else float("nan"),
        "rho_sameset_biascheck": round(s["bc_same"] / den, 5)
        if den > 0
        else float("nan"),
        "amp_ratio_BoverC": round(math.sqrt(bb / cc), 5)
        if bb > 0 and cc > 0
        else float("nan"),
        "cancel_frac": round(cancel, 5),
        "rel_cos_B": round(rel_b, 5),
        "rel_cos_C": round(rel_c, 5),
        "per_image_cos_mean": round(float(pa.mean()), 5) if len(pa) else None,
        "per_image_cos_sd": round(float(pa.std(ddof=1)), 5) if len(pa) > 1 else None,
        "n_images": int(s["n"]),
    }


def main() -> None:
    args = parse_args()
    run = Path(args.run).resolve()
    res_dir = run / "residuals"
    if not res_dir.is_dir():
        raise SystemExit(f"no residuals/ under {run} (run with --save_residuals)")
    native = str(args.tier)
    edges = [e for e in args.edges.split(",") if e]

    index: dict[tuple[str, str, int], Path] = {}
    stems, sis = set(), set()
    for f in sorted(res_dir.glob("*.npz")):
        stem, grid, s = f.stem.rsplit("__", 2)
        index[(stem, grid, int(s[1:]))] = f
        stems.add(stem)
        sis.add(int(s[1:]))
    stems_l = sorted(stems)
    n_sig = max(sis) + 1
    need = [native, "reenc"] + edges + [f"repromote{e}" for e in edges]
    for g in need:
        miss = [st for st in stems_l for si in range(n_sig) if (st, g, si) not in index]
        if miss:
            raise SystemExit(f"arm {g}: missing residuals for {sorted(set(miss))}")
    sigmas = [
        round(float(np.load(index[(stems_l[0], native, si)])["sigma"]), 6)
        for si in range(n_sig)
    ]
    log.info(f"{len(stems_l)} images, σ={sigmas}, routes {edges}")

    pools: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    percos: dict[tuple, list[float]] = defaultdict(list)

    for st in stems_l:
        for si in range(n_sig):
            nat1f, nat2f = load_halves(index[(st, native, si)])
            re1f, re2f = load_halves(index[(st, "reenc", si)])
            for e in edges:
                dem1, dem2 = load_halves(index[(st, e, si)])
                rp1f, rp2f = load_halves(index[(st, f"repromote{e}", si)])
                hw = dem1.shape[-2:]
                d1 = dem1.double().flatten()
                d2 = dem2.double().flatten()
                for op in ("area", "bicubic"):
                    nat1 = down(nat1f, hw, op)
                    nat2 = down(nat2f, hw, op)
                    nat = 0.5 * (nat1 + nat2)
                    nhat = (nat / nat.norm()).double().flatten()
                    rp1 = down(rp1f, hw, op).double().flatten()
                    rp2 = down(rp2f, hw, op).double().flatten()
                    re1 = down(re1f, hw, op).double().flatten()
                    re2 = down(re2f, hw, op).double().flatten()
                    refs = {
                        "reenc": (0.5 * (re1 + re2), re1 - re2),
                        "native": (
                            nat.double().flatten(),
                            (nat1 - nat2).double().flatten(),
                        ),
                    }

                    def pp(v: torch.Tensor) -> torch.Tensor:
                        return v - (v @ nhat) * nhat

                    for ref_name, (ref, ref_diff) in refs.items():
                        B1p, B2p = pp(rp1 - ref), pp(rp2 - ref)
                        C1p, C2p = pp(d1 - rp1), pp(d2 - rp2)
                        rd = pp(ref_diff)
                        s = pools[(e, ref_name, op, si)]
                        s["bc_cross"] += 0.5 * float(B1p @ C2p + B2p @ C1p)
                        s["bc_same"] += 0.5 * float(B1p @ C1p + B2p @ C2p)
                        s["bb_cross"] += float(B1p @ B2p) - float(rd @ rd) / 4.0
                        s["cc_cross"] += float(C1p @ C2p)
                        s["bb12"] += float(B1p @ B2p)
                        s["bb1"] += float(B1p @ B1p)
                        s["bb2"] += float(B2p @ B2p)
                        s["cc12"] += float(C1p @ C2p)
                        s["cc1"] += float(C1p @ C1p)
                        s["cc2"] += float(C2p @ C2p)
                        s["n"] += 1
                        Bm, Cm = 0.5 * (B1p + B2p), 0.5 * (C1p + C2p)
                        nb, nc = float(Bm.norm()), float(Cm.norm())
                        if nb > 0 and nc > 0:
                            percos[(e, ref_name, op, si)].append(
                                float(Bm @ Cm) / (nb * nc)
                            )
        log.info(f"  pooled {st}")

    # ---- assemble ---------------------------------------------------------
    result: dict = {
        "run_dir": str(run),
        "sigma_centers": sigmas,
        "n_images": len(stems_l),
        "estimand": "measured pooled ρ_r on the demoted grid, area-downsample, "
        "⊥ to D_e·r̄_native, cross-half debiased (shared-rp + shared-ref "
        "corrections), reenc ref primary — the estimand of prediction_191.json",
        "rel_gate": REL_GATE,
        "verdict_sigma_min": VERDICT_SIGMA_MIN,
        "routes": {},
    }
    tables: dict[tuple, list[dict]] = {}
    for e in edges:
        blk: dict = {}
        for ref_name in ("reenc", "native"):
            for op in ("area", "bicubic"):
                key = (e, ref_name, op)
                tab = [
                    pooled_row(pools[key + (si,)], percos[key + (si,)])
                    for si in range(n_sig)
                ]
                tables[key] = tab
                blk[f"{ref_name}_{op}"] = {
                    "table": tab,
                    "crossings_sigma": crossings(
                        sigmas, [t["amp_ratio_BoverC"] for t in tab]
                    ),
                }
        result["routes"][e] = blk

    # ---- score against the frozen prediction (pre-registered rule) --------
    pred_path = E19 / "prediction_191.json"
    frozen_path = E19 / "frozen_target_190.json"
    pred = json.loads(pred_path.read_text()) if pred_path.exists() else None
    gtab = (
        json.loads(frozen_path.read_text())["ref_reenc"]
        if frozen_path.exists()
        else None
    )
    if pred is not None:
        fro = pred["sigma_centers"]
        si_map = {
            si: next((j for j, f in enumerate(fro) if abs(f - s) < 1e-6), None)
            for si, s in enumerate(sigmas)
        }
        for e in edges:
            bins = []
            for si, s in enumerate(sigmas):
                j = si_map[si]
                if s < VERDICT_SIGMA_MIN or j is None or e not in pred:
                    continue
                r = tables[(e, "reenc", "area")][si]
                rho = r["rho"]
                band = sorted(
                    pred[e][cl]["rho"][j] for cl in ("diag", "block", "octave")
                )
                gated = (
                    r["rel_cos_B"] == r["rel_cos_B"]
                    and r["rel_cos_C"] == r["rel_cos_C"]
                    and r["rel_cos_B"] >= REL_GATE
                    and r["rel_cos_C"] >= REL_GATE
                )
                if rho != rho:
                    cls, vs = "nan", None
                elif rho >= 0:
                    cls, vs = "sign_positive_closure_falsified", None
                else:
                    cls = (
                        "deep_residual_owns"
                        if abs(rho) >= DEEP_MIN
                        else "weak_derivable_seed"
                        if abs(rho) <= WEAK_MAX
                        else "intermediate"
                    )
                    lo, hi = abs(band[0]), abs(band[-1])
                    vs = (
                        "below_band"
                        if abs(rho) < min(lo, hi)
                        else "above_band"
                        if abs(rho) > max(lo, hi)
                        else "within_band"
                    )
                bins.append(
                    {
                        "sigma": s,
                        "rho_r_measured": rho,
                        "rho_r_closure_band": [band[0], band[-1]],
                        "rho_g_e14": gtab[e]["table"][j]["rho"] if gtab else None,
                        "rel_ok": gated,
                        "vs_band": vs,
                        "class": cls,
                    }
                )
            gated_cls = [b["class"] for b in bins if b["rel_ok"]]
            result["routes"][e]["verdict"] = {
                "bins": bins,
                "n_gated": len(gated_cls),
                "classes_gated": {
                    c: gated_cls.count(c) for c in sorted(set(gated_cls))
                },
            }
    else:
        log.info("prediction_191.json not found — tables only, no verdict scoring")

    out_path = run / "measured_192.json"
    out_path.write_text(json.dumps(result, indent=2))
    paths = [out_path]
    if args.write_record:
        rec = E19 / "measured_192.json"
        rec.write_text(json.dumps(result, indent=2))
        paths.append(rec)

    for e in edges:
        log.info(f"\n== route {e} (reenc ref, area) ==")
        log.info(
            "σ        ρ_r     ρ_same   amp     relB    relC    closure band     ρ_g"
        )
        vbins = {
            b["sigma"]: b
            for b in result["routes"][e].get("verdict", {}).get("bins", [])
        }
        for si, s in enumerate(sigmas):
            t = tables[(e, "reenc", "area")][si]
            v = vbins.get(s)
            band = (
                f"[{v['rho_r_closure_band'][0]:+.2f},{v['rho_r_closure_band'][1]:+.2f}]"
                if v
                else "       -"
            )
            gflag = "" if v is None else ("" if v["rel_ok"] else "  GATE-FAIL")
            rho_g = f"{v['rho_g_e14']:+.3f}" if v and v["rho_g_e14"] else "     -"
            log.info(
                f"{s:<7} {t['rho']:+.3f}  {t['rho_sameset_biascheck']:+.3f}   "
                f"{t['amp_ratio_BoverC']:.3f}   {t['rel_cos_B']:.3f}   "
                f"{t['rel_cos_C']:.3f}   {band}   {rho_g}{gflag}"
            )
        if "verdict" in result["routes"][e]:
            log.info(
                f"gated verdict classes: "
                f"{result['routes'][e]['verdict']['classes_gated']}"
            )
    log.info("\nwrote " + " + ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
