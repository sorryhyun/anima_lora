#!/usr/bin/env python
"""E19.6 — does the shipped LoRA move the data leg? Operating-point diff of the
r-level ledger: base-DiT store (19.2) vs adapter-merged store (same probe,
``run_prior_distance.py --adapter``), per (route, σ).

The g-level ledger (E14) is measured AT the sincos operating point while the
r-level ledger (19.2) is base-DiT — the two operating points have never been
compared at matched estimand. r̄ lives in x-space, so cross-operating-point
direction comparisons are well-defined (unlike g, whose coordinates are
adapter-specific). Per (route e, σ, leg L ∈ {B, C, net}) plus the native
residual field itself (natres, per σ):

    dircos(L) = Σ_i⟨L̄_a⊥, L̄_b⊥⟩ / √(Σ_i debias|L_a⊥|² · Σ_i debias|L_b⊥|²)
    amp_ratio(L) = √(Σ debias|L_b⊥|² / Σ debias|L_a⊥|²)      (b/a = sincos/base)

on the demoted grid, area downsample, both sides projected ⊥ to store-a's
(base) native residual direction — one common subspace (own-projector and raw
controls; bicubic + native-ref controls mirror ``ledger_rho_r.py``). The
numerator is unbiased (draw noise independent across runs); each side's second
moment is cross-half debiased with the shared-ref correction for B and net.
Reliability gates (rel_cos ≥ 0.5 per side) carry over.

**Pre-registered reading (verdict bins σ ≥ 0.3, reenc/area, gated):** the
LoRA-moves-B account predicts dircos_B < dircos_C − 0.2 and/or
|log amp_ratio_B| ≫ |log amp_ratio_C|; dircos ≈ 1 and amp ≈ 1 for both legs →
leg-level operating-point invariance (extends E7's map-level null and closes
19.5's cross-operating-point caveat); natres dircos low with B/C dircos high →
the LoRA moves the score field mostly INSIDE the span the ⊥/legs discard.

CPU-only. Usage::

    python project/sigma_lowres/bench/ledger_operating_point.py \
        --run_a <base 19.2 run dir> --run_b <adapter run dir> [--write_record]
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

from bench._common import make_run_dir, write_result  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

E19 = Path(__file__).resolve().parents[1] / "paper_bench" / "experiments" / "e19"

REL_GATE = 0.5
VERDICT_SIGMA_MIN = 0.3
LEGS = ("B", "C", "net")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--run_a", required=True, help="base-DiT 19.2 run (residuals/)")
    p.add_argument("--run_b", required=True, help="adapter-merged run (residuals/)")
    p.add_argument("--tier", type=int, default=1024)
    p.add_argument("--edges", default="896,768,512")
    p.add_argument("--label", default=None)
    p.add_argument(
        "--write_record",
        action="store_true",
        help="also write opdiff_196.json into experiments/e19/",
    )
    return p.parse_args()


def down(x: torch.Tensor, hw: tuple[int, int], op: str) -> torch.Tensor:
    if x.shape[-2:] == tuple(hw):
        return x
    kw = {} if op == "area" else {"align_corners": False}
    return F.interpolate(x.unsqueeze(0), size=hw, mode=op, **kw).squeeze(0)


def index_store(run: Path) -> tuple[dict, list[str], int]:
    res = run / "residuals"
    if not res.is_dir():
        raise SystemExit(f"no residuals/ under {run}")
    idx: dict[tuple[str, str, int], Path] = {}
    stems, sis = set(), set()
    for f in sorted(res.glob("*.npz")):
        stem, grid, s = f.stem.rsplit("__", 2)
        idx[(stem, grid, int(s[1:]))] = f
        stems.add(stem)
        sis.add(int(s[1:]))
    return idx, sorted(stems), max(sis) + 1


def load_halves(f: Path) -> tuple[torch.Tensor, torch.Tensor]:
    z = np.load(f)
    return (
        torch.from_numpy(z["m1"].astype(np.float32)),
        torch.from_numpy(z["m2"].astype(np.float32)),
    )


def legs_for(
    idx: dict, st: str, native: str, e: str, si: int, hw, op: str, ref_name: str
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Per-store half-pairs of each leg on the lo grid + (ref parity diff, nat)."""
    nat1f, nat2f = load_halves(idx[(st, native, si)])
    re1f, re2f = load_halves(idx[(st, "reenc", si)])
    dem1, dem2 = load_halves(idx[(st, e, si)])
    rp1f, rp2f = load_halves(idx[(st, f"repromote{e}", si)])
    nat = down(0.5 * (nat1f + nat2f), hw, op)
    d1, d2 = dem1.double().flatten(), dem2.double().flatten()
    rp1 = down(rp1f, hw, op).double().flatten()
    rp2 = down(rp2f, hw, op).double().flatten()
    if ref_name == "reenc":
        ref = down(0.5 * (re1f + re2f), hw, op).double().flatten()
        rd = down(re1f - re2f, hw, op).double().flatten()
    else:
        ref = nat.double().flatten()
        rd = down(nat1f - nat2f, hw, op).double().flatten()
    halves = {
        "B": (rp1 - ref, rp2 - ref),
        "C": (d1 - rp1, d2 - rp2),
        "net": (d1 - ref, d2 - ref),
    }
    return halves, rd, nat.double().flatten()


def pooled_row(s: dict[str, float], pic: dict[str, list[float]]) -> dict:
    out: dict = {"n_images": int(s["n"])}
    rel_ok = True
    for leg in LEGS:
        ma, mb = s[f"{leg}_mm_a"], s[f"{leg}_mm_b"]
        den = math.sqrt(max(ma, 0.0) * max(mb, 0.0))
        rels = []
        for side in ("a", "b"):
            m1, m2 = s[f"{leg}_m1_{side}"], s[f"{leg}_m2_{side}"]
            rels.append(
                s[f"{leg}_m12_{side}"] / math.sqrt(m1 * m2)
                if m1 > 0 and m2 > 0
                else float("nan")
            )
        pa = np.array(pic[leg])
        out[f"dircos_{leg}"] = (
            round(s[f"{leg}_ab"] / den, 5) if den > 0 else float("nan")
        )
        out[f"amp_ratio_{leg}_BoverA"] = (
            round(math.sqrt(mb / ma), 5) if ma > 0 and mb > 0 else float("nan")
        )
        out[f"rel_cos_{leg}_a"] = round(rels[0], 5)
        out[f"rel_cos_{leg}_b"] = round(rels[1], 5)
        out[f"per_image_cos_{leg}_mean"] = (
            round(float(pa.mean()), 5) if len(pa) else None
        )
        if leg != "net":
            rel_ok = rel_ok and all(r == r and r >= REL_GATE for r in rels)
    out["rel_ok"] = bool(rel_ok)
    return out


def main() -> None:
    args = parse_args()
    run_a, run_b = Path(args.run_a).resolve(), Path(args.run_b).resolve()
    idx_a, stems_a, nsig_a = index_store(run_a)
    idx_b, stems_b, nsig_b = index_store(run_b)
    if stems_a != stems_b or nsig_a != nsig_b:
        raise SystemExit("stores disagree on stems or σ grid — not probe-matched")
    native = str(args.tier)
    edges = [e for e in args.edges.split(",") if e]
    n_sig = nsig_a
    sigmas = [
        round(float(np.load(idx_a[(stems_a[0], native, si)])["sigma"]), 6)
        for si in range(n_sig)
    ]
    sig_b = [
        round(float(np.load(idx_b[(stems_b[0], native, si)])["sigma"]), 6)
        for si in range(n_sig)
    ]
    if sigmas != sig_b:
        raise SystemExit("σ grids differ between stores")
    log.info(f"{len(stems_a)} images, σ={sigmas}, routes {edges}")

    variants = [
        ("reenc", "area", "a_nat"),
        ("reenc", "area", "own"),
        ("reenc", "area", "none"),
        ("reenc", "bicubic", "a_nat"),
        ("native", "area", "a_nat"),
    ]

    pools: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    percos: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    natpool: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for st in stems_a:
        for si in range(n_sig):
            # native residual field diff (no demotion involved), native grid
            na1, na2 = load_halves(idx_a[(st, native, si)])
            nb1, nb2 = load_halves(idx_b[(st, native, si)])
            ma = 0.5 * (na1 + na2).double().flatten()
            mb = 0.5 * (nb1 + nb2).double().flatten()
            np_ = natpool[si]
            np_["ab"] += float(ma @ mb)
            np_["mm_a"] += float(na1.double().flatten() @ na2.double().flatten())
            np_["mm_b"] += float(nb1.double().flatten() @ nb2.double().flatten())
            np_["n"] += 1
            for e in edges:
                hw = load_halves(idx_a[(st, e, si)])[0].shape[-2:]
                for ref_name, op, proj in variants:
                    ha, rd_a, nat_a = legs_for(
                        idx_a, st, native, e, si, hw, op, ref_name
                    )
                    hb, rd_b, nat_b = legs_for(
                        idx_b, st, native, e, si, hw, op, ref_name
                    )
                    if proj == "none":
                        nhat_a = nhat_b = None
                    elif proj == "own":
                        nhat_a = nat_a / nat_a.norm()
                        nhat_b = nat_b / nat_b.norm()
                    else:  # common subspace: base store's native direction
                        nhat_a = nhat_b = nat_a / nat_a.norm()

                    def pp(v, nhat):
                        return v if nhat is None else v - (v @ nhat) * nhat

                    s = pools[(e, ref_name, op, proj, si)]
                    pc = percos[(e, ref_name, op, proj, si)]
                    rda, rdb = pp(rd_a, nhat_a), pp(rd_b, nhat_b)
                    shared_ref = {"B": True, "C": False, "net": True}
                    for leg in LEGS:
                        a1, a2 = (pp(v, nhat_a) for v in ha[leg])
                        b1, b2 = (pp(v, nhat_b) for v in hb[leg])
                        ma_l, mb_l = 0.5 * (a1 + a2), 0.5 * (b1 + b2)
                        corr_a = float(rda @ rda) / 4.0 if shared_ref[leg] else 0.0
                        corr_b = float(rdb @ rdb) / 4.0 if shared_ref[leg] else 0.0
                        s[f"{leg}_ab"] += float(ma_l @ mb_l)
                        s[f"{leg}_mm_a"] += float(a1 @ a2) - corr_a
                        s[f"{leg}_mm_b"] += float(b1 @ b2) - corr_b
                        s[f"{leg}_m12_a"] += float(a1 @ a2)
                        s[f"{leg}_m1_a"] += float(a1 @ a1)
                        s[f"{leg}_m2_a"] += float(a2 @ a2)
                        s[f"{leg}_m12_b"] += float(b1 @ b2)
                        s[f"{leg}_m1_b"] += float(b1 @ b1)
                        s[f"{leg}_m2_b"] += float(b2 @ b2)
                        na_, nb_ = float(ma_l.norm()), float(mb_l.norm())
                        if na_ > 0 and nb_ > 0:
                            pc[leg].append(float(ma_l @ mb_l) / (na_ * nb_))
                    s["n"] += 1
        log.info(f"  pooled {st}")

    run_dir = make_run_dir(
        "sigma_lowres", args.label, root=Path(__file__).resolve().parent / "results"
    )
    result: dict = {
        "run_a_base": str(run_a),
        "run_b_adapter": str(run_b),
        "sigma_centers": sigmas,
        "n_images": len(stems_a),
        "estimand": "cross-operating-point pooled direction cosine + amplitude "
        "ratio per r-level leg, demoted grid, area, both sides ⊥ to base-store "
        "native direction (common subspace), cross-half debiased second moments, "
        "reenc ref primary",
        "rel_gate": REL_GATE,
        "verdict_sigma_min": VERDICT_SIGMA_MIN,
        "natres": [],
        "routes": {},
    }
    for si in range(n_sig):
        np_ = natpool[si]
        den = math.sqrt(max(np_["mm_a"], 0.0) * max(np_["mm_b"], 0.0))
        result["natres"].append(
            {
                "sigma": sigmas[si],
                "dircos": round(np_["ab"] / den, 5) if den > 0 else float("nan"),
                "amp_ratio_BoverA": round(math.sqrt(np_["mm_b"] / np_["mm_a"]), 5)
                if np_["mm_a"] > 0 and np_["mm_b"] > 0
                else float("nan"),
            }
        )
    for e in edges:
        blk: dict = {}
        for ref_name, op, proj in variants:
            tab = [
                pooled_row(
                    pools[(e, ref_name, op, proj, si)],
                    percos[(e, ref_name, op, proj, si)],
                )
                for si in range(n_sig)
            ]
            blk[f"{ref_name}_{op}_{proj}"] = {"table": tab}
        rows = []
        for si, s_ in enumerate(sigmas):
            if s_ < VERDICT_SIGMA_MIN:
                continue
            t = blk["reenc_area_a_nat"]["table"][si]
            rows.append(
                {
                    "sigma": s_,
                    "dircos_B": t["dircos_B"],
                    "dircos_C": t["dircos_C"],
                    "gap_CminusB": round(t["dircos_C"] - t["dircos_B"], 5)
                    if t["dircos_B"] == t["dircos_B"] and t["dircos_C"] == t["dircos_C"]
                    else float("nan"),
                    "amp_B": t["amp_ratio_B_BoverA"],
                    "amp_C": t["amp_ratio_C_BoverA"],
                    "rel_ok": t["rel_ok"],
                }
            )
        gated = [r for r in rows if r["rel_ok"]]
        blk["verdict"] = {
            "bins": rows,
            "n_gated": len(gated),
            "median_gap_CminusB_gated": round(
                float(np.median([r["gap_CminusB"] for r in gated])), 5
            )
            if gated
            else None,
        }
        result["routes"][e] = blk

    write_result(run_dir, script=__file__, args=args, metrics=result, label=args.label)
    if args.write_record:
        (E19 / "opdiff_196.json").write_text(json.dumps(result, indent=2))

    log.info("\nnatres (native residual field, base vs adapter):")
    for r in result["natres"]:
        log.info(
            f"  σ={r['sigma']:<7} dircos {r['dircos']:+.3f}  amp {r['amp_ratio_BoverA']:.3f}"
        )
    for e in edges:
        log.info(f"\n== route {e} (reenc, area, ⊥ base-native) ==")
        log.info("σ        B:cos   amp     C:cos   amp     net:cos amp")
        for si, s_ in enumerate(sigmas):
            t = result["routes"][e]["reenc_area_a_nat"]["table"][si]
            line = (
                f"{s_:<7} {t['dircos_B']:+.3f}  {t['amp_ratio_B_BoverA']:.3f}   "
                f"{t['dircos_C']:+.3f}  {t['amp_ratio_C_BoverA']:.3f}   "
                f"{t['dircos_net']:+.3f} {t['amp_ratio_net_BoverA']:.3f}"
            )
            line += "" if t["rel_ok"] else "  GATE-FAIL"
            log.info(line)
        log.info(
            f"median gated gap C−B: "
            f"{result['routes'][e]['verdict']['median_gap_CminusB_gated']}"
        )
    log.info(f"\nresult → {run_dir}")


if __name__ == "__main__":
    main()
