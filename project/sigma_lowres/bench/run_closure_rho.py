#!/usr/bin/env python
"""E19.1 — closure-predicted paired branch ledger ρ_r (theory-first).

Extends E17's Gaussian-closure machinery (``run_posterior_closure.py``) to
emit PAIRED branch predictions at the residual level, per (route, σ), on the
g-ledger's own grid: tier-1024 native, routes 1024→{896, 768, 512}, E14's
probe-matched 40-image list as holdout, σ = E14's 15 bin centers. The
prediction is frozen (committed to ``experiments/e19/``) BEFORE the measured
19.2 run is submitted — theory predicts, the instrument scores.

**Estimand (pre-registered; the A_e pin).** All vectors live on the demoted
(lo) grid of each (image, route); native-grid fields are brought down by the
instrument's area-downsample D_e (``F.interpolate(mode="area")`` — the same
operator as ``run_prior_distance.py``'s scalar read and E11's
``resid_structure.py``). With r̂_arm the closure-predicted mean residual:

    B_r = D_e r̂_rp − D_e r̂_ref     (data branch;  ref ∈ {native, reenc})
    C_r = r̂_dem − D_e r̂_rp         (graph branch)
    B_r + C_r = r̂_dem − D_e r̂_ref  (exact — the measured mismatch object)

so the split decomposes exactly the object the prior-distance probe
measures, mirroring the g-ledger's exactness (B + C = ḡ_dem − ḡ_native).
⊥ projects out the direction of D_e r̂_native per image (the r-space analog
of the ledger's P⊥ against ĝ). Population pooling is by summed inner
products across images (the concatenated-vector statistic):

    ρ_r(σ, e) = Σ_i⟨B⊥_i, C⊥_i⟩ / √(Σ_i|B⊥_i|² · Σ_i|C⊥_i|²)

— per-image directions are idiosyncratic (E11 norm-only), so per-image
cosines are reported as a secondary descriptive read only. The debiased
amplitude ratio √(Σ|B⊥|²/Σ|C⊥|²) is emitted per bin, so the closure also
predicts the crossing location per route.

**Coherence criterion (pre-registered).** The prediction counts as coherent
iff all three closures (diag / block / octave) agree in ρ_r sign per bin
across the mid/high-σ verdict bins (σ ≥ 0.3); the frozen profile is the
three-closure band. Low-σ bins are reported but not verdict-bearing (E17's
recorded low-σ caution).

**Controls.**

- *Cross-fit* (the shared-arm analog of the g-ledger's ``I_sameset`` bias
  check): B_r and C_r share the repromote arm's fitted model, whose
  estimation error enters both with opposite signs and biases a naive ρ_r
  negative. The fit split is halved; B_r uses half-A's repromote model while
  C_r uses half-B's (other arms full-fit), symmetrized over the swap —
  reported as ``rho_crossfit`` (block closure) against the same-model value.
- *Operator sensitivity*: the ledger re-pooled with bicubic instead of area
  downsampling (same predicted fields) — if ρ_r moves materially, the
  operator, not the model, owns part of the verdict.
- *reenc ref*: both data refs emitted, matching the E14 ledger pair
  (``ledger.json`` = reenc ref primary, ``ledger_native.json`` = native).

GPU is a one-time VAE encode of the demote/repromote/reenc arms into
``probe1024_closure/closure_latents/``; warm-cache runs are CPU-only.

Usage::

    make daemon-run ARGS="project/sigma_lowres/bench/run_closure_rho.py \
        --label e19-closure-rho"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402
from project.sigma_lowres.bench.run_posterior_closure import (  # noqa: E402
    ArmModel,
    dct2,
    idct2,
    predict_residuals,
)
from project.sigma_lowres.bench.tier_routing.redundancy import (  # noqa: E402
    score_corpus,
    select_probe_set,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

SIGMA = Path(__file__).resolve().parents[1]
PAPER_BENCH = SIGMA / "paper_bench"
E19 = PAPER_BENCH / "experiments" / "e19"
PROBE_LIST = PAPER_BENCH / "experiments" / "e13" / "e1b_probe_list.json"
FROZEN_TARGET = E19 / "frozen_target_190.json"

CLOSURES = ("diag", "block", "octave")
CONTROL_CLOSURE = "block"
VERDICT_SIGMA_MIN = 0.3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--vae", default=None)
    p.add_argument("--tier", type=int, default=1024)
    p.add_argument("--edges", default="896,768,512")
    p.add_argument("--num_fit", type=int, default=40)
    p.add_argument(
        "--data_root",
        default=str(Path(__file__).resolve().parent / "probe1024_closure"),
        help="cache root for the one-time closure latent encodes",
    )
    p.add_argument("--n_bands", type=int, default=10)
    p.add_argument("--shrink_R", type=float, default=0.2)
    p.add_argument("--clip_pc", type=float, default=0.6)
    p.add_argument("--quant_k", type=int, default=4)
    p.add_argument("--label", default=None)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="2 holdout + 4 fit images, one edge; never writes the frozen "
        "prediction file",
    )
    return p.parse_args()


# --------------------------------------------------------------------------
# latents (native from the lora cache; demote/repromote/reenc encoded once)
# --------------------------------------------------------------------------


def ensure_latents(records, edges, vae_path, cache_root: Path):
    from library.io.cache import load_cached_latents

    cache = cache_root / "closure_latents"
    cache.mkdir(parents=True, exist_ok=True)
    arms = [f"demote{e}" for e in edges] + [f"repromote{e}" for e in edges]
    arms.append("reenc")
    missing = [
        r
        for r in records
        if not all((cache / f"{r.stem}__{a}.npz").exists() for a in arms)
    ]
    if missing:
        log.info(f"VAE-encoding {len(missing)} images × {len(arms)} arms (one-time)")
        from project.sigma_lowres.bench.tier_routing.run_grad_probe import (
            VAE,
            encode_probe_latents,
        )

        extra = encode_probe_latents(
            missing,
            edges,
            vae_path or VAE,
            torch.device("cuda"),
            True,
            repromote=True,
        )
        for (stem, arm), lat in extra.items():
            np.savez_compressed(
                cache / f"{stem}__{arm}.npz", latent=lat.numpy().astype(np.float32)
            )
    out = {}
    for r in records:
        out[(r.stem, "native")] = (
            load_cached_latents(r.npz_path)[0].numpy().astype(np.float32)
        )
        for a in arms:
            out[(r.stem, a)] = np.load(cache / f"{r.stem}__{a}.npz")["latent"]
    return out


# --------------------------------------------------------------------------
# pooled paired ledger
# --------------------------------------------------------------------------


def down_to(x: np.ndarray, hw: tuple[int, int], mode: str) -> np.ndarray:
    if x.shape[-2:] == tuple(hw):
        return x
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0)
    kw = {} if mode == "area" else {"align_corners": False}
    return F.interpolate(t, size=hw, mode=mode, **kw).squeeze(0).numpy()


def _perp(v: np.ndarray, nhat: np.ndarray) -> np.ndarray:
    return v - float(v.flatten() @ nhat.flatten()) * nhat


class PairedPool:
    """Accumulates Σ⟨B⊥,C⊥⟩, Σ|B⊥|², Σ|C⊥|² (+ raw) over images."""

    def __init__(self):
        self.s = defaultdict(float)
        self.per_img_cos: list[float] = []

    def add(self, b: np.ndarray, c: np.ndarray, nat: np.ndarray) -> None:
        nn = float(np.linalg.norm(nat))
        nhat = nat / nn if nn > 0 else nat
        bp, cp = _perp(b, nhat), _perp(c, nhat)
        self.s["bc"] += float(bp.flatten() @ cp.flatten())
        self.s["bb"] += float((bp * bp).sum())
        self.s["cc"] += float((cp * cp).sum())
        self.s["bc_raw"] += float(b.flatten() @ c.flatten())
        self.s["bb_raw"] += float((b * b).sum())
        self.s["cc_raw"] += float((c * c).sum())
        nb, nc = float(np.linalg.norm(bp)), float(np.linalg.norm(cp))
        if nb > 0 and nc > 0:
            self.per_img_cos.append(float(bp.flatten() @ cp.flatten()) / (nb * nc))

    def row(self) -> dict:
        s = self.s
        den = (s["bb"] * s["cc"]) ** 0.5
        den_raw = (s["bb_raw"] * s["cc_raw"]) ** 0.5
        pic = np.array(self.per_img_cos)
        return {
            "rho": round(s["bc"] / den, 5) if den > 0 else float("nan"),
            "rho_raw": round(s["bc_raw"] / den_raw, 5) if den_raw > 0 else float("nan"),
            "amp_ratio_BoverC": round((s["bb"] / s["cc"]) ** 0.5, 5)
            if s["cc"] > 0
            else float("nan"),
            "per_image_cos_mean": round(float(pic.mean()), 5) if len(pic) else None,
            "per_image_cos_sd": round(float(pic.std(ddof=1)), 5)
            if len(pic) > 1
            else None,
        }


def crossings(sigmas: list[float], ratios: list[float]) -> list[float]:
    pts = [(s, r) for s, r in zip(sigmas, ratios) if r == r]
    out = []
    for (s0, r0), (s1, r1) in zip(pts[:-1], pts[1:]):
        if (r0 - 1.0) * (r1 - 1.0) < 0:
            out.append(round(s0 + (1.0 - r0) / (r1 - r0) * (s1 - s0), 4))
    return out


# --------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    start_heartbeat()
    edges = [int(e) for e in args.edges.split(",") if e]
    sigmas = json.loads(FROZEN_TARGET.read_text())["sigma_centers"]
    log.info(f"σ grid from frozen target: {sigmas}")

    # corpus: E14's exact probe list as holdout, stratified disjoint fit split
    probe_keys = {
        (p["artist"], p["stem"]) for p in json.loads(PROBE_LIST.read_text())["images"]
    }
    records = [
        r for r in score_corpus(k=args.quant_k) if r.complete() and r.tier == args.tier
    ]
    holdout = [r for r in records if (r.artist, r.stem) in probe_keys]
    if len(holdout) != len(probe_keys):
        raise SystemExit(f"probe list only {len(holdout)}/{len(probe_keys)} matched")
    rest = [r for r in records if (r.artist, r.stem) not in probe_keys]
    if args.smoke:
        holdout = holdout[:2]
        edges = edges[:1]
        args.num_fit = 4
    fit = select_probe_set(rest, args.num_fit, tier=args.tier)
    log.info(f"holdout (E14 probe): {len(holdout)}, fit: {len(fit)}")

    lat = ensure_latents(holdout + fit, edges, args.vae, Path(args.data_root))

    arms = (
        ["native", "reenc"]
        + [f"demote{e}" for e in edges]
        + [f"repromote{e}" for e in edges]
    )
    fit_sorted = sorted(fit, key=lambda r: r.stem)
    halves = (fit_sorted[0::2], fit_sorted[1::2])

    log.info("fitting arm models (full + 2 half splits for the rp cross-fit)")
    models: dict[str, ArmModel] = {}
    for a in arms:
        coeffs = [dct2(lat[(r.stem, a)]) for r in fit_sorted]
        models[a] = ArmModel.fit(coeffs, args.n_bands, args.shrink_R, args.clip_pc)
    half_rp: list[dict[str, ArmModel]] = []
    for hf in halves:
        hm = {}
        for e in edges:
            a = f"repromote{e}"
            coeffs = [dct2(lat[(r.stem, a)]) for r in hf]
            hm[a] = ArmModel.fit(coeffs, args.n_bands, args.shrink_R, args.clip_pc)
        half_rp.append(hm)

    run_dir = make_run_dir(
        "sigma_lowres", args.label, root=Path(__file__).resolve().parent / "results"
    )
    chain_cache: dict = {}

    # pools[(split, closure, ref, op, variant, edge, si)] -> PairedPool
    pools: dict[tuple, PairedPool] = defaultdict(PairedPool)

    for split, subset in (("holdout", holdout), ("fit", fit_sorted)):
        for i, r in enumerate(subset):
            coeff = {a: dct2(lat[(r.stem, a)]) for a in arms}
            for cl in CLOSURES:
                pred = {
                    a: [
                        idct2(x).astype(np.float32)
                        for x in predict_residuals(
                            coeff[a], models[a], sigmas, args.n_bands, cl, chain_cache
                        )
                    ]
                    for a in arms
                }
                rp_half = {}
                if cl == CONTROL_CLOSURE:
                    for hi_, hm in enumerate(half_rp):
                        for e in edges:
                            a = f"repromote{e}"
                            rp_half[(hi_, a)] = [
                                idct2(x).astype(np.float32)
                                for x in predict_residuals(
                                    coeff[a],
                                    hm[a],
                                    sigmas,
                                    args.n_bands,
                                    cl,
                                    chain_cache,
                                )
                            ]
                for e in edges:
                    lo_hw = lat[(r.stem, f"demote{e}")].shape[-2:]
                    for si in range(len(sigmas)):
                        dem = pred[f"demote{e}"][si]
                        for op in ("area", "bicubic"):
                            if op == "bicubic" and cl != CONTROL_CLOSURE:
                                continue
                            nat = down_to(pred["native"][si], lo_hw, op)
                            rp = down_to(pred[f"repromote{e}"][si], lo_hw, op)
                            for ref_name in ("reenc", "native"):
                                ref = (
                                    nat
                                    if ref_name == "native"
                                    else down_to(pred["reenc"][si], lo_hw, op)
                                )
                                pools[(split, cl, ref_name, op, "full", e, si)].add(
                                    rp - ref, dem - rp, nat
                                )
                            if cl == CONTROL_CLOSURE and op == "area":
                                # cross-fit: B from half-A's rp model, C from
                                # half-B's (and the swap), reenc ref
                                ref = down_to(pred["reenc"][si], lo_hw, op)
                                for ha, hb in ((0, 1), (1, 0)):
                                    rp_b = down_to(
                                        rp_half[(ha, f"repromote{e}")][si], lo_hw, op
                                    )
                                    rp_c = down_to(
                                        rp_half[(hb, f"repromote{e}")][si], lo_hw, op
                                    )
                                    pools[
                                        (split, cl, "reenc", op, "crossfit", e, si)
                                    ].add(rp_b - ref, dem - rp_c, nat)
            log.info(f"  [{split} {i + 1}/{len(subset)}] {r.artist}/{r.stem}")

    # ---- assemble ---------------------------------------------------------
    result: dict = {
        "sigma_centers": sigmas,
        "edges": edges,
        "n_holdout": len(holdout),
        "n_fit": len(fit_sorted),
        "closures": list(CLOSURES),
        "control_closure": CONTROL_CLOSURE,
        "verdict_sigma_min": VERDICT_SIGMA_MIN,
    }
    for split in ("holdout", "fit"):
        blk: dict = {}
        for cl in CLOSURES:
            for ref_name in ("reenc", "native"):
                for op in ("area", "bicubic"):
                    if op == "bicubic" and cl != CONTROL_CLOSURE:
                        continue
                    for variant in ("full", "crossfit"):
                        if variant == "crossfit" and (
                            cl != CONTROL_CLOSURE or ref_name != "reenc" or op != "area"
                        ):
                            continue
                        for e in edges:
                            key = (split, cl, ref_name, op, variant, e)
                            if (key + (0,)) not in pools:
                                continue
                            tab = [
                                pools[key + (si,)].row() for si in range(len(sigmas))
                            ]
                            name = f"{cl}_{ref_name}_{op}_{variant}_{e}"
                            blk[name] = {
                                "table": tab,
                                "crossings_sigma": crossings(
                                    sigmas,
                                    [t["amp_ratio_BoverC"] for t in tab],
                                ),
                            }
        result[split] = blk

    # coherence: all three closures agree in sign on verdict bins (holdout,
    # reenc ref, area, full)
    coher: dict = {}
    for e in edges:
        rows = []
        for si, s in enumerate(sigmas):
            if s < VERDICT_SIGMA_MIN:
                continue
            signs = {
                cl: np.sign(
                    result["holdout"][f"{cl}_reenc_area_full_{e}"]["table"][si]["rho"]
                )
                for cl in CLOSURES
            }
            rows.append(
                {
                    "sigma": s,
                    "signs": {k: int(v) for k, v in signs.items()},
                    "agree": len({v for v in signs.values()}) == 1,
                }
            )
        coher[str(e)] = {
            "bins": rows,
            "coherent": all(r["agree"] for r in rows),
        }
    result["coherence"] = coher

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=result,
        label=args.label,
        extra={
            "holdout": sorted(r.stem for r in holdout),
            "fit": sorted(r.stem for r in fit_sorted),
        },
    )

    # frozen-prediction summary for the committed record
    summary: dict = {
        "run_dir": str(run_dir),
        "sigma_centers": sigmas,
        "estimand": "pooled ρ_r on the demoted grid, area-downsample, ⊥ to "
        "D_e·r̂_native, reenc ref primary; see run_closure_rho.py docstring",
        "coherence": coher,
    }
    for e in edges:
        summary[str(e)] = {
            cl: {
                "rho": [
                    t["rho"]
                    for t in result["holdout"][f"{cl}_reenc_area_full_{e}"]["table"]
                ],
                "amp_ratio": [
                    t["amp_ratio_BoverC"]
                    for t in result["holdout"][f"{cl}_reenc_area_full_{e}"]["table"]
                ],
                "crossings": result["holdout"][f"{cl}_reenc_area_full_{e}"][
                    "crossings_sigma"
                ],
            }
            for cl in CLOSURES
        }
        summary[str(e)]["controls"] = {
            "rho_crossfit": [
                t["rho"]
                for t in result["holdout"][
                    f"{CONTROL_CLOSURE}_reenc_area_crossfit_{e}"
                ]["table"]
            ],
            "rho_bicubic": [
                t["rho"]
                for t in result["holdout"][f"{CONTROL_CLOSURE}_reenc_bicubic_full_{e}"][
                    "table"
                ]
            ],
        }
    pred_path = E19 / "prediction_191.json"
    if args.smoke:
        pred_path = run_dir / "prediction_191_smoke.json"
    pred_path.write_text(json.dumps(summary, indent=2))

    for e in edges:
        log.info(f"\n== route {e} (holdout, reenc ref, area, full) ==")
        log.info("σ        diag ρ   block ρ  octave ρ  amp(block)  crossfit  bicubic")
        for si, s in enumerate(sigmas):
            row = [
                result["holdout"][f"{cl}_reenc_area_full_{e}"]["table"][si]["rho"]
                for cl in CLOSURES
            ]
            amp = result["holdout"][f"block_reenc_area_full_{e}"]["table"][si][
                "amp_ratio_BoverC"
            ]
            cf = result["holdout"][f"block_reenc_area_crossfit_{e}"]["table"][si]["rho"]
            bc = result["holdout"][f"block_reenc_bicubic_full_{e}"]["table"][si]["rho"]
            log.info(
                f"{s:<7} {row[0]:+.3f}   {row[1]:+.3f}   {row[2]:+.3f}"
                f"    {amp:>7.3f}    {cf:+.3f}    {bc:+.3f}"
            )
    log.info(f"\nresult → {run_dir}\nfrozen prediction → {pred_path}")


if __name__ == "__main__":
    main()
