#!/usr/bin/env python
"""sigma_lowres — latent-only Gaussian-closure test of the residual identity (E17).

Under the Bayes-optimal flow-matching field the fixed-image mean residual is
``r̄*(σ; x) = σ·Q(σ)⁻¹(x − μ)`` for a Gaussian clean-latent model
``x|c ~ N(μ, C)``, ``Q(σ) = (1−σ)²C + σ²I`` (paper Appendix: the mean residual
as a posterior operator). This script asks whether any *second-order* closure
of that identity reproduces the measured cross-grid mean-residual mismatch
curve — the route-uniform, monotone-declining relative-L2 excess of the
``run_prior_distance.py`` probe — from paired clean latents ONLY: no DiT
forward, no denoiser. Three nested closures:

- ``diag``  — variance per (radial band, channel); diagonal in DCT modes.
  This is the destroyed-band Wiener null the paper already falsified at the
  measured level; here it gets its honest quantitative shot.
- ``block`` — adds a within-band 16×16 channel covariance (shrunk).
- ``octave`` — adds parent–child cross-band coupling between DCT modes
  (i,j) and (2i,2j) per channel — the cheapest wavelet-style non-diagonal
  structure, solved exactly along disjoint octave chains.

Covariances are estimated on the fit split (all complete tier-1280 records
NOT in the probe set) with band pooling + shrinkage; predictions are scored
on the held-out probe images — the same 16 images behind the measured curves
(``--measured`` run dir). The estimand is replicated exactly: per image and
adjacent pair, the hi-grid predicted residual is area-downsampled to the lo
grid and compared with relative L2 (no split-half floor: predictions are
analytic means). The clean-latent model is caption-MARGINAL (pooled over
images), a stated approximation: the true posterior is caption-conditional.

GPU use: a one-time VAE encode of the demoted/reenc arms, cached under
``<data_root>/closure_latents/``; with a warm cache the run is CPU-only.

Pre-registered reads (paper_bench/experiments/e17/README.md):

- **shape**: per route, Pearson r over σ between predicted and measured
  excess; the measured curve declines 0.89 → 0.36.
- **route-uniformity**: the measured curves are route-uniform (spread
  ≤ ~0.02–0.05); a destroyed-band-ordered prediction fails this.
- **amplitude**: RMSE of predicted vs measured excess over σ, held out.
- **verdict**: a closure PASSES iff mean held-out RMSE ≤ 0.05 AND its max
  route spread ≤ 2× the measured max spread. Any pass replaces part of the
  empirical curve with a data-only derivation; all-fail establishes that
  higher-order/non-Gaussian posterior structure or the frozen model's
  approximation error is essential — the measured curve stays the minimal
  honest closure either way.

Usage::

    make daemon-run ARGS="project/sigma_lowres/bench/run_posterior_closure.py \
        --label e17-closure"
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
import scipy.fft  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402
from project.sigma_lowres.bench.tier_routing.redundancy import (  # noqa: E402
    ImageRecord,
    score_corpus,
    select_probe_set,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

CLOSURES = ("diag", "block", "octave")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--vae", default=None, help="default: tier_routing VAE path")
    p.add_argument("--tier", type=int, default=1280)
    p.add_argument("--edges", default="1024,896,768,512")
    p.add_argument("--num_images", type=int, default=16, help="holdout probe size")
    p.add_argument(
        "--sigmas",
        default="0.125,0.375,0.625,0.875,1.0",
        help="csv σ list; default matches the measured resvec run",
    )
    p.add_argument(
        "--data_root",
        default=str(Path(__file__).resolve().parent / "probe1280_cache"),
    )
    p.add_argument(
        "--measured",
        default=str(
            Path(__file__).resolve().parent / "results" / "20260730-2054-resvec"
        ),
        help="run_prior_distance run dir holding the measured curves",
    )
    p.add_argument("--n_bands", type=int, default=10, help="radial bands above DC")
    p.add_argument(
        "--shrink_R", type=float, default=0.2, help="channel-corr shrink toward I"
    )
    p.add_argument(
        "--clip_pc", type=float, default=0.6, help="|parent-child corr| clip"
    )
    p.add_argument("--quant_k", type=int, default=4)
    p.add_argument("--label", default=None)
    return p.parse_args()


# --------------------------------------------------------------------------
# basis
# --------------------------------------------------------------------------


def dct2(x: np.ndarray) -> np.ndarray:
    """(C, H, W) → orthonormal 2D DCT-II per channel."""
    return scipy.fft.dctn(x, axes=(-2, -1), norm="ortho")


def idct2(x: np.ndarray) -> np.ndarray:
    return scipy.fft.idctn(x, axes=(-2, -1), norm="ortho")


def band_map(h: int, w: int, n_bands: int) -> np.ndarray:
    """(H, W) int map: -1 for DC, else 0..n_bands-1 by normalized radial freq.

    ρ = √((i/H)² + (j/W)²)/√2 ∈ [0, 1); geometric edges from a fixed floor so
    band identity is comparable across the free-fit shape population.
    """
    u = (np.arange(h) / h)[:, None]
    v = (np.arange(w) / w)[None, :]
    rho = np.sqrt(u * u + v * v) / np.sqrt(2.0)
    edges = np.geomspace(5e-3, 1.0, n_bands + 1)
    idx = np.clip(np.searchsorted(edges, rho, side="right") - 1, 0, n_bands - 1)
    idx = idx.astype(np.int64)
    idx[0, 0] = -1
    return idx


def octave_chains(h: int, w: int) -> list[list[tuple[int, int]]]:
    """Disjoint chains (i,j) → (2i,2j) → … partitioning all non-DC modes.

    A mode's parent is (i/2, j/2) iff both indices are even; roots are the
    modes with at least one odd index (plus even-index modes whose repeated
    halving bottoms out — handled by the not-both-even test after halving).
    """
    chains = []
    for i in range(h):
        for j in range(w):
            if (i, j) == (0, 0) or (i % 2 == 0 and j % 2 == 0):
                continue
            chain, ci, cj = [], i, j
            while ci < h and cj < w:
                chain.append((ci, cj))
                ci, cj = 2 * ci, 2 * cj
            chains.append(chain)
    # even-index chains rooted at modes like (0, odd·2^k) are covered above by
    # their odd root; the remaining all-even roots reduce to an odd root too,
    # except pure powers-of-two rows/cols through (0,0) — pick those up:
    seen = {m for c in chains for m in c}
    for i in range(h):
        for j in range(w):
            if (i, j) == (0, 0) or (i, j) in seen:
                continue
            chain, ci, cj = [], i, j
            while ci < h and cj < w and (ci, cj) not in seen:
                chain.append((ci, cj))
                seen.add((ci, cj))
                ci, cj = 2 * ci, 2 * cj
            if chain:
                chains.append(chain)
    return chains


# --------------------------------------------------------------------------
# covariance model (per arm): band stats pooled over fit images
# --------------------------------------------------------------------------


class ArmModel:
    """Second-order stats for one arm: per-(band, channel) variance, per-band
    channel correlation, per-(band, channel) parent-child correlation, plus
    DC mean/variance in per-pixel units (DC coeff = pixel-mean · √(HW))."""

    def __init__(self, n_bands: int, n_ch: int = 16):
        self.n_bands, self.n_ch = n_bands, n_ch
        self.var = np.zeros((n_bands, n_ch))
        self.R = np.stack([np.eye(n_ch)] * n_bands)
        self.pc = np.zeros((n_bands, n_ch))  # keyed by band of the CHILD
        self.dc_mean = np.zeros(n_ch)
        self.dc_var = np.ones(n_ch)

    @classmethod
    def fit(
        cls,
        coeffs: list[np.ndarray],
        n_bands: int,
        shrink_r: float,
        clip_pc: float,
    ) -> "ArmModel":
        m = cls(n_bands)
        # pass 1: per-(band, ch) variance + DC moments (per-pixel units)
        ssq = np.zeros((n_bands, 16))
        cnt = np.zeros(n_bands)
        dc = []
        for c in coeffs:
            h, w = c.shape[-2:]
            bands = band_map(h, w, n_bands)
            dc.append(c[:, 0, 0] / np.sqrt(h * w))
            for b in range(n_bands):
                mask = bands == b
                if mask.any():
                    ssq[b] += (c[:, mask] ** 2).sum(axis=1)
                    cnt[b] += int(mask.sum())
        cnt = np.maximum(cnt, 1)
        m.var = ssq / cnt[:, None]
        m.var = np.maximum(m.var, 1e-12)
        dc_arr = np.stack(dc)
        m.dc_mean = dc_arr.mean(axis=0)
        m.dc_var = np.maximum(dc_arr.var(axis=0), 1e-12)
        # pass 2: standardized channel correlation + parent-child correlation
        crs = np.zeros((n_bands, 16, 16))
        pc_num = np.zeros((n_bands, 16))
        pc_cnt = np.zeros(n_bands)
        std = np.sqrt(m.var)
        for c in coeffs:
            h, w = c.shape[-2:]
            bands = band_map(h, w, n_bands)
            for b in range(n_bands):
                mask = bands == b
                if mask.any():
                    z = c[:, mask] / std[b][:, None]
                    crs[b] += z @ z.T
            # parent (i,j) → child (2i,2j); child determines the band key
            hi = np.arange((h + 1) // 2)
            wj = np.arange((w + 1) // 2)
            pi, pj = np.meshgrid(hi, wj, indexing="ij")
            keep = (2 * pi < h) & (2 * pj < w)
            pi, pj = pi[keep], pj[keep]
            not_dc = ~((pi == 0) & (pj == 0))
            pi, pj = pi[not_dc], pj[not_dc]
            bp, bc = bands[pi, pj], bands[2 * pi, 2 * pj]
            zp = c[:, pi, pj] / std[bp].T
            zc = c[:, 2 * pi, 2 * pj] / std[bc].T
            for b in range(n_bands):
                mask = bc == b
                if mask.any():
                    pc_num[b] += (zp[:, mask] * zc[:, mask]).sum(axis=1)
                    pc_cnt[b] += int(mask.sum())
        corr = crs / cnt[:, None, None]
        for b in range(n_bands):
            m.R[b] = (1 - shrink_r) * corr[b] + shrink_r * np.eye(16)
        m.pc = np.clip(pc_num / np.maximum(pc_cnt, 1)[:, None], -clip_pc, clip_pc)
        return m


# --------------------------------------------------------------------------
# closure prediction: r̄*(σ) = σ·(a²C + σ²I)⁻¹ (x̃ − μ̃) per arm grid
# --------------------------------------------------------------------------


def _apply_eig(u: np.ndarray, wv: np.ndarray, z: np.ndarray, sigma: float):
    """(σ/(a²w + σ²)) in the eigenbasis of C, applied to z (…, D)."""
    a2 = (1.0 - sigma) ** 2
    gain = sigma / (a2 * np.maximum(wv, 0.0) + sigma**2)
    return (z @ u) * gain @ u.T


def predict_residuals(
    coeff: np.ndarray,
    model: ArmModel,
    sigmas: list[float],
    n_bands: int,
    closure: str,
    chain_cache: dict,
) -> list[np.ndarray]:
    """Per σ, predicted mean-residual (C, H, W) for one arm of one image."""
    ch, h, w = coeff.shape
    bands = band_map(h, w, n_bands)
    std = np.sqrt(model.var)
    out = [np.zeros_like(coeff) for _ in sigmas]

    # DC: scalar Gaussian with per-pixel-mean stats scaled to the DC coeff
    dc_dev = coeff[:, 0, 0] - model.dc_mean * np.sqrt(h * w)
    dc_var = model.dc_var * h * w
    for si, s in enumerate(sigmas):
        out[si][:, 0, 0] = s * dc_dev / ((1 - s) ** 2 * dc_var + s**2)

    if closure == "diag":
        for b in range(n_bands):
            mask = bands == b
            if not mask.any():
                continue
            x = coeff[:, mask]
            for si, s in enumerate(sigmas):
                gain = s / ((1 - s) ** 2 * model.var[b][:, None] + s**2)
                out[si][:, mask] = gain * x
        return out

    if closure == "block":
        for b in range(n_bands):
            mask = bands == b
            if not mask.any():
                continue
            c_b = std[b][:, None] * model.R[b] * std[b][None, :]
            wv, u = np.linalg.eigh(c_b)
            x = coeff[:, mask].T  # (n_modes, 16)
            for si, s in enumerate(sigmas):
                out[si][:, mask] = _apply_eig(u, wv, x, s).T
        return out

    # octave: exact solve along disjoint parent-child chains; chains sharing a
    # band signature share the covariance block, so one eig serves them all.
    key = (h, w)
    if key not in chain_cache:
        groups: dict[tuple, list[list[tuple[int, int]]]] = defaultdict(list)
        for chain in octave_chains(h, w):
            groups[tuple(bands[i, j] for i, j in chain)].append(chain)
        chain_cache[key] = groups
    for sig_bands, chains in chain_cache[key].items():
        ln = len(sig_bands)
        dim = 16 * ln
        c_blk = np.zeros((dim, dim))
        for t, b in enumerate(sig_bands):
            c_blk[16 * t : 16 * (t + 1), 16 * t : 16 * (t + 1)] = (
                std[b][:, None] * model.R[b] * std[b][None, :]
            )
            if t > 0:
                bp = sig_bands[t - 1]
                cross = np.diag(model.pc[b] * std[bp] * std[b])
                c_blk[16 * (t - 1) : 16 * t, 16 * t : 16 * (t + 1)] = cross
                c_blk[16 * t : 16 * (t + 1), 16 * (t - 1) : 16 * t] = cross
        wv, u = np.linalg.eigh(c_blk)
        ii = np.array([[m[0] for m in c] for c in chains])  # (n_chains, ln)
        jj = np.array([[m[1] for m in c] for c in chains])
        # (n_chains, ln, 16) → (n_chains, ln*16), channel-fastest per position
        x = coeff[:, ii, jj].transpose(1, 2, 0).reshape(len(chains), dim)
        for si, s in enumerate(sigmas):
            r = _apply_eig(u, wv, x, s).reshape(len(chains), ln, 16)
            out[si][:, ii, jj] = r.transpose(2, 0, 1)
    return out


# --------------------------------------------------------------------------
# instrument replication
# --------------------------------------------------------------------------


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / (0.5 * (a.norm() + b.norm())))


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(a.flatten().double().dot(b.flatten().double()) / (a.norm() * b.norm()))


def down_to(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x.unsqueeze(0), size=hw, mode="area").squeeze(0)


# --------------------------------------------------------------------------
# latents: native from the lora cache, demoted/reenc VAE-encoded once
# --------------------------------------------------------------------------


def ensure_latents(
    records: list[ImageRecord], edges: list[int], vae_path: str | None, root: Path
) -> dict[tuple[str, str], np.ndarray]:
    from library.io.cache import load_cached_latents

    cache = root / "closure_latents"
    cache.mkdir(exist_ok=True)
    arms = [f"demote{e}" for e in edges] + ["reenc"]
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
            missing, edges, vae_path or VAE, torch.device("cuda"), True
        )
        for (stem, arm), lat in extra.items():
            np.savez_compressed(
                cache / f"{stem}__{arm}.npz", latent=lat.numpy().astype(np.float32)
            )
    out: dict[tuple[str, str], np.ndarray] = {}
    for r in records:
        out[(r.stem, "native")] = (
            load_cached_latents(r.npz_path)[0].numpy().astype(np.float32)
        )
        for a in arms:
            out[(r.stem, a)] = np.load(cache / f"{r.stem}__{a}.npz")["latent"]
    return out


# --------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    start_heartbeat()
    edges = [int(e) for e in args.edges.split(",") if e]
    sigmas = [float(s) for s in args.sigmas.split(",") if s]
    data_root = Path(args.data_root).resolve()

    records = [
        r for r in score_corpus(k=args.quant_k, data_root=data_root) if r.complete()
    ]
    holdout = select_probe_set(records, args.num_images, tier=args.tier)
    hold_stems = {r.stem for r in holdout}
    fit = [r for r in records if r.stem not in hold_stems and r.tier == args.tier]
    log.info(f"fit: {len(fit)} images, holdout (probe set): {len(holdout)}")

    measured = json.load(open(Path(args.measured) / "result.json"))
    meas_probe = {p.split("/")[-1] for p in measured["extra"]["probe_set"]}
    if meas_probe != hold_stems:
        log.warning(
            "holdout differs from the measured run's probe set "
            f"({len(meas_probe & hold_stems)}/{len(meas_probe)} shared) — "
            "scoring is still valid but not image-matched"
        )
    if [round(s, 4) for s in measured["metrics"]["sigmas"]] != [
        round(s, 4) for s in sigmas
    ]:
        raise SystemExit("--sigmas must match the measured run's σ grid")

    lat = ensure_latents(fit + holdout, edges, args.vae, data_root)

    grid_names = ["native"] + [f"demote{e}" for e in edges]
    pairs = list(zip(grid_names[:-1], grid_names[1:])) + [("native", "reenc")]
    pair_label = {f"demote{e}": str(e) for e in edges} | {
        "native": str(args.tier),
        "reenc": "reenc",
    }

    # fit one ArmModel per arm on the fit split
    models: dict[str, ArmModel] = {}
    for g in grid_names + ["reenc"]:
        coeffs = [dct2(lat[(r.stem, g)]) for r in fit]
        models[g] = ArmModel.fit(coeffs, args.n_bands, args.shrink_R, args.clip_pc)
    log.info("arm models fitted")

    run_dir = make_run_dir(
        "sigma_lowres", args.label, root=Path(__file__).resolve().parent / "results"
    )
    rows_path = run_dir / "per_image.jsonl"
    chain_cache: dict = {}
    curves: dict[tuple[str, str, str], list[list[float]]] = defaultdict(list)

    for split, subset in (("holdout", holdout), ("fit", fit)):
        for i, r in enumerate(subset):
            preds: dict[tuple[str, str], list[torch.Tensor]] = {}
            for g in grid_names + ["reenc"]:
                coeff = dct2(lat[(r.stem, g)])
                for cl in CLOSURES:
                    res = predict_residuals(
                        coeff, models[g], sigmas, args.n_bands, cl, chain_cache
                    )
                    preds[(g, cl)] = [
                        torch.from_numpy(idct2(x).astype(np.float32)) for x in res
                    ]
            row: dict = {"stem": r.stem, "artist": r.artist, "split": split}
            for hi, lo in pairs:
                lo_hw = lat[(r.stem, lo)].shape[-2:]
                for cl in CLOSURES:
                    ms, cs = [], []
                    for si in range(len(sigmas)):
                        a = preds[(hi, cl)][si]
                        if a.shape[-2:] != lo_hw:
                            a = down_to(a, lo_hw)
                        b = preds[(lo, cl)][si]
                        ms.append(round(rel_l2(a, b), 5))
                        cs.append(round(cos(a, b), 5))
                    key = f"{pair_label[hi]}_{pair_label[lo]}"
                    row[f"m_{cl}_{key}"] = ms
                    row[f"cos_{cl}_{key}"] = cs
                    curves[(split, cl, key)].append(ms)
            with rows_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            log.info(f"  [{split} {i + 1}/{len(subset)}] {r.artist}/{r.stem}")

    # ---- score against the measured excess curves --------------------------
    adj_keys = [
        f"{pair_label[hi]}_{pair_label[lo]}" for hi, lo in pairs if lo != "reenc"
    ]
    metrics: dict = {
        "sigmas": sigmas,
        "n_fit": len(fit),
        "n_holdout": len(holdout),
        "closures": list(CLOSURES),
        "measured_run": str(args.measured),
    }
    meas_ex = {
        k: measured["metrics"][f"excess_{k}"]["mean"]
        for k in adj_keys
        if f"excess_{k}" in measured["metrics"]
    }
    spread_meas = [
        max(v[si] for v in meas_ex.values()) - min(v[si] for v in meas_ex.values())
        for si in range(len(sigmas))
    ]
    metrics["measured_excess"] = meas_ex
    metrics["measured_route_spread_max"] = round(max(spread_meas), 5)
    verdicts = {}
    for cl in CLOSURES:
        rmses, shapes = [], []
        pred_mean: dict[str, list[float]] = {}
        for k in adj_keys:
            arr = np.array(curves[("holdout", cl, k)])
            pm = arr.mean(axis=0)
            pred_mean[k] = [round(float(v), 5) for v in pm]
            if k in meas_ex:
                mv = np.array(meas_ex[k])
                rmses.append(float(np.sqrt(((pm - mv) ** 2).mean())))
                if np.std(pm) > 1e-9 and np.std(mv) > 1e-9:
                    shapes.append(float(np.corrcoef(pm, mv)[0, 1]))
        spread_pred = [
            max(pred_mean[k][si] for k in adj_keys)
            - min(pred_mean[k][si] for k in adj_keys)
            for si in range(len(sigmas))
        ]
        reenc_key = f"{args.tier}_reenc"
        reenc_curve = np.array(curves[("holdout", cl, reenc_key)]).mean(axis=0)
        fit_mean = {
            k: [
                round(float(v), 5)
                for v in np.array(curves[("fit", cl, k)]).mean(axis=0)
            ]
            for k in adj_keys
        }
        rmse = float(np.mean(rmses))
        ok = rmse <= 0.05 and max(spread_pred) <= 2 * max(spread_meas)
        verdicts[cl] = ok
        metrics[cl] = {
            "pred_excess_holdout": pred_mean,
            "pred_excess_fit": fit_mean,
            "pred_reenc_control": [round(float(v), 5) for v in reenc_curve],
            "rmse_mean": round(rmse, 5),
            "shape_pearson_mean": round(float(np.mean(shapes)), 5) if shapes else None,
            "route_spread_max": round(max(spread_pred), 5),
            "passes": ok,
        }
        log.info(
            f"[{cl}] rmse={rmse:.4f} shape_r={np.mean(shapes):.3f} "
            f"spread={max(spread_pred):.4f} (meas {max(spread_meas):.4f}) "
            f"→ {'PASS' if ok else 'FAIL'}"
        )
    metrics["any_pass"] = any(verdicts.values())

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["per_image.jsonl"],
        extra={"holdout": sorted(hold_stems), "fit": sorted(r.stem for r in fit)},
    )
    log.info(f"result → {run_dir}")


if __name__ == "__main__":
    main()
