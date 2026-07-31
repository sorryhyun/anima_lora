#!/usr/bin/env python
"""sigma_lowres — cross-route structure of the residual mismatch Δr̄.

CPU reanalysis of a ``run_prior_distance.py --save_residuals`` run. G7
established that ‖Δr̄‖ follows the same m(σ) curve on every route (±0.02)
while the routes' gradient floors span 0–0.3 — but a norm read cannot tell
a **rank-one common direction** (one universal mismatch mode every cutoff
excites → strong support for the account's single m(σ) input) from
**route-specific directions with matched norms** (which would undermine the
"universal m" interpretation and point at per-band structure).

Per image and σ: form each consecutive route pair's Δr̄ (hi half-means
area-downsampled to the lo grid, exactly as the run's scalar read), align
all pairs to that image's smallest grid, then report

- pairwise cosine matrices between route Δr̄s (raw + split-half
  attenuation-corrected via each Δ's parity-half reliability),
- the stacked-SVD top-mode energy share (rank-one-ness) per image,
- per pair: CROSS-IMAGE direction consistency (each image's Δr̄
  resampled to a common 64×64 grid, mean pairwise cosine across images —
  ≈ 0 means the direction is image-specific, killing any
  image-independent carrier such as a grid-conditional content prior)
  and the mean radial-spectral thirds (low/mid/high energy fractions),

averaged across images. High corrected cosines / top-share ⇒ one shared
mismatch direction; matched norms with low cosines ⇒ uniform-in-norm only.

Usage::

    uv run python project/sigma_lowres/paper_bench/experiments/e11/resid_structure.py \
        --run project/sigma_lowres/bench/results/<run>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--run", required=True, help="run dir holding residuals/")
    p.add_argument(
        "--out", default=None, help="output json (default <run>/resid_structure.json)"
    )
    return p.parse_args()


def down_to(x: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0)
    return F.interpolate(t, size=hw, mode="area").squeeze(0).numpy()


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return float("nan")
    return float(a.flatten() @ b.flatten()) / (na * nb)


def radial_thirds(x: np.ndarray) -> list[float]:
    """Energy fractions of the low/mid/high radial-frequency thirds."""
    fx = np.fft.fftshift(np.fft.fft2(x, axes=(-2, -1)), axes=(-2, -1))
    p = (np.abs(fx) ** 2).sum(0)
    h, w = p.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - h / 2, xx - w / 2) / (min(h, w) / 2)
    bands = ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 10))
    tot = float(p.sum())
    return [float(p[(r >= lo) & (r < hi)].sum() / tot) for lo, hi in bands]


def main() -> None:
    args = parse_args()
    run = Path(args.run).resolve()
    res_dir = run / "residuals"
    files = sorted(res_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"no residuals under {res_dir}")

    # {stem: {(grid, si): (m1, m2)}}
    per_img: dict[str, dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]] = (
        defaultdict(dict)
    )
    sigmas: dict[int, float] = {}
    for f in files:
        m = re.match(r"(.+)__([^_]+)__s(\d+)\.npz$", f.name)
        if not m:
            continue
        stem, grid, si = m.group(1), m.group(2), int(m.group(3))
        z = np.load(f)
        per_img[stem][(grid, si)] = (
            z["m1"].astype(np.float32),
            z["m2"].astype(np.float32),
        )
        sigmas[si] = float(z["sigma"])

    # grid names present (excluding reenc): sort numeric descending = native
    # first, then the demotion ladder; consecutive pairs mirror the run
    grids_all = sorted(
        {g for d in per_img.values() for g, _ in d if g != "reenc"},
        key=lambda g: -int(g),
    )
    pairs = list(zip(grids_all[:-1], grids_all[1:]))
    pair_names = [f"{hi}->{lo}" for hi, lo in pairs]
    n_sig = max(sigmas) + 1
    print(
        f"{len(per_img)} images, grids {grids_all}, pairs {pair_names}, "
        f"σ={[sigmas[i] for i in range(n_sig)]}"
    )

    out: dict = {"pairs": pair_names, "sigmas": [sigmas[i] for i in range(n_sig)]}
    for si in range(n_sig):
        cos_raw = np.full((len(per_img), len(pairs), len(pairs)), np.nan)
        cos_corr = np.full_like(cos_raw, np.nan)
        top_share = np.full(len(per_img), np.nan)
        norms = np.full((len(per_img), len(pairs)), np.nan)
        rels = np.full((len(per_img), len(pairs)), np.nan)
        common64: list[list[np.ndarray]] = [[] for _ in pairs]
        spectra: list[list[list[float]]] = [[] for _ in pairs]
        for ii, (stem, d) in enumerate(sorted(per_img.items())):
            if any((g, si) not in d for g in grids_all):
                continue
            small_hw = d[(grids_all[-1], si)][0].shape[-2:]
            deltas, delta_rel = [], []
            for hi, lo in pairs:
                h1, h2 = d[(hi, si)]
                l1, l2 = d[(lo, si)]
                lo_hw = l1.shape[-2:]
                if h1.shape[-2:] != lo_hw:
                    h1, h2 = down_to(h1, lo_hw), down_to(h2, lo_hw)
                # parity-half deltas -> reliability; mean delta -> the object
                d1, d2 = h1 - l1, h2 - l2
                dm = 0.5 * (d1 + d2)
                if dm.shape[-2:] != small_hw:
                    d1, d2, dm = (
                        down_to(d1, small_hw),
                        down_to(d2, small_hw),
                        down_to(dm, small_hw),
                    )
                deltas.append(dm)
                delta_rel.append(cos(d1, d2))
                dm64 = down_to(dm, (64, 64))
                common64[len(deltas) - 1].append(
                    dm64.flatten() / (np.linalg.norm(dm64) + 1e-12)
                )
                spectra[len(deltas) - 1].append(radial_thirds(dm64))
            for pi_ in range(len(pairs)):
                norms[ii, pi_] = float(np.linalg.norm(deltas[pi_]))
                rels[ii, pi_] = delta_rel[pi_]
                for pj in range(len(pairs)):
                    c = cos(deltas[pi_], deltas[pj])
                    cos_raw[ii, pi_, pj] = c
                    denom = (
                        np.sqrt(delta_rel[pi_] * delta_rel[pj])
                        if delta_rel[pi_] > 0 and delta_rel[pj] > 0
                        else np.nan
                    )
                    cos_corr[ii, pi_, pj] = c / denom if denom == denom else np.nan
            stack = np.stack([v.flatten() / (np.linalg.norm(v) or 1.0) for v in deltas])
            sv = np.linalg.svd(stack, compute_uv=False)
            top_share[ii] = float(sv[0] ** 2 / (sv**2).sum())
        xcos, xsem = [], []
        for vecs64 in common64:
            if len(vecs64) < 2:
                xcos.append(float("nan"))
                xsem.append(float("nan"))
                continue
            vs = np.stack(vecs64)
            off = (vs @ vs.T)[np.triu_indices(len(vecs64), 1)]
            xcos.append(round(float(off.mean()), 4))
            xsem.append(round(float(off.std() / np.sqrt(len(off))), 4))
        key = f"s{si}"
        out[key] = {
            "cos_raw_mean": np.nanmean(cos_raw, axis=0).round(4).tolist(),
            "cos_corrected_mean": np.nanmean(cos_corr, axis=0).round(4).tolist(),
            "reliability_mean": np.nanmean(rels, axis=0).round(4).tolist(),
            "norm_mean": np.nanmean(norms, axis=0).round(4).tolist(),
            "cross_image_cos": xcos,
            "cross_image_cos_sem": xsem,
            "spectral_thirds_mean": [
                [round(v, 4) for v in np.mean(s, axis=0)] if s else None
                for s in spectra
            ],
            "svd_top_share_mean": round(float(np.nanmean(top_share)), 4),
            "svd_top_share_sem": round(
                float(
                    np.nanstd(top_share, ddof=1) / np.sqrt(np.isfinite(top_share).sum())
                ),
                4,
            ),
        }
        print(
            f"\nσ={sigmas[si]}: svd top-mode share = "
            f"{out[key]['svd_top_share_mean']:.3f}±{out[key]['svd_top_share_sem']:.3f}"
            f"  (uniform rank-4 would be 0.25)"
        )
        print("corrected pairwise cosines | cross-image cos | spectrum L/M/H:")
        for pi_, name in enumerate(pair_names):
            row = out[key]["cos_corrected_mean"][pi_]
            sp = out[key]["spectral_thirds_mean"][pi_]
            print(
                f"  {name:<12} "
                + " ".join(f"{v:+.3f}" for v in row)
                + f"  | x-img {xcos[pi_]:+.3f}±{xsem[pi_]:.3f}"
                + (f"  | {sp[0]:.2f}/{sp[1]:.2f}/{sp[2]:.2f}" if sp else "")
            )

    dst = Path(args.out) if args.out else run / "resid_structure.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nresult → {dst}")


if __name__ == "__main__":
    main()
