#!/usr/bin/env python
"""E3 — the a + b/B batch-size fit over the pooled verdict grid.

Reads one ``run_sigma_probe.py --pool P --self_floor`` run and produces, per
(route, σ-bin), the paired-debiased gap at three batch sizes —

* **B=1**  — per-image rows (mean ± SEM over images),
* **B=P**  — the P-image strata (mean ± SEM over strata),
* **B=N**  — the all-images aggregate (single pooled read),

then the two-parameter fit ``gap(B) = a + b/B``: the intercept ``a`` is the
coherent drift no batch size removes (the §3.1 decomposition's first term),
``b`` the 1/B-averaging term. Conventions follow the E1(b) record: the
readable object is the **paired difference (arm − reenc)** at every level,
and cells where the debiased estimator blows up (|gap| > 1.5, small-floor
attenuation failure — the 512 endpoint bin) are masked.

Fit: weighted least squares on the three points with per-point SE weights;
the aggregate point has no replicate SE, so it borrows the stratum SEM
scaled by sqrt(P/N) (the 1/B variance law the fit itself assumes). CPU-only,
seconds.

Usage::

    uv run python project/sigma_lowres/paper_bench/experiments/e3/e3_fit.py \
        --run project/sigma_lowres/bench/results/20260804-0002-e3-pooled-verdict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# E1b's per-image rule (|gap| > 1.5) plus the non-physicality bound: gap =
# 1 − c_hat with c_hat a corrected cosine, so gap < −0.5 (c_hat > 1.5) is an
# attenuation-correction failure, not a measurement (the 512 endpoint cell,
# gap = −1.15, is the motivating case).
GAP_LO, GAP_HI = -0.5, 1.5


def paired(rows_or_stats: list[dict], edge: int) -> np.ndarray:
    """Per-entry paired-debiased curves (arm − reenc), blowup-masked → NaN.
    Shape (n_entries, n_bins)."""
    out = []
    for r in rows_or_stats:
        arm = np.array(
            [np.nan if v is None else v for v in r[f"debiased_gap_{edge}"]], float
        )
        ctrl = np.array(
            [np.nan if v is None else v for v in r["debiased_gap_reenc"]], float
        )
        arm[(arm < GAP_LO) | (arm > GAP_HI)] = np.nan
        ctrl[(ctrl < GAP_LO) | (ctrl > GAP_HI)] = np.nan
        out.append(arm - ctrl)
    return np.array(out)


def level_stats(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware per-bin mean and SEM over axis 0."""
    mean = np.nanmean(mat, axis=0)
    n = np.sum(~np.isnan(mat), axis=0)
    sd = np.nanstd(mat, axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sem = sd / np.sqrt(np.maximum(n, 1))
    return mean, sem


def fit_a_b(bs: np.ndarray, ys: np.ndarray, ses: np.ndarray) -> tuple[float, float]:
    """Weighted LS of y = a + b * (1/B); returns (a, b). NaN if <2 points."""
    ok = ~np.isnan(ys) & ~np.isnan(ses) & (ses > 0)
    if ok.sum() < 2:
        return float("nan"), float("nan")
    x = 1.0 / bs[ok]
    w = 1.0 / ses[ok] ** 2
    A = np.stack([np.ones_like(x), x], axis=1)
    coef = np.linalg.lstsq(A * np.sqrt(w)[:, None], ys[ok] * np.sqrt(w), rcond=None)[0]
    return float(coef[0]), float(coef[1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--run", required=True, help="e3 pooled-verdict run dir")
    ap.add_argument(
        "--out", default=None, help="output JSON (default <run>/e3_fit.json)"
    )
    args = ap.parse_args()

    run = Path(args.run)
    result = json.loads((run / "result.json").read_text())
    m = result["metrics"]
    pool = m["pool"]
    P, N = pool["size"], pool["aggregate"]["n_images"]
    centers = m["sigma_centers"]
    edges = [
        int(e) for e in (896, 768, 512) if f"debiased_gap_{e}" in pool["aggregate"]
    ]
    rows = [json.loads(ln) for ln in (run / "per_image.jsonl").open()]

    out: dict = {
        "run": str(run),
        "pool_size": P,
        "n_images": N,
        "sigma_centers": centers,
        "convention": "paired debiased gap (arm - reenc); gap outside "
        "[-0.5, 1.5] masked (non-physical c_hat); "
        "aggregate SE = stratum SEM * sqrt(P/N)",
        "routes": {},
    }
    for e in edges:
        per_img = paired(rows, e)
        strata = paired(pool["strata"], e)
        agg = paired([pool["aggregate"]], e)[0]
        m1, s1 = level_stats(per_img)
        mP, sP = level_stats(strata)
        sN = sP * np.sqrt(P / N)
        route: dict = {"levels": {}}
        route["levels"]["B1"] = {"mean": m1.tolist(), "sem": s1.tolist()}
        route["levels"][f"B{P}"] = {"mean": mP.tolist(), "sem": sP.tolist()}
        route["levels"][f"B{N}"] = {"mean": agg.tolist(), "se_borrowed": sN.tolist()}
        fits = []
        for j in range(len(centers)):
            a, b = fit_a_b(
                np.array([1.0, float(P), float(N)]),
                np.array([m1[j], mP[j], agg[j]]),
                np.array([s1[j], sP[j], sN[j]]),
            )
            fits.append({"sigma": centers[j], "a": a, "b": b})
        route["fit"] = fits
        out["routes"][str(e)] = route
        line = " ".join(
            f"σ{f['sigma']:g}:a={f['a']:+.3f}"
            if np.isfinite(f["a"])
            else f"σ{f['sigma']:g}:a=nan"
            for f in fits
        )
        print(f"[{e}] intercepts {line}")

    dest = Path(args.out) if args.out else run / "e3_fit.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
