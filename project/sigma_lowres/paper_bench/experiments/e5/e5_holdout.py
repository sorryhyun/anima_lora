"""E5 — Eq. 3 held-out validation (analysis-only, no GPU).

Fit the two-term account gap_e(sigma) ~= A_e * m(sigma)/G(sigma)^p + F_e
on the FIT routes {1024->896, 1024->512, 1280->1120}, fit the governor
models A(ratio) and F(target tokens), then PREDICT the held-out routes
{1024->768, 1280->1024} and score against (a) the spectral null and
(b) an oracle smooth fit (quadratic in sigma, fit directly on the
held-out data = the noise ceiling for a 3-dof curve).

Sources (all existing rows, no new instrument):
  - gap curves 1024 tier: runs/20260729-0014-e1b-debiased-map
    (paired per-image debiased delta, |d|>1.5 trimmed — Table 2 recipe)
  - gap curves 1280 tier: bench/results/20260726-2153 (G9; RAW paired —
    no self-floors exist there; D=4 bias caveat recorded in result.json)
  - m(sigma): bench/results/20260726-2133 (G7) — route-uniform mean of
    the four excess curves
  - G(sigma): each run's own bin-mean gnorm_native (kernel-path rule:
    G and gap always from the same process)

Pre-registered read (written before the numbers):
  PASS  = on both held-out routes, weighted RMSE(two-term prediction)
          <= 2x RMSE(oracle smooth) AND < RMSE(null) on the null's
          committed region (sigma > sigma_eq: null predicts gap 0);
          secondary: A_896 ~= A_1120 within 2 sigma (ratio governor),
          predicted F(2160 tok) inside the measured 768 floor CI
          (debiased 0.043..0.094 from E1a/E1c).
  FAIL  = otherwise; Eq. 3 stays a decomposition, governors stay
          "consistent with" (experiments/e5/README.md).

Usage:
    python project/sigma_lowres/paper_bench/experiments/e5/e5_holdout.py
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PAPER_BENCH = HERE.parents[1]  # project/sigma_lowres/paper_bench
SIGMA = HERE.parents[2]  # project/sigma_lowres
RUNS = PAPER_BENCH / "runs"
BENCH = SIGMA / "bench"

E1B = RUNS / "20260729-0014-e1b-debiased-map"
G9 = BENCH / "results" / "20260726-2153"
G5 = BENCH / "results" / "20260726-2017"
G7 = BENCH / "results" / "20260726-2133"

TRIM_ABS = 1.5

# route metadata: edge ratio + nominal target token count (EDGE_TOKEN_BANDS)
RATIO = {"896": 0.875, "768": 0.75, "512": 0.5, "1120": 0.875, "1024": 0.8}
TOKENS = {"896": 3012, "768": 2160, "512": 1016, "1120": 4825, "1024": 4116}
FIT_ROUTES = ["896", "512", "1120"]
HELD_OUT = ["768", "1024"]
SIGMA_EQ = {"768": 0.146, "1024": 0.15}  # RAPSD closed-form crossovers


def paired_stats(run_dir: Path, route: str, key: str):
    """Bin mean+SEM of the paired per-image delta (arm - reenc)."""
    recs = [json.loads(line) for line in (run_dir / "per_image.jsonl").open()]
    sigma = recs[0]["sigma_centers"]
    means, sems = [], []
    for b in range(len(sigma)):
        vals = []
        for r in recs:
            d, c = r[f"{key}_{route}"][b], r[f"{key}_reenc"][b]
            if d is None or c is None:
                continue
            delta = d - c
            if math.isfinite(delta) and abs(delta) <= TRIM_ABS:
                vals.append(delta)
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        means.append(mean)
        sems.append(math.sqrt(var / n))
    return np.array(sigma), np.array(means), np.array(sems)


def bin_gnorm(run_dir: Path):
    m = json.load((run_dir / "result.json").open())["metrics"]
    return np.array(m["gnorm_native"]["mean"])


def m_bar():
    """Route-uniform residual-mismatch shape from G7 (mean of 4 routes)."""
    m = json.load((G7 / "result.json").open())["metrics"]
    sig = np.array([float(s) for s in m["sigmas"]])
    curves = [
        np.array(m[k]["mean"])
        for k in ("excess_1280_1024", "excess_1024_896", "excess_896_768", "excess_768_512")
    ]
    return sig, np.mean(curves, axis=0)


def interp_extrap(x, xp, fp):
    """np.interp with linear extrapolation below xp[0]."""
    y = np.interp(x, xp, fp)
    lo = x < xp[0]
    if lo.any():
        slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
        y[lo] = fp[0] + (x[lo] - xp[0]) * slope
    return y


def wls_line(x, y, w):
    """Weighted LS for y = A*x + F. Returns A, F, se_A, se_F."""
    W = np.sum(w)
    xm, ym = np.sum(w * x) / W, np.sum(w * y) / W
    sxx = np.sum(w * (x - xm) ** 2)
    A = np.sum(w * (x - xm) * (y - ym)) / sxx
    F = ym - A * xm
    se_A = math.sqrt(1.0 / sxx)
    se_F = math.sqrt(1.0 / W + xm**2 / sxx)
    return A, F, se_A, se_F


def main() -> None:
    msig, mvals = m_bar()

    # measured paired curves
    curves = {}
    for route in ("896", "768", "512"):
        curves[route] = paired_stats(E1B, route, "debiased_gap")
    for route in ("1120", "1024"):
        curves[route] = paired_stats(G9, route, "gap")
    g5_1024 = paired_stats(G5, "1024", "gap")  # cross-check only

    gnorm = {"1024tier": bin_gnorm(E1B), "1280tier": bin_gnorm(G9)}
    tier_of = {"896": "1024tier", "768": "1024tier", "512": "1024tier",
               "1120": "1280tier", "1024": "1280tier"}

    def design(route, p):
        sig, _, _ = curves[route]
        return interp_extrap(sig, msig, mvals) / gnorm[tier_of[route]] ** p

    # ---- joint fit of shared p + per-route (A, F) on the FIT routes ----
    best = None
    for p in np.arange(1.0, 2.001, 0.01):
        sse = 0.0
        params = {}
        for route in FIT_ROUTES:
            _, y, sem = curves[route]
            x, w = design(route, p), 1.0 / sem**2
            A, F, se_A, se_F = wls_line(x, y, w)
            sse += float(np.sum(w * (y - (A * x + F)) ** 2))
            params[route] = dict(A=A, F=F, se_A=se_A, se_F=se_F)
        if best is None or sse < best[0]:
            best = (sse, p, params)
    sse, p_star, fit = best

    # ---- governors ----
    # A(ratio): linear through (0.5, A_512) and (0.875, mean(A_896, A_1120))
    A_hi = 0.5 * (fit["896"]["A"] + fit["1120"]["A"])
    var_A_hi = 0.25 * (fit["896"]["se_A"] ** 2 + fit["1120"]["se_A"] ** 2)
    ratio_consistency_z = abs(fit["896"]["A"] - fit["1120"]["A"]) / math.sqrt(
        fit["896"]["se_A"] ** 2 + fit["1120"]["se_A"] ** 2
    )

    def A_of_ratio(r):
        t = (r - 0.5) / (0.875 - 0.5)
        a = (1 - t) * fit["512"]["A"] + t * A_hi
        var = (1 - t) ** 2 * fit["512"]["se_A"] ** 2 + t**2 * var_A_hi
        return a, math.sqrt(var)

    # F(tokens): F0*exp(-n/tau) through the positive fit floors; 1120's
    # floor ~0 can't enter a log fit — used as a consistency check instead.
    fpts = [(TOKENS[r], fit[r]["F"], fit[r]["se_F"]) for r in FIT_ROUTES]
    logpts = [(n, math.log(F), se / F) for n, F, se in fpts if F > 0.005]
    (n1, l1, _), (n2, l2, _) = logpts[0], logpts[-1]
    tau = (n1 - n2) / (l2 - l1)
    F0 = math.exp(l1 + n1 / tau)

    def F_of_tokens(n):
        return F0 * math.exp(-n / tau)

    f1120_pred = F_of_tokens(TOKENS["1120"])

    # ---- held-out predictions + baselines ----
    results = {}
    for route in HELD_OUT:
        sig, y, sem = curves[route]
        x = design(route, p_star)
        A, se_A = A_of_ratio(RATIO[route])
        F = F_of_tokens(TOKENS[route])
        pred = A * x + F
        band = np.sqrt((x * se_A) ** 2 + np.median([f[2] for f in fpts]) ** 2)
        w = 1.0 / sem**2
        rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
        chi2 = float(np.mean(((y - pred) / sem) ** 2))
        # oracle: quadratic in sigma fit on the held-out data itself (3 dof)
        coef = np.polyfit(sig, y, 2, w=np.sqrt(w))
        oracle = np.polyval(coef, sig)
        rmse_oracle = float(np.sqrt(np.mean((y - oracle) ** 2)))
        # null: gap = 0 for sigma > sigma_eq (its committed region)
        mask = sig > SIGMA_EQ[route]
        rmse_null = float(np.sqrt(np.mean(y[mask] ** 2)))
        rmse_ours_nullregion = float(np.sqrt(np.mean((y[mask] - pred[mask]) ** 2)))
        results[route] = dict(
            sigma=sig.tolist(), measured=y.tolist(), sem=sem.tolist(),
            predicted=pred.tolist(), pred_band=band.tolist(),
            A=A, se_A=se_A, F=F, rmse=rmse, chi2=chi2,
            rmse_oracle=rmse_oracle, rmse_null_committed=rmse_null,
            rmse_ours_committed=rmse_ours_nullregion,
        )

    # in-sample quality for the fit routes (context, not verdict)
    insample = {}
    for route in FIT_ROUTES:
        sig, y, sem = curves[route]
        pr = fit[route]["A"] * design(route, p_star) + fit[route]["F"]
        insample[route] = dict(
            rmse=float(np.sqrt(np.mean((y - pr) ** 2))),
            chi2=float(np.mean(((y - pr) / sem) ** 2)),
            A=fit[route]["A"], se_A=fit[route]["se_A"],
            F=fit[route]["F"], se_F=fit[route]["se_F"],
        )

    verdict = dict(
        p_star=float(p_star),
        ratio_consistency_z=ratio_consistency_z,
        F_tau_tokens=tau, F0=F0,
        f1120_predicted=f1120_pred, f1120_fitted=fit["1120"]["F"],
        f768_predicted=F_of_tokens(TOKENS["768"]),
        pass_rmse={r: results[r]["rmse"] <= 2 * results[r]["rmse_oracle"] for r in HELD_OUT},
        pass_null={r: results[r]["rmse_ours_committed"] < results[r]["rmse_null_committed"] for r in HELD_OUT},
        pass_ratio_governor=bool(ratio_consistency_z < 2),
        pass_768_floor=bool(0.043 <= F_of_tokens(TOKENS["768"]) <= 0.094),
        g5_1024_crosscheck=dict(sigma=g5_1024[0].tolist(), measured=g5_1024[1].tolist(),
                                sem=g5_1024[2].tolist()),
    )
    verdict["PASS"] = bool(
        all(verdict["pass_rmse"].values())
        and all(verdict["pass_null"].values())
        and verdict["pass_ratio_governor"]
        and verdict["pass_768_floor"]
    )

    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M")
    out_dir = RUNS / f"{stamp}-e5-holdout"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- figure: measured vs predicted per held-out route ----
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    for ax, route in zip(axes, HELD_OUT):
        r = results[route]
        sig = np.array(r["sigma"])
        ax.errorbar(sig, r["measured"], yerr=r["sem"], fmt="o", ms=4,
                    color="C0", capsize=2, label="measured (paired)")
        ax.plot(sig, r["predicted"], "-", color="C3",
                label=f"two-term prediction (p={p_star:.2f})")
        ax.fill_between(sig, np.array(r["predicted"]) - np.array(r["pred_band"]),
                        np.array(r["predicted"]) + np.array(r["pred_band"]),
                        color="C3", alpha=0.15, lw=0)
        ax.axvline(SIGMA_EQ[route], color="0.6", ls=":", lw=1,
                   label=r"null $\sigma_{eq}$ (gap 0 to the right)")
        ax.axhline(0, color="0.8", lw=0.8)
        src = "1024→" + route if route != "1024" else "1280→1024"
        ax.set_title(f"{src}   RMSE {r['rmse']:.03f} (oracle {r['rmse_oracle']:.03f})")
        ax.set_xlabel(r"$\sigma$")
    axes[0].set_ylabel(r"paired gap $\bar\Delta$")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("E5: held-out prediction of the two-term account", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "e5_overlay.png", dpi=180, bbox_inches="tight")

    envelope = dict(
        schema_version=1,
        script="project/sigma_lowres/paper_bench/experiments/e5/e5_holdout.py",
        label="e5-holdout",
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        sources=dict(e1b=str(E1B), g9=str(G9), g5=str(G5), g7=str(G7)),
        caveats=[
            "1280-tier curves are RAW paired (no self-floors in G9/G5 runs; D=4)",
            "m(sigma) linearly extrapolated below sigma=0.125 for the two lowest 1024-tier bins",
            "A(ratio) interpolation rests on two distinct ratio values",
        ],
        fit=dict(p_star=float(p_star), insample=insample),
        governors=dict(A_of_ratio_pts={"0.5": fit["512"]["A"], "0.875": A_hi},
                       F0=F0, tau_tokens=tau),
        held_out=results,
        verdict=verdict,
    )
    (out_dir / "result.json").write_text(json.dumps(envelope, indent=1))

    print(f"p* = {p_star:.2f}   (joint weighted SSE {sse:.1f})")
    print(f"A: 512 {fit['512']['A']:.3f}±{fit['512']['se_A']:.3f}  "
          f"896 {fit['896']['A']:.3f}±{fit['896']['se_A']:.3f}  "
          f"1120 {fit['1120']['A']:.3f}±{fit['1120']['se_A']:.3f}  "
          f"(ratio-twin z={ratio_consistency_z:.2f})")
    print(f"F: 512 {fit['512']['F']:+.3f}  896 {fit['896']['F']:+.3f}  "
          f"1120 {fit['1120']['F']:+.3f} (exp-fit predicts {f1120_pred:+.3f})")
    print(f"F(2160) predicted for 768: {F_of_tokens(TOKENS['768']):+.3f} "
          f"(measured debiased floor 0.043..0.094)")
    for route in HELD_OUT:
        r = results[route]
        print(f"held-out {route}: RMSE {r['rmse']:.3f}  chi2/bin {r['chi2']:.2f}  "
              f"oracle {r['rmse_oracle']:.3f}  null(committed) {r['rmse_null_committed']:.3f} "
              f"vs ours {r['rmse_ours_committed']:.3f}")
    print("VERDICT:", "PASS" if verdict["PASS"] else "FAIL",
          {k: v for k, v in verdict.items() if k.startswith('pass_')})
    print("out:", out_dir)


if __name__ == "__main__":
    main()
