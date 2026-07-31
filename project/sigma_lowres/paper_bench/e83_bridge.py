"""E8.3 — spectral-account -> gap bridge overlay (analysis-only, no GPU).

required_experiments.md E8.3: convert each published tolerance delta into
a predicted *gap curve*, not just a boundary, and overlay it on the
measured debiased curves per (delta, route). Subsumes the continuous
t*(delta) sweep figure and the delta_reenc-anchored row.

Model (the diagonal-Gaussian premise of Eq. 4, on the measured RAPSD):
per band f with clean-latent power P(f), noising z = (1-sigma)x + sigma*eps.
The Bayes posterior mean is Wiener, x_hat = (1-sigma)P/D * z with
D = (1-sigma)^2 P + sigma^2, and the Bayes velocity is
v* = (z - x_hat)/sigma. The demoted grid carries no content above its
Nyquist cut f_cut = e/(2 e0) (normalized units, Nyquist = 0.5), so its
Bayes prediction in destroyed bands is the zero-content one; the
cross-grid *mean-residual* mismatch (the model analog of the measured
m(sigma) = ||dr_bar||, a draw-mean at fixed image) is, per band,

  E_x |E_eps[v*_P - v*_0 | x]|^2 = (1-sigma)^4 P^3 / (sigma^2 D^2),

integrated over destroyed bands with annulus weight f:

  m_spec_e(sigma)^2 = sum_{f > fcut_e} f df (1-sigma)^4 P^3/(sigma^2 D^2).

Convention check: the same model's Prop-1 criterion
E|v* - v*_0|-from-noise = (1-sigma)^2 P (1+P) / D reproduces the paper's
Eq. 4 boundary t = 1/(1 + sqrt(delta/(P(1+P-delta)))) exactly (and the
equal-power slice delta = (1+P)/2 gives sigma_eq = sqrt(P)/(1+sqrt(P))),
so the noising convention above is SPD's own.

Bridge (unit honesty: the account emits residual units only, so the
gap-unit curve borrows the two-term account's bridge):

  gap_spec_e(sigma; delta) = A * (m_spec_e(sigma)/G(sigma))^2
                             * 1[sigma < t*_e(delta)],

with ONE gain A calibrated on the single measured-safe route 1024->896
(floorless weighted LS through the origin — the account expresses no
floor, N3) and then FROZEN for 768/512: route identity enters only
through the model's own destroyed-band content. The delta-gate is
Prop 2: the route is committed safe (gap 0) above the t* of its
Nyquist band. The curve shape below the gate is delta-independent;
delta only moves the truncation.

Sensitivity row: the per-draw mismatch E|v*_P - v*_0|^2
= (1-sigma)^2 P^2/(sigma^2 D) is reported alongside (shape_perdraw);
the headline uses the mean-residual shape, matching the estimand of the
measured m(sigma).

Sources (all existing rows, no new instrument):
  - measured debiased curves + G(sigma): runs/20260729-0014-e1b-debiased-map
  - P(f): bench/results/20260730-0940-reenc-floor/curves.json (N=40 RAPSD,
    64 radial bins; same run anchors delta_reenc and t*_reenc)

Usage:
    python project/sigma_lowres/paper_bench/e83_bridge.py
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e5_holdout import E1B, bin_gnorm, paired_stats  # noqa: E402

REENC_FLOOR = HERE.parent / "bench" / "results" / "20260730-0940-reenc-floor"
FIGS = HERE.parent / "paper" / "figs"

ROUTES = ("896", "768", "512")
FCUT = {"896": 0.4375, "768": 0.375, "512": 0.25}
DELTA_REENC = 1e-5  # measured anchor (reenc-floor run; p10-p90 [1,2]e-5)
DELTA_SPD = 0.01  # SPD's published default


def t_star(delta, P):
    """Eq. 4 boundary: sigma above which the band is within delta of noise."""
    return 1.0 / (1.0 + np.sqrt(delta / (P * (1.0 + P - delta))))


def m_spec(sigma, freq, P, fcut, shape="meanres"):
    """Destroyed-band residual mismatch of the diagonal model, in
    (relative) velocity units. shape='meanres' is the draw-mean object
    (the measured m's estimand); 'perdraw' the per-draw one."""
    s = np.asarray(sigma)[:, None]
    f, p = freq[None, :], P[None, :]
    D = (1.0 - s) ** 2 * p + s**2
    if shape == "meanres":
        band = (1.0 - s) ** 4 * p**3 / (s**2 * D**2)
    else:
        band = (1.0 - s) ** 2 * p**2 / (s**2 * D)
    w = np.where(f > fcut, f, 0.0)
    return np.sqrt(np.sum(w * band, axis=1) * (freq[1] - freq[0]))


def main() -> None:
    curves = {r: paired_stats(E1B, r, "debiased_gap") for r in ROUTES}
    G = bin_gnorm(E1B)
    sigma = curves["896"][0]

    rap = json.load((REENC_FLOOR / "curves.json").open())
    freq, P = np.array(rap["freq"]), np.array(rap["P_mean"])
    P_cut = {r: float(np.interp(FCUT[r], freq, P)) for r in ROUTES}

    # three anchored tolerances; delta_eq is the equal-power slice, a
    # per-route delta = (1+P_cut)/2 whose t* is sigma_eq exactly
    anchors = {
        "eq": {r: (1.0 + P_cut[r]) / 2.0 for r in ROUTES},
        "spd": {r: DELTA_SPD for r in ROUTES},
        "reenc": {r: DELTA_REENC for r in ROUTES},
    }
    tstars = {
        a: {r: float(t_star(anchors[a][r], P_cut[r])) for r in ROUTES} for a in anchors
    }

    report = {}
    for shape in ("meanres", "perdraw"):
        x = {r: (m_spec(sigma, freq, P, FCUT[r], shape) / G) ** 2 for r in ROUTES}
        # calibrate the single gain on the safe route, floorless, WLS
        _, y896, sem896 = curves["896"]
        w = 1.0 / sem896**2
        A = float(np.sum(w * x["896"] * y896) / np.sum(w * x["896"] ** 2))
        rep = {"A_calibrated_on_896": A, "routes": {}}
        for r in ROUTES:
            _, y, sem = curves[r]
            base = A * x[r]
            entry = {
                "untruncated": {
                    "predicted": base.tolist(),
                    "rmse": float(np.sqrt(np.mean((y - base) ** 2))),
                }
            }
            for a in anchors:
                ts = tstars[a][r]
                pred = np.where(sigma < ts, base, 0.0)
                above = sigma > ts
                entry[a] = {
                    "delta": anchors[a][r],
                    "t_star": ts,
                    "predicted": pred.tolist(),
                    "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
                    "rmse_committed": (
                        float(np.sqrt(np.mean((y[above]) ** 2)))
                        if above.any()
                        else None
                    ),
                    "n_committed_bins": int(above.sum()),
                }
            rep["routes"][r] = entry
        report[shape] = rep

    # continuous t*(delta) sweep (family compression read)
    dgrid = np.geomspace(1e-3, 0.5, 200)
    sweep = {r: t_star(dgrid, P_cut[r]) for r in ROUTES}
    spread = np.max(sweep["512"] - sweep["896"])

    head = report["meanres"]

    # ---- figure: three overlay panels + the t*(delta) sweep ----
    fig, axes = plt.subplots(
        1, 4, figsize=(14.5, 3.4), gridspec_kw={"width_ratios": [1, 1, 1, 0.9]}
    )
    acolor = {"eq": "C2", "spd": "C3", "reenc": "C4"}
    alabel = {
        "eq": r"$\delta_{\mathrm{eq}}$ (equal power)",
        "spd": r"$\delta=0.01$ (SPD default)",
        "reenc": r"$\delta_{\mathrm{reenc}}$ (anchored)",
    }
    for ax, r in zip(axes[:3], ROUTES):
        sig, y, sem = curves[r]
        ax.errorbar(
            sig,
            y,
            yerr=sem,
            fmt="o",
            ms=4,
            color="C0",
            capsize=2,
            label="measured (debiased)",
            zorder=3,
        )
        for a, ls, lw in (("reenc", "-", 1.6), ("spd", "--", 1.3), ("eq", "-", 1.1)):
            e = head["routes"][r][a]
            ax.plot(
                sig,
                e["predicted"],
                ls,
                color=acolor[a],
                lw=lw,
                label=f"{alabel[a]}  RMSE {e['rmse']:.3f}",
            )
            ax.axvline(e["t_star"], color=acolor[a], lw=0.7, ls="--", alpha=0.45)
        ax.axhline(0, color="0.8", lw=0.8)
        ax.set_title(f"$1024\\to{r}$")
        ax.set_xlabel(r"$\sigma$")
        if r == "896":
            ax.set_ylabel(r"paired gap $\overline{\mathrm{gap}}$")
        ax.legend(fontsize=5.8, loc="upper right")
    ax = axes[3]
    rcolor = {"896": "C0", "768": "C1", "512": "C2"}
    for r in ROUTES:
        ax.plot(dgrid, sweep[r], color=rcolor[r], label=f"$1024\\to{r}$")
    for a in ("eq", "spd", "reenc"):
        d = anchors[a]["768"] if a != "eq" else None
        if d:
            ax.axvline(d, color=acolor[a], lw=0.7, ls="--", alpha=0.5)
    ax.axvline(DELTA_REENC, color=acolor["reenc"], lw=0.7, ls="--", alpha=0.5)
    ax.annotate(
        f"max spread {spread:.2f}",
        (0.96, 0.9),
        xycoords="axes fraction",
        fontsize=7,
        ha="right",
    )
    ax.set_xscale("log")
    ax.set_xlim(dgrid[0], dgrid[-1])
    ax.set_xlabel(r"tolerance $\delta$")
    ax.set_ylabel(r"predicted boundary $t^{*}(\delta)$")
    ax.set_title(r"$t^{*}(\delta)$: the whole family")
    ax.legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()

    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M")
    out_dir = HERE / "runs" / f"{stamp}-e83-bridge"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "e83_overlay.png", dpi=180, bbox_inches="tight")
    shutil.copy2(out_dir / "e83_overlay.png", FIGS / "e83_overlay.png")

    envelope = dict(
        schema_version=1,
        script="project/sigma_lowres/paper_bench/e83_bridge.py",
        label="e83-bridge",
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        sources=dict(e1b=str(E1B), rapsd=str(REENC_FLOOR)),
        caveats=[
            "gap-unit curves are the spectral account read through OUR "
            "bridge (G(sigma) + one gain calibrated on 1024->896, "
            "floorless); not a parameter-free prediction",
            "headline shape is the mean-residual (draw-mean) mismatch, "
            "matching the measured m's estimand; perdraw is sensitivity",
            "calibration uses all 896 bins of the ungated bridge; the "
            "896 floor (0.02-0.04) is small but nonzero, slightly "
            "inflating A in the account's favor",
        ],
        f_cut=FCUT,
        P_cut=P_cut,
        t_star=tstars,
        family_spread_max=float(spread),
        shapes=report,
    )
    (out_dir / "result.json").write_text(json.dumps(envelope, indent=1))

    print(f"A (meanres, calibrated on 896) = {head['A_calibrated_on_896']:.4g}")
    print(
        f"{'route':<7}{'t*_eq':>7}{'t*_spd':>8}{'t*_reenc':>10}"
        f"{'rmse_eq':>9}{'rmse_spd':>10}{'rmse_reenc':>11}{'rmse_ungated':>13}"
    )
    for r in ROUTES:
        e = head["routes"][r]
        print(
            f"{r:<7}{e['eq']['t_star']:>7.3f}{e['spd']['t_star']:>8.3f}"
            f"{e['reenc']['t_star']:>10.3f}{e['eq']['rmse']:>9.3f}"
            f"{e['spd']['rmse']:>10.3f}{e['reenc']['rmse']:>11.3f}"
            f"{e['untruncated']['rmse']:>13.3f}"
        )
    for r in ROUTES:
        e = head["routes"][r]
        cm = {a: e[a]["rmse_committed"] for a in ("eq", "spd", "reenc")}
        print(
            f"{r} committed-region rmse: "
            + "  ".join(
                f"{a}={v:.3f}" if v is not None else f"{a}=n/a" for a, v in cm.items()
            )
        )
    print(f"family spread max over delta in [1e-3, 0.5]: {spread:.3f}")
    print("out:", out_dir)


if __name__ == "__main__":
    main()
