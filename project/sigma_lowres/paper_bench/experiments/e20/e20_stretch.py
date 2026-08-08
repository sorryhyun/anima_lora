"""E20.4 — stretch: how derivable is the data term? (CPU, not verdict-bearing)

Pre-registration: experiments/e20/README.md §20.4. 19.5 showed the Gaussian
closure owns both leg directions at mid/high sigma with a smooth
sigma-dependent per-leg scale miss. This arm replaces the MEASURED mismatch
curve x(sigma) = m_bar/G with the CLOSURE-PREDICTED B-leg amplitude curve and
re-runs 20.1's lanes (identical chain, imported from e20_refit — fit routes ->
governors -> held-out 768 / LOO-896 / 1024), bounding how much of the
cancellation account can be *derived* rather than measured.

x arms (scale is absorbed by the per-route A, so only the shape matters):

  measured    m_bar/G — the published 20.1 lane, recomputed as cross-check
  measB       r-ledger MEASURED B amplitude (route-uniform mean of the three
              1024-tier curves)/G — diagnostic: decomposes "closure miss"
              vs "estimand mismatch (m_bar is not the B leg)"
  predB       closure-PREDICTED B amplitude (route-uniform mean)/G — the
              pre-registered derived arm
  predB_lin   (1 + b*sigma) * predB, one global b fitted on the 1024-tier
              fit routes (896+512 joint weighted SSE; 1120 is b-inert) —
              the pre-registered one-dof linear-in-sigma correction
  predB_pr    per-route closure curves (no pooling) — secondary, labeled;
              changes the A-governor's semantics (x no longer route-uniform)

Amplitude source: a rerun of bench/ledger_b_scoreshift.py patched to emit
absolute per-image RMS amplitudes (amp_pred_B / amp_meas_B), diag closure
(closures agree to ~0.01 in 19.5), same 19.2 store + 19.1 fit split. The
1280-tier routes (1120 fit leg, 1024 held-out) have no closure instrument
and keep the measured x in every arm — disclosed, so the clean read is the
768 held-out lane (fully inside closure coverage).

No bootstrap (point estimates; separate reading, not verdict-bearing for
20.1). rho_bar = -0.910 frozen as in 20.1.

Usage:
    uv run python project/sigma_lowres/paper_bench/experiments/e20/e20_stretch.py \
        --amp_run <bench/results/...-e204-amp>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
PAPER_BENCH = HERE.parents[1]
SIGMA = HERE.parents[2]
RUNS = PAPER_BENCH / "runs"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PAPER_BENCH / "experiments" / "e5"))

from e5_holdout import (  # noqa: E402
    E1B,
    FIT_ROUTES,
    G9,
    bin_gnorm,
    bin_widths,
    interp_extrap,
    m_bar,
    paired_stats,
)
from e5_refit import TIER_OF  # noqa: E402
from e20_refit import (  # noqa: E402
    RHO_BAR,
    WINDOW,
    dip_cross,
    fit_canc,
    rmse_pair,
    run_lane,
)

E20_REFIT_RUN = RUNS / "20260808-0924-e20-refit"  # 20.1 verdict run (reference)
ROUTES_1024 = ("896", "768", "512")
CLOSURE_KEY = "diag_reenc_area_meas"


def load_amp_curves(amp_run: Path):
    """(sigma_grid, {route: amp_pred_B}, {route: amp_meas_B}) from the
    patched ledger_b_scoreshift rerun."""
    rec = json.loads((amp_run / "result.json").read_text())
    m = rec["metrics"] if "metrics" in rec else rec
    sig = np.array(m["sigma_centers"], float)
    pred, meas = {}, {}
    for r in ROUTES_1024:
        tab = m["routes"][r][CLOSURE_KEY]["table"]
        pred[r] = np.array([row["amp_pred_B"] for row in tab], float)
        meas[r] = np.array([row["amp_meas_B"] for row in tab], float)
    return sig, pred, meas


def main(amp_run: Path, tag: str = "") -> None:
    # ---- measured curves + lane x, identical to 20.1 ----
    curves = {r: paired_stats(E1B, r, "debiased_gap") for r in ROUTES_1024}
    for r in ("1120", "1024"):
        curves[r] = paired_stats(G9, r, "gap")
    gnorm = {"1024tier": bin_gnorm(E1B), "1280tier": bin_gnorm(G9)}
    src = {"1024tier": E1B, "1280tier": G9}
    widths = {r: bin_widths(src[TIER_OF[r]], len(curves[r][0])) for r in curves}
    msig, mvals = m_bar()
    x_meas = {
        r: interp_extrap(curves[r][0], msig, mvals) / gnorm[TIER_OF[r]] for r in curves
    }

    asig, apred, ameas = load_amp_curves(amp_run)
    pred_bar = np.mean([apred[r] for r in ROUTES_1024], axis=0)
    meas_bar = np.mean([ameas[r] for r in ROUTES_1024], axis=0)

    def x_from(curve_sig, curve_vals):
        """One route-uniform amplitude curve -> per-route x on lane bins;
        1280-tier routes keep the measured lane x."""
        x = dict(x_meas)
        for r in ROUTES_1024:
            x[r] = (
                interp_extrap(curves[r][0], curve_sig, curve_vals) / gnorm["1024tier"]
            )
        return x

    arms = {
        "measured": x_meas,
        "measB": x_from(asig, meas_bar),
        "predB": x_from(asig, pred_bar),
    }
    # per-route secondary arm
    x_pr = dict(x_meas)
    for r in ROUTES_1024:
        x_pr[r] = interp_extrap(curves[r][0], asig, apred[r]) / gnorm["1024tier"]
    arms["predB_pr"] = x_pr

    # no-normalization secondary arms — the alternate literal reading of the
    # pre-registration ("replace x with the amplitude curve"), no /G
    def x_from_raw(curve_vals):
        x = dict(x_meas)
        for r in ROUTES_1024:
            x[r] = interp_extrap(curves[r][0], asig, curve_vals)
        return x

    arms["measB_noG"] = x_from_raw(meas_bar)
    arms["predB_noG"] = x_from_raw(pred_bar)

    # ---- predB_lin: one global b on the 1024-tier fit routes ----
    b_grid = np.linspace(-0.9, 4.0, 50)
    sse_b = []
    for b in b_grid:
        xb = x_from(asig, (1.0 + b * asig) * pred_bar)
        tot = 0.0
        for r in ("896", "512"):
            _, y, sem = curves[r]
            w = widths[r] / sem**2
            tot += fit_canc(xb[r], y, w, RHO_BAR)[0]
        sse_b.append(tot)
    b_star = float(b_grid[int(np.argmin(sse_b))])
    # one refinement pass around b_star
    b_grid2 = np.linspace(b_star - 0.15, b_star + 0.15, 13)
    sse_b2 = []
    for b in b_grid2:
        xb = x_from(asig, (1.0 + b * asig) * pred_bar)
        tot = 0.0
        for r in ("896", "512"):
            _, y, sem = curves[r]
            w = widths[r] / sem**2
            tot += fit_canc(xb[r], y, w, RHO_BAR)[0]
        sse_b2.append(tot)
    b_star = float(b_grid2[int(np.argmin(sse_b2))])
    arms["predB_lin"] = x_from(asig, (1.0 + b_star * asig) * pred_bar)

    # ---- run 20.1's chain per arm ----
    report = {}
    for name, x in arms.items():
        fit, gov, preds, meta = run_lane(curves, x, widths, RHO_BAR)
        entry = dict(fit={r: fit[r] for r in FIT_ROUTES}, gov=gov, lanes={})
        for r in ("768", "896", "1024"):
            if r not in preds:
                entry["lanes"][r] = dict(failed="LOO c-law invalid")
                continue
            full, det = rmse_pair(curves, r, preds[r])
            dip, crossings = dip_cross(curves, x, r, meta[r]["A"], meta[r]["c"])
            entry["lanes"][r] = dict(
                rmse_full=full,
                rmse_det=det,
                dip=dip,
                dip_in_window=bool(WINDOW[0] <= dip <= WINDOW[1]),
                crossing_high=crossings[-1] if crossings else None,
                A=meta[r]["A"],
                c=meta[r]["c"],
                lane=meta[r]["lane"],
            )
        for r in ("512", "1120"):
            full, det = rmse_pair(curves, r, preds[f"{r}_ins"])
            entry["lanes"][f"{r}_ins"] = dict(rmse_full=full, rmse_det=det)
        report[name] = entry
        report[name]["preds"] = {k: np.asarray(v).tolist() for k, v in preds.items()}

    # ---- shape agreement: normalized derived x vs lane x (896 grid) ----
    # normalization = mean 1 over sigma >= 0.3 (scale is absorbed by A);
    # localizes WHERE the derived curves part from the lane's x.
    sig896 = curves["896"][0]
    hi = sig896 >= 0.3
    shape = {}
    lane_n = x_meas["896"] / np.mean(x_meas["896"][hi])
    for name in ("predB", "measB"):
        arm_n = np.asarray(arms[name]["896"]) / np.mean(
            np.asarray(arms[name]["896"])[hi]
        )
        rel = np.abs(arm_n - lane_n) / np.abs(lane_n)
        shape[name] = dict(
            sigma=[float(s) for s in sig896],
            rel_dev=[float(v) for v in rel],
            max_rel_dev_mid=float(np.max(rel[(sig896 >= 0.3) & (sig896 <= 0.7)])),
            rel_dev_endpoint=float(rel[-1]),
            ratio_lowest_bin=float((arm_n / lane_n)[0]),
        )

    # ---- reference numbers from the 20.1 verdict run (same sources) ----
    ref = json.loads((E20_REFIT_RUN / "result.json").read_text())
    additive_det = {r: ref["rmse"][r]["additive_det"] for r in ("768", "896", "1024")}

    # cross-check: the measured arm must reproduce 20.1's central numbers
    xchk = abs(
        report["measured"]["lanes"]["768"]["rmse_det"] - ref["rmse"]["768"]["canc_det"]
    )

    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M")
    out_dir = RUNS / f"{stamp}-e20-stretch{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- figure: x shapes + the two safe lanes ----
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    colors = {
        "measured": "C0",
        "measB": "C4",
        "predB": "#2ca02c",
        "predB_lin": "C1",
        "predB_pr": "0.6",
    }
    label = {
        "measured": r"measured $\bar m/G$ (20.1 lane)",
        "measB": "measured B amp (r-ledger)",
        "predB": "closure-predicted B amp",
        "predB_lin": rf"closure B $\times(1+{b_star:.2f}\sigma)$",
        "predB_pr": "closure B, per-route (896)",
    }
    ax = axes[0]
    sig896 = curves["896"][0]
    for name in ("measured", "measB", "predB", "predB_lin", "predB_pr"):
        xv = np.asarray(arms[name]["896"], float)
        norm = np.mean(xv[sig896 >= 0.3])
        ax.plot(sig896, xv / norm, "-", color=colors[name], lw=1.6, label=label[name])
    ax.set_title(
        "x(σ) shapes on the 896 grid\n(normalized: mean 1 over σ≥0.3)", fontsize=9.5
    )
    ax.set_xlabel(r"$\sigma$")
    ax.legend(fontsize=6.5, loc="best")

    for ax, r in zip(axes[1:], ("768", "896")):
        sig, y, sem = curves[r]
        has_end = math.isclose(float(sig[-1]), 1.0, abs_tol=1e-9)
        n_in = len(sig) - 1 if has_end else len(sig)
        ax.errorbar(
            sig[:n_in],
            y[:n_in],
            yerr=sem[:n_in],
            fmt="o",
            ms=4,
            color="0.15",
            capsize=2,
            lw=1.0,
            zorder=4,
            label="measured (debiased)",
        )
        if has_end:
            ax.errorbar(
                sig[-1:],
                y[-1:],
                yerr=sem[-1:],
                fmt="o",
                ms=4.5,
                mfc="white",
                color="0.15",
                capsize=2,
                lw=1.0,
                zorder=4,
            )
        for name in ("measured", "measB", "predB", "predB_lin"):
            lane = report[name]["lanes"][r]
            if "failed" in lane:
                continue
            ax.plot(
                sig,
                report[name]["preds"][r],
                "-",
                color=colors[name],
                lw=1.7,
                label=f"{label[name]}  RMSE* {lane['rmse_det']:.3f}",
            )
        ax.axhline(0, color="0.8", lw=0.6, zorder=0)
        kind = "held-out" if r == "768" else "LOO-896"
        ax.set_title(
            f"$1024\\to{r}$ ({kind}; additive ref {additive_det[r]:.3f})", fontsize=9.5
        )
        ax.set_xlabel(r"$\sigma$")
        ax.legend(fontsize=6.5, loc="best")
    axes[1].set_ylabel("debiased gap")
    fig.suptitle(
        "E20.4 — derived data term through 20.1's lanes (not verdict-bearing)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "e20_stretch.png", dpi=180, bbox_inches="tight")

    envelope = dict(
        schema_version=1,
        script="project/sigma_lowres/paper_bench/experiments/e20/e20_stretch.py",
        label="e20-stretch",
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        sources=dict(
            e1b=str(E1B), g9=str(G9), amp_run=str(amp_run), e20_refit=str(E20_REFIT_RUN)
        ),
        frozen=dict(rho_bar=RHO_BAR, window=list(WINDOW), closure=CLOSURE_KEY),
        caveats=[
            "not verdict-bearing for 20.1 (pre-registered separate reading)",
            "1280-tier routes keep measured x in every arm (no closure "
            "instrument there); clean read = 768 held-out lane",
            "route-uniform arms pool the three 1024-tier amplitude curves "
            "(mirrors m_bar's route-uniform convention); predB_pr is the "
            "labeled per-route secondary",
            "point estimates only, no bootstrap",
            f"measured-arm cross-check vs 20.1 central: |d(RMSE*768)| = {xchk:.2e}",
        ],
        b_star=b_star,
        arms={
            n: {k: v for k, v in e.items() if k != "preds"} for n, e in report.items()
        },
        shape_agreement=shape,
        additive_reference_det=additive_det,
    )
    (out_dir / "result.json").write_text(json.dumps(envelope, indent=1))

    print(
        f"amp run: {amp_run.name}   b* = {b_star:+.2f}   "
        f"measured-arm cross-check d(768 RMSE*) = {xchk:.2e}"
    )
    hdr = f"{'arm':<11}{'768 RMSE*':>10}{'dip768':>8}{'896 RMSE*':>10}{'dip896':>8}{'1024':>8}{'512ins':>8}"
    print(hdr)
    for name in (
        "measured",
        "measB",
        "predB",
        "predB_lin",
        "predB_pr",
        "measB_noG",
        "predB_noG",
    ):
        ln = report[name]["lanes"]
        d768 = ln["768"]
        d896 = ln.get("896", {})
        print(
            f"{name:<11}{d768['rmse_det']:>10.4f}{d768['dip']:>8.3f}"
            f"{d896.get('rmse_det', float('nan')):>10.4f}"
            f"{d896.get('dip', float('nan')):>8.3f}"
            f"{ln['1024']['rmse_det']:>8.4f}"
            f"{ln['512_ins']['rmse_det']:>8.4f}"
        )
    print(
        f"{'(additive)':<11}{additive_det['768']:>10.4f}{'--':>8}"
        f"{additive_det['896']:>10.4f}{'--':>8}"
        f"{additive_det['1024']:>8.4f}{'--':>8}"
    )
    for name, s in shape.items():
        print(
            f"shape {name}: max rel dev {s['max_rel_dev_mid'] * 100:.0f}% over "
            f"sigma 0.3-0.7, {s['rel_dev_endpoint'] * 100:.0f}% at endpoint, "
            f"lowest-bin ratio {s['ratio_lowest_bin']:.2f}"
        )
    print("out:", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--amp_run",
        required=True,
        help="patched ledger_b_scoreshift run dir (e204-amp)",
    )
    ap.add_argument("--tag", default="")
    _a = ap.parse_args()
    main(Path(_a.amp_run).resolve(), _a.tag)
