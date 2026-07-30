"""E5 refit — three functional forms for the data-branch term (analysis-only).

paper_plan.md §6 step 2: the external theory review found the cosine
geometry derives a *quadratic* local law (1-cos ~= 0.5*||dg_perp||^2/G^2),
so the shipped `A*m/G^p` form is neither derived nor honestly-labeled
empirical. This refit runs the SAME fit -> governors -> held-out pipeline
as e5_holdout.py under three candidate forms for the data-branch term S_e
(shared additive floor F_e in all three):

  P  power (current empirical):        S = A * m(sigma) / G(sigma)^p,  p shared, scanned
  Q  derived small-perturbation:       S = A * (m(sigma)/G(sigma))^2
  X  exact angular link (saturating):  S = 1 - 1/sqrt(1 + (c * m/G)^2)

X is the exact 1-cos of a pure orthogonal perturbation with
||dg_perp||/G = c*m/G; its small-kappa limit is Q with A = c^2/2. The
severe 512 route (debiased gap ~0.3) sits far outside the small-kappa
regime, so Q cannot cover it by construction — the discriminating reads
are (a) in-sample chi2/bin on 512 and (b) held-out RMSE on {768, 1024}.

Deliverable for the claims ledger: name the "best empirical predictor"
(headlines, labeled empirical) and the "derived small-perturbation form"
(reported separately) — see paper_plan.md §6.2.

Sources: identical to e5_holdout.py (no new instrument, no GPU).

Usage:
    python project/sigma_lowres/paper_bench/e5_refit.py
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e5_holdout import (  # noqa: E402
    E1B,
    FIT_ROUTES,
    G9,
    HELD_OUT,
    RATIO,
    TOKENS,
    bin_gnorm,
    interp_extrap,
    m_bar,
    paired_stats,
    wls_line,
)

TIER_OF = {"896": "1024tier", "768": "1024tier", "512": "1024tier",
           "1120": "1280tier", "1024": "1280tier"}


def sat(kappa):
    """Exact angular gap of a pure orthogonal perturbation of relative size kappa."""
    return 1.0 - 1.0 / np.sqrt(1.0 + kappa**2)


def fit_sat_route(x0, y, w):
    """Weighted fit of y = sat(c*x0) + F over a two-stage c grid.

    Returns c, F, se_c (profile-likelihood 1-sigma from the chi2 curve).
    """
    c_med = 1.0 / max(float(np.median(x0)), 1e-12)
    best = None
    for grid in (np.geomspace(c_med * 1e-2, c_med * 1e2, 400), None):
        if grid is None:
            c0 = best[1]
            grid = np.geomspace(c0 / 1.3, c0 * 1.3, 400)
        for c in grid:
            s = sat(c * x0)
            F = float(np.sum(w * (y - s)) / np.sum(w))
            sse = float(np.sum(w * (y - s - F) ** 2))
            if best is None or sse < best[0]:
                best = (sse, c, F)
    sse_min, c_star, F_star = best
    # profile se: sweep c, refit F, find where sse crosses sse_min + 1
    lo = hi = c_star
    for c in np.geomspace(c_star, c_star * 3, 200):
        s = sat(c * x0)
        F = float(np.sum(w * (y - s)) / np.sum(w))
        if float(np.sum(w * (y - s - F) ** 2)) > sse_min + 1.0:
            hi = c
            break
    for c in np.geomspace(c_star, c_star / 3, 200):
        s = sat(c * x0)
        F = float(np.sum(w * (y - s)) / np.sum(w))
        if float(np.sum(w * (y - s - F) ** 2)) > sse_min + 1.0:
            lo = c
            break
    se_c = 0.5 * (hi - lo)
    # se_F from the weights at fixed c (same convention as wls_line)
    se_F = math.sqrt(1.0 / float(np.sum(w)))
    return c_star, F_star, se_c, se_F, sse_min


def exp_floor_law(fpts):
    """F0*exp(-n/tau) through the positive floors (e5_holdout convention)."""
    logpts = [(n, math.log(F)) for n, F, _ in fpts if F > 0.005]
    (n1, l1), (n2, l2) = logpts[0], logpts[-1]
    tau = (n1 - n2) / (l2 - l1)
    F0 = math.exp(l1 + n1 / tau)
    return F0, tau


def two_point_governor(vals, ses):
    """Linear-in-ratio interpolation through (0.5, v_512), (0.875, mean hi)."""
    v_hi = 0.5 * (vals["896"] + vals["1120"])
    var_hi = 0.25 * (ses["896"] ** 2 + ses["1120"] ** 2)
    z = abs(vals["896"] - vals["1120"]) / math.sqrt(ses["896"] ** 2 + ses["1120"] ** 2)

    def gov(r):
        t = (r - 0.5) / (0.875 - 0.5)
        v = (1 - t) * vals["512"] + t * v_hi
        var = (1 - t) ** 2 * ses["512"] ** 2 + t**2 * var_hi
        return v, math.sqrt(var)

    return gov, z


def main() -> None:
    msig, mvals = m_bar()
    curves = {}
    for route in ("896", "768", "512"):
        curves[route] = paired_stats(E1B, route, "debiased_gap")
    for route in ("1120", "1024"):
        curves[route] = paired_stats(G9, route, "gap")
    gnorm = {"1024tier": bin_gnorm(E1B), "1280tier": bin_gnorm(G9)}

    def x_ratio(route):
        """m(sigma)/G(sigma) on the route's own sigma grid."""
        sig, _, _ = curves[route]
        return interp_extrap(sig, msig, mvals) / gnorm[TIER_OF[route]]

    def x_power(route, p):
        sig, _, _ = curves[route]
        return interp_extrap(sig, msig, mvals) / gnorm[TIER_OF[route]] ** p

    forms = {}

    # ---- P: shared-p power (reproduces e5_holdout's fit) ----
    best = None
    for p in np.arange(1.0, 2.001, 0.01):
        sse, params = 0.0, {}
        for route in FIT_ROUTES:
            _, y, sem = curves[route]
            x, w = x_power(route, p), 1.0 / sem**2
            A, F, se_A, se_F = wls_line(x, y, w)
            sse += float(np.sum(w * (y - (A * x + F)) ** 2))
            params[route] = dict(A=A, F=F, se_A=se_A, se_F=se_F)
        if best is None or sse < best[0]:
            best = (sse, p, params)
    _, p_star, fitP = best
    forms["P"] = dict(kind="power", p=float(p_star), fit=fitP)

    # ---- Q: derived quadratic (m/G)^2 ----
    fitQ = {}
    for route in FIT_ROUTES:
        _, y, sem = curves[route]
        x, w = x_ratio(route) ** 2, 1.0 / sem**2
        A, F, se_A, se_F = wls_line(x, y, w)
        fitQ[route] = dict(A=A, F=F, se_A=se_A, se_F=se_F)
    forms["Q"] = dict(kind="quad", fit=fitQ)

    # ---- X: exact angular link ----
    fitX = {}
    for route in FIT_ROUTES:
        _, y, sem = curves[route]
        c, F, se_c, se_F, _ = fit_sat_route(x_ratio(route), y, 1.0 / sem**2)
        fitX[route] = dict(A=c, F=F, se_A=se_c, se_F=se_F)  # A slot holds c
    forms["X"] = dict(kind="exact", fit=fitX)

    # ---- shared evaluation ----
    def predict(form, route, gov_A, F_law):
        A, _ = gov_A(RATIO[route])
        F = F_law(TOKENS[route])
        if form["kind"] == "power":
            return A * x_power(route, form["p"]) + F
        if form["kind"] == "quad":
            return A * x_ratio(route) ** 2 + F
        return sat(A * x_ratio(route)) + F

    def insample_pred(form, route):
        prm = form["fit"][route]
        if form["kind"] == "power":
            return prm["A"] * x_power(route, form["p"]) + prm["F"]
        if form["kind"] == "quad":
            return prm["A"] * x_ratio(route) ** 2 + prm["F"]
        return sat(prm["A"] * x_ratio(route)) + prm["F"]

    report = {}
    for name, form in forms.items():
        fit = form["fit"]
        gov_A, z = two_point_governor(
            {r: fit[r]["A"] for r in FIT_ROUTES},
            {r: fit[r]["se_A"] for r in FIT_ROUTES},
        )
        fpts = [(TOKENS[r], fit[r]["F"], fit[r]["se_F"]) for r in FIT_ROUTES]
        F0, tau = exp_floor_law(fpts)
        F_law = lambda n, F0=F0, tau=tau: F0 * math.exp(-n / tau)

        ins = {}
        for route in FIT_ROUTES:
            _, y, sem = curves[route]
            pr = insample_pred(form, route)
            ins[route] = dict(rmse=float(np.sqrt(np.mean((y - pr) ** 2))),
                              chi2=float(np.mean(((y - pr) / sem) ** 2)))
        held = {}
        for route in HELD_OUT:
            sig, y, sem = curves[route]
            pred = predict(form, route, gov_A, F_law)
            w = 1.0 / sem**2
            coef = np.polyfit(sig, y, 2, w=np.sqrt(w))
            oracle = np.polyval(coef, sig)
            held[route] = dict(
                sigma=sig.tolist(), predicted=pred.tolist(),
                rmse=float(np.sqrt(np.mean((y - pred) ** 2))),
                chi2=float(np.mean(((y - pred) / sem) ** 2)),
                rmse_oracle=float(np.sqrt(np.mean((y - oracle) ** 2))),
            )
        report[name] = dict(
            kind=form["kind"], p=form.get("p"),
            params={r: fit[r] for r in FIT_ROUTES},
            ratio_twin_z=z, F0=F0, tau_tokens=tau,
            f768_predicted=F_law(TOKENS["768"]),
            insample=ins, held_out=held,
        )

    # ---- verdict ----
    chi512 = {n: report[n]["insample"]["512"]["chi2"] for n in report}
    ho_rmse = {n: float(np.mean([report[n]["held_out"][r]["rmse"] for r in HELD_OUT]))
               for n in report}
    best_512 = min(chi512, key=chi512.get)
    best_ho = min(ho_rmse, key=ho_rmse.get)
    verdict = dict(
        insample_512_chi2=chi512, held_out_mean_rmse=ho_rmse,
        best_512=best_512, best_held_out=best_ho,
        headline_empirical=best_ho,
        derived_form="Q",
        note=("X's small-kappa limit is Q with A=c^2/2; agreement of X and Q "
              "away from 512 plus X covering 512 supports the angular account. "
              "Whichever headlines is labeled empirical; Q is reported as the "
              "derived small-perturbation form (paper_plan §6.2)."),
    )

    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M")
    out_dir = HERE / "runs" / f"{stamp}-e5-refit"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- figures: held-out 768 first, then the fit routes 896/512 (paper
    # main figure); the second held-out route 1280->1024 as its own figure
    # (paper appendix). Three forms each.
    colors = {"P": "C3", "Q": "C2", "X": "C1"}
    label = {"P": f"P: A·m/G^p (p={p_star:.2f})", "Q": "Q: A·(m/G)² (derived)",
             "X": "X: 1−1/√(1+(c·m/G)²)"}

    def draw_panel(ax, route):
        sig, y, sem = curves[route]
        ax.errorbar(sig, y, yerr=sem, fmt="o", ms=4, color="C0", capsize=2,
                    label="measured (paired)")
        for name, form in forms.items():
            if route in FIT_ROUTES:
                pr = insample_pred(form, route)
                tag = f"{label[name]}  χ²/bin {report[name]['insample'][route]['chi2']:.1f}"
            else:
                pr = report[name]["held_out"][route]["predicted"]
                tag = f"{label[name]}  RMSE {report[name]['held_out'][route]['rmse']:.3f}"
            ax.plot(sig, pr, "-", color=colors[name], label=tag)
        ax.axhline(0, color="0.8", lw=0.8)
        kind = "in-sample" if route in FIT_ROUTES else "held-out"
        src = "1280→1024" if route == "1024" else f"1024→{route}"
        ax.set_title(f"{src} ({kind})")
        ax.set_xlabel(r"$\sigma$")
        ax.legend(fontsize=6.5, loc="best")

    panels = ["768", "896", "512"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for ax, route in zip(axes, panels):
        draw_panel(ax, route)
    axes[0].set_ylabel(r"paired gap $\bar\Delta$")
    fig.tight_layout()
    fig.savefig(out_dir / "e5_refit.png", dpi=180, bbox_inches="tight")

    fig2, ax2 = plt.subplots(figsize=(5.4, 3.8))
    draw_panel(ax2, "1024")
    ax2.set_ylabel(r"paired gap $\bar\Delta$")
    fig2.tight_layout()
    fig2.savefig(out_dir / "e5_refit_1280.png", dpi=180, bbox_inches="tight")

    envelope = dict(
        schema_version=1,
        script="project/sigma_lowres/paper_bench/e5_refit.py",
        label="e5-refit",
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        sources=dict(e1b=str(E1B), g9=str(G9)),
        caveats=[
            "same sources and caveats as e5_holdout.py (RAW 1280-tier, D=4; "
            "m extrapolated below sigma=0.125)",
            "X fit is per-route grid + profile-likelihood se on c",
        ],
        forms=report,
        verdict=verdict,
    )
    (out_dir / "result.json").write_text(json.dumps(envelope, indent=1))

    print(f"P: shared p* = {p_star:.2f}")
    print(f"{'form':<6}{'chi2/bin 512':>14}{'HO 768 rmse':>13}{'HO 1024 rmse':>14}{'HO mean':>9}")
    for n in ("P", "Q", "X"):
        r = report[n]
        print(f"{n:<6}{chi512[n]:>14.2f}{r['held_out']['768']['rmse']:>13.3f}"
              f"{r['held_out']['1024']['rmse']:>14.3f}{ho_rmse[n]:>9.3f}")
    for n in ("P", "Q", "X"):
        prm = report[n]["params"]
        print(f"{n} params: " + "  ".join(
            f"{r}: A={prm[r]['A']:.3g}±{prm[r]['se_A']:.2g} F={prm[r]['F']:+.3f}"
            for r in FIT_ROUTES)
            + f"   ratio-twin z={report[n]['ratio_twin_z']:.2f}"
            + f"   F(2160)={report[n]['f768_predicted']:+.3f}")
    print(f"VERDICT: best in-sample 512 = {best_512}, best held-out = {best_ho} "
          f"(headline empirical = {best_ho}; derived form = Q)")
    print("out:", out_dir)


if __name__ == "__main__":
    main()
