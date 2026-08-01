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

Also emitted (paper Fig 2 redesign, 2026-07-30): a leave-896-out lane
(governors re-derived from 512+1120 alone, post-hoc — the addendum of
experiments/e5/README.md, here under the X form) and a full-pipeline
image bootstrap (resample images -> re-bin -> refit routes -> re-derive
governors -> re-predict, B=1000, fixed seed) whose 68/95% quantiles band
the predictions in e5_main.png. m_bar and the bin-mean G are run-level
instruments with no per-image decomposition and are held fixed. The
bootstrap replicates exp_floor_law verbatim, including its positive-
floor filter — draws where the resampled 1120 floor clears the filter
re-anchor the law on the (896, 1120) legs; the branch tally lands in
the envelope.

Sources: identical to e5_holdout.py (no new instrument, no GPU).

Usage:
    python project/sigma_lowres/paper_bench/experiments/e5/e5_refit.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PAPER_BENCH = HERE.parents[1]  # project/sigma_lowres/paper_bench
SIGMA = HERE.parents[2]  # project/sigma_lowres
RUNS = PAPER_BENCH / "runs"
sys.path.insert(0, str(HERE))

from e5_holdout import (  # noqa: E402
    E1B,
    FIT_ROUTES,
    G9,
    HELD_OUT,
    RATIO,
    TOKENS,
    TRIM_ABS,
    bin_gnorm,
    bin_widths,
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


def per_image_matrix(run_dir, route, key):
    """(sigma, (n_images, n_bins)) paired deltas; trim -> nan."""
    recs = [json.loads(line) for line in (run_dir / "per_image.jsonl").open()]
    sigma = np.array(recs[0]["sigma_centers"])
    rows = []
    for r in recs:
        d = np.array([np.nan if v is None else v for v in r[f"{key}_{route}"]], float)
        c = np.array([np.nan if v is None else v for v in r[f"{key}_reenc"]], float)
        delta = d - c
        delta[np.abs(delta) > TRIM_ABS] = np.nan
        rows.append(delta)
    return sigma, np.array(rows)


def fit_sat_vec(x, y, w):
    """Vectorized two-stage grid version of fit_sat_route (c, F only)."""
    c_med = 1.0 / max(float(np.median(x)), 1e-12)
    grid = np.geomspace(c_med * 1e-2, c_med * 1e2, 300)
    for _ in range(2):
        s = sat(np.outer(grid, x))
        F = np.sum(w * (y - s), axis=1) / np.sum(w)
        sse = np.sum(w * (y - s - F[:, None]) ** 2, axis=1)
        i = int(np.argmin(sse))
        c0 = grid[i]
        grid = np.geomspace(c0 / 1.3, c0 * 1.3, 300)
    return float(c0), float(F[i])


def loo896_floor(F512, F1120):
    """Leave-896-out floor law: 512+1120 legs, fragile F_1120 enters the
    log fit clamped positive (addendum convention)."""
    F512, F1120 = max(F512, 1e-4), max(F1120, 1e-4)
    if F1120 >= F512:
        return None
    tau = (TOKENS["512"] - TOKENS["1120"]) / (math.log(F1120) - math.log(F512))
    F0 = math.exp(math.log(F512) + TOKENS["512"] / tau)
    return F0 * math.exp(-TOKENS["896"] / tau)


def bootstrap_predictions(mats, x_of, widths, B=1000, seed=0):
    """Full-pipeline bootstrap of the X-form fit->governor->predict chain.

    Resamples images within each run, re-bins, refits (c, F) per fit
    route, re-derives the ratio governor and both floor laws (pre-reg
    replica of exp_floor_law incl. its >0.005 filter and FIT_ROUTES
    ordering; leave-896-out via loo896_floor), then re-predicts the
    768 (pre-registered pipeline) and 896 (leave-out) curves.
    """
    rng = np.random.default_rng(seed)
    n_e1b = mats["896"].shape[0]
    n_g9 = mats["1120"].shape[0]
    pred768, pred896, f2160s, f3012s = [], [], [], []
    legs = {}
    n_clamp = n_drop = 0
    for _ in range(B):
        ie = rng.integers(0, n_e1b, n_e1b)
        ig = rng.integers(0, n_g9, n_g9)
        fit = {}
        ok = True
        for r in FIT_ROUTES:
            sub = mats[r][ie if TIER_OF[r] == "1024tier" else ig]
            n = np.sum(np.isfinite(sub), axis=0)
            y = np.nanmean(sub, axis=0)
            sem = np.nanstd(sub, axis=0, ddof=1) / np.sqrt(n)
            if not np.all(np.isfinite(sem)) or np.any(sem <= 0):
                ok = False
                break
            fit[r] = fit_sat_vec(x_of(r), y, widths[r] / sem**2)
        if not ok:
            n_drop += 1
            continue
        # pre-registered floor law: exp_floor_law replica
        fpts = [(TOKENS[r], fit[r][1]) for r in FIT_ROUTES]
        logpts = [(n, math.log(F)) for n, F in fpts if F > 0.005]
        if len(logpts) < 2 or logpts[0][1] == logpts[-1][1]:
            n_drop += 1
            continue
        (n1, l1), (n2, l2) = logpts[0], logpts[-1]
        tau = (n1 - n2) / (l2 - l1)
        if tau <= 0:
            n_drop += 1
            continue
        F0 = math.exp(l1 + n1 / tau)
        F3012_lo = loo896_floor(fit["512"][1], fit["1120"][1])
        if F3012_lo is None:
            n_drop += 1
            continue
        n_clamp += fit["1120"][1] <= 1e-4 or fit["512"][1] <= 1e-4
        key = "+".join(str(n) for n in sorted((n1, n2)))
        legs[key] = legs.get(key, 0) + 1
        t = (RATIO["768"] - 0.5) / (0.875 - 0.5)
        c768 = (1 - t) * fit["512"][0] + t * 0.5 * (fit["896"][0] + fit["1120"][0])
        F2160 = F0 * math.exp(-TOKENS["768"] / tau)
        pred768.append(sat(c768 * x_of("768")) + F2160)
        pred896.append(sat(fit["1120"][0] * x_of("896")) + F3012_lo)
        f2160s.append(F2160)
        f3012s.append(F3012_lo)
    return dict(
        pred768=np.array(pred768), pred896=np.array(pred896),
        f2160=np.array(f2160s), f3012=np.array(f3012s),
        kept=len(pred768), dropped=n_drop, loo_clamped=n_clamp,
        floor_law_legs=legs, B=B, seed=seed,
    )


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


def main(tier1024: Path = E1B, tag: str = "") -> None:
    msig, mvals = m_bar()
    curves = {}
    for route in ("896", "768", "512"):
        curves[route] = paired_stats(tier1024, route, "debiased_gap")
    for route in ("1120", "1024"):
        curves[route] = paired_stats(G9, route, "gap")
    gnorm = {"1024tier": bin_gnorm(tier1024), "1280tier": bin_gnorm(G9)}
    src = {"1024tier": tier1024, "1280tier": G9}
    widths = {r: bin_widths(src[TIER_OF[r]], len(curves[r][0])) for r in curves}

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
            x, w = x_power(route, p), widths[route] / sem**2
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
        x, w = x_ratio(route) ** 2, widths[route] / sem**2
        A, F, se_A, se_F = wls_line(x, y, w)
        fitQ[route] = dict(A=A, F=F, se_A=se_A, se_F=se_F)
    forms["Q"] = dict(kind="quad", fit=fitQ)

    # ---- X: exact angular link ----
    fitX = {}
    for route in FIT_ROUTES:
        _, y, sem = curves[route]
        c, F, se_c, se_F, _ = fit_sat_route(x_ratio(route), y, widths[route] / sem**2)
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
            w = widths[route] / sem**2
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
    out_dir = RUNS / f"{stamp}-e5-refit{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- figures: held-out 768 first, then the fit routes 896/512 (paper
    # main figure); the second held-out route 1280->1024 as its own figure
    # (paper appendix). Three forms each.
    colors = {"P": "C3", "Q": "C2", "X": "C1"}
    # m := ||dr_bar||, spelled out in the labels to match the manuscript,
    # which carries no separate symbol for it; G := ||g_bar_src(sigma)||.
    label = {
        "P": rf"P: $A\,\|\Delta\bar r\|/G^p$  ($p={p_star:.2f}$)",
        "Q": r"Q: $A\,(\|\Delta\bar r\|/G)^2$  (derived)",
        "X": r"X: $1-1/\sqrt{1+(c\,\|\Delta\bar r\|/G)^2}$",
    }

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

    # ---- leave-896-out lane (X form; post-hoc addendum) ----
    c_lo = fitX["1120"]["A"]
    F3012_lo = loo896_floor(fitX["512"]["F"], fitX["1120"]["F"])
    pred896_lo = sat(c_lo * x_ratio("896")) + F3012_lo
    _, y896, _ = curves["896"]
    loo896 = dict(
        c=c_lo, F3012=F3012_lo,
        rmse=float(np.sqrt(np.mean((y896 - pred896_lo) ** 2))),
    )

    # ---- full-pipeline bootstrap (X form) ----
    mats = {}
    for r in ("896", "512"):
        _, mats[r] = per_image_matrix(tier1024, r, "debiased_gap")
    _, mats["1120"] = per_image_matrix(G9, "1120", "gap")
    boot = bootstrap_predictions(mats, x_ratio, widths)

    def qtiles(a):
        q = np.percentile(a, [2.5, 16, 50, 84, 97.5], axis=0)
        return q

    # ---- paper main figure: X-only predictions with bootstrap bands +
    # full-gap parity across every route ----
    figm, (axa, axp) = plt.subplots(1, 2, figsize=(10.6, 3.8),
                                    gridspec_kw={"width_ratios": [1.35, 1]})
    F2160_pre = report["X"]["f768_predicted"]
    main_panels = [
        ("768", np.array(report["X"]["held_out"]["768"]["predicted"]),
         boot["pred768"], F2160_pre,
         f"held-out, pre-reg.; RMSE {report['X']['held_out']['768']['rmse']:.3f}"),
        ("896", pred896_lo, boot["pred896"], F3012_lo,
         f"leave-896-out, post-hoc; RMSE {loo896['rmse']:.3f}"),
    ]
    disp = {"768": "1024→768", "896": "1024→896", "1024": "1280→1024"}
    acolor = {"768": "C0", "896": "C1"}
    for r, center, draws, floor_val, tag in main_panels:
        sig, y, sem = curves[r]
        col = acolor[r]
        q = qtiles(draws)
        axa.fill_between(sig, q[0], q[4], color=col, alpha=0.10, lw=0)
        axa.fill_between(sig, q[1], q[3], color=col, alpha=0.22, lw=0)
        axa.plot(sig, center, "-", color=col, label=f"{disp[r]} predicted ({tag})")
        axa.errorbar(sig, y, yerr=sem, fmt="o", ms=4, color=col, capsize=2,
                     label=f"{disp[r]} measured")
        axa.axhline(floor_val, color=col, lw=1.0, ls="--", alpha=0.6)
        axa.annotate(f"$F({TOKENS[r]})$", (1.005, floor_val),
                     xycoords=("axes fraction", "data"), color=col,
                     fontsize=7, va="center")
    axa.axhline(0, color="0.8", lw=0.8)
    axa.set_xlabel(r"$\sigma$")
    axa.set_ylabel(r"paired gap $\bar\Delta$")
    axa.set_title("Predicted from governors alone (bands: 68%/95% bootstrap)",
                  fontsize=10)
    axa.legend(fontsize=6.5, loc="upper right")

    lims = [-0.08, 0.7]
    axp.plot(lims, lims, color="0.75", lw=0.8, zorder=0)
    for r in FIT_ROUTES:
        _, y, sem = curves[r]
        axp.errorbar(insample_pred(forms["X"], r), y, yerr=sem, fmt="o", ms=4,
                     mfc="white", color="0.45", capsize=0, lw=0.7, zorder=2)
    pcolor = {"768": "C0", "1024": "C2"}
    for r in HELD_OUT:
        _, y, sem = curves[r]
        axp.errorbar(report["X"]["held_out"][r]["predicted"], y, yerr=sem,
                     fmt="o", ms=5, color=pcolor[r], capsize=0, lw=0.9,
                     zorder=3, label=f"{disp[r]} (held-out)")
    axp.errorbar([], [], fmt="o", ms=4, mfc="white", color="0.45",
                 label="fit routes (in-sample)")
    axp.set_xlim(lims)
    axp.set_ylim(lims)
    axp.set_xlabel(r"predicted $\bar\Delta$")
    axp.set_ylabel(r"measured $\bar\Delta$")
    axp.set_title("All routes, one axis", fontsize=10)
    axp.set_aspect("equal")
    axp.legend(fontsize=6.5, loc="upper left")
    figm.tight_layout()
    figm.savefig(out_dir / "e5_main.png", dpi=180, bbox_inches="tight")

    envelope = dict(
        schema_version=1,
        script="project/sigma_lowres/paper_bench/experiments/e5/e5_refit.py",
        label="e5-refit",
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        sources=dict(e1b=str(tier1024), g9=str(G9)),
        caveats=[
            "same sources and caveats as e5_holdout.py (RAW 1280-tier, D=4; "
            "m extrapolated below sigma=0.125)",
            "X fit is per-route grid + profile-likelihood se on c",
        ],
        forms=report,
        verdict=verdict,
        loo896=loo896,
        bootstrap=dict(
            B=boot["B"], seed=boot["seed"], kept=boot["kept"],
            dropped=boot["dropped"], loo_floor_clamped=boot["loo_clamped"],
            floor_law_legs=boot["floor_law_legs"],
            F2160_quantiles=dict(zip(("p2_5", "p16", "p50", "p84", "p97_5"),
                                     qtiles(boot["f2160"]).tolist())),
            F3012_quantiles=dict(zip(("p2_5", "p16", "p50", "p84", "p97_5"),
                                     qtiles(boot["f3012"]).tolist())),
            held_fixed=["m_bar (G7, route-uniform shape)",
                        "bin-mean G(sigma) (run-level)"],
        ),
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
    print(f"LOO-896 (X): c={loo896['c']:.4f}  F(3012)={loo896['F3012']:+.4f}  "
          f"RMSE {loo896['rmse']:.3f}")
    print(f"bootstrap: kept {boot['kept']}/{boot['B']}  dropped {boot['dropped']}  "
          f"LOO-clamped {boot['loo_clamped']}  legs {boot['floor_law_legs']}")
    for nm, arr in (("F(2160)", boot["f2160"]), ("F(3012)", boot["f3012"])):
        q = qtiles(arr)
        print(f"{nm}: median {q[2]:+.3f}  68% [{q[1]:+.3f},{q[3]:+.3f}]  "
              f"95% [{q[0]:+.3f},{q[4]:+.3f}]")
    print("out:", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--tier1024_run",
        default=str(E1B),
        help="run dir supplying the 1024-tier debiased paired curves + gnorm "
        "(default: the published E1b run). Point at E13's segmented-grid run "
        "to refit against the resolved curve; the 1280-tier legs (G9) and "
        "m_bar (G7) are unaffected.",
    )
    ap.add_argument("--tag", default="", help="suffix on the output run dir")
    _a = ap.parse_args()
    main(Path(_a.tier1024_run), _a.tag)
