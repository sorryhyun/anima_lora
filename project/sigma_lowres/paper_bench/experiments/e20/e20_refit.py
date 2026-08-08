"""E20.1 — lane-matched refit of the cancellation-aware account (CPU).

Pre-registration: experiments/e20/README.md (frozen BEFORE this file was
written). The cancellation link

    d(sigma) = sat( sqrt( (A x)^2 + c^2 + 2 rho_bar (A x) c ) ),
    rho_bar = -0.910  (E14 gated ledger, licensed global by E19)

is run through the SAME fit -> governors -> held-out pipeline as the
additive lane (e5_refit / fig_accounts.build), with x(sigma) shared
bit-identically, so every difference is link-form only:

  fit routes 896/512/1120:  weighted two-stage (A, c) grid, weights
                            bin_widths/sem^2 (fit_sat_route protocol,
                            one extra grid axis; profile-likelihood se)
  governors:                A via two_point_governor on RATIO; c via the
                            exp-in-TOKENS family (exp_floor_law verbatim,
                            fit on c, sat(c) reported alongside)
  held-out:                 768 + 1024 from governors alone; 896 via the
                            leave-896-out lane (A from 1120, c from the
                            512+1120 two-point exp law = loo896_floor
                            applied to c); 512/1120 in-sample
  bootstrap:                full-pipeline B=1000, shared image resample;
                            canc bands for 768/896, additive bands for
                            1120/1024 (the 1280-tier guard envelope,
                            same recipe extended to that tier)
  ablations:                (a) rho=0 full re-run, (b) rho +/- 0.05
                            re-predict on frozen fits, (c) per-bin
                            residuals vs +/- eps*

E20.2 (free re-read): dip location (argmin d) and high-sigma crossing
(Ax = c) per route from the held-out parameters only, vs the E9 window
(0.56..0.81) and E9's 768 crossing estimate (0.688).

Verdict RMSEs are lane-matched against fig_accounts.build recomputed in
this process; the sigma=1 endpoint bin is detached (non-verdict) from
BOTH lanes symmetrically, with all-bin values reported alongside.

Usage:
    uv run python project/sigma_lowres/paper_bench/experiments/e20/e20_refit.py
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
PAPER_BENCH = HERE.parents[1]  # project/sigma_lowres/paper_bench
RUNS = PAPER_BENCH / "runs"
sys.path.insert(0, str(PAPER_BENCH))
sys.path.insert(0, str(PAPER_BENCH / "experiments" / "e5"))

import fig_accounts as fa  # noqa: E402  (bootstraps e5/e8 sys.path itself)
from e5_holdout import (  # noqa: E402
    E1B,
    FIT_ROUTES,
    G9,
    RATIO,
    TOKENS,
    bin_gnorm,
    bin_widths,
    interp_extrap,
    m_bar,
)
from e5_refit import (  # noqa: E402
    TIER_OF,
    exp_floor_law,
    fit_sat_vec,
    loo896_floor,
    per_image_matrix,
    sat,
    two_point_governor,
)

# ---- frozen constants (pre-registered in README.md; never tuned here) ----
RHO_BAR = -0.910
WINDOW = (0.56, 0.81)  # E9 dip/crossing window
E9_CROSS_768 = 0.688  # E9's 768 crossing estimate (context, not a gate)

C_MEAS, C_OURS, C_CANC = "0.15", "C0", "#2ca02c"
DISP = {
    "896": r"$1024\to896$",
    "768": r"$1024\to768$",
    "512": r"$1024\to512$",
    "1120": r"$1280\to1120$",
    "1024": r"$1280\to1024$",
}


def canc_pred(A, c, x, rho=RHO_BAR):
    ax = A * x
    return sat(np.sqrt(ax**2 + c**2 + 2.0 * rho * ax * c))


def fit_canc(x, y, w, rho=RHO_BAR):
    """Weighted 2D grid fit of (A, c), two-stage refinement.

    Mirrors fig_accounts_canc.fit_canc exactly (rho parameterized for the
    ablation); grids and refinement identical to the demo.
    """
    best = None
    A_med = 1.0 / max(float(np.median(x)), 1e-12)
    A_grid = np.geomspace(A_med * 1e-2, A_med * 1e2, 250)
    c_grid = np.geomspace(1e-3, 3.0, 250)
    for _stage in range(2):
        for A in A_grid:
            preds = canc_pred(A, c_grid[:, None], x[None, :], rho)
            sse = np.sum(w[None, :] * (y[None, :] - preds) ** 2, axis=1)
            i = int(np.argmin(sse))
            if best is None or sse[i] < best[0]:
                best = (float(sse[i]), float(A), float(c_grid[i]))
        _, A0, c0 = best
        A_grid = np.geomspace(A0 / 1.4, A0 * 1.4, 120)
        c_grid = np.geomspace(c0 / 1.4, c0 * 1.4, 120)
    return best  # (sse_min, A, c)


def profile_se(x, y, w, A_star, c_star, sse_min, rho=RHO_BAR):
    """Profile-likelihood 1-sigma on A and c (sse_min + 1 crossing),
    fit_sat_route's convention with the other parameter re-minimized."""
    c_grid = np.geomspace(c_star / 3, c_star * 3, 200)
    A_grid = np.geomspace(A_star / 3, A_star * 3, 200)

    def sse_min_over_c(A):
        preds = canc_pred(A, c_grid[:, None], x[None, :], rho)
        return float(np.min(np.sum(w[None, :] * (y[None, :] - preds) ** 2, axis=1)))

    def sse_min_over_A(c):
        preds = canc_pred(A_grid[:, None], c, x[None, :], rho)
        return float(np.min(np.sum(w[None, :] * (y[None, :] - preds) ** 2, axis=1)))

    ses = []
    for star, sweep in ((A_star, sse_min_over_c), (c_star, sse_min_over_A)):
        lo = hi = star
        for v in np.geomspace(star, star * 3, 200):
            if sweep(v) > sse_min + 1.0:
                hi = v
                break
        for v in np.geomspace(star, star / 3, 200):
            if sweep(v) > sse_min + 1.0:
                lo = v
                break
        ses.append(0.5 * (hi - lo))
    return ses[0], ses[1]


def fit_canc_vec(x, y, w, rho=RHO_BAR):
    """Fully vectorized two-stage (A, c) grid for the bootstrap."""
    A_med = 1.0 / max(float(np.median(x)), 1e-12)
    A_grid = np.geomspace(A_med * 1e-2, A_med * 1e2, 160)
    c_grid = np.geomspace(1e-3, 3.0, 160)
    A0 = c0 = None
    for _ in range(2):
        ax = np.outer(A_grid, x)[:, None, :]
        cc = c_grid[None, :, None]
        preds = sat(np.sqrt(ax**2 + cc**2 + 2.0 * rho * ax * cc))
        sse = np.sum(w * (y - preds) ** 2, axis=2)
        i, j = np.unravel_index(int(np.argmin(sse)), sse.shape)
        A0, c0 = float(A_grid[i]), float(c_grid[j])
        A_grid = np.geomspace(A0 / 1.4, A0 * 1.4, 160)
        c_grid = np.geomspace(c0 / 1.4, c0 * 1.4, 160)
    return A0, c0


def run_lane(curves, x, widths, rho):
    """The full fit -> governors -> held-out chain for one rho.

    Returns (fit, gov meta, per-route predictions, lane meta).
    """
    fit = {}
    for r in FIT_ROUTES:
        _, y, sem = curves[r]
        w = widths[r] / sem**2
        sse, A, c = fit_canc(x[r], y, w, rho)
        se_A, se_c = profile_se(x[r], y, w, A, c, sse, rho)
        fit[r] = dict(A=A, c=c, se_A=se_A, se_c=se_c, sat_c=float(sat(c)))

    gov_A, z = two_point_governor(
        {r: fit[r]["A"] for r in FIT_ROUTES},
        {r: fit[r]["se_A"] for r in FIT_ROUTES},
    )
    c0, tau_c = exp_floor_law(
        [(TOKENS[r], fit[r]["c"], fit[r]["se_c"]) for r in FIT_ROUTES]
    )
    c_law = lambda n: c0 * math.exp(-n / tau_c)  # noqa: E731
    c_loo = loo896_floor(fit["512"]["c"], fit["1120"]["c"])

    preds, meta = {}, {}
    for r in ("768", "1024"):
        A, _ = gov_A(RATIO[r])
        c = c_law(TOKENS[r])
        preds[r] = canc_pred(A, c, x[r], rho)
        meta[r] = dict(A=A, c=c, sat_c=float(sat(c)), lane="held-out (governors)")
    if c_loo is not None:
        preds["896"] = canc_pred(fit["1120"]["A"], c_loo, x["896"], rho)
        meta["896"] = dict(
            A=fit["1120"]["A"],
            c=c_loo,
            sat_c=float(sat(c_loo)),
            lane="LOO-896 (governors from 512+1120)",
        )
    for r in FIT_ROUTES:
        preds[f"{r}_ins"] = canc_pred(fit[r]["A"], fit[r]["c"], x[r], rho)
    gov = dict(
        ratio_twin_z=z,
        c0=c0,
        tau_c_tokens=tau_c,
        c_768=c_law(TOKENS["768"]),
        c_1024=c_law(TOKENS["1024"]),
        c_loo896=c_loo,
    )
    return fit, gov, preds, meta


def rmse_pair(curves, r, pred):
    """(all-bin, endpoint-detached) RMSE. Detached = verdict convention."""
    sig, y, _ = curves[r]
    pred = np.asarray(pred)
    full = float(np.sqrt(np.mean((y - pred) ** 2)))
    has_end = math.isclose(float(sig[-1]), 1.0, abs_tol=1e-9)
    n = len(sig) - 1 if has_end else len(sig)
    det = float(np.sqrt(np.mean((y[:n] - pred[:n]) ** 2)))
    return full, det


def dip_cross(curves, x, r, A, c, rho=RHO_BAR):
    """Dense-grid dip (argmin d) and Ax = c crossings for one route."""
    sig = curves[r][0]
    s = np.linspace(float(sig[0]), float(sig[-1]), 2001)
    xd = np.interp(s, sig, x[r])
    dcurve = canc_pred(A, c, xd, rho)
    dip = float(s[int(np.argmin(dcurve))])
    f = A * xd - c
    idx = np.nonzero(np.diff(np.sign(f)) != 0)[0]
    crossings = [
        float(s[i] + (f[i] / (f[i] - f[i + 1])) * (s[i + 1] - s[i])) for i in idx
    ]
    return dip, crossings


def bootstrap_lanes(mats, x, widths, B=1000, seed=0):
    """Shared-resample full-pipeline bootstrap of BOTH lanes.

    Per draw (same image indices for both lanes): re-bin, refit the three
    fit routes, re-derive governors, re-predict. Canc lane emits 768 +
    LOO-896 bands (the pre-registered bands); additive lane emits 1120 +
    1024 predictions (the 1280-tier guard envelope, e5_refit's recipe
    extended to that tier). Lane validity is tracked independently.
    """
    rng = np.random.default_rng(seed)
    n_e1b = mats["896"].shape[0]
    n_g9 = mats["1120"].shape[0]
    out = dict(canc768=[], canc896=[], add1120=[], add1024=[])
    drops = dict(stats=0, canc=0, add=0)
    t768 = (RATIO["768"] - 0.5) / (0.875 - 0.5)
    t1024 = (RATIO["1024"] - 0.5) / (0.875 - 0.5)
    for _ in range(B):
        ie = rng.integers(0, n_e1b, n_e1b)
        ig = rng.integers(0, n_g9, n_g9)
        yb, wb = {}, {}
        ok = True
        for r in FIT_ROUTES:
            sub = mats[r][ie if TIER_OF[r] == "1024tier" else ig]
            n = np.sum(np.isfinite(sub), axis=0)
            y = np.nanmean(sub, axis=0)
            sem = np.nanstd(sub, axis=0, ddof=1) / np.sqrt(n)
            if not np.all(np.isfinite(sem)) or np.any(sem <= 0):
                ok = False
                break
            yb[r], wb[r] = y, widths[r] / sem**2
        if not ok:
            drops["stats"] += 1
            continue

        # ---- canc lane ----
        fc = {r: fit_canc_vec(x[r], yb[r], wb[r]) for r in FIT_ROUTES}
        logpts = [
            (TOKENS[r], math.log(fc[r][1])) for r in FIT_ROUTES if fc[r][1] > 0.005
        ]
        c_loo = loo896_floor(fc["512"][1], fc["1120"][1])
        if len(logpts) < 2 or logpts[0][1] == logpts[-1][1] or c_loo is None:
            drops["canc"] += 1
        else:
            (n1, l1), (n2, l2) = logpts[0], logpts[-1]
            tau = (n1 - n2) / (l2 - l1)
            c0 = math.exp(l1 + n1 / tau)
            A768 = (1 - t768) * fc["512"][0] + t768 * 0.5 * (
                fc["896"][0] + fc["1120"][0]
            )
            out["canc768"].append(
                canc_pred(A768, c0 * math.exp(-TOKENS["768"] / tau), x["768"])
            )
            out["canc896"].append(canc_pred(fc["1120"][0], c_loo, x["896"]))

        # ---- additive lane (guard tier) ----
        fx = {r: fit_sat_vec(x[r], yb[r], wb[r]) for r in FIT_ROUTES}
        logpts = [
            (TOKENS[r], math.log(fx[r][1])) for r in FIT_ROUTES if fx[r][1] > 0.005
        ]
        if len(logpts) < 2 or logpts[0][1] == logpts[-1][1]:
            drops["add"] += 1
            continue
        (n1, l1), (n2, l2) = logpts[0], logpts[-1]
        tau = (n1 - n2) / (l2 - l1)
        if tau <= 0:
            drops["add"] += 1
            continue
        F0 = math.exp(l1 + n1 / tau)
        A1024 = (1 - t1024) * fx["512"][0] + t1024 * 0.5 * (
            fx["896"][0] + fx["1120"][0]
        )
        out["add1120"].append(sat(fx["1120"][0] * x["1120"]) + fx["1120"][1])
        out["add1024"].append(
            sat(A1024 * x["1024"]) + F0 * math.exp(-TOKENS["1024"] / tau)
        )
    return {k: np.array(v) for k, v in out.items()}, drops


def qtiles(a):
    return np.percentile(a, [2.5, 16, 50, 84, 97.5], axis=0)


def main(tier1024: Path = E1B, tag: str = "") -> None:
    # ---- shared data, bit-identical to the additive lane ----
    d = fa.build(tier1024)  # recomputes the additive account in-process
    curves = d["curves"]
    eps_star = d["eps_star"]
    gnorm = {"1024tier": bin_gnorm(tier1024), "1280tier": bin_gnorm(G9)}
    src = {"1024tier": tier1024, "1280tier": G9}
    widths = {r: bin_widths(src[TIER_OF[r]], len(curves[r][0])) for r in curves}
    msig, mvals = m_bar()
    x = {
        r: interp_extrap(curves[r][0], msig, mvals) / gnorm[TIER_OF[r]] for r in curves
    }

    # ---- 20.1 main chain ----
    fit, gov, preds, meta = run_lane(curves, x, widths, RHO_BAR)

    rmse = {}
    for r in ("768", "896", "1024"):
        if r not in preds:
            continue
        cf, cd = rmse_pair(curves, r, preds[r])
        af, ad = rmse_pair(curves, r, d["ours"][r])
        rmse[r] = dict(canc_full=cf, canc_det=cd, additive_full=af, additive_det=ad)
    for r in FIT_ROUTES:
        cf, cd = rmse_pair(curves, r, preds[f"{r}_ins"])
        af, ad = rmse_pair(curves, r, d["ours"][r])
        rmse[f"{r}_ins"] = dict(
            canc_full=cf, canc_det=cd, additive_full=af, additive_det=ad
        )

    # 512 diagnostic (pre-registered: not verdict-bearing) — mid-window
    # in-sample residual sign/size = the amplitude law's fingerprint
    sig5, y5, _ = curves["512"]
    m5 = (sig5 >= WINDOW[0]) & (sig5 <= WINDOW[1])
    diag512 = dict(
        mean_resid_window=float(np.mean(y5[m5] - preds["512_ins"][m5])),
        max_abs_resid_window=float(np.max(np.abs(y5[m5] - preds["512_ins"][m5]))),
    )

    # ---- 20.2: dips + crossings from held-out/LOO parameters only ----
    e202 = {}
    for r in ("768", "896", "1024"):
        if r not in meta:
            continue
        dip, crossings = dip_cross(curves, x, r, meta[r]["A"], meta[r]["c"])
        e202[r] = dict(
            dip=dip,
            dip_in_window=bool(WINDOW[0] <= dip <= WINDOW[1]),
            crossings=crossings,
            crossing_high=crossings[-1] if crossings else None,
            lane=meta[r]["lane"],
        )

    # ---- bootstrap (shared resample, both lanes) ----
    mats = {r: per_image_matrix(tier1024, r, "debiased_gap")[1] for r in ("896", "512")}
    mats["1120"] = per_image_matrix(G9, "1120", "gap")[1]
    boot, drops = bootstrap_lanes(mats, x, widths)

    # ---- 1280-tier guard: canc central inside the additive 95% envelope ----
    guard = {}
    for r, canc_curve in (("1120", preds["1120_ins"]), ("1024", preds["1024"])):
        q = qtiles(boot[f"add{r}"])
        sig = curves[r][0]
        has_end = math.isclose(float(sig[-1]), 1.0, abs_tol=1e-9)
        n = len(sig) - 1 if has_end else len(sig)
        inside = (canc_curve[:n] >= q[0][:n]) & (canc_curve[:n] <= q[4][:n])
        exc = np.maximum(q[0][:n] - canc_curve[:n], canc_curve[:n] - q[4][:n])
        guard[r] = dict(
            bins=n,
            outside=int(np.sum(~inside)),
            max_excursion=float(np.max(exc)),
            passed=bool(np.all(inside)),
        )
    guard_pass = all(guard[r]["passed"] for r in guard)

    # ---- ablations ----
    # (a) rho = 0 — full re-run of the chain
    fit0, gov0, preds0, meta0 = run_lane(curves, x, widths, 0.0)
    abl_rho0 = {}
    for r in ("768", "896", "1024"):
        if r in preds0:
            f, dt = rmse_pair(curves, r, preds0[r])
            abl_rho0[r] = dict(rmse_full=f, rmse_det=dt)
    # (b) rho +/- 0.05 — frozen fits + governors, re-predict only
    abl_sens = {}
    for rho in (RHO_BAR - 0.05, RHO_BAR + 0.05):
        entry = {}
        for r in ("768", "896", "1024"):
            if r not in meta:
                continue
            p = canc_pred(meta[r]["A"], meta[r]["c"], x[r], rho)
            f, dt = rmse_pair(curves, r, p)
            entry[r] = dict(rmse_full=f, rmse_det=dt)
        abl_sens[f"{rho:+.3f}"] = entry
    # (c) per-bin residuals vs +/- eps* (1024-tier grid only)
    abl_eps = {}
    for r in ("768", "896"):
        if r not in preds:
            continue
        sig, y, _ = curves[r]
        resid = np.asarray(y) - np.asarray(preds[r])
        abl_eps[r] = dict(
            resid=[float(v) for v in resid],
            eps=[float(v) for v in eps_star],
            n_exceed=int(np.sum(np.abs(resid) > eps_star)),
        )

    # ---- pre-registered verdict ----
    win768 = rmse["768"]["canc_det"] <= rmse["768"]["additive_det"]
    win896 = "896" in rmse and rmse["896"]["canc_det"] <= rmse["896"]["additive_det"]
    dips_ok = (
        "768" in e202
        and e202["768"]["dip_in_window"]
        and "896" in e202
        and e202["896"]["dip_in_window"]
    )
    if win768 and win896 and dips_ok and guard_pass:
        verdict = "WIN"
    elif not win768 and not win896:
        verdict = "KILL"
    else:
        verdict = "PARTIAL"
    verdict_detail = dict(
        verdict=verdict,
        win_768=bool(win768),
        win_896=bool(win896),
        dips_in_window=bool(dips_ok),
        guard_1280_pass=bool(guard_pass),
        note=(
            "WIN requires both safe-lane RMSE wins AND both dips in-window "
            "AND the 1280-tier guard; RMSEs are endpoint-detached, both "
            "lanes symmetrically (all-bin values reported alongside)"
        ),
    )

    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M")
    out_dir = RUNS / f"{stamp}-e20-refit{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- figure: lane-matched accounts_canc ----
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.8))
    panel_pred = {"896": preds.get("896"), "768": preds["768"], "512": preds["512_ins"]}
    for ax, r in zip(axes.ravel()[:3], ("896", "768", "512")):
        sig, y, sem = curves[r]
        ax.fill_between(
            sig,
            -eps_star,
            eps_star,
            color="0.5",
            alpha=0.35,
            lw=0,
            zorder=1,
            label=r"$\pm\varepsilon^{*}$",
        )
        key = {"896": "canc896", "768": "canc768"}.get(r)
        if key and len(boot[key]):
            q = qtiles(boot[key])
            ax.fill_between(sig, q[0], q[4], color=C_CANC, alpha=0.15, lw=0)
        ax.plot(
            sig,
            d["ours"][r],
            "-",
            color=C_OURS,
            lw=1.6,
            alpha=0.75,
            zorder=2,
            label="additive (Fig. 1 lanes)",
        )
        if panel_pred[r] is not None:
            ax.plot(
                sig,
                panel_pred[r],
                "-",
                color=C_CANC,
                lw=2.0,
                zorder=3,
                label=rf"canc, lane-matched ($\bar\rho={RHO_BAR}$)",
            )
        mkey = r if r in meta else None
        sat_c = meta[r]["sat_c"] if mkey else fit["512"]["sat_c"]
        ax.axhline(
            sat_c,
            color=C_CANC,
            lw=0.8,
            ls=":",
            alpha=0.6,
            zorder=1,
            label=r"canc floor analogue sat($c$)",
        )
        if r in e202 and e202[r]["crossing_high"] is not None:
            ax.axvline(
                e202[r]["crossing_high"], color=C_CANC, lw=0.8, ls=":", alpha=0.5
            )
        has_end = math.isclose(float(sig[-1]), 1.0, abs_tol=1e-9)
        n_in = len(sig) - 1 if has_end else len(sig)
        ax.errorbar(
            sig[:n_in],
            y[:n_in],
            yerr=sem[:n_in],
            fmt="o",
            ms=4.0,
            color=C_MEAS,
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
                color=C_MEAS,
                capsize=2,
                lw=1.0,
                zorder=4,
            )
        rk = r if r in rmse else f"{r}_ins"
        lane_tag = meta[r]["lane"] if r in meta else "in-sample"
        box = (
            f"{lane_tag}\n"
            f"RMSE* canc {rmse[rk]['canc_det']:.4f}\n"
            f"RMSE* addv {rmse[rk]['additive_det']:.4f}"
            + (f"\ndip $\\sigma\\approx${e202[r]['dip']:.3f}" if r in e202 else "")
        )
        ax.annotate(
            box,
            (0.97, 0.96),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=7.5,
            color="0.25",
            bbox=dict(fc="white", ec="0.8", alpha=0.85),
        )
        ax.set_title(DISP[r], fontsize=10.5)
        ax.axhline(0, color="0.8", lw=0.6, zorder=0)
        ax.set_xlabel(r"$\sigma$", fontsize=9)
        ax.set_ylabel("debiased gap", fontsize=9)
        ax.tick_params(labelsize=8)
        if r == "896":
            leg_h, leg_l = ax.get_legend_handles_labels()

    # 1280-tier guard panel
    ax = axes.ravel()[3]
    for r, canc_curve, col in (
        ("1120", preds["1120_ins"], "C1"),
        ("1024", preds["1024"], "C2"),
    ):
        sig, y, sem = curves[r]
        q = qtiles(boot[f"add{r}"])
        ax.fill_between(
            sig,
            q[0],
            q[4],
            color=col,
            alpha=0.15,
            lw=0,
            label=f"{DISP[r]} additive 95% env",
        )
        ax.plot(sig, canc_curve, "-", color=col, lw=1.8, label=f"{DISP[r]} canc")
        ax.errorbar(
            sig,
            y,
            yerr=sem,
            fmt="o",
            ms=3.5,
            color=col,
            mfc="white",
            capsize=2,
            lw=0.8,
            alpha=0.8,
        )
    ax.axhline(0, color="0.8", lw=0.6, zorder=0)
    ax.set_title("1280-tier guard (canc inside additive envelope)", fontsize=10)
    ax.set_xlabel(r"$\sigma$", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6.5, loc="upper left")
    ax.annotate(
        f"guard: 1120 {'PASS' if guard['1120']['passed'] else 'FAIL'} "
        f"({guard['1120']['outside']}/{guard['1120']['bins']} out)\n"
        f"guard: 1024 {'PASS' if guard['1024']['passed'] else 'FAIL'} "
        f"({guard['1024']['outside']}/{guard['1024']['bins']} out)",
        (0.97, 0.03),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="0.25",
        bbox=dict(fc="white", ec="0.8", alpha=0.85),
    )
    fig.legend(
        leg_h,
        leg_l,
        ncol=3,
        fontsize=8.5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
    )
    fig.suptitle(
        f"E20.1 — cancellation account under the paper's lanes (verdict: {verdict})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    fig.savefig(out_dir / "e20_main.png", dpi=180, bbox_inches="tight")

    # ---- envelope ----
    envelope = dict(
        schema_version=1,
        script="project/sigma_lowres/paper_bench/experiments/e20/e20_refit.py",
        label="e20-refit",
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        sources=dict(e1b=str(tier1024), g9=str(G9)),
        frozen=dict(rho_bar=RHO_BAR, window=list(WINDOW), e9_cross_768=E9_CROSS_768),
        caveats=[
            "same sources and caveats as e5_refit (RAW 1280-tier, D=4; "
            "m extrapolated below sigma=0.125)",
            "x(sigma) shared bit-identically with the additive lane",
            "verdict RMSEs endpoint-detached from BOTH lanes symmetrically; "
            "all-bin values reported alongside (fa.build convention)",
            "c governor = exp_floor_law verbatim on c (kappa-scale), sat(c) "
            "reported alongside; with all three c > 0.005 the two anchor "
            "legs are (896, 1120), unlike the additive floor law whose "
            "1120 floor ~0 fell to its filter",
        ],
        fit={r: fit[r] for r in FIT_ROUTES},
        governors=gov,
        held_out_meta=meta,
        rmse=rmse,
        diag_512=diag512,
        e20_2=e202,
        guard_1280=guard,
        ablations=dict(
            rho0=dict(fit={r: fit0[r] for r in FIT_ROUTES}, rmse=abl_rho0, gov=gov0),
            rho_sens=abl_sens,
            eps_residuals=abl_eps,
        ),
        bootstrap=dict(
            B=1000,
            seed=0,
            drops=drops,
            kept_canc=int(len(boot["canc768"])),
            kept_add=int(len(boot["add1120"])),
            canc768_quantiles=dict(
                zip(
                    ("p2_5", "p16", "p50", "p84", "p97_5"),
                    qtiles(boot["canc768"]).tolist(),
                )
            ),
            canc896_quantiles=dict(
                zip(
                    ("p2_5", "p16", "p50", "p84", "p97_5"),
                    qtiles(boot["canc896"]).tolist(),
                )
            ),
        ),
        additive_reference=dict(
            rmse={r: d["rmse"][r] for r in d["rmse"]},
            floors=d["floors"],
            F0=d["F0"],
            tau=d["tau"],
        ),
        verdict=verdict_detail,
    )
    (out_dir / "result.json").write_text(json.dumps(envelope, indent=1))

    # ---- prints ----
    print(f"rho_bar = {RHO_BAR} (frozen)")
    print("fit routes (A, c | profile se):")
    for r in FIT_ROUTES:
        f = fit[r]
        print(
            f"  {r:>5}: A={f['A']:.3f}±{f['se_A']:.3f}  c={f['c']:.4f}"
            f"±{f['se_c']:.4f}  sat(c)={f['sat_c']:.4f}"
        )
    print(
        f"governors: ratio-twin z={gov['ratio_twin_z']:.2f}  "
        f"c-law c0={gov['c0']:.4g} tau={gov['tau_c_tokens']:.0f} tok  "
        f"c(2160)={gov['c_768']:.4f}  c(4116)={gov['c_1024']:.4f}  "
        f"c_loo(3012)={gov['c_loo896'] if gov['c_loo896'] is None else round(gov['c_loo896'], 4)}"
    )
    print(f"{'route':<10}{'canc*':>8}{'addv*':>8}{'canc-all':>10}{'addv-all':>10}")
    for r in ("768", "896", "1024", "896_ins", "512_ins", "1120_ins"):
        if r not in rmse:
            continue
        e = rmse[r]
        print(
            f"{r:<10}{e['canc_det']:>8.4f}{e['additive_det']:>8.4f}"
            f"{e['canc_full']:>10.4f}{e['additive_full']:>10.4f}"
        )
    print("E20.2 (held-out params only):")
    for r, e in e202.items():
        cr = f"{e['crossing_high']:.3f}" if e["crossing_high"] is not None else "--"
        print(
            f"  {r}: dip={e['dip']:.3f} "
            f"({'IN' if e['dip_in_window'] else 'OUT of'} window "
            f"{WINDOW[0]}..{WINDOW[1]})  crossing_high={cr}  [{e['lane']}]"
        )
    print(
        f"512 diagnostic: mean mid-window resid "
        f"{diag512['mean_resid_window']:+.4f} "
        f"(max |r| {diag512['max_abs_resid_window']:.4f})"
    )
    print(
        "guard 1280: "
        + "  ".join(
            f"{r} {'PASS' if guard[r]['passed'] else 'FAIL'}"
            f"({guard[r]['outside']}/{guard[r]['bins']} out, "
            f"max exc {guard[r]['max_excursion']:+.4f})"
            for r in guard
        )
    )
    print(
        "ablation rho=0: "
        + "  ".join(f"{r} RMSE*={abl_rho0[r]['rmse_det']:.4f}" for r in abl_rho0)
    )
    for k, entry in abl_sens.items():
        print(
            f"ablation rho={k}: "
            + "  ".join(f"{r} RMSE*={entry[r]['rmse_det']:.4f}" for r in entry)
        )
    for r, e in abl_eps.items():
        print(
            f"eps residuals {r}: {e['n_exceed']}/{len(e['resid'])} bins exceed +/-eps*"
        )
    print(
        f"bootstrap: kept canc {len(boot['canc768'])}, add "
        f"{len(boot['add1120'])} of B=1000; drops {drops}"
    )
    print(f"VERDICT: {verdict}  {verdict_detail}")
    print("out:", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--tier1024_run",
        default=str(E1B),
        help="run dir supplying the 1024-tier curves (default: "
        "the published E1b verdict run)",
    )
    ap.add_argument("--tag", default="", help="suffix on the output run dir")
    _a = ap.parse_args()
    main(Path(_a.tier1024_run), _a.tag)
