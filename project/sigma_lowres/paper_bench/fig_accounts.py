"""Merged head-to-head figure: both accounts on the same axes.

Replaces the paper's two separate overlay figures (e83_overlay.png = the
spectral account against the measurement; e5_main.png = ours against the
same measurement) with ONE figure in which, per route, the measured
debiased curve carries both predictions. Same x-axis, same points, same
units -- "spectral view" and "ours" read off one panel.

No new instrument and no new fit: every number is recomputed by calling
the published helpers of experiments/e8/e83_bridge.py (spectral bridge)
and experiments/e5/e5_refit.py (our X-form fit -> governors -> held-out
prediction, plus the full-pipeline bootstrap), so this script cannot
drift from the two run envelopes it mirrors.

Panels (2x2):
  1024->896 / 1024->768 / 1024->512   measured points (sigma=1 endpoint
                                      bin detached as an open marker),
                                      spectral curve (delta_reenc-
                                      anchored; the family is delta-inert
                                      at the curve level) and ours (exact
                                      angular link) with 95% bootstrap
                                      band + floor line, read against the
                                      bin-level +/-eps* certification
                                      band (1.645 x route-median SEM,
                                      same recipe as plot_debiased_map)
  parity                              predicted vs measured, all bins;
                                      ours also on the two 1280-tier
                                      routes the spectral bridge does
                                      not cover

Per-panel status labels mirror Table "headtohead" exactly: ours is
held-out (pre-registered) on 768, leave-896-out (post-hoc) on 896, and
in-sample on 512; the spectral gain is calibrated on 896 and frozen.

Usage:
    python project/sigma_lowres/paper_bench/fig_accounts.py
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent  # project/sigma_lowres/paper_bench
SIGMA = HERE.parent  # project/sigma_lowres
RUNS = HERE / "runs"
sys.path.insert(0, str(HERE / "experiments" / "e5"))
sys.path.insert(0, str(HERE / "experiments" / "e8"))

from e5_holdout import (  # noqa: E402
    E1B,
    FIT_ROUTES,
    G9,
    HELD_OUT,
    RATIO,
    TOKENS,
    bin_gnorm,
    bin_widths,
    interp_extrap,
    m_bar,
    paired_stats,
)
from e5_refit import (  # noqa: E402
    TIER_OF,
    bootstrap_predictions,
    exp_floor_law,
    fit_sat_route,
    loo896_floor,
    per_image_matrix,
    sat,
    two_point_governor,
)
from e83_bridge import DELTA_REENC, FCUT, ROUTES, m_spec, t_star  # noqa: E402

REENC_FLOOR = SIGMA / "bench" / "results" / "20260730-0940-reenc-floor"
FIG_DIRS = (SIGMA / "paper_suggestion" / "figs", SIGMA / "paper" / "figs")

DISP = {"896": r"$1024\to896$", "768": r"$1024\to768$",
        "512": r"$1024\to512$", "1120": r"$1280\to1120$",
        "1024": r"$1280\to1024$"}
C_MEAS, C_SPEC, C_OURS = "0.15", "C3", "C0"


def build(tier1024: Path = E1B) -> dict:
    """Recompute both accounts on the same measured curves."""
    curves = {r: paired_stats(tier1024, r, "debiased_gap") for r in ROUTES}
    # bin-level certification resolution, identical to plot_debiased_map.py:
    # 1.645 x per-bin median SEM over the 1024-tier routes
    eps_star = 1.645 * np.median(
        np.stack([curves[r][2] for r in ROUTES]), axis=0)
    for r in ("1120", "1024"):
        curves[r] = paired_stats(G9, r, "gap")
    gnorm = {"1024tier": bin_gnorm(tier1024), "1280tier": bin_gnorm(G9)}
    src = {"1024tier": tier1024, "1280tier": G9}
    widths = {r: bin_widths(src[TIER_OF[r]], len(curves[r][0]))
              for r in curves}
    msig, mvals = m_bar()

    def x_ratio(route):
        sig, _, _ = curves[route]
        return interp_extrap(sig, msig, mvals) / gnorm[TIER_OF[route]]

    # ---- our account: X form, per-route fit -> governors -> held-out ----
    fitX = {}
    for route in FIT_ROUTES:
        _, y, sem = curves[route]
        c, F, se_c, se_F, _ = fit_sat_route(
            x_ratio(route), y, widths[route] / sem**2)
        fitX[route] = dict(A=c, F=F, se_A=se_c, se_F=se_F)
    gov_A, _ = two_point_governor(
        {r: fitX[r]["A"] for r in FIT_ROUTES},
        {r: fitX[r]["se_A"] for r in FIT_ROUTES},
    )
    F0, tau = exp_floor_law([(TOKENS[r], fitX[r]["F"], fitX[r]["se_F"])
                             for r in FIT_ROUTES])

    ours, floors = {}, {}
    for route in HELD_OUT:  # 768, 1024 -- governors only, zero route freedom
        A, _ = gov_A(RATIO[route])
        floors[route] = F0 * math.exp(-TOKENS[route] / tau)
        ours[route] = sat(A * x_ratio(route)) + floors[route]
    floors["512"] = fitX["512"]["F"]
    ours["512"] = sat(fitX["512"]["A"] * x_ratio("512")) + floors["512"]
    floors["1120"] = fitX["1120"]["F"]
    ours["1120"] = sat(fitX["1120"]["A"] * x_ratio("1120")) + floors["1120"]
    # 896 under the post-hoc leave-896-out lane (governors from 512+1120)
    floors["896"] = loo896_floor(fitX["512"]["F"], fitX["1120"]["F"])
    ours["896"] = sat(fitX["1120"]["A"] * x_ratio("896")) + floors["896"]

    mats = {r: per_image_matrix(tier1024, r, "debiased_gap")[1]
            for r in ("896", "512")}
    mats["1120"] = per_image_matrix(G9, "1120", "gap")[1]
    boot = bootstrap_predictions(mats, x_ratio, widths)
    bands = {"768": boot["pred768"], "896": boot["pred896"]}

    # ---- spectral account: destroyed-band bridge, one gain on 896 ----
    rap = json.load((REENC_FLOOR / "curves.json").open())
    freq, P = np.array(rap["freq"]), np.array(rap["P_mean"])
    sigma = curves["896"][0]
    P_cut = {r: float(np.interp(FCUT[r], freq, P)) for r in ROUTES}
    xs = {r: (m_spec(sigma, freq, P, FCUT[r]) / gnorm["1024tier"]) ** 2
          for r in ROUTES}
    _, y896, sem896 = curves["896"]
    w = widths["896"] / sem896**2
    A_spec = float(np.sum(w * xs["896"] * y896) / np.sum(w * xs["896"] ** 2))
    spec, gates = {}, {}
    for r in ROUTES:
        gates[r] = float(t_star(DELTA_REENC, P_cut[r]))
        spec[r] = np.where(sigma < gates[r], A_spec * xs[r], 0.0)

    rmse = {}
    for r in ROUTES:
        _, y, _ = curves[r]
        rmse[r] = dict(
            spectral=float(np.sqrt(np.mean((y - spec[r]) ** 2))),
            ours=float(np.sqrt(np.mean((y - ours[r]) ** 2))),
        )
    for r in ("1120", "1024"):
        _, y, _ = curves[r]
        rmse[r] = dict(spectral=None,
                       ours=float(np.sqrt(np.mean((y - ours[r]) ** 2))))
    return dict(curves=curves, ours=ours, spec=spec, floors=floors,
                bands=bands, gates=gates, rmse=rmse, A_spec=A_spec,
                boot=boot, F0=F0, tau=tau, fitX=fitX, eps_star=eps_star)


def draw(d: dict):
    curves, ours, spec = d["curves"], d["ours"], d["spec"]
    status = {  # (ours label, spectral label)
        "896": ("leave-896-out, post-hoc", "gain calibrated here"),
        "768": ("held out, pre-registered", "gain from 896"),
        "512": ("in-sample fit route", "gain from 896"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.8))
    handles = None
    for ax, r in zip(axes.ravel()[:3], ROUTES):
        sig, y, sem = curves[r]
        ax.fill_between(sig, -d["eps_star"], d["eps_star"], color="0.5",
                        alpha=0.35, lw=0, zorder=1,
                        label=r"$\pm\varepsilon^{*}$ (bin-level)")
        if r in d["bands"]:
            q = np.percentile(d["bands"][r], [2.5, 97.5], axis=0)
            ax.fill_between(sig, q[0], q[1], color=C_OURS, alpha=0.15, lw=0)
        ax.plot(sig, spec[r], "--", color=C_SPEC, lw=1.8, zorder=2,
                label="spectral account")
        ax.plot(sig, ours[r], "-", color=C_OURS, lw=1.8, zorder=2,
                label="ours (exact angular link)")
        ax.axhline(d["floors"][r], color=C_OURS, lw=0.9, ls="--", alpha=0.6,
                   zorder=1, label=r"our predicted floor")
        # sigma=1 is a different probe mode (exact endpoint, not a stratified
        # window draw) -- detached as an open marker, as in the verdict map
        has_end = math.isclose(float(sig[-1]), 1.0, abs_tol=1e-9)
        n_in = len(sig) - 1 if has_end else len(sig)
        ax.errorbar(sig[:n_in], y[:n_in], yerr=sem[:n_in], fmt="o", ms=4.5,
                    color=C_MEAS, capsize=2, lw=1.0, zorder=4,
                    label="measured (debiased)")
        if has_end:
            ax.errorbar(sig[-1:], y[-1:], yerr=sem[-1:], fmt="o", ms=5,
                        color=C_MEAS, mfc="white", mew=1.2, capsize=2,
                        lw=1.0, zorder=4,
                        label=r"$\sigma{=}1$ endpoint mode")
        ax.axhline(0, color="0.8", lw=0.8, zorder=0)
        ax.set_title(DISP[r], fontsize=11)
        ax.set_xlabel(r"$\sigma$")
        ax.set_ylabel(r"paired gap $\overline{\mathrm{gap}}$")
        lo, hi = ax.get_ylim()  # headroom for the RMSE box
        ax.set_ylim(lo, hi + 0.22 * (hi - lo))
        so, ss = status[r]
        ax.annotate(
            f"RMSE  spectral {d['rmse'][r]['spectral']:.3f}   ({ss})\n"
            f"{'':10s}ours {d['rmse'][r]['ours']:.3f}   ({so})",
            (0.5, 0.97), xycoords="axes fraction", ha="center", va="top",
            fontsize=7.0, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8", lw=0.6),
        )
        if r == "896":
            h, lab = ax.get_legend_handles_labels()
            by_label = dict(zip(lab, h))
            want = ["measured (debiased)", r"$\sigma{=}1$ endpoint mode",
                    "spectral account", "ours (exact angular link)",
                    r"our predicted floor",
                    r"$\pm\varepsilon^{*}$ (bin-level)"]
            keep = [k for k in want if k in by_label]
            handles = ([by_label[k] for k in keep], keep)

    # ---- parity ----
    ax = axes.ravel()[3]
    lims = [-0.09, 0.72]
    ax.plot(lims, lims, color="0.75", lw=0.9, zorder=0)
    for r in ROUTES:
        _, y, sem = curves[r]
        ax.errorbar(spec[r], y, yerr=sem, fmt="s", ms=4, color=C_SPEC,
                    capsize=0, lw=0.7, alpha=0.85, zorder=2,
                    label="spectral account" if r == "896" else None)
    for r in ROUTES + ("1120", "1024"):
        _, y, sem = curves[r]
        held = r in HELD_OUT
        ax.errorbar(ours[r], y, yerr=sem, fmt="o", ms=4.5 if held else 4,
                    color=C_OURS, mfc=C_OURS if held else "white",
                    capsize=0, lw=0.7, zorder=3,
                    label=("ours (held-out routes)" if r == "768" else
                           "ours (fit routes)" if r == "896" else None))
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel(r"predicted $\overline{\mathrm{gap}}$")
    ax.set_ylabel(r"measured $\overline{\mathrm{gap}}$")
    ax.set_title("All bins, one axis", fontsize=11)
    ax.legend(fontsize=7.0, loc="lower right", framealpha=0.9)
    fig.legend(*handles, ncol=3, fontsize=8.5, loc="lower center",
               bbox_to_anchor=(0.5, -0.005), frameon=False)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    return fig


def main(tier1024: Path = E1B, tag: str = "") -> None:
    d = build(tier1024)
    fig = draw(d)
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M")
    out_dir = RUNS / f"{stamp}-fig-accounts{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "accounts_headtohead.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    # the paper figure tracks the published verdict run only: an alternate
    # --tier1024_run render stays in its run dir and never touches figs/
    if tier1024 == E1B:
        for fd in FIG_DIRS:
            if fd.is_dir():
                shutil.copy2(out, fd / "accounts_headtohead.png")

    envelope = dict(
        schema_version=1,
        script="project/sigma_lowres/paper_bench/fig_accounts.py",
        label="fig-accounts",
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        sources=dict(tier1024=str(tier1024), g9=str(G9),
                     rapsd=str(REENC_FLOOR)),
        caveats=[
            "figure assembly only: every fit is recomputed by calling the "
            "published helpers of e83_bridge.py and e5_refit.py",
            "spectral curves are the account read through OUR bridge "
            "(G(sigma) + one gain calibrated on 1024->896, floorless)",
            "the delta_reenc gate is plotted; the family is delta-inert at "
            "the curve level (rmse identical to SPD's default, see e8)",
            "ours is in-sample on 512 and post-hoc leave-896-out on 896; "
            "only 768 (and the 1280-tier 1024) are pre-registered held out",
            "the gray +/-eps* band (1.645 x per-bin route-median SEM over "
            "the 1024-tier routes) reproduces plot_debiased_map.py's band "
            "so the verdict is readable off this figure",
        ],
        eps_star=[float(e) for e in d["eps_star"]],
        A_spectral_calibrated_on_896=d["A_spec"],
        floor_law=dict(F0=d["F0"], tau_tokens=d["tau"]),
        gates=d["gates"],
        rmse=d["rmse"],
        floors=d["floors"],
        bootstrap=dict(B=d["boot"]["B"], kept=d["boot"]["kept"],
                       dropped=d["boot"]["dropped"]),
    )
    (out_dir / "result.json").write_text(json.dumps(envelope, indent=1))

    print(f"{'route':<8}{'spectral':>10}{'ours':>8}{'floor':>9}{'gate':>8}")
    for r in ("896", "768", "512", "1120", "1024"):
        e = d["rmse"][r]
        s = f"{e['spectral']:.3f}" if e["spectral"] is not None else "--"
        print(f"{r:<8}{s:>10}{e['ours']:>8.3f}{d['floors'][r]:>9.3f}"
              f"{d['gates'].get(r, float('nan')):>8.3f}")
    print("out:", out_dir)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--tier1024_run", default=str(E1B),
        help="run dir supplying the 1024-tier curves (default: the "
             "published E1b verdict run; anything else skips the figs/ copy)",
    )
    ap.add_argument("--tag", default="", help="suffix on the output run dir")
    _a = ap.parse_args()
    main(Path(_a.tier1024_run), _a.tag)
