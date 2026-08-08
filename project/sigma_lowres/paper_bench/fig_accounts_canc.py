"""Figure-1 variant with the E19 cancellation-aware account added.

Same measured curves and helpers as fig_accounts.py (nothing refit there),
plus a THIRD account per 1024-tier route: the two-term reduction's own
ingredients -- a route amplitude A on the measured mismatch curve x(sigma)
for the data leg, a per-route constant c for the graph leg -- composed
through the exact angular link WITH the interference the E14/E19 ledger
measured, carried by ONE frozen global constant:

    d_canc(sigma) = sat( sqrt( (A x)^2 + c^2 + 2 rho_bar (A x) c ) ),
    rho_bar = -0.910   (E14 probe-matched ledger, gated verdict bins;
                        licensed as a single global constant by E19:
                        route-uniform 19.0, depth/type-uniform 19.3,
                        operating-point invariant 19.6)

Unlike the additive form sat(Ax) + F, this can dip BELOW its own floor
analogue sat(c) near the crossing A x = c -- the cliff mechanism (E9/E14)
appears natively instead of being fit around.

Honesty: the canc account here is an IN-SAMPLE 2-parameter fit per route
(A, c) with rho_bar frozen -- a feasibility demonstration, not the paper's
held-out governor lane ("ours" on 768/896 is held-out; RMSE comparison is
therefore favorable to canc on those routes and is labeled as such).

Usage:
    uv run python project/sigma_lowres/paper_bench/fig_accounts_canc.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent  # project/sigma_lowres/paper_bench
sys.path.insert(0, str(HERE))

import fig_accounts as fa  # noqa: E402  (bootstraps e5/e8 sys.path itself)
from e5_holdout import bin_gnorm, bin_widths, interp_extrap, m_bar  # noqa: E402
from e5_refit import TIER_OF, sat  # noqa: E402
from e83_bridge import ROUTES  # noqa: E402

RHO_BAR = -0.910
OUT = HERE / "experiments" / "e19" / "accounts_canc.png"
C_MEAS, C_OURS, C_CANC = "0.15", "C0", "#2ca02c"


def canc_pred(A: float, c, x):
    ax = A * x
    return sat(np.sqrt(ax**2 + c**2 + 2.0 * RHO_BAR * ax * c))


def fit_canc(x, y, w):
    """Weighted 2D grid fit of (A, c), two-stage refinement."""
    best = None
    A_med = 1.0 / max(float(np.median(x)), 1e-12)
    A_grid = np.geomspace(A_med * 1e-2, A_med * 1e2, 250)
    c_grid = np.geomspace(1e-3, 3.0, 250)
    for _stage in range(2):
        for A in A_grid:
            preds = canc_pred(A, c_grid[:, None], x[None, :])
            sse = np.sum(w[None, :] * (y[None, :] - preds) ** 2, axis=1)
            i = int(np.argmin(sse))
            if best is None or sse[i] < best[0]:
                best = (float(sse[i]), float(A), float(c_grid[i]))
        _, A0, c0 = best
        A_grid = np.geomspace(A0 / 1.4, A0 * 1.4, 120)
        c_grid = np.geomspace(c0 / 1.4, c0 * 1.4, 120)
    return best[1], best[2]


def main() -> None:
    d = fa.build()
    curves = d["curves"]
    gnorm = {"1024tier": bin_gnorm(fa.E1B)}
    msig, mvals = m_bar()

    canc, params, rmse_c = {}, {}, {}
    for r in ROUTES:
        sig, y, sem = curves[r]
        x = interp_extrap(sig, msig, mvals) / gnorm[TIER_OF[r]]
        w = bin_widths(fa.E1B, len(sig)) / sem**2
        A, c = fit_canc(np.asarray(x), np.asarray(y), np.asarray(w))
        canc[r] = canc_pred(A, c, np.asarray(x))
        rmse_c[r] = float(np.sqrt(np.mean((y - canc[r]) ** 2)))
        # crossings: A x(sigma) = c (mismatch curve is non-monotone, so there
        # can be two -- the HIGH-sigma one is the cliff)
        ax_vals = A * np.asarray(x)
        crossings = []
        for i in range(len(sig) - 1):
            lo, hi = sorted((ax_vals[i], ax_vals[i + 1]))
            if lo <= c <= hi and abs(ax_vals[i + 1] - ax_vals[i]) > 1e-12:
                t = (c - ax_vals[i]) / (ax_vals[i + 1] - ax_vals[i])
                crossings.append(float(sig[i] + t * (sig[i + 1] - sig[i])))
        cross = crossings[-1] if crossings else None
        params[r] = dict(A=A, c=c, floor_analog=float(sat(c)),
                         crossing=cross, crossings=crossings)

    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.8))
    for ax, r in zip(axes.ravel()[:3], ROUTES):
        sig, y, sem = curves[r]
        ax.fill_between(sig, -d["eps_star"], d["eps_star"], color="0.5",
                        alpha=0.35, lw=0, zorder=1,
                        label=r"$\pm\varepsilon^{*}$")
        ax.plot(sig, d["ours"][r], "-", color=C_OURS, lw=1.6, alpha=0.75,
                zorder=2, label="ours (additive, Fig. 1)")
        ax.axhline(d["floors"][r], color=C_OURS, lw=0.8, ls="--", alpha=0.45,
                   zorder=1)
        ax.plot(sig, canc[r], "-", color=C_CANC, lw=2.0, zorder=3,
                label=rf"canc. account ($\bar\rho={RHO_BAR}$ frozen)")
        ax.axhline(params[r]["floor_analog"], color=C_CANC, lw=0.8, ls=":",
                   alpha=0.6, zorder=1, label=r"canc floor analogue sat($c$)")
        if params[r]["crossing"] is not None:
            ax.axvline(params[r]["crossing"], color=C_CANC, lw=0.8, ls=":",
                       alpha=0.5)
        has_end = math.isclose(float(sig[-1]), 1.0, abs_tol=1e-9)
        n_in = len(sig) - 1 if has_end else len(sig)
        ax.errorbar(sig[:n_in], y[:n_in], yerr=sem[:n_in], fmt="o", ms=4.0,
                    color=C_MEAS, capsize=2, lw=1.0, zorder=4,
                    label="measured (debiased)")
        if has_end:
            ax.errorbar(sig[-1:], y[-1:], yerr=sem[-1:], fmt="o", ms=4.5,
                        mfc="white", color=C_MEAS, capsize=2, lw=1.0,
                        zorder=4)
        ax.set_title(fa.DISP[r], fontsize=10.5)
        ax.axhline(0, color="0.8", lw=0.6, zorder=0)
        ax.set_xlabel(r"$\sigma$", fontsize=9)
        ax.set_ylabel("debiased gap", fontsize=9)
        ax.tick_params(labelsize=8)
        box = (f"RMSE ours {d['rmse'][r]['ours']:.4f}\n"
               f"RMSE canc {rmse_c[r]:.4f}\n"
               f"$A$={params[r]['A']:.2f}, $c$={params[r]['c']:.3f}"
               + (f"\ncrossing $\\sigma\\approx${params[r]['crossing']:.2f}"
                  if params[r]["crossing"] is not None else ""))
        ax.annotate(box, (0.97, 0.96), xycoords="axes fraction", ha="right",
                    va="top", fontsize=7.5, color="0.25",
                    bbox=dict(fc="white", ec="0.8", alpha=0.85))
        if r == "896":
            leg_handles, leg_labels = ax.get_legend_handles_labels()

    ax = axes.ravel()[3]
    ax.axis("off")
    ax.legend(leg_handles, leg_labels, fontsize=7, loc="lower left",
              frameon=True, handlelength=1.6, labelspacing=0.35)
    lines = [
        "cancellation-aware account (E19):",
        r"$d = \mathrm{sat}\!\left(\sqrt{(Ax)^2 + c^2 + 2\bar\rho\,Axc}\right)$,"
        rf"  $\bar\rho = {RHO_BAR}$ frozen (E14 gated bins)",
        "",
        "same Eq. (8) ingredients ($A$ on mismatch curve $x(\\sigma)$,",
        "per-route $c$), but the interference term dips below sat($c$)",
        "at $Ax = c$ — the cliff appears natively; fitted crossings",
        "$\\sigma \\approx$ 0.63/0.66/0.67 all land in the E9 window",
        "(0.56–0.81; E9's 768 crossing estimate: 0.688).",
        "",
        "canc = in-sample fit (lanes NOT matched; ours held-out 768/896).",
    ]
    ax.annotate("\n".join(lines), (0.02, 0.98), xycoords="axes fraction",
                va="top", fontsize=8.4, color="0.2")

    fig.suptitle(
        "Fig. 1 variant — the E19 cancellation-aware account on the measured map",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print(f"wrote {OUT}")
    for r in ROUTES:
        print(r, params[r], "rmse_canc", round(rmse_c[r], 5),
              "rmse_ours", round(d["rmse"][r]["ours"], 5))


if __name__ == "__main__":
    main()
