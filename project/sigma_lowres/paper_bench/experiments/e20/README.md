# E20 — the cancellation-aware account under the paper's own lanes

| | |
|---|---|
| **Status** | **PLANNED 2026-08-08** — this file is the pre-registration. The frozen constant and reading rules below are committed **before** `e20_refit.py` exists (theory-first, mirroring 19.1). |
| **Question** | E19 located the near-cancellation's geometry (global, Jᵀ-born, ρ̄ ≈ −0.91) and the Fig.-1 demonstration (`fig_accounts_canc.py`, in `e19/accounts_canc.png`) showed the cancellation-aware link fits the measured map in-sample and lands its crossings in the E9 window. Open: does it survive the **same governor / held-out protocol** the paper's additive account runs under — same parameter count per route, same lanes, same bootstrap — and does it thereby earn the reserved full-σ-map confirmation spend? |
| **Depends on** | [E19](../e19/) (ρ̄ license: 19.0 route-uniform, 19.3 depth/type-uniform, 19.6 operating-point invariant), [E14](../e14/) (ledger; source of ρ̄), [E5](../e5/) (`e5_refit.py` lanes: FIT_ROUTES 896/512/1120, HELD_OUT 768/1024, LOO-896; governors; bootstrap), [E9](../e9/) (window ↔ crossing), `paper_bench/fig_accounts_canc.py` (feasibility demo, NOT lane-matched) |
| **Instruments** | 20.1 `e20_refit.py` (extends `e5_refit`; CPU); 20.2 free re-read of 20.1; 20.3 decision-gated GPU (the reserved spend); 20.4 stretch (closure-derived data term) |
| **In the paper** | If 20.1 wins: the successor of Eq. (8) in paper 2 §4.6 (the cancellation account's operational form), with `accounts_canc.png` upgraded to the lane-matched version. Paper 1's Fig. 1 lane is untouched either way. |

## Frozen constants (pre-registered — do not refit on curves)

- **ρ̄ = −0.910** — E14 probe-matched ledger, gated verdict bins
  (σ ≥ 0.3, rel ≥ 0.5, |ρ| ≤ 1), all three 1024-tier routes pooled,
  n = 26, range −0.99…−0.70. Frozen here; never tuned on gap curves.
- **Link**: d(σ) = sat(√((A·x)² + c² + 2ρ̄·(A·x)·c)),
  sat(κ) = 1 − 1/√(1+κ²) — the paper's eq:exact with κ_par dropped,
  exactly as the additive lane already drops it. x(σ) = the measured
  mismatch curve over ‖ḡ‖, **shared bit-identically with the additive
  lane** so every difference is link-form only.
- Under the hood this is the four-term expansion with assumption (ii)
  replaced by "ρ is one global constant" — a falsified assumption
  (E14: I deep-negative everywhere) swapped for a measured invariance
  (E19), at quadratic order, then kept licensed at plateau by the
  un-expanded parent form.

## 20.1 — lane-matched refit (CPU, ~minutes)

`e20_refit.py` mirrors `e5_refit`/`fig_accounts.build` exactly, link
swapped:

1. **Fit routes** (896 / 512 / 1120): weighted two-stage grid over
   (A_r, c_r), weights = bin_widths/sem² — identical protocol to
   `fit_sat_route`, one extra grid axis.
2. **Governors, zero route freedom on held-out routes**: A via the same
   `two_point_governor` on RATIO; c via the same exp-in-TOKENS family
   the floor law uses. **Pre-registered choice: the law is fit on c
   (the κ-scale quantity), with sat(c) reported alongside for
   comparability with the old floor.** LOO-896 lane mirrored
   (governors from 512+1120).
3. **Held-out predictions**: 768 (1024-tier) and 1024 (1280-tier) from
   governors; 896 via the LOO lane; 512/1120 in-sample.
4. **Bootstrap**: full-pipeline B = 1000, same recipe, bands for
   768/896.
5. **Ablations**: (a) ρ̄ = 0 — must degrade toward the additive fit or
   worse (sanity: the interference term does the work, not the extra
   grid axis); (b) ρ̄ ± 0.05 sensitivity (fit-free — same frozen fits,
   re-predict); (c) report per-bin residuals against ±ε*.

## 20.2 — the crossing as a zero-cost prediction

From 20.1's **held-out** parameters only: predicted dip location
(argmin of d) and crossing (Ax = c) per route, vs E9's window
(0.56–0.81) and E9's 768 crossing estimate (0.688). NB estimand
honesty: Ax = c locates the **gap-curve dip**, not the ledger's
√(S/F) = 1 (E14's 896 leg crossing ≈ 0.47–0.53 is a different
quantity; its 768 ledger has no leg crossing). The demo's in-sample
crossings were 0.63/0.66/0.67 — the pre-registered check is that the
held-out 768 dip stays in-window.

## Pre-registered reading rules

| outcome | verdict |
|---|---|
| held-out RMSE(canc) ≤ RMSE(additive) on **768 AND LOO-896**, dip in-window on both | **WIN** — canc account replaces Eq. (8) as the paper-2 operational form; proceed to 20.3 |
| better on exactly one safe lane, or RMSE tie with dip in-window | **PARTIAL** — "geometry right, amplitude law open" (19.0's split); record, no 20.3 |
| worse on both safe held-out lanes | **KILL** — the interference form does not transfer through governors; additive lane stands, this file records the failure |

**512 expected-failure pre-registration**: a global ρ̄ imposes more
cancellation than the unsafe route's mismatched-magnitude legs deliver
(19.0 item 2; the demo over-cancelled 512 mid-window, RMSE 0.114 vs
0.093). 512 improvement is NOT required; 512 worsening is NOT a kill.
Diagnostic value only: the sign/size of its mid-window residual is the
amplitude law's fingerprint.

**1280-tier guard**: canc predictions on 1120/1024 must stay within
the additive lane's bootstrap band envelope — the new link must not
break the tier the old account already covers.

## 20.3 — the reserved full-σ-map spend (GPU, decision-gated)

Only on a 20.1 **WIN**: freeze the 20.1 parameters, publish dense-grid
predictions per route in this file **before** submitting the run, then
spend the reserved verdict-grid measurement (E19 cost ladder: "the
full σ-map stays reserved for confirming whatever theory survives")
against them. Grid/budget to be specified in the 20.3 amendment after
20.1 lands; daemon-routed as usual.

## 20.4 — stretch: how derivable is the data term?

19.5 showed the Gaussian closure owns both leg *directions* at
mid/high σ with a smooth σ-dependent per-leg scale miss (B: 1.15–1.31
over mid-σ, 0.68–0.86 near endpoint). Optional arm: replace the
measured x(σ) with the closure-predicted B-leg amplitude curve (one
fitted scale, or a linear-in-σ correction) and re-run 20.1's lanes.
This bounds how much of the account can be *derived* rather than
measured. Not verdict-bearing for 20.1; separate reading.

## Kill switches / honesty

- ρ̄ frozen in this file before any fit; the ρ̄-sensitivity ablation is
  read-only.
- κ_par dropped in **both** lanes symmetrically — no asymmetric
  approximation.
- σ = 1 endpoint detached (open marker, non-verdict) as in Fig. 1;
  low-σ bins inherit the standing caution (assumption-(iii)-analogue:
  c constant is known to fail below σ ≈ 0.45 — inherited, not new).
- RMSE comparisons are lane-matched ONLY against `fig_accounts.build`
  outputs recomputed in the same process — never against the demo's
  in-sample numbers.
- The demo (`fig_accounts_canc.py`) stays labeled a demonstration; it
  is not evidence for 20.1's verdict.

## Cost ladder

| item | cost |
|---|---|
| 20.1 | CPU, ~minutes (grid fit + B=1000 bootstrap) |
| 20.2 | free (re-read of 20.1) |
| 20.3 | GPU, decision-gated — the reserved verdict-grid spend |
| 20.4 | CPU, optional |
