# E20 — the cancellation-aware account under the paper's own lanes

| | |
|---|---|
| **Status** | **20.1 + 20.2 DONE 2026-08-08 — verdict PARTIAL** (better on exactly one safe lane: 768 wins with dip in-window, LOO-896 loses; 1280-tier guard fails on 1024). Per the reading rules: geometry right, amplitude law open; **no 20.3 spend**. **20.4 DONE 2026-08-08 — NEGATIVE** (derived data term fails the lanes; failure is estimand-level, not closure-level — see §20.4 Results). Results below; runs `runs/20260808-0924-e20-refit/`, `runs/20260808-0944-e20-stretch/`. Pre-registration (frozen constants, lanes, reading rules) unchanged below. |
| **Question** | E19 located the near-cancellation's geometry (global, Jᵀ-born, ρ̄ ≈ −0.91) and the Fig.-1 demonstration (`fig_accounts_canc.py`, in `e19/accounts_canc.png`) showed the cancellation-aware link fits the measured map in-sample and lands its crossings in the E9 window. Open: does it survive the **same governor / held-out protocol** the paper's additive account runs under — same parameter count per route, same lanes, same bootstrap — and does it thereby earn the reserved full-σ-map confirmation spend? |
| **Depends on** | [E19](../e19/) (ρ̄ license: 19.0 route-uniform, 19.3 depth/type-uniform, 19.6 operating-point invariant), [E14](../e14/) (ledger; source of ρ̄), [E5](../e5/) (`e5_refit.py` lanes: FIT_ROUTES 896/512/1120, HELD_OUT 768/1024, LOO-896; governors; bootstrap), [E9](../e9/) (window ↔ crossing), `paper_bench/fig_accounts_canc.py` (feasibility demo, NOT lane-matched) |
| **Instruments** | 20.1 `e20_refit.py` (extends `e5_refit`; CPU); 20.2 free re-read of 20.1; 20.3 decision-gated GPU (the reserved spend — **not taken**: 20.1 PARTIAL); 20.4 stretch (closure-derived data term) |
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

## Results (2026-08-08) — 20.1 + 20.2, verdict PARTIAL

Run: `runs/20260808-0924-e20-refit/` (`e20_main.png`, `result.json`).
Verdict RMSEs (RMSE*) are σ=1-endpoint-detached from **both** lanes
symmetrically; all-bin values in the envelope agree in direction on
every verdict lane.

**Fit routes** (A, c | profile se): 896 → 0.389±0.040 / 0.340±0.033;
512 → 1.808±0.100 / 1.339±0.074; **1120 → 0.142±0.016 / c ≈ 0.0007**.
The graph-leg constant **collapses to ~0 on the 1280-tier**, exactly
mirroring the additive lane's F₁₁₂₀ ≈ 0 — so the c-law's >0.005 filter
drops 1120 and the exp-in-TOKENS law anchors on the same (896, 512)
legs as the additive floor law (c₀ = 2.69, τ = 1456 tok, c(2160) =
0.610). The LOO-896 c comes out nearly floorless (c = 0.026). Honesty
note: the canc A's ratio-twin z = 5.69 (A₈₉₆ vs A₁₁₂₀ far apart) — the
A-governor is strained in this lane.

| lane | RMSE* canc | RMSE* additive | dip (window 0.56–0.81) |
|---|---|---|---|
| **768 held-out (safe)** | **0.0778** | 0.0987 | **0.673 IN** |
| **LOO-896 (safe)** | 0.0856 | **0.0769** | 0.871 OUT |
| 1024 held-out (guard tier) | 0.1993 | **0.0462** | 0.856 OUT |
| 512 in-sample (diagnostic) | 0.1190 | 0.0961 | — |

- **768 is the clean win**: RMSE beat + the canc curve reproduces the
  measured non-monotone shape (peak σ≈0.3, dip σ≈0.67) that the
  additive account structurally cannot. From held-out parameters only,
  **dip 0.673, crossing Ax = c at 0.655** — inside the E9 window and
  close to E9's 768 crossing estimate 0.688 (20.2's zero-cost check).
- **LOO-896 loses** because its lane inherits the 1120 collapse:
  c_loo ≈ 0.026 makes the canc curve floorless-flat, missing the low-σ
  bump the additive floor F(3012) partially captures.
- **1280-tier guard: 1120 PASS (0/4 bins out), 1024 FAIL (3/4 out,
  max excursion +0.196)** — extrapolating the 1024-tier c-law up-tier
  (c(4116) = 0.159) over-shapes 1024, whose own tier wants c ≈ 0.
- **512 diagnostic** (pre-registered expected failure, not a kill):
  mean mid-window residual **+0.054** — the global ρ̄ over-cancels the
  mismatched-magnitude route, same sign as the demo. Amplitude-law
  fingerprint confirmed.
- **Ablations**: ρ̄ = 0 degrades 768 to RMSE* 0.168 (the interference
  term does the work, not the extra grid axis — sanity holds) while
  *improving* 1024 to 0.082 (the interference + extrapolated c is what
  breaks the guard tier). ρ̄ ± 0.05: 768 stays 0.076–0.082 — verdict
  insensitive. ±ε*: 5/9 (768) and 4/9 (896) bins exceed the band.
- Bootstrap kept 999 (canc) / 981 (additive) of B = 1000.

**Reading**: the interference *geometry* transfers through governors on
the tier that licensed ρ̄ (768: win + in-window dip/crossing), but the
amplitude law does not — c is a **1024-tier quantity, not
token-governed across tiers**. Per the pre-registered table this is
PARTIAL ("geometry right, amplitude law open", 19.0's split): recorded
here, **no 20.3**; Eq. (8) stands as paper 2's operational form and
`accounts_canc.png` stays labeled a demonstration. Reopening would
need a per-tier c account (e.g. c ≈ 0 above the 1024 tier) — that is a
new pre-registration, not a refit of this one.

## 20.3 — the reserved full-σ-map spend (GPU, decision-gated) — **NOT TAKEN** (20.1 returned PARTIAL; the reserved spend remains unspent)

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

### 20.4 Results (2026-08-08) — NEGATIVE, failure is estimand-level, not closure-level

Instruments: `e20_stretch.py` (imports 20.1's chain from `e20_refit`,
so the lanes are bit-identical; measured-arm cross-check reproduces
20.1's 768 RMSE* to 0.0e+00) over amplitude curves from a rerun of
`bench/ledger_b_scoreshift.py` patched to emit absolute per-image RMS
leg amplitudes (`amp_pred_B`/`amp_meas_B`; new `--closures` arg, diag
closure — closures agree to ~0.01 in 19.5). Runs:
`bench/results/20260808-0939-e204-amp` (amp dump, daemon job
`20260808-093836-7bb1f0`) → `runs/20260808-0944-e20-stretch/`.
1280-tier routes keep measured x in every arm (no closure instrument
there); the clean read is the 768 held-out lane.

| arm (768 held-out lane) | RMSE* | dip |
|---|---|---|
| measured m̄/G (20.1) | **0.078** | 0.673 IN |
| measured B amp (r-ledger, diagnostic) | 0.412 | 0.883 |
| closure-predicted B amp (the pre-registered arm) | 0.436 | 0.874 |
| closure B × (1 + 0.93σ) (the one-dof correction) | 0.510 | 0.914 |
| per-route closure B / no-/G variants | 0.29–0.46 | 0.06–0.88 |
| (additive reference) | 0.099 | — |

- **Every derived arm fails at lane level** (4–6× the measured lane's
  RMSE*, worse than the additive account; dips out of window), robust
  to pooled-vs-per-route, /G-vs-raw, and the linear-σ dof (whose
  in-sample b* = +0.93 makes 768 *worse*).
- **The failure is not the closure**: the *measured* B-amplitude arm
  fails equally (0.412 vs 0.436) — swapping measured→closure
  amplitudes moves almost nothing. The ledger's B-leg amplitude is
  simply **not the account's x(σ)**: an estimand mismatch, not a
  prediction miss.
- **Where they part** (normalized, mean 1 over σ ≥ 0.3, 896 grid): the
  derived x agrees with the lane's x to **≤ 5–7 % over σ ≈ 0.3–0.7**
  — the band where E19 licensed the closure — but diverges at both
  tails: 0.60× (pred) / 1.47× (meas) at the lowest bin, and **missing
  the σ→1 upturn entirely** (67 % / 175 % rel. dev. at the endpoint).
  The tails are exactly what the lanes lean on: low-σ bins carry the
  largest fit weights (smallest SEMs) and the high-σ upturn carries
  the dip/crossing geometry.
- **Reading**: the data term is derivable *only over the mid-σ band*;
  the account's operative structure lives in the tails, where the
  r-level leg estimand and the mismatch-curve estimand part ways.
  Derivability of the account through the closure B-leg: **none at
  lane level**. Any future "derive the amplitudes" attempt (the E21
  sketch in the 20.1 reading) must first build the estimand bridge
  between the ledger legs and the gap account's x — the closure is
  not the bottleneck. *(Naming note: "the E21 sketch" here is the
  informal amplitude-derivation idea, **not** the cell-level g-ledger
  that later ran as [E21](../e21/) — see the numbering notes in
  e21/e22. And the bridge has since been closed from the
  **measurement** side at σ = 0.7 by E22.4 (PER-SAMPLE HOLDS); the
  objective-side bridge this paragraph requires remains open.)*

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
| 20.3 | GPU, decision-gated — the reserved verdict-grid spend (not taken — gate did not fire) |
| 20.4 | CPU, optional |
