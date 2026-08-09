# E5 — Eq. 3 held-out validation, and the three-form refit

| | |
|---|---|
| **Status** | **DONE 2026-07-29** — qualified PASS (held-out), plus the functional-form refit the same day |
| **Verdict** | The two-term account **predicts routes it was not fit on** at ~0.09 RMSE; the floor exp-law is 2-for-2 on floors it never saw. The **exact angular link X** wins held-out and headlines; the derived quadratic Q is its small-κ limit. |
| **Runs** | `runs/20260729-1130-e5-holdout/` (result.json + `e5_overlay.png`) · `runs/20260729-1322-e5-refit/` (result.json + 3-panel `e5_refit.png`) · later refits `runs/20260730-1025-e5-refit/`, `runs/20260730-1145-e5-refit/` |
| **Scripts** | `e5_holdout.py` (pass criteria pre-registered in the script header **before** the numbers) · `e5_refit.py` (three candidate forms) — analysis-only, no GPU |
| **Sources** | [E1](../e1/)b paired-debiased (1024 tier), G9 raw paired (1280 tier — no self-floors exist there, D=4 caveat), G7 m(σ), per-run gnorm |
| **Consumed by** | [E8.3](../e8/) (imports `e5_holdout`'s fit/paired-stats helpers) |
| **In the paper** | §4.6 "the account confronted"; `figs/e5_refit.png` is the §4.6 figure (`figs/e5_overlay.png` also staged) |

## E5 — held-out validation

**Question.** Does the two-term account *predict* routes it was not fit
on? Fit A_e, Floor_e on {1024→896, 1024→512, 1280→1120}, fit governor
models A(ratio) and F(target tokens), predict {1024→768, 1280→1024}
from measured m̄(σ) (G7, route-uniform mean) and each run's own G(σ).

### Results — all four pre-registered gates fire; PASS with character

- **p\* = 2.00** (grid boundary — but that *is* the small-mismatch
  cosine-geometry limit the account pins a priori).
- **Ratio governor, amplitude-level:** A_512 0.077±0.010,
  A_896 0.0076±0.0053, A_1120 0.0068±0.0015 — the two ratio-0.875
  routes agree at **z = 0.14** despite 1.6× different target capacity.
  G9 showed the ratio governor at the σ\* level; this is the same
  verdict at the fitted-amplitude level. (Refuted as a *test* by E13:
  `A` carries an unstated per-run G normalization, so z is
  scale-dependent — 0.14 → 8.17 under bin-width reweighting,
  reproducing at 8.2–8.7 under all three forms. The governor's
  *predictions* are unaffected.)
- **Floor governor, interpolation-level:** fitted floors 512 +0.264,
  896 +0.039, 1120 +0.002; exp-law F(n) = 0.70·exp(−n/1041 tok) fit on
  512+896 predicts F(4825) = +0.007 (1120 fitted +0.002 ✓) and
  **F(2160) = +0.088 for 768 vs measured debiased endpoint
  +0.092±0.012** — the held-out floor is hit dead-on.
- **Held-out 1024→768:** RMSE 0.093 — *better than the oracle
  quadratic fit on the held-out data itself* (0.105); null committed
  region (σ > σ_eq: gap ≈ 0) RMSE 0.164 vs ours 0.096.
- **Held-out 1280→1024:** RMSE 0.092 vs oracle 0.056 (ratio 1.64,
  inside the 2× gate); null 0.104 vs ours 0.100 (marginal).

### Misses, recorded honestly (the "character")

1. **768 mid-σ window:** measured ≈ 0 in σ∈[0.56,0.94] vs predicted
   flat ~0.09–0.14 — the two-term form cannot dip below Floor_e. Same
   window-vs-endpoint anomaly [E1](../e1/)(b) already recorded; E5
   sharpens it into the account's one visible structural failure.
   *(Later resolved by [E9](../e9/): it is the interaction term,
   I_768 < 0. Two further updates: E13 H2's re-flattening triggered the
   pre-registered re-score of the "768 dip outside the 95 % band"
   claim, and E19 found no probe-matched in-window 768 crossing — the
   crossing↔window localization survives on 896 only.)*
2. **1280→1024 peak overshoot** (predicted 0.37 vs measured 0.18 at
   σ=0.375): A(0.8) comes from linear-in-ratio interpolation resting on
   only two distinct ratio values; A(r) is evidently convex between
   0.5 and 0.875. *(Retired by the refit below — it was quadratic-form
   overshoot, not convexity.)*
3. **χ²/bin 7.6 / 9.2 held-out (0.9–5.2 in-sample):** the prediction is
   NOT within instrument resolution anywhere — the claim licensed is
   "shape + magnitude class + governors at ~0.09 RMSE", never
   "predicts within ε\*".

### Post-hoc addendum (leave-896-out, NOT pre-registered; same session)

896 was a fit route, so it had no held-out prediction; re-deriving the
governors from 512+1120 alone and predicting 896 gives: A(0.875) from
the ratio-twin 1120 alone = 0.0068 (vs 0.0076 fitted on 896 itself);
floor exp-law refit through 512+1120 (fragile leg: F_1120 =
0.002±0.014 enters a log fit) → **F(3012) = +0.019 vs E1a's measured
896 debiased floor +0.019 [+0.010,+0.030] — dead center. The floor law
is 2-for-2 on floors it never saw** (768 +0.088/+0.092, 896
+0.019/+0.019). Curve-level RMSE 0.072 vs oracle 0.064 (ratio 1.13),
χ²/bin 6.4 — same character: low-σ bins under-predicted (measured
+0.09..+0.21 vs predicted +0.02..+0.05 at σ ≤ 0.31), mid/high-σ
tracked, and the E1b σ=1 endpoint reads +0.042±0.011 vs +0.019
predicted — the endpoint-vs-window anomaly again, now visible on 896.
(E13 later resolved the pieces: the low-σ under-prediction is real
curve change — E13 puts that route's signal in the slope where E1b put
it in the floor; 896's σ→1 approach is a certified instrument limit —
H3; and the σ = 1 step is a correction-regime discontinuity — H1.)

### Voice decision

`../../paper_plan.md` §2 resolved → Eq. 3 headlines as a *predictive*
first-order account with stated ~0.09 RMSE resolution. Of the two
governor statements, "absolute size sets floor" upgraded to a
measured-governor statement (floor exp-law hitting the held-out 768)
and stands; "ratio sets amplitude" rested on the ratio-twin z = 0.14,
which E13 later showed is a scale-dependent statistic (per-run G
normalization of A) — the *prediction* survives ("E5 predicts routes it
was not fit on at ~0.09 RMSE" is intact under E13), the governor *test*
does not. The mid-σ 768 dip and the A(r) convexity go into the
account's stated limits.

## E5 refit — three functional forms for the data-branch term

`../../paper_plan.md` §6 step 2 (analysis-only, no GPU): same fit →
governors → held-out pipeline as `e5_holdout.py` under three candidate
forms for S_e (shared additive floor in each): **P** power `A·m/G^p`
(p shared, scanned — the shipped form), **Q** derived
small-perturbation quadratic `A·(m/G)²`, **X** exact angular link
`1−1/√(1+(c·m/G)²)` (exact 1−cos of a pure orthogonal perturbation with
‖δg⊥‖ = c·m; small-κ limit = Q with A = c²/2).

### Results

- **P's scan lands on p\* = 2.00 exactly** — the "free" exponent picks
  the quadratic on its own; P and Q are near-duplicates everywhere.
- **In-sample 512** (the severe route, gap ~0.3): χ²/bin 3.5–3.9
  across all three forms (Q narrowly best) — the floor absorbs most of
  512, so the small-κ worry does not discriminate in-sample.
- **Held-out is decisive: X wins** — mean RMSE 0.071 vs 0.093 (P) /
  0.094 (Q). The entire margin is the 1280→1024 route (0.049 vs
  ~0.095): **the saturation of the exact link removes the peak
  overshoot E5 had booked as "A(r) evidently convex"** (miss #2). On
  768 all three tie (0.089–0.093) — the mid-σ dip is form-independent
  (it is the interaction term I_768 < 0, not a form artifact).
- **Form-invariant governors:** floor exp-law F(2160) = +0.087/+0.088
  in all three; ratio-twin z = 0.14 (P) / 0.44 (Q) / 0.15 (X) — the
  floor half stands; the ratio-twin z is the scale-dependent statistic
  E13 refuted as a test (see above).

### Ledger naming (consumed by main.tex §3.3/§4.6)

The **exact angular link X headlines as the best empirical predictor**
— its empirical ingredient is the linear mismatch loading ‖δg⊥‖ = c·m,
and it is labeled empirical on that account; **Q is reported as the
derived small-perturbation form** and is X's small-κ limit. One
geometry at two orders: the derived quadratic is not a competitor to
the headline form but its local law, and E5 miss #2 (A(r) convexity) is
retired — the "convexity" was quadratic-form overshoot.
