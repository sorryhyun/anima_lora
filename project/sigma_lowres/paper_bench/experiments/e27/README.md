# E27 — σ-rotation law: fixed-plane geodesic vs SLERP, with the RoPE-phase generator anchor

| | |
|---|---|
| **Status** | **PRE-REGISTERED 2026-08-09** — committed before the instrument (`e27_rotlaw.py`) existed. CPU-only; no GPU spend anywhere in this experiment. This is [Q7](../../paper_v2/questions.md)'s combined read: candidate 2 (fixed-plane geodesic) carries the law test, candidate 1 (RoPE-phase-driven) enters only as the *generator anchor* — the two are complements (the plane supplies the direction family, the phase mechanism is a candidate for its rate/generator), not rivals. |
| **Question** | Q7: does "the cancellation axis rotates smoothly with σ" upgrade from description to model? Estimand frozen by Q7: **direction cosines under leave-one-σ-out, never scalar map RMSE**; a law earns a claim only if it beats the flanking-bin SLERP baseline. Plus the one mechanism hint made quantitative: E19.4's phase-borne component of C, extracted as a vector from e194's stored π arms — does it align with the rotation *generator* (the in-plane tangent of the axis field), or merely with the axis itself? |
| **Depends on** | [E24](../e24/) (axis field STRUCTURED; Gram machinery + debias conventions inherited verbatim), [E25.0](../e25/) (transport NO-GAIN — the ĝ-frame family is dead and not re-entered here; R-leg machinery + 25.0-1 pass set), [E19.4](../e19/) (phase mediation, `--pi_align` arms preserved in the e194 store — first re-read of those arms since; probe use only, G11 untouched), [E14](../e14/) (leg/debias estimand). |
| **Instruments** | `e27_rotlaw.py` (this dir; CPU) — validation gates, LOO plane + geodesic fits, SLERP/copy baselines, V_phase anchor. |
| **In the paper** | Per Q7's decision gate: if 27.2 lands LAW-BEATS-SLERP before revision_plan §7 step 5, the geometry section may carry one claim + one panel under the axis-field scope rungs; otherwise the question transfers to paper 2 intact (no placeholder language). 27.4 feeds paper 2's mechanism bridge either way. |

## Sources (the three surviving arm_sums stores, one operating point)

The E24 stores verbatim (re-verified on disk 2026-08-09): e193
depth-ledger (σ 0.3/0.4333/0.5667/0.7/1.0 × routes 896,768), e194
pi-causal (σ 0.7/0.8333/0.9625/1.0 × routes 896,768 — **π arms now
read**: `896pi`/`768pi` + half-sets, ignored by E24/E25), e221
per-image-ledger (descriptive only, never fit/flank/target — selection
differs; the E24 house rule). Same adapter ⇒ one operating point.
Merged σ grid per route: **{0.3, 0.4333, 0.5667, 0.7, 0.8333, 0.9625,
1.0}** — 7 bins; σ = 0.7 and σ = 1.0 exist in both standard stores
(cross-store duplicates; E24 measured their axis agreement ≈ 1.0).

## Frozen conventions (inherited + new; no tuning on outputs)

Inherited unchanged from E24/E25.0: legs verbatim
`vector_ledger.bc_ledger` with `data_ref = reenc`, each leg ⊥ against
its condition's own ĝ; rel gate **0.5** per leg per condition
(B-reads gate on rel_cos_B, C on rel_cos_C, R on rel_cos_R with
nR² > 0); cross-condition numerators debiased (draw noise independent
across bins/stores; same-store same-bin cross-route corrections as in
E24/E25); denominators = within-condition cross-set debiased norms;
debiased values may exceed 1 — reported as computed, clamped only in
figures and inside arccos/SLERP angle extraction (clamp to [−1, 1]
there only). R legs via Gram linearity (R = B + C), exactly as E25.0.

New frozen items:

- **Fold structure (LOO over σ)**: one fold per target σ ∈
  **{0.4333, 0.5667, 0.7, 0.8333, 0.9625}** (interior bins only; the
  grid ends 0.3 and 1.0 are never targets — 1.0 also keeps its E24
  endpoint detachment). A fold holds out **every condition at the
  target σ** (both routes, both standard stores — no cross-store
  leakage through the σ = 0.7 duplicate).
- **Fit set per fold**: all gated standard-corpus conditions at
  non-target σ, both routes, endpoints included when gated (σ = 1.0
  may serve as fit/flank, never as target). e221 never enters.
- **Orientation**: an axis is a line — scoring uses |cos|. For angle
  extraction and interpolation, conditions are chain-oriented by
  ascending σ (order ties: 896 before 768, e193 before e194): each
  unit direction's sign is flipped to keep a non-negative cosine with
  the previously oriented one. The fitted plane is sign-invariant
  (it derives from the second-moment operator Σᵢ ûᵢûᵢᵀ).
- **Duplicate pooling (baselines and tangents)**: where a flank σ is
  populated by several gated conditions (routes and/or stores), the
  flank direction is the sign-aligned unit-mean of them, renormalized
  (Gram arithmetic). Predictions are therefore **route-shared** (the
  E24 axis is route-shared at fixed σ); they are scored against each
  gated held-out condition separately.
- **Baselines** at target σ_t: **(a) nearest-bin copy** — pooled unit
  direction at the nearest gated σ (tie → lower σ); **(b) SLERP** —
  spherical interpolation between the pooled flanking directions at
  the nearest gated σ below and above σ_t, parameter
  t = (σ_t − σ_lo)/(σ_hi − σ_lo).
- **Law (candidate 2, two pre-named variants)**: per fold and leg,
  the top-2 eigenplane of the debiased normalized Gram over the fit
  conditions (kernel-PCA relation gives an orthonormal ambient basis
  e₁, e₂ as combinations of fit-condition units); each fit condition
  contributes (σᵢ, θᵢ) with θᵢ = atan2(⟨ûᵢ,e₂⟩, ⟨ûᵢ,e₁⟩) after chain
  orientation; θ(σ) fit by unweighted least squares. Prediction at
  σ_t = cos θ(σ_t)·e₁ + sin θ(σ_t)·e₂.
  - **L-log (primary)**: θ(σ) = θ₀ + ω·ln σ — designated primary
    before any output is seen, on the committed observation that the
    axis (and ĝ) rotate fastest at low σ.
  - **L-lin (secondary)**: θ(σ) = θ₀ + ω·σ. Reported alongside; a
    secondary-only win is recorded as a hint for paper 2 and earns
    **no claim** this revision (multiplicity honesty).
- **Scoring**: debiased |cos(prediction, held-out condition)| per
  gated target condition; Δ = |cos_law| − |cos_slerp| per condition;
  medians and win shares over target conditions. The B table carries
  the verdict; C and R tables reported alongside (E24/E25
  convention). SLERP-vs-copy reported as a sanity row.

## Pre-registered readings

**27.1 — plane σ-stability** (Q7 candidate 2's "first check";
descriptive gate, context for 27.2, B legs; C reported): per fold,
the in-plane share ⟨û_t,e₁⟩² + ⟨û_t,e₂⟩² of each gated held-out
direction in the LOO-fitted plane.

| outcome (B, all folds) | label |
|---|---|
| median share ≥ 0.8 and min ≥ 0.6 | **PLANE-STABLE** — the fixed-plane family is well-posed |
| median share < 0.6 | **PLANE-UNSTABLE** — candidate 2 dies here regardless of 27.2 |
| in between | **PLANE-MIXED** — per-fold pattern recorded |

**27.2 — law vs SLERP** (the Q7 verdict; B legs, primary variant
L-log). Precedence: BEATS → WORSE → NO-GAIN.

| outcome (gated target conditions) | verdict |
|---|---|
| median Δ ≥ 0.02 **and** Δ > 0 on ≥ 2/3 of target conditions | **LAW-BEATS-SLERP** — Q7's claim gate is met; one claim + one panel licensed per Q7's decision gate |
| median Δ ≤ −0.02 **and** Δ < 0 on ≥ 2/3 | **LAW-WORSE** — the geodesic is a worse model than local interpolation; candidate 2 closed |
| otherwise | **NO-GAIN** — smoothness already buys SLERP; "rotates smoothly" stays a description (Q7's explicit null outcome) |

The 0.02 margin mirrors E25.0-2's NO-GAIN threshold. Expected-honest
note, recorded up front: SLERP is a locally optimal discretization of
exactly the geodesic being tested, so a genuine win can only come
from denoising (the fit pools ~6 σ bins × 2 routes where SLERP uses 2
noisy flanks) or from flank-quality gaps — NO-GAIN is the likely
outcome and is a fully respectable verdict.

**27.3 — R̂ transfer** (E25a-facing, descriptive — no paper claim):
identical LOO scoring on the R legs over the 25.0-1-passing
conditions. Outcome shapes only the E25a lookup language
(interpolate-by-law vs measure-every-bin, and whether the
768/σ = 0.4333 hole can be covered by the law); it does not touch
27.2's verdict.

**27.4 — phase generator anchor** (Q7 candidate 1 made quantitative;
the only new estimand): per e194 (route, bin), the pooled phase-borne
component **V_phase = (C − C_pi)⊥ = perp(ḡ_dem − ḡ_dem,π)** (rp
cancels by linearity; ⊥ against the condition's ĝ), with its own
half-set reliability rel_cos_Vp (gate 0.5) and debiased power
nVp² = ⟨Vp₁⊥, Vp₂⊥⟩. Same-(store,bin) inner products involving
V_phase and C are cross-half debiased (the dem arm's draw noise is
shared within a half-set). The comparison object is the **C-leg
in-plane tangent** t̂_C(σ_j): central difference
Ĉ(σ_{j+1}) − Ĉ(σ_{j−1}) (chain-oriented; neighbors prefer the same
store, else the other standard store), projected ⊥ Ĉ(σ_j),
normalized — all Gram arithmetic. Readable bins: 768 ×
{0.7, 0.8333, 0.9625} carry the reading (E19.4's primary route;
0.9625 flagged — its tangent leans on the σ = 1.0 flank); 896
reported. Both |cos(V̂p, t̂_C)| and |cos(V̂p, Ĉ(σ_j))| reported per
readable bin, plus V̂p's in-plane share in the full-data C plane
(descriptive).

| outcome (768 readable bins) | label |
|---|---|
| median \|cos(V̂p, t̂_C)\| ≥ 0.5 **and** per-bin > \|cos(V̂p, Ĉ)\| | **TANGENT-ALIGNED** — the phase component points along the rotation generator; candidate 1 has a measured foothold (paper 2's opening move) |
| median \|cos(V̂p, Ĉ)\| ≥ 0.5 **and** per-bin > \|cos(V̂p, t̂_C)\| | **AXIS-ALIGNED** — the phase component lives along the axis itself; phase mediates the *depth*, not the *rotation* — candidate 1's LOO hope dies cheaply |
| both medians < 0.3 | **NULL** — V_phase is unrelated to the axis-field geometry at these bins |
| otherwise | **MIXED** — per-bin pattern recorded, no label claimed |

## Validation gates (all must pass before any real quantity is read)

1. E24 gates rerun via import: synthetic mini-store agrees with
   `bc_ledger` exactly; e221's committed `ledger.json` reproduced
   exactly (all bins, all six scalars).
2. **Planted geodesic**: a synthetic axis field rotating in one fixed
   plane with θ(σ) = θ₀ + ω ln σ plus per-condition draw noise — the
   instrument must recover the plane (held-out in-plane share ≈ 1),
   the fitted (θ₀, ω) within tolerance, and score the law above SLERP
   (median Δ > 0.02). A **non-planar control** (rotation through a
   third dimension) must NOT pass: held-out in-plane share degraded
   and no law-beats-SLERP.
3. **V_phase**: linearity identity (perp(dem − dem_π) = C − C_pi to
   fp tolerance on the real store); planted synthetic where the phase
   component lies along a known tangent — recovered at cos > 0.9 —
   with an orthogonal control that must not inflate.
4. nR² identity (E25.0's gate) on every condition whose R leg is
   read; max deviation ≤ 1e−6 in ledger units.

## Kill switches / honesty

- Read-only CPU analysis of committed stores; nothing refit on
  outputs; all thresholds frozen in this file before the instrument
  existed.
- π arms are read as a **probe of the phase mechanism only** —
  exactly E19.4's licensed use; no PI training lever (G11 untouched).
- No objective-side use of anything here (E20.4 closed at estimand
  level); a passing law is descriptive/lookup-side only (E25a).
- Per-sample anything stays gated on E22 → 22.4 → E23a, unchanged.
  Wording: pooled directions at this operating point, never
  per-sample.
- e221 descriptive only. σ = 1.0 never a target. No post-hoc
  re-thresholding: if the verdict is NO-GAIN, the manuscript keeps
  "rotates smoothly" as a description (Q7's no-placeholder rule).
- The ĝ-frame transport family is dead (E25.0-2 NO-GAIN) and is not
  re-entered — the fixed plane here is the axis's **own** eigenplane,
  the explicitly different claim Q7 records.

## Outputs

`e27_rotlaw.json` (the record), `e27_loo.png` (per-fold LOO cosines:
law vs SLERP vs copy + plane shares), `e27_phase.png` (anchor read).
Expected cost: one chunked fp64 Gram over ~70 × 77.7M vectors —
~10 min CPU, ≤ ~30 GB resident, no GPU.
