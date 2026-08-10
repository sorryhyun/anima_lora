# E27 — σ-rotation law: fixed-plane geodesic vs SLERP, with the RoPE-phase generator anchor

| | |
|---|---|
| **Status** | **DONE 2026-08-09** (pre-registered same day before the instrument existed; CPU-only as planned, one 567 s run). Verdicts: **27.1 PLANE-MIXED** (no fixed plane: LOO share median 0.77 B / 0.83 C, collapsing to 0.24–0.44 at σ = 0.9625), **27.2 LAW-WORSE** (the 2-parameter geodesic loses to SLERP at *every* B target, median Δ = −0.166; both variants), **27.3 R LAW-WORSE** (E25a keeps measure-every-bin + neighbor interpolation, which scores well), **27.4 NULL** (V_phase is a highly reproducible direction, rel 0.83–0.97, that lives almost entirely **outside** the axis-field plane — share 0.01–0.09 — and aligns with neither the rotation generator nor the axis). Net: **Q7 answers NO at this data resolution** — "rotates smoothly" stays a description; the σ-binned lookup + SLERP interpolation is not just the engineering floor but the measured best; and the RoPE-phase mechanism is a within-bin effect whose direction is unrelated to the σ-rotation. This was [Q7](../../paper_v2/questions.md)'s combined read: candidate 2 (fixed-plane geodesic) carried the law test, candidate 1 (RoPE-phase-driven) entered only as the *generator anchor* — the two are complements (the plane supplies the direction family, the phase mechanism was a candidate for its rate/generator), not rivals. |
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
   the fitted (θ₀, ω) within tolerance, score the law above the
   nearest-copy baseline (median gain > 0.02, a true geometry gap),
   and hold **parity** with SLERP (median Δ > −0.02). A **non-planar
   control** (rotation through a third dimension) must NOT pass:
   held-out in-plane share degraded and no law-beats-SLERP.
   *Amended pre-run — see the defect note below.*

**Pre-registration defect (recorded 2026-08-09, before any real store
was read; discovered while building the instrument's synthetic
gates).** This gate originally required the planted planar geodesic to
*beat* SLERP at median Δ > 0.02. That requirement is mathematically
unsatisfiable under the frozen debiased scoring: the debiased
estimator is unbiased for both predictions, and SLERP between flanking
bins is an exact discretization of a fixed-plane geodesic — its only
systematic error against the target is the t-parametrization mismatch
(linear-in-σ vs the planted ln σ rate), which is ≪ 0.02 for any
rotation field in this line's regime (< 90° total span, no
orientation flips). The gate is amended to copy-gap + SLERP-parity as
stated above. **The 27.2 verdict thresholds are untouched** — but the
defect sharpens their interpretation, recorded here up front:
LAW-BEATS-SLERP, if it occurs on real data, must come from
flank-quality gaps or estimation-noise averaging, not from geodesic
geometry per se; and a NO-GAIN outcome does **not** refute the
fixed-plane geodesic family — it means SLERP already realizes it,
i.e. 27.1 (plane stability) carries the model-vs-description content
in that branch, and the E25a lookup keeps per-bin interpolation
regardless.

Two further gate-design facts measured on synthetics while building
the instrument (recorded pre-run; they refine the *controls*, not the
verdict rules): (i) a **smooth** non-planar curve is not a usable
non-planarity control — over 7 grid points any smooth one-parameter
curve is nearly rank-2 (mean + tangent dominate; sample-correlation
between planes absorbs the extra dimensions; a strongly twisted
torus curve still read held-out share 0.86–0.98) — so the control
plants per-bin out-of-plane excursions instead, and, correspondingly,
a PLANE-STABLE outcome on real data must not be oversold: at 7 bins
it excludes bin-idiosyncratic off-plane structure, not smooth plane
wander. (ii) On that control the law *legitimately* out-predicts the
junk-laden flanks (LAW-BEATS-SLERP fires while the plane share
correctly collapses) — so the certified object is the **pair**
(PLANE-STABLE ∧ LAW-BEATS-SLERP), which is what the control gate now
asserts the instrument refuses; a real-data BEATS under PLANE-MIXED
would inherit exactly this caution via 27.1's frozen reading.
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

Post-verdict illustration (added 2026-08-10, committed digests only —
no new quantities): **`e27_rhat_sphere.png`** (`e27_rhat_sphere_fig.py`)
— R̂ = (B+C)̂ per σ bin as a point on the unit sphere, one panel per
route, with **neighbor SLERP great-circle arcs** between adjacent bins:
the figure draws exactly the settled model (27.3 measure-every-bin +
neighbor interpolation) and nothing more — no plane anywhere;
consecutive arcs bend (bend angle from the committed cosine triple via
the spherical law of cosines, annotated; 0° = one great circle), the
27.1 rolling plane made visible. Embedding honesty: the 896 chain
{0.3…0.7} has all six committed pairwise cosines (exact top-3 embed,
captured share annotated), 768 is an exact triangle; σ = 0.8333 rides
its single committed cosine (dashed/hollow, drawing convention);
768/σ = 0.4333 = the 25.0-1 hole, annotated not invented. Per-point:
|R| = √(2·R_quad) and the held-out LOO SLERP |cos| (the certified
interpolation quality at that bin).
Expected cost: one chunked fp64 Gram over ~70 × 77.7M vectors —
~10 min CPU, ≤ ~30 GB resident, no GPU.

## Results (2026-08-09)

Instrument: `e27_rotlaw.py` (this dir). All validation gates passed
before any real quantity was read (E24 synthetic + e221 exact
reproduction; planted geodesic: plane + |ω| recovered, law > copy
+0.030, law ~ SLERP +0.001; per-bin-excursion control: share 0.51 and
certification refused via 27.1; planted V_phase tangent recovered
0.9+, orthogonal control flat; nR² identity 4.2e−14; V_phase
linearity identity < 1e−9). Run: daemon job `20260809-162632-a66fac`,
567 s CPU, 44 × 77.7M Gram. Record: `e27_rotlaw.json`.

### 27.1 — PLANE-MIXED: there is no fixed plane

LOO in-plane share of held-out directions: **median 0.769 / min 0.336
(B)**, 0.831 / 0.392 (C). The per-fold pattern is σ-structured, not
noise: interior folds 0.59–0.94, the σ = 0.9625 fold collapses to
0.24–0.44 on every leg. Read against the synthetic gate-design fact
(a smooth one-parameter curve reads share ≈ 0.98 at 7 bins): these
numbers are a *strong* negative — the axis field carries real
bin-specific components outside any fixed 2-plane, concentrated
toward the endpoint region (coherent with 19.2's σ→1 tail being
different physics). Yet SLERP's success (below) shows each bin *is*
nearly in the span of its two neighbors — the field is **locally
2-D with a rolling plane**, not planar. Note the per-fold fit-Gram
top-2 shares (0.72–0.82) sit well below E24's pooled 0.92: the
pooled number includes the fit conditions' own overfit and excluded
the endpoint bins; the LOO read is the honest version.

### 27.2 — LAW-WORSE: SLERP wins everywhere (B: 12/12 targets)

Median |cos| at held-out targets, B legs: **SLERP 0.961, copy 0.850,
L-log 0.733, L-lin 0.773** — median Δ(primary) = **−0.166**, law win
share 0.0 (C: −0.102, win share 2/12 — both at the σ = 0.9625 fold
where SLERP itself degrades to 0.57–0.59). Both pre-named variants
lose; the θ-fits are fold-unstable (the L-log coefficients flip sign
between folds), i.e. the in-plane angle is not affine in ln σ or σ.
Per the defect note's interpretation: for a true fixed-plane geodesic
the expected outcome was *parity*; LAW-WORSE is therefore a positive
refutation — the misfit is real geometry (imperfect plane + non-affine
rate), not scoring noise. **Q7's claim gate is not met; the
manuscript keeps "rotates smoothly" as a description** (no-placeholder
rule executes).

### 27.3 — R̂ transfer: LAW-WORSE; the E25a lookup shape is settled

Same pattern on the residual direction (median: SLERP 0.855, copy
0.802, L-log 0.719; Δ = −0.176, win share 0.0). E25a, if ever frozen,
stays **measure-every-bin + neighbor (SLERP) interpolation** — now
with its quality quantified: neighbor interpolation reconstructs
held-out R̂ at |cos| ≈ 0.74–1.01 across interior verdict bins. The
768/σ = 0.4333 reliability hole keeps only the 896 read at that fold
(SLERP 0.855) — neighbor coverage of the hole is *plausible but
route-crossed*; record, don't over-claim.

### 27.4 — NULL: the phase component is not the rotation generator

V_phase = (C − C_pi)⊥ is **the most reproducible direction in this
experiment** (rel_cos_Vp 0.87–0.97 on 768, 0.83–0.89 on 896 — higher
than either leg) and yet: |cos(V̂p, t̂_C)| = 0.165 / 0.039 / 0.339 and
|cos(V̂p, Ĉ)| = 0.154 / 0.389 / 0.036 at 768 σ = 0.7 / 0.8333 /
0.9625 (medians 0.165 / 0.154 → NULL), with **in-plane share 0.01–0.09
in the full-data C plane**. So the phase-borne share of the
cancellation (E19.4: cos(C, C_pi) 0.50–0.67 — a large within-bin
rotation) points in a direction essentially orthogonal to the entire
axis-field geometry. Q7 candidate 1 dies at its foothold: the
σ-rotation of the axis is **not** phase-generated. For paper 2 this
is a sharp new fact, not merely a null — the cancellation axis and
the phase-mediated component are *separate directions*, so the
mechanism bridge must carry them as distinct objects (896 mirrors the
768 read where readable; 896/σ = 0.9625 unreadable — its tangent's
σ = 1.0 flank fails the C gate, as anticipated).

### Honest notes

- The fit set froze endpoint conditions (σ = 1.0, gated) as
  fit-eligible; the global θ-fits are partly dragged by the endpoint
  region where the field departs the mid-σ plane. An
  endpoints-excluded law was **not** computed (it would be post-hoc
  re-thresholding); if paper 2 wants it, it needs its own
  pre-registration.
- σ = 0.9625 is simultaneously the most *reliable* region (rel 0.87–0.95)
  and the least *plane-shared* — reliability and geometric regularity
  are different axes; don't conflate them when reading `e27_loo.png`.
- All numbers are pooled directions at this operating point
  (`anima_soup_sincos`); nothing per-sample, nothing objective-side.

## Cost ladder (planned → actual)

| item | planned | actual |
|---|---|---|
| full read | ~10 min CPU, ≤ ~30 GB | 567 s, 44-vector Gram (validation synthetics add ~20 s) |
| GPU | none | none |
