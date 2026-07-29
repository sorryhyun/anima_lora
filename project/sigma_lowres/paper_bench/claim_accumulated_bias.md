# Claim — the in-band verdict is valence-blind; the residual risk is accumulated sub-band bias

*2026-07-29, out of a discussion of whether the reenc control could be
"biased toward a beneficial direction". Status: acknowledged limitation,
closed by E4; no instrument change required.*

## The claim

An in-band read (Δ_e = d_e − d_reenc ≤ ε*) certifies **per-step,
first-order gradient equivalence at instrument resolution** — it does
not by itself certify training-outcome equivalence. The cosine gap
measures the *magnitude* of a gradient deviation, never its *valence*
(whether the deviation helps or harms learning). The gap between the two
readings is exactly what E4 exists to close.

## Why the "reenc is beneficially biased" form of the worry collapses

1. **There is no measurable reenc direction to have a valence.**
   `gap_reenc ≈ 0` with split-half reliability indistinguishable from
   zero (E7-flat: bin means +0.013/+0.004/+0.010/−0.002, split-half
   −0.25): the reenc gradient matches the native population gradient as
   well as a second native redraw does. The criterion therefore reduces
   by transitivity to "demoted gradient ≈ **native** gradient", and
   native is the baseline by definition — there is no separate reenc
   direction whose valence could mislead.
2. **A benign reenc makes the test stricter, not looser.** The control
   is a subtracted credit: cheap reenc → small d_reenc → small
   allowance. The dangerous direction for the criterion would be an
   unusually *expensive* control (inflated allowance); empirically
   d_reenc sits at the floor, so the test operates near its most
   stringent point (allowance ≈ 0, band set by ε* alone).

## The form of the worry that survives: accumulated sub-band bias

- **Systematic, not stochastic.** Demotion error is structured — it
  always removes the same super-Nyquist band. Redraw noise averages out
  over steps; a *consistent* sub-ε* directional bias (cos ≈ 0.99, same
  direction every step) can accumulate over ~10⁴ steps into an outcome
  effect the per-bin instrument cannot see. In-band ≠ zero integrated
  effect.
- **Direction match ≠ trajectory match.** The probe reads directions;
  `gnorm` differs across arms (gnorm_896 ≠ gnorm_native), so Adam's
  second-moment state — and hence the realized parameter trajectory —
  can diverge even under exact per-bin direction match.
- **Per-bin equivalence composes only to first order.** Equivalence at
  each σ separately does not guarantee equivalence of the full schedule
  under curvature + optimizer state.

## Consequence for the paper

Every efficiency number stays "a projected ceiling" until E4 lands.
E4's three arms answer this claim in both directions: if sub-band
accumulation is real, the σ-conditional arm degrades vs native
(instrument too lenient); if the unconditional-768 negative control does
*not* degrade, the instrument is too conservative. Either outcome is
informative; only "conditional ≈ native ≠ 768-unconditional" validates
the map as an outcome-level safety criterion.

## Optional direct probe (deprioritized below E4)

A phase-scrambled-residual arm (add x_reenc − x_nat with randomized
phase: same spectrum and magnitude, random direction) would separate
"reenc is a generic small perturbation" (d_scram ≈ d_reenc) from "the
VAE round trip is a specially manifold-aligned perturbation"
(d_scram ≫ d_reenc). By §"collapses" above, neither outcome moves the
current verdicts — run only if reviewers press on the control's
neutrality.
