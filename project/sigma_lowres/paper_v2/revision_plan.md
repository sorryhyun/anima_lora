# revision_plan — reframe around the cancellation geometry (RESOLVED)

Written 2026-08-09; **executed in full 2026-08-11** (§7 steps 3–9 of
the plan landed in the tex; the reframe draft compiles at 35 pp,
tectonic clean, no undefined refs). This file is condensed to the
resolved state: the verbatim plan — direction paragraph, claim
ladder, carry list, E26 §5 block, figure debts, order of work, the
2026-08-09 gate-check record (§9), and the §10 placement note — lives
at commit `c1534b00` and its predecessors (`351dd178`, `05deaab5`)
and is not restated here. What remains below is (1) the execution
record with its deviations, (2) the **standing constraints** that
bind any future edit of this tex — the scope rungs and the honesty
guards — and (3) the open debts.

Successor doc for anything runnable: `roadmap.md` (this dir); nothing
there is licensed to enter the tex without its own pre-registration.

## 1. Execution record (2026-08-11)

§7 step 1 (E26 — flat REPLICATES, dirty REPLICATES) and step 2
(survey pass / placement note) were DONE 2026-08-10. Steps 3–9
executed 2026-08-11 per the placement note. Deviations and
realizations, recorded:

- **Claims-ledger table (the §4 carry item): does not exist in the
  frozen draft** — the v1 plan's table was never realized in the tex,
  so "extended with the §2.2–§2.5 rows" was inapplicable. The E26
  rows landed instead as an appendix table (`tab:e26` appended to
  `app:e7` with the frozen criteria and the identity-consistency
  caveat); the E26 "replication panel" figure debt is discharged by
  that table, not a figure.
- **New figures** (assembly-only script
  `../paper_bench/fig_canc_axis.py`, committed digests, nothing
  refit): `figs/canc_panel.png` (h(B)/h(C)/h(B+C) per route + ρ
  strip, from the E14 ledger) = Fig. 2; `figs/axis_field.png`
  (family cosines + across-σ rotation, from `e24_axis.json` +
  `e250_read.json`) = Fig. 3. `accounts_canc.png` (E20 lane caption,
  demonstration-labeled) and `e250_rel/frame.png` promoted into a new
  appendix section `app:geometrypanels`.
- **Appendix (ii)-assumption spots re-voiced** (two places:
  `app:geometry` drop-bounds paragraph, `sec:headtohead` mechanism
  paragraph) — outside the placement note's body-file inventory but
  required for label consistency once (ii) became a measured entry.
  The label slots (i)/(iii)/(iv) are unchanged, so every
  `assumption (iv)` cross-reference survives.
- E27's SLERP number (held-out |cos| median 0.96) is cited in the
  observations section as licensed by §7 step 5's bracket; the
  rotation-law negative is stated as a disclaimer, not a claim.
- E20.1 lane numbers quoted are the endpoint-detached RMSE* values
  (`result.json` `_det`: 768 held-out 0.078 vs 0.099, LOO-896 0.086
  vs 0.077, 1024 guard 0.199 vs 0.046; the frozen gate-check block
  quoted `_full` — both committed).
- Step 9's hedge grep swept clean: no dip-concession, no "derived",
  no rotation-law claim, no E28/mechanism language, ρ̄ only with
  license, canc lane reported-not-adopted, the three −14.6 %
  attribution sentences intact (896 route + one-sided gate named in
  abstract, intro contribution 2, and conclusion).

## 2. Standing constraints on this tex (do not regress)

### Scope rungs (keep distinct wherever a claim is introduced, and in the abstract's hedge)

- **Map claims** (broadest): the full debiased campaign, both tiers.
- **Cancellation claims**: every (route, σ) of the operating adapter
  (E14 grid), leg-level operating-point invariance (E19.6), the E7
  map-level 2×2 null, and E26 — **three adapters of this base at 768
  across the window; same-base qualifier stays; 896 an open cell**.
  The depth-scaling upgrade is NOT claimed (identity-consistency:
  arithmetic at 9/10 adapter-bins; sole exception dirty σ = 0.7,
  recorded in `tab:e26`'s caption).
- **Axis-field claims** (narrowest): one adapter
  (`anima_soup_sincos`), three vector stores, standard corpus, 11/12
  conditions with the 768/σ = 0.4333 hole reported beside the claim.

### Honesty guards (a violation appearing in any future pass is a regression)

- **No "derived."** The two-term account is an *effective* law; E20.4
  closed estimand-level derivation of its data term.
- **The canc lane is not adopted.** E20.1 PARTIAL is reported at lane
  parity (both RMSEs, both losses) wherever the cancellation-aware
  form appears; the additive lane keeps Fig. 1.
- **Chronology guard.** Presentation order ≠ discovery order: the
  account was fitted and held-out-validated before the geometry
  existed; E5's voice transfers verbatim and the text never implies
  the geometry produced the account.
- **Unit honesty.** Magnitudes via h(·)/exact link only; S/Φ/I
  quadratic shares only as sign/decomposition/localization with the
  truncation-domain caveat (now stated in §3.4 of the tex).
- **ρ̄ citation license.** ρ̄ = −0.91 appears only with its E14 source
  and E19.0/19.3/19.6 uniformity license.
- No lever result claimed or implied; the one Outlook paragraph is
  flagged as such and the paper does not depend on it.
- The matched-angle result (E25.0-2) stays reported even though it
  weakens the tidiest geometry story.
- Per-sample language stays out ("pooled residual direction at this
  operating point").
- **No rotation-law language** (E27 ran and did not pass — permanent
  for this revision); "rotates smoothly" is the ceiling, optionally
  with the SLERP citation as a description.
- **No mechanism-bridge / E28 language** (28-B PARTIAL refuted the
  strong conditioning-frame version; the two-block object is paper-2
  material under its own registration — F1 resolved it
  MISMATCH-CARRIED, see `roadmap.md` §1(b)).
- **Same-environment qualifier.** Cross-store vector reads are
  licensed only within one numerical environment (the 2026-08-10
  cross-boot break); the qualifier lives in §4.1's instrument
  sentence, the observations section, and the limitations — keep all
  three.

## 3. Open debts

- Carried v1 figure debts: Fig-1 enlargement, waterfall.
- 896-route cancellation replication: stated open cell (a run would
  need its own registration under `../paper_bench/experiments/`).
- The upper-tail (σ > 0.94) sweep remains the designated probe named
  in the limitations.
