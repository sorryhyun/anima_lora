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

### Post-freeze amendment (2026-08-11, later same day): Eq. (5) made operational-exact

Trigger: the "small remainder" chain was indefensible as written —
the branch-level R_e (ΔJᵀΔr̄ + covariance-coupling difference) is
never separately measurable (the lattice's fourth corner does not
exist), a scale estimate via the measured relative residual mismatch
(0.89→0.36) puts the mixed term at ~half the C leg, and assumption
(i)'s "designated probe" promise had no probe in §4. Resolution:
define the branches operationally in the tex, matching what the E14
ledger always measured.

- `eq:branches` (5) rewritten as the telescoping identity
  B = ḡ_rp − ḡ_src, C = ḡ_dem − ḡ_rp — exact at every amplitude, no
  remainder; product-rule forms (J̄ᵀΔr̄, ΔJᵀr̄) demoted to the legs'
  leading-order *interpretation*; second-order coupling + covariance
  differences stated to ride inside the legs; fourth-corner
  uniqueness argument added (`\rp` macro added to main.tex).
- Assumption **(i)** re-purposed (label slot kept, so all
  (i)/(iii)/(iv) cross-references survive): now "geometric remainder,
  handled by domain" — R'_e carries only beyond-quadratic cosine
  geometry, controlled by the existing h(·)/exact-link unit rule; the
  untestable Cauchy–Schwarz footnote (ρ_e, never measured) removed.
  List intro re-voiced: "one measured term, one domain rule, and two
  assumptions". Reduction sentence now "Under (iii) and (iv), … and
  magnitudes read under (i)'s rule".
- `app:branches` re-derived (telescoping first, mean-factor expansion
  second, computability remark re-scoped); `app:geometry` four-term
  expansion now exact at the quadratic level (no R_e-bearing cross
  terms); drop-bounds paragraph keeps |I| ≤ 2√(SΦ), replaces the ρ_e
  bound with the domain statement tied to the recorded 3–4×
  truncation underprediction.
- §4.3 arm intro now cites the §3.3 repromote construction and the
  reenc-vs-src referencing ~4% agreement; `app:notation` arm list and
  the E14 ledger implementation paragraph tied to `eq:branches`.
- Also resolves the ρ (anti-alignment) vs ρ_e (remainder share)
  notation collision — ρ_e no longer exists.
- New §3.3 figure `fig:arms` (`figs/arms_schematic.png`, assembly-only
  script `../paper_bench/fig_arms_schematic.py`): the three arms as
  one concrete image walking the lattice path — per-panel arm
  constructions (𝓔(y) / 𝓔(upscale(𝓡_e(y))) / 𝓔(𝓡_e(y))), intervention
  arrows, token-grid glyphs. Layout: src→dem straight dashed = the
  demotion training actually does; rp hangs below it as the hidden
  waypoint (visibly pixelated — nearest-neighbor upscale drawn).
  Content is a crop of the model's own fig:kaaiyuki render (no new
  IP); downscale drawn at ratio 1/3 for visibility; caption states
  reenc pricing + re-referencing + the drawn-for-visibility caveats.
- Follow-up trim (same day): the body's leading-order ≈ claims
  (B_e ≈ J̄ᵀΔr̄, C_e ≈ ΔJᵀr̄) removed — with operational legs they
  were a fresh attack surface (the mixed term is ~half the C leg by
  the scale estimate above). §3.3 keeps only the physical
  characterization + the Δr̄(σ) definition (needed by (iv) /
  eq:twoterm); ΔJ now appears nowhere in the body (§3.4 and
  app:geometry control arguments re-voiced as "graph leg vanishes
  identically"); the mean-factor reading lives only in app:branches,
  hedged as interpretation.
- Sweep leftover found 2026-08-11 (post-amendment J-hedge audit):
  related work's ANT/DeMe contrast still called our object "the
  product-rule split of one gradient's change" — re-voiced to "the
  data--graph split" (sec_intro §2, matches the operational legs).
  Audit outcome otherwise: no stale J-hedges; remaining J mentions are
  §3.3 motivation (g = Jᵀr), the §4.3 Jᵀ-born *finding*, the spectral
  account's calibrated-gain sentence, and the deliberate appendix
  interpretation/computability remarks. Repromote prior-art check: no
  spectral-account precedent for the src/rp/dem gradient-level split
  (SwD/SPD claims live at the data/distribution level); closest
  relatives are blur-vs-decimate factorization (BlurPool) and
  upsampled-LR corruption designs (relay/cascaded) — positioning
  sentence NOT added (open option); wu2025phasealigned already cited
  and is not a repromote precedent.
- §3 J-notation consolidation (2026-08-11, user-directed): the
  g = Jᵀr product + noise-mean/fluctuation split + r̄_a definition
  moved out of §3.3 into a new §3.1 "Inside the mean" paragraph
  (single display, flagged "interpretive — no claim computes through
  it", pointing at app:branches); §3.3 now opens repromote-first,
  motivating the rp arm by the SRCNN-family pre-upsampling precedent
  (new bib entries dong2016srcnn, kim2016vdsr — the rp data state =
  the pre-upsampled-LR input of that literature). Body voice is B/C
  throughout; app:branches g_a-product ref repointed
  §3.3 → §3.1. Verified: A.4 gap derivation (exact link → four-term →
  excess → drop bounds) is J-free; A.4's only J is the "What is not
  derived" σ_max(J̄) envelope paragraph (honesty guard for (iv),
  kept). Compiles clean, no undefined refs.
- fig:arms caption reduced to concept-only ("Decomposing demoted
  training"): Eq.-5 ref, arm names/formulas, and the reenc-pricing
  sentence dropped (arm labels remain in the image; reenc pricing +
  re-referencing still stated in §3.1/§4.3); drawn-for-visibility
  caveats and render-content note kept.
- §3.3 ¶1–2 re-voiced problem-first (user-directed): ¶1 now states the
  non-attributability of δg under demoted training (both changes
  applied together) BEFORE introducing the SRCNN-precedent repromote
  state as the splitter; post-Eq.-5 defensive prose slimmed — "exact
  at every amplitude / noise-covariance couplings ride inside the
  legs / fourth corner / no cross-grid matrix product" all moved to
  appendix-only (app:branches unchanged, body keeps one exactness
  sentence: "an identity rather than an expansion" + rp-path
  uniqueness pointer, needed by §3.4's "legs being exact" reference).
  fig:ledgergeom body ref dropped (still referenced twice in
  appendix); "unlike any per-branch matrix product" clause trimmed.
- Two supporting sentences added (user-directed, same pass): §3.1
  "Inside the mean" now ends with the factor-coupling explanation
  (J moves with the data, r moves with the graph — each factor
  responds to both interventions; app:branches pointer), grounding
  §3.3's non-attributability opening; §3.2 limits list closes with
  the shared-root sentence — Eq. (7) models the data alone, so the
  graph contribution to a measured gap is neither expressible nor
  extractable by the spectral account (the attribution problem §3.3
  solves). Consistent with the existing "keeps only the
  input-mediated part of the data branch" read; no new claims.
- Mean-factor reading of the cancellation added (user-directed): the
  pre-amendment Jᵀr terms revived as *explanation of the
  anti-alignment phenomenon*, NOT as claim structure — body §3.3 gets
  a formula-free "why cancellation is natural" passage (factor
  coupling ⇒ compensation inside the product only when data+graph
  move together ⇒ waypoint mismatch response enters legs with
  opposite sign; grid-co-moving part survives as floor), detail in a
  new app:branches paragraph "The mean-factor reading of the
  near-cancellation" (J̄ᵀΔr̄ / ΔJᵀ leading terms, matched-corner
  product invariance ↔ tier-consistency of the base computation,
  factor-level restatement of app:instrument's round-trip geometry,
  consistency with the Jᵀ-born/uniform/phase-borne reads, explicit
  interpretation-not-derivation limits: identification-only ΔJ,
  mixed term not small, §4.3 closures bar estimand-level
  re-derivation). "ΔJ nowhere in the body" invariant preserved (body
  passage has no formulas). Honors No-"derived" + no-mechanism-bridge
  guards (no conditioning-frame/E28 content).
- §3.3 post-Eq.-5 two paragraphs compressed (user-directed): body now
  reads conjecture-first — identity + factor coupling ⇒ the branches
  can push in opposite directions and collide; the split's value is
  making that observable — then one short measured paragraph
  (qualitative near-cancellation + the ratio-through-1 collapse
  signature). Moved out of body: ρ definition + ρ̄ = −0.91 + pooling
  license summary (now §4.3-only — ρ inline-defined at its first
  §4.3 use, keeping the citation-license guard satisfied), the
  factor-coupling why-passage (app:branches has the full version),
  "only realizable split" uniqueness (app:branches). Kept in body:
  identity clause (needed by §3.4 "legs being exact"), Δr̄
  definition (needed by (iv)), crossing signature (referenced by
  §3.4 dip discussion + §4.3 "crossing geometry of §3.3").

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
