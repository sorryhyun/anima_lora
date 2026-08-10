# revision_plan — reframe around the cancellation geometry

Written 2026-08-09. This dir is a copy of `paper_suggestion/` (the
2026-07-31 "two accounts head-to-head" draft, frozen there as
draft-of-record; pre-rewrite v1 archived at
`_archive/paper/sigma_lowres_v1/`). The v1 restructure plan is
discharged and frozen at `../record/paper_plan.md` — this plan is its
successor and covers only the delta: what the E19–E25.0 vector-ledger
results change about the manuscript, and the one new run (E26) the
reframe owes.

**Amended 2026-08-09 (same day)** after the derivation gate check
(§9): the reframe is upgraded from "geometry as a third act appended
after the head-to-head" to **cancellation-first** — the B/C
near-cancellation moves into the theory section as the organizing
fact, the two-term account is restated as an *effective* law on the
cancellation residual, and the ledger observations sit between the
map and the scored section. What is **not** upgraded: the additive
account keeps the paper's predictive lane (E20.1 PARTIAL), and no
text may claim the account is *derived* from the geometry
(E20.4 NEGATIVE). See §9 for the record.

**Amended 2026-08-10** after the E28 gate-2 diagnosis; **rewritten
wholesale later the same day** once the E28 768-only read landed and
the E26 full-grid freeze was executed (commits `2b935563`/`9d8bda69`
→ `29f7a0ce`/`fa55b2a4`; memory `project_crossboot_arm_store_break`).
This block replaces the morning's version rather than layering on it.
The facts, and what each does to this plan:

- **The cross-environment break.** A same-protocol native rerun
  agrees with the committed e193/e194 stores at only **B 0.32 /
  C 0.41 / ĝ 0.47** across the 2026-08-09 reboot pair (fresh inductor
  autotune; adapter/models/caches/torch/driver all unchanged), vs
  0.96–1.02 within one environment. Committed cross-boot *vector*
  reads are dead; within-boot reads and scalar rows (drift
  Δρ ≈ 0.03–0.05, ΔG ≈ 27 %) are unaffected/tolerable.
- **§2.4 / §3 scope**: the axis-field rung's "across stores (≈ 1
  debiased)" claim gains an explicit **same-numerical-environment
  qualifier** — e193/e194 shared a boot; the pooled directions carry a
  large kernel-path-dependent component. Wording hook already in the
  draft: the instrument sentence "finite-draw cosines are
  compiler-kernel-path sensitive, so all pairings are computed within
  one run" (`sec_experiments.tex` §4.1) extends to pooled cross-store
  reads. The qualifier lands next to the 768/σ = 0.4333 reliability
  hole (§8 honesty-guard list, same sentence family).
- **§5 E26 reference**: comparisons against *committed* sincos rows
  are now cross-environment reads. The realized fix is **stronger**
  than the morning sketch's "in-run re-reference cells": the E28
  native seed-twin store (`20260810-0658-e28-native-twin-768` — same
  boot, same 5-bin grid, same seed) **is** the in-session sincos
  reference for the E26 grid; committed e193/e221 rows are context,
  never verdict denominators. §5 below is rewritten to match the
  frozen amendment.
- **E28 itself ran** (registered as deferred-to-paper-2; explicit
  user GO same day): 768 stage + native twin + read are DONE — 28-A
  READABLE, **28-B PARTIAL**: pinning the adaln frame neither stops
  the rotation (SHARED-AXIS fails, median |cos| 0.483) nor preserves
  it (matched-pair median |Δcos| 0.217 vs the twin); under the pinned
  frame the field reorganizes into two internally-coherent σ-blocks
  split at the conditioning value. 28-C(i): the cancellation survives
  the pinned frame at every bin, systematically shallower
  (ρ −0.68…−0.86 vs twin −0.84…−0.92). 896 completion undecided.
  **None of this enters this revision's tex**: the registration's "In
  the paper: Nothing in this revision" stands, and the §8
  no-mechanism-language guard is now *adjudicated* (the
  conditioning-frame "strong version" is refuted at this operating
  point/route), not merely precautionary.

**Status: PLAN — no reframe tex edited yet.** Order of work in §7;
nothing in §2–§6 is licensed to land in the tex before the §7 step
that carries it. §7 step 1 (E26 — flat REPLICATES, dirty REPLICATES,
§5 resolution block) and step 2 (survey pass — placement note in §10)
DONE 2026-08-10; next actionable step is 3 (§3.3/§3.4 rewrite). **One out-of-plan tex fix applied 2026-08-10** (external
review comment; wording only, numbers untouched, application numbers
stay verbatim per §4): the −14.6 %/−15.1 % savings (E4-measured on
the `sigma896` arm — 1024→896 above the one-sided σ > 0.5 gate,
no 768-window contribution) were attributed by "the window(s) the map
validates" phrasing in the abstract (main.tex), intro contribution 3
(sec_intro.tex), and the conclusion (sec_discussion.tex, which also
juxtaposed "stackable 768 window" directly before the number). All
three now name the 896 route + one-sided gate as the arm; the 768
window is stated as stacking onto the *late* rule at no endpoint
cost. The §4.3 use ("the always-gated arm realizes −14.6 %") was
already correctly attributed and is unchanged. Step 9's
hedge-consistency pass should re-verify these three sentences.

---

## 1. Direction in one paragraph

The current draft's centerpiece is a *scalar* account: the demoted
gradient's deviation reduces to a ratio-governed decaying term plus a
σ-independent floor "carried by the compute graph," and this account
beats the spectral account head-to-head. Everything since the freeze
showed what that deviation *is*: at every measured (route, σ) the data
branch B and graph branch C are individually far larger than the
realized gap and strongly anti-aligned (ρ ≈ −0.63…−0.96, E14), the
anti-alignment is Jᵀ-born, global, and operating-point invariant with
ρ̄ ≈ −0.91 licensed as one constant (E19), the cancellation is locally
enforced (E21), and the residual R = B + C is a σ-indexed axis field —
one direction per σ, shared across routes and stores, rotating
smoothly, with the miss angle-borne (E24/E24.2/E25.0). The frozen
draft carries this upside down: §3.4's reduction rests on assumption
(ii) "interaction negligible," which the committed ledger violates
maximally at every condition (S+F overshoots the realized gap 3–10×
grid-wide, §9), and §4.5 concedes the 1024→768 window dip as "a dip
the two-term reduction does not predict" when the crossing mechanism
is measured (E9/E14) and reproduces it natively
(`fig_accounts_canc.py`). The reframe puts the cancellation first:
the theory section presents the deviation as the residual of a
near-cancellation and *anticipates* the dip; the two-term account is
introduced as the effective law of that residual (its fitted
amplitudes absorb the cancellation — this is stated, not hidden); the
ledger observations (cancellation + axis field) land between the map
and the scored section; the scored section keeps the additive
account's Fig.-1 lane verbatim and reports the cancellation-aware
form's lane test honestly (E20: wins 768 with the dip, loses
LOO-896). The two-accounts spine is kept; E5's held-out voice
transfers verbatim; presentation order is not discovery order and the
text must never imply the account was historically derived from the
geometry.

## 2. Claim ladder (what the reframed paper asserts, in order)

Carried claims are **not re-litigated** — their voice, hedges, and the
claims ledger transfer verbatim (§4).

1. *Metrology* (carried): debiased estimator, self-floors, draw-count
   extrapolation, ε\* certification resolution.
2. *Cancellation* (NEW position — moves ahead of the scalar account):
   - B/C decomposition is exact in shared adapter-parameter space;
     cross-set debiasing removes draw noise (`vector_ledger.py`,
     E9/E10/E14 machinery — promoted from appendix to main text).
   - At every measured (route, σ) the branches are individually large
     and strongly anti-aligned; the measured gap curve is the
     **residual of a near-cancellation** (E14). Magnitude reads via
     the exact counterfactual angles h(·) only — the S/F/I quadratic
     is sign/decomposition/localization outside its truncation domain
     (E9 unit-honesty rule, carried into the manuscript).
   - The anti-alignment is Jᵀ-born, depth/type-uniform, magnitude ∝
     branch energy, partially RoPE-phase-mediated at noise-dominated
     bins, and operating-point clean; ρ̄ = −0.91 cited only with its
     license (E14 source; E19.0/19.3/19.6 uniformity).
   - The σ ≈ 0.4–0.7 cliffs and the 768 window are the
     |B⊥|/|C⊥| = 1 crossings (E9); the in-window excess sitting
     *below* the route's own floor is the crossing signature, stated
     in the theory and pointed to in the map — no longer conceded as
     unpredicted.
3. *Effective scalar account* (carried, re-voiced): the two-term form
   is the effective law of the cancellation residual — a route
   amplitude on the measured mismatch curve plus a per-route floor,
   with the fitted amplitudes absorbing the cancellation. E5
   predictive voice verbatim ("predicts routes it was not fit on at
   ~0.09 RMSE"); head-to-head win over the spectral account; the
   spectral account's miss now has a mechanism (it transports
   amplitude without the interference structure). The
   cancellation-aware operational form is reported next to it at lane
   parity: E20.1 PARTIAL (wins 768 held-out with the dip, 0.075 vs
   0.093; loses LOO-896; 1280-tier guard fails) — reported, not
   adopted; the additive lane stands.
4. *Geometry of the residual* (the reframe's fine structure):
   - The cancellation is locally enforced, with adaln carrying
     86–87 % of the phase-response amplitude (E21).
   - The residual is a σ-indexed **axis field**: one direction per σ
     across routes (median |cos| 0.95) and stores (≈ 1 debiased),
     rotating smoothly in σ (E24; E25.0 R̂ descriptive tables).
   - The residual is angle-borne (E24.2 knob read).
   - The rotation is **matched-angle, not planar**: transporting by
     the anchor's rotation buys nothing (E25.0-2 NO-GAIN — the
     "co-rotates with the σ-conditioned native gradient" intuition is
     retired; say so in the text, it is a natural reviewer guess).
   - Reliability scope: the pooled residual direction exists at 11/12
     verdict conditions with a recorded hole at 768/σ = 0.4333 and
     softness at mid-window 768 generally (E25.0-1 PARTIAL).
5. *Closures* (one honest subsection): deriving the account's data
   term from the geometry is closed at the estimand level (E20.4);
   re-deriving a ledger term as an objective correction likewise;
   damping-form levers are dead at probe level (E23.0 DAMP-DEAD); a
   fixed subspace is refuted at lever level — any projector must be
   per-σ-bin (E23.0-C PER-BIN-ONLY); per-sample exploitation stays
   gated (E22 → E23a). These are results, not future work.
6. *Application* (carried): the validated-window routing, −14.6 % wall
   at fixed steps.
7. *Outlook* (ONE paragraph, no promise): population-level σ-local
   levers (E25a restricted / E25b sketch) are licensed-but-unrun; the
   paper does not depend on them.

## 3. The scope sentences (non-negotiable)

Breadth now has three rungs, and the draft must keep them distinct
where each claim is introduced *and* in the abstract's hedge:

- **Map claims** (broadest): the full debiased campaign, both tiers,
  the coverage the frozen draft already states.
- **Cancellation claims**: every (route, σ) of the operating adapter
  (E14 grid), with leg-level operating-point invariance (E19.6), the
  E7 map-level 2×2 null, and the E26 full grid (flat REPLICATES,
  dirty REPLICATES) as the cross-checkpoint evidence. **Final stated
  breadth (E26 §5, resolved 2026-08-10): three adapters of this base
  model at 768 across the window — same-base qualifier stays, 896 an
  open cell.**
- **Axis-field claims** (narrowest): one adapter
  (`anima_soup_sincos`), three vector stores, standard corpus, the
  E24 debias conventions, 11/12 conditions.

Conflating rungs invites exactly the R1-style estimator-confound
review the line already survived once. E26 has landed
(REPLICATES/REPLICATES): the cancellation rung takes the widened
wording above at §7 step 7; the axis-field rung stays at its
one-adapter scope regardless (the grid pre-registered no vector
reads, and the smoke's cross-adapter direction read was
frame-confounded).

## 4. Carries over verbatim (do-not-touch list)

- The instrument + debiasing subsection and the "debiased units only
  from here" convention.
- The (route, σ) map numbers, boundary scoring, endpoint ≡ x-zero ≡
  α-flat. **One scheduled exception**: the §4.5 dip sentence ("a dip
  the two-term reduction does not predict") flips to the crossing
  read (§7 step 6) — the numbers around it do not move.
- The E5 held-out + three-form refit material and its qualified
  voice. Its *position* may move with the scored section; its words
  do not.
- The claims-ledger table (extended with the §2.2–§2.5 rows, existing
  rows unedited).
- The head-to-head structure and the `accounts_headtohead` figure —
  the additive lane stays Fig. 1 (E20's recorded decision); the
  cancellation-aware lane is text + appendix material, not a Fig. 1
  replacement.
- The application section and its numbers.
- All appendix raw/historical tables.

Supersession note: E19/E20 records earmark the cancellation account
as "the theoretical spine of paper 2." This plan supersedes that
earmark for the *narrative* (the cancellation moves into this
paper's theory and observations); the *operational canc lane as
predictive account* remains unadopted per E20.1 — that part of the
earmark stands.

## 5. E26 — cross-adapter cancellation check (frozen & running)

**RESOLVED 2026-08-10 (after the freeze below ran)**: full grid DONE —
**flat REPLICATES, dirty REPLICATES** (5/5 bins readable and passing
on both; read + tables in `../paper_bench/experiments/e26/`
`e26_grid_read.{py,json}` and README "Full-grid results"). §7 step 7
is unblocked: the scope sentence takes the REPLICATES wording (breadth
= three adapters of this base at 768 across the window; same-base
qualifier per the frame-confound limitation; 896 an open cell). The
depth-scaling upgrade is NOT claimed (identity-consistency column
lands in the arithmetic branch at 9/10 adapter-bins; sole exception
dirty σ = 0.7, recorded). dirty's first submission crashed on a full
disk and was resubmitted bit-identically in the same boot — recorded
in the E26 README run record; same-environment status preserved.

**Question**: does the B–C cancellation geometry replicate on LoRA
adapters other than the line's operating point?
**Status: E26.0 smoke DONE 2026-08-09 — both adapters PASS. Full-grid
amendment FROZEN 2026-08-10 (before any grid gradient existed); both
grid runs daemon-submitted the same day** (`e26-grid-flat` /
`e26-grid-dirty`). The pre-registration, frozen thresholds, protocol,
and results are the authority at
`../paper_bench/experiments/e26/README.md` (+ `e260_smoke.json`);
this section keeps only what the *manuscript plan* needs. This
section was rewritten 2026-08-10 to match the executed freeze — the
2026-08-09 "economize sketch" (768 σ-dense + 896 verdict bins,
~2.9 GPU-h) it replaced is superseded and recorded only in the E26
README's deviation note.

- **Adapters**: the preserved E7 pair (`output/paper/e7/`, verbatim
  shipped recipe, designed style axis, zero training cost; seed
  siblings = pre-declared extension tier, not run without an
  amendment). E7's probe *runs* are not reusable (no `arm_sums/`).
- **Frozen grid = 768-only, the E28-twin window**: bins {0.3, 0.4333,
  0.5667, 0.7, 0.8333}, no endpoint, 40 images, D = 12 — the exact
  grid of `20260810-0658-e28-native-twin-768`; measured cost
  3.8 GPU-h per adapter. The sketch's "896 at the verdict bins" is
  **dropped** (deviation recorded in the freeze): one probe run
  cannot carry per-route bin sets, a separate 896 run forfeits the
  shared native/reenc arms (≈ full-grid cost), and after the
  environment amendment its cells would need their own same-boot
  sincos reference (the twin covers 768 only). 768 carries the
  paper's window/dip claims and the larger residual; **896
  replication stays an open cell**, stated as such in the limitation
  paragraph next to the model-vs-adapter frame confound.
- **In-session sincos reference = the E28 native twin store** (same
  boot, same grid, same estimand path). Committed e193/e221 rows are
  context, never verdict denominators. If a reboot interleaves the
  queue: the scalar verdict stands (margins far above the measured
  cross-environment scalar drift), but any vector read (none
  pre-registered) is void.
- **Frozen verdict (per adapter, over the 5 bins)**: a bin is
  *readable* iff rel_cos_B ≥ 0.5 ∧ rel_cos_C ≥ 0.5; < 3 readable ⇒
  INCONCLUSIVE (remedy = the smoke's single pre-declared D = 24
  top-up, nothing else). **REPLICATES** iff ≥ 4/5 readable AND at
  every readable bin I < 0, ρ ≤ −0.5, h(B+C) < min(h(B), h(C));
  **PARTIAL** iff the criteria hold at ≥ 3 readable bins including
  σ = 0.7; **FAILS** otherwise. Scope-sentence consequences
  unchanged: REPLICATES → the cancellation-rung scope widens to
  "N adapters, one model"; PARTIAL → one-operating-point scope kept,
  pattern reported; FAILS → scoped to the operating adapter,
  limitation paragraph. **No outcome removes the cancellation or
  geometry material from the paper** — E26 only sets stated breadth,
  and the smoke's PASS already bounds the worst case at "replicates
  at the favorable bin, pattern elsewhere reported as measured."
- **Identity-consistency column** (pre-registered, carried into the
  freeze): per bin, ρ_implied from the measured h-triple next to
  measured ρ. The smoke's "enforcement depth scales with the
  perturbation" candidate (ρ deepens sincos −0.890 → flat −0.939 →
  dirty −0.959 as the legs grow) is claimed **only if** the measured
  deepening exceeds what the identity already forces from the
  magnitudes; otherwise it is reported as the arithmetic consequence
  it is.
- **Frame-free cross-adapter axis estimand: DROPPED**, not
  pre-registered: the smoke showed raw-parameter cross-adapter axis
  cosines sit at the ĝ frame baseline (0.27–0.35 vs baseline
  0.37–0.39 — frames don't overlap), and an induced-ΔW/function-space
  estimand is new instrument work the paper does not need. The
  "property of the model vs of the adapter" question is stated open
  in the limitation paragraph.
- **Secondary deliverable** (unchanged, no verdict weight): localize
  E7's flat-good / dirty-bad floor ordering as amplitude vs direction
  — flat's h(B+C) ≈ sincos's (0.049 vs 0.044, legs 2×) while dirty's
  is ~8× (0.347), mirroring E7's checkpoint-dependent cos_floor.

## 6. Figure debts

- **NEW — cancellation panel** (now required, no longer
  verify-first): h(B), h(C), h(B+C) by σ per route — the gap curve
  visibly *below both branches* is the reframe's first figure;
  material exists in the E14/E19 ledgers
  (`e19/appendix.md` per-σ geometry figure, `fig_ledger_geometry.py`).
- **NEW — axis-field summary** (the fine-structure signature): the
  σ-binned R̂ cosine structure (across-route / across-store /
  across-σ) in one panel; candidate base: extend
  `fig_ledger_geometry.py`.
- `accounts_headtohead` unchanged as Fig. 1 (additive lane).
  The cancellation-aware lane figure (`e19/accounts_canc.png`, E20
  lane-matched numbers) goes to the appendix with the E20 PARTIAL
  read in the caption.
- Promote/adapt `e250_rel.png` + `e250_frame.png` (reliability + the
  matched-angle-not-planar read) — appendix at minimum.
- E26 replication panel (after §7 step 7).
- Carried figure debts from the v1 plan (Fig-1 enlargement, waterfall)
  remain owed and unchanged.

## 7. Order of work

1. **E26 amendment + runs + read — DONE 2026-08-10**: trimmed grid
   frozen, both runs landed (dirty via a same-boot bit-identical
   resubmit after a full-disk crash), read executed against the
   frozen criteria — **flat REPLICATES, dirty REPLICATES** (§5
   resolution block). Step 7 is unblocked and waits only on the tex
   steps before it.
2. Survey pass over `sec_theory.tex` / `sec_experiments.tex` /
   `sec_discussion.tex`: fix the insertion points (cancellation into
   §3.3/3.4, observations section between map and scored, closures
   subsection), and mark every sentence the assumption-(ii) rewrite
   invalidates (the §3.4 assumption block, the §4.5 dip concession,
   the limitations "reduction's domain" paragraph). No edits yet — a
   placement note appended to this file.
3. §3.3/§3.4 rewrite: B/C formalism promoted from appendix;
   cancellation stated as the measured organizing fact (ρ̄ with its
   E19 license); the four-term expansion kept as the bridge;
   assumption (ii) **deleted as an assumption** and replaced by the
   measured interference; the two-term form introduced as the
   effective law of the residual with the crossing/dip as its
   anticipated signature; unit-honesty rule stated (h(·) for
   magnitudes, S/F/I for sign/decomposition/localization).
4. Abstract + intro rewrite (question unchanged; the two-accounts
   framing keeps its role; "carried by the compute graph itself"
   replaced by the cancellation-residual object with the §3 rung
   hedges). Also carries the related-work citation additions from the
   2026-08-10 paper scan (§10 last subsection: ANT / DeMe / PMA).
5. New observations section (instrument-adjacent placement per step
   2): cancellation panels (E14, h-units) → local enforcement (E21) →
   axis field (E24 → E25.0 order) → closures subsection (§2.5).
   **Gated option** (`questions.md` Q7): if the rotation-law read is
   frozen, run, and lands PASS before this step, one claim + one
   panel may be added under Q7's own pre-registration; otherwise
   "rotates smoothly" stays a description — no placeholder language.
   **[Resolved 2026-08-09 before this step: E27 ran and the verdict
   is negative (LAW-WORSE / PLANE-MIXED / anchor NULL) — the option
   does not fire. "Rotates smoothly" stays, now as an *adjudicated*
   description; the sentence may cite E27's SLERP numbers (held-out
   |cos| median 0.96) if a citation is wanted.]**
6. Scored-section edits: dip sentence flips to the crossing read;
   one lane-parity paragraph on the cancellation-aware form with the
   E20 numbers (wins 768 + dip, loses LOO-896, guard fail — not
   adopted); Fig. 1 and the E5 material untouched in wording.
7. E26 lands → cancellation-rung scope sentence finalized,
   replication panel added, claims ledger rows updated.
8. Discussion/limitations rework: the "reduction's domain" paragraph
   shrinks to what §3 now states up front; frame-confound limitation
   (model-vs-adapter open); outlook trim (the one lever paragraph).
9. Full-tex compile + hedge-consistency pass (reuse the "polish
   experiment notes" discipline: no stale hedges against later
   verdicts — in particular no surviving "does not predict" language
   about the dip, and no "derived" language about the account).

## 8. Honesty guards

- **No "derived."** The two-term account is an *effective* law; E20.4
  closed estimand-level derivation of its data term. Any sentence
  implying the account follows from the geometry is out.
- **The canc lane is not adopted.** E20.1 PARTIAL is reported at lane
  parity (both RMSEs, both losses) wherever the cancellation-aware
  form appears; the additive lane keeps Fig. 1.
- **Chronology guard.** Presentation order ≠ discovery order: the
  account was fitted and held-out-validated before the geometry
  existed; E5's voice transfers verbatim and the text never implies
  the geometry produced the account.
- **Unit honesty.** Manuscript magnitudes via h(·)/exact link only;
  S/F/I quadratic shares appear only as sign/decomposition/
  localization reads with the truncation-domain caveat (in-window
  S+F+I under-predicts h(B+C) by ~1.5–2.7×, §9).
- **ρ̄ citation license.** ρ̄ = −0.91 appears only with its E14 source
  and E19.0/19.3/19.6 uniformity license.
- No lever result is claimed or implied; E25a is unfrozen/restricted,
  E25b is a sketch. One outlook paragraph, flagged as such.
- The matched-angle result (E25.0-2) is reported even though it
  *weakens* the tidiest version of the geometry story — it is the
  pre-registered read and it scopes the lookup claim honestly.
- The 768/σ = 0.4333 reliability hole and mid-window 768 softness are
  reported next to the axis-field claim, not footnoted.
- E26 trimmed-grid thresholds were frozen (2026-08-10) before any
  grid gradient existed — discharged; no post-hoc widening of the
  adapter list; the depth-scaling upgrade stays gated on the
  identity-consistency column (§5).
- Per-sample language stays out of the paper ("pooled residual
  direction at this operating point" — the E25.0 wording).
- No rotation-*law* language unless `questions.md` Q7 is frozen, run,
  and PASS against its SLERP baseline before §7 step 5 — the measured
  claim is "rotates smoothly," nothing stronger. **[E27 ran and did
  NOT pass — this guard is now permanent for this revision: any
  rotation-law sentence that appears in review is a regression.]**
- No mechanism-bridge language either: E28 ran (768-only) and landed
  **28-B PARTIAL** — the conditioning-frame "strong version" is
  refuted at this operating point/route, and the surviving two-block
  structure is a paper-2 object awaiting its own registration.
  Nothing from E28 enters this revision's tex (per its registration's
  "In the paper: Nothing in this revision"); a mechanism sentence
  appearing in review is a regression, same as the rotation-law case.

## 9. Gate check record (2026-08-09)

Question posed before amending this plan: can the two-term account be
*derived* from the B/C cancellation, and should the manuscript be
restructured cancellation-first? Checked against committed data only
(no new runs): the E14 probe-matched ledger
(`runs/20260801-2304-e14-ledger-probematched/ledger{,_native}.json`),
the E19/E20 records, and `fig_accounts_canc.py`.

- **Grid re-read (reenc ref, matches the paper's excess convention)**:
  ρ ∈ [−0.96, −0.63] at every non-degenerate (route, σ) on all three
  1024-tier routes. S+F overshoots the realized gap h(B+C) by 3–10×
  grid-wide (768 in-window: S+F 0.10–0.20 vs h(B+C) 0.016–0.035).
  S+F+I equals the quadratic prediction identically
  (I ≡ 2√(SF)·ρ, so (√S−√F)² + 2√(SF)(1+ρ) is a rearrangement, not
  a new form) and under-predicts h(B+C) in-window by ~1.5–2.7×
  (truncation) — magnitudes must be read via h(·) (E9 rule).
- **Already adjudicated by the line** (nothing new to run):
  feasibility of the cancellation-aware link — `fig_accounts_canc.py`
  (in-sample, dips below its own floor natively); its license —
  E19.0/19.3/19.6; its lane test — E20.1 **PARTIAL** (768 held-out:
  canc 0.0753 vs additive 0.0931, dip in-window; LOO-896: canc
  0.0819 vs additive 0.0732; 1280-tier guard fails on 1024; "geometry
  right, amplitude law open; no 20.3 spend"); derivation of the data
  term — E20.4 **NEGATIVE** (estimand-level).
- **Verdict**: cancellation-first restructure LICENSED as narrative
  (assumption (ii) is untenable against committed data and the dip
  concession is unnecessary); derivation claim REFUSED (E20.4);
  account-lane swap REFUSED (E20.1). The restructure itself needs no
  new GPU runs; E26 remains the only owed run.

## 10. Placement note (§7 step 2 — survey pass, 2026-08-10)

Anchors are file:line at commit `9d8bda69`. No tex edited.

### sec_theory.tex

- **§3.3 `sec:ouraccount` (161–234)**: the B/C decomposition
  (Eq. `eq:branches`, 201–207) is already main-text — step 3's
  "promoted from appendix" is partly discharged; what is missing is
  the *measured* cancellation. Insertion: new paragraph after 217
  ("…sets the rest to zero.") stating (a) both branches individually
  large, (b) strong anti-alignment with ρ̄ = −0.91 cited under its
  E14 + E19.0/19.3/19.6 license, (c) the |B⊥|/|C⊥| = 1 crossing as
  the anticipated in-window signature (forward-ref to the
  observations subsection).
- **§3.4 assumption (ii) (284–290, item + footnote)**: DELETE as an
  assumption. Keep the label slot — (ii) becomes the *measured
  interference* entry (large, negative, in-window; carried as
  measurement, not assumption) so the cross-reference
  "assumption~(iv)" (sec_experiments 230) and the (i)/(iii)/(iv)
  labels survive unrenumbered. "Under (i)–(iv)" (305) re-voices to
  "under (i), (iii), (iv), with the interference as measured".
- **Two-term reduction intro (305–336)**: re-voice as the effective
  law of the cancellation residual (fitted amplitudes absorb the
  cancellation — stated). The dip-anticipation sentence lands here.

### sec_experiments.tex

- **Observations section inserts between 187 (end of §4.2 map) and
  189 (`\subsection{Both accounts, scored}`)**: order per §7 step 5 —
  cancellation panels (E14, h-units) → local enforcement (E21) →
  axis field (E24 → E25.0, with the matched-angle retirement
  sentence) → closures (§2.5). Scope sentences per §3 rungs, with
  the NEW same-environment qualifier (2026-08-10 amendment) attached
  to the across-store sentence and the 768/σ = 0.4333 hole beside it.
- **Instrument hook (121)**: "computed within one run" sentence
  extends to pooled cross-store reads (same-environment qualifier).
- **Dip sentence (174–176)**: "a dip the two-term reduction does not
  predict (the vector-resolved probe traces it to negative
  data–graph interference)" → flips to the crossing read (§7 step 6);
  surrounding numbers untouched.
- **Lane-parity paragraph (E20.1)**: lands at the end of §4.3 after
  the governors paragraph (276–303), with both RMSEs and both losses.
- "assumption~(iv)" (230–231) survives unedited under the
  keep-labels decision above.

### sec_discussion.tex

- **"Account resolution and domain" (105–115)**: step 8 shrink — the
  "reduction's domain excludes the 768 mid-σ window … still does not
  predict" clause reduces to a pointer at §3's up-front statement.
- **One-operating-point paragraph (87–98)**: E26 scope sentence +
  frame-confound (model-vs-adapter open) land here after §7 step 7.
- **"Instrument resolution and coverage" (122–142)**: add the
  kernel-path/same-environment instrument sentence (one line, cites
  the within-one-run discipline already stated in §4.1).
- **Conclusion (144–170)**: step 9 hedge pass only; the two-term
  voice stays additive-lane.

### sec_intro.tex + main.tex

- **Intro 43–47** ("We therefore decompose … into two terms"): step 4
  — one added sentence presenting the decomposition as the effective
  law of a measured near-cancellation; contribution 1 (52–74) gains
  the geometry clause with the rung hedge.
- **Abstract (main.tex 57–62)**: "carried by the compute graph
  itself" → the cancellation-residual object with the §3 rung hedges
  (step 4). Title, contributions structure, and the 14.6 % sentence
  untouched.

### Invalidated-sentence inventory (assumption-(ii) rewrite)

1. sec_theory 284–290 — the (ii) item + footnote (replaced, see above).
2. sec_theory 305 — "Under (i)–(iv)" (re-voiced).
3. sec_experiments 174–176 — the dip concession (flips to crossing).
4. sec_discussion 110–114 — the domain-exclusion clause (shrinks).
No other sentence in the four body files asserts (ii) or leans on it.

### Related-work additions (2026-08-10 paper scan; lands with §7 step 4)

Three papers read in full this date (PDFs + notes in session
scratchpad); each gets a citation, none gets a mechanism sentence.

- **ANT** (Go et al., NeurIPS 2023, arXiv:2306.00354) and **DeMe**
  (Ma et al., CVPR 2025, arXiv:2410.06664) — the σ-indexed
  gradient-*direction* literature: per-timestep task affinity
  (pairwise cosine of per-timestep loss gradients) decays smoothly
  with noise-level gap (ANT O1) and goes negative between distant
  timesteps (DeMe Fig. 1a); interval clustering / decouple-then-merge
  as remedies. Placement: a NEW fourth paragraph in
  `sec_intro.tex` §Related work (after "Positional extension",
  129–138), ~3 sentences, "noise levels as tasks" framing; the
  observations section's "rotates smoothly" sentence (§7 step 5) MAY
  cite them as the raw-gradient precedent for smooth σ-indexed
  direction structure. **Delta to state**: they measure conflict
  *between* per-timestep full gradients; our object is the
  product-rule split of one gradient's *change* under input
  downscaling — the near-cancellation has no counterpart there.
  **Guard**: framing citations only. The "conditioning frame smooths
  the intrinsic task-block structure" reading of 28-B PARTIAL that
  these papers suggest is paper-2 material under E28's follow-up
  registration — it does not enter this revision's tex (§8
  no-mechanism-language guard applies verbatim).
- **PMA** (Wu et al., arXiv:2511.19778, "Phase-Aligned RoPE for
  Mixed-Resolution DiT") — measures the content-free RoPE attention
  bias κ(∆) (sinusoidal phase filter, timestep-stable), shows PI-style
  position rescaling fails because rescaled distances sample the
  filter at wrong phases, and ranks PI worst < NTK < YaRN < native at
  the sample level — the forward-pass twin of G11's off-manifold
  verdict and the yarnsig probe ordering (gap_896pi > gap_896yarn,
  gate unchanged). Placement: appended to the "Positional extension"
  paragraph (129–138), 1–2 sentences. **Delta to state**: PMA is
  inference-time, mixed-resolution *within one attention op*,
  attributed at the attention-score/sample level; our substitution is
  train-time whole-grid (no mixed-scale attention arises) and the
  attribution is at the training-gradient level — the floor
  decomposition (RoPE_e + Resid_e) remains ours. Their
  timestep-stability of κ(∆) is *consistent* with the floor's
  σ-independence but that connection is discussion-grade at most —
  same no-mechanism guard.
- `references.bib` keys to add with step 4: `go2023ant`,
  `ma2025deme`, `wu2025phasealigned` (verify none already present
  under other keys before adding).
