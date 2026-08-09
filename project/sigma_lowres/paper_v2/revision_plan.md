# revision_plan — reframe around the cancellation geometry

Written 2026-08-09. This dir is a copy of `paper_suggestion/` (the
2026-07-31 "two accounts head-to-head" draft, frozen there as
draft-of-record; pre-rewrite v1 archived at
`_archive/paper/sigma_lowres_v1/`). The v1 restructure plan is
discharged and frozen at `../record/paper_plan.md` — this plan is its
successor and covers only the delta: what the E19–E25.0 vector-ledger
results change about the manuscript, and the one new run (E26) the
reframe owes.

**Status: PLAN — no tex edited yet.** Order of work in §7; nothing in
§2–§5 is licensed to land in the tex before the §7 step that carries it.

---

## 1. Direction in one paragraph

The current draft's centerpiece is a *scalar* account: the demoted
gradient's deviation reduces to a ratio-governed decaying term plus a
σ-independent floor "carried by the compute graph," and this account
beats the spectral account head-to-head on the measured (route, σ) map.
Everything since the freeze upgraded the floor from an amorphous residue
to a *vector object with measured geometry*: the data and graph branches
B and C nearly cancel, the cancellation is **locally enforced** (E21),
the residual R = B + C is **one direction per σ, shared across routes,
stores, and corpus** (E24 STRUCTURED; E25.0's R̂ tables), and the miss
is **angle-borne, not amplitude-borne** (E24.2: closing ρ removes
70–100 % of the residual, amplitude matching ≤ 45 %). The reframe makes
this the paper's third act — question → account (scalar, beats
spectral) → geometry (what the floor *is*) → application — and replaces
"carried by the compute graph itself" phrasing with the measured
object. The two-accounts spine is kept, not replaced: the geometry
section is where the winning account's leftover term stops being a
fitted constant and becomes a characterized direction.

## 2. Claim ladder (what the reframed paper asserts, in order)

Carried claims are **not re-litigated** — their voice, hedges, and the
claims ledger transfer verbatim (§4).

1. *Metrology* (carried): debiased estimator, self-floors, draw-count
   extrapolation, ε\* certification resolution.
2. *Scalar account* (carried): two-term reduction of the four-term
   expansion; E5 predictive voice ("predicts routes it was not fit on
   at ~0.09 RMSE"); head-to-head win over the spectral account.
3. *Geometry* (NEW — the reframe's payload):
   - B/C decomposition is exact in shared adapter-parameter space;
     cross-set debiasing removes draw noise (`vector_ledger.py`, E9/E10
     machinery — already partly in the appendix, promoted to main text).
   - The branches nearly cancel and the cancellation is locally
     enforced, with adaln carrying 86–87 % of the phase-response
     amplitude (E21).
   - The residual is a σ-indexed **axis field**: one direction per σ
     across routes (median |cos| 0.95) and stores (≈ 1 debiased),
     rotating smoothly in σ (E24; E25.0 R̂ descriptive tables).
   - The residual is angle-borne (E24.2 knob read).
   - The rotation is **matched-angle, not planar**: transporting by the
     anchor's rotation buys nothing (E25.0-2 NO-GAIN — the "co-rotates
     with the σ-conditioned native gradient" intuition is retired; say
     so in the text, it is a natural reviewer guess).
   - Reliability scope: the pooled residual direction exists at 11/12
     verdict conditions with a recorded hole at 768/σ = 0.4333 and
     softness at mid-window 768 generally (E25.0-1 PARTIAL).
4. *Closures* (NEW, one honest subsection): re-deriving a ledger term
   as an objective correction is closed at the estimand level (E20.4);
   damping-form levers are dead at probe level (E23.0 DAMP-DEAD); a
   fixed subspace is refuted at lever level — any projector must be
   per-σ-bin (E23.0-C PER-BIN-ONLY); per-sample exploitation stays
   gated (E22 → E23a). These are results, not future work.
5. *Application* (carried): the validated-window routing, −14.6 % wall
   at fixed steps.
6. *Outlook* (ONE paragraph, no promise): population-level σ-local
   levers (E25a restricted / E25b sketch) are licensed-but-unrun; the
   paper does not depend on them.

## 3. The scope sentence (non-negotiable)

Every geometry claim in §2.3 is measured at **one operating point**:
one adapter (`anima_soup_sincos`), three vector stores, standard
corpus, the E24 debias conventions. The map claims have far broader
coverage than the geometry claims, and the draft must say so where the
geometry is introduced *and* in the abstract's hedge — otherwise the
paper invites exactly the R1-style estimator-confound review the line
already survived once. **E26 (§5) is the widening**; its verdict
finalizes the sentence's wording (§7 step 6). Until E26 lands, the
geometry section is written with the one-operating-point scope stated,
not deferred.

## 4. Carries over verbatim (do-not-touch list)

- The instrument + debiasing subsection and the "debiased units only
  from here" convention.
- The (route, σ) map, boundary scoring, endpoint ≡ x-zero ≡ α-flat.
- The E5 held-out + three-form refit material and its qualified voice.
- The claims-ledger table (extended with the §2.3/§2.4 rows, existing
  rows unedited).
- The head-to-head structure and `accounts_headtohead` figure.
- The application section and its numbers.
- All appendix raw/historical tables.

## 5. E26 — cheap cross-adapter cancellation check (the new run)

**Question**: does the B–C cancellation geometry replicate on LoRA
adapters other than the line's operating point?

**Instrument — nothing new**: `run_sigma_probe.py` already takes
`--adapter <ckpt>`; the read is `vector_ledger.py` + the E24/E25.0
machinery on the resulting `arm_sums/`, all CPU. The only delta from
E19.3's reference run is a slimmed grid and a different checkpoint.

- **Protocol sketch** (to be frozen in
  `../paper_bench/experiments/e26/README.md` *before* any run — line
  convention, pre-registration first):
  `--repromote --keep_arm_sums --self_floor --deterministic`, verdict
  σ bins only ({0.3, 0.4333, 0.5667, 0.7, 0.8333}), routes 1024→768 +
  1024→896, draws/D matched to e193 per-condition counts.
- **Cost**: e193 was 5.7 GPU-h at 16 draws × 15 σ × 3 routes; 5 σ ×
  2 routes extrapolates to **~1.3 GPU-h per adapter** (measure on the
  first and record). CPU read ~minutes per store. Daemon-submitted
  (`--queue`), like every probe run.
- **Adapters**: 2–4 plain-LoRA checkpoints. Cross-adapter *direction*
  comparison requires an identical parameter space (same rank + target
  modules — same config family as the operating adapter); adapters
  with a different config can still carry the scalar reads. Prefer
  spanning different training slices (different artists/pools) over
  different seeds of the same slice — seed-twins would understate the
  generalization claim.
- **Readouts, in verdict order**:
  1. *Cancellation replicates* (the paper-relevant read): h(B+C) ≪
     min(h(B), h(C)) at the verdict bins; I < 0 where the operating
     point has it; ledger scalar signs/ordering reproduced.
  2. *Axis field within adapter*: rel_cos_R ≥ 0.5 per condition;
     across-route sharing at fixed σ.
  3. *Cross-adapter axis* (descriptive bonus, no verdict weight
     unless pre-registered otherwise at freeze time):
     cos(R̂_a(σ), R̂_b(σ)) in the shared parameter space — decides
     "property of the model/route" vs "property of the adapter."
     Either answer is a finding; an adapter-universal axis would also
     upgrade any future E25a lookup from per-adapter to global.
- **Verdict shapes** (exact thresholds set at freeze):
  REPLICATES → the scope sentence widens to "N adapters, one model";
  PARTIAL → geometry claims keep the one-operating-point scope and the
  replication pattern is reported as-is; FAILS → the geometry section
  is explicitly scoped to the operating adapter and the cross-adapter
  result becomes a limitation paragraph. **No outcome removes §2.3
  from the paper** — the claims are true at the measured operating
  point under any E26 result; E26 only sets their stated breadth.

## 6. Figure debts

- **NEW — axis-field summary** (the reframe's signature figure): the
  σ-binned R̂ cosine structure (across-route / across-store / across-σ)
  in one panel; candidate base: extend `fig_ledger_geometry.py`.
- Promote/adapt `e250_rel.png` + `e250_frame.png` (reliability + the
  matched-angle-not-planar read) — appendix at minimum.
- Cancellation magnitude vs residual (h(B), h(C), h(B+C) by σ) —
  likely exists in the E19/E21 material; verify before drawing new.
- E26 replication panel (after §7 step 6).
- Carried figure debts from the v1 plan (Fig-1 enlargement, waterfall)
  remain owed and unchanged.

## 7. Order of work

1. **Freeze E26 pre-registration** (`e26/README.md`: thresholds,
   adapter list, exact grid) and submit the runs — GPU time is the
   long pole; everything below overlaps with the queue.
2. Survey pass over `sec_theory.tex` / `sec_experiments.tex` to fix
   the geometry section's insertion point and what moves out of the
   appendix (B/C formalism). No edits yet — a placement note appended
   to this file.
3. Abstract + intro rewrite (question unchanged; the two-accounts
   framing keeps its role; the geometry claim enters the abstract with
   the §3 scope hedge).
4. New geometry section (§2.3 ladder, E21 → E24 → E25.0 order) + the
   closures subsection (§2.4).
5. Theory section: promote the B/C vector formalism from appendix,
   keep the four-term expansion as the bridge from the scalar account.
6. E26 lands → scope sentence finalized, replication panel added,
   claims ledger rows updated.
7. Discussion/outlook trim (the one lever paragraph), full-tex
   compile + hedge-consistency pass (reuse the "polish experiment
   notes" discipline: no stale hedges against later verdicts).

## 8. Honesty guards

- No lever result is claimed or implied; E25a is unfrozen/restricted,
  E25b is a sketch. One outlook paragraph, flagged as such.
- The matched-angle result (E25.0-2) is reported even though it
  *weakens* the tidiest version of the geometry story (no normalized
  frame) — it is the pre-registered read and it scopes the lookup
  claim honestly.
- The 768/σ = 0.4333 reliability hole and mid-window 768 softness are
  reported next to the axis-field claim, not footnoted.
- E26 thresholds are frozen before any run; no post-hoc widening of
  the adapter list after seeing a result.
- Per-sample language stays out of the paper ("pooled residual
  direction at this operating point" — the E25.0 wording).
