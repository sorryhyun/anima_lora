# E25 — σ-local angle lever (population-level exploitation of the cancellation axis)

| | |
|---|---|
| **Status** | **SKETCH 2026-08-08 — NOT pre-registered** (arms E25a/E25b unfrozen; freezing still requires an explicit E20.4-adjacency justification). **E25.0 DONE 2026-08-09** (pre-registered same day before the `e250_read.py` instrument existed; CPU-only as planned) — **25.0-1 PARTIAL** (11/12 verdict conditions have a reliable pooled residual direction; hole at 768/σ = 0.4333), **25.0-2 NO-GAIN** (the E24 co-rotation is matched-angle, not planar — E25a's lookup must interpolate per σ bin). See Results. **E23.0 DONE 2026-08-09** ([../e23/](../e23/)) — the one-step counterfactual read: **23.0-B PROJ-PARTIAL** (D = 12 excess filter voids the σ ≥ 0.4333 verdict targets; recorded structure: all six σ = 0.7 transfers close 0.70–0.88 at SR ≈ 1, and the standard-corpus lookup closes 0.76–0.81 of the *resolved* e221/e224 excess out-of-corpus), **23.0-C PER-BIN-ONLY** (span projector actively harmful, SR 0.75–0.82 — a fixed subspace is refuted at lever level, not just descriptively). Per the frozen Stage-2 map: **E25a may only be frozen restricted — σ = 0.7, one direction per bin, route-shared, no span projector** — with its closure denominator anchored on resolved-excess evidence (e224; optionally a D ≥ 48 σ = 0.7 top-up as its own amendment) plus the standing E20.4-adjacency paragraph. E23a's damping form is DAMP-DEAD (closed at probe level); E25b unaffected. |
| **Question** | The line now knows *where* the residual gap lives: the cancellation is locally enforced (E21), its axis is one direction per σ across routes/stores (E24 STRUCTURED), and the residual is **angle-borne** — closing ρ → −1 removes 70–100 % of it while amplitude matching removes ≤ 45 % (E24.2). Can a **population-level** lever exploit this — either by filtering the residual direction out of demoted-sample gradients, or by making the implicit resolution dimension explicit — without touching anything the line has closed? |
| **Licensed by** | E24 STRUCTURED clause (this is the named follow-up; the lever **must be σ-local** — a single fixed subspace is refuted), E24.2 knob read (target = angle/residual direction, never amplitude rebalancing), E21 (adaln carries 86–87 % of the phase-response amplitude — the natural injection/damping site), pooled evidence only. |
| **Explicitly NOT licensed** | Anything per-sample (gated on E22 → 22.4 → E23a, unchanged); any PI-RoPE revival (G11); and — the sharpest adjacency — anything that re-derives a ledger term as an objective correction (E20.4 closed that at the estimand level). A frozen E25 must argue explicitly why its lever is an in-expectation filter / conditioning change and not the closed derived-data-term family. |

## Candidate arms (sketch only)

- **E25a — σ-local residual projection guard**: during demoted training
  steps, damp/project the component of the LoRA gradient along the
  σ-binned pooled residual direction (B+C)̂(σ), read from a probe-built
  lookup (per σ bin; co-rotating with the axis field per E24). Optimizer-
  side, in expectation, no per-sample estimate anywhere.
- **E25b — explicit resolution conditioning**: inject a resolution/scale
  embedding into the adaln pathway during LoRA training (micro-
  conditioning precedent), making the dimension the network currently
  absorbs implicitly (E21: adaln amplitude concentration) explicit.
  Prediction to gate on: ρ deepens / h(B+C) shrinks at the verdict σ,
  measured by the existing σ probe at the same operating-point protocol.

Both are Tier 1.5 if pursued: bench + invariant test required; quality
gate = no CMMD regression on the E4-style yardstick, pitch stays
wall-clock-at-fixed-steps (autoscale lesson).

## E25.0 — prerequisite read (cheap, CPU, must precede any freeze)

**Pre-registered 2026-08-09, before the instrument (`e250_read.py`)
existed.** Two facts the sketch depends on that are not yet measured:

1. **Residual-direction reliability**: is the pooled (B+C)̂ direction
   reproducible across draw sets per (route, σ)? (B and C are each
   reliable, but their sum is the small difference of large legs — its
   direction may not be.) Free read on the three existing arm_sums
   stores with the E24 machinery. If unreliable pooled, E25a is dead
   before it starts.
2. **Frame-relative stationarity**: is the axis stationary in the ĝ(σ)
   frame (E24's descriptive co-rotation, promoted to a pre-registered
   estimand)? Decides whether E25a's lookup can be one direction in a
   normalized frame or must interpolate per σ bin.

### Sources

The three E24 stores verbatim (e193 depth-ledger, e194 pi-causal, e221
per-image-ledger — re-verified on disk 2026-08-09), same adapter ⇒ one
operating point. Conditions, legs, debias, and gates are **inherited
from E24 unchanged**: legs verbatim `vector_ledger.bc_ledger` with
`data_ref = reenc`, each leg ⊥ against its condition's own ĝ; rel gate
0.5; verdict σ ∈ {0.3, 0.4333, 0.5667, 0.7, 0.8333}; standard-corpus
stores only in verdict statistics, e221 descriptive; π arms not read
(G11); cross-condition numerators debiased per the E24 conventions
(draw noise independent across bins/stores; same-store same-bin
cross-route B pairs subtract the shared reenc ref-noise power
|d⊥|²/4); denominators are the within-condition cross-set debiased
norms.

### Frozen quantities (no tuning on outputs)

- **Residual per half-set**: R₁⊥ = B₁⊥ + C₁⊥ (= the perp'd
  ḡ_dem − ḡ_reenc of draw set 1; rp cancels), R₂⊥ likewise.
  - `rel_cos_R` = raw cosine(R₁⊥, R₂⊥) — the exact mirror of
    `rel_cos_B`/`rel_cos_C` (raw, undebiased, per condition).
  - Debiased residual power **nR² = 2G²·(S + F + I)**, from the exact
    identity ⟨R₁⊥, R₂⊥⟩ − ref_noise = 2G²(S+F+I) (validation gate:
    both sides agree to ≤ 1e−6 in ledger units on every condition —
    the shared reenc ref-noise term carries over from B unchanged
    since C has no ref dependence).
  - A condition **passes read (1)** iff rel_cos_R ≥ 0.5 **and**
    nR² > 0.
- **Frame transport**: for an across-σ pair (x, y) within one (store,
  route), the minimal rotation Rot in span(ĝₓ, ĝᵧ) taking ĝₓ → ĝᵧ,
  applied to the pooled leg mean; **cos_fr = debiased cos(Rot·B̄⊥ₓ,
  B̄⊥ᵧ)** (rotation preserves norms ⇒ denominators unchanged; numerator
  computed exactly from Gram entries — ⟨Bₓ,Bᵧ⟩, ⟨ĝₓ,Bᵧ⟩, ⟨ĝᵧ,Bₓ⟩,
  ⟨ĝₓ,ĝᵧ⟩ — with no orthogonality shortcut). Reported next to the raw
  E24 cosine with **Δ = |cos_fr| − |cos_raw|** per pair. Pairs: E24
  family (i) (across-σ, same route, standard corpus), gated per leg as
  in E24. B table carries the verdict; C table reported alongside.
  Second-order noise products inside the transport numerator are
  accepted under the same cross-condition independence assumption as
  E24's numerators (noted, not corrected). Debiased values may exceed
  1; reported as computed, clamped only in figures.
- **What the transport distinguishes** (recorded up front): literal
  co-rotation of the axis inside ĝ's own motion plane predicts
  cos_fr ≈ 1; matched-angle rotation in an unrelated plane predicts
  cos_fr ≈ cos_raw. E24's descriptive readout (cos(B̂,B̂) ≈ cos(ĝ,ĝ))
  cannot separate these; this estimand does.

### Pre-registered readings

**25.0-1 (residual reliability; decides E25a's life):**

| outcome (12 verdict conditions) | verdict |
|---|---|
| every condition passes (rel_cos_R ≥ 0.5, nR² > 0) | **RELIABLE** — E25a's pooled lookup object exists at this operating point |
| median rel_cos_R < 0.5 | **UNRELIABLE** — E25a dead (sketch kill criterion executes); E25b unaffected |
| in between | **PARTIAL** — E25a, if ever frozen, is restricted to the passing (route, σ) set; the pattern is recorded |

**25.0-2 (frame-relative stationarity; decides the lookup's shape).**
Precedence: FRAME-STATIONARY checked first, then NO-GAIN, else PARTIAL.

| outcome (B table, gated across-σ verdict pairs) | verdict |
|---|---|
| median \|cos_fr\| ≥ 0.7 **and** min \|cos_fr\| ≥ 0.5 (the exact E24 SHARED-AXIS across-σ criterion, which the **raw** table failed at min 0.442) | **FRAME-STATIONARY** — one direction in the ĝ-co-rotating frame suffices; the lookup is the transport map itself |
| median Δ ≤ 0.02 | **NO-GAIN** — the E24 co-rotation was matched-angle, not planar; a lookup must interpolate per σ bin |
| in between | **PARTIAL** — per-σ-bin interpolation is the conservative default; the surviving structure recorded |

- **Descriptive rows (no verdict weight)**: transported and raw R̂
  pairwise cosines over read-1-passing verdict conditions (all three
  E24 families — does the residual direction share the axis-field
  structure; this directly shapes the E25a lookup); e221 rows; the
  C tables.

### Validation gates (all must pass before any new quantity is read)

1. e221 committed `ledger.json` reproduced exactly (all bins, all six
   scalars) — the E24 gate, rerun by this instrument.
2. Synthetic mini-store: within-condition scalars agree with
   `bc_ledger` exactly; planted residual direction recovered by
   rel_cos_R (and destroyed by large noise); the nR² identity holds.
3. Synthetic transport: a planted rigid co-rotation in the ĝ-motion
   plane yields cos_fr ≈ 1 where raw ≈ cos(ĝ,ĝ); an out-of-plane
   control (axis motion in a plane not containing ĝ's motion) must
   NOT inflate (cos_fr ≈ cos_raw).
4. nR² identity on the real stores: max deviation ≤ 1e−6 (ledger
   units) over all conditions.

### Kill switches / honesty

- Read-only CPU analysis of committed stores; nothing refit, no
  constant tuned, no objective term derived (E20.4 stands closed); no
  lever is implemented or licensed by this read — E25 freezing still
  separately requires the E20.4-adjacency paragraph.
- Pooled directions only; per-sample variants stay gated on
  E22 → 22.4 → E23a, unchanged. Wording: "pooled residual direction
  at this operating point", never per-sample.
- Outputs (this dir): `e250_read.json` (the record), `e250_rel.png`,
  `e250_frame.png`. Expected cost: one chunked fp64 Gram over the E24
  vector set (54 × 77.7M) — ~6 min CPU, ≤ ~26 GB resident, no GPU.

## E25.0 Results (2026-08-09)

Instrument: `e250_read.py` (this dir). All validation gates passed
before any new quantity was read: the E24 synthetic + e221
exact-reproduction gates rerun clean; the transport synthetic recovers
a planted in-plane co-rotation (raw 0.501 → transported 1.000) and
does **not** inflate the out-of-plane control (0.498 → 0.498); the nR²
identity holds on every real condition (max dev 4.2e−14, ledger
units). Record: `e250_read.json`; figures `e250_rel.png`,
`e250_frame.png`. Runtime 470 s CPU (one chunked fp64 Gram, as E24).

### 25.0-1 verdict: **PARTIAL** — the pooled residual direction exists, with one hole

- **11 of 12 verdict conditions pass** (rel_cos_R ≥ 0.5 and nR² > 0);
  median rel_cos_R 0.68. The residual direction is *less* reliable
  than its legs everywhere (rel_B 0.74–0.93, rel_C 0.60–0.85 on the
  same conditions) — the small-difference-of-large-legs worry was
  right in degree, wrong in kind: the direction survives pooling.
- **The one failure: e193/768/σ = 0.4333** (rel_cos_R 0.374).
  Suggestive context (descriptive, not verdict): e221/768 also fails
  at its two lowest bins (0.385 at σ = 0.4333, 0.491 at 0.5667) while
  its σ = 0.7 passes (0.639) — mid-window **768 is where the residual
  direction is least reproducible**; 896 passes at every verdict σ
  (min 0.514). Endpoint bins (detached): σ = 0.9625 is the most
  reliable of all (0.92–0.94); σ = 1.0 mixed (0.39–0.94).
- Per the frozen reading: **E25a is not dead**; if ever frozen it is
  restricted to the passing (route, σ) set, with the 768/σ ≈ 0.43
  hole recorded.

### 25.0-2 verdict: **NO-GAIN** — the co-rotation is matched-angle, not planar

- B table, 14 gated across-σ pairs: median |cos_fr| 0.796 vs raw
  0.791; min 0.416 vs 0.442; **median Δ = −0.0005**. Precedence:
  FRAME-STATIONARY fails first (min 0.416 < 0.5 — the transport even
  slightly *hurts* the extreme-span pairs), then NO-GAIN fires
  (Δ ≤ 0.02). C table agrees (median Δ 0.007).
- Mechanistic reading (the separation the estimand was built for):
  E24's descriptive "the axis rotation tracks the anchor's rotation"
  is an **equal-angle coincidence, not a rigid co-rotation inside ĝ's
  motion plane** — transporting by the ĝ rotation does nothing
  because the axis's component in that plane is small. The axis
  rotates through the same angle as ĝ but in its own plane.
- Consequence (frozen): **E25a's lookup must interpolate per σ bin**;
  a single direction in a ĝ-normalized frame is refuted. This also
  retires the "network merely co-rotates with the σ-conditioned
  native gradient" intuition from E24's descriptive note.

### Descriptive: the residual direction is itself an axis-field

R̂ = (B+C)̂ over the 11 passing verdict conditions mirrors the E24 leg
structure: **across-route, same σ: median |cos| 0.95** (min 0.75, 5
pairs); **across-store, σ = 0.7: ≈ 1.05** (debiased, ≈ 1; both
routes); across-σ: median 0.78 (min 0.56; transport again ≈ no gain,
0.80/0.58). So the exact object E25a would look up — a σ-binned pooled
residual direction, shared across routes, reproducible across runs —
exists per σ bin, except at the recorded hole.

### What this decides for E25

- **E25a: alive but narrowed.** Lookup = per-σ-bin interpolation (no
  normalized frame), restricted to read-1-passing conditions; the
  route-shared R̂(σ) supports one direction per σ bin (not per route).
  The freeze still separately owes the E20.4-adjacency paragraph.
- **E25b: unaffected** by either read (as pre-registered for a 25.0-1
  failure; 25.0-2 touches only the lookup shape). The NO-GAIN
  mechanism note is context for its adaln-conditioning rationale, not
  a gate.

## Kill criteria (sketch level)

- E25.0 (1) fails → E25a dead; E25b unaffected.
- Any formulation that requires a per-sample quantity → out of scope,
  full stop (that is E23a's gated territory).
- If the frozen version cannot distinguish itself from E20.4's closed
  family in one paragraph, it does not run.
