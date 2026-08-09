# E23 — training-facing lever gate: the one-step counterfactual read (E23.0)

| | |
|---|---|
| **Status** | **PLANNED — E23.0 pre-registered 2026-08-09, before the `e230_read.py` instrument existed** (CPU-only, committed stores, no GPU). E23a/E23b remain the gated sketches in [E22](../e22/)'s README — E23a's drafting is licensed by 22.4 (PER-SAMPLE HOLDS at σ = 0.7) but is **not** frozen here; this read decides whether it ever is. |
| **Question** | Two implementable gradient-surgery levers are now on the table for the demoted-step training path: **adaln-branch damping** (E23a's sketch, per E21's 86–87 % amplitude concentration) and **σ-binned residual projection** (E25a's sketch, per E24 STRUCTURED + E25.0 PARTIAL/NO-GAIN). Both act on the same object — the demoted sample's LoRA gradient — and both are *exactly simulable, one-step, pooled,* from the committed `arm_sums/` stores: damping is a band mask-scale of the demoted arm, projection is a rank-1/rank-k removal, and every debiased gap they produce is algebra on per-band Gram entries. Before any training bench is frozen (Stage 2 = one of E23a / E25a / E25b), measure for free: does each lever actually move the pooled demoted gradient *toward* native, at what cost in retained signal, and does the projection lookup transfer out-of-sample (across stores / routes / σ)? |
| **Licensed by** | E22 → 22.4 (E23a drafting licensed, single-σ caveat standing); E24 STRUCTURED + knob read (any lever must be σ-local and angle-targeted); E25.0 (25.0-1 PARTIAL — the pooled lookup object exists per σ bin, route-shared, hole at 768/σ = 0.4333; 25.0-2 NO-GAIN — per-σ-bin lookup mandatory). This experiment is measurement only — pooled, one operating point, no lever is implemented in training code and none is shipped from here. |
| **Explicitly NOT licensed** | Any ledger-derived objective term (E20.4 closed at estimand level — nothing here is a loss term; the levers are optimizer-side filters evaluated in counterfactual only); any PI arm read (G11); any per-sample estimated quantity (E23a's per-sample *license* is about the mechanism holding sample-by-sample, not about estimating per-sample directions — no such estimate appears here or in any lever this read can license). |
| **Instruments** | 23.0 `e230_read.py` (CPU; per-band Gram over the four committed stores → damping sweep, projection transfers, lookup-shape read; outputs `e230_read.json`, `e230_damp.png`, `e230_proj.png`, `e230_leak.png`). |
| **In the paper** | The §5/discussion bridge from mechanism to prescription gets its cheapest possible next sentence either way: a lever family that survives the one-step counterfactual is licensed for a training bench (Stage 2); a family that fails it is closed without a GPU-hour spent — and if both gradient-surgery families fail, the shipped scheduler-side recipe is reconfirmed as the only sanctioned family at this operating point, which is itself a clean claim. |

**Numbering note**: E23a (per-sample adaln lever) and E23b
(preprocess-resize causal transport) keep their reserved names in E22's
sketch. E23b additionally lost its empirical motivation on 2026-08-09
(22.4's resize-factor null stands at healthy gated counts — no g-space
hypothesis was generated); it stays parked and is not touched here.
E23.0 is also, jointly, **E25a's remaining prerequisite** — its
projection tables complete what E25.0 started ([E25](../e25/) carries
the pointer); the Stage-2 selection map is frozen at the bottom of this
file.

## Sources (verified on disk 2026-08-09)

| store | σ centers loaded | routes | n_img | draws | role |
|---|---|---|---|---|---|
| `bench/results/20260807-0745-e193-depth-ledger/arm_sums/` | 0.3, 0.4333, 0.5667, 0.7 | 896, 768 | 40 | 12 | verdict |
| `bench/results/20260807-1400-e194-pi-causal/arm_sums/` | 0.7, 0.8333 | 896, 768 | 40 | 12 | verdict (π arms not read) |
| `paper_bench/runs/20260808-1633-e221-per-image-ledger/arm_sums/` | 0.4333, 0.5667, 0.7 | 768 | 16 | 24 | descriptive (stratified corpus) + validation target |
| `paper_bench/runs/20260809-0031-e224-per-image-d96/arm_sums/` | 0.7 | 768 | 16 | 96 | descriptive (stratified corpus, highest SNR) + validation target |

All four share the adapter (`anima_soup_sincos`) ⇒ one operating point.
Endpoint bins (0.9625, 1.0) are not loaded — every read here is
in-window. e224 joins the source list for the first time (its arm store
was kept by the 22.4 `--keep_arm_sums` flag and is hereby put to use);
like e221 it is stratified-corpus, so **never** in verdict statistics.
Band ranges from each store's `arm_sums/groups.json`; the instrument
asserts the four stores' type ranges are identical before any read (one
adapter ⇒ one parameter layout). The 10 core types are a disjoint cover
of all 77.7M dims (E21's verified partition); the adaln band = the three
`adaln_up_*` types, other bands per E21's `BANDS`.

## Frozen conventions (pre-registered — no tuning on outputs)

**Estimand — debiased gap to native.** Per condition c = (store, route,
σ-bin), arms a, b (native half-sets), re1/re2 (reenc), dem1/dem2
(demoted); g0 = (a+b)/2, ĝ = g0/‖g0‖ (raw ledger anchor, as
`bc_ledger`):

- d_dem = 1 − [(⟨dem1, g0⟩ + ⟨dem2, g0⟩)/2] / √(⟨dem1, dem2⟩·⟨a, b⟩)
  — numerator symmetrized over independent sets, both norms cross-set
  debiased. d_reenc analogously from re1/re2. Values may exceed the
  [0, 2] cosine range under noise — reported as computed (E24
  convention).
- **closure(lever) = (d_dem − d_lever)/(d_dem − d_reenc)** — the
  fraction of the demotion excess (over the reenc pipeline floor, which
  is the ceiling of any lever acting on the demoted arm: removing B+C
  entirely reproduces reenc by construction) that the lever closes.
- **Twin noise**: σ_twin(d) = |d⁽¹⁾ − d⁽²⁾|/2 from the two single-set
  numerator halves (shared denominators); σ_twin(Δ) adds dem and reenc
  in quadrature. **Resolvable-excess filter**: a condition enters
  closure statistics iff Δ_c = d_dem − d_reenc ≥ max(0.005,
  3σ_twin(Δ)). Filtered-out conditions reported by count with their Δ.
- **Gates inherited, not re-derived**: condition gating (rel_cos_B ≥
  0.5 ∧ rel_cos_C ≥ 0.5) from the committed `e24_axis.json` (all 12
  verdict conditions passed); read-1 pass (rel_cos_R ≥ 0.5 ∧ nR² > 0)
  from the committed `e250_read.json` (11/12; hole e193/768/σ = 0.4333).
  rel_cos_R is recomputed by this instrument as a cross-check
  (validation gate 3), not re-gated.

**Lever 1 — adaln damping** (E23a's implementable form: scale the adaln
band of the demoted sample's gradient by λ):

- M(λ) = diag mask, λ on the adaln band, 1 elsewhere;
  d_damp(λ) = 1 − [(⟨M dem1, g0⟩ + ⟨M dem2, g0⟩)/2] /
  √(⟨M dem1, M dem2⟩·⟨a, b⟩) — every term a polynomial in λ over the
  10 per-band Gram entries.
- λ* = argmin over the 0.01 grid on [0, 1]; display grid {0, 0.25,
  0.5, 0.75, 0.9, 0.95, 1.0}. **Signal retention** SR(λ) =
  num(λ)/num(1) (retained native-parallel component); update-norm
  retention √(⟨M dem1, M dem2⟩/⟨dem1, dem2⟩) as context.
- Descriptive rows (no verdict weight): the same sweep with the mask on
  each other band (self_attn, mlp, cross_attn) — localization context
  only.

**Lever 2 — residual projection** (E25a's implementable form: remove a
probe-frozen direction from the demoted step's gradient):

- Set-pure residual directions r̂_s = normalize(P⊥ĝ (dem_s − re_s)),
  s ∈ {1, 2}; condition-mean direction r̂ = normalize(P⊥ĝ (dem̄ − re̅))
  (the E25.0 object). P[u] = I − ûûᵀ; span projectors orthonormalize
  their direction set via its small Gram.
- **Own-bin (cross-set)**: d_proj = 1 − [(⟨P[r̂₂] dem1, g0⟩ +
  ⟨P[r̂₁] dem2, g0⟩)/2] / √(⟨P[r̂₂] dem1, P[r̂₁] dem2⟩·⟨a, b⟩) — each
  set projected by the *other* set's direction; projector-noise terms
  in the norm are second-order and accepted (E24 convention, noted not
  corrected). Context row: an upper bound on what a same-condition
  lookup could do.
- **Transfers**: direction = the source condition's mean r̂, evaluated
  on the target's dem sets with the fixed-projector debiased formula
  (numerator halves each independent of the direction's source across
  stores; within-store cross-route transfers share the reenc/native
  arms with the target — second-order, recorded). SR_proj =
  numerator ratio vs no projection.
- **Lookup directions (23.0-C)**: window σ ∈ {0.5667, 0.7, 0.8333}
  (the shipped σ > 0.5 demotion gate's in-window probe bins). For
  evaluation on condition c: ℓ̂_σ(−c) = normalize(Σ r̂_x) over verdict
  conditions x at that σ that pass read-1, **excluding c itself**
  (leave-self-out ⇒ every 23.0-C row is out-of-sample; route-shared
  pooling per 25.0's descriptive R̂ read). Span projector for c: the
  span of the three window ℓ̂_σ'(−c). e221/e224 rows use the same
  verdict-store-built lookups (they are never in the pool).

## Pre-registered readings

**23.0-A — damping viability** (gated verdict conditions passing the
excess filter; both routes). Precedence: VIABLE → DEAD → PARTIAL.

| outcome | verdict |
|---|---|
| median closure(λ*) ≥ 0.25 **and** median SR(λ*) ≥ 0.95 **and** λ* ≤ 0.99 in ≥ 75 % of conditions | **DAMP-VIABLE** — E23a's damping family survives its one-step mechanism check; it may compete for Stage 2 |
| median closure(λ*) ≤ 0.05 **or** median λ* ≥ 0.99 **or** every λ reaching closure ≥ 0.25 has median SR < 0.9 | **DAMP-DEAD** — branch damping moves the pooled demoted gradient no closer to native (or only at unacceptable signal cost); E23a's damping form is closed in this line absent new mechanism evidence, with **no GPU spent against the 22.4 license** (the license itself stays on record) |
| else | **DAMP-PARTIAL** — structure recorded; conservative default = DEAD for spend decisions unless a passing subset is named |

**23.0-B — projection transfer** (the E25a viability read; gated +
read-1-passing verdict conditions passing the excess filter). Verdict
family = (ii) + (iii) pooled; (i) is context.

- (i) own-bin cross-set closure (context; within-condition upper bound)
- (ii) cross-store transfers at σ = 0.7 (e193 ↔ e194, per route, both
  directions — 4 readings; fully independent runs)
- (iii) cross-route transfers, same store and σ (896 → 768 and reverse
  at every verdict σ with both routes)

| outcome | verdict |
|---|---|
| median transfer closure ≥ 0.5 **and** min ≥ 0.25 **and** median SR ≥ 0.98 | **PROJ-TRANSFERS** — the pooled lookup object does its job out-of-sample; E25a's freeze is licensed for drafting (shape from 23.0-C) |
| median transfer closure < 0.25 | **PROJ-WEAK** — the training-time benefit of a probe-built lookup is marginal; E25a is not frozen on this evidence |
| else | **PROJ-PARTIAL** — surviving structure recorded (e.g. transfers pass only at high σ); any restricted E25a freeze must name it |

**23.0-C — lookup shape** (gated window conditions in verdict stores;
e221/e224 descriptive). Rows per condition: closure + SR for (a)
own-bin cross-set, (b) leave-self-out own-σ lookup, (c) each other
window σ's lookup (σ-leakage), (d) the span projector. Descriptive:
the ⟨ℓ̂_σx, ĝ_y⟩ leakage matrix. Precedence: SPAN-OK → PER-BIN-ONLY →
PARTIAL.

| outcome | verdict |
|---|---|
| at every gated window condition: closure_span ≥ 0.8·closure_own **and** SR_span ≥ 0.98 | **SPAN-OK** — a fixed rank-≤3 projector over the window directions is batch-compatible and loses nothing material; E25a's freeze may use it (no per-sample σ segregation needed) |
| any gated window condition has SR_span < 0.98 **or** closure_span < 0.5·closure_own | **PER-BIN-ONLY** — the lever must apply per-σ (per-sample σ is known at training; the freeze must specify the micro-batch mechanism) |
| else | **SHAPE-PARTIAL** — per-σ application is the conservative default |

**Stage-2 selection map (frozen here, executed after the readings):**

| 23.0-A | 23.0-B | Stage 2 |
|---|---|---|
| DAMP-DEAD | PROJ-TRANSFERS | **E25a** is the Stage-2 lever (freeze drafted with the 23.0-C shape; still owes its E20.4-adjacency paragraph; E25b optional second arm). E23a closed at probe level. |
| DAMP-VIABLE | PROJ-WEAK | **E23a** (damping bench, single-σ caveat carried per 22.4) is Stage 2; E25a not frozen. |
| DAMP-VIABLE | PROJ-TRANSFERS | The lever with the **higher median closure at matched SR ≥ 0.95** goes to Stage 2; the other is recorded and needs its own pre-registration to ever run. |
| DAMP-DEAD | PROJ-WEAK | No gradient-surgery Stage 2. E25b (resolution conditioning — not simulable from stores) remains the only candidate arm and would need its own freeze; the shipped scheduler-side recipe stands reconfirmed as the only sanctioned family at this operating point. |

PARTIAL outcomes take their nearest weak/dead side for spend decisions
unless the recorded structure names a passing subset explicitly.

## Validation gates (all must pass before any new quantity is read)

1. The committed e221 **and** e224 `ledger.json` (reenc ref) reproduced
   exactly at output rounding — all bins, all six scalars
   (S/F/I/ρ/rel_cos_B/rel_cos_C) — through this instrument's band-Gram
   path (B/C legs via the e221/e224 rp arms, loaded for this gate).
2. Partition exactness: the 10 per-band Grams resum to the full-vector
   Gram at machine precision (they are the same sums reordered).
3. rel_cos_R recomputed here matches the committed `e250_read.json`
   values on shared conditions to ≤ 1e−4.
4. Synthetic mini-store: (a) a planted adaln-band-borne excess with the
   in-band reference component parallel to ĝ recovers its analytic λ*;
   excess planted outside adaln recovers λ* = 1; (b) a planted residual
   direction gives own-bin projection closure ≈ 1 and transfer closure
   ≈ cos²φ at a planted angle φ; (c) debias negative control: on a
   noise-dominated store the *raw-mean* h(λ) sweep must show the
   spurious small-λ preference (damping also damps noise) while
   d_damp(λ) stays flat — the reason the debiased estimand is the
   frozen one; (d) λ = 1 and the empty projector reproduce d_dem
   exactly (identity gate, also asserted on the real stores).

## Kill switches / honesty

- Read-only CPU analysis of committed stores; nothing refit, no
  constant tuned, no objective term derived (E20.4 stands closed), no
  training code touched, no PI arms read (G11).
- **One-step, pooled, one operating point.** A lever that wins here is
  licensed for a Stage-2 *training bench* — never shipped from this
  read (training dynamics are unmeasured). A lever that loses here is
  dead in this line absent new mechanism evidence: a training run
  cannot rescue a lever that fails its own one-step mechanism.
- **The per-image spread of the projection's effect is not obtainable**
  from committed stores (per-image records are scalar reductions; no
  ⟨R_i, r̂_pool⟩ products were stored). Bounding per-sample harm of a
  pooled lookup is Stage 2's business — or one added scalar line in any
  future per-image GPU amendment. Recorded as the known gap.
- λ grid, thresholds, window bins, lookup construction, and the
  Stage-2 map are frozen above before the instrument exists. Wording:
  "pooled one-step counterfactual at this operating point", never a
  training-outcome claim.
- Stratified-corpus stores (e221/e224) are descriptive everywhere.

## Cost ladder (planned)

| item | planned |
|---|---|
| 23.0 | CPU only: one chunked fp64 band-Gram over ~80 condition vectors × 77.7M dims (~2.5× the E25.0 Gram work ⇒ ~10–20 min), ≤ ~28 GB resident (fp32-resident means; streaming fallback if RAM is tight); all readings are small-matrix algebra on the Grams |
| GPU | none |
