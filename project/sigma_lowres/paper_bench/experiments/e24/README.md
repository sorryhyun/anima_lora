# E24 — cancellation-axis geometry: is the cancelled direction one global mode?

| | |
|---|---|
| **Status** | **DONE 2026-08-08 — verdict STRUCTURED** (σ-indexed axis field: one direction across route/store/corpus at fixed σ, smooth rotation across σ; knob read: the residual is angle-borne — see Results). Pre-registered same day before the `e24_axis.py` instrument existed. CPU-only as planned. **Update 2026-08-09**: the descriptive "rotation tracks the anchor's own rotation" reading was tested and **retired** by [E25.0-2](../e25/) — matched-angle, not planar co-rotation (ĝ-frame transport buys Δ ≈ 0); see the annotated bullet in Results. |
| **Question** | ρ̄ ≈ −0.91 means that per (route, σ) the pooled legs B⊥, C⊥ span a nearly 1-D subspace — a "cancellation axis" along which the data damage and the graph response slide against each other. Everything so far measures the *angle inside* each condition; nothing yet asks whether the axis itself is **one shared direction** across σ, routes, and runs ("a scale mode of the adapter-gradient space" — the geometric reading: demotion as an approximate symmetry the network absorbs, residual = failure of equivariance), or a per-condition direction that merely always cancels locally. Secondary, free from the same scalars: what actually dominates the residual gap — the incomplete **angle** (ρ > −1) or the **amplitude mismatch** (\|B\| ≠ \|C\|)? That pins which knob any population-level lever should target, *before* one is proposed. |
| **Depends on** | [E19](../e19/) 19.3 + [E19.4] surviving `arm_sums/` stores and [E22](../e22/) 22.1's `--keep_arm_sums` store (all three verified on disk 2026-08-08, same adapter `anima_soup_sincos` — one operating point); `paper_bench/vector_ledger.py` (leg/debias conventions, `Sums` loader); [E21](../e21/) (LOCAL — the axis question is the *cross-condition* complement of E21's *within-condition* cell read); [E20](../e20/) 20.4 (closed: no ledger-derived objective term — this experiment derives nothing, it only measures geometry). |
| **Instruments** | 24.1 `e24_axis.py` (CPU; cross-condition axis cosines + subspace rank + figure); 24.2 free re-read of 24.1's scalars (residual knob decomposition); 24.4 `e24_axis_fig.py` (illustration-only figures from the committed digest — added post-verdict, no new quantities). |
| **In the paper** | The §5 geometric paragraph gets its missing measurement: SHARED-AXIS licenses the "scale mode" language (and, downstream, the population-level lever family sketched in E25 below); LOCAL-AXIS confines the geometric reading to "locally 1-D, globally unstructured" and kills projection-style levers in one stroke. The knob decomposition feeds the discussion of *why* the shipped scheduler-side recipe is the right family (or what a training-facing family should target if ever licensed). |

**Numbering note**: E23a/E23b are reserved by [E22](../e22/)'s gated lever
sketch (not pre-registered, gate not met) — this experiment is **not** them
and does not touch their gates. E24 is un-gated: pooled geometry only.

## Sources (verified on disk 2026-08-08)

| store | σ centers | routes | n_img | draws | note |
|---|---|---|---|---|---|
| `bench/results/20260807-0745-e193-depth-ledger/arm_sums/` | 0.3, 0.4333, 0.5667, 0.7, 1.0 | 896, 768 | 40 | 12 | standard corpus |
| `bench/results/20260807-1400-e194-pi-causal/arm_sums/` | 0.7, 0.8333, 0.9625, 1.0 | 896, 768 | 40 | 12 | standard corpus; π arms ignored |
| `paper_bench/runs/20260808-1633-e221-per-image-ledger/arm_sums/` | 0.4333, 0.5667, 0.7 | 768 | 16 | 24 | **stratified corpus — selection differs**; internal-consistency rows only, flagged in every output |

All three share the adapter ⇒ one operating point. e193↔e194 share the
corpus ⇒ their σ = 0.7 overlap is the clean cross-store pair (already
consistency-checked at cell level by E21's guard (c)).

## Frozen conventions (pre-registered — no tuning on outputs)

- **Condition** = (store, route, σ-bin). **Legs verbatim
  `vector_ledger.bc_ledger`, `data_ref = reenc`**: B = ḡ_rp − ḡ_reenc,
  C = ḡ_dem − ḡ_rp; each leg ⊥ against that condition's own ĝ (the
  ledger estimand). π arms are not read (G11 untouched).
- **Within-condition scalars** (rel_cos_B/C, S, F, I, ρ, ref-noise) must
  reproduce `vector_ledger.py --data_ref reenc` **exactly** at output
  rounding on e221 (its `ledger.json` is committed) — instrument
  validation gate, run before any new quantity is read.
- **Cross-condition cosines are debiased**: numerator ⟨B̄⊥ₓ, B̄⊥ᵧ⟩ (draw
  noise independent across bins/stores ⇒ unbiased), **except** same-store
  same-bin cross-route B pairs, which share the reenc reference — there
  the numerator subtracts the shared ref-noise power (|d⊥|²/4, the
  bc_ledger convention). Denominators are the within-condition
  **cross-set debiased** norms √(2G²·S) (B) and √(2G²·F) (C). Same
  convention for C (no shared-arm correction needed). Values may exceed
  1 in magnitude under noise — reported as computed, clamped only in
  figures.
- **Gate**: a condition enters verdict statistics only if the leg's
  rel_cos ≥ 0.5 (B-reads gate on rel_cos_B, C-reads on rel_cos_C).
  Gated-out conditions reported by count. **Verdict σ** = in-window
  {0.3, 0.4333, 0.5667, 0.7, 0.8333}; endpoint bins (0.9625, 1.0)
  detached, descriptive only. e221 conditions are **never** in verdict
  statistics (selection differs) — descriptive rows.
- **Verdict quantity**: the set of pairwise debiased cosines
  |cos(B̂⊥ₓ, B̂⊥ᵧ)| over all gated verdict-condition pairs from the
  standard-corpus stores (e193, e194), split into three pre-named
  families: (i) **across-σ, same route**; (ii) **across-route, same σ**;
  (iii) **across-store, σ = 0.7**. Same table for Ĉ. Absolute value:
  an axis is a line, not a ray — sign is reported but does not enter
  the verdict.
- **Subspace rank** (descriptive): participation ratio and top-1 share
  of the debiased Gram over gated verdict conditions, per leg per route
  (and pooled routes).
- **Context rows** (descriptive, no verdict weight): cos(ĝₓ, ĝᵧ) matrix
  (how much the perp anchor itself moves); undebiased/raw-B̄ variant of
  the cosine table (robustness of the debias); cross-leg cos(B̂ₓ, Ĉᵧ)
  (do B and C span one *plane* globally).
- **24.2 residual knob decomposition** (free re-read, per gated verdict
  condition, from S/F/ρ): R = S + F + 2√(SF)ρ (the quadratic residual);
  counterfactuals **R_angle** = (√S − √F)² (ρ → −1, amplitudes fixed)
  and **R_amp** = 2S(1 + ρ) (amplitudes matched at the B leg, angle
  fixed). Read: which counterfactual leaves the smaller residual share
  R_·/R, per condition.

## 24.3 — pre-registered readings

| outcome (gated verdict pairs, standard-corpus stores) | verdict |
|---|---|
| median \|cos\| ≥ 0.7 **and** min ≥ 0.5 in **all three** pre-named families, for the B table (C table reported alongside) | **SHARED-AXIS** — one global cancellation direction exists at this operating point; the "scale mode" §5 language is licensed; E25 (population-level lever family: projection guard / explicit resolution conditioning of adaln) may be sketched with its own pre-registration. |
| median \|cos\| ≤ 0.3 in family (i) or (ii) | **LOCAL-AXIS** — the cancellation is locally 1-D but globally unstructured; projection-style levers are dead in this line; the geometric §5 paragraph is confined to the within-condition statement (E21). |
| in between / family-dependent (e.g. σ-shared but route-local) | **STRUCTURED** — record which family carries the alignment; any downstream use needs a follow-up pre-registration naming the surviving structure. |

- **Knob reading** (24.2, descriptive but pre-named): ANGLE-DOMINATED if
  R_amp < R_angle at every gated verdict condition (closing the angle
  buys less than matching amplitudes ⇒ the residual is angle-borne);
  AMP-DOMINATED for the reverse; else MIXED — recorded per condition.
- **e221 consistency row**: e221's within-family cosines (across-σ,
  768) reported against the e193 equivalents — agreement within the
  twin-based noise is an internal-consistency note, disagreement is a
  **selection-sensitivity flag** on the axis (worth its own follow-up,
  not a verdict change here).

## Kill switches / honesty

- Read-only CPU analysis of committed stores; nothing is refit, no
  constant tuned, no objective term derived (E20.4 stands closed).
- Pooled directions only: per-image axis spread is **not obtainable**
  from these stores and not claimed — wording "pooled cancellation
  axis at this operating point", never per-sample. The per-sample
  version of any E25 lever remains gated on E22's 22.4 → E23a chain.
- The perp anchor ĝ differs per condition; the cos(ĝ, ĝ) context matrix
  is reported so an axis read is never quietly an anchor-motion read.
- Anti-scope: no PI arms (G11), no per-cell axis atlas (E21's lattice
  already covers within-condition cell structure; a cell-level axis
  comparison would need its own pre-registration), no lever
  implementation in this experiment.

## Results (2026-08-08)

Instrument: `e24_axis.py` (this dir). Validation gates passed before any
new quantity was read: synthetic mini-store agrees with the committed
`bc_ledger` exactly (S/F/I/ρ/rel at output rounding) and recovers a
planted cross-store axis cosine to < 0.05; e221's committed `ledger.json`
reproduced **exactly** (all 3 bins, all six scalars). Every one of the 12
verdict conditions passes both rel gates (relB 0.74–0.93, relC
0.60–0.85) — no reliability caveat anywhere (pooled estimand; the
per-image reliability wall of E22 does not apply here). Record:
`e24_axis.json`; figures `e24_axis_cos.png`, `e24_knobs.png`. Runtime
348 s CPU (single chunked fp64 Gram over 54 × 77.7M vectors).

### 24.3 verdict: **STRUCTURED** (B table; C table agrees)

| family | n | median \|cos\| | min \|cos\| |
|---|---|---|---|
| across-σ, same route | 14 | 0.791 | **0.442** |
| across-route, same σ | 6 | 0.971 | 0.866 |
| across-store, σ = 0.7 | 2 | 1.003 | 0.999 |

SHARED-AXIS fails only on the across-σ min (0.442 < 0.5); LOCAL-AXIS is
nowhere close. The pre-registered STRUCTURED clause applies — the
alignment structure, recorded:

- **At fixed σ the axis is one direction.** Across routes 0.87–0.99
  (rising with σ), across stores at σ = 0.7 ≈ 1.00 (debiased values may
  exceed 1; both legs, both routes). The damage direction is not a
  route property — 896 and 768 demotion push the adapter along the same
  line, and two independent runs agree to within noise.
- **Across σ the axis rotates smoothly and monotonically with σ
  separation**: adjacent bins 0.89–0.97, extreme span (0.3 ↔ 0.7)
  0.44–0.60; all signs positive (a coherent rotating field, no flips).
  The C table shows the same pattern (min 0.572).
- **The rotation is matched-angle with the anchor's, but in its own
  plane** (settled 2026-08-09 by [E25.0-2](../e25/)): at matched spans,
  cos(B̂, B̂) ≈ cos(ĝ, ĝ) — 0.44 vs 0.43 (0.3↔0.7), 0.74 vs 0.82
  (0.4333↔0.7), 0.94 vs 0.94 (0.5667↔0.7), 0.90 vs 0.91 (0.7↔0.8333).
  The pre-registered frame-relative estimand shows these equal angles
  are a coincidence of magnitude, not a rigid co-rotation in ĝ's motion
  plane: transporting by the ĝ rotation buys nothing (median
  Δ = −0.0005), so the axis is **not** σ-stationary in the ĝ frame — it
  rotates through the same angle as ĝ in its own plane.
- **Gram** (12 gated verdict conditions): top-1 share 0.775 / PR 1.61
  (B), 0.792 / 1.55 (C) — one dominant mode plus one secondary mode
  (λ₂ ≈ 1.6–1.7), consistent with "dominant shared axis + σ-rotation
  component". Cross-leg context rows confirm B and C stay in one plane
  across σ (cos(B̂ₓ, Ĉᵧ) mirrors the B–B decay with the expected sign).
- **e221 consistency row**: the stratified corpus sees the same axis —
  e221 across-σ cosines 0.71/0.90/0.95 vs the e193/768 equivalents
  0.74/0.91/0.94. **No selection-sensitivity flag.**
- Honest note on the near-miss: the SHARED-AXIS gate failed on the
  single most-distant pair (0.3 ↔ 0.7, 768). Restricted to σ ≥ 0.4333
  the min is 0.74 and SHARED-AXIS would have passed — recorded as
  description, not a verdict change (no post-hoc re-thresholding); σ =
  0.3 is also where the anchor itself rotates fastest (ĝ: 0.54 to its
  nearest verdict neighbor).

### 24.2 knob decomposition: the residual is angle-borne (numeric read)

Across all 12 gated verdict conditions: **R_angle/R = 0.0001–0.298**
(closing the angle, ρ → −1, removes 70–100 % of the residual) while
**R_amp/R = 0.55–1.04** (matching amplitudes removes ≤ 45 % and is
sometimes slightly counterproductive). 896 conditions are the extreme
cases (R_angle/R ≤ 0.11 at every 896 row).

**Pre-registration defect, recorded**: the frozen "Knob reading" bullet
is self-contradictory as written — its parenthetical ("⇒ the residual is
angle-borne") inverts the inequality it names. Per the literal
inequality the outcome would be labeled AMP-DOMINATED; semantically the
measurement says the opposite. Mirroring E21's dual-read honesty: **no
label is claimed**; downstream use must cite the numbers, whose reading
is unambiguous — the residual gap lives in the incomplete angle, not in
amplitude mismatch, at every gated verdict condition.

### What this buys / licenses

- §5 geometric paragraph: the cancellation axis is **global across
  routes and runs at fixed σ and rotates smoothly with σ** — "a
  σ-indexed scale mode of the adapter-gradient space", not one frozen
  direction; the rotation lives in the axis's own plane, matched-angle
  with ĝ but not co-rotating (E25.0-2). The lattice (E21) + axis-field
  (E24) pair gives the mechanism figure its cross-condition complement.
- Per the STRUCTURED clause, downstream use needs a follow-up
  pre-registration naming the surviving structure. The surviving
  structure is strong: any projection/conditioning-style lever (the E25
  family sketched in the pre-registration) must be **σ-local** (or
  σ-conditioned) rather than a single fixed subspace — which composes
  naturally with the shipped σ-gated recipe and with E21's adaln
  amplitude concentration. The knob read adds: such a lever should
  target the **angle** (the residual direction), not amplitude
  rebalancing. **Confirmed and sharpened 2026-08-09** by E25.0: the
  σ-local requirement is now bin-level mandatory (a single direction
  in a ĝ-normalized frame is refuted — 25.0-2), and the lookup object
  exists per σ bin, route-shared, except the recorded 768/σ = 0.4333
  hole (25.0-1 PARTIAL).
- Population-level licensing only; every per-sample variant stays gated
  on E22 → 22.4 → E23a, unchanged.

### 24.4 figures (illustration only — verdicts live in `e24_axis.json`)

`e24_axis_fig.py`, reading only the committed digest:

- **`e24_axis_field.png`** — the axis *fan*: every gated verdict
  condition's B̂ / Ĉ drawn in the top-2 eigenplane of its Gram (plane
  share 0.92 / 0.93; per-arrow in-plane share annotated — the
  `fig_bc_plane` out-of-plane honesty convention). Route pairs at each σ
  nearly coincide; the σ = 0.7 quadruple (768/896 × e193/e194) is one
  bundle. Third panel: the descriptive co-rotation curves (B̂ vs ĝ
  rotation away from σ = 0.7) — no frame-relative claim (correctly so:
  E25.0-2 later showed the frame-relative version fails — the curves
  depict matched angles, not planar co-rotation).
- **`e24_bc_comb_rot.png`** — E19's `bc_comb` redrawn with the
  **measured** between-bin orientation (θ_B from the shared top-2
  plane) replacing the original "C always vertical" per-bin convention.
  Within-bin lengths (√2S, √2F in ‖g‖ units) and mutual angle
  (arccos ρ) stay exact as before, one true scale — σ = 0.3's ~4×
  larger legs are drawn as they are. C's side of B remains a drawing
  convention (annotated on the figure). What the upgrade makes visible:
  the pairs are *not* parallel copies along σ — the internal
  near-antiparallel geometry persists while the whole frame rotates
  (+61° → −16°) and shrinks.

- **`e24_bc_comb_3d.png`** — the rot comb lifted into 3D in the E19
  `bc_comb` visual style (axis-off, teal σ axis, direct labels, scale
  bar): pairs planted on the σ axis, with the shared top-2 eigenplane
  mapped rigidly onto the cross-plane (y, z) at each σ. One ‖g‖ scale
  for shape *and* magnitude — the B→C chord's length IS the realized
  gap |B + C| = √2R (per-bin values annotated; dashed envelope through
  the C⊥ tips, E19 convention). The frame is rotated so Ĉ(σ = 0.7)
  draws vertical — that bin reproduces the E19 comb exactly, and every
  other pair *twists about the σ axis* by its measured θ_B (+61° at
  σ = 0.3 → −16° at σ = 0.8333), which is the E24 upgrade made visible.

- **`e24_bc_comb_fan.png`** — the σ-rotation fan: every pair planted at
  one shared origin, σ entering purely as rotation (each pair drawn at
  its measured θ_B in the shared top-2 eigenplane, true ‖g‖ scale, light
  rings as the ruler; arrow color = σ). B thick into the origin, C thin
  out of it, gray chord = realized gap. The most convention-free of the
  comb family — it is literally the measured configuration overlaid in
  the shared plane (remaining conventions: plane projection, in-plane
  share in parens; C's side of B).

- **`e24_bc_comb_sphere.png`** — the fan made 3D, protractor-style: one
  shared origin, each pair rigidly rotated in its own plane so its
  **B+C chord runs along ẑ** (drawing convention, replacing E19's
  "C vertical"), and the pair's vertical half-plane placed at
  **azimuth = σ mapped onto 0–180°** (a σ *coordinate* — low σ left,
  high σ right; the measured θ_B is annotated per bin, not drawn).
  Within-bin geometry exact, one true ‖g‖ scale (unit half-shell as the
  ruler).

The E19 originals (`bc_comb*`, `fig_ledger_geometry`) are not modified —
they made no cross-bin orientation claim and remain valid; these are the
E24-informed successors for the paper to choose from.

## Cost ladder (planned → actual)

| item | planned | actual |
|---|---|---|
| 24.1 | CPU, ~minutes (memmap reads, ≤ ~10 GB resident) | 348 s, ~25 GB peak (first attempt with per-pair fp64 conversion was ~6× slower and crashed at serialization — rewritten as one chunked Gram pass; no GPU used or needed) |
| 24.2 | free (re-read of 24.1 scalars) | free |
| GPU | none | none |
