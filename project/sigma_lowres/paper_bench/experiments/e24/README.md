# E24 — cancellation-axis geometry: is the cancelled direction one global mode?

| | |
|---|---|
| **Status** | **PLANNED — pre-registered 2026-08-08** (committed before the `e24_axis.py` instrument exists, mirroring 21/22). CPU-only; no GPU item anywhere in this experiment. |
| **Question** | ρ̄ ≈ −0.91 means that per (route, σ) the pooled legs B⊥, C⊥ span a nearly 1-D subspace — a "cancellation axis" along which the data damage and the graph response slide against each other. Everything so far measures the *angle inside* each condition; nothing yet asks whether the axis itself is **one shared direction** across σ, routes, and runs ("a scale mode of the adapter-gradient space" — the geometric reading: demotion as an approximate symmetry the network absorbs, residual = failure of equivariance), or a per-condition direction that merely always cancels locally. Secondary, free from the same scalars: what actually dominates the residual gap — the incomplete **angle** (ρ > −1) or the **amplitude mismatch** (\|B\| ≠ \|C\|)? That pins which knob any population-level lever should target, *before* one is proposed. |
| **Depends on** | [E19](../e19/) 19.3 + [E19.4] surviving `arm_sums/` stores and [E22](../e22/) 22.1's `--keep_arm_sums` store (all three verified on disk 2026-08-08, same adapter `anima_soup_sincos` — one operating point); `paper_bench/vector_ledger.py` (leg/debias conventions, `Sums` loader); [E21](../e21/) (LOCAL — the axis question is the *cross-condition* complement of E21's *within-condition* cell read); [E20](../e20/) 20.4 (closed: no ledger-derived objective term — this experiment derives nothing, it only measures geometry). |
| **Instruments** | 24.1 `e24_axis.py` (CPU; cross-condition axis cosines + subspace rank + figure); 24.2 free re-read of 24.1's scalars (residual knob decomposition). |
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

## Cost ladder

| item | cost |
|---|---|
| 24.1 | CPU, ~minutes (memmap reads over the three stores, ≤ ~10 GB resident fp32 condition means) |
| 24.2 | free (re-read of 24.1 scalars) |
| GPU | **none** |
