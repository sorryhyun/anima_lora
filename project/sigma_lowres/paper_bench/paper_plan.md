# paper_plan — revision direction for the sigma_lowres paper

Companion to `required_experiments.md` (E1–E8): that file says *what must
run*; this file says *what the manuscript becomes*. Drafted 2026-07-28
from the gap-native restructure discussion; supersedes nothing — the
current `paper/main.tex` stays the working draft until E1 lands.

**STATUS 2026-07-29 — Branch A written into `paper/main.tex`.** E1(a,b,c)
landed and rule 1 fired (512 floor confirmed debiased), so the §2 spine
below is now the manuscript: new §3 metrology (gap def + instrument +
finite-draw-bias subsection + ε\* and the safe(ε) definition), §4 account
(softened, "derive"→account; Table 1 + retrodict moved out), §5
Measurements (raw tables kept as marked historical record, new debiased
verdict-map + debiased floor tables, floor waterfall ledger, x-zero≡
endpoint equality, PI/iso-severity/depth re-anchored with the
one-signed-bias argument), §6 "The null read in gap units" (Table 1 +
scoring + E8.3 bridge stub), §7 two-maps framing + projected-ceiling
wording, limitations gain ε\*-relative + debiased-coverage items, repro
statement now matches what is public (paper_bench/runs in-repo; raw-run
tarball pending). Abstract/intro/conclusion rewritten (narrowed headline,
metrology as contribution, 14% as projected ceiling). Compiles clean
under tectonic (0 overfull, no broken refs). Still open in the
manuscript, each marked **[pending]** in place: E2 α-sweep confirmation,
E3 pooled-with-self-floors run, E4 A/B + `reenc_noise_floor.py`
(δ_reenc row + D(f) numbers), E7 membership probe, E8.3 overlay +
t\*(δ) figures, results tarball. Figures not yet regenerated in debiased
units (raw figs marked as raw in captions); Fig-1 enlargement + waterfall
+ overlay figures owed with E8 analysis.

---

## 1. Direction in one paragraph

The paper moves from "a two-term account confronts a spectral null" to
**gap-native metrology**: gap() is promoted from verdict quantity to the
paper's protagonist, and every claim is stated in its units and at its
measured resolution. Three-part spine: (1) what the instrument can
detect — a derived detectability threshold ε\*, replacing the ad-hoc
reenc ±0.04 band; (2) the guarantee region — the (route, σ) set where
the debiased gap is non-inferior to ε\* at one-sided 95%; (3) the prior
work's tolerances read *in gap units* — each published δ converted into
a predicted gap curve through our measured bridge, overlaid on the
measured curves. The two-term account is not demoted by this: it becomes
the load-bearing unit bridge (m/G^p) that part (3) needs, rather than a
free-standing theory chapter. Everything is restated in **debiased
units** (E1) — the restructure is conditional on E1 and must not be
written before it.

Headline narrows to: *spectral sufficiency of the noisy input does not
guarantee gradient equivalence under resolution substitution* — and the
converse asymmetry gets one explicit sentence: the VAE round trip has
measurable input-level error (D(f), the δ_reenc anchor) yet a gradient
cost below instrument resolution. Input-level error neither implies nor
excuses gradient-level cost; the paper's thesis, both directions.

## 2. New section spine (old → new)

| new | content | built from |
|---|---|---|
| §1 Intro | narrowed headline; 14% stated as *projected ceiling* until E4 | R4 fix |
| §2 Related | + explicit "neither SPD nor SwD claims naive gradient equivalence; we test the tempting extension, not their methods" | framing fix |
| §3 The metric and its resolution | gap definition, redraw floor, reenc control, **ε\*(N, D, floor) derivation**, debiased estimator (self-floors, attenuation correction), "safe" ≡ one-sided 95% CI below ε\* | old §4.1 + E1 + E8.1 |
| §4 What demotion can cost | two-term account, presented as a **first-order account whose terms are individually evidenced** (drop "derive" from abstract); null as model of the input branch; graph term | old §3, softened per review |
| §5 Measurements | gap curves, floor, input branch, governors, RoPE — all debiased. Floor section gains the **waterfall decomposition** (see §4 below) and the R2 relabel: "endpoint gap" → "high-noise endpoint gap"; x-zero is the graph-only control; 768's text stops claiming the plateau *is* the floor | old §4.2–4.6 + E1/E2 |
| §6 The null in gap units | predicted-gap-curve overlay per (δ, route) via the m_null/G^p bridge with A calibrated on the safe route; continuous t\*(δ) sweep figure; δ_reenc-anchored row. States plainly: "the null read through our bridge" | old §3.3 Table 1 + §4.7, upgraded by E8.3 |
| §7 Guarantee region + trainer | **two maps** — per-example (batch-1 worst case) and batch-aggregate (what the shipped trainer consumes); non-inferiority wording throughout; cost accounting as ceiling until E4 lands | old §5 + E3 |
| §8 Limitations / conclusion | + operating-point item resolved or measured by E7 if it ran | old §6–7 |

Reordering rationale: the confrontation (§6) moves *after* the
measurements because its bridge consumes measured G(σ) and a calibrated
A — presenting it before the curves would hide that dependence.

## 3. Claim-level revisions

**Strengthened (if E1 confirms):**
- The 512 floor, reported raw *and* debiased — survives the variance
  confound explicitly instead of ignoring it.
- "Never safe" restated per aggregation object (R3): per-example map vs
  batch-aggregate map, each with its own verdict column. The lenient
  read (batch-aggregate on 768/512) is reported either way — if 768
  stays out of band under the most lenient read, the verdict hardens;
  if it enters band, that is a finding, and the map says so.

**Softened / relabeled (regardless of E1):**
- Abstract "we first derive" → first-order account language; "ratio sets
  amplitude / token count sets floor" → "consistent with" unless E5's
  held-out prediction passes.
- Endpoint ≠ pure graph (R2): target still carries x at σ=1. All
  "graph-only" claims re-anchored to x-zero; 768's endpoint split into
  target-content share (E2's α-slope) + graph share.
- Every "14%" → "projected ceiling of ~14%" until E4's A/B lands.
- Reproducibility statement matched to what is actually public
  (results tarball or un-ignored verdict runs).

**New content:**
- ε\* derivation (§3) — the one genuinely new theory-ish element, and it
  is metrology, not physics.
- The reverse-asymmetry sentence (reenc: input error real, gradient cost
  ≤ band) + D(f) numbers once `reenc_noise_floor.py` runs.
- Null→gap overlay figure + t\*(δ) sweep (§6).
- Waterfall decomposition of the floor (§5).

## 4. The floor waterfall

Present Floor_e as an explicit additive ledger, one stacked bar or
4-column table per route, components measured by designated probes:

    Floor_e = reenc (≤ band, by control)
            + target-content share (E2 α-slope at endpoint)
            + RoPE_e (erased by PI at endpoint)
            + Resid_e (remainder; carries the capacity governor)

Numbers currently in hand (raw units; all to be re-stated debiased):
896 ≈ 0+0+0+0; 768 ≈ 0 + [E2] + 0.081 + ~0; 512 ≈ 0 + ~0.02 + 0.096 +
~0.22. The 768 row is the one E2 materially changes (half its endpoint
gap is target-content per the x-zero comparison). Caveat kept from the
discussion: reenc control (native decode→re-encode) is a *proxy* for the
pipeline cost demotion actually pays (source downscale→encode) — one
sentence, and the optional demote→re-promote arm (below) closes it
empirically if run.

## 5. Branch plan on E1's decision rule

The restructure is written three ways depending on E1 (pre-registered in
`required_experiments.md`):

- **Branch A — floors confirmed** (`gap_∞(512) ≥ 0.15` debiased): the
  spine above verbatim. E5 held-out prediction becomes the optional
  §4-strengthener.
- **Branch B — 768 melts into band**: safety map and abstract rewritten
  (768 safe at high σ; "never safe" claims retreat to 512); the
  absolute-size-governs-floor story loses its middle point and is
  re-examined against {896≈0, 512 large}; RoPE/depth decompositions
  re-checked in debiased units before any of §5's mechanism text
  survives.
- **Branch C — everything collapses into band**: the headline becomes
  the (still true, still novel) low/mid-σ result + the metrology
  contribution (ε\*, debiased estimator, the confound itself as a
  finding — "naive gap estimators manufacture token-count floors").
  Kill-switch honesty: the paper says its own earlier reads were
  variance-inflated.

Do not draft §5–§7 prose before E1(b); §3–§4 and the figure scaffolding
can be written now.

## 6. Figure/table plan

- **Fig 1** (enlarged, first page): measured debiased gap curves + the
  null's predicted curves at its published δ — the confrontation visible
  before any prose.
- **Fig: t\*(δ) sweep** — x: δ (log), y: t\*, three near-coincident route
  curves (family spread ≤ 0.13), measured boundaries as markers, δ_reenc
  as a vertical anchor line. Replaces/absorbs Table 1's sampled rows.
- **Fig: floor waterfall** (§4 above).
- **Table: two safety maps** (per-example / batch-aggregate), columns:
  route, Floor_e (debiased), σ\*, verdict at ε\*.
- Existing estimator-context and RAPSD figures survive in §3/§6.

## 7. Optional items (not gating, decide after E1)

- **Demote→re-promote arm** (pixel down→up at native grid, encode):
  Floor = 0 by construction, isolates the input branch across the full σ
  axis, and tests two-term additivity per bin (gap_demote ≈
  gap_repromote + Floor_e). Built-in falsification: its endpoint must be
  in band. Worth one run if E1 confirms the floors and E5 is attempted —
  it is the cleanest independent check of the form. Not yet in
  `required_experiments.md`; add as E9 if adopted.
- **E6/E7 generalization arms** per their existing entries.

## 8. Order of work

1. E1(a) endpoint sweep → E1(b,c) full debiased map (GATE).
2. Pick branch (A/B/C). Write §3 (ε\*) + §4 (account) — branch-invariant.
3. E2 + E3 runs; §5 prose + waterfall in debiased units.
4. E8 analysis: null→gap bridge + t\*(δ) figure → §6.
5. E4 (Phase 1b A/B + `reenc_noise_floor.py`) → clear pending markers,
   14% becomes measured, δ_reenc row lands.
6. Hygiene pass per `required_experiments.md` [FIX] list (abstract,
   pre-registered/post-hoc labels, reproducibility artifacts).
