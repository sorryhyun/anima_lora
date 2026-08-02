# E18 — the per-step → trajectory bridge: two-sample distinguishability × the E16 propagator

| | |
|---|---|
| **Status** | **PROPOSED 2026-08-02** — companion to [E16](../e16/): 16.0's deterministic twins are 18.2's checkpoint source, **contingent on a one-line 16.0 amendment** (save T/4 and T/2 intermediate checkpoints; the registered 16.0 design saves only finals). Numbered after E17 (DONE); file named `proposal.md` until adopted into the index. Verifier pass 2026-08-02 folded in (per-draw store, C2ST validity, E7 floor confound, scope corrections) |
| **Question** | What licenses reading the per-step (route, σ) map as a statement about *training outcomes*? Reframed: is the demoted per-step gradient **distribution** distinguishable from the native one *beyond its mean* — and does the surviving mean component compose through the trajectory propagator in a measured regime? |
| **Depends on** | [E16](../e16/) (the propagator identity + regime probe + twin checkpoints), [E4](../e4/)'s [`claim_accumulated_bias.md`](../e4/claim_accumulated_bias.md) (the three surviving worries, mapped 1:1 onto 18.0/18.1/18.2), [E9](../e9/) (paired in-window difference structure), [E11](../e11/) (probe harness + probe set), [E15](../e15/) (per-sample second moment — see Groundings for why it does *not* pre-empt 18.0), [E7](../e7/) (cross-checkpoint map replication — prior for 18.2), `methods.md` `--deterministic` (no chaos floor in 18.2) |
| **Instrument** | probe harness **plus a probe-kernel delta**: `bench/sigma_probe/kernel.py::grad_estimate_binned` accumulates arm sums in place and never materializes a per-draw gradient (E15 g9 records the same), so 18.0/18.1 need a per-draw projection dump added to the kernel + a GPU recompute — there is no CPU-only path. Readout: logistic classifier two-sample test (C2ST, Lopez-Paz & Oquab 2017) on projected gradients with a **within-pair** permutation null; `bench/compare_ckpt_dw.py` and the endpoint probe (with per-checkpoint self-floors) for 18.2 |
| **In the paper** | a **bridge lemma** paragraph in §2.3 (or opening §5) + one appendix subsection; rewrites the "Gradient-level, not end-task" limitation from a concession into a decomposition with three measured facts and one bounded residual |

## The reframe (and what it does *not* claim)

The map certifies per-step, first-order equivalence of the **mean**
gradient at instrument resolution. `claim_accumulated_bias.md` records
the three worries that survive that certification:

1. a coherent sub-ε* directional bias accumulating over ~10⁴ steps;
2. **direction match ≠ trajectory match** — `gnorm` differs across
   arms, so Adam's second-moment state can diverge under exact
   direction match;
3. per-bin equivalence composes only to first order (curvature +
   optimizer state).

The reframe: treat the demoted arm's per-step gradient as a *sample
from a distribution* and ask whether that distribution is
distinguishable from the native one. Decompose the per-step difference
into

- a **mean shift** b_t(σ) — the object the map already measures, and
  the only component that accumulates coherently: to first order the
  final-weight deviation is E16's propagator sum
  Δθ_T ≈ Σ_t M_{t→T} · η_t · b_t;
- a **mean-centered fluctuation** component — distribution shape and
  variance at fixed mean. In θ it washes out at O(1/√(BT)); its one
  non-washing channel is Adam's per-coordinate second-moment state ν,
  which consumes it at full weight.

The bridge from "gradient similarity" to "actual weight change" then
needs exactly **three measured facts**, no accumulation analysis:

| fact | claim | measured by |
|---|---|---|
| (i) | the mean shift b_t is small in the certified window | the map (done) |
| (ii) | the fluctuation channel is at-chance vs the reenc control, and Adam-inert | **18.0 / 18.1 (new)** |
| (iii) | the propagator regime on the bias direction is washout or linear | **E16 16.0** |
| residual | beyond-first-order feedback (the map is a θ₀ statement) | **18.2** bounds it |

Neither the map nor E16 alone closes worry-list above; the map says
nothing about composition, E16 takes b_t as given and says nothing
about per-step channels beyond the mean. Interlocked, they do — and a
failure of any single row localizes which sentence of the bridge lemma
is wrong, instead of collapsing the whole claim.

**Not claimed**: the TV/data-processing composition bound (per-step
distributional distance δ ⇒ trajectory distance ≤ Tδ). It is formally
valid and quantitatively vacuous at T ≈ 10⁴ (it would require
certifying δ ≲ 10⁻⁴, below any reachable instrument resolution) — and
a coherent mean shift is precisely the term that is near-invisible to
per-step TV while composing linearly. The composition claim here runs
through the decomposition + the measured regime, never through DPI.

**Terminology**: "two-sample test / distinguishability" throughout;
never "real/fake gradients" in the manuscript — the GAN/DMD collision
(this repo ships a fake-critic distiller) invites exactly the wrong
reading.

## 18.0 — mean-centered two-sample probe (the distinguishability test)

At the map's checkpoint, per (image, σ-bin): D CRN-paired draws per
arm {src, reenc, dem-route}. Features per draw: a fixed k-dim random
projection of the flattened adapter gradient (shared across arms,
frozen seed) + per-layer gradient norms. Two variants, both scored as
logistic C2ST AUC with a label-permutation null:

- **raw** — sanity gate: must reproduce the map's separations (512
  separates hard, in-window 896 ≈ reenc). If it doesn't, the features
  are too weak — fix before reading the centered variant.
- **centered + unit-normalized** — the new estimand: per-arm mean
  removed, draws normalized. Note this is a **stronger** invariance
  than the paper's — the cosine estimand is blind to parallel
  rescaling of the *mean* gradient only, whereas per-draw
  normalization also quotients out per-draw magnitude structure. That
  is a deliberate, stated choice (isolate distribution *shape* at
  fixed mean; route every magnitude question to 18.1), and the bridge
  paragraph must state it as such rather than claim it "keeps the
  paper's invariance class".

The reenc arm's centered AUC is the control effect size, playing the
same role d_reenc plays for the map.

**Validity constraints (non-negotiable).** CRN partners share image,
ε, and σ, and E15's per-image paired ‖Δ‖ is O(‖g‖) — a classifier
that sees one partner in train and the other in test can exploit
image identity and inflate AUC toward the *paired* effect size. So:
label permutations **within CRN pairs only**; CV folds **grouped by
image** (no image straddles a split); and the effective sample size
for power purposes is the number of probe images (~tens), not
images × draws. The calibration below must run under the same
grouping.

**Instrument calibration (the ε* analogue).** C2ST failure is
one-sided — "at chance *at this power*". Calibrate power by injecting
known synthetic shifts into src draws (scaled copies of the measured
Δḡ direction and random directions) and finding the smallest
detectable effect; report the detection threshold alongside every
at-chance verdict, exactly as ε* is reported for the map.

**Pre-registered reads** (frozen before the run):
- certified window (1024→896, σ ∈ (0.5, 0.94]): centered AUC
  indistinguishable from the reenc control's AUC at the calibrated
  power → the fluctuation channel carries nothing demotion-specific;
- 512 all-σ: positive control, expected to separate in both variants;
- centered AUC ≫ control *inside* the certified window → the map
  under-describes the per-step change; the safe-window claim then
  leans on 18.1 showing the surplus is Adam-inert, and the bridge
  paragraph must say so.

## 18.1 — the Adam functional (worry #2, tested where it acts)

The generic discriminator answers "is anything different"; Adam only
consumes one specific functional. Test it directly: per-coordinate
second-moment ratio ρ_j = E[g²_dem]_j / E[g²_src]_j (shrunk,
summarized per layer), plus the induced preconditioner distortion —
cosine between the native mean step preconditioned by ν_src vs by
ν_dem at Adam's fixed point. This is `claim_accumulated_bias.md`'s
"gnorm differs ⇒ ν-state can diverge" made quantitative.

**Pre-registered**: in-window ρ spread within the reenc control's
spread ⇒ ν divergence bounded by the control; 768/512 expected to
fail (gnorm differences are measured facts there). Scope honesty: a
preconditioner distortion below the map's ε* closes the Limitations
caveat **only for the shipped optimizer** (AdamW's ν functional is
what 18.1 measures); the general "a magnitude-sensitive estimand
could apportion the endpoint differently" sentence is about the
estimand class itself and survives regardless — 18.1 gives it a
number for the one optimizer the trainer actually runs.

## 18.2 — on-trajectory stationarity (rides E16's twins, no new training)

The map is measured at one θ₀; feedback — trajectories drift, and the
gap is re-evaluated at drifted weights — is the beyond-first-order
residual no per-step argument closes. Bound it empirically: at E16
16.0's deterministic twin checkpoints (native + early/late/spread,
t ∈ {T/4, T/2, T} — **requires amending 16.0 to save the two
intermediate checkpoints**, its registered design saves finals only),
re-run a reduced probe grid (the endpoint sweep + 2–3 interior bins,
in-window route + 768 off-map control) at each checkpoint's weights.

**Floors are per-checkpoint, mandatory.** E7's NEW FACT is that the
redraw-floor *level* is checkpoint-dependent (cos_floor 0.73 vs 0.50
across its two adapters) — a raw "gap within the θ₀ map's band" read
across checkpoints would confound floor drift with gap drift. Every
checkpoint gets its own self-floors and the comparison is stated in
debiased units only; expected resolution is E7-like (~0.03–0.05 per
bin, i.e. of order ε* itself), so 18.2 bounds *gross* feedback, not
sub-ε* drift.

**Horizon honesty.** 480-step tenth-scale twins bound feedback over
480 steps; the accumulated-bias worry names ~10⁴. A stationary read
here is evidence, not proof, for production horizons — stated as
such, with the E7 factorial (independently *trained* checkpoints, not
trajectory-drifted ones, map shape replicated with interaction below
resolution) as the complementary long-range prior.

**Pre-registered mapping**:
- **stationary** (per-checkpoint debiased gaps consistent across t at
  the per-checkpoint resolution) + E16 regime ∈ {linear, washout} ⇒
  the bridge closes at twin scale: first-order composition licensed,
  gross feedback bounded over the measured horizon;
- **growing along demoted arms** ⇒ amplification-by-feedback: the map
  is a θ₀ statement only; §5's trainer claim stays purely empirical
  (E4 yardstick) and the bridge paragraph reports the growth rate
  instead of a composition claim;
- discordant (grows on native too) ⇒ checkpoint drift dominates route
  effects — record, widen E7's factorial before interpreting.

## Why this is more robust *combined* with E16 (the user's instinct, made explicit)

E16 measures **how a fixed per-step bias composes** (the M_{t→T}
regime) but treats b_t as a scalar given. 18.0/18.1 certify that the
mean is the **only** per-step channel at instrument power (and that
the one non-washing side channel, ν, is inert). 18.2 certifies the
map is **stationary along the very trajectories E16 orders**. Each
alone is attackable — "you measured composition of a bias you didn't
fully characterize" / "you characterized a step at one checkpoint" —
interlocked they form one lemma with every premise measured:

> Δθ_T ≈ Σ_t M_{t→T} η_t b_t, with b_t = the measured map (i),
> no other channel at instrument power (ii), M-regime measured (iii),
> stationarity of (i) along the trajectory bounded (18.2).

## Manuscript delta

- §2.3 (after the aggregations paragraph) or §5 opening: 4–6 sentence
  bridge paragraph + the displayed accumulation identity (shared with
  E16's theory block — one derivation, cited twice).
- Limitations, "Gradient-level, not end-task": rewritten from
  concession to decomposition — three measured facts + one bounded
  residual, with the honest one-sidedness (calibrated power) stated
  in the same breath as ε*.
- If 16.0 lands "linear-dilution" and 18.x land at-chance/stationary:
  §5 gains the sentence the trainer currently can't say — *why* a
  per-step certification was the right thing to gate on.

## Groundings

- `claim_accumulated_bias.md` — the three surviving worries map 1:1
  to 18.0 (coherent accumulation → decomposition), 18.1 (ν-state),
  18.2 (composition beyond first order). **Positioning**: the
  *recorded* closer of that claim is the outcome-level full-band CMMD
  rescoring, and E15's verdict independently endorses the E4 A/B —
  not any per-step read — as the certification instrument. E18 does
  not replace either; it is the *paper-bridge* companion (why the
  per-step map was the right gate), and it sequences behind the
  already-owed items (E3 pooled grid, E13 write-in, E11 `--uncond`)
  unless the manuscript timeline pulls it forward.
- **E15 does not pre-empt 18.0 — and constrains it even less than a
  "measured second moment" would.** E15's E‖g_nat − g_dem‖² ≈
  V_total/2 is a *lower bound* assembled from mean-difference norms
  (per-draw dispersion absent by construction, its g9/h4), and it is
  a *paired* object under CRN. 18.0's estimand is *marginal*
  distinguishability — can a classifier tell which arm a single draw
  came from — which is what the optimizer actually experiences (it
  never computes both arms). Large paired difference with at-chance
  marginals is exactly E15's own "aggregate-coherence" reading:
  per-sample differences are O(‖g‖) and cancel; what survives in
  aggregate is the small mean — the component the map measures.
- E9: in-window paired difference is 20–30% of the additive |B|+|C|
  sum (I < 0 cancellation) — the CRN pairing 18.0 inherits.
- E7 factorial: map *shape* replicates across two independently
  trained, style-disjoint checkpoints with adapter × probe-style
  interaction below resolution — the long-range prior for 18.2; its
  checkpoint-dependent floor *level* (0.73 vs 0.50) is why 18.2
  mandates per-checkpoint self-floors.
- E16 16.0: identical eligible sets across placement arms (E4 CRN
  property) + `--deterministic` twins (no 0.413 chaos floor) — the
  checkpoints 18.2 probes are clean, once intermediate saves are
  added to its design.
- E14 fp32 ledger: stores **per-bin arm sums only** (240 vectors =
  16 arm-keys × 15 bins, accumulated over images and draws;
  `kernel.py::grad_estimate_binned` never materializes a per-draw
  gradient — E15 g9). Nothing in it serves 18.0/18.1 directly; the
  run is also still incomplete (31/40 images) at proposal time.

## Cost

- **18.0/18.1**: one GPU probe recompute is unavoidable (the E14
  store is arm-sums-only, see Groundings): D draws × 3–4 arms ×
  reduced σ grid on a probe subset, daemon-queued, few GPU-hours.
  Requires a probe-kernel delta — a per-draw hook in
  `grad_estimate_binned` that projects each draw's gradient to k dims
  (and accumulates per-coordinate g² for 18.1) *before* the in-place
  sum, so the dump stays MBs, not the 78 GB fp32 regime. Readout
  itself is CPU-trivial.
- **18.2**: 3 checkpoints × 4 arms × reduced grid **plus
  per-checkpoint self-floors** (the floor draws dominate this cost —
  budget it like a small E7 cell per checkpoint, not a bare grid);
  zero new training (twins exist once 16.0 runs with intermediate
  saves).
- Trainer delta: none for 18.x; one line in 16.0's runner to save
  T/4 and T/2 checkpoints. New code: the kernel per-draw hook + the
  C2ST readout script (CPU, testable on synthetic Gaussians with
  grouped folds).

## Kill switches / failure honesty

- 18.0 raw variant fails to reproduce the map ⇒ features insufficient;
  no verdicts read until fixed.
- Centered AUC separates in-window and 18.1 shows ν distortion above
  ε* ⇒ the bridge does **not** close; the paper keeps the current
  empirical stance and this record becomes the documented reason the
  limitation paragraph survives.
- 18.2 growth on demoted arms ⇒ composition claim withdrawn; the
  growth curve itself is the finding (feedback rate of the demotion
  bias), reported as such.
