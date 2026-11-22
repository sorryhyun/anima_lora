# DAVE — DC Attenuation for diVersity Enhancement (Anima port)

Training-free, representation-level intervention to recover same-prompt sample
diversity: a port of *Breaking the Lock-in* (ICML'26, arXiv 2606.06813). Phases 0–2a
were built from the abstract and teaser figure; Phases 2b–2e align it with the full paper.

## The idea

Text-to-image samples from one prompt look overly similar. DAVE's diagnosis:
the **DC component** of intermediate Transformer-block features — the spatial
average `μ^ℓ` of `h^ℓ` over the patch grid, per channel — converges across seeds,
pinning global layout and collapsing later variation. The fix is a per-block
output rewrite that attenuates the DC while leaving the AC residual intact:

```
ĥ^ℓ = α · μ^ℓ + (h^ℓ − μ^ℓ)        α < 1,  μ^ℓ = mean over (T,H,W)
```

The teaser figure pairs this with a **Power Ratio** (`‖μ‖²/‖h‖²`) gate — i.e. the
attenuation is meant to be *block-selective*, hitting only where the DC carries
energy.

### How it maps onto Anima

- Anima blocks emit 5D `(B, T, H, W, D)`; the DC is `h.mean(dim=(1,2,3))`. The
  intervention is a **post-`forward` hook** (forward_hook-not-override invariant),
  branchless (`h − (1−α)·μ`, α=1 → exact no-op) so it survives `compile_blocks()`.
- It is a *representation* edit inside the forward, **not** a sampler-boundary
  correction like CNS/SMC — so it lives as a hook + per-step model buffer
  (the Spectrum/mod-guidance pattern), not in `library/inference/corrections/`'s
  post-step callback shape.

## Phase 0 — premise probe ✅ DONE (2026-06-05)

`probe_dc_convergence.py` — read-only. Generates the **`make test` prompt** across
N seeds (real `generate()`, model injected via `shared_models`, hooks on every
block), and per (step, block) measures cross-seed cosine similarity of the **DC**
vs the **AC residual `h−μ`** (pooled to GxG, DC-disjoint by construction), plus
the DC power ratio. Verdict restricted to **DC-heavy blocks** (early-step power
ratio ≥ 0.30) — block-averaging smears the bimodal per-block power structure.

**Question:** is the DC the shared/locked component while the AC carries the
seed-specific diversity? If DC sim ≈ AC sim even where the DC has energy, there's
nothing to unlock and we stop.

### Result — PASS (`results/20260605-1442/`, 8 seeds × 24 steps, CFG 4.0)

| early steps | DC sim | AC sim | gap |
|---|---|---|---|
| all blocks | **0.989** | 0.482 | +0.51 |
| DC-heavy blocks | **0.9993** | 0.611 | +0.39 |

The DC is near-perfectly cross-seed-shared (it's the conditioning); the AC
residual that holds seed structure sits at 0.48–0.61. DAVE's decomposition is
real on Anima — there is a shared, energetically-significant channel whose
attenuation would let the diverse AC breathe.

> ⚠️ **False start, kept as a lesson:** the first run (`results/20260605-1432/`)
> compared DC against a plain **coarse 2×2 pool**, which *contains* the DC. The
> gap collapsed to +0.022 and the auto-verdict printed `premise_holds: false`.
> It was a measurement artifact — DC and AC must be **disjoint** (`h−μ`) to be
> compared. Don't reintroduce a DC-containing AC proxy.

### What the probe settled about *where* to intervene

From the power-ratio heatmap + the DC−AC gap heatmap (`dc_minus_ac_gap.png`):

- **Block-selective, not global.** Strongest targets (high DC power ∩ high
  DC−AC gap) are the **late blocks ~19–27** and mid blocks ≥11.
- **Block 0 is a false target.** DC-heavy by power, but its AC is *also*
  cross-seed-locked (gap ≈ 0) — attenuating DC there perturbs a shared signal
  without unlocking anything. Exclude it.
- **Temporal story differs from the paper.** On Anima the DC is shared at *all*
  steps (sim ≈0.99 even at σ≈1.0), not "converging early." Its *energy* is small
  early (ratio 0.28) and grows late (0.63, σ<0.45). So the paper's "attenuate
  early" isn't obviously the lever here — the early-vs-late attenuation window is
  an **open knob**, not a settled default.

Artifacts per run: `result.json`, `convergence_curves.png` (all vs DC-heavy),
`power_ratio_heatmap.png`, `dc_minus_ac_gap.png`, `per_block.npz` (the (S,L)
matrices — reuse instead of re-running).

## Build-out (Phases 1–2, both done)

The probe localized the targets and the saved `per_block.npz` let us derive the
**per-block α schedule offline (no new GPU run)**: weight the attenuation by
`power_ratio × max(0, DC−AC gap)` so block 0 and AC-shared regions self-zero —
that *is* the figure's "Power Ratio" gate, fit from our own data. Phase 1 was that
desk exercise; Phase 2 wired the intervention and eyeballed it.

### Phase 1 — derive the per-block α mask ✅ DONE (2026-06-05)
`derive_alpha_mask.py` — offline, no GPU. Turns `per_block.npz` into a per-block
weight `w(ℓ) = power_ratio(ℓ) · max(0, DC−AC gap(ℓ))` (both averaged over the
early window), normalized to `[0, 1]`, shipped as
`networks/calibration/dave_alpha.npz`. The product self-zeros exactly the two
families DAVE must not touch — **block 0** (gap ≈ 0) and the **early-mid AC-shared
blocks 1–8** (~zero DC power). Sanity gates pass: block 0 → 0.026, peak at **block
22 (1.00)**, late blocks 19–27 dominate (0.59–1.00). The figure's "Power Ratio"
gate, fit from our own data.

At inference `--dave` reads `w(ℓ)` and forms the attenuation factor
`(1−α_ℓ) = strength · w(ℓ)`, so `--dave_strength` is a single live knob and the
self-zeroed blocks stay no-ops.

### Phase 2 — `--dave` intervention ✅ DONE (2026-06-05) — eyeballed, not bench'd
**Shipped** (the README design, faithfully): per-block **post-`forward` hook**
`h ← h − (1−α_ℓ)·μ` (`library/inference/corrections/dave.py`), branchless (α=1 →
no-op, survives `compile_blocks()`), + a per-step model buffer `_dave_cur_sigma`
restamped from the timestep so the hooks can σ-gate without seeing the step.
Standard loop only (no Spectrum/SPD compose), applied to both CFG passes. Knobs:
`--dave_strength`, `--dave_sigma_lo/hi`, `--dave_block_lo/hi`, `--dave_tau` (see
Phase 2b). Make lever:
`make test DAVE=1 [DAVE_STRENGTH= DAVE_SIGMA='lo,hi' DAVE_TAU= DAVE_BLOCKS='lo,hi']`.

> **`strength` is NOT the paper's α.** `strength = (1−α)` at the most-implicated
> block (w=1) — the *fraction of DC removed*, not kept (the paper's `ĥ=α·μ+(h−μ)`
> keeps α). So `α_peak = 1 − strength`: our `s0.5 ⇔ α=0.5`, `s0.8 ⇔ α=0.2`, landing
> on the paper's recommended α∈[0.2,0.5]. Non-peak blocks scale by `w(ℓ)<1`, so they
> remove less (α closer to 1).

We chose to **eyeball** the diversity/quality tradeoff (`eyeball.py`, 2 seeds ×
{baseline, s0.05/s0.10 all-blocks, s0.10/s0.20 mid-only}) rather than build the
PE-Core diversity bench — there's no text tower wired for an honest "alignment
held constant" number, and the failure mode turned out to be visually obvious.

**Findings:**
- **The default must be small.** At `strength=0.6` the output collapses to a
  color-blown **grid** — the mask weights the *latest* blocks hardest, but there
  the DC carries 60–97% of the energy (it *is* the image content, not a removable
  shared-layout signal), so attenuating it across 9 blocks before `final_layer`
  goes fully off-manifold. Default is now **0.1**.
- **The knee is `s≈0.05–0.10` all-blocks.** `s0.05` is a subtle tone/background
  nudge; `s0.10` gives genuine **per-seed recomposition** (framing pulls back,
  layout changes) while staying coherent and keeping the "ANIMA" sign legible.
- **The cost is tonal flatness, not collapse.** Removing the global per-channel
  DC removes tonal/color mean, so quality degrades *gracefully* as **posterization**
  (flatter range, higher saturation) that scales with strength — not the grid.
- **Diversity lives in the late blocks, not the mid blocks.** Mid-only (9–18) is a
  weak lever: `s0.10` ≈ baseline, `s0.20` warms tone and **breaks the sign text
  before it recomposes**. This matches the probe (late blocks had the highest DC
  power ∩ DC−AC gap); the "mid is the safe lever" guess was wrong.

**Verdict — plausibly ports, with a caveat.** There is a usable regime (≈0.05–0.10
all-blocks) where composition diversifies at near-fixed quality, so DAVE is not a
no-go on Anima. But it is **not free**: above ~0.1 the global-DC removal trades
tonal richness for diversity. Open follow-up: **RMS-renormalize after the DC
subtraction** to conserve tonal energy (recolor-not-remove, à la CNS) — would test
whether the posterization is separable from the diversity gain.

### Phase 2b — the temporal cutoff τ ✅ overturns the "stay gentle" caveat (2026-06-09)

The real paper went public (**arXiv 2606.06813**, *Breaking the Lock-in*, Kwon/Lee/
Choi). Its load-bearing lever is the one our Phase-2 default ignored: a **temporal
cutoff τ** — attenuate **only the first ~15–20% of denoising steps** (τ=0.15), not
the whole schedule. Wired as `--dave_tau` (`library/inference/corrections/dave.py`):
it reconstructs the live `--infer_steps`/`--flow_shift` σ-grid and converts τ→σ_lo,
so "first 15% of steps" tracks the actual schedule (at flow_shift=3 that's σ∈[0.95,
1.0] — an unguessable raw-σ window, which is why a step-fraction knob was needed).

**This is what made the intense dose safe.** Phase 2 found `s0.6` full-schedule
collapses to a grid — because at *late* steps the late-block DC carries 60–97% of
the energy (it's content). Confined to the *early* steps, that same late-block DC is
the low-energy (0.28), cross-seed-shared signal that's safe to nuke. Re-swept
(`eyeball.py`, 3 seeds, the **same power·gap mask**, just τ-gated):

| config | diversity vs baseline | cost |
|---|---|---|
| baseline | none — near-identical clones | — |
| τ0.15 · s0.30 | mild (reframing, added scene elements) | text legible, safe |
| **τ0.15 · s0.50** | **strong** — pose / crop→full-body / sign-type change | warm tint + some sign-text garble |
| τ0.15 · s0.80 | strong | warmer tint; text seed-dependent, still **coherent** |
| τ0.20 · s0.50 | strong | **over-attenuates** → poster-flat, bold outlines |

**Findings:**
- **No grid collapse at s0.5–0.8 once τ-gated.** The exact dose that blew up on the
  full schedule stays detailed and coherent confined to the early window. The τ-gate,
  not the dose, was the missing piece — *the Phase-2 "default must stay ≤0.1"
  conclusion was an artifact of full-schedule application.*
- **Sweet spot: τ=0.15, strength ≈0.3–0.5** (= paper α∈[0.5,0.7]) — far more
  per-seed recomposition than the shipped `s0.10` full-schedule default, at
  comparable coherence. **τ must not exceed ~0.15**; τ=0.20 tips into posterization.
- **Residual cost is a warm tonal shift + sign-text wobble**, traced to the mask
  *still* weighting the late content-blocks (22–27) even when early-gated. Confirms
  the mask-design lead below: the power·gap weighting points at the recolor-prone
  blocks; a **gap-emphasis** mask (lifts blocks 9–11: DC-locked, AC-free, low-energy)
  or the paper's flat `DC_sim≥0.99` pool should hold the diversity at lower tonal cost.

**Two things the probe data settled against the paper:** (1) the paper's pool rule
`DC_sim≥0.99` would *include block 0*, but on Anima block 0's **AC is also
seed-locked** (sim 0.984, gap +0.01) — attenuating it unlocks nothing; our `gap`
term correctly zeroes it. (2) The paper's pool on our data is 22/28 blocks (≥block 7),
far broader than our ~6 power-weighted targets — "their schedule" is *both* the τ
cutoff *and* a broad, flat, non-power-weighted block pool.

### Phase 2c — the dots/tint/text damage is ALL the late blocks (2026-06-09)

The Phase-2b residual cost wasn't one artifact, it was three (patch-grid **dots** in
foliage, a warm **tonal tint**, **sign-text garble**) — and a single block-range test
showed they share one cause. Holding τ0.15·s0.50 fixed and varying only `--dave_block_hi`
(`eyeball.py`, 3 seeds):

| block range | diversity | dots | tint | text |
|---|---|---|---|---|
| `0–27` (all) | max | dotted foliage | warm/orange | garbled |
| **`0–18`** (spare 19–27) | **strong, held** | **gone — smooth** | **gone** | **legible** |
| `0–11` (mid-only) | weak (≈baseline) | gone | gone | clean |

**All three costs were the late content-blocks 19–27.** Sparing them removed the dots,
the tint *and* the text-garble at once, while the recomposition survived. Why the dots
specifically: subtracting the per-channel spatial DC shrinks the dominant component, so
the next block's per-token LayerNorm/adaLN **renormalizes and boosts the AC residual**;
the highest spatial frequency the token stream can carry is the per-patch grid (~64×64
for 1024px), so the boosted HF energy aliases onto that grid as stippling — worst in
already-HF regions (foliage), invisible in smooth ones (sky → just the tint). Attenuating
blocks 19–27 imprints this **directly** on the output patches (they feed `final_layer →
unpatchify` with no downstream attention left to diffuse it); attenuating mid blocks lets
~15 blocks of attention smear it out. Hence: late blocks = patch-imprint, mid blocks = clean.

**Where the diversity actually lives: blocks ~12–18.** `0–11` alone barely moves off
baseline (the high-gap blocks 9–11 are too weak a lever), so the recomposition in `0–18`
comes from the mid band 12–18 — moderate gap (~0.18), moderate power — *not* the high-gap
early-mid blocks and *not* the late content-blocks.

**This inverts the shipped mask.** The current `power·gap` mask weights 19–27 hardest
(peak 22) — it concentrates the lever on the artifact source and underweights the clean
band. **Production setting → `τ0.15, s≈0.5, blocks 0–18`** (a hard cap at 18, zeroing
19–27 in the mask). Open: with the artifact ceiling lifted, retest **s0.8 × blk0–18** —
higher strength may now buy more diversity without the damage. The `gap²·power` /
paper-pool mask-design sweep is now mostly mooted by the block cap, but RMS-renorm is
still worth a look as a second-order tonal cleanup. `--dave_strength`'s code default
stays 0.1 until the capped mask is wired.

### Phase 2d — flat statistical pool + production defaults ✅ (2026-06-09)

The full paper's **Appendix A.1.1** ("Block Selection") settled the mask design that
Phase 2b/2c were converging on by eyeball. The paper selects its block pool **L**
*statistically*, not by a power gate: pairwise cross-seed cosine of the DC across
many seeds, BH-FDR-tested, **threshold ≥ 0.99** — a **flat** membership, then
**"final-stage blocks are excluded from the candidate pools"** because they have
*limited impact on diversity*. That is exactly our Phase-2c finding, arrived at
independently (on Anima the final blocks aren't merely useless — they imprint the
patch-grid **dots**), now with the authors' own rule as backing.

`derive_alpha_mask.py` rewritten to that rule (no GPU re-run — same probe):

    member(ℓ) = DC_sim(ℓ) ≥ 0.99   AND   gap(ℓ) > 0.03   AND   ℓ ≤ 18
    w(ℓ)      = 1.0 if member else 0.0          # flat, not power-weighted

- `DC_sim ≥ 0.99` — the paper's statistical pool (early window).
- `gap > 0.03` — **Anima-specific correction the paper lacks**: block 0 passes the
  paper's 0.99 threshold (DC_sim 0.998) but its **AC is also seed-locked** (gap
  0.017), so attenuating it unlocks nothing. The gap term drops it. (On SD3 the
  paper keeps block 0 — its AC isn't locked there.)
- `ℓ ≤ 18` — the paper's "exclude final-stage" cap = our dot fix, baked into the
  mask so `--dave` "auto" can't reintroduce the dots.

**Shipped mask → flat blocks 8–18** (`networks/calibration/dave_alpha.npz`). This
**inverts** the old `power·gap` mask (which peaked at block 22, i.e. on the dot
source) and **structurally forecloses the dots** — the single most critical
degradation — rather than relying on a runtime `--dave_block_hi` override.

**Production defaults** (`library/inference/args.py`): `--dave_tau` 0 → **0.15**
(paper τ), `--dave_strength` 0.1 → 0.5 → **0.3 (final)**. The 0.5 bump (= paper
α=0.5) was walked back after the production-mask sweep below — Anima's text + anime
hands need a gentler dose than the paper's text-free ImageNet/COCO eval implies.

**Production sweep — s0.3 vs s0.5 vs s0.8** (`output/tests/dave/`, flat 8–18 mask,
τ0.15, 3 seeds). Consistent across all seeds:

| dose | diversity | rendered "ANIMA" | hands / structure |
|---|---|---|---|
| **τ0.15 · s0.30** | strong — pose/framing/hair all vary, clearly off-baseline | **legible** 3/3 | coherent 3/3 |
| τ0.15 · s0.50 | stronger + heavy restyle | **garbled** 2/3 ("NNM", melted) | merged/distorted fingers 2/3 |
| τ0.15 · s0.80 | max | warm tint returns, hat artifacts | proportions drift |

**s0.3 buys most of the diversity at much better fidelity** — the hand/text breakage
is specifically the s0.5→s0.3 delta. The dots (the original critical degradation) are
gone at every dose (flat-pool + cap + τ did that); the remaining tradeoff is purely
diversity-vs-text/hand-coherence.

### Phase 2e — τ0.10 dominates τ0.15: the damage is *window width*, not dose (2026-06-09)

Tightening the window to **τ0.10** (first ~10% of steps) at the flat 8–18 mask, 3
seeds:

| config | diversity | rendered "ANIMA" | hands |
|---|---|---|---|
| τ0.10 · s0.30 | good (framing/hair/pose vary) | legible (1/3 mildly melty) | clean 3/3 |
| **τ0.10 · s0.80** | **strongest of any sweep** (incl. autumn/season swap) | legible, stylized 3/3 | acceptable 3/3 |
| τ0.15 · s0.50 (prior) | strong | garbled 2/3 | distorted 2/3 |

**The hand/text damage tracks how many steps the dose touches, not the dose
magnitude.** Confined to the first 10%, even s0.80 holds legible text + clean hands
while diversifying *harder* than τ0.15·s0.30 — cleaner than s0.50 was at τ0.15. So
**τ0.10 strictly dominates τ0.15 at equal dose**, and the earlier "lower the dose to
0.3" knee was really "the window was too wide." This reframes the two knobs: tighten
τ first, then spend the recovered headroom on dose.

**Production defaults (final): flat 8–18 pool, `--dave_tau 0.10`, `--dave_strength
0.3`.** τ dropped 0.15→0.10 (dominates on fidelity). Dose kept conservative at 0.3
(cleanest text/hands); 0.3–0.8 are all usable at τ0.10 — `s0.8` for max diversity if
you don't mind stylized text. `eyeball.py` configs now sweep τ0.10 × {s0.3, s0.8}.

> **Still empirical: the `block ≤ 18` cap and `gap > 0.03` cutoff.** The pool
> *membership* is now statistical, but the dot-avoidance cap is a hand-set
> structural guess (validated by eyeball, endorsed by the paper, but not *measured*
> on Anima). The analytic upgrade — a one-time **patch-grid-imprint probe** that
> measures each block's dot susceptibility directly, plus a **multi-prompt/artist**
> pool probe (A.1.1 uses 100 seeds × many prompts; we used 1 prompt × 8 seeds) — is
> the open Phase 3 below.

## Phase 3 (proposed) — analytic block selection, measured dot-avoidance

Two probes to replace the two remaining empirical knobs (`block_cap`, single-prompt
pool). Both are grounded in the full paper:

1. **Multi-prompt / per-artist pool probe.** A.1.1 derives **L** from 100 seeds ×
   many prompts; Table 4 shows the pool is dataset-agnostic (Jaccard 0.98 across
   independent 20-prompt pools). Our pool came from **one** prompt. Extend
   `probe_dc_convergence.py` to sweep N artists/prompts (Anima's natural
   conditioning axis), intersect the per-prompt ≥0.99 pools → an artist-robust,
   statistically-tested membership instead of a single-prompt read.
2. **Patch-grid-imprint probe (the analytic dot-predictor).** ⚠️ **BUILT + RUN +
   REFUTED** (`probe_patch_imprint.py`, `results/20260609-1613-imprint/`). The metric
   does **not** predict quality and must not gate the cap — two reasons, both in the
   data:
   - **The fraction normalization inverts the signal.** `imprint = patch-energy /
     *total* Δ-energy`. The decodes show block 17 (highest imprint, 0.008) is the
     *cleanest* (near-baseline) while block 9 (lowest, 0.004) recomposes hardest and
     **garbles hands + sign text**. A no-op block has tiny total Δ → its little patch
     noise reads as a high fraction; a hard-recomposing block dilutes patch energy to
     near-zero. So it rewards do-nothing blocks and punishes the productive ones.
   - **Dots are cumulative, not per-block.** Every single τ-gated block scores <0.01
     (all 0.004–0.008) — no block dots alone. The Phase-2c dots were the *full-schedule
     multi-block* late-stage attenuation; one τ-gated block can't reproduce that regime,
     so there was never a dot to measure. "Lines" (1D axis banding) is all that flickers.

   **Net:** the `≤18` cap stays justified by Phase-2c eyeball + the paper's
   "exclude final-stage" rule, **not** by this probe. The probe's real value is
   incidental — its `--decode` contact sheet is a clean per-block *diversity-vs-damage*
   catalog (block 17 = no-op, block 9 = recompose+break, block 2 = restyle), which is
   the right shape of tool but needs a **perceptual/VLM judge**, not a 2D-FFT (the paper
   scores blocks with Gemini for exactly this reason — Fig 9 / F.1). The live question
   has shifted from "which block dots" (solved) to **"how much diversity before
   hands/text break"** — being settled by the production `eyeball.py` s0.3-vs-s0.5 sweep.

   What the probe computes: for each block ℓ it re-generates with DAVE attenuating **ℓ alone** (a one-hot mask through the real
   `generate()` path — same τ-gate, same σ window, same dose), diffs the final
   latent against a baseline (`Δ_ℓ = lat_ℓ − lat_base`), and scores

       imprint(ℓ) = power(Δ_ℓ) at the patch-grid harmonics / total power(Δ_ℓ)

   The patch grid has period `spatial_patch_size` in the latent (=2 → the dots sit
   at the **latent Nyquist**), so the metric is the fraction of the perturbation's
   2D-FFT power on the patch-harmonic axes (DC dropped). High = the block moves the
   output as a patch-grid pattern (dot-causer); low = broadband/low-freq
   recomposition (safe). Verified on synthetics: smooth recomposition → 0.00, a pure
   patch checkerboard → 1.00, mostly-smooth+few-dots → 0.03. The run prints a
   per-block ranking + an agreement check against the shipped `≤18` cap.

   ```bash
   make daemon-run ARGS="bench/dave/probe_patch_imprint.py"              # all blocks, 3 seeds
   make daemon-run ARGS="bench/dave/probe_patch_imprint.py --decode 4"   # + VAE-decode worst/best for eyeball
   ```

**Why the cap can't be derived from theory alone.** Appendix E proves *why DC and
why early* (E.6: the ensemble mean is DC-dominated because AC cancels across the
data distribution; E.7: a capacity bound — early DC-lock caps recoverable diversity,
so intervening early is necessary). But the dots are a *downstream-architecture*
property (distance-to-output × LayerNorm renorm), which the latent-space theory
doesn't see — so dot-avoidance must be **measured** (probe 2), while pool membership
*can* be statistical (probe 1 / A.1.1). F.1 further warns the block→attribute map is
**model-specific** (Flux ≠ SD3.5), so any attribute-targeted-diversity feature must
be calibrated on Anima, not ported.

## Spin-off — DAVE as a training diversity signal (turbo / DP-DMD)

The intervention is the less interesting half; the **diagnostic** (DC = seed-shared
conditioning, AC = seed-specific diversity) ports to training as a *measurement*.
DP-DMD's whole pitch is *diversity-preserved* distillation, but its in-loop `div`
loss only measures how close the student's first step lands to the teacher's
K-step anchor — it says nothing about whether the student's own same-prompt
samples have collapsed across seeds (the canonical DMD failure mode). The DAVE
cross-seed **AC sim** is exactly that mode-collapse detector.

Wired as a `validate_every_n_steps`-style pass in the turbo loop (mirrors
`distill_mod`'s validation): fix one **held-out** conditioning, roll the student's
N-step Euler grid across N seeds under `no_grad` (the live `_forward` /
`set_student_step` / `student_sigmas` primitives), hook every block, and log:

- `val/div_ac_sim` — cross-seed cosine of the AC residual; **lower = more diverse**
  (rising over training = collapse). The headline signal.
- `val/div_dc_sim` — reference; should stay high (~conditioning lock).
- `val/div_gap` (`dc−ac`) and `val/div_xpred_ac_sim` (same split on the final latent).
- `val/fm_mse` — flow-matching reconstruction MSE on the held-out sample (the
  **fidelity** half of the fidelity↔diversity tradeoff view). ⚠️ FM val loss has
  *not* tracked sample quality on Anima (CMMD replaced it as the quality signal),
  so read it as a divergence/sanity number paired against AC-sim, not a score.

The metric is layout-robust (DC = mean over all non-(batch,channel) axes), so it
reads correctly under both eager 5D blocks and the compiled native-flatten.

```bash
make turbo  # then add to the turbo config or CLI:
#   --validate_every_n_steps 750 --val_diversity_seeds 8 [--val_prompt_idx N]
```

Code: `scripts/distill_turbo/diversity.py` (probe), `config.py` (`io.validate_every_n_steps`
/ `val_diversity_seeds` / `val_prompt_idx`), `distill.py` (held-out capture + in-loop call).
Status: wired + unit-smoke-tested (layout-invariance, cosine math, config round-trip);
**not yet exercised in a live distill run**.

## Phase 4 — MOD-guidance compose probe (2026-06-11): orthogonal, not antagonistic

Field observation (ComfyUI, orthoinit LoRA): mod-guidance alone sometimes lands in
degenerate layout attractors — *every* sample back-facing, or thick letterbox
borders — and composing DAVE (τ0.10·s0.5) rescued exactly those cases. Hypothesis
tested here: **does MOD deepen the early DC lock, and does DAVE restore it?**

Four arms of `probe_dc_convergence.py` (new `--lora/--mod/--dave` flags), same 8
seeds × 24 steps, CFG 4, `anima_channel_ortho` armed, `pooled_text_proj-0611`,
stock step_i8_skip27 schedule, w=3. Capture is **pre-attenuation** (probe hooks
register before `generate()` arms DAVE's), so the +DAVE numbers are the emergent
state change, not our own subtraction. Fixed pool = blocks 8–18:

| arm | DC s0–8 | AC s0–8 | AC s8–24 | DC power 8–26 (early) |
|---|---|---|---|---|
| vanilla | 0.998 | 0.581 | 0.538 | 0.327 |
| dave | 0.996 | **0.477** | **0.265** | 0.291 |
| mod | 0.998 | 0.586 | 0.550 | 0.328 |
| mod+dave | 0.996 | **0.477** | **0.265** | 0.292 |

AC-sim deltas vs vanilla: dave **−0.104**, mod **+0.005**, mod+dave **−0.104** →
interaction **−0.005 ≈ 0**. Results: `results/20260611-205*-{base-lora,dave,mod,mod-dave}/`.

1. **"MOD deepens the lock" is REFUTED.** MOD moves *nothing* the lock stats can
   see — DC sim, AC sim, even DC power ratio in its own 8–26 window are all
   vanilla-identical. The AdaLN delta rides `t_emb`, a **common-mode re-aim** of
   the already-locked DC: every seed's DC shifts together, which cross-seed
   similarity is blind to (and it's already saturated at 0.999).
   *Not a no-op artifact*: `probe_mod_presence.py` (same-seed MOD-on/off diff)
   shows MOD re-aims every block's DC by 0.6–4% at every step (cos ≥ 0.9994,
   |ΔDC| 45–67) — blocks 0–7 shift *more* than the steering window, because the
   base `proj(main)` t_emb injection reaches all blocks. Present, common-mode,
   lock-invariant.
2. **DAVE's unlock is emergent and persistent.** 2–3 attenuated early steps halve
   the *late*-trajectory AC sim (0.538 → 0.265) — divergence compounds through
   the latent long after the hooks go quiet.
3. **Zero interaction = orthogonality.** DAVE's effect under MOD is the same as
   without; MOD's (non-)effect likewise.

Follow-up proposal: `docs/anima/proposal/dave_mod_bestofn.md` — make "why the compose
gives better images" falsifiable via best-of-N order statistics with correlated
draws (MOD moves μ, DAVE moves ρ/N_eff; predicted vs observed best-of-n curves).

So the compose story is **not** compensation-by-opposition. It is: the DC lock
means all seeds share one layout basin; MOD re-aims that shared basin (tone win,
occasionally a degenerate basin — and because the lock is intact, **every seed
inherits the failure**, which is why MOD failures look like "everyone facing
away" and seed re-rolls don't fix them). DAVE weakens the lock so seeds spread
across basins again — it doesn't fight MOD, it **restores the seed lottery that
MOD's failures had made useless**. Practical rule: MOD steers, DAVE de-correlates;
compose them when a MOD failure survives re-rolling.

## Files

| File | Role |
|---|---|
| `probe_dc_convergence.py` | Phase-0 premise probe (read-only block hooks → `per_block.npz`). |
| `derive_alpha_mask.py` | Phase-1/2d offline mask: `per_block.npz` → flat pool `networks/calibration/dave_alpha.npz`. |
| `eyeball.py` | Phase-2 sweep: baseline vs DAVE configs × seeds → `output/tests/dave/`. |
| `probe_patch_imprint.py` | Phase-3 analytic dot-predictor: per-block patch-grid imprint → measured block cap. |
| `probe_mod_presence.py` | Phase-4 sanity: same-seed MOD-on/off ΔDC per block (MOD is present + common-mode, not a no-op). |
| `results/<ts>/` | Standard `result.json` envelope + PNGs + `per_block.npz` / `imprint.npz`. |

Intervention code lives outside `bench/`: `library/inference/corrections/dave.py`
(hooks), the `_dave_*` buffers in `library/anima/models.py`, `--dave*` flags in
`library/inference/args.py`, the `DAVE=1` lever in `scripts/tasks/inference.py`.
