# Anima LoRA: effective batch, lr, EMA and grad clip as one noise budget

Status: PROPOSED 2026-11-05. Preliminary runs (below) on ama mitsuki only; no
arm run under `--deterministic` yet. Instrumentation landed: the training loop
logs `grad/norm`, `grad/norm_sq`, `grad/norm_max`, `grad/clip_frac` and, with
gradient accumulation, `grad/micro_norm_sq` (`library/training/loop.py`).

## Question

Which effective batch, lr, EMA span and clip rule give the cleanest LoRA at a
given amount of learning, and how much of the difference between settings is
real rather than run-to-run chance? The goal is a Pareto front, not one recipe.

Constraint from the dataset owner: the images that cause gradient spikes stay
in the set. Their pull is to be bounded by the clip rule, not removed by
curation.

## Measurements and controls

- Quantitative: the grad statistics above, plus displacement, the median over
  modules of `||dW||` against a reference arm (`bench/anima_ema/render_ab.py`
  reports raw against EMA; the cross-arm ratio is the same computation).
- Judgment: renders checked by eye, 2–3 seeds × several artists, at the
  `make test` recipe (28 steps euler, cfg 4, shift 3, 1024²).
- Control: every arm runs `--deterministic --seed 42 --paired_step_rng`. Without
  it, two runs of one config differ by more than the arms do (next section).

## Preliminary runs (2026-11-05)

Common setup: ama mitsuki (106 images), 8 epochs, rank 32 / alpha 128,
`weight_svd` slice 2, REPA, `constant_with_warmup` (5 %), AdamW,
`max_grad_norm 1.0`, σ ~ logit-normal(0, 1) with shift 1, no timestep mask,
bf16 DiT with the fp32 stream, adapter EMA span 0.1.

| arm | updates | `‖dW‖` vs bs1 EMA | raw → EMA (norm, cos) | renders |
|---|---|---|---|---|
| bs1 5e-5 | 848 | 1.10 raw / 1.00 EMA | 0.91, 0.974 | speech bubbles and a stray "?" from the dataset; run 1 also oversaturated magenta neon and a twisted bikini pose |
| GA2 5e-5 | 424 | 0.68 EMA | 0.92, 0.982 | cleanest; style strength not yet judged |
| GA2 1e-4 | 424 | 1.41 EMA | — | no dataset leakage, but tangled arms and hands in the hug and maid prompts |

Reading: noise and displacement both matter. At similar displacement, bs1
leaks dataset features (bubbles) that GA2 does not; at higher displacement, GA2
breaks anatomy. EMA 0.1 reduces the bubbles at bs1 without removing them.

**Run-to-run floor.** The same config run twice gives a median dW cosine of
0.146 (bs1 5e-5) and 0.115 (GA2 1e-4), with displacement reproducing to 1.005
and 0.978. GA2 against bs1 reads 0.145, so direction cosines between arms carry
no signal at this setup (the floor in
[`paired_dw_chaos_floor_deterministic.md`](../findings/paired_dw_chaos_floor_deterministic.md)
was 0.41 on another config). In renders the bubbles came back in the bs1 twin;
the magenta saturation did not, so that one was the run, not the setting.

**Grad norm (bs1 5e-5).** Mean per epoch falls from 2.20 to about 0.95 (median
1.01 over the run). With the fixed cap of 1.0 the clip fraction falls from
94 % in epoch 1 to 22–29 % in epochs 5–8, 46 % overall. Spikes reach 12.8 and
correlate weakly with the loss (r = 0.27). GA2 1e-4 clips 43 % overall, so a
fixed cap does not separate the two arms.

**Noise scale (GA2 1e-4).** From one micro-batch's `‖g‖²` against the
accumulated mean's, over the run:

| aggregate | `E‖ḡ‖² / E‖g₁‖²` | `B = tr(Σ) / ‖G‖²` |
|---|---|---|
| mean | 0.52 | 25 |
| 10 % trimmed mean | 0.75 | 1.0 |
| median | 0.89 | 0.3 |

The typical update is mostly signal (B near 1). Most of `tr(Σ)` sits in a few
spike updates. The trimmed and median rows are a qualitative read, not unbiased
estimators.

## What this implies

- A larger batch has little to average on typical updates; GA2 is likely
  enough and GA4 is the upper end worth trying.
- The noise that matters is the spike tail. A fixed cap of 1.0 sits at the
  median norm, so it normalizes half the updates instead of catching spikes,
  and what it does changes over the run as the norm drifts down.
- Under accumulation, clipping after the mean is too late: a norm-10 micro-batch
  next to a norm-1 one sets the mean's direction, and the clip only shrinks it.
- Unclipped, a spike pulls Adam's first moment its way for ~10 updates and
  inflates the second moment for hundreds. A clip before the optimizer bounds
  both; that placement stays.

## Proposed changes

1. **Median clip.** `clip_mode = "fixed" | "median"`, `clip_k` (default 2): the
   cap is `clip_k ×` the median pre-clip norm of the last 50 updates. Until the
   window holds 10 updates, the cap is `max_grad_norm`. Typical updates pass
   unclipped; a spike is scaled to `clip_k ×` a typical update and keeps its
   direction, so the image still trains.
2. **Per-micro clip.** `clip_per_micro`: under accumulation, clip each
   micro-batch's own gradient before it joins the sum (one fp32 buffer of the
   adapter's gradients, ~100 MB at rank 32), so a spike carries at most `1/accum`
   of an update. Equal to the global clip at `accum = 1`.
3. Tests: the median clip leaves a norm stream with no outlier untouched and
   caps an injected spike at `clip_k ×` the median; per-micro clip at accum 1
   matches the global clip bit for bit.

## Plan

**Phase 1, clip rule** (5 arms × 2 artists, ~10 runs, about 1.5 h with renders):

| arm | batch | lr | clip |
|---|---|---|---|
| A | bs1 | 5e-5 | fixed 1.0 (today) |
| B | bs1 | 5e-5 | median, k 2 |
| C | bs1 | 5e-5 | off |
| D | GA2 | 5e-5 | median, k 2, after the mean |
| E | GA2 | 5e-5 | median, k 2, per micro |

Gate: B or E removes the dataset leakage on both artists without visible style
loss → it becomes the default. If C matches A, the clip rule is not what bs1
was missing.

**Phase 2, batch × lr** at the winning clip rule: bs1 / GA2 / GA4, lr along the
square-root rule for Adam (`lr ∝ √batch`, Malladi et al. 2022) and one step
off it each side, reported against displacement. Cut arms whose displacement
leaves 0.7–1.0 × the bs1 5e-5 EMA, where today's renders were usable.

**EMA** rides along at no cost: `ema_extra = "0.05,0.3"` beside the main 0.1 on
every arm, compared in `render_ab.py`. Because the span is a fraction of
optimizer updates, it already follows the EMA scaling rule across batch sizes
(Busbridge et al. 2023, decay `ρ → ρ^κ` for batch `κ×`).

Every phase is hour-scale and gets an explicit go before it runs.

## Out of scope

- Curating or down-weighting the spike images.
- σ-side variance reduction (loss weighting, stratified σ across micro-batches,
  training shift 3 to match inference): a separate line once the clip rule is
  settled. At shift 1, training draws σ > 0.9 for 1.4 % of updates while
  inference spends ~25 % of its steps there.
