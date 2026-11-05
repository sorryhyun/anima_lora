# Anima LoRA: effective batch, lr, EMA and grad clip as one noise budget

Status: RUNNING 2026-11-05. Median and per-micro clip are implemented
(`library/training/grad_clip.py`); a blind A/B of the default recipe against
the proposed one is training (Phase 1). Preliminary runs (below) are ama mitsuki
only. The training loop logs `grad/norm`, `grad/norm_sq`, `grad/norm_max`,
`grad/clip_frac`, `grad/clip_cap` (median / per-micro clip) and, with gradient
accumulation, `grad/micro_norm_sq` (`library/training/loop.py`).

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
- Judgment: blind A/B pairs at the `make test` recipe (28 steps euler, cfg 4,
  shift 3, 1024²), graded in the `anima-blind-pairs` repo.
- Control: several training seeds per arm, not one `--deterministic` draw. Two
  runs of one config differ by more than the arms do (next section), and a
  deterministic pair still cannot rule out that its one seed favours one arm.

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

## Changes (implemented 2026-11-05)

1. **Median clip.** `--clip_mode median`, `--clip_k` (default 2),
   `--clip_window` (default 50): the cap is `clip_k ×` the median pre-clip norm
   of the last `clip_window` norms, excluding the one being clipped. Until 10
   are recorded, the cap is `max_grad_norm`. Typical updates pass unclipped; a
   spike is scaled to `clip_k ×` a typical update and keeps its direction, so
   the image still trains. Ring buffer, median and coefficient stay on the card.
   The window is not checkpointed: a resumed run warms up again.
2. **Per-micro clip.** `--clip_per_micro`: under accumulation, clip each
   micro-batch's own gradient before it joins an fp32 sum of the adapter's
   gradients (~100 MB at rank 32), so a spike carries at most `1/accum` of an
   update. The median then runs over micro-batch norms (in full-update units),
   and `grad/norm` logs them. One process only, not under fp16 loss scaling.
3. `--clip_mode fixed` without `--clip_per_micro` keeps the old
   `accelerator.clip_grad_norm_` call. Stamped as `ss_clip_mode`, `ss_clip_k`,
   `ss_clip_per_micro`.
4. `tests/test_grad_clip.py`: typical norms pass the median clip untouched, an
   injected spike lands at `clip_k ×` the median with its direction kept, the
   fixed clip matches `torch.nn.utils.clip_grad_norm_` bit for bit, per-micro at
   accum 1 matches the global clip, per-micro bounds a spike to its share, and
   the per-micro sum of a real backward matches plain accumulation.

## Plan

### Phase 1: blind A/B, default against the proposed recipe (running)

The question is whether to change the default, so the arms are the shipped
recipe and the whole proposed one, not a factorial over its parts. A win does
not say which change carried it.

| arm | config |
|---|---|
| A | `configs/methods/lora.toml`: 2e-5 cosine, bs1, clip fixed 1.0, EMA 0.1 |
| B | A with 5e-5 `constant_with_warmup`, GA2, `clip_mode median`, `clip_k 2`, `clip_per_micro` |

- **Training:** seeds 42–45 per arm × two artists (channel (caststation), 30
  images; ama mitsuki, 106), 8 epochs, 16 LoRAs
  (`output/ckpt/abclip_<cast|ama>_<A|B>_s<seed>.safetensors`).
- **Renders:** each LoRA renders 4 prompts (classroom, hug, maid, bikini on a
  beach: the ones where the preliminary runs showed leakage or broken anatomy)
  × 2 render seeds, at the `make test` recipe.
- **Pairs:** a pair holds the A and B LoRAs of one (artist, training seed) at one
  prompt and render seed, sides shuffled. One set per artist,
  `s29_ABCLIP_cast` and `s29_ABCLIP_ama`, 32 pairs each, in `anima-blind-pairs`.
  The key sits in `output/blind/<set>/key.json`, outside that repo. The grader
  picks the better side overall: style fidelity, anatomy, stray dataset
  artifacts.
- **Statistics:** training noise acts per LoRA, so renders of one LoRA are not
  independent. The decision test is a sign flip over the 8 (artist, seed)
  clusters, with B − A wins per cluster as the statistic (one-sided p floor
  1/256). The binomial over the 64 pairs is reported beside it (B ≥ 43/64 is
  one-sided p 0.004), plus wins per artist and per prompt.
- **Gate:** B becomes the default if the cluster test gives p ≤ 0.05 and B wins
  on both artists. If A wins, or the result is flat, the default stays.
- `bench/anima_abclip/run_ab.py render` (daemon job: renders and median
  `||dW||` per LoRA into `bench/anima_abclip/results/`), `make --push` (builds
  and pushes the sets), `score` (pulls verdicts, writes
  `bench/anima_abclip/report.md`).

First B run (caststation s42, 120 updates, 240 micro-batches): 13 % of
micro-batches clipped at k 2, median cap 3.06, the largest norm 10.8. The tail is
heavier than "a few spikes". Phase 2 asks whether that tail is σ.

### Owed: Phase 2, σ-side

At shift 1, training draws σ > 0.9 for 1.4 % of updates while inference spends
~25 % of its steps there, and the grad norm depends on σ. A median taken across
all σ may read normal σ-to-σ variation as spikes. Next, one of:

- **Per-timestep clip:** the median kept per σ band (a few bands over [0, 1]),
  so each draw is capped against draws at a similar σ.
- **Training timestep shift:** shift 3 to match inference, which moves draws
  into the band inference uses and changes the norm distribution the clip sees.

What decides between them is first logging σ beside each micro-batch norm (no
GPU run beyond a normal training run): if the clipped tail concentrates in a σ
band, the per-timestep clip; if the tail spreads across σ, the shift.

**GA4 waits.** Under per-micro clip, a spike already carries at most
`cap / accum` of an update, so GA4's extra dilution buys little. On typical
updates B ≈ 1 (most of the update is signal), so averaging 4 instead of 2
reduces little noise, and it halves the update count: caststation would get 60
updates, where EMA 0.1 spans 6. GA4 is worth a run only if B still leaks
dataset artifacts after the σ-side change.

**EMA** rides along on any later arm: `ema_extra = "0.05,0.3"` beside the main
0.1, compared in `render_ab.py`. The span is a fraction of optimizer updates, so
it already follows the EMA scaling rule across batch sizes (Busbridge et al.
2023, decay `ρ → ρ^κ` for batch `κ×`).

Every phase is hour-scale and gets an explicit go before it runs.

## Out of scope

- Curating or down-weighting the spike images.
- σ-side loss weighting and σ stratified across micro-batches (Phase 2 takes
  only the per-σ clip and the training shift).
