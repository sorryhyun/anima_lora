# Anima LoRA gradients: a heavy tail at the σ edges, and the recipe that bounds it

> **STATUS (2026-11-05).** A/B RESULT + MEASUREMENT. Recipe B (5e-5
> `constant_with_warmup`, GA2, median clip k 2 per micro-batch, adapter EMA 0.1)
> beat the old default in a blind A/B and is the shipped
> `configs/methods/lora.toml`. A per-σ-band clip on top of B won a second,
> ||dW||-matched blind A/B at the margin (cluster p 0.047) and is in the shipped
> config since (`clip_sigma_bands = "0.2,0.4,0.6,0.8"`). Plan and per-phase detail:
> [`../proposal/lora_batch_clip_ema.md`](../proposal/lora_batch_clip_ema.md).

## What the gradient looks like

Setup unless noted: rank 32 / alpha 128, `weight_svd` slice 2, REPA, AdamW, bf16
DiT with the fp32 stream, σ ~ logit-normal(0, 1) at shift 1, batch 1 per
micro-batch, 8 epochs.

**The grad norm falls and is heavy-tailed.** At bs1 / 5e-5 (ama mitsuki, 848
updates) the mean pre-clip norm per epoch falls from 2.20 to about 0.95 (median
1.01 over the run). Spikes reach 12.8 and correlate weakly with the loss
(r = 0.27). A fixed cap of 1.0 therefore sits at the median: it clips 94 % of
updates in epoch 1 and 22–29 % in epochs 5–8, normalizing half the updates rather
than catching spikes.

**Typical updates are mostly signal; the noise is in the tail.** From one
micro-batch's `‖g‖²` against the GA2 mean's (GA2 1e-4), `B = tr(Σ) / ‖G‖²` is 25
by the mean, 1.0 by a 10 % trimmed mean and 0.3 by the median. A few spike
updates carry most of `tr(Σ)`. A larger batch therefore has little to average on a
typical update, and the lever is the clip, not the batch.

**The tail is σ, not images.** Logged per micro-batch under recipe B
(`--grad_sigma_log`; caststation, ama mitsuki and sweetonedollar, seed 42):

| σ | 0–0.2 | 0.2–0.4 | 0.4–0.6 | 0.6–0.8 | 0.8–1.0 |
|---|---:|---:|---:|---:|---:|
| share of draws (pooled) | 6.7 % | 23.5 % | 34.3 % | 28.0 % | 7.4 % |
| median pre-clip norm | 1.6–2.7 | 0.9–1.4 | 0.9–1.3 | 1.4–2.2 | 2.5–4.9 |
| clip rate under the global median clip (pooled) | 33 % | 2.6 % | 1.1 % | 9.8 % | 62 % |

The norm is U-shaped in σ (Spearman +0.23 to +0.41). The edges (σ < 0.2 or ≥ 0.8)
hold 13–15 % of the draws and 59–80 % of the clipped micro-batches. The clip rate
does not fall over the run. No image dominates: at most 3–4 clips per image over 8
epochs, spread across the set. A median taken over all σ reads normal edge-σ
draws as spikes. The first 10 micro-batches sit at the warmup cap
(`max_grad_norm` 1.0) and are all clipped; leave them out of any clip-rate count.

## What it means for the recipe

- **Clip against a running median, per micro-batch.** `--clip_mode median`
  (cap `clip_k ×` the median of the last 50 pre-clip norms) passes typical updates
  and scales a spike to `k ×` a typical one, keeping its direction, so the image
  still trains. `--clip_per_micro` clips each micro-batch before it joins the sum.
  Clipping after the mean is too late, because a norm-10 micro-batch next to a norm-1
  one has already set the direction. The clip stays before the optimizer: unclipped,
  a spike pulls Adam's first moment for ~10 updates and inflates the second for
  hundreds.
- **GA2 is enough.** Under the per-micro clip a spike carries at most `cap / accum`
  of an update, and typical updates are mostly signal. GA4 would halve the update
  count (60 on a 30-image set, where EMA 0.1 spans 6) for little extra averaging.
- **EMA needs a constant lr.** Under cosine decay the last ~10 % of updates barely
  move, and EMA 0.1 lands on the raw weights (ama mitsuki: ||dW|| 0.997×, cos
  1.000). At constant 5e-5 it is 0.91× / cos 0.974 and drops raw's stray dataset
  artifacts (speech bubbles, a stray "?") (`bench/anima_ema/render_ab.py`). EMA
  0.1 reduces the bubbles at bs1 but does not remove them. GA2 at a similar
  displacement does not show them.
- **lr sets displacement, and displacement has a ceiling.** GA2 at 1e-4 (1.41× the
  bs1 EMA's ||dW||) shows no leakage but tangles arms and hands in the hug and maid
  prompts. GA2 at 5e-5 (0.68×) is the clean point.

## The A/Bs

Both blind A/Bs use seeds 42–45 per arm, 4 prompts (classroom, hug, maid, bikini) × 2
render seeds, 32 pairs per artist, at the `make test` recipe. The decision test is a
sign flip over the 8 (artist, seed) clusters, because training noise acts per LoRA.
`bench/anima_abclip/run_ab.py`.

| | arms | result | caveat |
|---|---|---|---|
| s29 (Phase 1) | old default (2e-5 cosine, bs1, clip 1.0) vs B | **B 41–16**, 7 ties; cluster p 0.0078; caststation 22–8, ama mitsuki 19–8 | B moved 2.3–2.5× as far (median ||dW||), so part of the win can be strength |
| s30 (Phase 2) | B vs B + `--clip_sigma_bands 0.2,0.4,0.6,0.8`, band rescaled to B's median ||dW|| | **band 39–21**, 4 ties; cluster p 0.047; caststation 20–10, sweetonedollar 19–11 | marginal; cast s44 (8–0) carries much of it, and band lost cast s45 (3–5) |

The per-band clip changes little in magnitude (band moved within ±7 % of B before
matching), so s30 compares direction only. It wins most clearly on the bikini prompt
(6–1 and 7–1) and loses only sweetonedollar hug (2–4).

## Traps

- **Endpoint ΔW cosine between arms carries no signal at this setup.** The same
  config run twice gives a median dW cosine of 0.146 (bs1 5e-5) and 0.115 (GA2
  1e-4); GA2 against bs1 reads 0.145. Displacement reproduces to ±3 %. Use several
  training seeds and blind renders, not one run per arm (compare
  [`paired_dw_chaos_floor_deterministic.md`](paired_dw_chaos_floor_deterministic.md),
  0.41 on another config).
- **One render of one run is not a property of the setting.** The bs1 twin
  reproduced the speech bubbles but not the oversaturated magenta of the first run.
  Read an artifact as the setting's only if it recurs across training seeds.
- **Same seed is not a paired run.** Two arms with the same seed see the same σ
  draws, but they decorrelate as soon as one clip lands differently. Pairs differ
  as much as independent runs do.

## Open

- The σ-band clip ships on a marginal gate (p 0.047, two artists); a third artist
  would firm it up.
- sweetonedollar under B (s42–s44, no band clip) draws malformed and oversized hands at
  x1.0. They shrink at x0.8 / x0.6, so they scale with the adapter, as do the style and
  the uniform. Lower lr with more epochs would land near the same displacement; what
  separates hands from style (σ-gated LoRA, render aspect, the data) is untested.
- Training shift 3, to match inference: at shift 1, σ > 0.9 gets 0.6–1.4 % of draws,
  while inference spends ~25 % of its steps there. Untested.
- The GUI variant (`configs/gui-methods/lora.toml`: alpha 32, 4 epochs) kept 2e-5 and
  was not tested.
