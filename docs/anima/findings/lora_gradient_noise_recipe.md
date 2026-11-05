# Anima LoRA gradients: a heavy tail at the σ edges, and the recipe that bounds it

> **STATUS (2026-11-05).** A/B RESULT + MEASUREMENT. Recipe B (5e-5
> `constant_with_warmup`, GA2, median clip k 2 per micro-batch, adapter EMA 0.1)
> beat the old default in a blind A/B and is the shipped
> `configs/methods/lora.toml`. A per-σ-band clip on top of B won a second,
> ||dW||-matched blind A/B at the margin (cluster p 0.047) and is in the shipped
> config since (`clip_sigma_bands = "0.2,0.4,0.6,0.8"`). The T-LoRA rank mask is
> commented out there since a third A/B (s31) found the published T-LoRA no better
> than plain LoRA. s33 reran the s29 recipe A/B without the mask and at matched
> ||dW||: **A 52–42, cluster p 0.17**, so at equal step length B's direction is no
> better, and s29's win came from the mask mismatch or from B's 2.2–2.5× larger
> ||dW||. Open questions (B at its own ||dW|| without the mask, training σ shift, the mask):
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

s29 and s30 trained with the T-LoRA mask on (`min_rank` 1) and rendered at full
rank, so every arm trained one model and rendered another. That is likely the largest
confound in both. s33 reran s29 without the mask
([proposal §0](../proposal/lora_batch_clip_ema.md)) and B lost its lead. s29 and s30 use seeds 42–45 per arm, 4 prompts (classroom, hug, maid, bikini) × 2
render seeds, 32 pairs per artist, at the `make test` recipe. The decision test is a
sign flip over the 8 (artist, seed) clusters, because training noise acts per LoRA.
`bench/anima_abclip/run_ab.py`.

| | arms | result | caveat |
|---|---|---|---|
| s29 (Phase 1) | old default (2e-5 cosine, bs1, clip 1.0) vs B | **B 41–16**, 7 ties; cluster p 0.0078; caststation 22–8, ama mitsuki 19–8 | B moved 2.3–2.5× as far (median ||dW||), so part of the win can be strength |
| s33 (s29 rerun) | A vs B (one clip window), both `use_timestep_mask` off; B rescaled to A's median ||dW|| (×0.40–0.46); ama mitsuki, sweetonedollar, caststation; render seeds 9 / 10 | **A 52–42**, 2 ties; cluster p 0.17 (A side), 0.88 (B side); ama mitsuki 17–13, sweetonedollar 19–13, caststation 16–16 (A–B) | the seeds split 6 A, 4 B, 2 even, and single seeds swing 6–2 either way, so the training seed moves the result more than the recipe does; B was not tested at its own ||dW|| |
| s30 (Phase 2) | B vs B + `--clip_sigma_bands 0.2,0.4,0.6,0.8`, band rescaled to B's median ||dW|| | **band 39–21**, 4 ties; cluster p 0.047; caststation 20–10, sweetonedollar 19–11 | marginal; cast s44 (8–0) carries much of it, and band lost cast s45 (3–5) |
| s31 | band (T-LoRA mask in training) rendered with the mask (`--lora_timestep_mask`) vs the same recipe without the mask, full rank; nomask rescaled to band's ||dW|| (x0.95–0.97) | **tie, band 11–13**; sweetonedollar s42 4–4, s43 1–7, s44 6–2 | three training seeds; a pair's two arms share composition and differ in detail |

The per-band clip changes little in magnitude (band moved within ±7 % of B before
matching), so s30 compares direction only. It wins most clearly on the bikini prompt
(6–1 and 7–1) and loses only sweetonedollar hug (2–4).

## Where the LoRA acts at inference (σ gate)

`inference.py --lora_sigma_range LO HI` turns the LoRA off outside `[LO, HI)` (P-GRAFT
hooks). sweetonedollar under B, s42–s44, the four A/B prompts at render seed 3, `make
test` recipe (28 steps, shift 3). Mean absolute pixel difference, 0–255
(`bench/anima_abclip/sigma_gate.py`, `results/20261105-2110-sigma_gate_sweet/`):

| | vs LoRA on everywhere (hooks) | vs bare base |
|---|---:|---:|
| static merge x1.0 (noise floor of the hook path) | 1.6–2.0 | — |
| off at σ < 0.4 (5 of 28 steps) | 2.4–2.6 | — |
| x0.8 / x0.6 | 10.5–16.5 / 18.5–26.5 | — |
| off at σ >= 0.8 (12 of 28 steps) | 48–58 | **8.5–8.9** |
| LoRA on everywhere | — | 50–59 |
| T-LoRA mask per step (`--lora_timestep_mask`) | 21–27 | 45–57 |

Off at σ >= 0.8, the render is the base's: same composition, palette and lighting, with
small changes (eye colour, a ribbon). Off at σ < 0.4 changes nothing past the noise
floor. Between training seeds the renders differ by 40.9 with the LoRA on and 7.4 with
it off at σ >= 0.8, so the seed-to-seed variation is all in that band. At shift 1 the
band gets 7.4 % of draws, about 18 micro-batches over sweetonedollar's 8 epochs. The
T-LoRA mask is training-only: there these draws train 1–8 of the 32 ranks (`min_rank` 1,
linear), while inference runs all 32 at every σ, so the high-σ effect includes ranks
trained only on lower σ. The x0.6 / x0.8 sweep moves the same composition. One artist, whose tag the base already renders in a painterly look;
caststation is unchecked.

Masking at inference as in training (σ >= 0.8 runs 1–8 of 32 ranks) does not fall back
to the base: the render stays as far from it, with the composition kept and the details
redrawn. Training seeds differ by 39.3 under the mask against 40.9 without it.

## Implementation

- `library/training/grad_clip.py`. `--clip_mode median`: the cap is `clip_k ×` the
  median of the last `clip_window` pre-clip norms, excluding the one being clipped,
  and `max_grad_norm` until 10 are recorded. Ring buffer, median and coefficient stay
  on the card; the window is not checkpointed, so a resumed run warms up again.
- `--clip_per_micro` clips each micro-batch before it joins an fp32 sum of the
  adapter grads (~100 MB at rank 32), so the median runs over micro-batch norms.
  Under several processes or fp16 loss scaling it clips the accumulated update
  instead, with a warning. `--clip_mode fixed` without it keeps
  `accelerator.clip_grad_norm_`.
- `--clip_sigma_bands` keeps one ring buffer per band and uses the global cap until
  the band holds 5 norms; a unit of several micro-batches takes their mean σ.
- Stamped: `ss_clip_mode`, `ss_clip_k`, `ss_clip_per_micro`, `ss_clip_sigma_bands`.
  `--grad_sigma_log` writes `<output_name>.grad_sigma.jsonl` (per clipped unit: norm,
  cap, σ, loss, image key; `bench/anima_abclip/sigma_tail.py` reads it).
- Logged: `grad/norm`, `grad/norm_sq`, `grad/norm_max`, `grad/clip_frac`,
  `grad/clip_cap`, and `grad/micro_norm_sq` under accumulation.
- `tests/test_grad_clip.py`: typical norms pass the median clip, a spike lands at
  `clip_k ×` the median with its direction kept, fixed matches
  `torch.nn.utils.clip_grad_norm_` bit for bit, per-micro at accum 1 matches the
  global clip, and the per-micro sum of a real backward matches plain accumulation.

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
- Training σ shift and the T-LoRA mask: the proposal.
- The GUI's LoRA variant (`configs/gui-methods/lora.toml`) carries the same recipe
  since 2026-11-05 (alpha 128, 8 epochs, band clip, EMA 0.1, no mask); T-LoRA and
  Hydra GUI variants keep their own lr and fixed clip.
