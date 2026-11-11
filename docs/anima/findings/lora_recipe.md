# Anima LoRA recipe: what the blind A/Bs settled (s29–s42)

> **STATUS (2026-11-06).** A/B RESULTS + MEASUREMENTS, line closed. Fourteen blind A/Bs on
> the plain-LoRA recipe of `configs/methods/lora.toml`. Each shipped knob below either won
> one or tied the alternative. The gradient measurements explain the clip. REPA is off
> since s37 / s38. Rank, alpha against lr, and σ shift were rerun without REPA and stayed
> null.

## The shipped recipe and what backs each knob

| knob | value | evidence | without REPA? |
|---|---|---|---|
| REPA | off | off beat on at alpha 512 / lr 1.25e-5, 32–16 (s37 + s38, pair p 0.015); tied at alpha 128 / lr 5e-5, 23–22 (s40) | — |
| σ sampling | logit-normal(0, 1), shift 1 (`timestep_sampling = "sigmoid"`) | shift 2 tied, 22–23 (s32); without REPA shift 1 led 28–20 (s39) | yes |
| clip | median of the last 50 micro-batch norms × `clip_k` 2, per micro-batch, one window per σ band (0.2 / 0.4 / 0.6 / 0.8) | σ-edge heavy tail (§ Gradients); band beat one window 39–21 (s30, p 0.047) | profile yes, A/B no |
| batch | bs 1, GA 2 | a spike carries at most `cap / accum` of an update (§ Gradients) | the argument does not involve REPA |
| lr / schedule | 5e-5, `constant_with_warmup` (5 % warmup) | beat 2e-5 cosine bs1 clip 1.0, 41–16 (s29), a win that did not survive the mask-off, ‖dW‖-matched rerun (s33, A 52–42); 1e-4 tangles hands | no |
| adapter EMA | 0.1 (`--ema_window` default) | needs a constant lr; at 5e-5 it drops raw's stray bubbles | the argument does not involve REPA |
| rank / alpha | 32 / 128 | rank 16 null (s34 28–20, s41 17–15); alpha 512 / lr 1.25e-5 null (s35 25–22, s36 18–13, s42 23–22) | yes (s41, s42) |
| T-LoRA mask | off | the published method (mask in training and inference) tied plain LoRA 11–13 (s31) | no |
| other | AdamW, 8 epochs, `weight_svd` down init, caption dropout 0.1, fp32 stream | set before this line | — |

What a tie means here: the training seed moves a result more than most of these knobs do.
Single (artist, seed) clusters swing 6–2 either way in every A/B. A tie keeps the simpler or
cheaper side, and the shipped side when neither is.

## Protocol

- Several training seeds per arm (42–44, sometimes 45), two or four render seeds, four
  prompts (p0 classroom / school uniform, p1 hetero hug, p2 maid / tray / cafe, p3 bikini /
  beach), blind pairs at the `make test` recipe (28 steps euler, cfg 4, shift 3, 1024²),
  graded in `anima-blind-pairs`. One grader.
- **Match ‖dW‖** (`match_dw`, from s30): the treatment's `lora_up` is rescaled so its
  median `‖(alpha / rank) · up @ down‖` over modules equals the control's at the same
  (artist, seed). The pairs then compare direction, not step length. s29 without it was
  confounded by B moving 2.3–2.5× as far.
- **Decision test:** a sign flip over (artist, seed) clusters, since training noise acts
  per LoRA; the binomial over pairs is reported beside it. Two training seeds put the
  cluster floor at 0.25, three at 0.125.
- Sets: caststation (30 images, ~120 updates under GA2), sweetonedollar (31, ~124), ama
  mitsuki (106, 424), sincos (354, ~1416).
- Runner: `_archive/bench/anima_abclip/run_ab.py --exp <name> {train,render,make,score,answer,mad}`;
  per-experiment tallies in `_archive/bench/anima_abclip/report_<exp>.md`.

## All A/Bs

| id | exp | control vs treatment | result (control–treatment) | cluster p | note |
|---|---|---|---|---:|---|
| s29 | `abclip` | old default (2e-5 cosine, bs1, fixed clip 1.0) vs B (5e-5 constant, GA2, median clip k2 per micro) | **16–41**, 7 ties | 0.0078 | T-LoRA mask on, B 2.3–2.5× ‖dW‖, unmatched |
| s30 | `sigband` | B vs B + σ-band clip | **21–39**, 4 ties | 0.047 | mask on; cast s44 8–0 carries much of it |
| s31 | `nomask` | T-LoRA as published (mask in training and inference) vs no mask | 11–13 | — | sweetonedollar only |
| s32 | `sigshift` | shift 1 vs shift 2 (σ ≥ 0.8 draws 7.4 % → 24 %) | 22–23, 3 ties | 0.50 | cast 8–14 (every seed), sweet 14–9 |
| s33 | `abclip2` | s29 rerun, no mask, B matched to A (×0.40–0.46) | **A 52–42**, 2 ties | 0.17 (A side) | three artists; B's direction no better |
| s34 | `rank` | r32 / a128 vs r16 / a64 | 20–28 | 0.25 | two clusters 7–1 each carry it |
| s35 | `alpha` | a128 / 5e-5 vs a512 / 1.25e-5 | 25–22, 1 tie | 0.62 | cast 16–8, sweet 9–14 |
| s36 | `alpha_sincos` | s35 on sincos | 18–13, 1 tie | 1.0 | ~1416 updates |
| s37 | `norepa` | a512 REPA on vs off (caststation) | **8–16** | 0.375 | s43 8–0 |
| s38 | `norepa_sweet` | s37 on sweetonedollar | **8–16** | 0.125 | every seed to off |
| s39 | `sigshift_norepa` | s32, REPA off in both | 28–20 | 0.16 (shift 1 side) | cast 12–12, sweet 16–8 (every seed) |
| s40 | `repa128` | a128 REPA on vs off | 23–22, 3 blank | 0.69 | cast 12–10, sweet 11–12 |
| s41 | `rank_ama` | r32 / a512 vs r16 / a362 (alpha ∝ √r), REPA off, ama mitsuki | 17–15 | — (2 seeds) | bikini 7–1 r32 |
| s42 | `alpha_norepa` | s35, REPA off in both | 23–22, 3 blank | 0.64 | seed 42 6–2 to a512 on both artists |
| s43 | `nodc` | `weight_svd` vs `weight_svd_nodc` + `channel_scaling_floor` 0.01, matched | 25–23 | 0.73 | by prompt, both artists: classroom / hug 19–5 svd, maid / beach 6–18 nodc |

Every arm through s36 trained with relational REPA on. s29 and s30 trained with the T-LoRA
mask on and rendered at full rank, so each arm was trained as one model and rendered as
another.

## Gradients: a heavy tail at the σ edges

Setup unless noted: rank 32 / alpha 128, `weight_svd`, AdamW, bf16 DiT with the fp32
stream, σ ~ logit-normal(0, 1) at shift 1, batch 1 per micro-batch, 8 epochs.

**The grad norm falls and is heavy-tailed.** At bs1 / 5e-5 (ama mitsuki, 848 updates) the
mean pre-clip norm per epoch falls from 2.20 to about 0.95 (median 1.01 over the run).
Spikes reach 12.8 and correlate weakly with the loss (r = 0.27). A fixed cap of 1.0 sits at
the median: it clips 94 % of updates in epoch 1 and 22–29 % in epochs 5–8, normalizing half
the updates rather than catching spikes.

**Typical updates are mostly signal; the noise is in the tail.** From one micro-batch's
`‖g‖²` against the GA2 mean's, `B = tr(Σ) / ‖G‖²` is 25 by the mean, 1.0 by a 10 % trimmed
mean and 0.3 by the median. A few spikes carry most of `tr(Σ)`, so a larger batch has
little to average on a typical update: the lever is the clip, not the batch.

**The tail is σ, not images** (`--grad_sigma_log`, recipe B; caststation, ama mitsuki,
sweetonedollar, seed 42):

| σ | 0–0.2 | 0.2–0.4 | 0.4–0.6 | 0.6–0.8 | 0.8–1.0 |
|---|---:|---:|---:|---:|---:|
| share of draws (pooled) | 6.7 % | 23.5 % | 34.3 % | 28.0 % | 7.4 % |
| median pre-clip norm | 1.6–2.7 | 0.9–1.4 | 0.9–1.3 | 1.4–2.2 | 2.5–4.9 |
| clip rate under the global median clip | 33 % | 2.6 % | 1.1 % | 9.8 % | 62 % |

The norm is U-shaped in σ (Spearman +0.15 to +0.45). The edges (σ < 0.2 or ≥ 0.8) hold
13–16 % of the draws and 50–80 % of the clipped micro-batches; the clip rate does not fall
over the run. No image dominates (at most 2–4 clips per image over 8 epochs). A median
taken over all σ reads normal edge-σ draws as spikes, hence one window per band.

The profile holds across every variant measured: with and without the T-LoRA mask
(`sigmalog2_*`: σ ≥ 0.8 at 3.0× / 2.9× the 0.4–0.6 median), at rank 16 (s34, 2.6–3.4×), and
without REPA (s39 `norepa128_*`, six runs: σ 0.8–0.9 at 2.2–4.8 against 0.9–1.3 at
0.4–0.6). Under the fixed, global-median and band clips, σ ≥ 0.8 takes 7 %, 11–12 % and
13–15 % of the summed post-clip norm, so the A → B change shifted the σ weighting more
than B → band did. The first 10 micro-batches sit at the warmup cap (`max_grad_norm` 1.0);
leave them out of a clip-rate count.

What this gives the recipe:

- **Clip against a running median, per micro-batch.** A spike is scaled to `k ×` a typical
  norm with its direction kept, so the image still trains. Clipping after the mean is too
  late: a norm-10 micro-batch next to a norm-1 one has already set the direction. Unclipped,
  a spike pulls Adam's first moment for ~10 updates and inflates the second for hundreds.
- **GA2 is enough.** Under the per-micro clip a spike carries at most `cap / accum` of an
  update. GA4 would halve the update count (60 on a 30-image set, where EMA 0.1 spans 6).
- **EMA needs a constant lr.** Under cosine decay the last ~10 % of updates barely move and
  EMA 0.1 lands on the raw weights (ama mitsuki: ‖dW‖ 0.997×, cos 1.000). At constant 5e-5
  it is 0.91× / cos 0.974 and drops raw's stray artifacts (speech bubbles, a stray "?")
  (`bench/anima_ema/render_ab.py`).
- **lr sets displacement, and displacement has a ceiling.** GA2 at 1e-4 (1.41× the bs1 EMA's
  ‖dW‖) tangles arms and hands in the hug and maid prompts; GA2 at 5e-5 (0.68×) is clean.
  This is a read of renders, not a blind A/B.

## σ: where the LoRA acts, and training shift

`inference.py --lora_sigma_range LO HI` turns the LoRA off outside `[LO, HI)` (P-GRAFT
hooks). sweetonedollar under B, s42–s44, the four prompts at render seed 3. Mean absolute
pixel difference, 0–255 (`_archive/bench/anima_abclip/sigma_gate.py`):

| | vs LoRA on everywhere (hooks) | vs bare base |
|---|---:|---:|
| static merge ×1.0 (noise floor of the hook path) | 1.6–2.0 | — |
| off at σ < 0.4 (5 of 28 steps) | 2.4–2.6 | — |
| ×0.8 / ×0.6 | 10.5–16.5 / 18.5–26.5 | — |
| off at σ ≥ 0.8 (12 of 28 steps) | 48–58 | **8.5–8.9** |
| LoRA on everywhere | — | 50–59 |
| T-LoRA mask per step (`--lora_timestep_mask`) | 21–27 | 45–57 |

Off at σ ≥ 0.8 the render is the base's. Off at σ < 0.4 nothing changes past the noise
floor. The seed-to-seed variation sits in that band too: between training seeds the renders
differ by 40.9 with the LoRA on and 7.4 with it off at σ ≥ 0.8. At shift 1 the band gets
7.4 % of the draws, about 18 micro-batches per run, against 12 of 28 inference steps.

So training shift was the obvious lever: more high-σ draws fit that band from more samples
but also learn more composition from 30 images. s32 (shift 2) tied with the artists split
(caststation to shift 2 on every seed, sweetonedollar to shift 1); without REPA (s39)
caststation went even and sweetonedollar stayed with shift 1, so caststation's gain needed
REPA. Shift 1 stays.

## REPA

Relational REPA (weight 0.05, layer 8, PE-Spatial) had shipped on sample-grid reads and
CMMD, A/B'd only against the absolute form (`_archive/repa/repa.md`).

- **Its cost depends on the step.** At alpha 512 / lr 1.25e-5 it lost 16–32 (s37 + s38;
  5 of 6 clusters, p2 maid 1–11). At alpha 128 / lr 5e-5 it tied 23–22 (s40; p2 5–7).
  Combined with s42: a128 with REPA ≈ a128 without ≈ a512 without, and only a512 with REPA
  loses. The LoRAs trained at the shipped step with REPA on lost nothing measurable.
- **Loss cannot screen it.** `loss/current` holds the fm loss plus `0.05 · repa/align_loss`
  (`library/training/losses.py::_repa_loss`); with the REPA term taken out, the fm loss of
  the two arms matches (caststation s42: 0.0731 vs 0.0723). Grad norms, the σ profile and
  ‖dW‖ (match factors 0.94–1.06) barely move either. REPA changes the render without
  changing the fm fit.
- Off in `configs/methods/{lora,lora_const_ema}.toml`, `configs/gui-methods/{lora,tlora,hydralora}.toml`,
  `configs/soup/soup.toml` (and `configs/base.toml` had it off). EasyControl task configs
  keep it: cond ≠ target is a different regime, not tested here.

## Rank, alpha and the bilinear drift

arXiv 2606.12883 splits one LoRA step under Adam or Muon into a task-aligned signal
`s·(BΔA + ΔB·A)`, Θ(αη), and a bilinear drift `s·ΔB·ΔA`, Θ(αη²). At a fixed α·η the update
is the same to first order and the drift falls as 1/η; the paper puts the optimal alpha at
`α ≈ C·√r` with a large C. For Anima it predicts that high alpha / low lr draws cleaner
LoRAs, and that alpha for another rank scales as √r (the rsLoRA rule `adaln_alpha` already
follows).

For reference, Cosmos-Predict2-2B's full fine-tune runs fused AdamW at 1.7e-5 peak
(`nvidia-cosmos/cosmos-predict2`, `post-training_text2image.md`), its LoRA r16 / α16 at
2.6e-5. The shipped 5e-5 at scale 4 is about 3× that. The paper's own rule at rank 16 would
take a step ~5× the shipped one, past the displacement ceiling above, so the tests held α·η
at the shipped value and moved only the split.

- **Same α·η gives the same ‖dW‖**: a512 lands at 0.96–0.99× of a128 on 30 images, 0.97× on
  sincos (~1416 updates), 1.00–1.08 factors without REPA.
- **At fixed alpha / rank, ‖dW‖ scales as √r under Adam**: r16 at alpha 64 moved 0.72× of
  r32 (s34). With alpha ∝ √r (r16 at alpha 362 against r32 at 512) the steps match in
  training, 0.95× / 1.09× (s41).
- **The drift is real and does not reach the render.** `_archive/bench/anima_abclip/drift_share.py`
  rebuilds `A0` from the base DiT and splits `ΔW = s·B·A0 + s·B·D`, `D = A − A0`. Median over
  modules:

  | | `‖B·D‖ / ‖B·A0‖` | mlp.layer2 | adaln self-attn | `‖D‖ / ‖A0‖` |
  |---|---:|---:|---:|---:|
  | a128, 6 runs | **0.15–0.17** | 0.26–0.30 | 0.32–0.40 | 0.08–0.09 |
  | a512, 6 runs | **0.06** | 0.11–0.14 | 0.12–0.15 | 0.03 |

  A sixth of ΔW at the shipped lr, a third in mlp.layer2 and adaln; alpha 512 cuts it 2.7×,
  and five blind sets (s35, s36, s42; rank: s34, s41) cannot tell the arms apart. The
  `down_init` A/Bs found the same for the starting basis (`bench/grad_init/README.md`).
- Rank is null on three artists with and without REPA; rank 16 only halves the file.
- If a longer run ever shows an effect: Hydra's router lr is `unet_lr × router_lr_scale`
  (lower lr 4× → scale 4×); the turbo student starts from an extracted delta, so alpha
  changes its initial ΔW; MOSS LoRA (Muon, 5e-5) is already at its pretraining lr.

## T-LoRA mask

`use_timestep_mask` (on in `lora.toml` until 2026-11-05, `min_rank` 1) masked ranks in
training only: the static merge cannot vary with σ and the eval forward skipped the mask,
so inference ran every rank at every σ. The paper (§3.1, arXiv 2507.05964) masks at
inference too; `inference.py --lora_timestep_mask` does that on the P-GRAFT hooks.

| | training | inference | status |
|---|---|---|---|
| T-LoRA as published | mask | mask (hooks; a custom node in ComfyUI) | s31: tied plain LoRA, 11–13 |
| plain LoRA | none | full rank | default since 2026-11-05 |
| shipped until 2026-11-05 | mask | full rank | every s29 / s30 LoRA; untested against either |

The published method costs a custom node at inference and gained nothing on one artist,
so the default dropped the mask. Masked at inference, a T-LoRA does not fall back to the
base at high σ: the render stays as far from it, composition kept, details redrawn.

## Traps

- **Endpoint ΔW cosine between arms carries no signal.** The same config run twice gives a
  median dW cosine of 0.146 (bs1 5e-5) and 0.115 (GA2 1e-4); GA2 against bs1 reads 0.145.
  Displacement reproduces to ±3 %. Compare
  [`paired_dw_chaos_floor_deterministic.md`](paired_dw_chaos_floor_deterministic.md).
- **One render of one run is not a property of the setting.** The bs1 twin reproduced the
  speech bubbles but not the first run's oversaturated magenta; read an artifact as the
  setting's only if it recurs across training seeds.
- **Same seed is not a paired run.** Two arms with one seed see the same σ draws but
  decorrelate as soon as one clip lands differently.
- **Cross-seed MAD as a pre-screen.** In s32, s34 and s35 the arm whose training seeds
  agreed more won its artist's blind set in 5 of 6 cases. s39–s42 do not bear it out: the
  lower-MAD arm won 2 artist sets (s39 sweet, s41), lost 3 (s40 both, s42 sweet, each by
  one or two pairs) and tied 2. Most gaps are ±3, about one arm's render-seed spread; do not
  use MAD to skip a blind set.

## Open

- **B at its own ‖dW‖ against A**, both without the mask. s33 matched the step and found
  B's direction no better; if B wins unmatched, the lever is the step length (lr / schedule),
  not the clip, and the clip / GA part has no blind win of its own. s30 (band clip) rests on
  the masked B.
- **REPA on a long run** (sincos, ~1416 updates); every check here is 30–106 images.
- T-LoRA, Hydra and soup inherited REPA off untested.
- The σ-band clip shipped on p 0.047 over two artists.
- sweetonedollar under B draws malformed, oversized hands at ×1.0 that shrink at ×0.8 /
  ×0.6, scaling with the adapter like the style does. What separates hands from style
  (σ-gated LoRA, render aspect, the data) is untested.
- If the T-LoRA mask returns, stamp `use_timestep_mask` / `min_rank` / `alpha_rank_scale`
  in the LoRA so inference can apply it without flags.

## Implementation

- `library/training/grad_clip.py`. `--clip_mode median`: the cap is `clip_k ×` the median of
  the last `clip_window` pre-clip norms, excluding the one being clipped, and
  `max_grad_norm` until 10 are recorded. Ring buffer, median and coefficient stay on the
  card; the window is not checkpointed, so a resumed run warms up again.
- `--clip_per_micro` clips each micro-batch before it joins an fp32 sum of the adapter grads
  (~100 MB at rank 32). Under several processes or fp16 loss scaling it clips the
  accumulated update instead, with a warning. `--clip_mode fixed` without it keeps
  `accelerator.clip_grad_norm_`.
- `--clip_sigma_bands` keeps one ring buffer per band and uses the global cap until the band
  holds 5 norms; a unit of several micro-batches takes their mean σ.
- Stamped: `ss_clip_mode`, `ss_clip_k`, `ss_clip_per_micro`, `ss_clip_sigma_bands`.
  `--grad_sigma_log` writes `<output_name>.grad_sigma.jsonl` (per clipped unit: norm, cap, σ,
  loss, image key); `_archive/bench/anima_abclip/sigma_tail.py` reads it.
- Logged: `grad/norm`, `grad/norm_sq`, `grad/norm_max`, `grad/clip_frac`, `grad/clip_cap`,
  and `grad/micro_norm_sq` under accumulation.
- `tests/test_grad_clip.py`: typical norms pass the median clip, a spike lands at `clip_k ×`
  the median with its direction kept, fixed matches `torch.nn.utils.clip_grad_norm_` bit for
  bit, per-micro at accum 1 matches the global clip, and the per-micro sum of a real backward
  matches plain accumulation.

## Files

- `_archive/bench/anima_abclip/run_ab.py` — every A/B above (`--exp`), `report_<exp>.md` beside it.
- `_archive/bench/anima_abclip/sigma_tail.py`, `sigma_gate.py`, `drift_share.py` — σ profile, σ gate,
  drift split.
- `bench/anima_ema/render_ab.py` — EMA renders and the shared render loop.
