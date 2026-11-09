# MOSS t2i LoRA by self-distillation: the target image as the teacher's TE context

Status: proposal, 2026-11-09. Nothing run yet.

Source: D-OPSD, arXiv 2605.05204 (Jiang et al., HKUST / Alibaba Z-Image team). This proposal
takes the paper's **off-policy** arm (§ 3.3, Fig. 6a, arm 3), not the on-policy method.

## Idea

A LoRA step today regresses the student on the flow target `ε − x₀` at a noised real image.
Here the target is the same model's prediction on the same noisy latent, conditioned on the
caption **plus the target image through the TE's vision tower**:

```
x_t   = (1 − σ)·x₀ + σ·ε                       (as now, same σ draw)
u_s   = v_θ (x_t, t, c_s)       c_s = TE(caption)                    grad
u_t   = v_θ̄ (x_t, t, c_t)       c_t = TE(target image, caption)      no grad
loss  = ‖u_s − sg(u_t)‖²
```

- **Student:** the LoRA being trained, θ, on the cached caption embeds, as today.
- **Teacher:** the same DiT and the same LoRA with EMA weights θ̄, no grad. It differs from the
  student in its conditioning: the TE has seen the target image.
- **What it trains:** the student learns to produce from text alone what the image-conditioned
  model produces. The concept moves from the TE context into the LoRA. The target image is
  never a regression target.

The paper's premise is that a DiT on an LLM/VLM encoder reproduces an image's concept zero-shot
when the image enters the encoder. On MOSS this is measured
([vision_tower_reference.md](../findings/vision_tower_reference.md)): image + caption through
Qwen3.5-9B's own vision tower brings characters, outfits, hair colours and layout back, with no
training. The paper had to graft Qwen3-4B weights into Qwen3-VL-4B to get a usable teacher
(App. D, Fig. 9). It names Qwen3.5-style native multimodal encoders as the case where that
problem goes away, which is MOSS's TE.

## Why off-policy, and what the paper says about it

D-OPSD proper rolls the student out from noise along the inference schedule and distils on the
visited states (Eq. 6, Algorithm 1). On a 4–8 step distilled model the rollout reuses the
student's own forwards. MOSS samples 50 steps with CFG 3, so a rollout costs roughly 7–11× an SFT
step. This proposal keeps the SFT states (noised real images) and changes only the target.

Fig. 6a (Z-Image-Turbo LoRA) compares four arms. Below is read off the curves; the paper gives
no numbers.

| arm | DINO similarity to targets | Quality score |
|---|---|---|
| SFT on target images | slowest | drops sharply mid-run |
| SFT on teacher samples | dips, then rises | about flat |
| **off-policy distillation (this proposal)** | rises steadily, second | about flat |
| on-policy distillation (D-OPSD) | fastest, highest | highest |

The paper's reading: self-distillation stops SFT's quality loss; on-policy adds convergence speed.

**Caveat for MOSS.** The SFT collapse in the paper is a step-distilled model losing its few-step
dynamics. App. B says multi-step models tolerate the train-test mismatch, and MOSS Base LoRAs
trained by SFT keep their quality (ama mitsuki, `docs/moss/optimization/int8.md`). So the quality
argument may not carry over. What could still help:

- **Lower-variance target.** The teacher's velocity is a posterior mean given the image; `ε − x₀`
  is one sample.
- **Target on the base model's manifold.** Dataset artifacts (scan noise, JPEG, one artist's line
  weight) reach the LoRA only as far as the teacher reproduces them.

## Teacher weights

The paper's EMA decay is 0.9999 over 1K updates. Over 848 updates (the ama mitsuki run) that
shadow moves 1 − 0.9999⁸⁴⁸ ≈ 8 % of the way to the student, so the teacher is mostly the base
model plus the image context. The paper reports that a frozen base teacher is "stable and
effective", that 0.9999 is best, and that the student copy itself collapses.

The shipped `ema_window 0.1` (decay 0.988 at 848) is far faster than 0.9999 and is not the teacher.
The teacher needs its own shadow, independent of the saved EMA.

Arms (below): frozen base (`multiplier 0`, as `preserve_loss` already does) and EMA 0.9999.

## Implementation

1. **Teacher cache:** `{stem}.moss_vis.safetensors` beside each TE file.
   - Writes `vision_te.encode(target image, caption)` for each caption the student trains on:
     `embeds`, and `embeds_plain` for `plain_caption_rate`.
   - Image at ~1 MP, about 1,000 vision rows. The finding's `img_lo` at ~130 rows drew looser.
   - Uses the same TE weights as the student cache (int8 decoder, vision tower bf16:
     `vision_te.load_te(precision="int8")`).
   - `prompt_tail off` on the teacher, the setting the vision finding measured; generate with
     `ref_vision` forces it off too. The student keeps its cache's tail.
   - Stamps encoder, caption format, max pixels and image digest.
   - ~0.4 s per encode, so a 106-image set takes about a minute after loading.
2. **Train loop** (`library/moss/train.py`, `training_loss`): a `distill` request field
   (`off | base | ema`) plus `distill_ema_decay`.
   - `training_loss` already writes `noisy` / `timestep` to `state`.
   - Before the student forward, run the teacher under `torch.no_grad()` on the same `noisy`
     with the teacher embeds:
     - `base`: `network.set_multiplier(0)`.
     - `ema`: `_foreach_copy_` the teacher shadow into the adapter params, then copy them back.
       That needs one LoRA-sized scratch buffer.
   - The target becomes the teacher output in place of `noise − latent`. Nothing else in the loss
     changes.
   - Student and teacher both use the cond branch only, as SFT does. CFG stays an inference-time
     matter.
3. **Compile range:** the teacher's token axis is longer by the vision rows. `seq_range`
   (`train.py:279`) must span `n_img + n_text_teacher`, or the teacher forward recompiles.
4. **Memory:** the teacher forward holds no activations past its output. The peak stays the
   student's fwd/bwd (10.95 GB on the int8 plan) plus the longer no-grad forward's transient.
   Check once with `activation_memory_budget 0.3`.

Cost per step: the SFT step (1.08 s on ama mitsuki) plus one no-grad forward about 25 % longer.
Estimate ~1.5 s/step, about 21 min for 848 steps.

## Gate before training (minutes, no LoRA)

Fig. 7: when the teacher cannot draw the concept, training fails. Two checks on the training set:

1. **Renders.** A few ama mitsuki captions, Base, CFG 3, 50 steps, three conditionings:
   caption only, image + caption, and image + a held-out prompt. Does image + caption draw her
   (face, hair, eyes, outfit) where caption only does not?
2. **One-step x̂₀ by σ.** This is the measurement the λ(σ) schedule comes from.
   - On the cached training latents, noise each latent at a fixed σ grid (e.g. 0.05, 0.1, 0.2,
     0.35, 0.5, 0.7, 0.85, 0.95), several ε draws per (image, σ).
   - Take the one-step estimate `x̂₀ = x_t − σ·v` under three conditionings: caption only (`c_s`),
     image + caption (`c_t`), and image only. Base model, no LoRA, cond branch.
   - Score x̂₀ against x₀:
     - latent MSE;
     - decoded PSNR (full and 64 px luma, as in the vision finding);
     - the velocity residual `‖v − (ε − x₀)‖`.
   - Plot each against σ, with the paired gap `c_t − c_s` per image.
   - Expected shape:
     - the gap is near 0 at low σ, where x_t pins the image;
     - it opens at mid and high σ, where the caption alone leaves content open.
   - The σ band where the gap is clearly positive is where the teacher adds information. If the gap
     is positive at low σ too, the teacher also helps with detail.
   - If no band shows a gap, the teacher carries nothing the student lacks; stop.
   - The x̂₀ variance across ε draws at fixed (image, σ) sizes the variance argument directly.
     This is the Stable Target Field quantity (Xu et al., ICLR 2023).
   - Cost: forwards only, 106 images × 8 σ × 3 conditionings × a few ε, a few minutes compiled.
     Decode only a subset for PSNR (the VAE decode wants the DiT off the card).

Stop here if the renders do not reproduce her or the gap is flat.

## Experiment

ama mitsuki (`post_image_dataset/moss/ama_mitsuki_layout`, 106 images), the shipped
`configs/moss/lora.toml` (int8 stream, Muon, lr 5e-5, rank 32, 848 steps), same seed for every arm:

| arm | target |
|---|---|
| SFT | `ε − x₀` (control; the existing 848-step run if its config matches) |
| distill-base | frozen base + image context |
| distill-ema | EMA 0.9999 + image context |

Each arm takes ~20 min on the daemon, about an hour for all three.

Read with `bench/moss_ab/lora_ab.py --arms er_sde:1.0` and `ab_cols.py`, judged by eye; the
loss never separated arms in [t2i_lora_recipe.md](../findings/t2i_lora_recipe.md), and here
the arms do not even share a target. Look at:

- likeness on training captions;
- held-out prompts (new scenes, outfits, poses) for overfit;
- dataset artifacts and style leakage on prompts without her.

## Related work

Cited from memory, not read for this proposal. Read the PDFs before relying on any number.

- **Privileged-information distillation.** LUPI (Vapnik 2009); Generalized Distillation
  (Lopez-Paz et al. 2016). This is the general form: a teacher with train-only information
  teaches a student without it.
- **Context distillation.** Askell et al. 2021; Snell et al. 2022, "Learning by Distilling
  Context". The LLM form of this idea: the same model with a prompt context is the teacher.
- **Guidance distillation.** Meng et al., CVPR 2023, "On Distillation of Guided Diffusion
  Models". The same structure on a diffusion model: the student regresses on the same model's
  stronger-conditioned (CFG) velocity. Here the stronger condition is the image context.
- **Stable Target Field.** Xu et al., ICLR 2023. Replaces the one-sample `ε − x₀` with a
  reference-batch posterior average at high noise to cut target variance. This is the argument
  for teacher targets at high σ.
- **DreamBooth prior preservation.** Ruiz et al., CVPR 2023. FM mixed with a base-model
  self-target; `preserve_loss` is of this kind.
- **On-policy line.** DAgger (Ross et al. 2011); GKD (Agarwal et al., ICLR 2024); MiniLLM
  (Gu et al.); Self Forcing (Huang et al. 2025, rollout training for autoregressive video
  diffusion).
- **Contrast: step distillation.** Reflow (Liu et al. 2022) and DMD / DMD2 (Yin et al.).
  Matching the teacher's instantaneous velocity on rollouts does not make a few-step model; the
  Anima turbo is DMD.

## Open

- **λ mix with the FM loss.** On the shared `x_t`, `λ‖u_s − (ε − x₀)‖² + (1 − λ)‖u_s − u_t‖²`
  equals `‖u_s − (λ(ε − x₀) + (1 − λ)u_t)‖²` plus a constant: one regression onto an interpolated
  target, at no extra cost. λ can follow σ: data target at low σ, where the teacher is not
  pixel-aligned and x_t already holds the detail; teacher at high σ, where `ε − x₀` is noisiest.
  The paper does not mix (§ 4 lists extra training constraints as future work). In an on-policy
  version the FM term needs its own noised-real states, so it costs a second student
  forward/backward and brings back the off-policy states. Add it only if pure distill loses likeness.
- **Fidelity ceiling.** The teacher is not pixel-aligned (luma PSNR in the finding), so fine
  detail (accessories, eye shape) may cap at the teacher's level. x_t carries the real image at
  low σ, where the teacher should follow it.
- **On-policy.** A rollout variant (short no-grad schedule, distil on one or two visited states)
  is the next step only if off-policy shows a signal.
- **Not for text or glyph tasks.** The vision tower is 32 px per token. Colorize on vision alone
  reached luma r 0.42 (`project/finished/moss_lora/done/i2i.md`), so swap_text and OCR-style
  targets are out of scope.
