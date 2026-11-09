# MOSS t2i LoRA by self-distillation: the target image as the teacher's TE context

Status: implemented 2026-11-09 (`--distill`, `scripts/moss/distill_cache.py`). First runs on
channel (caststation).

Source: D-OPSD, arXiv 2605.05204 (Jiang et al., HKUST / Alibaba Z-Image team). Implemented here:
the paper's **off-policy** arm (§ 3.3, Fig. 6a, arm 3), not its on-policy method.

## Idea

A LoRA step today regresses the student on the flow target `ε − x₀` at a noised real image.
Here the target is the same model's prediction on the same noisy latent, conditioned on the
caption **plus the target image through the TE's vision tower**:

```
x_t   = (1 − σ)·x₀ + σ·ε                       (as SFT, same σ draw)
u_s   = v_θ (x_t, t, c_s)       c_s = TE(caption)                    grad
u_t   = v_θ̄ (x_t, t, c_t)       c_t = TE(target image, caption)      no grad
loss  = ‖u_s − sg(u_t)‖²
```

- **Student:** the LoRA being trained, θ, on the cached caption embeds, as in SFT.
- **Teacher:** the same DiT, either at Base (LoRA multiplier 0) or with an EMA shadow θ̄ of the
  LoRA. It differs from the student only in its conditioning: the TE has seen the target image.
- The student learns to produce from text alone what the image-conditioned model produces.

On MOSS, image + caption through Qwen3.5-9B's own vision tower brings characters, outfits, hair
colours and layout back zero-shot ([vision_tower_reference.md](../findings/vision_tower_reference.md)).
The paper had to graft Qwen3-4B into Qwen3-VL-4B to get such a teacher (App. D, Fig. 9) and names
native multimodal encoders like Qwen3.5 as the case where that goes away.

**Paper context.** D-OPSD proper rolls the student out from noise and distils on the visited
states (Eq. 6, Algorithm 1); on MOSS (50 steps, CFG 3) a rollout costs ~7–11× an SFT step. In
Fig. 6a (Z-Image-Turbo LoRA, curves only) SFT on target images loses quality mid-run, off-policy
distillation holds quality and is second on DINO similarity, on-policy is best on both. That SFT
collapse is a step-distilled model losing its few-step dynamics (App. B); MOSS Base LoRAs trained
by SFT keep their quality. The paper's EMA teacher decays at 0.9999 and it reports a frozen base
teacher as "stable and effective".

## Implementation

- **Teacher cache** (`library/moss/distill.py`, `scripts/moss/distill_cache.py`,
  `DistillCacheRequest`): `{stem}.moss_vis.safetensors` beside each TE file, one `embeds_<kind>`
  per caption the student can draw (`full`, `plain`, `v{k}`, `empty` = the image alone). The page
  is prefilled once and every caption continues from it (`vision_te.encode_variants`). Image at
  the latent's size (~1,000 vision rows at 1 MP). Same TE weights as the student cache (int8
  decoder, vision tower bf16). `--tail` sets the teacher's prompt tail (default `off`),
  independent of the student cache's. Stamped with captions, source digest, encoder and tail;
  re-encodes only what changed. 30 images take about a minute after loading.
- **Train loop** (`library/moss/train.py`, `training_loss(teacher=…)`):
  - `--distill off|base|ema`. `base`: multiplier 0 during the teacher forward. `ema`: a shadow of
    the adapter params at `--distill_ema_decay` (default 0.9999), swapped in through one scratch
    copy (`TeacherWeights`), independent of the saved `ema_window` EMA.
  - The teacher forward runs first, no grad, on the same `x_t` / timestep, cond branch only.
  - `--distill_mix λ`: target `λ·(ε − x₀) + (1 − λ)·u_t`; on the shared `x_t` this equals
    `λ·FM + (1 − λ)·distill` up to a constant.
  - `--distill_sigma_min s`: teacher only on steps with σ ≥ s, the flow target below (no teacher
    forward there). `--distill_sigma_ramp r`: the teacher's share rises linearly from 0 at
    `s − r` to 1 at `s`.
  - The flow MSE is logged as `fm` on teacher steps. The run JSON counts `distill_steps`.
  - The compile range spans the teacher's longer token axis (caststation: 4192–5613).
  - Refused with `delta_rows`, `trigger_registers` and `ref_adapters all`: their row masks are
    built for the student's sequence, and the teacher's has the image rows ahead of the caption.
- **Gate probe** `bench/moss_self_distill/xhat_probe.py`: one-step flow loss and x̂₀ by σ under
  `cap` / `img_cap` / `img_cap_sure` / `img`, Base, no LoRA, plus full renders per conditioning.
- Tests: `tests/test_moss_distill.py` (teacher weight swap, target replacement).

Cost (caststation, int8 Base, fp32 stream, 30 images ~4,100 image tokens): 1.45 s/step with a
teacher forward every step against 1.10 s for SFT; peak 13.20 GB against 13.19 GB. A σ-gated run
pays the teacher forward only on its teacher steps (hybrid95: 1.08 s/step).

## Probe on channel (caststation), 2026-11-09

`bench/moss_self_distill/results/20261109-1638-caststation/`, 30 images, int8 Base, fp32 stream:

| σ | 0.1 | 0.25 | 0.4 | 0.6 | 0.8 | 0.95 |
|---|---|---|---|---|---|---|
| FM loss, `cap` | 0.4735 | 0.2661 | 0.1957 | 0.1747 | 0.2323 | 0.4576 |
| FM loss, `img_cap` (tail off) | 0.4726 | 0.2653 | 0.1952 | 0.1739 | 0.2306 | 0.4402 |
| images with `img_cap` lower | 100 % | 100 % | 100 % | 90 % | 97 % | 100 % |
| ‖v_img_cap − v_cap‖² | 0.0041 | 0.0026 | 0.0018 | 0.0018 | 0.0029 | 0.0187 |

- `img_cap` is lower at every σ on nearly every image; the gap is ~10× larger at σ 0.95 than below
  0.8. x_t pins most of the image below 0.8, so the one-step gap there is small by construction.
- x̂₀ spread across ε is the same for both (0.0382 vs 0.0377 at σ 0.8).
- Image alone (`img`) is worse than `cap` below σ 0.95; the `sure` tail on the teacher adds nothing.
- Renders (CFG 3, 50 steps): `img_cap` brings back the composition, outfits, props and side
  doodles, and draws far cleaner figures than `cap`. Line weight and the pale flat palette stay
  closer to Base's.

## Runs on channel (caststation)

`channel_(caststation)_layout_sure` cache, `configs/moss/lora.toml` with lr 1e-4 unless noted,
`ema_window 0.139`, 240 steps, seed 0. The σ sequence is the same in every run, so a σ-gated arm
differs from SFT on its teacher steps only. Renders: `scripts/moss/generate.py` (int8, CFG 3, 50
steps), the three `lora_ab.py` caststation prompts, seeds 1–6, sheets in
`output/tests/caststation_distill_ab/`.

| LoRA (`output/ckpt_moss/caststation_*`) | target | teacher steps | FM loss, last 60 steps |
|---|---|---|---|
| `fwdpv` (2026-11-03) | FM | 0 | 0.2827 |
| `distill_base` | teacher (Base) | 240 | 0.2910 |
| `distill_base_lr2e4` | teacher (Base), lr 2e-4 | 240 | 0.2950 |
| `hybrid95` | teacher at σ ≥ 0.95, FM below | 19 | 0.2811 |
| `hybrid90` | teacher at σ ≥ 0.9, FM below | 50 | |
| `ramp85_95` | teacher share 0 at σ 0.85 → 1 at 0.95 | 85 (Σ share 49.5) | |

- Pure distillation's loss sits at 0.005–0.008 throughout. Muon normalises each matrix's update,
  so the loss scale does not set the step size; the LoRA's ΔW came out at ~90 % of SFT's.
- hybrid95 shows some promise by eye against SFT.

## Open questions

- **What the distill target carries.** `u_t − u_s` is the direction the image adds beyond the
  caption. Measure it per σ band and per content: composition, identity, palette, line work.
  Decompose the trained LoRA's effect the same way: renders with the LoRA under `cap` vs Base under
  `img_cap`, and the hybrid arms against SFT on held-out prompts. FM regresses on one noisy sample
  of the image; the teacher target is a model-side posterior mean. Where these differ is what FM
  cannot deliver, which is why the gain may be orthogonal to it.
- **Transfer.** Does the hybrid's gain hold on other artists (a line-art-heavy one, a painterly
  one), on a character set (ama mitsuki, where the concept is identity, the content the vision
  tower conveys best), and on i2i tasks? For glyph tasks the tower is 32 px per token and
  colorize on vision alone reached luma r 0.42 (`project/finished/moss_lora/done/i2i.md`), so
  there the teacher would carry layout, not strokes.
- **Other contexts for the teacher.** The teacher's embeds need not be image + the same caption:
  - a sparse student caption (trigger only, tags dropped) against image + full caption: the
    privileged-information form, the LoRA learns to fill in what the caption leaves out;
  - an instruction or style description placed before the image, which shapes the rows the DiT
    reads (the TE is causal);
  - the image at a different resolution (`img_lo` ~130 rows drew looser), or a crop;
  - an amplified difference, `u_s_base + γ·(u_t − u_s_base)` with γ > 1, so the image direction
    is not drowned when mixed with FM.
- **Where the teacher steps go.** σ ≥ 0.95 is 8 % of steps under the shipped σ sampling. A
  separate high-σ teacher term per step (a second student forward/backward) or a shifted σ draw
  on teacher steps would give it more updates.
- **EMA teacher.** At 240 updates a 0.9999 shadow moves 2.4 % toward the student, so it is Base.
  A faster shadow (0.99) makes the teacher the LoRA plus the image: FM teaches the style, the
  image context the composition.
- **On-policy at high σ.** Rolling the student out only over the first steps (σ 1 → 0.85, about 5
  of 50) and distilling there costs a few no-grad forwards and puts the teacher on the student's
  own composition errors, the regime where the paper's on-policy arm gained most.

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
