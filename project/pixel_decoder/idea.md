# Idea: perceptual loss through the head, into the DiT (2026-11-19)

The P5 topic "Pixel losses through the head, into the DiT" of `plan.md`, written down from a
chat on pixel-space DiT losses. Nothing here has run. The question: can the released head
(`mixr_1119_1455`) stand in for the VAE as a cheap differentiable decode, so that a LoRA on the
Anima DiT trains with a PixelGen-style LPIPS term on x̂0?

## Sources

### PixelGen (arXiv:2602.02493v2): adopt its form

Read from the PDF.

- Loss `L = L_FM + g(t)·(λ1·LPIPS + λ2·P-DINO) + L_REPA` on the x-prediction x̂θ against the
  real image; λ1 = 0.1, λ2 = 0.01. P-DINO = 1 − cos of DINOv2-B last-layer patch features
  (layers 6 / 9 / multi-layer worse, Table 5e).
- Noise gate `g(t) = 1[t ≥ 0.3]` in their convention (t = 1 clean), i.e. **on at σ ≤ 0.7** in
  ours. Without it recall drops (Table 5f: 0.58 at τ 0 vs 0.60 at τ 0.3; τ 0.6 costs FID
  7.53 → 10.72). App. A: perceptual-to-FM gradient-norm ratio is largest at high noise.
- ImageNet 256, 200K steps, no CFG (Table 5a): JiT 23.67 → + LPIPS 10.00 → + P-DINO 7.46 →
  + gate 7.53. Cost 0.19 → 0.20 s/it, 29.5 → 33.9 GB.
- Higher λ lowers recall (Table 5b, 5c).
- The authors' own caveat: the latent VAE is already trained with LPIPS + GAN, and latent
  diffusion "can also benefit from perceptual supervision" (LPL, Berrada et al. 2024, decoder
  features at low noise; PixelGen notes LPL's recall stayed stable across timesteps in latent
  space). Much of PixelGen's gain fixes uniform pixel-MSE weighting, which a latent model
  mostly does not have.

### AnisoFlow (arXiv:2611.17767v1): set aside

Read from the PDF. It learns a prediction target, not a loss: `u_W = W·ε − x` with PSD W, exact
closed-form velocity recovery, same FM loss. In pixel space it gives JiT-H/16 1.90 → 1.55.
In latent space it gives no change: SiT-XL/2-REPA 1.44 → 1.44, RAEv2 1.05 → 1.05, with the
learned weights near the default target (Table 4, Fig. 7). Anima is a latent velocity model
(64-dim 2×2 latent patch), the SiT case, and a LoRA starts from a converged velocity
predictor.

### DWD, "Decoding Without a Decoder" (arXiv:2611.22287v2): the same family as our head

Read from the PDF. Same group as AnisoFlow. A pixel head on a latent DiT, reading the same tap
as ours: the last-layer tokens after the final norm and modulation, i.e. the input of the
denoising head (`final_layer.linear` here), at the final step only, at a small nonzero t.

| | DWD | `mixr_1119_1455` |
|---|---|---|
| DiT | last K blocks trained (T2I 4 of 32, SiT 16 of 28, Wan 8 of 40), prefix frozen | frozen |
| head | linear, least-squares init from 10k images | ConvNeXt, 25M |
| input | tokens h | h + x_t 2×2 patch |
| target | the real image (Oklab out, losses on RGB) | VAE(x̂0) |
| loss | L1 + LPIPS 1:1, StyleGAN-T GAN late (256² crops, adaptive weight) | 0.5·L1 + 5·MSE + 0.05·grad |
| trajectory | anchoring: MSE of v to a frozen copy at t ∈ [0.04, 1], λ 5 – 500 | untouched (frozen DiT) |
| final-step CFG | cond only, no pair | f_u + g·(f_c − f_u) |
| cost | 8 – 4,300 H100 h | ~1 h on the 5070 Ti |

- Algorithm 1: the frozen prefix runs once under no_grad; the reference tail and the tuned tail
  share it. p_pix = [0, 0.1] before shift (4-step models: [0, 0.5] or [0.02, 0.3]).
- A linear head is enough once the tail trains. DWD-conv (a residual conv refiner on top) gives no
  consistent gain (Table 1). Their Fig. 3 frozen linear probe on h is our P1 (26.28 dB). With a
  frozen DiT the capacity has to sit in the head, as ours does.
- Real-pixel target beats the decoder. Wan2.1 14B warping error 1.52 → 1.18. The Wan decoder
  finetuned on the same data and losses reaches only 1.39 (B.1), so the gain comes from the
  method, not the data. On FLUX.2, whose decoder is already strong, the result is a tie
  (Limitations).
- Nonzero final t is needed (Fig. 4). At t_final = 0 the gain vanishes. Their reading: the noise
  at t > 0 leaves the head room to create detail the denoised latent would have smoothed.
- **Shared-head ablation.** Pixel losses put on the denoising output itself, at the same small t,
  with anchoring on: AnisoFlow FID 1.55 → 1.71, JiT 1.90 → 1.91 (Table 1). Their reading:
  ℓ2 favours mode coverage, LPIPS / GAN collapse diversity, and anchoring cannot hold it. This is
  the form of step 2 below.
- 4-step FLUX.2 klein: tied with the baseline with only the t ranges moved; final step at
  t = 0.7 after shift. 4-step Wan 1.3B: a self-distillation target (reference final step →
  renoise → small-t DWD) for the ambiguity of pairing a few-step output with real pixels (B.2).
- GAN helps pFID but is not critical (Table 2, "w/o adversarial loss").

## What carries over (Claude's reading)

- **LPIPS only.** P-DINO acts on global semantics, which live at higher σ than the head covers,
  and overlaps REPA (PE).
- **σ coverage is the main gap.** Under `configs/base.toml` (sigmoid sampling, scale 1, shift 1)
  σ ≤ 0.7 is ~80 % of training samples; the head's base range σ ≤ 0.136 is ~3 %. With the
  current head the term only touches the last steps' texture: line weight and screentone, the
  places the head itself is weakest against the VAE. Reaching PixelGen's window needs the P5
  topic "a head at every step" (σ up to ~0.7 is enough, since PixelGen gates out the blurry
  x̂0 regime anyway).
- **Null-space hacking.** The head reads the full 2048-dim input of `final_layer.linear`; only
  its 64-dim row space changes v. A loss backpropagated through the head can move f in the
  1984-dim null space and improve the head's output while the latent (and the VAE decode)
  stays the same. Fix: `f_in = f.detach() + P_W(f − f.detach())`. Forward is unchanged, and
  the gradient reaches f only through v, the same path as a VAE decode.
- **Target = VAE(x0), not the original.** The P1–P4 lesson: the latent → VAE path cannot
  carry detail past the VAE's ceiling. A real-image target asks for detail no v can deliver,
  and that gradient has nowhere honest to go.
- **Head Jacobian.** The head was trained on values (regressing VAE(x̂0)), not on derivatives,
  and may lean on null-space features. Whether its row-space gradient matches the VAE's is
  unknown; this is the first thing to measure.
- **Small-dataset LoRA.** Lower recall here means copying the training images' texture. Start
  λ below PixelGen's 0.1.
- An honest alternative decoder with no null space: a small latent-only decoder on x̂0 (a TAE
  for the Wan 2.1 / Qwen-Image VAE). Comparison arm if the head fails the probe.
- **DWD's shared-head ablation argues against step 2.** LPIPS into v at low σ is "pixel losses on
  the denoising output", which cost FID there even with anchoring. Their ablation is on pixel
  models (the denoising output is the image); ours would reach v through the head, but the
  conflict they name (conditional mean vs perceptual) is the same. Step 1 still holds as a
  measurement (also for cjk_rl arm D); keep step 2's expectations low.

## What DWD suggests for the head itself (Claude's reading)

Our head is DWD's frozen-DiT, VAE-distilled variant. DWD points at the levers for the head rather
than for the DiT:

- **Real-pixel target + LPIPS on the head** (P5 "a paired real-pixel target"). Head training is
  teacher-forced from cached latents, so each rollout has its resized image. The Anima VAE is
  21.5 – 39.8 dB from the original on manga pages, so there is room, unlike DWD's FLUX.2 tie.
  The pixel losses go on the head only, never on v, which is the separation DWD's ablation
  argues for.
- **Cond-only final step.** The head was trained on teacher-forced single-forward features (CFG 1)
  and is read on CFG-combined ones. A cond-only last step matches training and drops one forward
  (1/56 of a 28-step CFG render). One-seed read against the current decode; minutes.
- **A final-step-only tail LoRA** (not in the paper). DWD anchors v because its tuned tail runs at
  every step. A LoRA on the last K blocks switched on for the final forward only (`set_enabled`, as
  the turbo point already does) leaves the trajectory untouched by construction, with no anchoring.
  It gives the head attention-level context ahead of its 3×3 depthwise convs. Costs: the final
  step's v is discarded, and training backprops through K blocks (K 2 – 4 should fit 16 GB, not
  measured).

## Order

1. **Gradient-agreement probe.** At σ 0.06–0.14 on 16 training latents, compute
   ∂ LPIPS(decode(x̂0), VAE(x0)) / ∂v by (a) the VAE decode, (b) the head with the row-space
   projection, (c) the head without it. Read the cosine and norm ratio against (a), plus time
   and memory per step. Shares its measurement with cjk_rl arm D's D0 cost probe. A forward-
   and-backward probe, a few minutes on the daemon. Gate set before the run.
   Run 2026-11-21 (`lgrad_1121_1845`, `reports/lpips_grad_2026_11_21.md`): (b) not passed; steps
   2 and 3 as written here do not follow. The windowed VAE decode takes the head's place.
2. If (b) passes: one LoRA arm, `λ·LPIPS` on the head path at low σ only, VAE(x0) target,
   against FM-only on the same data. A texture-heavy set (screentone, line art). DWD's
   shared-head ablation predicts no gain or a diversity loss here.
3. If that shows an effect: widen the head's σ range toward 0.7 (P5 "a head at every step").

On the head side, from DWD, independent of steps 1 – 3: the cond-only final-step read (minutes),
then a real-pixel target with LPIPS on the head, then the final-step tail LoRA if the head alone
stalls.

`lpips` is in the venv (installed for the ×2 work, `bench/pixel_readout/x2_head.py`); nothing
in `library/` uses it yet.
