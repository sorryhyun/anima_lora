# Pixel readout head — a light pixel decoder on the frozen DiT's last features

Status: the proposal the line opened on (2026-11-18). P1 and P2 are done; state and order of
work: [README.md](README.md), [plan.md](plan.md). Probe code and results: `bench/pixel_readout/`.
Moved here from `docs/anima/proposal/pixel_readout_head.md` when the line became its own
project (the user, 2026-11-19). The P2 head below is described with a P1 linear skip in the
code's first version; the skip was dropped (`reports/p2_skip_2026_11_19.md`).

## TL;DR

FastFrames (Chou et al., Apple, arXiv:2611.23226, §3.2 and App. C) drops the Wan VAE decoder and
reads pixels straight off the frozen DiT's pre-head features: a 64M head, three ConvNeXt blocks
at token resolution and a per-token linear to a 16×16 RGB patch, trained against the VAE decode
and then with a pixel-space GAN. Decoding goes from 147 ms to 1 ms, and the GAN stage gave the
same DiT weights sharper output than its baseline (CF++ → FastFrames-S).

This is the output half of L2P (`docs/anima/findings/l2p_pixel_transfer.md`). L2P swapped the
*input* to RGB patches and the frozen core left its native latent-token manifold; here the input
stays latent and the core runs as trained, and the head is a readout, not a generator. On Anima a
single linear readout already gives recognisable images on held-out artists (below), so the
question is no longer whether the features carry pixels but what a real head buys and what it
costs to keep one per model.

## Evidence so far

### Facts

Probe 1, `readout_1118_2219` (`bench/pixel_readout/linear_probe.py`, results
`bench/pixel_readout/results/20261118-2221-readout_1118_2219/` (local)):

- anima-base, features = the input of `final_layer.linear` (LayerNorm + adaLN-modulated,
  2048-dim per token), taken at σ 0.0577 (the last σ of the 50-step, shift-3 grid) from cached
  real latents with their cached captions, one forward per image.
- Readout: per-token linear 2048 → 768 (16×16×3) plus bias, ridge least squares (λ 1e-6 × mean
  diag) on 256 images; target = bf16 VAE decode of the clean latent.
- Test: 30 images from 8 artists held out of the fit (audirenze, asou_(asabu202), gin00, kimoshi,
  dorontabi, butterchalk, fizz_(pixiv34498626), sushispin).

| Test PSNR vs VAE decode | dB |
|---|---|
| linear readout | 26.28 |
| VAE image averaged per 16×16 patch, bilinear up | 22.07 |
| same, nearest up | 21.57 |

- Linear is above the patch-average reference on 30 / 30 images (+2.6 … +6.4 dB). Ridge 1e-4:
  26.13, 1e-2: 24.83.
- Claude's read (labelled sheets 00 and 08): faces, eyes, fingers, hair strands and the
  `Audirenze @audinze` watermark are legible; lower contrast and slightly washed colour; a fine
  paper-grain noise over flat regions (white background, skin); thin dark outlines thinner or
  broken (bikini edges).
- The user's read: "진짜 잘나오네".
- Timings in the same job (5070 Ti): VAE decode 0.14 s / image at ≤ 1024-tier, bf16, 2D-folded;
  DiT forward 0.30 s / image under `compile_blocks(dynamic_seq=True)`, mixed shapes.

### Interpretation

The residual errors are the ones a per-token linear map cannot fix: it has no view of the
neighbouring tokens, so edges that cross a 16-px patch border and flat regions spanning many
patches are reconstructed patch by patch. That is what FastFrames' 3×3 depthwise blocks add.
The probe is teacher-forced (real latents at σ_last), not read off a sampled trajectory; at
σ 0.058 the input is almost the clean latent, and the readout is applied once, so the L2P
failure mode (a field that does not compose over steps) does not apply, but it is still
unchecked on generated images.

## What a head would buy, in order

1. Pixel-space losses at a fraction of the VAE's memory. cjk_rl arm D
   (`project/cjk_rl/proposal.md`) backpropagates a reader's NLL through x̂₀ → VAE decode → crop,
   and decodes only a latent window with gradient because a full-page decode backward does not
   fit beside the DiT and the reader. Through a head the whole page fits. A pixel GAN or a
   perceptual loss on the head is the same saving for turbo.
2. Sharpness from a GAN on the head alone, DiT untouched (FastFrames-S vs CF++, same DiT weights).
   For a 4-step turbo student this is a cheaper lever than another distillation run.
3. Decode memory and time. Anima's VAE at 1024² is 0.14 s against 28–50 DiT forwards, so speed
   alone is not a reason here. On MOSS the fp32 FLUX.2 decode at ~2 MP does not fit beside the
   DiT; a head would. Not before the Anima phases read out.

What it costs: a head is tied to the DiT whose features it reads. FastFrames states it has to be
fine-tuned per model, where the VAE is shared across a family. Anima ships as merges, LoRAs and a
turbo student, and users stack LoRAs, so whether a base-trained head survives an adapter decides
whether it can be released or stays an internal training tool.

## Phases

Each phase names its question and what each answer leads to.

### P1 — linear readout (done)

Recognisable on held-out artists → P2.

### P2 — ConvNeXt head, reconstruction only

- Head: the frozen `final_layer` adaLN modulation reused for the input (or a fresh one, same
  shape), 1×1 projection 2048 → width, three ConvNeXt blocks (3×3 depthwise, LayerNorm, 4×
  GELU MLP) at token resolution, per-token linear → 768, unpatchified. No upsampler. Width
  1024 to start (FastFrames keeps 1536 on a 1536-dim DiT).
- Input also carries x_t's latent patch (64 values per token) through a linear embed. It is
  near-redundant at σ 0.058 but costs nothing and is needed for P4 (below).
- Data: anima-base cache, all artists except P1's 8 held-out ones; features computed online at
  σ_last (no feature cache: 16 MB per 1024² image); targets = VAE decodes, decoded once to disk
  as uint8.
- Loss: FastFrames' spatial terms, 0.5 L1 + 5 MSE + 0.05 L_grad (finite differences); no
  temporal terms.
- Compile with the free-fit `seq_bands` of the data (P1 compiled one band, 4032–4200, and ran
  other sizes as static specialisations).
- Cost: ≈ 0.3 s / step at batch 1 on the 5070 Ti, so 5k steps ≈ 25–30 min plus the target decode.
- Read: P1's 12 test images, plus 8 held-out images with Japanese text (manga panels from the
  OCR sets) for glyphs and screentone, crops at 2×; three columns VAE | P1 linear | P2 head,
  labelled. Numbers beside: test PSNR vs VAE against P1's 26.28 (spread unknown). The user's read
  decides.
- Outlines and flat-region grain at VAE level → P3. Still visibly softer than the VAE → widen
  or deepen once (5 blocks, width 1536) before P3; if that does not move it, the GAN stage (P5)
  becomes a requirement, not an option.

### P3 — generated images and adapter transfer

- Render 16 prompts × 2 seeds with the standard sampler (50 steps, shift 3) and capture the last
  step's features; decode each final latent with the VAE and with the P2 head.
- Same with a merged LoRA (m14 or an artist LoRA from `output/`), the head still trained on base.
- Holds on base and on the adapter → a releasable decoder is on the table; P4/P5 apply to every
  derivative. Holds on base, breaks on the adapter → the head is an internal tool trained per
  model (arm D, turbo), and release is dropped. Breaks on generated base images → the
  teacher-forced training set is the gap; train on sampled final latents instead and repeat P3.

### P4 — σ-conditioned head (needed by arm D and turbo)

The 4-step shift-3 turbo grid ends its last step at σ 0.5, and arm D reads x̂₀ at steps 0–1. At
those σ the features predict a velocity, and x̂₀ = x_t − σ·v needs x_t, which is why P2 already
feeds x_t. P4 trains the same head with σ drawn over [0, 1] (adaLN from the DiT's t-embedding)
and the target the VAE decode of x̂₀ the DiT itself predicts at that σ (what the VAE path would
show), read at σ ∈ {0.06, 0.5, 0.9} against VAE(x̂₀).

### P5 — pixel losses through the head

Separate proposals once P2–P4 read out: (a) GAN on the head (FastFrames uses DINOv3 ViT-S/16
heads at layers 4/8/12 with a relativistic loss and R1/R2; the turbo line's disc code is in git
history); (b) arm D with the reader NLL through the head, its D0 cost probe rerun against the
VAE-window path; (c) a ResShift refiner in place of the GAN (below).

What the target is. FastFrames' reconstruction stage regresses the VAE decode of a sampled
rollout (App. C, Eq. 4): no ground-truth pixels exist for it, so the VAE is its ceiling. Only
the GAN stage reaches past it, through an unpaired discriminator on 10K real videos, with the
VAE-target reconstruction loss kept on. Our P2 is teacher-forced from cached latents, so each
training pair has its real image (`post_image_dataset/resized/`): a paired real-pixel target is
available here without a GAN. Generated images (P3) have no such pair.

ResShift, the finished SR sidecar (`project/finished/sr/`: ×2 / ×4 1-step RSD students, input
`lq` = LR pixels), connects in three places:

- **Chain swap, no training.** Head output in place of the VAE decode ahead of ResShift ×2.
  Forward only, minutes. Read: whether ResShift sharpens P1/P2's flat-region grain into texture
  that the VAE → ResShift path does not show (same images, side by side).
- **Head at 2× (32×32 per token).** Decode and ×2 in one readout. Under L1 / MSE a 1024-tier
  feature has no 2048 detail to regress, so this alone gives a blurred mean; it needs (a) or
  the refiner below.
- **ResShift refiner as the P5 lever.** Head output as the residual-shift start, a 1-step
  refiner to the VAE decode or to the real pixels, compared against (a) as "head + GAN" vs
  "head + ResShift 1-step". The SR line placed its ×2 ceiling in the teacher and the VQ-f4
  reconstruction floor (`project/finished/sr/STATUS.md`); the same backbone carries that floor.

## Not claimed

- No speed gain on Anima's 28–50 step sampling; decode is 0.14 s of it.
- Nothing about MOSS until the Anima phases read out (different DiT, 128-channel latents at /16,
  the same 16-px token).
- FastFrames reports DiT fine-tuning at low LR as unstable; this proposal keeps the DiT frozen
  throughout.
