# Plan (2026-11-19)

Phases are the proposal's (`proposal.md`); this file holds their order, what is settled and what
is open. Each run's header and numbers live in `bench/pixel_readout/README.md`.

## Done

### P1 — linear readout

`readout_1118_2219`: test 26.28. The user's read: "진짜 잘나오네".

### P2 — ConvNeXt head, reconstruction only

`convnext_1119_0938` (no skip): test 32.93, text
27.45, against 26.50 / 21.58 for the head with a P1 skip (`reports/p2_skip_2026_11_19.md`).
The user's verdict (2026-11-19): "대 합격". Claude's labelled read has the grid and grain gone
and the head a little softer than the VAE; on manga pages (`sheets_1119_1032`) the smallest
text and screentone dots stay soft.

## Recipe (settled unless a run says otherwise)

- No skip, out at default init, layer scale γ 1 (`--no_skip --ls_init 1.0`). The three were
  changed together and are not separated; no run is planned to separate them.
- Loss 0.5 L1 + 5 MSE + 0.05 L_grad, AdamW, cosine, clip 1.0, one DiT forward and one VAE decode
  per rollout.
- LR 2e-4 at batch 1, 4 updates per rollout (`0938`). `convnext_1119_0950` (gradient over 4
  rollouts, LR 1e-3, the user's arm) ended level with it: test 32.96 against 32.93, text 27.46
  against 27.45, after a faster start (27.57 against 25.69 at rollout 600).

## Next

### 1. A longer P2 run on the winning recipe

`0938` was still rising when its schedule ended
(+0.29 dB on test over the last 600 rollouts) and sits 4.1 dB under VAE(x̂0). Question: where
does the reconstruction-only head level off? Read: the same sheets and PSNR beside `0938`; the
user's read decides whether the softness left is worth P5. Cost: hour-scale, proposed to the
user before it is queued.

### 2. P3 — generated images and adapter transfer

16 prompts × 2 seeds with the standard
sampler, last-step features, VAE decode against the head; the same with a merged LoRA, head
still trained on base. Holds on both → a releasable decoder is on the table. Holds on base
only → an internal tool trained per model. Breaks on generated base images → train on sampled
final latents and repeat. The target here is VAE(x̂0), the decode of what the DiT produced,
which is FastFrames' target (`ceiling_1119_0920`: on teacher-forced inputs VAE(x̂0) against
the clean-latent target is 37.00 dB on test).

### 3. P4 — σ-conditioned head

Needed by cjk_rl arm D and turbo: σ over [0, 1], target
VAE(x̂0) at that σ, read at σ ∈ {0.06, 0.5, 0.9}.

### 4. P5 — pixel losses through the head

GAN on the head, arm D's reader NLL through the head,
or a ResShift refiner. Separate proposals once P3 and P4 read out.

## Open

- An upsample stage. One 2× stage at width 512 in place of the third block costs the same
  multiplies per block (4× the positions, ¼ per position) and about 2× that block's activations,
  and puts a 3×3 depthwise at 8-px spacing with a 512 → 192 output instead of 1024 → 768. Raised
  by the user 2026-11-19. Tried if grid or thin-outline errors remain after step 1; it replaces
  the proposal's "5 blocks, width 1536" as the first capacity lever.
- The features are taken under bf16 autocast; whether their rounding costs readout PSNR is
  unchecked.
- No spread for any number here: one seed per run. A seed twin of the recipe is owed before a
  gap under about 1 dB is read as a gain.
- MOSS: nothing until the Anima phases read out.
