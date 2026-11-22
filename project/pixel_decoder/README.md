# pixel_decoder — a light pixel head on the frozen Anima DiT's last features

The algorithm and what was found. What is ahead: the `*plan.md` files. Runs: `reports/`, and
`archive/` (local).

## Algorithm

After FastFrames (arXiv:2611.23226 §3.2, App. C): a small head replaces the VAE decoder at the last
sampling step. The DiT is frozen.

### Input and decode

- f: the input of `final_layer.linear` at the last forward (LayerNorm + adaLN, 2048 per token).
- x_t's 2×2 latent patch, 64 values per token.
- Under CFG, f = f_u + g·(f_c − f_u). The tap sits before a linear map, so it maps to v_cfg.
- Output: RGB in [-1, 1], 16×16 pixels per token at ×1, 32×32 at ×2.
- A file is valid at the σ points it was trained on (stamped `sigma`). Its scale is the file's.

### Head (`library/anima/pixel_head.py`)

| stage | spacing | width | operation |
|---|---|---|---|
| input | 16 px | 1024 | `proj(f) + lat_embed(x_t)` |
| trunk | 16 px | 1024 | 2 ConvNeXt blocks |
| upsample | 8 px | 512 | LayerNorm, linear, pixel shuffle ×2, 1 ConvNeXt block |
| ×1 output | 8 px | | LayerNorm, linear to an 8×8 RGB patch per position |
| ×2 second stage | 4 px of the ×1 image | 256 | from the upsample stage: the same shuffle, 2 ConvNeXt blocks |
| ×2 output | 4 px of the ×1 image | | LayerNorm, linear to an 8×8 patch of the 2× image per position |

A block is a 3×3 depthwise convolution, LayerNorm and a 4× MLP, with layer scale starting at 1.0.
There is no linear skip from f. Output layers run in fp32. A ×2 file has no ×1 output layer.

### A rollout

1. Draw a decode point σ and noise ε; x_t = (1 − σ)·x0 + σ·ε from a cached latent.
2. One forward of the DiT (fp32 stream, compiled blocks) gives f and v; x̂0 = x_t − σ·v.
3. Push (f, x_t patches, target) into a replay buffer of 64.
4. Make 16 head updates, each on one random sample of the buffer: AdamW, gradient clip 1.0, one
   cosine cycle with a 200-update warmup.

| | ×1 (`base_x1.py`, `train_x1.py`) | ×2 (`train_x2.py`) |
|---|---|---|
| decode points | σ 0.0577 – 0.1364 and turbo's σ 0.5, in turn | σ 0.0577 – 0.1364 |
| target | base point: the original image; turbo point: VAE(x̂0) | the real 2× image of the same crop |
| data | cached pairs, manga pages at a share of 0.33 | `x2flex`: masters downscaled to 2×, or exact master crops |
| replay sample | the whole image | a window of 24 tokens + a margin of 3, left out of the loss on sides inside the image |
| loss | rec; + 0.05 LPIPS on original-image targets | rec + 0.1 LPIPS + 1.0 dc |

- rec = 0.5 L1 + 5 MSE + 0.05 gradient L1.
- dc = L1 on 32-px block means (the tone term).
- The scripts are in `scripts/pixel_head/`, their values in `configs/pixel_head/`.

### GAN term

- Discriminator: frozen PE-Spatial-B16, blocks 3 / 7 / 11 (0-based), each with a small
  convolutional head writing one logit per token.
- Paired relativistic loss on the output and the real image at the same place:
  L_D = softplus(−(D(real) − D(fake))), L_G = softplus(−(D(fake) − D(real))).
- R1 / R2 in noise-perturbation form: 0.1 · mean((D(x + σε) − D(x))²), σ 0.01. LR 1e-4.
- Generator weight, set once per rollout on one replay sample:
  λ = ratio · ‖∇ regression‖ / ‖∇ L_G‖ over the head, at most 1.0.
- ×1 finish: original-image samples only, the whole image. ×2: a 512-px crop of the 2× output.

## Findings

dB is PSNR. "test / text" at ×1: 30 / 8 held-out images; at ×2: 12 / 5, against the real 2× image.
One seed per number. Where no twin gap is given, the spread is unknown.

### Form and input

- A per-token linear readout gives test 26.28 dB against VAE(x0); the patch-mean reference 22.07
  ([record](archive/p1_readout_2026_11_18.md), local).
- With a linear skip, zero-init output and layer scale 1e-6 the head stayed at 26.50. Without the
  skip and at layer scale 1.0: 32.93. VAE(x̂0), the DiT's own prediction: 37.00
  ([record](archive/p2_skip_2026_11_19.md), local).
- The 2× upsample stage: test / text 32.93 / 27.45 → 33.64 / 28.04, 38 / 38 images up
  ([record](archive/upsample_2026_11_19.md), local).
- Linear probe of other inputs: the previous step's feature adds 0.24 dB to the last one's, block
  18 adds 0.10. Alone they are 0.39 and 2.74 under it
  ([record](archive/feat_probe_2026_11_20.md), local).
- Far off its σ: a head trained at 0.0577 gave 13.73 dB on turbo's σ 0.5 colour renders, 34.58
  after a warm-started training at σ 0.5
  ([record](archive/p4_t2_2026_11_19.md), local).
- The int8 DiT against bf16, 32 renders each: 32.32 / 32.44 on base, 34.60 / 34.61 on turbo. The cond-only feature against the CFG one at CFG 4: within 0.1 dB on the mean
  ([record](archive/int8_cond_2026_11_19.md), local).
- Decode, medians at about 1 MP on a 5070 Ti: head 6.7 ms (14.0 ms with the CFG combine) against
  141 ms for a resident VAE; 0.24 GB against 1.30 GB above the DiT
  ([record](archive/release_2026_11_19.md), local).

### ×1 training

At σ 0.0577 against the original image, dB / LPIPS. The twin gap of two 3,000-rollout runs is at
most 0.06 / 0.0013 on test ([record](reports/x1_recipe_2026_11_22.md)).

| change from 3,000 rollouts at LR 2e-4 | test | text |
|---|---|---|
| none | 33.75 / 0.0807 | 28.61 / 0.0671 |
| σ range 0.03 – 0.25 | 33.54 / 0.0837 | 28.42 / 0.0716 |
| page share 0.1 | 33.91 / 0.0758 | 28.62 / 0.0633 |
| LR 5e-5 | 33.97 / 0.0769 | 28.87 / 0.0634 |
| 12,000 rollouts | 34.13 / 0.0741 | 29.06 / 0.0596 |
| 12,000 rollouts at LR 5e-5 | 34.25 / 0.0725 | 29.20 / 0.0580 |
| reference: VAE(x̂0) | 35.11 / 0.0560 | 30.29 / 0.0375 |

- A warm start at LR 2e-4 stays under its init's dB until about rollout 9,500 of 12,000. At 5e-5
  the dip is about a quarter as deep.
- A GAN finish (2,000 rollouts, ratio 2) on the last head: test 33.33 / 0.0828. High-pass
  amplitude on base renders, colour / mono: 0.880 / 0.850 → 0.986 / 0.941 of the VAE's. The
  user's read: "gan ft가 날카롭게 만드는건 확인했다" ([record](plan.md)).
- That finish has no term on the turbo point. Its amplitude there is 0.933 / 0.944, against
  0.920 / 0.931 for the head without a finish at 3,000 rollouts ([record](plan.md)).

### ×2 training

- A second stage regressed onto an upscaler's output: 30.70 dB against that teacher, where the
  teacher on the head's ×1 output reads 33.71. The stage fits 2 windows at up to 41.75
  ([record](archive/x2_arm_a_2026_11_19.md), local).
- The upscaler's critic as DMD: 29.65 / 0.152 on test against its control's 29.76 / 0.141
  ([record](archive/x2_critic_2026_11_20.md), local).
- The real 2× image as the target, 4,000 rollouts: 30.45 / 25.92 dB, LPIPS 0.141 / 0.137. The ×1
  head followed by the upscaler: 30.06 / 25.98, 0.171 / 0.114. With no ×1 loss the file's ×1
  output falls to 24.76 dB on test ([record](reports/x2_b4k_2026_11_21.md)).
- GAN from rollout 1000 of 4,000, on a head not yet converged: 29.82 / 25.52, 0.158 / 0.145. The
  user reported pinwheel tiles on fine texture
  ([record](reports/x2_tile_gft_2026_11_22.md)).
- GAN finetune on the converged head, 1,000 rollouts: 29.74 / 25.40, 0.161 / 0.149; twin gap
  0.08 / 0.07 dB, 0.002 / 0.001. Claude's read, labelled: no pinwheel on either seed, a diagonal
  weave on screentone (same record).
- The same for 3,000 rollouts: 29.86 / 25.48, 0.152 / 0.135. The user's read: "확실히 길게 학습하는게 좋네".
  Claude's read, labelled: parallel stripes (same record).
- The discriminator's margin on held-out heads is 6.33 against a null of 0.05
  ([record](archive/x2_disc_probes_2026_11_21.md), local). In training it stays at 5.9 – 7.3 after
  its rise ([record](reports/x2_full_run_deploy_2026_11_21.md)).
- A regular pattern is read on crops: whole-image lag-16 autocorrelation is +0.0043 on the tiled
  head and +0.0051 on a tile-free one ([record](reports/x2_tile_gft_2026_11_22.md)).

### What the head loses

- Fine strokes: the user reported the region blurred at ×2. Claude's read, labelled: merged
  already by every ×1 head tried, separate in the VAE decode
  ([record](reports/x1_recipe_2026_11_22.md)).
- Screentone: small dots are smoothed into shading (the user's read; [record](plan.md)).
- Tone: darker than the VAE on textured regions, about 2.5 levels in linear light on
  screentone-like blocks of base mono renders, 1 on turbo, under 1 over a whole image
  ([record](archive/p4_mix_2026_11_19.md), local).

### Gradient into the DiT

- The head's LPIPS gradient as v receives it has cosine 0.115 / 0.154 / 0.166 to the fp32 VAE's at
  σ 0.0577 / 0.1 / 0.1364. The floor (another noise draw) is 0.060 / 0.078 / 0.081, the bf16 VAE
  0.511 / 0.495 / 0.434 ([record](reports/lpips_grad_2026_11_21.md)).
- 0.032 – 0.038 of that gradient lies in the 64 directions that set v; 64 / 2048 = 0.031 (same
  record).
- One 512² window backward: head 0.019 s / 0.70 GB, bf16 VAE 0.087 s / 2.46 GB (same record).

### Alpha

- On light backgrounds the RGB head keeps 0.864 – 0.933 of thin strokes' contrast, VAE(x0)
  0.963 – 0.975 ([record](reports/rgba_r0_r1_2026_11_22.md)).
- A whole head fits alpha on 16 images to L1 0.0047, 0.0065 on new backgrounds, 0.0387 on 16
  images not fitted. Training the output layer alone: 0.2485 (same record).
- The user's read of generated renders after 1,000 rollouts: "아예 블록처럼 쪼개지는게 문제"
  ([record](rgba_plan.md)).

### Traps

- An empty `init` in `train_x2.py` loads the catalog's head, not the ×1 base.
- `seed` also moves the held-out noise, so two seeds' evals differ in more than training.
- The inductor cache does not key on token bands: compile through `compile_dit`, which isolates
  it per signature.
