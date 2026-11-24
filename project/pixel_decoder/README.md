# pixel_decoder — a light pixel head on the frozen Anima DiT's last features

The algorithm and what was found. What is ahead: `rgba_plan.md` and `x2_plan.md`; the weight-space
and LoRA proposal for ×2: `x2_idea.md`. Runs: `reports/`, and
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
| ×2 output | 4 px of the ×1 image | | LayerNorm, linear to an 8×8 patch of the 2× image per position; or, in a file stamped `x2_overlap`, to a 16×16 patch centred on the position, summed with its neighbours' under a sin² window |

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
| loss | rec; + 331.6 DINO on original-image targets | rec + 245.2 DINO + 1.0 dc |

- rec = 0.5 L1 + 5 MSE + 0.05 gradient L1.
- dc = L1 on 32-px block means (the tone term).
- DINO = the frozen DINOv3 ViT-S/16 distance (`DinoFeat`, after ViTok-v2): 4 random 224-px tiles
  per update at the same places of output and target, patch tokens after blocks 3 / 6 / 9 / 12,
  L2-normalised per token, MSE. Each weight puts the term's gradient norm over the head at that
  of the LPIPS-VGG term it replaced (0.05 on ×1, 0.1 on ×2), read on a trained head
  (`reports/x1_dino_2026_11_22.md`, `reports/x2_dino_2026_11_23.md`). The heads trained before
  2026-11-23 carry the LPIPS term.
- The entry points are in `scripts/pixel_head/` (modules in `train/`), their values in
  `configs/pixel_head/`.

### GAN term

- Discriminator: frozen PE-Spatial-B16, blocks 3 / 7 / 11 (0-based), each with a small
  convolutional head writing one logit per token.
- PE over DINOv3 ViT-S/16 and ConvNeXt-T as the tower: a linear probe of the original against
  `x1dll_1123_0821`'s output, 112 whole images, image-level split, best block's held-out
  accuracy, unpaired at σ 0.0577 / 0.1364: PE 0.880 / 0.894 (block 3), ViT-S 0.709 / 0.743
  (block 2), ConvNeXt-T 0.700 / 0.728 (the 8-px stage). Paired: PE 0.999 / 1.000, ViT-S 0.973 /
  0.989, ConvNeXt-T 0.924 / 0.938. The user's rule: a DINO tower that loses keeps PE
  (`bench/pixel_readout/experiments/disc_backbone_probe.py`, `discprobe_1123_1108`).
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
  user's read: "gan ft가 날카롭게 만드는건 확인했다".
- That finish has no term on the turbo point. Its amplitude there is 0.933 / 0.944, against
  0.920 / 0.931 for the head without a finish at 3,000 rollouts.

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
- The GAN crop, a 2,000-rollout finetune of the 12,000-rollout B head (30.55 / 26.16, 0.137 /
  0.134): at 1024 px 29.63 / 25.42, 0.178 / 0.181, 3.73 s per rollout; at 512 px 30.00 / 25.78,
  0.155 / 0.145, 1.54 s. Stamp rise over the parent on three images: 1024 +0.14 / +0.23 / +0.52,
  512 +0.02 / +0.05 / +0.21; on the two grainy originals the stamp is 1.59 / 1.89 at 1024 and
  1.59 / 1.75 at 512. The user: "512가 훨씬 낫다" ([record](x2_plan.md)).
- Noise inputs on the GAN finish (64 Gaussian channels per position before each second-stage
  block, zero-init): two decodes of one input differ by 0.08 – 0.09 levels at the end, after 0.14 –
  0.16 at rollout 500; tile numbers and dB level with the finish without them
  ([record](reports/x2_noise_2026_11_23.md)).
- On halftone the heads write one or two fixed token tiles in place of the dots: 16 centroids hold
  0.28 of those tiles' energy with the LPIPS head, 0.76 with the DINO head, 0.73 after the GAN
  finish; the chain's tiles at the same places 0.06
  ([record](reports/x2_dll_dgf_2026_11_23.md)).
- On flat regions every ×2 head with the 8 × 8 output steps at the patch borders (× 1.23 – 1.37
  of the inner first difference) and at the token borders (× 1.33 – 1.37); the chain reads × 1.0
  ([record](reports/x2_noise_2026_11_23.md)). With 16 × 16 patches at stride 8 summed under a
  window: × 0.99 / × 0.98, at 30.42 / 26.04 dB against its init's 30.48 / 26.13 and a high-pass
  ratio of 1.000 / 0.987 (σ 3 / 1 px); the decode takes 18.0 ms against 12.4
  ([record](reports/x2_overlap_2026_11_23.md)).
- The GAN finish on the overlapping head (`x2ovgf_1124_0920`, 4,000 rollouts at LR 2e-5), 32
  renders: amp 0.771 / 0.698 of the chain's against 0.770 / 0.692 for the finish on the 8 × 8
  output, token border × 1.08 against × 1.37, departed tiles' share on 16 centroids 0.718 against
  0.731; held out 29.69 / 25.62 dB ([record](reports/x2_ovgf_mix_2026_11_24.md)).
- A GAN finish and its parent mix linearly in weight space: at α 0.5 / 1 / 1.5 of the finish's
  change, amp 0.667 / 0.771 / 1.005, stamp32 0.517 / 0.657 / 1.115, dB against the chain 27.19 /
  26.86 / 25.38 (the parent 0.603, 0.414, 27.07). One stage's change alone: the second ×2 stage
  amp 0.787 with stamp32 1.068, the output layer 0.625 with 1.289, each with a tone shift of
  about 2 levels (same record; `bench/pixel_readout/experiments/head_merge.py`).
- The same change cut by singular directions: the top 8 per matrix (13 – 37 % of its energy)
  give amp 0.829 with stamp32 1.188 and a first difference on flat tokens of 0.91 levels; the
  rest 0.632, 1.695 and 1.94; the whole finish 0.771, 0.657 and 0.39 (same record).
- The discriminator's margin on held-out heads is 6.33 against a null of 0.05
  ([record](archive/x2_disc_probes_2026_11_21.md), local). In training it stays at 5.9 – 7.3 after
  its rise ([record](reports/x2_full_run_deploy_2026_11_21.md)).
- A regular pattern is read on crops: whole-image lag-16 autocorrelation is +0.0043 on the tiled
  head and +0.0051 on a tile-free one ([record](reports/x2_tile_gft_2026_11_22.md)).

### What the head loses

- Fine strokes: the user reported the region blurred at ×2. Claude's read, labelled: merged
  already by every ×1 head tried, separate in the VAE decode
  ([record](reports/x1_recipe_2026_11_22.md)).
- The ×1 form fits such windows: on the 16 held-out windows where the base head is furthest
  under VAE(x̂0) on the high-pass, an overfit of the released form reads 43.49 dB against the
  original, VAE(x̂0) 24.46 and the base head 23.65; 3 more blocks at 8 px 45.92, a 4-px stage
  42.04 ([record](reports/x1_fit_2026_11_24.md)).
- Screentone: small dots are smoothed into shading (the user's read).
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
- A perceptual term on a LoRA, decoded through the bf16 VAE window: LPIPS and DINOv3 at ratios 0.3
  and 1.0 each moved their own distance on the training draws by at most 3 %, and the user's blind
  reads stayed inside the spread (DINO 15 – 15 over two seeds at 0.3, 8 – 8 on one set at 1.0).
  Closed and out of the code since 2026-11-23
  ([LPIPS](reports/lpips_lora_2026_11_21.md), [DINO](reports/dino_lora_2026_11_23.md)).

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
- The ×1 base is local only, at `output/pixel_head/x1ll_1122_1000/head.safetensors`: `--init` /
  `--head_run` take the folder path, the bare run name does not resolve to a head.
- A GAN arm at a 1024 crop on ×2 needs window 32 and `manifest_w32.json`.
- The head's turbo point was trained on the local `models/loras/anima_turbo_4step.safetensors`
  (the turbo Hub's `old/` file).

## Rules

The user's calls on how this line is run and recorded.

### Records

- `README.md` is the algorithm and the facts found, with no history. `x2_plan.md` and
  `rgba_plan.md` hold what is waiting or ahead; a finished item leaves the plan, and an Open item
  names a concrete step.
- One report per phase or arm. `reports/` keeps the records still in use; a finished one moves to
  `archive/` (gitignored, cited with "(local)").
- The region where dense thin lines merge is called "fine strokes", not named by body part or
  prompt.

### Runs

- A GAN arm takes the PE term on a 512² crop, not 1024 (the user, 2026-11-22: "512가 훨씬 낫다고
  적어두자").
- The PE tower is compiled on ×2 (one crop size) and eager on the ×1 term (the whole image, a
  shape per image).
- The Hub ×2 file stays as it is until the user has read the new head's renders.
- The start eval is off by default (`start_eval`).
- The perceptual term is the DINOv3 ViT-S/16 term, in training and in the reads; LPIPS is out of
  both, and of the LoRA terms (the user, 2026-11-23: "dino로 전면 교체하는걸로 하자", "lpips는
  없애자"). Tables before that date carry LPIPS columns; new reads do not.
- A paper's head is ported in the paper's form first. When a small trainable part stalls, the
  16-image overfit probe (`bench/pixel_readout/experiments/x2_overfit.py`, about 70 s an arm) and
  a forward-only ceiling probe come before any argument about LR or budget.

### Reads

- ×2 is read by the DINO distance and crops, not dB (bicubic wins dB). No number sees the tile or the
  stripes.
- Texture is measured on native renders over many images.
- Renders shown to the user are Anima renders through the head only, with no GT or parent-run
  columns.

### RGBA

- Backgrounds are white and the light colours of the dataset's simple backgrounds.
- The whole head trains; no arm trains a part of it. A submit report names the arm list.
- The cache holds the DiT's inputs only (VAE latent, TE, the cutout with its colour); the frozen
  DiT runs in the training loop.
- `library/anima/` keeps the head and the decode only. The entry points are
  `scripts/pixel_head/*.py`; the training modules (trainers, losses, data splits, the RGBA
  read) `scripts/pixel_head/train/`, the RGBA cache and its data `scripts/pixel_head/cache/`.
- Artist names stay out of the code: the ×1 held-out split is the local
  `output/pixel_head/x1_split.json`.
- A read leads with the transparent / opaque split, translucency second (`rgba_plan.md`
  Priority).
