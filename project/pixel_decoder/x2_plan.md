# ×2 output — plan

- Done: `x2llgf1024_1122_1919` (GAN finetune 2k, 1024 crop, eager PE) from `x2bll_1122_1629`
  (B only 12k). Design and step 1: [the report](reports/x2_b12k_gft2k_2026_11_22.md).
- Done: `x2llgf512_1122_2209`, the same finetune at a 512 crop with the PE tower compiled
  ("The crop, 1024 against 512", at the end).
- The crop is settled at 512 (the user, 2026-11-22: "512가 훨씬 낫다고 적어두자"; README, Rules).
- Done 2026-11-23: `x2dll_1123_1537` (B only 12k, DINO, from `x1dll_1123_0821`) and its GAN finish
  `x2dgf512_1123_1701`, read on held-out sheets and 32 generated renders
  ([report](reports/x2_dll_dgf_2026_11_23.md)). Every ×2 head writes token seams (border / within
  first difference 1.23 – 1.28, chain 1.02).
- Done 2026-11-23: `x2nz_1123_1920`, noise inputs on the GAN finish: level with `x2dgf512` on
  every tile number, draw gap 0.08 – 0.09 levels ([report](reports/x2_noise_2026_11_23.md)).
- Done 2026-11-23: `x2ov_1123_2012`, the overlapping output (16 × 16 patches at stride 8 under a
  window) as a 2,000-rollout B finetune of `x2dll`: the patch and token steps are gone (flat
  tokens × 0.99 / × 0.98, token edge ratio on textured tiles 0.96 / 1.10), sharpness level with
  `x2dll`, 30.42 / 26.04 dB against 30.48 / 26.13 ([report](reports/x2_overlap_2026_11_23.md)).
  It closes the seam item (the shifted double decode and the border term are dropped).

## The ×2 form from here (2026-11-23)

- The user: "그럴거면 x2 자체를 stride 버전으로 하고, 거기에 gan을 얹는게 맞나? 뭐... 12k 학습을
  해야하는지는 모르겠다만... 일단 이건 x2 plan으로 갱신하자".
- The ×2 head's output is the overlapping one; the B head is trained with it and the GAN finish
  goes on top of that head. `x2dll` / `x2dgf512` (the 8 × 8 output) stay as the references.

Open, none queued:

1. The GAN finish on the overlapping output: `x2dgf512_1123_1701`'s recipe with resume `x2dll` →
   `x2ov_1123_2012` as the one change, about 30 min; the render read runs from
   `output/pixel_head/render_cache/` (local). Read against `x2dgf512` on the same 32 renders: the
   line gradient and high-pass (the user on `x2ov`: "진짜 좀 무르네"), the flat-token border
   ratios with the GAN's grain on, `tile_dict.py`, and the user's sheets of candle, kotatsu and
   manga_page.
2. Waiting on the user's read: `x2ovll_1124_0139` (the B stage retrained with the overlap from
   the start, 12,000 rollouts) against `x2ov` on the pair sheets
   ([report](reports/x2_overlap_12k_2026_11_24.md)). The numbers: held out 30.45 / 26.11
   dB against 30.42 / 26.04, DINO the same, the token border step on flat tokens × 1.19 against
   × 0.98. The GAN finish of step 1 resumes from `x2ov` unless the read picks `x2ovll`.
3. The fixed tile on halftone and the GAN's net (neither moved by the output form): the unpaired
   discriminator, `x2dgf512`'s recipe on the overlapping parent with the real crop drawn from
   another replay sample as the one change, about 30 min + the render read; read by the margin's
   course (6.3 – 7.0 with the pairing), `tile_dict.py` and the user's sheets. It leaves the
   paper's paired form.
4. The noise inputs again with a nonzero init and their own LR, if 3 moves the margin: only then
   does a loss reward the draw.

## A fixed pattern after the GAN finetune (2026-11-22)

### Facts

- The user (2026-11-22, on `x2llgf1024_1122_1919`'s sheets): "x2 gan은 뭔가 일정한 패턴이
  생겨버리네 … 뭔가 그 애매한게 있어".
- Measured on the training job's end sheets `<run>/<image>_full.png` (local; left half the 2×
  original, right half the head), the mean of the three channels, 5 images:
  - stamp: the head's high-pass (minus a Gaussian blur at σ 3 px) folded at 32 px (one DiT token
    at ×2) and averaged over all tiles; the std of that 32 × 32 mean, 8-bit levels. On the
    original of `test_00` the same measure reads 0.19 (0.06 at 8 px, 0.10 at 16 px).
  - seam: in flat regions of the original (smoothed gradient under 1 level), the residual
    head − original averaged per 8 × 8 cell; mean |difference| of horizontally adjacent cells
    across a 32-px token border / within a token.

  | image | stamp, `x2bll` | stamp, `x2llgf1024` | seam border / within, `x2bll` | `x2llgf1024` |
  |---|---|---|---|---|
  | `test_00_10153824` | 1.15 | 1.59 | 0.61 / 0.60 | 0.88 / 0.76 |
  | `test_01_10808923` | 1.18 | 1.89 | 0.56 / 0.54 | 0.75 / 0.68 |
  | `test_02_13340817` | 0.24 | 0.38 | 0.11 / 0.11 | 0.21 / 0.16 |
  | `test_03_13351964` | 0.26 | 0.49 | 0.43 / 0.42 | 0.55 / 0.48 |
  | `text_01_11318466` | 0.42 | 0.94 | 0.24 / 0.20 | 0.35 / 0.25 |

- The two highest stamps are the two originals with a visible paper grain (`test_00`, `test_01`).
- Claude's read (labelled; 1:1 crops of flat regions, original | head | difference × 4): on
  `test_02`'s skin the difference shows square patches of about one token at a slight colour
  offset; on `test_00`'s grain the difference is grain-like noise with no visible tile edge.
- Spread unknown: no seed twin of either run.

### Interpretation

- Two patterns, of different causes.
  1. Stamp: the same high-frequency texture in every token. Already in the B-only parent, raised
     1.4 – 2.2× by the GAN. The grain the originals carry is random and mostly not in the latent;
     the head has no source of randomness, so the texture that LPIPS and the GAN ask for can only
     be a fixed function of position inside the token where the feature is flat.
  2. Token seams: a per-token colour offset on flat regions. Absent in the parent (border = within),
     present after the GAN.
- A full-resolution refine stage after the output layer lets pixels on either side of a seam see
  each other and can address 2. It is deterministic as well, so it can learn the stamp as easily
  as the trunk; it does not address 1. On the alpha head the same stage (`rgbarf_1122_2110`,
  reverted) left that head's patches as they were, a decision error rather than a seam
  ([report](reports/rgba_refine_2026_11_22.md)); the ×2 seams are a colour offset, not shown to
  be the same kind.

### Proposal (not queued)

- Noise injection: per-pixel Gaussian noise channels, a fresh draw per forward, fed into the last
  ×2 stage (the up blocks), StyleGAN-style. With a refine stage written again for ×2.
- Run: the GAN finetune 2k again from `x2bll_1122_1629`, everything else as
  `x2llgf512_1122_2209` (the 512 crop). Cost about 50 min (2,000 rollouts at 1.54 s) plus the
  new stage's.
- Read: the stamp and seam numbers above on the same 5 sheets against both runs, and the sheets
  side by side; the user's read decides. Bars: spread unknown.
- What each answer leads to: stamp and seams down with grain kept → the form for the ×2 ship read.
  Stamp down, seams kept → the seams are the trunk's, not the output layer's. Stamp kept → the
  pattern does not come from missing randomness; look at the discriminator's crop phase
  (interpolated PE positions at 1024).

## The crop, 1024 against 512 (2026-11-22)

### Facts

- `x2llgf512_1122_2209` (job `20261122-220918-06c2f3`, rc 0, 2,000 rollouts) and the read
  `x2sp_1122_2209` (`bench/pixel_readout/experiments/x2_stamp_phase.py`; local:
  `bench/pixel_readout/results/20261122-2301-x2sp_1122_2209/`).
- Against the real 2× image, test / text:

  | run | dB | LPIPS | s per rollout | peak GB |
  |---|---|---|---|---|
  | `x2bll_1122_1629` (parent) | 30.55 / 26.16 | 0.137 / 0.134 | 0.78 | 9.1 |
  | `x2llgf1024_1122_1919` | 29.63 / 25.42 | 0.178 / 0.181 | 3.73 | 13.8 |
  | `x2llgf512_1122_2209` | 30.00 / 25.78 | 0.155 / 0.145 | 1.54 | 13.0 |

- Stamp at 32 px and seam border / within, as `x2_stamp_phase.py` reads them on the end sheets:

  | image | stamp parent | 1024 | 512 | seam parent | 1024 | 512 |
  |---|---|---|---|---|---|---|
  | `test_00_10153824` | 1.15 | 1.59 | 1.59 | 0.61 / 0.59 | 0.88 / 0.75 | 0.66 / 0.63 |
  | `test_01_10808923` | 1.18 | 1.89 | 1.75 | 0.56 / 0.53 | 0.77 / 0.68 | 0.62 / 0.57 |
  | `test_02_13340817` | 0.24 | 0.38 | 0.26 | 0.14 / 0.13 | 0.25 / 0.19 | 0.18 / 0.14 |
  | `test_03_13351964` | 0.26 | 0.49 | 0.31 | 0.43 / 0.41 | 0.55 / 0.47 | 0.50 / 0.44 |
  | `text_00_5847100` | 0.18 | 0.33 | 0.23 | 0.30 / 0.28 | 0.39 / 0.33 | 0.35 / 0.30 |
  | `text_01_11318466` | 0.42 | 0.94 | 0.63 | 0.29 / 0.23 | 0.42 / 0.28 | 0.33 / 0.24 |
  | `text_02_dan_5341366` | 0.26 | 0.49 | 0.38 | 0.25 / 0.23 | 0.33 / 0.28 | 0.30 / 0.25 |
  | `text_03_12540050` | 0.48 | 0.70 | 0.53 | 0.26 / 0.22 | 0.40 / 0.27 | 0.31 / 0.23 |
  | `text_04_10841155` | 0.45 | 0.99 | 0.77 | 0.55 / 0.54 | 0.72 / 0.66 | 0.62 / 0.57 |

- Stamp rise over the parent on `test_02` / `test_03` / `text_01`: 512 crop +0.02 / +0.05 /
  +0.21; 1024 crop +0.14 / +0.23 / +0.52; the 512-crop finetunes of `x2b4k` +0.02 – 0.03 /
  +0.05 – 0.06 / +0.11 – 0.25 (the config header's record).
- PE phase, d(run) / d(parent) at the last tap, the highest ratio off the 16-px grid minus the
  lowest on it, over the 9 images: 1024 crop +0.06 … +0.16, 512 crop +0.00 … +0.06.
- The user (2026-11-22, on these numbers): "512가 훨씬 낫다고 적어두자". The README's crop rule
  now reads 512.
- Sheets: `output/pixel_head/x2llgf512_1122_2209/*_crop.png` (local). Spread unknown: no seed
  twin.

### Interpretation

- By the rule in `archive/configs/x2llgf512_1122_2209.toml` (local): the 512 crop's stamp rise is inside the
  earlier 512 runs' and its off-grid penalty is small, so the 1024 crop raised the fixed pattern
  and costs 2.4× the time.
- On the two grainy originals (`test_00`, `test_01`) the stamp is as high at 512 as at 1024:
  the crop does not explain that part, and the noise-injection proposal above still stands
  for it.
