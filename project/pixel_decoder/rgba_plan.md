# RGBA output — plan (2026-11-22)

The P5 topic "RGBA" of `plan.md`, as its own plan: an alpha readout at ×1 for renders on a white
or light simple background, trained on real cutouts composited onto such backgrounds. This file is
the training strategy.
Crawling and filtering are the user's and are not planned here. R0 and R1 ran 2026-11-22
(`reports/rgba_r0_r1_2026_11_22.md`); the steps below are what is ahead.

## Priority

### Facts (as set)

- The user, 2026-11-22, on 8 generated renders through R2's head (`rgbar2_1122_1550`, base 28 steps
  CFG 4, 4 prompts × seeds 0, 1; `output/pixel_head/rgbar2_1122_1550/renders/` (local)): "반투명이고
  나발이고가 아니라 아예 블록처럼 쪼개지는게 문제니까 그 부분만 플랜에 달아놔 반투명은 2순위고 1순위는
  무조건 깔끔하게 투명으로 나뉘어야해".
  1. A clean split into transparent and opaque, with no blocks: none inside the foreground, none
     left in the background.
  2. Translucency, after 1.
- It replaces the order of R2's question (`rgbar2_1122_1550`: "translucency first") and holds for
  the steps below.

## State

### Facts

- The user (2026-11-22): "white background, simple background가 들어간 이미지 페어만 사용, 흰 배경을
  투명으로 변환, 그걸 그대로 학습"; then, on labels from real alpha: "transparent background 및
  너가 말한대로 필터링해서 크롤링하고 그걸로 학습하면 괜찮을듯?"; then: "여기는 내가 크롤해오면 학습 전략만
  rgba_plan.md 만 쓰자 내가 알아서 필터링해올게".
- The user (2026-11-22, after the cutouts arrived): "흰색 말고도 연한 다양한 색 (실제 image dataset에서
  나오는 simple background 의 느낌을 쓰거나 합성하면 좋을듯) 을 쓰면 좋을듯? plan에서 split은 무시하고
  r0 r1 해보자".
- The delivered folder `$CAPTION_CORPUS_DIR/transparent` (local), measured 2026-11-22: 327 RGBA + caption
  pairs, 322 after reposts and near-identical variants, 8 artists, 290 of the files from one.
- Simple backgrounds of `image_dataset`, measured 2026-11-22: of the tagged images whose border
  one affine colour field holds, 574 in all, 488 are white and 80 light (luma ≥ 180), neutral greys
  and pale tints, some with a gradient and grain of a few levels.
- R0 `rgbar0_1122_1515`, done: at σ 0.0577 – 0.1364 the base head keeps 0.864 – 0.933 of thin
  strokes' contrast to the background on white, flat and gradient, VAE(x0) 0.963 – 0.975.
  Generated `white background` borders: median 255, 20 of 20. Colour prompts: 8 of 10 outside the
  light palette.
- R1 `rgbar1_1122_1515`, done: the whole head on 16 images × 3 backgrounds, 6,000 updates at LR
  5e-5: alpha L1 0.0047 fit, 0.0065 the same images on new backgrounds, 0.0387 on 16 images not
  fitted. Decoding with the estimated B is within 0.01 levels of the true B.
- The user (2026-11-22, after R1): "누가 up out 돌리래; full만 돌려". The whole head trains; the
  smaller arms are not run again.
- The LR check `rgbar1lr_1122_1535` (whole head at 2e-4) was cancelled by the user at 4,500+ of
  6,000 updates; its last log line: fit L1 0.0015, transition 0.0224, against 0.0049 / 0.0768 at
  5e-5 and the same step.
- The user (2026-11-22): "지금 학습 취소하고, 5e-5로 바꾸고 1k로, lpips 넣어".
- Init is the ×1 base `output/pixel_head/x1ll_1122_1000/head.safetensors` (local), written out as a
  path, and not the release (`plan.md`, State).
- The base at σ 0.0577 against the original, test 30: 34.25 dB / LPIPS 0.0725. VAE(x̂0): 35.11 /
  0.0560 (`plan.md`, "The ×1 base recipe").
- A head trained on real latents read generated colour images at 30.90 dB against VAE(x_final),
  26 renders, and generated mono at 22.51, 6 renders. Its CFG feature and cond feature read
  within 0.03 dB (`archive/p3_2026_11_19.md` (local)).
- Background of generated white-background images, measured 2026-11-22 on existing sheets (local:
  `bench/pixel_readout/results/20261122-1215-x1llb28_1122_1000/`,
  `20261122-1222-x1llturbo_1122_1000/`): prompts `maid` and `chibi` × seeds 0, 1, the 12-px image
  border, VAE column and base-head column, 8-bit.

  | render | median | p1 – p99 |
  |---|---|---|
  | base, 28 steps, CFG 4 | 255 | 255 – 255 |
  | turbo v4_longer, 4 steps | 254 – 255 | 253 – 255 |

- `_flatten_alpha` of the resize stage already composites transparency onto 255 white.
- Local weights for the post-hoc matting baselines: none (the BiRefNet cache folder holds no
  snapshot).

### Interpretation

- R1 leaves two things open: fine strokes and translucency off the fit images (R2). On the fit
  images the cancelled LR check points at LR / length for the strokes before the head's form.
- The content need not be something Anima would generate: the head is teacher-forced on low-σ
  features of a given latent. It has to cover what the decoder will be used on. The weak axis so
  far is a content type (fine strokes, screentone), which alpha at hair tips may share.
- Exact 255 is what the DiT paints under `white background` (R0). The user's light colours widen
  what the head is for; saturated and dark ones stay outside it unless the palette is widened.
- With the background known, alpha is the only unknown: the semi-transparent foreground colour
  that `plan.md` lists as out of the head's reach follows from C, α and B. On a light colour B is
  read from the pixels the head calls background.
- A foreground of the background's colour has no pixel evidence. There the head can only use what
  the feature
  knows about the object, which is also where it could beat a colour or flood rule.

## The form

### Facts (as set)

- A separate alpha file in `PixelHead`'s form with one output channel: the base's weights, the
  output layer replaced (512 → 8 × 8 × 1, 32,832 parameters).
- Target a = 2α − 1. Loss `rec_loss` on it (L1, MSE, gradient) + 0.05 LPIPS on a as three
  channels, 512 crop (the user, 2026-11-22). No GAN or composite term.
- Decode, with C the RGB decode in [0, 1]:

      α = clamp((a + 1) / 2, 0, 1)
      B = affine colour field fitted on C where α ≤ 0.02
      P = clamp(C − (1 − α) · B, 0, α)      premultiplied foreground
      F = P / α where α > 0, else 0         straight colour, for PNG

- C can come from any ×1 decode: the base, the release or the VAE.

### Interpretation

- The GAN finish trains the whole trunk, so an alpha layer on the base's trunk does not fit the
  release's. A file with its own trunk keeps RGB untouched and works beside each decoder, for one
  more head pass at decode.
- Recomposited on a background B', an alpha error Δα becomes a colour error |Δα|·(1 − B'): none on
  white, all of it on black. So a composite loss on constant backgrounds adds nothing to L1 on α,
  and reads are made on black and on a checker, never on white alone.

## Data

### Facts (as set)

- Input: the delivered cutout folder, RGBA files with caption sidecars, as filtered by the user.
  It is the only source: no pairs from `image_dataset/`, no labels made from white backgrounds.
- A builder in `build_pages.py`'s form, one file per image:

  | item | rule |
  |---|---|
  | size | `fit_size`, about 4,100 tokens, native aspect |
  | resize | LANCZOS on premultiplied colour and alpha together |
  | input | composite on a background (below), VAE-encoded |
  | label | the resized alpha, uint8, no threshold |
  | background | `white` 255; `flat`, a light colour of `image_dataset`'s simple backgrounds ± 6 levels, half of the draws tinted; `field`, such a background's affine gradient with its grain |
  | caption | the sidecar through `parse_caption` / `compose_caption`, `transparent background` → `simple background`, plus `white background` on white; TE max-padded |
  | split | none by artist (the user); R2 leaves out R1's 16 read-only images |

- Checked once at build: counts, sizes, alpha histogram, composite | alpha sheets of the stress set.
- `pixel_pages` and the cached training set are not in the mix.
- The cache holds the DiT's inputs only (VAE latent, TE embed, alpha), no feature (the user,
  2026-11-22: "f를 왜 미리만들어 … 입력단만 해"). Backgrounds are drawn at cache time, `draws` per
  cutout and background kind; σ and noise per rollout, fixed for the read rows. A
  changed input re-encodes its rows only (`scripts/pixel_head/_alpha_cache.py`).

### Interpretation

- Per-image captions and not the pages' single caption, because foreground against background is
  semantic and inference runs with a prompt.

## Steps

Each run gets `project/pixel_decoder/configs/<handle>_mmdd_hhmm.toml` with this design in its
header. Common to all: base DiT, fp32 stream, DiT and head compiled, σ 0.0577 – 0.1364, through
the daemon.

### R2. Does it generalise, and beat post-hoc matting? (micro, 1,000 rollouts)

Facts (as set):

- Two jobs, extraction and training apart (the user, 2026-11-22: "코드 다시 짜 특징추출 (캐싱)이랑
  학습 분리"; `rgbar2_1122_1542`, which had both in one job, was cancelled at about 50 rollouts):
  1. cache `rgbacache_1122_1550`: `scripts/pixel_head/cache_alpha.py`, 918 train rows + 96 read, inputs only;
  2. train `rgbar2_1122_1550`, one arm, the whole head: `scripts/pixel_head/train_alpha.py`, the frozen DiT
     and the head on the card with LPIPS, no VAE or TE.
  The generated-image read and the matting references below follow them, not part of either job.
- Amount against R1, at 16 updates a rollout and LR 5e-5 from the base:

  | run | rollouts | updates | Σ LR against R1 | updates an image |
  |---|---|---|---|---|
  | R1 | 375 (equivalent) | 6,000 | 1 | 375 (16 images) |
  | micro | 1,000 | 16,000 | 2.72 | 52 (306 images) |

- Parent: the ×1 base recipe (`configs/pixel_head/base_x1.toml`). 16 updates per rollout, replay
  64, LR 5e-5, one cosine, warmup 200, seed 0.
- Changed together, not separated: target α for RGB, the cutout set for cache + pages, base points
  alone (no turbo point), LPIPS on the alpha. The whole head trains (the user, 2026-11-22).
- References on the same images:

  | reference | input | role |
  |---|---|---|
  | all opaque | — | null |
  | colour key against B | the head's C | null |
  | flood from the border, within 5 levels of B without its grain | the head's C | cheap baseline |
  | BiRefNet, skytnt anime-segmentation | the head's C | baseline to beat; weights to fetch first |

- Held-out numbers, per image and σ: alpha L1 over the GT regions:
  1. transition, 0.02 < α < 0.98;
  2. enclosed background, α ≤ 0.02 and not connected to the border;
  3. near-white foreground, α ≥ 0.98 and min RGB ≥ 240;
  4. translucent interior, 0.02 < α < 0.98 more than 4 px inside;
  5. thin strokes, under 7 px;
  6. translucency within 5 levels of B (no pixel evidence), reported apart;
  7. the whole image.
- The same regions for the flood and matting references on the original composite, so a decode error
  is not read as a matting error.
- Generated images: the working set, base 28 steps CFG 4, α from the CFG feature. No GT. Blind
  pairs alpha head | best post-hoc baseline, on black and on checker, one (prompt, seed) per file.
  Beside them: L1 between α of the CFG and of the cond feature.
- Bars: spread unknown. The user's read decides.
- Cost: with the DiT in the loop the cancelled run was at about 1.0 s a rollout at 50, so about
  17 min for 1,000; the base's 0.674 s is no estimate for this head.

Interpretation:

- Ahead of post-hoc matting on generated hair, holes or white on white → R3.
- Ahead on held-out, behind on generated → the gap is the domain. Read CFG against cond and the
  prompts' content before adding data.
- Level with post-hoc matting on both → the head's case is speed alone; the user decides whether
  that is worth a file.

### R3. The full run (hour-scale, on the user's word)

Facts (as set):

- R2's config with rollouts 1,000 → 12,000, nothing else changed; at most about 2 h 15 min.
- A seed twin for the spread, with the held-out noise seed fixed apart from `seed` first (today
  `--seed` moves both).
- Ship read on the second set below, against the best post-hoc baseline, blind. The user decides.

Interpretation:

- 12,000 at 5e-5 is the RGB base's length (`reports/x1_recipe_2026_11_22.md`); that it carries
  over to alpha is unmeasured.

## Prompt sets

Prefix `masterpiece, best quality, score_7, safe. `, each prompt ends with `white background.`

Working set, seeds 0, 1:

- P3's `maid` and `chibi`, as they are;
- `1boy, old man, long white hair, long beard, white robe, full body,`
- `1other, armor, white cape, spear, full body,`
- `no humans, white cat, long fur, whiskers, sitting,`
- `no humans, bicycle, side view,`
- `no humans, wine glass, glass bottle, transparent,`
- `no humans, campfire, smoke, sparks,`
- `no humans, potted fern, thin leaves,`
- `no humans, wooden chair, lace cloth,`

Ship set, seeds 7, 8, read by no arm before R3:

- `1boy, wizard, grey hair, wide sleeves, staff, full body,`
- `1other, robot, cables, antennae, full body,`
- `no humans, white horse, flowing mane,`
- `no humans, birdcage, thin bars,`
- `no humans, jellyfish, translucent, tentacles,`
- `no humans, dandelion, seeds,`

## Code to write

- Written 2026-11-22: `library/anima/pixel_alpha.py` (head, α → colour),
  `scripts/pixel_head/{cache_alpha,train_alpha,_alpha_data,_alpha_cache,_alpha_read}.py`,
  `tests/test_pixel_alpha.py`. The channel
  count goes into the vendored `pixel_head.py` at the ship step.
- `p3_generated.py`: a prompt-set switch and the alpha read.
- Decode, catalog, node and docs only after the ship read.

## Not in this plan

- Turbo σ 0.5: x̂0's outline can leave the original's, so GT alpha may not align. Its own probe
  after R3.
- Scenic, dark or patterned backgrounds.
- ×2 alpha; tuning the DiT for transparency (LayerDiffuse).
