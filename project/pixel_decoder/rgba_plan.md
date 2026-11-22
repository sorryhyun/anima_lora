# RGBA output — plan (2026-11-22)

The P5 topic "RGBA" of `plan.md`, as its own plan: an alpha readout at ×1 for renders on a white
background, trained on real cutouts composited onto white. This file is the training strategy.
Crawling and filtering are the user's and are not planned here. No step below has run.

## State

### Facts

- The user (2026-11-22): "white background, simple background가 들어간 이미지 페어만 사용, 흰 배경을
  투명으로 변환, 그걸 그대로 학습"; then, on labels from real alpha: "transparent background 및
  너가 말한대로 필터링해서 크롤링하고 그걸로 학습하면 괜찮을듯?"; then: "여기는 내가 크롤해오면 학습 전략만
  rgba_plan.md 만 쓰자 내가 알아서 필터링해올게".
- Init is the ×1 base `output/pixel_head/x1ll_1122_1000/head.safetensors` (local), written out as a
  path, and not the release (`plan.md`, State).
- The base at σ 0.0577 against the original, test 30: 34.25 dB / LPIPS 0.0725. VAE(x̂0): 35.11 /
  0.0560 (`plan.md`, "The ×1 base recipe").
- A head trained on real latents read generated colour images at 30.90 dB against VAE(x_final),
  26 renders, and generated mono at 22.51, 6 renders. Its CFG feature and cond feature read
  within 0.03 dB (`reports/p3_2026_11_19.md`).
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

- The content need not be something Anima would generate: the head is teacher-forced on low-σ
  features of a given latent. It has to cover what the decoder will be used on. The weak axis so
  far is a content type (fine strokes, screentone), which alpha at hair tips may share.
- Compositing onto exact 255 matches the base renders measured, so there is no background
  augmentation. Two prompts are a small sample; R0 widens it.
- With the background known, alpha is the only unknown: the semi-transparent foreground colour
  that `plan.md` lists as out of the head's reach follows from C and α.
- White foreground on white has no pixel evidence. There the head can only use what the feature
  knows about the object, which is also where it could beat a colour or flood rule.

## The form

### Facts (as set)

- A separate alpha file in `PixelHead`'s form with one output channel: the base's weights, the
  output layer replaced (512 → 8 × 8 × 1, 32,832 parameters).
- Target a = 2α − 1. Loss `rec_loss` on it (L1, MSE, gradient). No LPIPS, GAN or composite term.
- Decode, with C the RGB decode in [0, 1]:

      α = clamp((a + 1) / 2, 0, 1)
      P = clamp(C − (1 − α), 0, α)          premultiplied foreground
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
  | input | composite on 255 white, VAE-encoded |
  | label | the resized alpha, uint8, no threshold |
  | caption | the sidecar through `parse_caption` / `compose_caption`, `transparent background` → `white background` (added if neither is there); TE max-padded |
  | split | by artist; posts without an artist by id |
  | held-out | 30 test images from artists outside train, plus a fixed stress set of 10 (hair tips, holes, white on white, semi-transparent) picked before any run is read |

- Checked once at build: counts, sizes, alpha histogram, composite | alpha sheets of the stress set.
- `pixel_pages` and the cached training set are not in the mix.

### Interpretation

- Per-image captions and not the pages' single caption, because foreground against background is
  semantic and inference runs with a prompt.

## Steps

Each run gets `project/pixel_decoder/configs/<handle>_mmdd_hhmm.toml` with this design in its
header. Common to all: base DiT, fp32 stream, DiT and head compiled, σ 0.0577 – 0.1364, through
the daemon.

### R0. Does the tap hold the cutouts? (forward only, minutes)

Facts (as set):

- The base head on held-out composites at σ 0.0577 and 0.1364, against the composite, beside
  VAE(x̂0): dB and LPIPS on the foreground's box (flat white inflates whole-image dB), and sheets
  original | VAE(x̂0) and VAE(x̂0) | head on the stress set.
- The working set below rendered once: the border values of the backgrounds.

Interpretation:

- Strokes present in VAE(x̂0) → R1.
- Strokes gone in VAE(x̂0) already → that detail is out of reach at this tap; it is recorded and
  not chased by training.
- Backgrounds off 255 → the composite colour and the decode's B follow the measured values before
  the cache is built for training.

### R1. How much of the head has to train? (16-image overfit, minutes)

Facts (as set):

- Features computed once for 16 train images, stress content included; the arms differ only in
  what trains:

  | arm | trains |
  |---|---|
  | `out` | the new output layer |
  | `up` | + `up_blocks`, `norm_out` (about 2.1 M) |
  | `full` | the whole head (about 23 M) |

- `rec_loss`, the run's optimizer, LR 5e-5. `out` also at 2e-4: its layer starts from random.
- Read: GT | arm, one file per (image, arm), on black; L1 per region beside it.

Interpretation:

- The smallest arm that holds hair tips and holes is R2's form.
- `full` does not fit either → the form question of `plan.md` (Open) comes first; a longer run
  would not answer it.

### R2. Does it generalise, and beat post-hoc matting? (micro, 2,000 rollouts)

Facts (as set):

- Parent: the ×1 base recipe (`configs/pixel_head/base_x1.toml`). 16 updates per rollout, replay
  64, LR 5e-5, one cosine, warmup 200, seed 0.
- Changed together, not separated: target α for RGB, the cutout set for cache + pages, base points
  alone (no turbo point), no LPIPS, R1's trainable part.
- References on the same images:

  | reference | input | role |
  |---|---|---|
  | all opaque | — | null |
  | colour key, 1 − min(RGB) | the head's C | null |
  | flood from the border, min(RGB) ≥ 250 | the head's C | cheap baseline |
  | BiRefNet, skytnt anime-segmentation | the head's C | baseline to beat; weights to fetch first |

- Held-out numbers, per image and σ: alpha L1 over four GT regions:
  1. transition, 0.02 < α < 0.98;
  2. enclosed background, α ≤ 0.02 and not connected to the border;
  3. near-white foreground, α ≥ 0.98 and min RGB ≥ 240;
  4. the whole image.
- The same four for the flood and matting references on the original composite, so a decode error
  is not read as a matting error.
- Generated images: the working set, base 28 steps CFG 4, α from the CFG feature. No GT. Blind
  pairs alpha head | best post-hoc baseline, on black and on checker, one (prompt, seed) per file.
  Beside them: L1 between α of the CFG and of the cond feature.
- Bars: spread unknown. The user's read decides.
- Cost: at most about 25 min at the base run's 0.674 s per rollout, plus evals (estimate; this run
  decodes no VAE target).

Interpretation:

- Ahead of post-hoc matting on generated hair, holes or white on white → R3.
- Ahead on held-out, behind on generated → the gap is the domain. Read CFG against cond and the
  prompts' content before adding data.
- Level with post-hoc matting on both → the head's case is speed alone; the user decides whether
  that is worth a file.

### R3. The full run (hour-scale, on the user's word)

Facts (as set):

- R2's config with rollouts 2,000 → 12,000, nothing else changed; at most about 2 h 15 min.
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

- `library/anima/pixel_head.py`: an output channel count on `PixelHead`, stamped `channels`
  (absent = 3), read by `load_pixel_head`.
- The cache builder (Data).
- `scripts/pixel_head/train_alpha.py` + `configs/pixel_head/alpha.toml` on `X1Trainer`'s loop, and
  an invariant test: a stamped one-channel file round-trips, a three-channel file loads as before.
- `p3_generated.py`: a prompt-set switch and the alpha read.
- Decode, catalog, node and docs only after the ship read.

## Not in this plan

- Turbo σ 0.5: x̂0's outline can leave the original's, so GT alpha may not align. Its own probe
  after R3.
- Coloured or scenic backgrounds: they need random-background composites and an estimate of B.
- ×2 alpha; tuning the DiT for transparency (LayerDiffuse).
