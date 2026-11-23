# RGBA output — plan (rewritten 2026-11-23)

One ×1 head that writes RGB and alpha together, for renders on a white or light simple background.
State: R5 `rgbar5_1123_2305` (the first run of the form) running, 2026-11-23.
The plan before this date (an alpha-only head in a file of its own, steps R0 – R4) is in git up to
`222d5dd0`; its results are under Records below.

## Goal and priority

### Facts

- The user (2026-11-23), on learning that the alpha head writes α alone and takes its colour from
  another decode: "지금 rgba pixel decoder는 오로지 a 하나만 출력해?", "대체 누가 그렇게 하랬어",
  "base에서 init한것도 아무 쓸모가 없잖아"; on a changed RGB target: "rgb는 냅두면 되잖아 그걸 왜
  타겟을 바꿔"; then: "일단 rgba plan 갈아엎어 다 지우고 새로써".
- The user (2026-11-23), on what the alpha-only form asks of the person using it: "니가 유저면
  투명배경 만들려고 헤드 두개 달아놓고 레이어 뽑고 그거 포토샵켜서 다시합성하고 참 좋겠다 그치?".
  What the head is for: one head file picked in the ComfyUI decode node, a transparent image out.
  No second head, no layer to extract, no compositing by hand.
- The alpha-only form was Claude's (the old plan, "The form"); no word of the user's set it.
- The user (2026-11-22), on 8 generated renders through `rgbar2_1122_1550`: "반투명이고 나발이고가
  아니라 아예 블록처럼 쪼개지는게 문제니까 그 부분만 플랜에 달아놔 반투명은 2순위고 1순위는 무조건
  깔끔하게 투명으로 나뉘어야해".
  1. A clean split into transparent and opaque, with no blocks: none inside the foreground, none
     left in the background.
  2. Translucency, after 1.
- The user (2026-11-23), after R3: "데이터 병목이네 그럼.. 쩝 일단 알겠다 데이터 늘리는걸로 하자".

## The form

### Facts (as set by the user, 2026-11-23)

- One head, one file, four output channels: RGB and α.
- The target is the RGBA image itself: "투명 a 채널이 붙어있는 이미지의 rgb는 그대로 동일하게 쓰고 a는
  추가로 하는건데". Where α = 0 the image has no colour; there the target is the composite's, the
  background the cutout was laid on (the user: "a=0이면 배경색 어 맞아").
- No extra measurement of the RGB (the user: "측정은 하지마 굳이 할필요 없는듯").

### As written (2026-11-23)

- `PixelHead`'s upsampled form with the output layer 512 → 8 × 8 × 4. Trunk and the RGB rows of the
  output layer from the init head, the α rows fresh (32,832 parameters more than the RGB head).
- Init `x1dll_1123_0821`, the head behind Hub `x1/anima_pixel_decoder_base.safetensors`. R1 – R4
  started from `x1ll_1122_1000`.
- Target (`Cutout.target`): RGB = the cutout's straight colour where α > 0, the composite where
  α = 0; a = 2α − 1.
- Loss: `lambda_rgb` · rec(RGB) + rec(a) + 331.6 · DINO(RGB) + 331.6 · DINO(a as three channels).
  331.6 is the ×1 weight on both (the base head trained its RGB with it); not measured on alpha.
- The whole head trains (`README.md`, Rules).
- Decode: the head's RGB is straight colour; saved with α = clamp((a + 1) / 2, 0, 1) as it is.
  No fringe step.

### Interpretation

- At α = 1 and at α = 0 the RGB target is the composite, what the base head was trained to write;
  only the partly transparent pixels move to the image's own colour.
- The trunk is shared: what keeping RGB costs α is not measured. It is R5's question.
- RGB is held on what the run sees: cutouts on white and light backgrounds.

## Records

### Facts

The runs' and caches' configs (`rgbar*`, `rgbacache*`) were deleted on the user's word (2026-11-23:
"rgbar config랑 rgbacache config도 다 지우고"); those up to R3 are in git up to `222d5dd0`, R4's and
its cache's were never committed. Outputs: `output/pixel_head/<run>/` (local).

- R0 `rgbar0_1122_1515`: on light backgrounds the RGB head keeps 0.864 – 0.933 of thin strokes'
  contrast, VAE(x0) 0.963 – 0.975. Generated `white background` borders: median 255, 20 of 20
  ([report](reports/rgba_r0_r1_2026_11_22.md)).
- R1 `rgbar1_1122_1515`: the whole head on 16 images × 3 backgrounds, alpha L1 0.0047 fit, 0.0065
  on new backgrounds, 0.0387 on 16 images not fitted; the output layer alone 0.2485 (same report).
- R2 `rgbar2_1122_1550`, 1,000 rollouts, 918 train rows: flat mid tiles on the 8 working renders
  598. The refine arm `rgbarf_1122_2110` (reverted): 573
  ([report](reports/rgba_refine_2026_11_22.md)).
- R3 `rgbar3_1123_1701`, 4,000 rollouts, a file at 2,000, R2's cache.
- R4 `rgbar4_1123_2053`: R3 (4,000 rollouts, LR 2e-5, DINO 331.6, init `x1ll_1122_1000`, warmup
  200, seed 0, a file at 2,000) on a wider cache (+ 500 SkyTNT cutouts, all 1girl solo;
  ootomo_takuji 279 → 36 train cutouts; train rows 918 → 1,701). Done 2026-11-23, rc 0, 2,075 s.
- Flat mid 8 × 8 tiles (`scripts/pixel_head/render_alpha.py`; 4 prompts × seeds 0, 1, base 28
  steps CFG 4; n = 1 run per column, spread unknown):

  | render | R3 2k | R3 4k | R4 2k | R4 4k |
  |---|---|---|---|---|
  | all 8 | 69 | 144 | 120 | 137 |
  | oldman s0 | 0 | 3 | 0 | 0 |
  | oldman s1 | 1 | 65 | 0 | 0 |
  | bicycle s0 | 21 | 24 | 44 | 76 |
  | bicycle s1 | 47 | 51 | 76 | 58 |
  | maid s0 + s1 | 0 | 1 | 0 | 0 |
  | glass s0 + s1 | 0 | 0 | 0 | 3 |

- Held-out alpha L1 (`result.json` history, 16 images × 3 backgrounds a set), R3 → R4 at 4k:
  `other` all 0.0247 → 0.0279, transition 0.2219 → 0.2255, translucent 0.2656 → 0.2787, thin
  0.1863 → 0.1901; `new_bg` all 0.0169 → 0.0180. 11 of the 16 `other` images are ootomo_takuji.
- Claude's read of R4 (labelled; `renders/cmp_{oldman_s1,bicycle_s0,bicycle_s1}.png`, on black +
  alpha): the robe's patches are gone in both R4 files; one small hole high in the beard in both;
  blocks inside the wheels in all four files; in R4 2k the space between the spokes is filled
  almost opaque.
- The user (2026-11-23), on these: "ㅇㅋ".

### Interpretation

- R4's blocks went where its new data was: people. The bicycle, an object with none in the new
  set, kept them. Object cutouts are the data to add (the user crawls and filters).
- The held-out `other` set leans on the artist whose train share R4 cut, so its small rise is not
  read as a loss.

## Data

### Facts

- Sources (the user's, filtered by the user): `$CAPTION_CORPUS_DIR/transparent` (322 cutouts after dedup, 8
  artists) and `$CAPTION_CORPUS_DIR/anime_seg/skytnt` (500, train only). Local.
- Cache `post_image_dataset/pixel_alpha_r4` (local), built by `scripts/pixel_head/cache_alpha.py`
  with `cutouts` the first folder, `extra_cutouts` the second, `thin = "ootomo_takuji:0.1"` (the
  user, 2026-11-23: "10%만"), `bg_source = "image_dataset"`, `draws = 1`, `n_read = 16`, `seed = 0`:
  the DiT's inputs only (VAE latent of the composite, TE, the cutout), 1,701 train rows, read sets
  `new_bg` and `other` of 16 images × 3 backgrounds. Backgrounds: white, a flat light colour, a
  light affine field with grain, from `image_dataset`'s simple backgrounds.
- Every cutout is stored with its colour (`rgbacache_1123_2254`, 2026-11-23: 551 train cutout files
  rewritten, no latent re-encoded).

## Next

### 1. R5 `rgbar5_1123_2305`: the first run of the form (running)

Facts (config `configs/rgbar5_1123_2305.toml`):

- Parent `rgbar4_1123_2053` (settings under Records). Changed together, not separated: the output
  form (1 → 4 channels), the RGB term and its DINO term, the init (`x1ll_1122_1000` →
  `x1dll_1123_0821`). Cache (rerun for colour), rollouts, LR, schedule, seed and the α term as R4.
- Question: does α on the shared trunk hold R4's, with RGB held at the base?
- Read, per file, beside R4:
  1. flat mid tiles on the 8 working renders (table above), with the on-black + alpha sheets of
     oldman s1 and bicycle s0 / s1, labelled;
  2. held-out alpha L1, the same regions;
  3. RGB dB as the scripts print it: against the target on the read sets, against the init head on
     the 8 renders.
- Bars: spread unknown (no seed twin). The user's read decides.
- Cost: about 41 min at 1.65 rollouts/s (R4: 2,075 s); one more DINO forward an update.

Interpretation:

- α level with R4 and RGB at the base → that file is the form; the R1 – R4 files go.
- α behind R4 → the RGB term's weight, before anything else.
- RGB off the base → the same weight, the other way.

### 2. After it

- Object cutouts into the cache (the user's data), then the same run.
- The ComfyUI node: a four-channel file returns a four-channel IMAGE; `make vendor-sync` for the
  pixel node. Local until the ship read; no publish, no Hub file.
- Ship read on the second prompt set below. The user decides.

## Prompt sets

Prefix `masterpiece, best quality, score_7, safe. `, each prompt ends with `white background.`

Working set, seeds 0, 1 (`render_alpha.py` runs the four marked *):

- `1girl, maid, frills, lace, detailed clothes, holding tray, full body,` *
- P3's `chibi`, as it is;
- `1boy, old man, long white hair, long beard, white robe, full body,` *
- `1other, armor, white cape, spear, full body,`
- `no humans, white cat, long fur, whiskers, sitting,`
- `no humans, bicycle, side view,` *
- `no humans, wine glass, glass bottle, transparent,` *
- `no humans, campfire, smoke, sparks,`
- `no humans, potted fern, thin leaves,`
- `no humans, wooden chair, lace cloth,`

Ship set, seeds 7, 8, read by no arm:

- `1boy, wizard, grey hair, wide sleeves, staff, full body,`
- `1other, robot, cables, antennae, full body,`
- `no humans, white horse, flowing mane,`
- `no humans, birdcage, thin bars,`
- `no humans, jellyfish, translucent, tentacles,`
- `no humans, dandelion, seeds,`

## Not in this plan

- Turbo σ 0.5: x̂0's outline can leave the original's, so GT alpha may not align.
- Scenic, dark or patterned backgrounds.
- The sharp (GAN-finished) RGB in the RGBA file.
