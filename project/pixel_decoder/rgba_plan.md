# RGBA output — plan

One ×1 head that writes RGB and alpha together, for renders on a white or light simple background.
One of the line's two open tracks since 2026-11-26 (the other: `x1_plan.md`). The run record up to
the preview: [reports/rgba_r2_r6_2026_11_24.md](reports/rgba_r2_r6_2026_11_24.md). The plan before
2026-11-23 (an alpha-only head in a file of its own, steps R0 – R4) is in git up to `222d5dd0`.

## Goal and priority

### Facts

- The user (2026-11-26): "당분간은 rgba가 crisp하게 혹은 x1이 좀 더 넓은 0.75 - 0 범위를 커버하도록
  학습하는 데 집중하려고 해". What "crisp" is read on (the α edge and the blocks, or the RGB's
  sharpness) is not said yet.
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

### Interpretation

- Claude's reading of "crisp", not confirmed: priority 1 above (the α edge and the blocks between
  thin structure), since that is what the preview fails on. If it is the RGB's sharpness, the
  last item of "Not in this plan" comes back in.

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
- RGB is held on what the run sees: cutouts on white and light backgrounds.

## Where it stands

### Facts

- The preview is R6 `rgbar6_1124_0034` (config `configs/rgbar6_1124_0034.toml`; R5 at LR 5e-5),
  its 4,000-rollout file: Hub `x1/anima_pixel_decoder_rgba_preview.safetensors`, catalog
  `anima_pixel_head_rgba`, node 0.4.0 (shipped 2026-11-24). Trained at the base points only.
- Alpha L1 at 4k (`result.json` summary, n = 1 run each, spread unknown). `new_bg` is 16 train
  cutouts on new backgrounds, `other` 16 cutouts outside train; R1 is the alpha-only head fitted
  on the 16 `new_bg` images alone:

  | band | R1 `new_bg` | R6 `new_bg` | R6 `other` |
  |---|---|---|---|
  | all | 0.0065 | 0.0170 | 0.0276 |
  | transition | 0.099 | 0.197 | 0.218 |
  | translucent | 0.075 | 0.209 | 0.272 |
  | thin | 0.095 | 0.149 | 0.177 |
  | holes | 0.025 | 0.068 | 0.115 |

- Flat mid 8 × 8 tiles on the working set's 8 renders: 264, of them bicycle s0 / s1 127 / 135;
  oldman, maid 0; glass 2. On the ship set (seeds 7 / 8): wizard 5 / 20, robot 0 / 0, horse 6 / 1,
  birdcage 309 / 199, jellyfish 351 / 109, dandelion 28 / 37.
- Claude's read of the ship set (labelled, on a checker and as alpha): wizard, robot and horse
  split with clean outlines; birdcage has blocks between the bars in both seeds; the jellyfish's
  veil is opaque white with blocks at its rim.
- R4 – R6 make 113 updates a cutout; R1 made 375 an image.
- No overfit of the four-channel form has been run; R1's fit is the alpha-only form's.
- The ×2 head's overlapping output (16 × 16 patches at stride 8 under a window, `PixelHeadX2`)
  took the patch-border step on flat regions from × 1.23 to × 0.99
  ([record](reports/x2_overlap_2026_11_23.md)); `PixelHead` (×1, RGBA) has no such output.

### Interpretation

- Holes: the train images fell to about half of R5's with the higher LR and the unseen ones moved
  0.016; what is left between 0.068 and 0.115 is the data's. The wheels and the cage are object
  cases, with no object cutouts in the cache.
- Transition and translucent stay near 0.2 on both sets, twice what R1 reached on the same
  images. The LR does not reach the edge band and more cutouts of the same kind would not either;
  budget, a GAN term on α and overlapping output patches are the untried levers. No ceiling for
  the band is measured on this form.

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

Nothing is queued. Which of the two comes first is the user's call.

### 1. Object cutouts

- Object cutouts into the cache (the user's data), then the run on it: R6's config (the user
  picks R5's LR or R6's).
- The file that follows replaces the preview on the Hub under a name the user picks; its ship
  read needs a prompt set no arm has been read on.

### 2. The edge band and the blocks (proposed, not taken)

- First, the overfit probe of the four-channel form on the 16 `new_bg` images, read by band
  against R1's column above (`README.md`, Rules: the overfit probe comes before any argument
  about LR or budget). Under R1's numbers on transition → the form (the 8 × 8 output, the shared
  trunk) is the limit and the overlapping output is the arm; at R1's → budget and data are, and
  the arm is a longer run. Not written as a config; cost not measured (R1's arm ran 375 updates
  an image on 16 images).
- A GAN term on α waits on that read.

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

Ship set, seeds 7, 8, read for R6's ship (2026-11-24); the next ship read takes a new set:

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
