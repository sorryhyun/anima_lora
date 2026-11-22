# RGBA refine arm — a full-resolution refine stage on the alpha head (2026-11-22)

`rgbarf_1122_2110` against its parent `rgbar2_1122_1550`, both in `../configs/`; the plan:
`../rgba_plan.md`. The refine code was reverted after this read; the config no longer parses.

## Facts

### Why it ran

- The user, on R2's renders (2026-11-22): "반투명이고 나발이고가 아니라 아예 블록처럼 쪼개지는게
  문제" (`rgba_plan.md` Priority).
- On R2's 8 renders (`output/pixel_head/rgbar2_1122_1550/renders/` (local)), as saved 8-bit PNGs:
  flat mid 8 × 8 tiles (mean alpha in (0.1, 0.9), std < 0.05) oldman_s1 227, bicycle_s1 191,
  bicycle_s0 120, oldman_s0 50, maid 0 – 8, glass 0; mid alpha (0.02 – 0.98) more than 2 px from
  the 0.5 contour, oldman_s1 4.79 %, bicycle_s1 4.83 % of the frame.
- Thresholding R2's alpha at 0.5 at decode, and a guided filter on it with the RGB as guide, on
  oldman_s1 and bicycle_s1 (Claude's read, labelled): the patches stay, as hard-edged holes and
  opaque white squares under the threshold.
- The user (2026-11-22), after the per-cell output layer was explained: "안해볼 이유가 없을듯 refine
  conv 붙이고 1k 돌려보자".

### The arm

- `out` wrote 1 + 8 channels per pixel; four full-resolution 3 × 3 convs (width 32, dilation 1 / 2
  / 4 / 1, 17 px receptive field, the last at zero) added a correction to the alpha. No RGB input.
  Nothing else changed from `rgbar2_1122_1550`.
- Training 1,020 s against 628 s; 0.89 s a rollout against 0.55.

### Held-out scores at 1,000 rollouts

Alpha L1 per region, per-image mean (the run's `result.json`, 96 read rows):

| region | new_bg, rgbar2 | new_bg, refine | other, rgbar2 | other, refine |
|---|---|---|---|---|
| all | 0.0151 | 0.0150 | 0.0254 | 0.0254 |
| transition | 0.1744 | 0.1687 | 0.2104 | 0.2062 |
| translucent | 0.1830 | 0.1751 | 0.2608 | 0.2543 |
| thin | 0.1363 | 0.1359 | 0.1778 | 0.1764 |
| holes | 0.0461 | 0.0415 | 0.1209 | 0.1174 |
| white foreground | 0.0128 | 0.0139 | 0.0432 | 0.0460 |
| background-like foreground | 0.0142 | 0.0152 | 0.0400 | 0.0428 |

- The eval line at 250 and 500 rollouts was within 0.002 of R2's on every region.
- Spread unknown: no seed twin.

### Renders

- The same 8 renders (maid, oldman, glass, bicycle × seeds 0, 1; base 28 steps CFG 4) through
  each head, one script (`render_alpha.py`, Claude's scratchpad (local)); alpha before 8-bit
  rounding. Outputs (local): `output/pixel_head/rgbar2_1122_1550/renders_v2/`,
  `output/pixel_head/rgbarf_1122_2110/renders/`, each with `blocks.json`.

  | render | flat mid tiles, rgbar2 | refine | interior mid %, rgbar2 | refine |
  |---|---|---|---|---|
  | maid_s0 | 8 | 14 | 0.93 | 1.06 |
  | maid_s1 | 0 | 1 | 0.56 | 0.57 |
  | oldman_s0 | 51 | 36 | 1.70 | 1.92 |
  | oldman_s1 | 227 | 261 | 5.04 | 6.22 |
  | glass_s0 | 0 | 0 | 0.43 | 0.51 |
  | glass_s1 | 0 | 0 | 0.30 | 0.31 |
  | bicycle_s0 | 121 | 99 | 4.25 | 4.45 |
  | bicycle_s1 | 191 | 162 | 4.98 | 5.37 |
  | total | 598 | 573 | | |

- Claude's read (labelled; robe of oldman s0 / s1, wheels of bicycle s1, composite and alpha map):
  with refine the patches sit in the same places, their square edges softer.
- Claude's read (labelled; the 8 read sheets GT | rgbar2 | refine, generated maid s0 / s1 and
  glass): generated characters and glass alike in both. Held out, both heads alike: holes in the
  white dress on a light flat background (`other_white_fg`) in the same places, the translucent
  skirt and float opaque, 8-px steps on the skirt's ribbon edge, thin purple lines thickened; the
  birdcage's lower bars a little cleaner with refine.
- The user (2026-11-22): "그럼 refine 쪽 자체 코드 변경을 그냥 revert하자 rgba에는 좀 더 길게 돌릴
  계획으로". Code reverted the same day.

## Interpretation

- The patches are the cells' decision in white-on-white regions, not the seams between cells:
  letting pixels across a seam see each other moved neither where they are nor how many.
- The held-out `other_white_fg` dress shows the same failure as the generated robe, and it has GT:
  the white-foreground and background-like-foreground L1 measure it on real labels, where the
  render tile count has none.
- What is left is length or data: R3 with checkpoints, read on those regions and on the render
  tile count, then white-on-white cutouts if the count does not fall.
