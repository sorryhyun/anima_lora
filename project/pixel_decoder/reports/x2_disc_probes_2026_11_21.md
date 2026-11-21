# ×2 discriminator probes: `x2dprobe_1121_1015` and the band probe `x2dband_1121_1125` (2026-11-21)

What the PE-Spatial discriminator separates the ×2 output (`x2b4k_1121_0950`) from the real 2×
image on: noise arms, `chain` and `null` in the first probe, low / high band swaps and
`real_downup` in the second. Moved from `x2_plan.md` on 2026-11-21.

## Design: discriminator probe, `x2dprobe_1121_1015` (queued behind `x2b4k_1121_0950`)

### Facts

- Code: `bench/pixel_readout/x2_dprobe.py`, entered by `x2_head.py --dprobe N`. The head is
  frozen (`--resume_x2 x2b4k_1121_0950`, `--rollouts 0`). 32 train images (seeded sample of
  the `x2flex` train rows, `down` and `exact`) are rendered whole at the base point; one
  fresh `x2_pegan.PEDisc` per arm (the B+GAN runs' settings: taps after blocks 4 / 8 / 12,
  512-px paired crops, one crop per step, LR 1e-4, R1 / R2 0.1 at σ 0.01) trains 1,600
  steps on them and is read on every non-overlapping 512-px crop of the `test` 12 /
  `text` 5 images.
- Arms, by what stands on the fake side against the real 2× crop: `head` (the ×2 output),
  `chain` (ResShift ×2 of the ×1 release head's output), `null` (the real 2× image of
  another picture), `head_n0.03` and `head_n0.1` (`head` with Gaussian noise of that σ,
  [-1, 1] units, added to both crops in training and in the read).
- Read: held-out margin `D(real) − D(fake)` per arm and per tap, over all PE tokens and
  over the tokens whose 16-px cell is flat (range of the σ 2-px blurred luma under 4
  levels) or holds an edge (at least 24) in the real image; the training margin by row
  kind; the std of the luma's high-pass residual on flat cells for real / head / chain;
  margin maps on four held-out crops per arm. Bars: none; `null` is the meter's floor.
- Records it rests on: the discriminator margin 7 – 8 in every B+GAN run (`x2_arm_b_2026_11_20.md`).
  Claude's read, labels visible, one sheet (`~/x2pair_1120/release_b1k/…test_00…`
  (local)): the original carries grain over the whole frame, both heads' outputs none.
- CPU check of the code on synthetic images (a smooth image against itself plus σ 0.02
  noise, 8 steps): `head` held margin 2.24, `null` −0.11, `head_n0.1` 0.02.
- Cost: about 12 minutes.

### Interpretation

- Question: what does the discriminator separate the ×2 output from the real image on?
  If the noise arms take most of the margin away and flat cells read as high as edge
  cells, the cue is grain the head cannot produce from its features, and the GAN arm
  takes the discriminator with the noise level that leaves an edge margin. If the margin
  stays on edge cells under noise, the cue is the lines and the GAN arm keeps the
  discriminator and changes only its weight.
- `chain` against `head`: if the chain reads as far from the real image as the head, a
  discriminator on the real masters does not ask for what the user reads as clean in the
  ResShift output.
- The flat / edge split is weak on the deep taps (a PE token past block 8 sees the whole
  crop); the noise arms carry the read.

## Outcome of `x2dprobe_1121_1015`

### Facts

- 10 minutes; results and margin maps (local)
  `bench/pixel_readout/results/20261121-1053-x2dprobe_1121_1015/`. Pool 32 train images
  (20 `down`, 12 `exact`), read on 17 held-out images (220 crops, 225,280 cells: 42.8 %
  flat, 31.9 % edge). Head = `x2b4k_1121_0950`. One seed per arm.

  | arm (fake side) | train margin, first → last 200 steps | held margin | tokens with real above | held margin flat / edge, tap 4 | train margin `down` / `exact` |
  |---|---|---|---|---|---|
  | `head` | 3.67 → 6.53 | 6.33 | 0.997 | 6.27 / 7.19 | 6.21 / 6.90 |
  | `chain` | 3.52 → 6.18 | 5.17 | 0.998 | 5.60 / 6.24 | 5.50 / 7.04 |
  | `null` | 0.01 → 0.03 | 0.05 | 0.536 | 0.03 / −0.03 | 0.01 / 0.03 |
  | `head_n0.03` | 4.13 → 8.22 | 7.97 | 0.998 | 8.30 / 9.58 | 7.81 / 8.82 |
  | `head_n0.1` | 3.68 → 8.38 | 8.30 | 0.988 | 8.04 / 9.87 | 7.99 / 9.18 |

- Std of the luma's high-pass residual on flat cells, 8-bit levels: real 0.87, head 0.48,
  chain 0.32.
- Claude's read of two margin maps, labels visible (`dprobe_head_test_13340817.png`,
  `dprobe_head_n0.1_test_10808923.png`): the margin is near-uniform over the crop at
  every tap, blank white background included.

### Interpretation

- The grain reading is rejected: noise 4 to 15 times the real images' flat-cell residual
  on both crops leaves the discriminator separating 99 % of tokens. The cue survives
  noise of 13 levels, so it is not a low-amplitude texture.
- The margin's size is not comparable between the noise arms and the others: under noise
  the R1 / R2 term (a σ 0.01 perturbation) costs the discriminator less.
- The chain separates as the head does, so this discriminator does not place the ResShift
  output near the real masters either.
- The per-token logits act as one decision per crop: a PE token after block 4 already
  mixes the whole 512-px crop, so the flat / edge split and the maps do not localise the
  cue, and the generator's gradient from it is not local to lines.
- What the cue is stays open between the head's tone (dark pixels lifted 6 – 14 levels,
  shared with the chain) and its lines; the band probe below separates them.

## Design: band probe, `x2dband_1121_1125`

### Facts

- `x2dprobe_1121_1015`'s command with `--dprobe_arms band`, nothing else changed (same
  pool, head, discriminator settings and seeds). The fake side is built from the real
  image and the head's output, split by a Gaussian of σ px at the 2× scale:
  `low<σ>real_high<σ>head` (the head's output with the real image's low band),
  `low<σ>head_high<σ>real` (the real image with the head's low band), σ 8 and 2, and
  `real_downup` (the real crop bicubic ×½ then ×2).
- Read: held margin and the share of tokens with real above, against `head` 6.33 / 0.997
  and `null` 0.05 / 0.536 of `x2dprobe_1121_1015`. Bars: none.
- Cost: about 10 minutes.

### Interpretation

- Question: is the head separated from the real image by its low band (tone, colour) or
  its high band (lines, texture)? `low8head_high8real` near `null` and
  `low8real_high8head` near `head` → the lines; the GAN term then asks for what the user
  asks for, and the arm changes the weight and the discriminator's locality. The reverse
  → tone; the discriminator then needs the low band taken out of its input before a GAN
  arm says anything about lines.
- `real_downup` reads how separable a 1× image's content is at all, with no model in it.

## Outcome of `x2dband_1121_1125`

### Facts

- 10.5 minutes; results (local) `bench/pixel_readout/results/20261121-1107-x2dband_1121_1125/`.
  Same pool, held-out crops, head and seeds as `x2dprobe_1121_1015`.

  | arm (fake side) | held margin | tokens with real above | train margin `down` / `exact` |
  |---|---|---|---|
  | `head` (`x2dprobe_1121_1015`) | 6.33 | 0.997 | 6.21 / 6.90 |
  | `low8real_high8head` | 6.40 | 0.996 | 6.18 / 6.94 |
  | `low2real_high2head` | 5.99 | 0.996 | 5.95 / 6.85 |
  | `low2head_high2real` | 4.52 | 0.996 | 4.57 / 5.22 |
  | `low8head_high8real` | 2.51 | 0.997 | 2.88 / 2.60 |
  | `real_downup` | 6.64 | 0.995 | 6.60 / 7.20 |
  | `null` (`x2dprobe_1121_1015`) | 0.05 | 0.536 | 0.01 / 0.03 |

### Interpretation

- The margin follows the head's high band: with the real image's tone under it the head
  separates as before (6.40 against 6.33), and the band finer than σ 2 px carries most of
  it (5.99). The real image under the head's tone separates at well under half the margin
  (2.51), though still on 99.7 % of tokens: tone is a second, weaker cue.
- A real image taken down to 1× and back separates as the head does (6.64). What the
  discriminator reads is what a 1× image lacks at 2×, which is the sharpness the user
  asks for; it is the right teacher for the lines.
- With the first probe: the earlier B+GAN runs did not fail on a wrong cue. What is left
  is the term's weight and length (the largest, `x2bganx_1120_1730`, ran 250 rollouts on a
  head that was still 3,750 rollouts from where B alone ends) and the crop-level decision.
