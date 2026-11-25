# ×2 output — plan

What is waiting or ahead on the ×2 head. What was found: `README.md`, Findings, "×2 training".
The records: `reports/README.md`. Tidied on 2026-11-24 (the user: "x2 plan이 좀 너저분하니까
한번 쭉 정리하면서 돌아보자"); the 2026-11-22 sections went to
[reports/x2_stamp_crop_2026_11_22.md](reports/x2_stamp_crop_2026_11_22.md), the comparison
with the reference code to [reports/x2_slope_2026_11_24.md](reports/x2_slope_2026_11_24.md).

## Where the head stands

- The form is the overlapping output (16 × 16 patches at stride 8 under a window), the B head
  trained with it and the GAN finish on top (the user, 2026-11-23: "그럴거면 x2 자체를 stride
  버전으로 하고, 거기에 gan을 얹는게 맞나?"). B head: `x2ovll_1124_0139`, 12,000 rollouts.
- The GAN finishes of that head, on the 32 renders (amp = high-pass share of the chain's; the
  tile column is `tile_dict.py`'s departed share on castle s0; held out against the real 2×
  image, test / text):

  | head | what | amp, colour / mono | flat-token grain | sky patch, 8-px fold | tile, castle s0 | held out, dB |
  |---|---|---|---|---|---|---|
  | `x2ovll_1124_0139` | B head, no GAN | 0.603 / 0.620 | 0.23 | 0.09 | 0.002 | 30.45 / 26.11 |
  | `x2ovgf_1124_0920` | ratio 2.0, 4,000 rollouts | 0.771 / 0.698 | 0.39 | 0.32 | 0.054 | 29.69 / 25.62 |
  | `x2r05gf_1124_1941` | ratio 0.5, 4,000 | 0.710 / 0.646 | 0.29 | 0.11 | 0.003 | 30.20 / 25.94 |
  | `x2r10gf_1125_0104` | ratio 1.0, 4,000 | 0.739 / 0.673 | 0.33 | 0.16 | 0.013 | 29.95 / 25.77 |
  | `x2slope_1124_2055` | ratio 2.0 + slope term, 1,000 | 0.782 / 0.733 | 0.36 | 0.19 | 0.093 | 29.63 / 25.35 |
  | `x2slr05_1124_2316` | ratio 0.5 + slope term, 1,000 | 0.754 / 0.718 | 0.33 | 0.14 | 0.066 | 29.81 / 25.49 |
  | `x2slcap_1125_0011` | the same, the weight's cap lifted | 0.754 / 0.716 | 0.32 | 0.14 | 0.072 | 29.82 / 25.50 |
  | `x2r200_1125_0036` | the same, slope term at 200 | 0.702 / 0.680 | 0.31 | 0.14 | 0.028 | 30.01 / 25.69 |
  | `x2r05_1124_1908` | ratio 0.5, no slope term, 1,000 | 0.687 / 0.659 | 0.29 | 0.15 | 0.003 | 30.20 / 25.89 |

- No ship candidate. `x2r05gf_1124_1941` has the best balance so far and is not enough to ship
  (the user's read, below). The Hub ×2 file stays.
- The user's reads so far:
  - of `x2ovgf_1124_0920`, two regions ([record](reports/x2_ovgf_mix_2026_11_24.md)): the
    sharp line is the identity of the upscale; the lines are distorted (in `x2ovll` too,
    sharper here) and the flat blue carries a slight pattern.
  - of the PE-GAN term (2026-11-24): "선명함을 확실히 주는게 꾸준히 보였으니 ㅇㅋ". It stays.
  - of `x2slope_1124_2055`: "가능성은 있어 보이는데 03 s0 의 경우는 좀...".
  - of `x2r05gf_1124_1941` (2026-11-24): "r05gf가 제일 무난하긴 했지만 배포후보로까진 아쉬워 그래도 여태까지중에 제일 밸런스는 좋아".
  - of `x2r05gf_1124_1941` | `x2r10gf_1125_0104` (2026-11-25): "사실상 둘이 동률이다".
- Three patterns are open, each with its own source (README, "×2 training", the last items):
  1. The token tile on halftone and the breaks in lines: the B head's. No GAN setting removed
     them.
  2. The 8-px grid on flat colour: the GAN finish's. Lower at ratio 0.5 and with the slope
     term.
  3. The token tile on fine colour texture (castle, neon_rain): the GAN finish's. Near zero at
     ratio 0.5, higher with the slope term at either ratio.

## Waiting on the user's read

Labelled pair sheets of the 32 renders, local, under `bench/pixel_readout/results/`:

| pair | folder | what it decides |
|---|---|---|
| `x2slcap` \| `x2r200` | `20261125-0056-tex_x2r200_1125_0036/pairs_x2slcap_1125_0011_x2r200_1125_0036/` | slope weight 2500 against 200 at ratio 0.5 |
| `x2slope` \| `x2slr05` | `20261124-2336-tex_x2slr05_1124_2316/pairs_x2slope_1124_2055_x2slr05_1124_2316/` | ratio 2.0 against 0.5 under the slope term |

The pair sheets of the closed arms (unpaired, flat-cell cut, the LoRA finishes, the 1,000-rollout
ratio and slope arms) are named in their records.

## Next

1. Nothing queued; the next step is the user's call. `x2r10gf_1125_0104` (`x2r05gf_1124_1941`
   at ratio 1.0, the user's pick; [record](reports/x2_ratio_2026_11_24.md)) read half of the
   amp between ratio 0.5 and 2.0 (0.739 / 0.673) with a fifth to a quarter of the castle tile
   and the flat-colour fold; the user's read of the pair: "사실상 둘이 동률이다".
2. The settings of this GAN term are read through: ratio (2.0, 0.5), slope
   weight (0.1, 200, 2500), the weight's cap, the pairing and the tower each move along one
   line where more amp comes with more of a token-locked pattern
   ([record](reports/x2_slope_2026_11_24.md), the last section). `x2r200_1125_0036` (the
   user's value, "ㅇㅋ 200으로 가자") read between the two ends: castle s0 0.028, amp 0.702 /
   0.680, the sky patch's fold 0.14, margin 1.28. The next step is the user's call; what
   Claude would propose is in the chat record of 2026-11-25 and is not a plan item until
   the user takes one.

## Parked, no step proposed

- The DINOv3 discriminator (`DinoDisc`, `pegan_tower = "dino"`): closed by the user
  (2026-11-25: "아냐 dino는 그냥 취소하자 그냥 안좋아") after `x2dino_1124_2348` (margin 0.012,
  the weight cut at a twentieth of the ratio, the head where its parent is) and 102 rollouts
  of `x2dinocap_1125_0007` ([record](reports/x2_slope_2026_11_24.md)).

- The reference's cap on the adversarial gradient, per sample at the RGB output: the other
  difference from the reference not tried. After Next 1.
- An α under 1 of a finish as a sharpness setting: `head_merge.py --alphas <α>` writes the file
  in seconds ([record](reports/x2_ovgf_mix_2026_11_24.md)). A strength setting only; no α
  separates sharpness from the pattern.
- The noise inputs with a nonzero init and their own LR: they were waiting on an arm that moves
  the discriminator's margin. The slope term does (0.17 – 0.19 against 7); not proposed while
  the tile on fine texture is open.
- The GAN finish as a LoRA (`x2_idea.md`): rank 8 at two LRs ran; rank 64 and the factor reads
  of its steps 2 – 4 are not proposed. The finish stays a full finetune.
- A Jacobian lens on the second stage, to fold pattern directions out of the head without
  training (`x2_idea.md`, the last section): written down on the user's word (2026-11-25);
  nothing run.
- A convolution tail at pixel resolution: the reference decoder has none; not proposed.
- A high-pass penalty on the pixels whose target is flat: the user read that direction as
  "애매"; dropped.
