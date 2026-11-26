# ×2 output — plan (parked 2026-11-26)

Parked on the user's word (2026-11-26: "당분간은 rgba가 crisp하게 혹은 x1이 좀 더 넓은 0.75 - 0
범위를 커버하도록 학습하는 데 집중하려고 해"). Nothing here is queued or proposed; this file is
where the head stands and what a return would start from. What was found: `README.md`, Findings,
"×2 training". Records: `reports/README.md`; the finished ones, the finished runs' configs and the
weight-space / LoRA / lens proposal (`x2_idea_2026_11_24.md`) are in `archive/` (local).

## Where the head stands

- The form is the overlapping output (16 × 16 patches at stride 8 under a window), the B head
  trained with it and the GAN finish on top (the user, 2026-11-23: "그럴거면 x2 자체를 stride
  버전으로 하고, 거기에 gan을 얹는게 맞나?"). B head: `x2ovll_1124_0139`, 12,000 rollouts.
- On the Hub: `x2/anima_pixel_head_x2.safetensors` (`x2bgan2_1121_1340`, the catalog's ×2 row,
  the node README's and the card's samples) and, since 2026-11-26,
  `x2/anima_pixel_head_x2_v3.safetensors`: `x2r10gf_1125_0104` with the edit `f8_k4` (the 8-px
  grid on flat colour folded out of `out2`, no training;
  [record](reports/x2_lens_2026_11_26.md)): amp 0.739 / 0.673, flat-token grain 0.31, the sky
  patch's 8-px fold 0.10, castle s0 0.014. Whether v3 becomes the catalog's default is the
  user's call, not asked yet.
- The GAN finishes of the B head, on the 32 renders (amp = high-pass share of the chain's; the
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

  Records: [ratio](reports/x2_ratio_2026_11_24.md), [slope term](reports/x2_slope_2026_11_24.md).
- The user's reads:
  - of `x2ovgf_1124_0920`, two regions (`archive/x2_ovgf_mix_2026_11_24.md`, local): the
    sharp line is the identity of the upscale; the lines are distorted (in `x2ovll` too,
    sharper here) and the flat blue carries a slight pattern.
  - of the PE-GAN term (2026-11-24): "선명함을 확실히 주는게 꾸준히 보였으니 ㅇㅋ". It stays.
  - of `x2slope_1124_2055`: "가능성은 있어 보이는데 03 s0 의 경우는 좀...".
  - of `x2r05gf_1124_1941` (2026-11-24): "r05gf가 제일 무난하긴 했지만 배포후보로까진 아쉬워 그래도 여태까지중에 제일 밸런스는 좋아".
  - of `x2r05gf_1124_1941` | `x2r10gf_1125_0104` (2026-11-25): "사실상 둘이 동률이다".
  - of `x2r05gf_1124_1941` (2026-11-26): "선명도가 약간 더 아쉽긴 했어".
  - of `x2r05gf_1124_1941` | `x2r10gf_1125_0104` + `f8_k4` and | `x2ovgf_1124_0920` + `f8_k4`
    (2026-11-26): "ㅇㅋ 그래도 ratio 1.0 + 편집이 manga s0을 볼때 더 낫다 이걸로 v3 업로드하자".
  - of the DINOv3 discriminator (2026-11-25): "아냐 dino는 그냥 취소하자 그냥 안좋아". Closed.
- Three patterns are open, each with its own source (`README.md`, "×2 training"):
  1. The token tile on halftone and the breaks in lines: the B head's. No GAN setting removed
     them; the B head's held-out numbers are flat from rollout 11,000.
  2. The 8-px grid on flat colour: the GAN finish's. Lower at ratio 0.5 and with the slope
     term; `f8_k4` folds its 8-px part out.
  3. The token tile on fine colour texture (castle, neon_rain): the GAN finish's. Near zero at
     ratio 0.5, higher with the ratio and with the slope term; no `out2` edit reaches it.
- The settings of this GAN term are read through: ratio (0.5, 1.0, 2.0), slope weight (0.1, 200,
  2500), the weight's cap, the pairing and the tower each move along one line where more amp
  comes with more of a token-locked pattern ([record](reports/x2_slope_2026_11_24.md), the last
  section).

## Pair sheets the user has not read

Labelled, of the 32 renders, local, under `bench/pixel_readout/results/`:

| pair | folder | what it compares |
|---|---|---|
| `x2slcap` \| `x2r200` | `20261125-0056-tex_x2r200_1125_0036/pairs_x2slcap_1125_0011_x2r200_1125_0036/` | slope weight 2500 against 200 at ratio 0.5 |
| `x2slope` \| `x2slr05` | `20261124-2336-tex_x2slr05_1124_2316/pairs_x2slope_1124_2055_x2slr05_1124_2316/` | ratio 2.0 against 0.5 under the slope term |

## Local files

Tidied on the user's word (2026-11-26: "output/pixel_head 랑 bench/pixel_readout/results/ 도
정리하자"). All local.

- Heads kept under `output/pixel_head/`: `x2ovll_1124_0139` (B head), `x2r10gf_1125_0104` and
  `x2f8_1126_2227/` (the v3 file and `x2ovgf` + `f8_k4`), `x2r05gf_1124_1941`, `x2ovgf_1124_0920`,
  and the 1,000-rollout arms of the unread pair sheets with their control (`x2slope_1124_2055`,
  `x2slr05_1124_2316`, `x2slcap_1125_0011`, `x2r200_1125_0036`, `x2gf1k_1124_1256`). The 32
  cached renders are under `output/pixel_head/render_cache/`: a kept head is read on them again
  with no DiT run.
- Deleted there: `x2dll_1123_1537`, `x2ov_1123_2012`, `x2nz_1123_1920`, `x2unp_1124_1256`,
  `x2flat_1124_1341`, `x2r05_1124_1908`, `x2lora_1124_1113`, `x2lora_1124_1151`,
  `x2dino_1124_2348`, and the mixes `x2mg_1124_1025` / `x2rk_1124_1100` (`head_merge.py` writes
  them again from `x2ovll` and `x2ovgf`).
- Under `bench/pixel_readout/results/`: the renders, sheets and reads of v3 (`*x2f8*`,
  `20261126-2217-x2lens_1126_2225/`) and the two unread pair-sheet folders above are kept. The
  render, texture and tile folders of `x2ovll`, `x2r05`, `x2r05gf`, `x2r10gf`, `x2slope`,
  `x2slr05`, `x2slcap`, `x2r200` and `x2dino` were deleted; a path to one in a report now holds
  only its `result.json`, under `bench/pixel_readout/results/archive/<the same folder name>/`.
  The deleted heads' `result.json` / `merge.json`: `…/archive/output-<run>/`.

## Levers not taken

A return to ×2 starts from the user's pick among these or a new one; none is proposed.

- The reference code's cap on the adversarial gradient, per sample at the RGB output: the one
  difference from the reference not tried ([record](reports/x2_slope_2026_11_24.md)).
- An α under 1 of a finish as a sharpness setting: `head_merge.py --alphas <α>` writes the file
  in seconds. A strength setting only; no α separates sharpness from the pattern.
- `f8_k4` on another finish: `lens_read.py --heads <run> --p32_sets --f8_k 4 --sheets
  --write_heads f8_k4 --write_only` (CPU, 2 min).
- The noise inputs with a nonzero init and their own LR: they waited on an arm that moves the
  discriminator's margin; the slope term does (0.17 – 0.19 against 7).
- The GAN finish as a LoRA: rank 8 at two LRs ran; rank 64 and the factor reads (steps 2 – 7 of
  the proposal) did not. The finish stays a full finetune.
- A convolution tail at pixel resolution: the reference decoder has none.
- A high-pass penalty on the pixels whose target is flat: the user read that direction as
  "애매"; dropped.
