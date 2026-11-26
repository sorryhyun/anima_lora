# ×1 output — plan (opened 2026-11-26)

The ×1 head over a wider decode range, σ 0.75 – 0. One of the line's two open tracks (the other:
`rgba_plan.md`). What was found on ×1: `README.md`, Findings, "×1 training" and "What the head
loses". Records: [reports/x1_recipe_2026_11_22.md](reports/x1_recipe_2026_11_22.md),
[reports/x1_fit_2026_11_24.md](reports/x1_fit_2026_11_24.md).

## Goal

### Facts

- The user (2026-11-26): "당분간은 rgba가 crisp하게 혹은 x1이 좀 더 넓은 0.75 - 0 범위를 커버하도록
  학습하는 데 집중하려고 해".
- What the files on the Hub cover (`configs/pixel_head/base_x1.toml`,
  `points = "0.0577-0.1364:real 0.5:adapter"`): the base `x1dll_1123_0821` and the sharp
  `x1dgan512_1123_1214` were trained at σ 0.0577 – 0.1364 on the base DiT against the original
  image, and at σ 0.5 with the turbo LoRA against VAE(x̂0). The RGBA preview has the base points
  only.
- The last σ of shift-3 Euler at N steps is 3 / (N + 2): 0.75 at 2 steps, 0.6 at 3, 0.5 at 4,
  0.25 at 10, 0.1364 / 0.1 / 0.0577 at 20 / 28 / 50, 0.029 at 100.
- The head takes no σ input: it reads f (after the final layer's LayerNorm + adaLN) and x_t's
  patch.

### Not set yet (the user's)

- What is decoded above σ 0.14: the last step of a 2 – 10-step sampler (with the turbo LoRA, on
  the base DiT, or both), or a preview in the middle of sampling. This sets the DiT at those
  points (`:adapter` or not) and their target (`:real`, or VAE(x̂0) as on the turbo point).
- Whether the wide head replaces the base file or sits beside it.

## What the records say about the range

### Facts

- Below the range, on the base DiT (`x1sig_1124_1250` / `x1sig_1124_1252`, 3 images, the mean of 2
  noise draws, dB against the original; VAE(x̂0) / `x1dll`):

  | σ | 9482149 `win2` | 12692758 `win0` | 6378235, whole image |
  |---|---|---|---|
  | 0 | 27.24 / 22.93 | 34.99 / 30.43 | 37.59 / 32.22 |
  | 0.01 | 26.94 / 24.79 | 34.83 / 31.96 | 37.13 / 33.69 |
  | 0.03 | 26.01 / 25.30 | 33.58 / 32.41 | 36.28 / 35.14 |
  | 0.0577 | 24.96 / 24.94 | 31.75 / 31.45 | 35.20 / 35.04 |
  | 0.1364 | 22.96 / 23.54 | 28.63 / 28.79 | 32.76 / 32.97 |

- A trained range of σ 0.03 – 0.25 (`x1wide_1121_2230`, 3,000 rollouts, the LPIPS term) against
  the parent recipe's twin `x1twin_1121_2230`, test 30, dB / LPIPS against the original:

  | σ | `x1twin` | `x1wide` | VAE(x̂0) |
  |---|---|---|---|
  | 0.03 | 33.88 / 0.0777 | 34.00 / 0.0799 | 37.04 / 0.0412 |
  | 0.0577 | 33.78 / 0.0794 | 33.54 / 0.0837 | 35.11 / 0.0560 |
  | 0.1364 | 31.18 / 0.1038 | 31.08 / 0.1063 | 31.33 / 0.0916 |
  | 0.25 | 27.92 / 0.1531 | 28.12 / 0.1469 | 27.88 / 0.1378 |

  The `realr` / `x1twin` gap inside the range is at most 0.06 dB / 0.0013; at σ 0.03 and 0.25
  the spread is unknown. On generated renders against the VAE decode: 28 steps 31.47 against
  31.87, 10 steps (σ 0.25) 29.75 against 30.25, 100 steps 30.32 against 30.36; `realr` and
  `x1twin` differ by 0.03, 0.03 and 0.15 dB on those rows.
- The 12,000-rollout base at LR 5e-5 (`x1ll_1122_1000`) at the two σ it was not trained on, test:
  34.68 / 0.0684 at 0.03, 27.95 / 0.1481 at 0.25.
- Far above: a head trained at σ 0.0577 reads 13.73 dB on the turbo's σ 0.5 colour renders, 34.58
  after a warm-started training at σ 0.5 (`archive/p4_t2_2026_11_19.md`, local). One head over
  the base range and the turbo point: `mixr_1119_1455` (`archive/p4_mix_2026_11_19.md`, local).
- No head has been read at σ 0.6 or 0.75, or on the base DiT at σ 0.5.

### Interpretation

- The 2026-11-22 report did not take the wide range up: `x1wide` lost more than the twin gap
  inside the range and gained 0.1 – 0.2 dB off it. That was a 3,000-rollout arm up to σ 0.25; it
  does not say what a full-length run or a range up to 0.75 gives.
- Below 0.0577 VAE(x̂0) goes on to VAE(z0) and the head peaks near σ 0.03 and falls: those σ are
  outside what it was trained on. The low end is where the fine-stroke gap to the VAE sits
  (`README.md`, "What the head loses").
- Above 0.25 on the base DiT a one-step x̂0 is far from the image, so the target there decides
  what the head is asked to write; that is the first open point above.

## Next

Nothing is queued.

1. Proposed, not taken: a forward-only read of `x1dll_1123_0821` over the whole range, σ 0, 0.01,
   0.03, 0.0577, 0.1364, 0.25, 0.5, 0.6, 0.75, on the held-out test / text / manga sets, on the
   base DiT and with the turbo LoRA, against the original and VAE(x̂0). It says where the
   released head falls off and how far VAE(x̂0) itself is from the original at each σ, which is
   the ceiling a `:real` target is read against. No script does this as it is:
   `bench/pixel_readout/experiments/decode_point.py` reads named images on the base DiT, and the
   trainer reads `eval_points` inside a run. Minutes of forwards; cost not measured.
2. The training arm (points, targets, length, the file it becomes) is written after the two
   points under "Not set yet" and the read of 1. Its bar inside the old range is the
   `realr` / `x1twin` gap above, measured on the LPIPS recipe; no twin exists on the DINO recipe.
