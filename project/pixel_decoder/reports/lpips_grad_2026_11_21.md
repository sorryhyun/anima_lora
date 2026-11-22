# LPIPS gradient into v: the pixel heads against the VAE decode (2026-11-21)

`lgrad_1121_1845`: step 1 of `idea.md`, a forward + backward probe of whether a pixel head can be
the differentiable decode of an LPIPS term on a LoRA. Design and the reading rule: the last
section of this file.

## Facts

- The user (2026-11-21): "let's go ahead with item 1 and if any, proceed to 2" (item 1 = this
  probe, item 2 = a head over a wider σ range).
- `bench/pixel_readout/lpips_grad_probe.py --label lgrad_1121_1845`: 16 of P1's held-out test
  images, base DiT on the fp32 stream, teacher-forced, two noise draws per (image, σ), one window
  of 32 × 32 tokens per image (24 + a 4-token margin, LPIPS-VGG on the inner 384² px), target
  VAE(x0) from the fp32 VAE, every decode clamped to [-1, 1], head and LPIPS in fp32. One seed for
  images and windows. Results (local) `bench/pixel_readout/results/20261121-1842-lgrad_1121_1845/`.
- Arms. a: ∂LPIPS/∂v through the fp32 VAE decode of the window. a16: the same through the bf16
  VAE. b: a head's gradient as v receives it, (W Wᵀ)⁻¹ W ∂L/∂f (W = `final_layer.linear.weight`,
  64 × 2048, cond(W Wᵀ) 2.9). Heads `mixr_1119_1455` (target VAE(x̂0)) and `realr_1119_2200`
  (original image + LPIPS at the base points); both trained on base σ 0.0577 … 0.1364.
- Step check: v − η·ĝ along each unit gradient, η set so that a's first-order LPIPS decrease is
  1 % of the loss, the loss re-read through the fp32 VAE. "share" = an arm's decrease over a's;
  "null" = the same for a head's gradient taken from the other noise draw. a's own
  actual / predicted decrease: 1.02 … 0.81 at 0.1 %, 0.77 … 0.48 at 1 %, 0.50 … 0.12 at 10 %
  (σ 0.0577 … 0.7). The 0.1 % shares are not read (the mean at σ 0.1 is −4.1 for a16).
- Means over 32 (image, draw) rows per σ. `floor` = cos(b on one draw, a on the other).

  | σ | 0.0577 | 0.1 | 0.1364 | 0.3 | 0.5 | 0.7 |
  |---|---|---|---|---|---|---|
  | LPIPS of VAE(x̂0) against VAE(x0) | 0.041 | 0.064 | 0.082 | 0.158 | 0.255 | 0.366 |
  | cos(a16, a) | 0.511 | 0.495 | 0.434 | 0.343 | 0.174 | 0.115 |
  | cos(a, a) across the two draws | 0.154 | 0.126 | 0.104 | 0.053 | 0.025 | −0.003 |
  | cos(a, FM gradient) | 0.087 | 0.076 | 0.068 | 0.041 | 0.017 | 0.008 |
  | `mixr` cos(b, a) | 0.115 | 0.154 | 0.166 | 0.103 | 0.052 | 0.009 |
  | `mixr` floor | 0.060 | 0.078 | 0.081 | 0.049 | 0.020 | 0.005 |
  | `realr` cos(b, a) | 0.138 | 0.159 | 0.164 | 0.100 | 0.053 | 0.010 |
  | `realr` floor | 0.067 | 0.075 | 0.079 | 0.043 | 0.020 | 0.003 |
  | a16 share, 1 % step | 0.686 | 0.662 | 0.639 | 0.486 | 0.338 | 0.256 |
  | `mixr` share / null, 1 % step | 0.136 / 0.073 | 0.197 / 0.110 | 0.225 / 0.099 | 0.156 / 0.070 | 0.085 / 0.007 | 0.013 / 0.014 |
  | `realr` share / null, 1 % step | 0.160 / 0.086 | 0.210 / 0.115 | 0.210 / 0.089 | 0.145 / 0.048 | 0.080 / 0.017 | 0.027 / 0.001 |
  | a16 share, 10 % step | 0.884 | 0.863 | 0.907 | 0.759 | 0.705 | 0.561 |
  | `mixr` share / null, 10 % step | 0.210 / 0.107 | 0.329 / 0.160 | 0.414 / 0.193 | 0.341 / 0.157 | 0.302 / 0.114 | 0.087 / 0.047 |
  | `mixr` row-space share of ∂L/∂f | 0.032 | 0.034 | 0.038 | 0.060 | 0.062 | 0.067 |
  | `mixr` ‖b‖ / ‖a‖ | 1.57 | 0.82 | 0.58 | 0.28 | 0.12 | 0.10 |
  | `mixr` PSNR against VAE(x̂0), dB | 37.58 | 38.00 | 38.02 | 35.87 | 35.74 | 30.95 |

  64 of 2048 dimensions is 0.031. `mixr`'s 1 % share is above its null on 26 / 31 / 31 of 32 rows at
  the three in-range σ. cos(c, Wᵀ g_a), the raw head gradient in f space: 0.021 … 0.032 in range.
- Cost of one backward on the 512² window (median seconds, peak GB over the 4.45 GB of models on
  the card): fp32 VAE 0.154 s / 4.33, bf16 VAE 0.087 s / 2.46, a head 0.019 s / 0.70.
- Claude's reading rule, fixed before the result (`plan.md`): "yes" if `mixr`'s cos(b, a) at the
  three in-range σ sits nearer cos(a16, a) than the floor. Result: 0.115 / 0.154 / 0.166 against
  floors 0.060 / 0.078 / 0.081 and a16 0.511 / 0.495 / 0.434: nearer the floor at each. Not passed.
- Not run: a spread over window or image seeds, the training images, a head trained on wider σ, a
  bf16 VAE with its own bf16 target.

## Interpretation

- The heads read the image from the whole feature, not from v: the share of their gradient that
  lies in the 64 directions setting v equals the share those directions have of the space. Under
  the row-space projection about 3 % of the gradient energy is left, and its direction buys a
  fifth of what the VAE gradient buys per unit step, half of that surviving a change of noise
  draw (an image-generic part). Without the projection a LoRA can lower the loss in directions
  that leave the latent unchanged.
- The two heads read alike, so the target a head was trained on (VAE(x̂0) or the original) is not
  what decides this; the input is. A head that reads f was never asked how the image moves when v
  alone moves.
- Widening the σ range of this head form does not address it. The values already hold outside the
  trained range (35.7 dB at σ 0.5 on the base DiT) while the gradient agreement falls with σ.
- The windowed VAE decode is affordable as the decode of an LPIPS term at any σ: 0.09 s and
  2.5 GB in bf16 per 512² window. Its bf16 gradient is itself only at cos 0.5 with the fp32 one
  and still buys 64 – 69 % of the decrease at a 1 % step, 86 – 91 % at 10 %: the fp32 gradient
  carries high-curvature components that a finite step does not collect, so the cosine understates
  a decoder and the step check is the meter to carry forward.
- The LPIPS gradient is nearly orthogonal to the FM gradient (cos under 0.09) and differs between
  noise draws of one image (cos 0.15 at most), so whatever it adds to a LoRA is a direction FM
  does not supply, arriving as a noisy per-draw signal.
- The same reading applies to cjk_rl arm D's "reader NLL through the head": its D0 cost probe
  should take the windowed VAE path, or a decoder whose only inputs are x_t and v.

## Design as written in `plan.md` (moved 2026-11-22)

Step 1, the gradient-agreement probe (the user, 2026-11-21: "let's go ahead with item 1 and if
any, proceed to 2"; item 2 = a head over a wider σ range).

Facts (the design, set before the run):

- `bench/pixel_readout/lpips_grad_probe.py --label lgrad_1121_1845`: forward + backward, no step.
  16 of P1's held-out test images, base DiT on the fp32 stream, teacher-forced at σ 0.0577 / 0.1 /
  0.1364 (the heads' range) and 0.3 (outside it), two noise draws each, one 24-token window with a
  4-token margin per image, target VAE(x0), both decodes clamped.
- Arms: a = ∂LPIPS/∂v through the fp32 VAE decode of the window; a16 = the same through the bf16
  VAE; b = a head's gradient as v receives it under the row-space projection,
  (W Wᵀ)⁻¹ W ∂L/∂f; c = the head's raw ∂L/∂f against Wᵀ g_a. Heads: `mixr_1119_1455` (target
  VAE(x̂0)) and `realr_1119_2200` (target the original image + LPIPS at the base points).
- Read: cos(b, a) per head and σ, beside its two in-run references: cos(a16, a) and the floor
  cos(b on one draw, a on the other); cos(a, a) across the draws is what a perfect decoder reads on
  that floor. Also the norm ratio, the row-space share of c, PSNR of the head against VAE(x̂0), and
  seconds and peak GB of each backward. Added after the first smoke, before the full run: σ 0.5
  and 0.7, and a step check (each gradient's LPIPS decrease through the fp32 VAE at three step
  sizes, with the head's gradient from the other draw as its null).
- Bars: spread unknown, no earlier run on this quantity.
- Cost: minutes.
- Result (`reports/lpips_grad_2026_11_21.md`): `mixr` cos(b, a) 0.115 / 0.154 / 0.166 at σ 0.0577 /
  0.1 / 0.1364, floor 0.060 / 0.078 / 0.081, cos(a16, a) 0.511 / 0.495 / 0.434; `realr` level with
  it; row-space share of the head's gradient 0.032 – 0.038. By the reading rule below: not passed.
  The wider-σ head (item 2) was not trained.
- One 512² window backward: fp32 VAE 0.154 s / 4.33 GB, bf16 VAE 0.087 s / 2.46 GB, head 0.019 s /
  0.70 GB.

Interpretation (the reasoning for the arm):

- Question: does a head send LPIPS into v the way the VAE decode does? Yes → the head is the
  decode of the LoRA arm and the wider-σ head is trained with target VAE(x̂0). No → the wider head is
  not trained for this purpose; the windowed VAE decode (its cost is in this run) or a latent-only
  decoder carries the term.
- cos(b, a) in v space is the first-order share of the VAE path's LPIPS decrease that a step along
  the head's gradient buys, so it reads without a bar. Claude's reading rule, fixed before the
  result: "yes" if `mixr`'s cos(b, a) at the three in-range σ sits nearer cos(a16, a) than the
  floor; the numbers go to the user either way.
- After the result: the decode of an LPIPS term on a LoRA is the windowed VAE (bf16), at any σ, or
  a decoder whose only inputs are x_t and v. Next arm, proposed and not run: FM against FM +
  λ·LPIPS through the bf16 VAE window on one LoRA, λ from the measured gradient-norm ratio.
