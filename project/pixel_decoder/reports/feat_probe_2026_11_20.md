# Feature probe: the last step's feature, the step before, a mid block (2026-11-20)

`featprobe_1120_0947`: P1's ridge readout with the input changed, to answer the user's
question whether the previous step's feature helps the pixel decode.

## Facts

- The user (2026-11-20): "이전 스텝의 특징을 보면 업스케일링 픽셀디코드에 도움이 되려나?"
- `bench/pixel_readout/feat_probe.py --compile`: P1's split (256 train images, 30 test images
  from 8 held-out artists), fp32 stream, base DiT, teacher-forced on cached latents. Three
  2048-d feature groups per token: L, the final layer's input at the last σ of a 28-step
  shift-3 grid (0.1000); P, the same tap at the σ before it (0.1875), same image and same
  noise draw; M, block 18's output (of 28) at the last σ, layer-normed per token. A per-token
  ridge linear to a 16×16 RGB patch, solved in closed form per arm, ridge picked on the test
  set from 1e-6 / 1e-4 / 1e-2. One seed. Results (local)
  `bench/pixel_readout/results/20261120-1020-featprobe_1120_0947/`.
- Test PSNR, dB, and the paired per-image gap to L (30 images):

  | arm | against VAE(x0) | against the original | gap to L, VAE target |
  |---|---|---|---|
  | L | 26.47 | 26.26 | |
  | P | 26.08 | 25.89 | −0.39 (0 / 30 up, −0.86 … −0.05) |
  | M | 23.73 | 23.61 | −2.74 (0 / 30 up, −3.92 … −1.74) |
  | L + P | 26.71 | 26.48 | +0.24 (30 / 30 up, +0.08 … +0.60) |
  | L + M | 26.57 | 26.36 | +0.10 (30 / 30 up, +0.05 … +0.21) |

  The gaps against the original are within 0.03 dB of those against the VAE decode.
- Not run: a null arm at equal width (L plus features that carry no image), the turbo point
  (last σ 0.5, previous 0.75), a ×2 target, the trained head with P as a second input.
- Claude's forecasts before the run: the previous step would add almost nothing; a mid block
  was the likelier source. The mid block added less than the previous step.

## Interpretation

- The previous step's feature holds a little pixel information the last one does not expose
  to a linear map: a quarter of a dB on every image. Against the 10 dB between this readout
  and the trained head, and with a second DiT tap and a two-forward training rollout as the
  price, it is not a lever for ×2 sharpness.
- A linear probe cannot say what a non-linear head would draw from P; the head also reads
  x_t, which this probe leaves out.
- The last step's feature is the best single source; block 18 alone is 2.7 dB under it.
