# pixel_readout

Pixel readout from the frozen Anima DiT's pre-head features in place of the VAE decoder
(FastFrames, arXiv:2611.23226 §3.2 and App. C).

## linear_probe.py — probe 1

Question: does a per-token linear map from the features `final_layer.linear` reads
(2048-dim, LayerNorm + adaLN-modulated) to a 16×16 RGB patch reproduce the VAE decode on
Anima? Recognisable → a ConvNeXt head (probe 2). Not recognisable → Anima's last-layer
features do not carry pixels the way Wan's do, and the line stops.

Per image: VAE-decode the cached clean latent (target), noise it to the last σ of the default
grid (50 steps, shift 3 → σ 0.0577), one DiT forward with the cached caption, tap the final
layer's input. Ridge least squares on 256 train images; scored on 30 images from 8 artists held
out of the fit. Reference: the VAE image averaged per 16×16 patch and upsampled (one colour per
token).

```bash
make daemon-run ARGS="--queue --stall-timeout 600 bench/pixel_readout/linear_probe.py --compile --label readout_mmdd_hhmm"
```

Output: `results/<stamp>-<label>/` with `result.json` (per-image PSNR), `sheet_*.png` (VAE |
linear | token-mean bilinear, full image and a 2× centre crop) and `readout_linear.safetensors`.

### readout_1118_2219 (anima-base, 2.4 min)

| | test PSNR vs VAE (dB) |
|---|---|
| linear readout (ridge 1e-6) | 26.28 |
| token mean, bilinear | 22.07 |
| token mean, nearest | 21.57 |

Linear is above token-mean bilinear on all 30 test images (+2.6 … +6.4 dB). Ridge 1e-4: 26.13,
1e-2: 24.83. Features are teacher-forced (real latents at σ_last), not taken from a sampled
trajectory.
