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

## convnext_head.py — P2 (`docs/anima/proposal/pixel_readout_head.md`)

Head: P1's ridge linear as an fp32 skip (init from `readout_1118_2219`, LR 2e-5) plus
`proj(f) + embed(x_t 2×2 patch)` → 3 ConvNeXt blocks (width 1024, 3×3 depthwise, 4× GELU MLP,
layer scale 1e-6) → zero-init per-token linear to 768; 29.7M params. Same tap and σ as P1.
Train: anima-base cache minus P1's 8 test artists and 8 text images; one DiT forward + one VAE
decode per rollout, 4 AdamW steps on it (FastFrames App. C), LR 2e-4 cosine, warmup 200.
Loss 0.5 L1 + 5 MSE + 0.05 L_grad.

```bash
make daemon-run ARGS="--queue --stall-timeout 900 bench/pixel_readout/convnext_head.py --compile --label convnext_mmdd_hhmm"
```

- Question: does neighbour context fix what P1 leaves (broken thin outlines, flat-region grain,
  low contrast)? At VAE level → P3. Visibly softer than VAE → widen / deepen once (5 blocks,
  1536), then GAN (P5) is a requirement.
- Read: P1's 12 test sheets + 8 text sheets (`text_*.png`, crop at the densest-edge window),
  VAE | P1 linear | P2 head, labelled; test / text PSNR vs VAE beside P1 (26.28). Bars: spread
  unknown. The user's read decides.
- Cost: 3600 rollouts × 4 steps, ≈ 30 min plus compile.

### convnext_1119_0835 (anima-base, 26 min, 0.43 s / rollout, 7.7 GB)

Results `results/20261119-0900-convnext_1119_0835/` (local). Train 2981 images.

| test PSNR vs VAE (dB) | test (30) | text (8) |
|---|---|---|
| P1 linear (fp32, same features) | 26.28 | 21.61 |
| P2 head, rollout 600 / 1200 / 1800 / 2400 / 3000 | 26.09 / 26.25 / 26.30 / 26.37 / 26.46 | 21.50 / 21.51 / 21.56 / 21.56 / 21.57 |
| P2 head, rollout 3600 (final) | 26.50 | 21.58 |

Per image, head − P1: test −0.12 … +0.88 (23 / 30 up), text −0.10 … +0.06 (2 / 8 up). Spread
unknown. Layer-scale γ at the end, mean |γ| per block: 2.1e-3, 1.1e-3, 1.8e-3 (init 1e-6);
mean |out.weight| 5.7e-3 (init 0).

Note: the ConvNeXt path's output stayed small next to the skip (γ ≈ 1e-3), so this run reads
mostly as P1 plus a small correction.

### convnext_1119_0905 — layer scale γ init 1e-6 → 0.1

Parent `convnext_1119_0835`, `--ls_init 0.1`, nothing else changed. Question: does the
ConvNeXt path, once open, remove the 16-px grid / grain / thin-outline errors P1 leaves?
Read: same sheets and PSNR beside `0835` and P1; spread unknown; the user's read decides.

Stopped at rollout ~1000 (the user's call): train loss matched `0835` to the third decimal at
every print (same data order and noise seeds); eval @ 600 test 26.13 / text 21.51 against
`0835`'s 26.09 / 21.50.

## grad_probe.py — where the head's gradient goes

### gradprobe_1119_0914 (16 train images, forward + backward, no step)

| | init γ 1e-6 | init γ 0.1 | `0835` final |
|---|---|---|---|
| total ‖g‖ / clip factor | 0.179 / 1.000 | 0.179 / 1.000 | 0.159 / 1.000 |
| grad share: skip linear / blocks / out | 0.77 / 0 / 0.23 | 0.77 / 0 / 0.23 | 0.79 / 0.17 / 0.03 |
| pairwise cos of per-image out.weight grads | 0.025 | 0.025 | 0.085 |
| output RMS, residual path / skip | 0 / 0.651 | 0 / 0.651 | 0.021 / 0.643 |

No null run for the cosine. P1's error RMS on test ≈ 0.097 (from 26.28 dB).

Note: clipping never engaged and the out layer gets gradient at init under either γ, so neither
clipping nor the init held the residual path back.

## render_sheets.py — sheets for chosen images

VAE | P1 | P2 head on cached items or raw images (raw: resized to ~4100 tokens, VAE-encoded,
one `--caption` through the TE). Forward only.

### sheets_1119_0904 (head = convnext_1119_0835)

Results `results/20261119-0904-sheets_1119_0904/` (local). PSNR vs VAE, dB, P1 → P2:

| image | P1 | P2 |
|---|---|---|
| sushispin 10043045 / 10186995 / 12210581 / 12688803 / 9859367 (rated safe, held-out artist, in neither P1 nor P2 training) | 26.27 / 24.64 / 26.09 / 23.83 / 27.20 | 26.46 / 24.79 / 26.24 / 23.98 / 27.41 |
| Manga109-s p.14 spreads ARMS / DollGun / AosugiruHaru / BakuretsuKungFuGirl (caption `monochrome, greyscale, comic, speech bubble, japanese text`) | 16.61 / 16.53 / 22.14 / 18.62 | 16.53 / 16.44 / 22.16 / 18.52 |

Claude's read (labelled, sheets 00, 05, 07): colour images as in the test set (16-px block grid,
washed contrast, grain). Manga: panels, faces and large SFX kept; small balloon text keeps its
layout but not readable glyphs; blacks lift to grey and white paper takes grain; screentone and
hatching blur to grey.
