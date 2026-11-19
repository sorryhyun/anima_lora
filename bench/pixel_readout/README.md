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

## convnext_head.py — P2 (`project/pixel_decoder/proposal.md`)

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

## ceiling_probe.py — which target the features can reach

### ceiling_1119_0920 (forward only, P2's 30 test + 8 text images, same noise seeds; head = `0835`)

Results `results/20261119-0920-ceiling_1119_0920/` (local). x̂0 = x_t − σ·v, the DiT's one-step
estimate at σ 0.0577. PSNR, dB:

| | test (30) | text (8) |
|---|---|---|
| VAE(x_t) vs VAE(x0) | 30.73 | 27.08 |
| VAE(x̂0) vs VAE(x0) | 37.00 | 32.02 |
| P1 / P2 vs VAE(x0) | 26.30 / 26.50 | 21.62 / 21.59 |
| P1 / P2 vs VAE(x̂0) | 26.47 / 26.69 | 21.73 / 21.70 |

Latent RMSE to x0: x_t 0.068, x̂0 0.032 (latent RMS 0.63). VAE(x̂0) is above the P2 head on
38 / 38 images, VAE(x_t) on 38 / 38.

Note: the clean-latent target is not what holds the head at P1's level; the VAE decode of the
head's own x_t input is 4.2 dB above the head on test.

## overfit_probe.py — can the head fit 16 images?

Features, x_t and VAE targets of 16 train images computed once (fixed noise), then a fresh head
per arm trained on those pairs with `0835`'s optimizer, 4000 steps, batch 1, 18 ms / step.
Arms: `parent` = `0835`'s head (P1 skip at LR 2e-5, out zero-init, γ 1e-6); `open` = γ 1;
`noskip` = no skip, out default init, γ 1. On the 16 fit images: P1 25.71, VAE(x_t) 31.07,
VAE(x̂0) 37.05 (all vs VAE(x0)); P1 on test 26.30.

### overfit_1119_0924 (head LR 2e-4) and overfit_lr1e3_1119_0925 (head LR 1e-3)

Results `results/20261119-0929-overfit_1119_0924/`, `results/20261119-0933-overfit_lr1e3_1119_0925/`
(local). PSNR vs VAE(x0) at step 4000, dB; one seed, spread unknown.

| arm | LR 2e-4: fit (16) / test (30) | LR 1e-3: fit / test | steps clipped, 2e-4 / 1e-3 |
|---|---|---|---|
| `parent` | 31.51 / 25.67 | 40.48 / 23.79 | 0 / 1 |
| `open` | 35.05 / 24.66 | 47.53 / 25.21 | 0 / 0 |
| `noskip` | 30.83 / 27.16 | 39.42 / 26.74 | 413 / 500 |

`noskip` test by step: LR 2e-4 24.05 @ 500, 26.94 @ 2000, 27.16 @ 4000; LR 1e-3 24.98 @ 1000,
26.91 @ 2000, 26.74 @ 4000. Both skip arms' test falls from P1's 26.30 at every eval after
step 1000 (LR 2e-4) or from the first eval (LR 1e-3). `parent`'s γ ended at 0.16–0.17 (LR 2e-4)
against 1e-3 in `0835`.

Note: every arm fits 16 images, so the head is trainable as built; with the P1 skip what the
residual path learns does not carry to held-out images, and without it 16 images already put
test above `0835`'s 26.50. Fit values above 37 are past what the DiT's own x̂0 gives and can
only be memorised noise draws. The probe does not separate the skip from γ 1 and the out init.

### convnext_1119_0938 — no skip

Parent `convnext_1119_0835`, `--no_skip --ls_init 1.0` (no P1 skip, out default init, γ 1; the
three change together as in `overfit_1119_0924`'s `noskip` arm, and the run does not separate
them). Head LR 2e-4, data, order, noise seeds and schedule as the parent.
Question: does the no-skip head, on the full train set, pass `0835` and remove the 16-px grid /
grain / thin-outline errors? Yes → P3 on this head, LR 1e-3 as the next arm. Level with `0835`
→ gradient accumulation over rollouts.
Records: `overfit_1119_0924` `noskip` test 27.16 from 16 images; `ceiling_1119_0920` VAE(x̂0)
37.00 / VAE(x_t) 30.73 on test.
Read: same sheets and PSNR as `0835`, labelled; spread unknown; the user's read decides.
Cost: 3600 rollouts × 4 steps, ≈ 26 min.

Result (25 min, 0.42 s / rollout; `results/20261119-0959-convnext_1119_0938/` (local)):

| test PSNR vs VAE (dB) | test (30) | text (8) |
|---|---|---|
| P1 linear | 26.28 | 21.61 |
| `0835` final | 26.50 | 21.58 |
| `0938`, rollout 600 / 1200 / 1800 / 2400 / 3000 | 25.69 / 29.39 / 31.01 / 32.21 / 32.64 | 22.63 / 25.14 / 26.21 / 26.99 / 27.36 |
| `0938`, rollout 3600 (final) | 32.93 | 27.45 |

Per image, head − P1: test +5.04 … +8.27 (30 / 30 up), text +4.95 … +6.66 (8 / 8 up). Train
PSNR (EMA) 30.7 at rollout 1950 against 24.3–25.0 throughout `0835`.

Claude's read (labelled, 2× crops of sheets 00, 08 and text 02): the 16-px grid and the grain
are gone and outlines are continuous; against the VAE the head is a little softer, thin lines
waver slightly (bikini strings, sheet 08), and small balloon text is legible with softer strokes.

### convnext_1119_0950 — gradient over 4 rollouts, LR 1e-3 (the user's arm)

Parent `convnext_1119_0938` (no skip, γ 1), `--accum 4 --opt_steps 1 --lr 1e-3`: each update
averages the gradient over the last 4 rollouts (sliding window; shapes differ, so accumulation),
one update per rollout, head LR 2e-4 → 1e-3. The three change together and the run does not
separate them. Kept: each rollout enters 4 gradient evaluations, head compute per rollout, data,
order, noise seeds, cosine, clip 1.0. Changed with them: 3600 updates against 14,400; warmup
200 updates is 200 rollouts against 50; summed LR 1.25×.
Question: at the same rollout budget, does 4-rollout averaging at LR 1e-3 end above batch 1 at
2e-4? Above → the recipe for longer runs; level or below → `0938`'s recipe stays.
Read: test / text PSNR at each eval beside `0938`, same sheets, labelled; spread unknown; the
user's read decides.
Cost: 3600 rollouts, ≈ 26 min.

Result (25 min, 0.41 s / rollout; `results/20261119-1024-convnext_1119_0950/` (local)):

| test PSNR vs VAE (dB), rollout 600 / 1200 / 1800 / 2400 / 3000 / 3600 | test (30) | text (8) |
|---|---|---|
| `0938` | 25.69 / 29.39 / 31.01 / 32.21 / 32.64 / 32.93 | 22.63 / 25.14 / 26.21 / 26.99 / 27.36 / 27.45 |
| `0950` | 27.57 / 29.61 / 31.15 / 32.20 / 32.78 / 32.96 | 22.96 / 24.76 / 26.04 / 26.83 / 27.34 / 27.46 |

Per image at the end, `0950` − `0938`: test −0.09 … +0.18 (20 / 30 up), text −0.05 … +0.07
(5 / 8 up). Verdict: level; `0938`'s recipe stays.

### ceiling_1119_1003 — the `0938` head against both targets

`ceiling_probe.py --head_run …convnext_1119_0938`, `results/20261119-1025-ceiling_1119_1003/`
(local). Head vs VAE(x0): test 32.93, text 27.45. Head vs VAE(x̂0): test 34.30, text 28.44.
VAE(x̂0) vs VAE(x0): 37.00 / 32.02 as in `ceiling_1119_0920`.

## render_sheets.py — sheets for chosen images

VAE | P1 | P2 head on cached items or raw images (raw: resized to ~4100 tokens, VAE-encoded,
one `--caption` through the TE). Forward only.

### sheets_1119_0904 (head = convnext_1119_0835)

Results `results/20261119-0904-sheets_1119_0904/` (local). PSNR vs VAE, dB, P1 → P2:

| image | P1 | P2 |
|---|---|---|
| sushispin 10043045 / 10186995 / 12210581 / 12688803 / 9859367 (rated safe, held-out artist, in neither P1 nor P2 training) | 26.27 / 24.64 / 26.09 / 23.83 / 27.20 | 26.46 / 24.79 / 26.24 / 23.98 / 27.41 |
| Manga109-s p.14 spreads ARMS / DollGun / AosugiruHaru / BakuretsuKungFuGirl (caption `monochrome, greyscale, comic, speech bubble, japanese text`) | 16.61 / 16.53 / 22.14 / 18.62 | 16.53 / 16.44 / 22.16 / 18.52 |

### sheets_1119_1032 (head = convnext_1119_0938, same nine inputs)

Results `results/20261119-1042-sheets_1119_1032/` (local). PSNR vs VAE, dB, P1 → head:

| image | P1 | head |
|---|---|---|
| sushispin 10043045 / 10186995 / 12210581 / 12688803 / 9859367 | 26.27 / 24.64 / 26.09 / 23.83 / 27.20 | 32.20 / 29.87 / 31.12 / 28.57 / 33.79 |
| Manga109-s p.14 spreads ARMS / DollGun / AosugiruHaru / BakuretsuKungFuGirl | 16.61 / 16.53 / 22.14 / 18.62 | 21.52 / 22.17 / 27.49 / 24.26 |

Claude's read (labelled, 2× crops of sheets 00, 05, 07): colour image close to the VAE, no grid
or grain, fine line work (flower petals) a little softer. Manga: blacks are black and paper is
white, panel lines and hatching hold; the smallest balloon text keeps its layout with soft,
partly unreadable strokes; a dotted ellipsis column reads as a faint line; screentone dots
smooth into grey.

### sheets_1119_0904 — read

Claude's read (labelled, sheets 00, 05, 07): colour images as in the test set (16-px block grid,
washed contrast, grain). Manga: panels, faces and large SFX kept; small balloon text keeps its
layout but not readable glyphs; blacks lift to grey and white paper takes grain; screentone and
hatching blur to grey.
