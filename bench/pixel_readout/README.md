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

## p3_generated.py — P3, the head on sampled images

Renders each (prompt, seed) with `generate_body` (Euler, 50 steps, shift 3, CFG 3.5, bench
negative prompt, vocab pack attached), hooks the last step's x_t, cond / uncond features and
velocities, and reads the head three ways against VAE(x_final): CFG-combined feature, cond
feature, and teacher-forced on the generated latent. `--adapter` repeats every prompt at
multiplier 0 and 1. `cfg_check` confirms x_final = x_t − σ·v_cfg.

### p3_1119_1056 (head `0938`, adapter `anima_merged15`, 16 prompts × seeds 0, 1)

Results `results/20261119-1057-p3_1119_1056/` (local). PSNR vs VAE(x_final), dB, mean:

| | base cfg / cond / tf | LoRA cfg / cond / tf |
|---|---|---|
| colour (26) | 30.90 / 30.94 / 30.17 | 31.24 / 31.28 / 30.61 |
| mono (6) | 22.51 / 22.53 / 21.86 | 22.85 / 22.87 / 22.26 |

`cfg_check` ≤ 1.71e-3. Per-prompt table and reads: `project/pixel_decoder/reports/p3_2026_11_19.md`.

### manga_1119_1052 (`render_sheets.py`, head `0938`)

`render_sheets.py` now also reports VAE(x̂0) vs VAE(x0), the head vs VAE(x̂0), and for raw images
VAE vs original and head vs original (sheet columns original | VAE | head). Inputs: 4 Manga109-s
p.14 spreads, page 8 of 6 hitocrawl JA works, 2 sushispin. Results
`results/20261119-1054-manga_1119_1052/` (local); table in the P3 report. Manga: VAE vs original
21.46–39.79, VAE(x̂0) vs VAE(x0) 24.31–36.25, head vs VAE(x0) 21.51–33.24.

### mangaup_1119_1152 (head `convnext_1119_1106`, same 12 inputs and noise as `manga_1119_1052`)

Results `results/20261119-1201-mangaup_1119_1152/` (local). Head vs VAE(x0), `0938` → `1106`:
Manga109-s 21.51 → 21.82, 22.15 → 22.47, 27.53 → 27.87, 24.26 → 24.67; hitocrawl +0.32 … +0.79;
sushispin +0.60 / +0.68; 12 / 12 up. Head vs original on the 10 pages: +0.14 … +0.74.

### convnext_1119_1106 — one 2× upsample stage (the user's arm, 2026-11-19)

Parent `convnext_1119_0938` (`--no_skip --ls_init 1.0`), `--up 1`: blocks 1–2 at token
resolution (width 1024) as before; the third block runs after one 2× upsample (LayerNorm +
per-token linear 1024 → 4 × 512, pixel shuffle) at width 512, so its 3×3 depthwise sits at 8-px
spacing, and `out` writes an 8×8 patch per upsampled position (512 → 192) in place of 16×16
per token (1024 → 768). Same multiplies in the third block's MLP (4× positions, ¼ per
position); 23.3M against 28.2M parameters. Data, order, noise seeds, LR 2e-4, 4 updates per
rollout, 3600 rollouts, cosine, clip 1.0 unchanged.
Question: does a block at 8-px spacing pass `0938` on test / text and take out the softness
left in thin lines, small text and screentone? Above → the upsample stage joins the recipe
(P3's manga training runs on it). Level → `0938`'s head stays and capacity goes to width or
data.
Records: `0938` test 32.93 / text 27.45; `0950` (different recipe) 32.96 / 27.46.
Read: same sheets and PSNR beside `0938`, then `render_sheets.py` on `manga_1119_1052`'s
inputs; labelled; bars: spread unknown (no seed twin of `0938`); the user's read decides.
Cost: 3600 rollouts, ≈ 27 min.

Result (26 min, 0.43 s / rollout, 24.9M parameters; `results/20261119-1151-convnext_1119_1106/` (local)):

| test PSNR vs VAE (dB), rollout 600 / 1200 / 1800 / 2400 / 3000 / 3600 | test (30) | text (8) |
|---|---|---|
| `0938` | 25.69 / 29.39 / 31.01 / 32.21 / 32.64 / 32.93 | 22.63 / 25.14 / 26.21 / 26.99 / 27.36 / 27.45 |
| `1106` | 26.94 / 30.57 / 31.73 / 32.89 / 33.44 / 33.64 | 23.47 / 26.01 / 26.93 / 27.62 / 27.96 / 28.04 |

Per image, `1106` − `0938`: test +0.31 … +1.31 (30 / 30 up), text +0.42 … +0.87 (8 / 8 up). Spread
unknown. Claude's read (labelled, VAE | 0938 | 1106 crops of text 00, 02 and sheet 08): balloon
glyph strokes sharper with `1106`, closer to the VAE. Adopted into the recipe (Claude's call under
the user's proceed-unless-flagged rule, 2026-11-19).

## build_pages.py — manga pages for the training mix

Precaches VAE latents of Manga109-s images and hitocrawl JA pages (resized to ~4100 tokens) into
`post_image_dataset/pixel_pages/{train,eval}/`; the eval split is `manga_1119_1052`'s 10 pages
with their original pixels. `convnext_head.py --pages <dir> --page_share p` mixes `train/` in
(one fixed tag caption, `--page_caption`) and scores `eval/` as a `manga` set; `--target xhat0`
trains against VAE(x̂0). Every eval now reports each set against VAE(x0) and VAE(x̂0).

Build 2026-11-19: pools 8,140 Manga109-s / 9,683 hitocrawl JA (pages 1–2 of each work skipped),
1,500 + 1,500 drawn (seed 0), 471 s, 1.6 GB. `smoke_pages` (20 rollouts, `--target xhat0
--page_share 0.33`, no `--up`): 9 page rollouts; manga eval VAE vs original and VAE(x̂0) vs
VAE(x0) match `manga_1119_1052` within 0.1 dB.

## e2e_speed.py — decode cost on the `make test SPECTRUM=1` sampler

The 16 P3 prompts, anima-base, Euler 28 / shift 3 / CFG 4, Spectrum flags of
`scripts/tasks/inference.py` (last step an actual forward, σ_last 0.10), compiled blocks, seed 0.
The final latent is decoded four ways: the `1106` head (resident), the VAE kept on CPU and moved
in and out (`vae_swap`, the interactive path), the VAE loaded from disk (`vae_disk`, the
single-prompt path), the VAE resident (reference). Times with CUDA sync, pixels on CPU; peak VRAM
above the resident DiT.

### e2e_1119_1209 (`results/20261119-1212-e2e_1119_1209/` (local))

| per image, median of 16 | seconds | peak above resident DiT (MB) |
|---|---|---|
| DiT sampling | 6.58 | 1349 |
| head | 0.007 | 244 (+95 weights, resident) |
| VAE resident | 0.141 | 1298 |
| VAE swap | 0.178 | 1384 |
| VAE disk | 0.828 | 1384 |

16 images: DiT 105.6 s; decode adds 0.2 s (head), 2.8 s (swap), 12.9 s (disk). Head vs
VAE(x_final) PSNR: 27.09 mean, colour (13) 28.52, mono (3) 20.89; P3 (`0938` head, σ 0.0577,
50 steps, CFG 3.5) was 30.90 / 22.51. The run changes σ, steps, CFG, Spectrum and the head at
once; one seed.

## rollout_profile.py — where a rollout's time goes

12 train images (~4,100 tokens), medians (`results/20261119-1219-prof_bf16_1119_1217`,
`…-1220-prof_fp32_1119_1217`, local): load 2 ms, DiT forward 203 ms (bf16 stream) / 207 ms
(fp32), one VAE decode 141 ms, 4 head updates 73 ms eager / 62 ms under
`torch.compile(dynamic=True)`. DiT + VAE are 82 % of a rollout.

`bench/_anima.build_anima` does not apply `stream_fp32_`, so every head up to `convnext_1119_1106`
read bf16-stream features while `inference.py` defaults to `--stream fp32`. `convnext_head.py`
now takes `--stream` (default fp32), compiles the head's training forward under `--compile`, and
takes `--replay N`.

### replay_1119_1225 (the user's arm, 2026-11-19)

Parent `convnext_1119_1106`; changed together: `--stream fp32`, head compile, and
`--rollouts 900 --opt_steps 16 --replay 64` (each update draws one of the last 64 rollouts;
14,400 updates as the parent, a quarter of the DiT forwards and VAE decodes). `--eval_every 150`.
Question: does a quarter of the rollouts, reused from a buffer, hold `1106`'s test 33.64 / text
28.04? The user's rule: no large drop → replay goes into the recipe and the saved time goes to
more data. Held-out features are on the fp32 stream here, so the comparison carries the stream
change too.


`replaybf16_1119_1231` (the same arm on `--stream bf16`) was queued and cancelled by the user
(2026-11-19): fp32 stream is settled on other evidence, so a drop in `replay_1119_1225` reads as
replay's.

Result `replay_1119_1225` (`results/20261119-1236-replay_1119_1225/` (local)), 11.3 min against
`1106`'s 25.7 (0.76 against 0.43 s / rollout, a quarter of the rollouts):

| updates 2400 / 4800 / 7200 / 9600 / 12000 / 14400 | test (30) | text (8) |
|---|---|---|
| `1106` | 26.94 / 30.57 / 31.73 / 32.89 / 33.44 / 33.64 | 23.47 / 26.01 / 26.93 / 27.62 / 27.96 / 28.04 |
| `replay_1119_1225` | 29.03 / 31.30 / 31.97 / 32.63 / 33.21 / 33.35 | 24.45 / 26.01 / 26.72 / 27.20 / 27.54 / 27.68 |

Per image, replay − `1106`: test −0.47 … −0.13 (0 / 30 up), text −0.41 … −0.28 (0 / 8 up).
Against VAE(x̂0): test 34.93, text 28.77. P1 linear (fit on bf16 features) reads 25.21 on the
fp32-stream features against 26.28 on bf16. Spread unknown.
Reading: ahead to 7,200 updates, behind from 9,600 on; the buffer's 900 distinct rollouts
against 3,600 is the likely limit late in the run. The user's rule (no large drop → replay in the
recipe, time to data) applies.
The user's read (2026-11-19, native-resolution crops of the 8 text windows and sheet 08, VAE |
`1106` | replay): no difference, or replay slightly better, which the user attributes to the
fp32 stream.

### pagemix_1119_1242 (P3 manga arm 2 on the replay recipe; the user's call, 2026-11-19)

`replay_1119_1225`'s flags with `--target xhat0 --pages post_image_dataset/pixel_pages
--page_share 0.33 --rollouts 2000 --eval_every 400` (32,000 updates). Arm 1 (no pages in the mix)
was dropped by the user: target, rollout count and pages change together against
`replay_1119_1225`, and the reading is not held to a separated comparison.
Question: does the page mix take out the soft small text, screentone and the grey tint on mono,
without losing colour? Read: test / text / manga PSNR (vs VAE(x0) and VAE(x̂0)) beside
`replay_1119_1225` and `mangaup_1119_1152`, manga sheets (original | VAE | head), labelled; the
user's read decides.
