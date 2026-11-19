# Turbo point on v4_longer, and a real-image target with LPIPS on the base points (2026-11-19)

Two micro arms from the release head `mixr_1119_1455`. The real-target arm follows DWD
(arXiv:2611.22287; `../idea.md`).

## Facts

### Setup

- `output/ckpt/release/anima_turbo_4step_v4_longer.safetensors` (local; built by
  `project/anima_turbo/deploy/deploy.sh … lora`: merged15 soup ++ the official turbo v1.1 delta,
  exact concat, ranks 352 / 224, bf16, 1.25 GB). Uploaded to the public Hub repo
  `sorryhyun/anima-turbo-4step` on the user's word (commit `67a5ce82`), file only.
- `convnext_head.py` gained `--points …:real` (that point trains against the image its latent was
  encoded from: `post_image_dataset/resized/`, or a page's source resized as in `build_pages.py`),
  `--lambda_lpips` (VGG LPIPS on one random 512² crop per update, `:real` points only),
  `--grad_ratio`, an eval at rollout 0 for `--init` runs, and per-set PSNR / LPIPS against the
  original for the head, VAE(x0) and VAE(x̂0).
- Both arms: the `mixr_1119_1455` argv with `--adapter` → v4_longer, `--init mixr_1119_1455`,
  `--rollouts 1000` (500 per point), fp32 stream, compiled. One seed, spread unknown.

| run | points | changed from the release recipe | time |
|---|---|---|---|
| `mixt_1119_2106` (T) | `0.0577-0.1364 0.5:adapter` | the adapter | 14.8 min, 0.76 s / rollout |
| `real_1119_2106` (R) | `0.0577-0.1364:real 0.5:adapter`, `--lambda_lpips 0.05` | the adapter; base-point target VAE(x̂0) → original; LPIPS on the base points | 15.1 min, 0.78 s / rollout |

R changes the target and adds LPIPS together; the run does not separate them. λ 0.05: the
unweighted LPIPS / reconstruction gradient-norm ratio on the release head against real targets
was 71 (45 … 113, 8 rollouts, `smoke_real_1119_2105`) and 62 (33 … 153) in R, so λ 0.05 puts the
LPIPS term at about 3× the reconstruction term, inside the 2 – 5× of the ×2 sweep's λ 0.1
(`x2lp*_1119_1944`).

### Held-out, teacher-forced (30 test / 8 text / 10 manga)

Both arms dip after the warm start and recover: T's turbo point (test, against VAE(x̂0)) reads
36.49 at rollout 0, 34.86 at 250, 37.38 at 1,000; R's base point (σ 0.0577 test, LPIPS against the
original) 0.104, 0.181, 0.083.

Turbo point σ 0.5 on v4_longer, PSNR against VAE(x̂0), test / text / manga:

| head | |
|---|---|
| `mixr_1119_1455` as loaded | 36.49 / 30.27 / 29.49 |
| T | 37.38 / 30.76 / 30.25 |
| R | 36.85 / 30.36 / 29.84 |

`mixr_1119_1455` on the old turbo file it was trained with: 37.17 / 31.63 / 30.71 (a different
adapter, so different features and targets).

Base points against the original image, PSNR dB / LPIPS:

| σ, set | VAE(x̂0) | T | R |
|---|---|---|---|
| 0.0577 test | 35.10 / 0.056 | 33.95 / 0.100 | 33.57 / 0.083 |
| 0.0577 text | 30.27 / 0.038 | 28.87 / 0.072 | 28.39 / 0.070 |
| 0.0577 manga | 26.95 / 0.062 | 26.19 / 0.114 | 25.84 / 0.104 |
| 0.0970 test | 32.99 / 0.075 | 32.52 / 0.111 | 32.34 / 0.093 |
| 0.0970 text | 28.08 / 0.052 | 27.53 / 0.081 | 27.28 / 0.078 |
| 0.0970 manga | 25.23 / 0.080 | 25.03 / 0.126 | 24.84 / 0.116 |
| 0.1364 test | 31.32 / 0.092 | 31.12 / 0.124 | 31.07 / 0.106 |
| 0.1364 text | 26.41 / 0.067 | 26.22 / 0.093 | 26.12 / 0.090 |
| 0.1364 manga | 23.93 / 0.097 | 23.95 / 0.139 | 23.88 / 0.129 |

- VAE(x0), the clean latent's decode, against the original: 39.21 / 0.029 test, 34.92 / 0.019
  text, 30.43 / 0.039 manga.
- R − T per image over the 48: PSNR −0.39 (0 up) at σ 0.0577, −0.19 (0 up) at 0.0970, −0.06
  (8 up) at 0.1364; LPIPS lower on 32, 35 and 35 of 48.
- Base points against VAE(x̂0): T 38.13 / 32.34 / 31.27 at σ 0.0577 (release head 38.05 / 32.36 /
  31.30); R 36.78 / 31.07 / 29.88.

### Generated images (P3's 16 prompts × seeds 0, 1; head against the VAE decode of the same latent)

| read | run | `mixr_1119_1455` | T | R |
|---|---|---|---|---|
| base, 28 steps, CFG 4 | `real_base28_1119_2106` | 32.44 (`mixr_base28_1119_1455`) | 32.45 (colour 33.86, mono 26.34) | 31.41 (33.00, 24.54); R − T −2.82 … −0.16, 0 / 32 up |
| v4_longer, 4 steps, CFG 1 | `real_turbo_1119_2106` | 32.90 (33.95, 28.36) | 33.66 (34.69, 29.22); T − mixr +0.53 … +1.48, 32 / 32 up | 33.16 (34.19, 28.73); R − T −0.81 … −0.20, 0 / 32 up |

- Claude's read, labels visible, `sheet_base_13_manga_page_s0.png` (VAE | T | R): on grey
  regions the VAE shows screentone dots, T a smooth grey, R a fine regular pattern at about the
  dot pitch; R's greys look slightly darker than T's. Line work reads the same in T and R.
- Claude's read, labels visible, `sheet_lora_01_sakura_s0.png`: the v4_longer LoRA on plain
  anima-base renders a clean 4-step image; the three heads are not told apart at sheet size.
- The user's read (2026-11-19, the base and turbo sheets, labelled): "괜찮은듯? 톤이 살짝 달라지는것
  같긴 한데 원본에 충실한쪽은 우측이네" (the right column is R). On replacing the release head: "릴리스
  헤드를 바꿔도 되긴 하는데 흠... x2 preview 느낌으로 릴리스하는건 어때?"
- Sheets: `bench/pixel_readout/results/20261119-2136-real_base28_1119_2106/`,
  `20261119-2143-real_turbo_1119_2106/`, and each training run's own directory (local).

## Interpretation

- T is the turbo-point update for v4_longer: every render moved up and the base points held. It
  was still rising at 1,000 rollouts, and a third of the budget went to recovering from the
  warm start at the full learning rate, so a lower peak LR or a longer run is the next lever.
- On the original, the head sits under the VAE path at the same σ on both meters at low σ and
  meets it in PSNR at σ 0.136. The gap to close is head against VAE(x̂0), not VAE against the
  original.
- R trades about 0.1 – 0.4 dB against the original for 10 – 18 % lower LPIPS on test and manga;
  text barely moves. It closes roughly a third of the LPIPS gap to the VAE path on test and
  leaves the PSNR gap wider. The drop against the VAE decode on renders is by construction (R no
  longer regresses onto the VAE's output) and says nothing by itself.
- Whether R's pattern on screentone regions is the wanted texture or an artefact is a read of the
  sheets. The run cannot say whether the real target or the LPIPS term produced it; the arm that
  separates them is the T recipe with LPIPS on the VAE(x̂0) target.
- The shared head pays for R on the turbo point (−0.5 dB against T) although that point's target
  and loss did not change.

## Full run `realr_1119_2200`

### Facts

`real_1119_2106` with `--rollouts` 1000 → 3000, nothing else changed (36.3 min, 0.68 s per
rollout). One seed, spread unknown. The replacement of the release head is decided on this run
(the user, `../x2_plan.md`).

Held-out, against the original, PSNR dB / LPIPS (T = `mixt_1119_2106`, R = `real_1119_2106`):

| σ, set | VAE(x̂0) | T | R, 1,000 | `realr`, 3,000 |
|---|---|---|---|---|
| 0.0577 test | 35.10 / 0.056 | 33.95 / 0.100 | 33.57 / 0.083 | 33.75 / 0.081 |
| 0.0577 text | 30.27 / 0.038 | 28.87 / 0.072 | 28.39 / 0.070 | 28.61 / 0.067 |
| 0.0577 manga | 26.95 / 0.062 | 26.19 / 0.114 | 25.84 / 0.104 | 26.02 / 0.102 |
| 0.0970 test | 32.99 / 0.075 | 32.52 / 0.111 | 32.34 / 0.093 | 32.47 / 0.091 |
| 0.1364 test | 31.32 / 0.092 | 31.12 / 0.124 | 31.07 / 0.106 | 31.16 / 0.105 |
| 0.1364 manga | 23.93 / 0.097 | 23.95 / 0.139 | 23.88 / 0.129 | 23.98 / 0.128 |

- Turbo point σ 0.5 against VAE(x̂0), test / text / manga: 37.22 / 30.69 / 30.22 (T 37.38 /
  30.76 / 30.25; R 36.85 / 30.36 / 29.84).
- Eval history, σ 0.0577 test, LPIPS against the original: 0.104 at rollout 0, 0.205 at 500,
  0.156 at 1,000, 0.115 at 1,500, 0.083 at 2,500, 0.081 at 3,000.
- Generated images, head against the VAE decode of the same latent, 32 renders each:
  base 28 steps CFG 4 (`realr_base28_1119_2240`) T 32.45, `realr` 31.80 (colour 33.34, mono
  25.14; −1.93 … +0.21, 2 / 32 up); v4_longer 4 steps (`realr_turbo_1119_2240`) T 33.68, `realr`
  33.53 (−0.36 … +0.05, 2 / 32 up).
- One flat grey region of `sheet_base_13_manga_page_s0.png` (128², luma levels): standard
  deviation VAE 6.0, T 4.8, `realr` 8.7; share of the AC power at the 4 / 8 / 16-px lattice
  frequencies VAE 1.1 %, T 5.8 %, `realr` 7.8 %. One region of one image.
- Claude's read, labels visible, the same sheet and `sheet_base_08_beach_s0.png`: on the manga
  page `realr` carries a fine regular pattern over the grey regions where T is smooth and the VAE
  shows irregular dots; on the colour sheet the three columns are not told apart at sheet size.
- Sheets (local): `bench/pixel_readout/results/20261119-2242-realr_base28_1119_2240/`,
  `20261119-2249-realr_turbo_1119_2240/`.
- The user's read (2026-11-19, these sheets and numbers): "흠 뭐 이정도면 괜찮을듯 커밋하자".

### Interpretation

- Three times the rollouts bought about +0.2 dB and −3 % LPIPS over the micro arm and brought the
  turbo point back to T's level. The history spends half the run under its starting point, so the
  outcome is set by the LR decay more than by the length; a lower peak LR for warm starts is the
  cheaper lever.
- On the one flat region the head's texture is stronger than the VAE's own, and a small part of
  it sits on the patch lattice. Whether that reads as screentone or as a grid is the user's call
  on the sheets.
