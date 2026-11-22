# Plan (2026-11-19)

P1 – P4 are complete, with the release, the promotion and the ComfyUI node below (the user,
2026-11-19: "일단은 p4까지 완료인걸로", then "p4는 완료된걸로 갱신하고 커밋하자"). Their record and the settled
training recipe are in `reports/phases_p1_p4_2026_11_19.md`; each run's design and numbers are in
`reports/`. The release head is `mixr_1119_1455`
(`bench/pixel_readout/results/20261119-1527-mixr_1119_1455/head.safetensors`, local). This file
holds the P4 wrap-up, the P5 topics and what is open.

## P4 wrap-up (done 2026-11-19)

Order (the user, 2026-11-19): "배포되면 프로젝트 모델 코어는 라이브러리에 편입하는 쪽으로
계획을 잡자. p5는 아마 픽셀로스 등 integration일텐데 배포 및 승격을 선행하게 될듯". Release,
then promotion of the model core into `library/`, then the P5 topics.

### 1. Release

- Done 2026-11-19: `mixr_1119_1455` uploaded to the private repo `sorryhyun/anima-pixel-decoder`
  (the user: private first, they edit the card later). Card source `deploy/README.md`; decode
  cost on turbo and Spectrum and the upload record in `reports/release_2026_11_19.md` (turbo:
  VAE swap 15 % of DiT + decode, head 7 ms).
- Left to the user: the card text, making the repo public.
- Open: the head's turbo point was trained with the Hub turbo repo's
  `old/anima_turbo_4step.safetensors`; on v3 / v4 it reads 31.50 / 31.40 against 34.61
  (`reports/release_2026_11_19.md`). A turbo point on v3 or v4 (`--points … :adapter`,
  ~7 min per 500 rollouts) is the fix; not run.

### 2. Promotion — the model core into `library/`

Done 2026-11-19 (the user: "승격 ㄱㄱ"):

- `library/anima/pixel_head.py`: `PixelHead`, `ConvNeXtBlock`, `load_pixel_head` (shape and
  trained σ ranges from the metadata), `lat_patches`, `from_patches`, `PATCH`, `PixelTap`
  (hooks on the DiT forward and `final_layer.linear`, CFG combine, decode), `out_of_range`.
  `bench/pixel_readout/` imports the model code from it.
- Catalog row `anima_pixel_head`, pack `anima-pixel-head` (`models/vae/`, fetched on first use;
  GUI pack strings in four languages).
- `inference.py --decoder pixel_head [--pixel_head <file>]` (`library/inference/pixel_decoder.py`)
  on the single, interactive and batch paths; `PIXEL=1` / `PIXEL_HEAD=` on `test-*` and
  `test-turbo`; `GenerationRequest(decoder=…, pixel_head=…)`. Warns outside the trained σ, on a
  Spectrum cached last step and under `--cfgpp` / `--smc_cfg`; refuses `--tiled_diffusion`;
  `--latent_path` falls back to the VAE with a warning.
- Façade: `anima_lora.models.{PixelHead, load_pixel_head, PixelTap}`,
  `anima_lora.inference.{ensure_pixel_decoder, pixel_decode, save_decoded}`.
- Tier 2: bench `e2e_speed.py`, test `tests/test_pixel_head.py` (CPU), doc
  `docs/anima/inference/pixel_head.md`.
- Check through the daemon (one seed, one prompt, head PNG vs VAE decode of the saved latent):
  base 28 steps CFG 4 46.2 dB, turbo 38.2 dB; no warnings.

- ComfyUI node: built and published 2026-11-19, GitHub `sorryhyun/ComfyUI-Anima-PixelDecoder`,
  registry `comfyui-anima-pixel-decoder` 0.1.0, linked from the card with colour / mono samples. `AnimaPixelHeadTap` + `AnimaPixelDecode`,
  `_vendor/pixel_head.py` via `make vendor-sync`. Turbo in ComfyUI: total 1.127 s (VAE Decode)
  vs 1.005 s (head), 36.91 dB (`reports/comfy_node_2026_11_19.md`). Not run in ComfyUI: the
  Spectrum KSampler, the base DiT at CFG 4.

## P5 topics

After the release and the promotion, on the library's head (the user, 2026-11-19: "기존 plan은
이제 p5에서 unnest해서 여러 주제를 적자. x2 or x4, rgba, pixel loss 등"). Each topic is its own
line of work with its own proposal; order not set. The first steps are Claude's drafts.

### ×2 / ×4 output

Its own plan: `x2_plan.md` (pointers into `reports/` for everything that ran, the state, the
sequence ahead, next arms). In short: the head
writes 32×32 px per token through a second 2× stage; arm A (`x2a_1119_1802`, regression onto the
released ResShift ×2 student's output) came out soft, 1.7 – 3.0 dB under the teacher run on the
release head's ×1 output; LPIPS and a tone term (`x2lt_1119_2013`) and the SR line's DMD / GAN
critic (`reports/x2_critic_2026_11_20.md`) did not close it. A ×2 file shipped as a test file
(`reports/x2_preview_release_2026_11_19.md`). Next, on the user's word (2026-11-20): arm B, the
real 2× image as the target at the base points, the turbo point left out of ×2 for now.

### RGBA

An alpha channel on the head's output is cheap to add, and labels can come from real RGBA
cutouts composited onto random backgrounds (exact alpha) or a matting model on generated images.
What the head alone cannot give: the DiT renders a composite, so semi-transparent regions (hair
tips, glass, smoke) carry no background-free foreground colour; LayerDiffuse tunes the DiT for
that. The baseline to beat is post-hoc matting (BiRefNet, anime-segmentation) on hair-edge
sheets.

### Pixel losses on the head

What is left against the VAE on `mixr_1119_1455`, on the user's read: screentone dots smoothed
into soft shading; earlier, speed lines and footnote-size glyphs soft under zoom.
Measured beside it: textured regions 1 – 2.5 levels darker than the VAE in linear light
(`tone_probe.py`), the meter for any arm here.

- GAN on the head: FastFrames uses DINOv3 ViT-S/16 heads at layers 4 / 8 / 12 with a
  relativistic loss and R1 / R2; the turbo line's disc code is in git history.
- A paired real-pixel target: head training is teacher-forced from cached latents, so each pair
  has its real image (`post_image_dataset/resized/`), a target past the VAE without a GAN.
- A ResShift 1-step refiner from the head's output, read against the GAN arm.

### Pixel losses through the head, into the DiT

The head as the differentiable decode for losses that train the DiT or a LoRA: cjk_rl arm D
(reader NLL through the head, its D0 cost probe rerun against the VAE-window path), a GAN or
refiner term for the turbo student.
Idea and first probe (PixelGen-style LPIPS through the head; DWD, arXiv:2611.22287, the same family as our head): `idea.md`.

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

Step 2, the LoRA arm on ama mitsuki (the user, 2026-11-21: "그래 차라리 그게 낫겠다. ama mitsuki 로
해보자").

Facts (the design, set before the runs):

- Code: `library/training/pixel_lpips.py` (`--pixel_lpips_ratio` / `--pixel_lpips_weight` /
  `--pixel_lpips_measure`, σ gate 0.7, window 24 + 2 × 4 tokens), test `tests/test_pixel_lpips.py`.
  Per sample at σ ≤ 0.7: LPIPS-VGG of the bf16 VAE decode of x_t − σ·v on one random window against
  the same decode of x0; the term's gradient into v is scaled per sample to `ratio` × the FM
  gradient's norm.
- Arms, configs in `configs/` (linked into `configs/gui-methods/custom/`), seed 42, ama mitsuki
  (106 images, 424 updates), `configs/methods/lora.toml` as it stands plus `paired_step_rng` and
  `gradient_checkpointing`: `lpctl_1121_1900` (the term measured, not added), `lpr03_1121_1900`
  (ratio 0.3), `lpr10_1121_1900` (ratio 1.0).
- Smokes: at the default preset the first VAE decode ran out of memory with 14.34 GB allocated
  (job 20261121-185603); with gradient checkpointing 16 updates ran at ~0.86 s per micro-step, the
  term on 26 / 32 batches, λ 0.045 at ratio 1.0 (job 20261121-185722).
- Read: (1) the paired training logs, `pixel_lpips/lpips` at the logged steps of the last two
  epochs; (2) `bench/pixel_readout/lora_ab.py --label lpab_1121_1900`: the lora_recipe prompts
  p0 – p3 × render seeds 42 / 43 at the `make test` recipe, pair sheets control | arm, blind sets
  with the sides shuffled; the user's read decides. ‖dW‖ reported, not matched.
- Bars: spread unknown, one training seed.
- Result (`reports/lpips_lora_2026_11_21.md`): paired training-log LPIPS against the control,
  ratio 0.3 −0.0003 … −0.0007, ratio 1.0 −0.0026 … −0.0043 (control 0.153); ‖dW‖ within 0.7 %;
  render seeds 42 – 45, blind sets `s45_LPIPS03_ama` / `s45_LPIPS10_ama` pushed to
  `anima-blind-pairs` on the user's word. The user's blind read (2026-11-21), control – arm:
  ratio 0.3 5 – 11, ratio 1.0 11 – 5 (16 pairs each, one training seed; p1 7 – 1 to the control
  and p3 1 – 7 to the arm over both sets). Ratio 1.0 is not carried on; ratio 0.3 waits on a seed
  twin (control + ratio 0.3 at seed 43), proposed, not queued.

Interpretation (the reasoning for the arms):

- The ratio has no source: PixelGen's λ 0.1 is on a pixel x-prediction and does not translate, so
  0.3 and 1.0 bracket it. The per-sample ratio rule is Claude's choice over a fixed λ, because the
  term's gradient scales with σ and the FM gradient's norm moves with σ too.
- A visible gain on the user's read → a second training seed, then prompts outside the working set.
  Level or worse at both ratios → the term is parked; the ratio and the σ gate are the knobs left.

### The ×1 recipe: spread, length, LR, σ range, page share (batch `1121_2230`)

The user (2026-11-21): "x1같은경우는 이미 어느정도 학습할만큼 했다고 볼수가 있나? robust 할만하다면
하고싶긴 한데 현재 배포본이 '완성' 레벨인지 궁금하긴 하네. … 오늘 새벽에 학습 돌리는걸 x1이랑 x2로
예약해두고 싶어서"; on the four arms proposed: "ㅇㅋ 그러자"; then, on the pages: "망가는 아니마
도메인하고 좀 멀어서 비중을 낮춰야 하지 않을까 싶어" (the fifth arm).

Facts (the design, set before the runs):

- Parent: the release head's run `realr_1119_2200` (`convnext_head.py`, init `mixr_1119_1455`,
  points `0.0577-0.1364:real 0.5:adapter`, λ_lpips 0.05, pages 0.33, LR 2e-4, 3,000 rollouts,
  seed 0; 36 minutes). Train set 2,981 images and 3,000 pages.
- Arms, each the parent's argv with one change, queued in this order:

  | run | change |
  |---|---|
  | `x1twin_1121_2230` | `--seed` 0 → 1 |
  | `x1wide_1121_2230` | base range 0.0577 – 0.1364 → 0.03 – 0.25 |
  | `x1page_1121_2230` | `--page_share` 0.33 → 0.1 |
  | `x1lowlr_1121_2230` | `--lr` 2e-4 → 5e-5 |
  | `x1long_1121_2230` | `--rollouts` 3000 → 12000 (`--eval_every` 1000) |

- Changed in all five against the parent's command: `--eval_points` (new, read-only σ), so each
  head is read at σ 0.03, 0.0577, 0.097, 0.1364 (0.14 on `x1wide`), 0.25 and the turbo point.
  Smoke of the flag: `smoke_x1night_1121_2230` (20 rollouts), first in the queue.
- σ 0.03 … 0.25 is the last step of shift-3 Euler at 100 … 10 steps (0.029 / 0.25).
- The values 5e-5, 12,000, 0.1 and the range have no source in the records: the page share was
  run at 0 (`mangaup_1119_1152`) and 0.33 only; `reports/real_target_2026_11_19.md` names a lower
  peak LR for warm starts without a value.
- Records they rest on (`reports/real_target_2026_11_19.md`): `realr` against the original at
  σ 0.0577, test / text / manga: 33.75 / 0.081, 28.61 / 0.067, 26.02 / 0.102 (dB / LPIPS);
  VAE(x̂0) there 35.10 / 0.056, 30.27 / 0.038, 26.95 / 0.062; 1,000 → 3,000 rollouts +0.2 dB;
  the eval history under its start until past rollout 1,500 (LPIPS 0.104 → 0.205 → 0.115).
- Read: (1) held-out, teacher-forced, test 30 / text 8 / manga 10 at every eval point, each arm
  against `realr` and `x1twin`; (2) generated images, P3's 16 prompts × seeds 0, 1, every head on
  the same renders against the VAE decode: `x1b28_1121_2230` (base 28 steps, CFG 4),
  `x1turbo_1121_2230` (v4_longer, 4 steps), and on `realr` / `x1twin` / `x1wide` only
  `x1s10_1121_2230` (10 steps, σ 0.25) and `x1s100_1121_2230` (100 steps, σ 0.029); (3) sheets
  for the user's read, which decides.
- Bars: the `realr` / `x1twin` gap at the shared points (σ 0.0577, 0.097, 0.1364, turbo). At
  σ 0.03 and 0.25 only `x1twin` holds the parent recipe: spread unknown there.
- Cost: 4 × 36 minutes + about 2.3 hours + the four reads, about 5.5 hours.
- Smoke `smoke_x1night_1121_2230` (20 rollouts, rc 0, 8 minutes; held-out features 288 in
  164 s): the flag reads the two extra points. Its rollout-0 eval is `mixr_1119_1455` as loaded
  (target VAE(x̂0)), head against VAE(x̂0), test 30 / manga 10: σ 0.03 36.43 / 30.21, 0.0577
  38.05 / 31.30, 0.097 38.47 / 31.63, 0.1364 38.46 / 31.55, 0.25 36.77 / 30.43.

Interpretation (the reasoning for the arms):

- `x1long`: does more of the same recipe raise the head? A gain past the twin gap → the release
  recipe was short and the next head is a long run; level → the recipe is at its plateau and what
  is left (screentone, the 1.35 dB to the VAE path at low σ) needs another lever.
- `x1lowlr`: half of the parent's run went to recovering from the warm start; level with or above
  the parent at a quarter of the LR → warm starts take the lower LR, and a long run pairs with it.
- `x1wide`: what does the release recipe read outside its range (`x1twin` at σ 0.03 and 0.25),
  and what does covering the range cost inside it? Off-range gain with the in-range rows inside
  the twin gap → the next head trains on the wide range.
- `x1page`: the pages are real manga pages under one fixed caption, far from what Anima renders.
  Colour and generated-mono rows level or up with the manga rows down → the share drops; the
  generated mono renders, not the held-out pages, are the read that matters for it.
- The arms are single-variable on purpose; a head that combines the winners is a later run.

#### Outcome (2026-11-22)

Facts:

- All ten jobs rc 0. Results (local): `bench/pixel_readout/results/*-x1{twin,wide,page,lowlr,long}_1121_2230/`,
  `*-x1{b28,turbo,s10,s100}_1121_2230/`; sheets and renders
  `bench/pixel_readout/results/20261122-0610-night_read/`, `20261121-2300-x1night_renders/`.
  0.66 – 0.75 s per rollout; `x1long` 135 minutes.
- Held-out, teacher-forced, the head against the original image at σ 0.0577, dB / LPIPS:

  | run | test 30 | text 8 | manga 10 |
  |---|---|---|---|
  | `mixr_1119_1455` (the init, rollout 0) | 33.94 / 0.1036 | 28.88 / 0.0739 | 26.21 / 0.1168 |
  | `realr_1119_2200` (parent, release) | 33.75 / 0.0807 | 28.61 / 0.0671 | 26.02 / 0.1020 |
  | `x1twin` | 33.78 / 0.0794 | 28.65 / 0.0661 | 26.04 / 0.1016 |
  | `x1wide` | 33.54 / 0.0837 | 28.42 / 0.0716 | 25.88 / 0.1056 |
  | `x1page` | 33.91 / 0.0758 | 28.62 / 0.0633 | 25.98 / 0.1011 |
  | `x1lowlr` | 33.97 / 0.0769 | 28.87 / 0.0634 | 26.22 / 0.1022 |
  | `x1long` | 34.13 / 0.0741 | 29.06 / 0.0596 | 26.32 / 0.0962 |
  | VAE(x̂0) | 35.11 / 0.0560 | 30.29 / 0.0375 | 26.95 / 0.0618 |

- The `realr` / `x1twin` gap, the largest over the four shared points: test 0.06 dB / 0.0013,
  text 0.05 dB / 0.0010, manga 0.03 dB / 0.0012.
- Off the trained range (`x1twin` holds the parent recipe; one run, spread unknown), test 30,
  the head against the original, dB / LPIPS, with VAE(x̂0) beside it:

  | σ (Euler steps) | `x1twin` | `x1wide` | `x1long` | VAE(x̂0) |
  |---|---|---|---|---|
  | 0.03 (100) | 33.88 / 0.0777 | 34.00 / 0.0799 | 34.29 / 0.0717 | 37.04 / 0.0412 |
  | 0.0577 (50) | 33.78 / 0.0794 | 33.54 / 0.0837 | 34.13 / 0.0741 | 35.11 / 0.0560 |
  | 0.1364 (20) | 31.18 / 0.1038 | 31.08 / 0.1063 | 31.31 / 0.0997 | 31.33 / 0.0916 |
  | 0.25 (10) | 27.92 / 0.1531 | 28.12 / 0.1469 | 27.94 / 0.1461 | 27.88 / 0.1378 |

  Against VAE(x̂0), test: `x1twin` 35.62 / 37.24 / 37.56 / 36.79 dB at σ 0.03 / 0.0577 / 0.1364 /
  0.25; `x1wide` 35.88 / 36.75 / 36.90 / 36.23.
- `x1long`'s eval history at σ 0.0577, test dB / LPIPS: 33.94 / 0.104 at rollout 0, 32.03 / 0.176
  at 1,000, 32.82 / 0.143 at 3,000, 33.46 / 0.102 at 6,000, 33.88 / 0.083 at 9,000, 34.12 / 0.075
  at 11,000, 34.13 / 0.074 at 12,000. `x1lowlr`: 33.49 / 0.149 at 500, 33.51 / 0.102 at 1,000,
  33.97 / 0.077 at 3,000.
- Generated images, 16 prompts × seeds 0 and 1, every head on the same latents, PSNR of the head
  against the VAE decode (dB); colour 26 / mono 6 split for the first two:

  | read | `realr` | `x1twin` | `x1wide` | `x1page` | `x1lowlr` | `x1long` |
  |---|---|---|---|---|---|---|
  | base 28 steps, CFG 4 | 31.84 | 31.87 | 31.47 | 32.00 | 32.25 | 32.53 |
  | … colour / mono | 33.38 / 25.14 | 33.39 / 25.31 | 32.99 / 24.93 | 33.64 / 24.92 | 33.74 / 25.79 | 34.12 / 25.62 |
  | turbo v4_longer, 4 steps | 33.53 | 33.55 | 33.27 | 33.74 | 33.81 | 34.24 |
  | … colour / mono | 34.54 / 29.15 | 34.56 / 29.20 | 34.27 / 28.92 | 34.82 / 29.05 | 34.82 / 29.41 | 35.25 / 29.85 |
  | base 10 steps (σ 0.25) | 30.22 | 30.25 | 29.75 | — | — | — |
  | base 100 steps (σ 0.029) | 30.21 | 30.36 | 30.32 | — | — | — |

  The twin gap on these: 0.04, 0.03, 0.03 and 0.15 dB; on the six mono images at 28 steps 0.17 dB.
- Renders (`night_renders.py`, base 28 steps, CFG 4, one latent per prompt and seed): the card's
  monochrome prompt (seed 42) and the fine-stroke prompt of `x2_plan.md` (seeds 42, 43) through
  `mixr`, `realr` and the five arms. Sheets `x1_stroke_s42.png`, `x1_stroke_s43.png`,
  `x1_mono_shorts.png`.
- Claude's read of those sheets, labelled: on both fine-stroke seeds every ×1 head, `mixr` and
  `x1long` included, gives the strokes merged into a soft dark area where the VAE gives separate
  strokes; no arm stands apart from the others at 4× zoom. On the monochrome crop the heads give a
  fainter dot screen than the VAE; `mixr` the faintest.
- No read by the user yet.

Interpretation:

- `x1long`: past the twin gap on every row (test +0.39 dB / −0.0066 LPIPS against 0.06 / 0.0013).
  The release recipe was short. At LR 2e-4 the long run's dB stayed under the init's until
  about rollout 9,500 (its LPIPS passed the init's by 6,000), so the length bought less than it
  could.
- `x1lowlr`: above the parent on every row at the same length, with a dip a quarter as deep. Warm
  starts take the lower LR. The pair to run next is the two together (12,000 rollouts at 5e-5):
  neither arm reads whether the gains add.
- `x1wide`: inside the range it loses more than the twin gap; off the range it gains 0.1 – 0.2 dB
  against the original and reads lower against the VAE decode on the 10-step renders. The
  release recipe already degrades smoothly off its range (no cliff at 10 or 100 steps), so the
  wide range is not taken up.
- `x1page`: the held-out manga rows moved by about the twin gap (−0.04 dB against 0.03, LPIPS
  level), the colour rows rose (test +0.16 dB, −0.005 LPIPS; generated colour +0.26 dB), and the
  generated mono renders read 0.22 dB under the parent against a twin gap of 0.17 dB on those
  six images. Dropping the share gives a little on colour and costs at most the edge of the
  spread on mono; whether mono lines look the same is the user's read of `x1b28`'s manga sheets.
- The fine strokes did not move with any arm, the VAE(x̂0)-target head `mixr` included, so the
  blur is not the real-image target's and not the run length's. What is left is the head's form:
  every nonlinear layer runs at the 16-px token spacing and one linear layer writes the 16 × 16
  patch. The check before any long run is whether this form can fit such strokes at all: a
  16-window overfit on fine-stroke crops with the VAE decode as target, the ×1 form beside one with a
  pixel-shuffle stage (`bench/pixel_readout/x2_overfit.py` is the ×2 version of this probe).

#### The long run at the lower LR, `x1ll_1122_0855` (postponed by the user)

The user (2026-11-22): "x2는 어차피 x1에서 init한 형태이니까, x1를 먼저 5e-5로 돌려보고 x1를
확정한다음 x2도 longer and better recipe (b only -> gan 마무리) 로 하면 좋겠어."

Queued at 08:53 and cancelled a few minutes in, with its three read jobs, on the user's word
("longer은 나중에 돌릴거라 일단은 취소해줘"); no result. The design stands for the later run,
which takes a new name.

Facts (the design, set before the run):

- `x1long_1121_2230`'s argv with `--lr` 2e-4 → 5e-5, nothing else changed (12,000 rollouts, init
  `mixr_1119_1455`, pages 0.33, seed 0, the same eval points). It is equally `x1lowlr_1121_2230`
  with 3,000 → 12,000 rollouts, and is read against both.
- Records it rests on: the Outcome table above (`x1long` 34.13 / 0.0741, `x1lowlr` 33.97 / 0.0769,
  `realr` 33.75 / 0.0807 on test at σ 0.0577); `x1lowlr`'s history (33.51 / 0.102 at rollout
  1,000, where `x1long` read 32.03 / 0.176).
- Read: the held-out evals at every point beside `x1long` and `x1lowlr`; the generated-image
  reads `x1llb28_1122_0855` and `x1llturbo_1122_0855` (`realr`, `x1lowlr`, `x1long`, this head on
  the same latents); the monochrome and fine-stroke renders through this head. The user's read
  of the sheets decides the ×1 head.
- Bars: the `realr` / `x1twin` gap (test 0.06 dB / 0.0013 LPIPS, 3,000-rollout runs). No twin of
  a 12,000-rollout run: spread unknown at that length.
- Not in this run: the page share stays 0.33 (`x1page` read level on mono, a little up on
  colour; the user has not decided it).
- Cost: 12,000 rollouts at 0.66 – 0.75 s, about 2 h 15 min, and about 25 minutes of reads.

Interpretation (what each answer leads to):

- Above `x1long` past the twin gap → the gains add; this head is the candidate for the release
  and for the ×2 sequence's init (`x2_plan.md`, "The sequence after the ×1 head is fixed").
- Level with `x1long` → length is the lever and the LR does not matter at that length; either
  head is the candidate, on the user's read.
- Below `x1long` → the lower LR runs out of schedule on a long cosine; `x1long` stands.

### A head at every step

Spectrum's cached steps (σ 0.90 … 0.265) read a Chebyshev-forecast feature at the tap, and
cjk_rl arm D reads x̂0 at steps 0 – 1; both want a head over σ in [0, 1]. `mixr_1119_1455`
covers σ 0.058 … 0.136 on base and 0.5 on turbo. Use: a preview at every step.

### MOSS

A head on MOSS-Image (different DiT, 128-channel latents at /16, the same 16-px token); decode
memory is the motive there (the fp32 VAE decode does not fit beside the DiT). Nothing run.

## Open

- Screentone: small dots smoothed into soft shading, a systematic difference from the VAE (the
  user's read on `mixr_1119_1455`, good enough to release). Taken up under P5, pixel losses on
  the head.
- Tone: the head is darker than the VAE on textured regions (the user's read; `tone_probe.py`,
  `reports/p4_mix_2026_11_19.md`): about 2.5 levels in linear light on screentone-like blocks of
  mono renders on the base DiT, 1 on turbo, under 1 level over a whole image; flat regions level.
  The loss has no term on local tone. Same P5 topic.
- Spectrum renders of `mixr_1119_1455` have numbers only (34.37 / 26.00); no sheet was made.
- The head's flat grey on screentone carries a faint regular weave at the patch scale (Claude's
  read, labelled, base 20 / 50 `manga_page` crops); not measured.
- Speed lines (강조선) slightly soft, footnote-size glyphs blur, under zoom only (the user's read
  and Claude's on `pagemix_1119_1242`).
- The P3a generations (6 mono, 26 colour) not re-run on the pagemix head (`p3_generated.py`).
- Training on sampled final latents: in reserve (P3a did not break on generated images).
- The DiT GEMMs run under bf16 autocast (the stream is fp32 since `replay_1119_1225`); whether
  that rounding costs readout PSNR is unchecked.
- The compiled head (`torch.compile(head, dynamic=True)`) hits `recompile_limit` 12 on image
  shape and runs eager after that, in every run of the replay recipe
  (`reports/p4_mix_2026_11_19.md`); speed only.
- No spread for any trained head: one seed per run. A seed twin of the recipe is owed before a
  gap under about 1 dB is read as a gain. P3's seed-to-seed gap of one prompt (mean 1.98 dB) is
  content variation between images, not a run spread.
- Grey tint on mono generations (Claude's read on `p3_1119_1056`): on teacher-forced greyscale
  pages the pagemix head adds about 0.5 level of chroma over the VAE (`reports/pagemix_2026_11_19.md`);
  generated mono images are not re-checked.
