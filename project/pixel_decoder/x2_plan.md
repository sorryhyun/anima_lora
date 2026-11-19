# ×2 output — plan (2026-11-19)

The P5 topic "×2 / ×4 output" of `plan.md`, as its own plan: decode and 2× upscale in one
readout of the DiT's last features, so a 1024-tier render leaves the head at 2048. Runs and
probes: `reports/x2_arm_a_2026_11_19.md`. Code: `bench/pixel_readout/x2_head.py` (head and
training), `x2_overfit.py` (fit probe); results under `bench/pixel_readout/results/` (local).

## State

### Facts

- The form: the release head `mixr_1119_1455` plus a second 2× stage (512 → 4 × 256 pixel
  shuffle, ConvNeXt blocks at 4-px spacing, an 8×8 RGB patch per position, 32×32 output pixels
  per token), default init, no skip. The ×1 output stays, so one file decodes at ×1 and ×2.
  Head time, ×1 and ×2 together: 12.9 ms on 80 × 52 tokens (2560 × 1664 output), eager.
- Arm A, `x2a_1119_1802` (target: the released ResShift ×2 student on VAE(x̂0), recipe loss,
  1,000 rollouts). PSNR against the teacher's output, dB, test 12 / text 8 / manga 4, one seed:

  | | ×2 head | chain: release head ×1 → teacher (`x2chain_1119_1832`) | bicubic ×2 of VAE(x̂0) | second teacher draw |
  |---|---|---|---|---|
  | base σ 0.097 | 30.70 / 27.25 / 26.49 | 33.71 / 29.55 / 28.17 | 29.82 / 28.25 / 27.00 | 42.45 / 42.72 / 45.91 |
  | turbo σ 0.5 | 29.96 / 26.50 / 26.11 | 32.66 / 28.24 / 27.57 | 29.28 / 27.93 / 27.01 | 42.52 / 42.57 / 44.57 |

- The ×1 output against VAE(x̂0) went from the parent's 37.44 / 32.67 / 32.77 (base) to
  36.37 / 31.58 / 31.68 over the run.
- Fit probes: on 16 train windows every trainable arm stops at 32.4 – 33.3 (base point; 2 → 6
  blocks 32.56 → 32.38, width 256 → 512 with 6× the parameters 33.25, frozen base 29.73 against
  bicubic 30.39); on 2 windows the same stage reaches 37.59 and 41.75 (width 512) against a second
  teacher draw at 42.28.
- Tone, read off the whole-image sheets in 8-bit levels, candidate minus the teacher on VAE(x̂0)
  (inline measurement, 4 – 8 sheets per set, not `tone_probe.py`, whose sheet layout differs):

  | | whole-image luma | pixels the teacher has under luma 64 | chroma range (max − min channel) |
  |---|---|---|---|
  | ×2 head, base test / text / manga | +0.27 / −0.06 / +0.01 | +7.92 / +9.30 / +5.22 | −0.75 / −0.30 / −0.06 |
  | ×2 head, turbo | +0.34 / +0.03 / −0.01 | +6.16 / +9.49 / +5.80 | −0.76 / −0.38 / +0.02 |
  | chain, base | +0.17 / −0.20 / −0.03 | +3.40 / +4.79 / +3.74 | −0.20 / −0.13 / −0.26 |
  | chain, turbo | +0.21 / −0.12 / −0.06 | +2.69 / +5.59 / +4.13 | −0.27 / −0.21 / −0.20 |

- The user (2026-11-19), on skipping the chain sheet before arm A: "1번은 어느정도 자명해서
  건너뛰어도 될듯? resshift x2 배포본은 톤 차이까지 다 깐깐하게 본거라".
- The user (2026-11-19), while the fit probe ran: "아니면 resshift에서 gan 구조 썼는지는 모르겠는데
  gan 쓰면 좀 도움되지 않을까?" The ResShift distillation used one (a DMD2-style head on the fake
  critic's bottleneck, `--lambda_gan 3e-3`, with VGG-LPIPS at `--lambda_lpips 2.0` and a tone term
  `dc_loss`, L1 on 32-px block means; `project/finished/sr/distill_rsd/train.py`).
- The user's read (2026-11-19, on the `x2a_1119_1802` and `x2chain_1119_1832` crops): "톤이 좀
  바뀌긴 하는거같아서 lpips나 tone loss가 있긴 있어야겠군".
- Claude's read (labelled, same crops): the ×2 head's lines are thicker and softer than the
  teacher's; the chain's are as crisp; washed iris detail and smeared dense glyphs show in both.
- Supply for a real-pixel target: of 3,242 images in `post_image_dataset/resized`, the master in
  `image_dataset/` is at least 2× the resized side on 1,518 (46.8 %), at least 1.9× on 1,718;
  whether a master is itself an upscale or a JPEG is unchecked.

### Interpretation

- The level to read a ×2 head against is the chain, not the teacher's second draw. The target is
  a function of VAE(x̂0), and the head knows VAE(x̂0) only to its ×1 decode error, which grows
  through the teacher. Arm A's design assumed otherwise.
- What the ×2 stage lacks is edge sharpness: 1.7 – 3.0 dB under the chain, soft where the chain is
  crisp. The stage can express the teacher's output (2 windows) and does not learn it as a mapping
  under the recipe loss at this size (16 windows; width and depth buy little). In SR distillation
  that gap is what a perceptual or adversarial term carries, which is the user's suggestion.
- The tone the user saw is local, not global: whole-image luma moves under 0.5 level, dark
  pixels are lifted 5 – 9 levels on the ×2 head and 3 – 6 on the chain. Softened dark lines read
  lighter, so part of the lift is the softness itself and part comes with the ×1 decode; the two
  are not separated. LPIPS does not pin tone (the SR line added `dc_loss` for the tint LPIPS left
  free), so a tone term is its own item.
- Washed iris detail and smeared glyphs come from the ×1 decode and belong to `plan.md`'s "pixel
  losses on the head".
- A ×2 stage beside an untouched release head is not available in this form (frozen base under
  bicubic), and training the base cost the ×1 output about 1 dB.

## Arm A — the design as written before the run

Moved from `plan.md`. Parent `mixr_1119_1455` (points `0.0577-0.1364 0.5:adapter`, pages 0.33,
replay 64 × 16 updates, LR 2e-4, fp32 stream). Changed together: the second 2× stage; its
target, `rsd_student_18k` at the ComfyUI node's defaults on VAE(x̂0), recipe loss; windowed
training, one 30 × 30-token window per rollout with the loss on the inner 24 × 24 away from image
edges and the teacher run on the window only. Question: does the ×2 output read level with
VAE → ResShift ×2? Level → promote the stage and read arm B against it; soft → capacity before
arm B; ×1 under the parent → a frozen base. Bars: spread unknown. Smoke `smoke_x2` gave the
reference levels (bicubic 31.86 / 28.25, second draw 46.65 / 40.36, 2 test / 8 text images).

Outcome: soft, and ×1 under the parent; the capacity and frozen-base branches were both read by
the fit probes (above). No user verdict was asked on arm A's sheets beyond the tone read.

## Arms 1 + 2 together — tone term and LPIPS (`x2lt_1119_2013`, done)

The user (2026-11-19): "1, 2 같이 적용해도 될듯? lpips나 tone term 모두 이미 resshift든 moss든
검증된거라. 다만 hyperparameter만 가볍게 보고 본 학습 돌리자".

Design: `x2a_1119_1802` with two terms added on the ×2 output against the teacher, nothing else
changed (window 24, margin 3, 1,000 rollouts, LR 2e-4, points `0.0577-0.1364 0.5:adapter`, pages
0.33, init `mixr_1119_1455`): `--lambda_dc` (L1 on 32-px block means, `dc_loss` of the SR line)
and `--lambda_lpips` (VGG-LPIPS). The two cannot be separated in this run. Question: do the ×2
head's lines and dark tone come up to the chain's? Yes → the ×2 stage is a candidate for
promotion, read by the user on crops. No → arm 3 (GAN).

Sizing, `smoke_x2lp` (λ 0 with both terms in the graph, 150 rollouts): unweighted gradient norms
over the head, rollouts 50 / 100 / 150: rec2 0.38 / 0.31 / 0.33, rec1 0.29 / 0.24 / 0.35, LPIPS
7.1 / 7.1 / 16.0, dc 0.57 / 0.48 / 0.38. 1.71 – 1.78 s per rollout against arm A's 1.09.
λ_dc = 1.0 (the SR line's value; its gradient is about rec2's). λ_lpips sweep at 250 rollouts,
`x2lp{001,003,01}_1119_1944`: 0.01 / 0.03 / 0.1, about 0.3× / 1× / 3× rec2's gradient. The
SR line's 2.0 sat beside a DMD loss and does not carry over. The full run takes the λ whose
crops are crisp without artefacts, read against `x2ref_1119_1908` (arm A and the chain on the
new meters: LPIPS against the teacher, dark-pixel luma lift).

Void runs: `smoke_x2lp` and the first sweep (`x2lp{001,003,01}_1119_1919`). The loss sum was
named `total`, which shadowed the cosine schedule's step count, so `set_lr` computed
cos(π·step / loss). After warmup the LR multiplier landed anywhere in 0 … 1 each step, with no
decay. Fixed (`loss`, `total_steps`). The smoke's gradient-norm ratios are kept for sizing
only. Arm A and `x2ref_1119_1908` predate the change or do not train.

Sweep `x2lp{001,003,01}_1119_1944` (250 rollouts, test 4 / text 8 / manga 2 per point). Base
point against the teacher, PSNR test / text / manga; LPIPS test / text / manga; dark lift
test / text:

| λ_lpips | dB | LPIPS | dark lift | ×1 test |
|---|---|---|---|---|
| 0.01 | 30.20 / 26.45 / 25.21 | 0.176 / 0.241 / 0.271 | +14.1 / +12.3 | 38.28 |
| 0.03 | 30.10 / 26.38 / 25.15 | 0.167 / 0.232 / 0.260 | +14.6 / +12.5 | 38.28 |
| 0.1 | 29.91 / 26.24 / 25.02 | 0.161 / 0.218 / 0.245 | +15.6 / +13.2 | 38.17 |

The same images in `x2ref_1119_1908`: arm A (1,000 rollouts) 32.12 / 27.25 / 25.90, LPIPS
0.135 / 0.204 / 0.224, dark lift +8.0 / +9.5. The turbo point moves the same way. On Claude's
labelled crops (test 00, text 01), the three arms cannot be told apart, and none shows an
artefact. The full run takes λ_lpips 0.1 (the lowest LPIPS, clean on the crops); the dark lift
grows with λ and is read there.

Read: the full run's two-column crops `chain | ×2 head` (`--chain`), test 12 / text 8 / manga 4 at
both points, labelled; numbers beside them: dB, LPIPS and dark lift against the teacher, the ×1
output against the parent. The user's read decides. Bars: spread unknown.
Cost: sweep 3 × ~11 min, full run ~40 min.

Outcome, `x2lt_1119_2013` (λ_lpips 0.1, λ_dc 1.0, 1,000 rollouts, 1.39 s per rollout, peak
9.2 GB; head 11.8 ms on 80 × 52 tokens). Against the teacher, test 12 / text 8 / manga 4:

| | dB | LPIPS | dark lift | ×1 against VAE(x̂0) |
|---|---|---|---|---|
| ×2 head, base σ 0.097 | 29.83 / 26.84 / 26.10 | 0.134 / 0.161 / 0.172 | +13.4 / +11.4 / +6.4 | 36.21 / 31.48 / 31.55 |
| chain, base | 33.71 / 29.55 / 28.17 | 0.152 / 0.100 / 0.100 | +5.7 / +4.9 / +3.8 | |
| ×2 head, turbo σ 0.5 | 29.09 / 26.15 / 25.80 | 0.135 / 0.165 / 0.174 | +12.9 / +11.3 / +7.0 | 34.87 / 30.40 / 30.58 |
| chain, turbo | 32.66 / 28.24 / 27.57 | 0.148 / 0.109 / 0.105 | +5.9 / +5.7 / +4.2 | |

Bicubic: 29.82 / 28.25 / 27.00 (base), 29.28 / 27.93 / 27.01 (turbo). Claude's read, labelled,
two crops (`base_s0_09705_test_00`, `…_text_01`): the ×2 head's dark regions (hair, a bow) are
lighter than the chain's and its lines softer; the glyphs are legible in both. The user has not
read this run's crops.

Interpretation: LPIPS against the teacher went under the chain's on the test set and stayed far
above it on text and manga; the tone term at 1.0 did not hold the dark pixels (+13 against arm
A's +10), and PSNR fell to bicubic's level. Arm 3 (GAN) stays the next arm on sharpness.

## ×2 preview — release plan (2026-11-19)

### Facts

- The user, on replacing the release head with the real-target head
  (`reports/real_target_2026_11_19.md`): "릴리스 헤드를 바꿔도 되긴 하는데 흠... x2 preview 느낌으로
  릴리스하는건 어때? x2를 12ms만으로 만드는건 꽤 매력적이라". Then: "본런이랑 x2 스테이지 돌리고,
  라이브러리랑 comfyui 노드 배포는 x2_plan.md 에 적어놓자".
- The user, asked whether `mixr_1119_1455` stays as the default beside a preview file: "본런
  결과 보고 정하지 뭐 옆에 두기보단 교체하는 느낌으로 가자". The new head replaces the release head
  (`anima_pixel_head` in the catalog, the Hub file, the node's default) once the user has read
  run 1; it is not shipped beside it.
- The new head: ×1 on the real-target recipe with the turbo point on
  `anima_turbo_4step_v4_longer`, and the ×2 stage on it.
- Open until the read: whether the replacement is run 1's ×1 file with the ×2 file as a second,
  preview file, or run 2's one file for both scales. `x2a_1119_1802` moved the ×1 output about
  1 dB while the ×2 stage trained; run 2's ×1 meters against run 1's say which.
- Run 1, `realr_1119_2200` (×1): `real_1119_2106` with `--rollouts` 1000 → 3000, nothing else
  changed (`--init mixr_1119_1455`, points `0.0577-0.1364:real 0.5:adapter`, `--lambda_lpips
  0.05`, adapter v4_longer). About 45 min.
- Run 2, `x2real_1119_2242` (×2 stage): `x2lt_1119_2013` with
  `--init` → run 1's head and `--adapter` → v4_longer, plus `--points 0.0577-0.1364:real` and
  `--lambda_lpips_x1 0.05`: the ×1 output's loss at the base points is run 1's (the original
  image + LPIPS) instead of VAE(x̂0). The ×2 target is unchanged: the teacher on VAE(x̂0). About
  40 min. `x2_head.py` gained the `:real` tag, `--lambda_lpips_x1` and ×1-against-original
  meters; `smoke_x2real_1119_2205` (20 rollouts on `real_1119_2106`) checks the path first.
- What run 2 cannot separate from `x2lt_1119_2013`: the base head, the adapter and the ×1 loss
  all change.

### Interpretation

- Without the ×1 change, run 2's ×1 term would pull the base head back onto VAE(x̂0) and undo
  what the user read on run 1's recipe.
- The ×1 output aims at the original and the ×2 output at the teacher's upscale of VAE(x̂0), two
  slightly different images; at the base points they are about 35 dB apart (VAE(x̂0) against the
  original, σ 0.058), so the two outputs of one file may differ in tone. A ×2 target on the
  original (the master at 2×, 1,518 images) is arm 4 below and not in the preview.

### Read before deployment

Claude's draft. The run's own crops `chain | ×2 head` (test 12 / text 8 / manga 4, both points,
labelled) and the numbers above against `x2lt_1119_2013` and the chain; then renders through the
×2 head, which no run has made yet: P3's 16 prompts on base 28 steps CFG 4 and on v4_longer
4 steps, `VAE → ResShift ×2 | ×2 head`, with end-to-end time (the teacher is 3.16 s per image).
`p3_generated.py` reads ×1 heads only and needs a ×2 path for this. The user's read decides;
bars: spread unknown.

### Deployment: library

Claude's draft, after the read passes. It follows the ×1 promotion (`plan.md` §2).

- `library/anima/pixel_head.py`: `PixelHeadX2` moves in from `bench/pixel_readout/x2_head.py`
  (the bench script imports it back). `load_pixel_head` returns it for a file stamped
  `x2_depth` (keys `base.*`, `up2_*`, `out2.*`; `d_feat` from `base.proj.weight`), with the σ
  ranges as now. `PixelTap.decode(head, guidance_scale, scale=1 | 2)`: a ×2 file gives either
  output, a ×1 file refuses `scale=2` with the reason in the message.
- `inference.py --decoder pixel_head --pixel_scale 2` (`library/inference/pixel_decoder.py`),
  `PIXEL=1 PIXEL_SCALE=2` on `test-*` and `test-turbo`, `GenerationRequest(pixel_scale=…)`. The
  saved image is 2× `--image_size`; `save_decoded` and the PNG metadata are checked for an
  assumed size.
- Catalog: `anima_pixel_head` points at the replacement. If the ×2 file ships as a second file,
  a row `anima_pixel_head_x2` (`models/vae/`, fetched on first use), pack `anima-pixel-head-x2`,
  GUI pack strings in four languages, and `--pixel_scale 2` without `--pixel_head` resolves to
  it; if one file serves both scales, no new row. `docs/anima/inference/pixel_head.md` and the
  card's numbers (turbo point, σ table, PSNR against the VAE) are re-read on the new head: it no
  longer regresses onto the VAE decode, so "dB against the VAE" drops by construction.
- Warnings as for ×1 (σ outside the stamp, a Spectrum cached last step, `--cfgpp` /
  `--smc_cfg`); the file's `adapter` stamp (v4_longer) goes into the log line at load.
- Façade: `anima_lora.models.PixelHeadX2`.
- Tier 2: `e2e_speed.py --scale 2` (head against VAE decode → ResShift ×2), a CPU test in
  `tests/test_pixel_head.py` (loader round trip, output shapes ×1 and ×2, the ×1-file refusal),
  a ×2 section in `docs/anima/inference/pixel_head.md`; `docs/anima/experimental/pixel_decoder.md`
  updated from "in progress".
- One check through the daemon as for ×1: base 28 steps CFG 4 and turbo, the ×2 PNG against
  ResShift ×2 of the VAE decode of the saved latent; and one on `--int8`
  (`reports/int8_cond_2026_11_19.md` read ×1 only).

### Deployment: ComfyUI node

Claude's draft. Repo `~/ComfyUI-Anima-PixelDecoder` (GitHub `sorryhyun/ComfyUI-Anima-PixelDecoder`,
registry `comfyui-anima-pixel-decoder`, 0.1.0 now); publish flow in the `custom-nodes` skill.

- `make vendor-sync` rewrites `_vendor/pixel_head.py` from the library (`build_pixel_vendor`);
  no hand copy.
- `AnimaPixelDecode` takes `scale` (1 / 2). 2 needs a ×2 head file and returns the IMAGE at 2×
  the latent's pixel size; with a ×1 file it raises with the reason. `AnimaPixelHeadTap` is
  unchanged (same tap, same features).
- Head files stay in `models/pixel_heads/` (or a VAE folder under a name containing
  `pixel_head`); the README names the ×2 file and that it also decodes ×1.
- `bench/pixel_readout/comfy_e2e.py --scale 2`: turbo (v4_longer) in ComfyUI, VAE Decode →
  the ResShift ×2 node against Anima Pixel Decode at scale 2, time and PSNR / LPIPS; record in
  `reports/`. Still not run in ComfyUI from 0.1.0: the Spectrum KSampler, the base DiT at CFG 4.
- Version 0.2.0: README (a ×2 sample beside the ×1 ones), `pyproject.toml`, registry publish,
  GitHub push.

### Deployment: Hub

Claude's draft. `sorryhyun/anima-pixel-decoder` (the user edits the card): the replacement
under the release head's file name, with `mixr_1119_1455` kept under `old/` (it is the head
for the Hub turbo repo's `old/` file); the ×2 file as `anima_pixel_head_x2_preview.safetensors`
if it ships separately; and a "×2 preview" card section stating what the file holds (×1 on the original-image target + LPIPS, ×2 stage on the
ResShift ×2 student's output, turbo point trained on `anima_turbo_4step_v4_longer`), the
measured time, and the limits read at that point (tone of dark regions, line softness). Upload
on the user's word.

## Next arms

Claude's drafts, none queued. Each is a micro arm on arm A's recipe (about 25 min at 1.09 s per
rollout) unless noted, read on the same held-out set as two-column crops `chain | ×2 head` at the
same windows as arm A's, labelled; the user's read decides. Numbers beside them: ×2 against the
teacher and against the chain's level, the ×1 output against the parent's, the tone table above.
Bars: spread unknown; a seed twin of `x2a_1119_1802` is owed before a gap under about 1 dB is
read as a gain.

1. Tone term. `x2a_1119_1802` with an L1 on block means of the ×2 output against the teacher's
   (`dc_loss`, 32-px blocks at ×2), nothing else changed. Question: does the dark-pixel lift
   close without a sharpness term? Its weight is set from the loss ratio on a smoke at the run's
   own settings.
2. Perceptual term. Arm 1's recipe plus VGG-LPIPS on the ×2 output. Question: do the lines
   come up to the chain's? `lpips` is not in the venv (the `sr` dependency group); the weight
   starts from the SR line's 2.0 and is set the same way as arm 1's.
3. GAN on the ×2 output (the user's suggestion), if arm 2's lines stay soft. Discriminator
   candidates: FastFrames' DINOv3 ViT-S/16 heads (`plan.md`, "pixel losses on the head"), the
   turbo line's disc code in git history. Hour-scale to build; proposed again when arm 2 reads.
4. Target variants, each one variable against the best arm above: the teacher on the head's own
   ×1 output, detached (the target becomes a function of what the head holds); real masters at
   the base points (arm B, 1,518 images), which at σ 0.5 would ghost where x̂0 and the master
   disagree.
5. Keeping ×1: the ×1 loss weight, or a lower LR on the base, against the 1 dB drift. Read on the
   ×1 column of any arm above before a separate run.

## Open

- No seed twin, no spread, for any ×2 number.
- 16-window fit: a ceiling or slow convergence is not separated (the cosine schedule ends as
  the curve flattens); the 2-window numbers rest on one window per point.
- Only teacher-forced reads. No turbo or base rollouts through the ×2 head, no end-to-end time
  beside VAE → ResShift (teacher 3.16 s per image against the head's 12.9 ms).
- The dark-pixel lift is not split into softness and tone; `tone_probe.py` does not read the
  two-column sheets.
- The ×2 stage lives in `bench/pixel_readout/x2_head.py::PixelHeadX2`; `library/anima/pixel_head.py`
  is untouched until an arm passes.
- ×4 not started.
