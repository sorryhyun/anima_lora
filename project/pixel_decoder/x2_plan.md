# ×2 output — plan (2026-11-19, updated 2026-11-20)

The P5 topic "×2 / ×4 output" of `plan.md`, as its own plan: decode and 2× upscale in one
readout of the DiT's last features, so a 1024-tier render leaves the head at 2048. Finished
arms are in `reports/`: arm A and its design `x2_arm_a_2026_11_19.md`, arms 1 + 2
`x2_lpips_dc_2026_11_19.md`, the preview release `x2_preview_release_2026_11_19.md`, arm 3
`x2_critic_2026_11_20.md`, the feature probe `feat_probe_2026_11_20.md`. Code:
`bench/pixel_readout/x2_head.py` (head and training), `x2_overfit.py` (fit probe), `x2_dmd.py`
(arm 3's critic), `x2_masters.py` (real 2× targets), `x2_pair.py` (pair sheets); results under
`bench/pixel_readout/results/` (local).

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
- Arm 3 (2026-11-20, `reports/x2_critic_2026_11_20.md`): RSD's DMD and GAN terms on the ×2
  output, each against a control continued from `x2real_1119_2242`; DMD + GAN level with
  its control, DMD alone a little under its control on dB, LPIPS and dark lift. The user
  closed the branch and chose arm B (below), with the turbo point left out of ×2 for now.
- Supply for a real-pixel target: of 3,242 images in `post_image_dataset/resized`, the master in
  `image_dataset/` is at least 2× the resized side on 1,518 (46.8 %), at least 1.9× on 1,718;
  whether a master is itself an upscale or a JPEG is unchecked.

- Run 2 `x2real_1119_2242` (real-target base, v4_longer) reads level with `x2lt_1119_2013` on the
  ×2 output (base 29.74 / 26.70 / 25.94 dB, LPIPS 0.132 / 0.158 / 0.169, dark +13.8 / +11.7 /
  +6.6 against the teacher); numbers in `reports/x2_preview_release_2026_11_19.md`.
- The user's read of run 2 on renders, `VAE → ResShift ×2 | ×2 head | bicubic` (`x2bic_1119_2335`,
  base 28 steps CFG 4, five random prompts, seed 510): "kotatsu에선 bicubic보다 톤을 잘 지키지만 털
  세부 표현은 아쉽고, shop 예시에선 각 책이 bicubic보다 망가진 느낌이 들기도 한다. 하지만 스크린톤
  자체는 bicubic보다 톤을 잘 보존하면서 특유의 질감이 약간 더 나는 느낌이 난다. manga같은 경우 선이
  좀 더 뚜렷해야 할것 같아". On the same renders bicubic is higher than the ×2 head against the chain
  on all five (e.g. kotatsu 21.67 against 17.90 dB).

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
- After arm 3 (2026-11-20): every arm so far asked the ×2 stage for a ResShift model's
  upscaling function, by regression, by LPIPS or by its critic, and none closed the gap to
  the chain. Arm B changes what is asked: the 2× image the latent was encoded from.

## Finished arms

| arm | runs | what changed | verdict |
|---|---|---|---|
| A | `x2a_1119_1802` | a second 2× stage regressed onto ResShift ×2 of VAE(x̂0) | not passed |
| 1 + 2 | `x2lt_1119_2013` | LPIPS 0.1 and a tone term 1.0 on the ×2 output | not passed |
| preview | `x2real_1119_2242` | the ×2 stage on the real-target ×1 head | shipped |
| 3, DMD + GAN | `x2dmd_1120_0848` / `x2ctl_1120_0848` | RSD's critic terms, head LR 5e-5 | level |
| 3, DMD alone | `x2dmdo_1120_0925` / `x2ctl2_1120_0925` | the DMD term, head LR 2e-4 | not passed |

Against the teacher at the base point, test 12 / text 8 / manga 4: arm A 30.70 / 27.25 /
26.49 dB; arms 1 + 2 29.83 / 26.84 / 26.10, LPIPS 0.134 / 0.161 / 0.172, dark lift +13.4 /
+11.4 / +6.4; the preview level with it; arm 3's DMD-only run 29.65 / 26.47 / 25.71, 0.152 /
0.176 / 0.185, +13.7 / +12.0 / +6.6 against its control's 29.76 / 26.54 / 25.75, 0.141 /
0.165 / 0.175, +13.2 / +11.7 / +6.5. The chain: 32.66 / 28.68 / 26.98, 0.081 / 0.077 /
0.088, +8.2 / +6.7 / +4.9. One seed everywhere, spread unknown.

Notes: the preview shipped as a test file on the user's word, not on a pass. In the DMD + GAN
pair the control did not move from its parent at LR 5e-5, so that pair reads neither term.
The ResShift-critic branch was closed on the user's word (2026-11-20).

## Arm B — the real 2× image as the ×2 target (decided 2026-11-20, not built; next action: control / B / B+GAN)

The user (2026-11-20): "resshift 등의 업스케일링 모델은 주로 주어진 입력을 일반적으로 '새로 확대 생성'
하는 방향인거고, 픽셀 디코더는 주어진 feature를 픽셀로 해석하는, vae보다는 좁은 영역을 커버하는거잖아?
근데 무리하게 새로 생성하는 능력을 가르치려고 하는 느낌이 드는데"; after arm 3's sheets: "arm b가
맞다고 보고, 일단 터보는 x2에서 뒤로 미루는걸로 하자".

### Facts

- Supply and alignment, `bench/pixel_readout/x2_masters.py` (`x2masters_1120_1035`, CPU;
  manifest (local) under `bench/pixel_readout/results/`): all 3,242 resized images have a
  master; the 2× target is a downscale of the master on 1,518 (the "native set"), an
  upscale by at most 1.05 on 1,717 and at most 1.25 on 2,156. Master formats in the native
  set: webp 1,419, jpeg 55, png 44; whether a master is itself an upscale is unchecked.
- The 2× target is the resize stage's geometry at exactly twice the cover size and twice the
  crop offset (`x2_masters.target_2x`). On 48 native-set images its LANCZOS ×½ reads 51.5 –
  61.1 dB against the resized image (mean 56.1); the 2×2 box mean 37.9 – 49.4. Doubling the
  bucket in `resize_to_bucket` instead left 35 of 48 at 24 – 40 dB (the cover size and the
  crop offset round by themselves, half a ×1 pixel off).
- On the 12 held-out images composed in `~/x2pair_1120/dmd_vs_ref/` (local; test 4, text
  8) the 2× target's scale against the master is 0.46 – 0.80 on the four test images and
  0.60, 1.00, 1.03, 1.19, 1.47, 1.62, 1.73, 1.83 on the text images.
- Decided by the user: arm B is the next arm; the turbo point is left out of the ×2 work
  for now.
- The user (2026-11-20), on the target supply: "원본이 좀 작은경우만 줄이면 어때? 상한 1024
  하한 768에서 자유롭게. 즉 원본이 1536이면 우리식 리사이즈는 1024가 되니까 충족을 안하는데
  768로 줄이면 2배가 되니까 성립. 2048이면 그대로 1024로, 1024면 512까지 줄여야 하는데 이건
  하한 조건을 충족 못하니 생략"; then: this rule as the default target, and control / B /
  B+GAN as the next action of this plan.
- Tier-flex supply (Claude, 2026-11-20, CPU header scan over the `x2masters_1120_1035`
  manifest, script and per-image manifest in the session scratchpad (local), not in the
  repo): the ×1 input takes the largest tier of 1024 / 896 / 768 whose free-fit bucket
  (`freefit_bucket` on the master after its crop margins, `band_for_tier`) the master
  covers at 2× (scale ≤ 1). Picked: 1024 on 1,518, 896 on 395, 768 on 400, none on 929;
  2,313 of 3,242 usable (71 %). Scale of the 2× target against the master, median (min –
  max): 1024 0.70 (0.21 – 1.00), 896 0.91 (0.85 – 1.00), 768 0.91 (0.85 – 1.00). Formats
  on 896 / 768: webp 374 / 392, jpeg 17 / 4, png 4 / 4. 4 of 91 artist folders have no
  usable image; the thinnest: `kase_daiki` 2 / 54, `sumiyao_(amam)` 5 / 61,
  `tottotonero` 11 / 80, `ru_zhai` 11 / 80.
- The tiers' token bands (`EDGE_TOKEN_BANDS`): 768 (2160, 2160), 896 (3000, 3024), 1024
  (4032, 4200).
- The user (2026-11-20), on the build: "그럼 896 768은 입력이 작아지고 타겟은 원본인걸로
  보장되는거지? 내생각에 이쪽이 업스케일링 task로썬 좋은 방향일듯".
- Built set, `bench/pixel_readout/x2_flex.py` (`x2flex_1120_1700`, CPU; images and
  `manifest.json` under `post_image_dataset/x2flex/` (local), 7.6 GB). In place of the tier
  scan above, a master that does not cover 2× the 1024 bucket is cropped to a multiple of
  32 px per side at the stem's anchor (after the free-fit aspect clamp) and kept as the 2×
  target without resampling; its LANCZOS ×½ is the ×1 input, kept when the ×1 holds 2,160 –
  4,200 tokens, at any size in between, not at a tier bucket. Rows: `down` (×1 = the
  resized image and its cached latent, 2× = `target_2x`) 1,518; `exact` 820 (×1 tokens
  2,161 – 3,024 on 421, 3,025 – 4,200 on 395, 2,160 on 4); dropped 904.
- Split: every image of the P1 test artists out of training (182 rows); read set `test`
  = 2 `down` rows per test artist with any (12), `text` = per text artist the `down` row
  with the longest quoted caption (5 of 8 artists have one). Train: 1,345 `down` + 794
  `exact`. On 200 sampled rows the 2× image is exactly twice the ×1 size, sides multiples
  of 16.
- Code (not yet run): `x2_head.py --x2flex <manifest>` takes train / test / text from the
  manifest and reads every ×2 meter and sheet against the real 2× image (`*_vs_real`
  keys; the teacher on VAE(x̂0) against the real image is logged as `<set>_teacher_*`);
  `--x2_target real` trains on the real 2× window; `--pegan` / `--lambda_pegan` add the
  PE-Spatial discriminator (`x2_pegan.py`: taps after PE blocks 4 / 8 / 12, a per-token
  conv head each, paired relativistic loss on 512-px crops, R1 / R2 in the
  noise-perturbation form, weight 0.1, σ 0.01, head LR 1e-4).
- Smoke `x2pgsm_1120_1555` (B's recipe with `--pegan --lambda_pegan 0 --term_log`, 100
  rollouts; `bench/pixel_readout/results/20261120-1640-x2pgsm_1120_1555/` (local)). Read
  against the real 2× image, test 12 / text 5, one seed:

  | | dB | LPIPS | dark lift |
  |---|---|---|---|
  | bicubic ×2 of VAE(x̂0) | 30.35 / 26.39 | – | – |
  | teacher on VAE(x̂0) | 29.50 / 25.47 | 0.155 / 0.094 | +4.06 / +8.38 |
  | chain (`realr_1119_2200` ×1 → teacher) | 30.06 / 25.98 | 0.171 / 0.114 | +5.83 / +12.07 |
  | ×2 head `x2real_1119_2242`, as loaded | 29.77 / 25.71 | 0.196 / 0.176 | +7.91 / +14.03 |
  | after 100 rollouts on the real target | 29.74 / 25.45 | 0.191 / 0.180 | +7.19 / +13.68 |

  ×1 against the original image over the 100 rollouts: 32.88 → 32.39 / 27.83 → 27.24 dB.
  Unweighted gradient norms over the head at rollouts 50 / 100: rec2 0.149 / 0.111, LPIPS
  4.21 / 3.66, dc 0.369 / 0.257, PE-GAN 71.4 / 25.3. Discriminator (not opposed, λ 0): margin
  5.36 / 6.46, loss 0.018 / 0.008, R1 0.050 / 0.035. 2.60 s per rollout, disc 42 ms per
  step, peak 10.0 GB.

### Interpretation (supply)

- The `exact` latents live in `post_image_dataset/x2flex/cache/`, not
  `post_image_dataset/lora/`, where an extra `{stem}_{WxH}_anima.npz` would add buckets
  to LoRA training. TE caches are per stem and carry over.
- On `exact` rows the 2× target is the master's own pixels, so its compression shows at
  1:1, where the `down` set's median 0.70 averages it out.
- The deployment case is 1024 → 2048; `exact` inputs put up to about 1.4× more content in
  a token. Held-out reads stay on the `down` set.

### Design (Claude's draft; to be fixed in a run header before it is queued)

- Parent `x2ctl2_1120_0925`'s recipe (resume `x2real_1119_2242`, LR 2e-4, window 24, margin
  3, λ_lpips 0.1, λ_dc 1.0, λ_lpips_x1 0.05, replay 64 × 16), with the points reduced to
  the base range `0.0577-0.1364:real`, the train set reduced to the `x2flex` set above,
  no manga pages (no 2× source checked for them), no turbo adapter, `--chain` on.
- Three arms, queued as one batch after the running training job ends:
  - control: the ×2 target unchanged (the teacher on VAE(x̂0));
  - B: the ×2 target at the same windows changed to the real 2× image, recipe loss, LPIPS
    and the tone term against it; nothing else changed;
  - B+GAN: B plus the PE-Spatial discriminator (Next arms, 1), real = the real 2× images,
    `--pegan --pegan_ratio 1.0`: the weight is set each step so the PE-GAN gradient at the
    ×2 output equals rec2's (cap 0.1). A fixed λ was not taken from the smoke because its
    PE-GAN gradient fell 71 → 25 in 50 rollouts. Ratio 1.0 is a choice; at the smoke's
    norms the SR line's λ_gan 3e-3 gives 0.5 – 1.5× rec2 over the head. 250 rollouts, as
    the others.
  - Labels `x2bgan_1120_1650`, `x2bctl_1120_1650`, `x2b_1120_1650`, queued in that order
    (B+GAN first, so the new weight path fails early if it does); shared flags: `--compile
    --init realr_1119_2200 --resume_x2 x2real_1119_2242 --x2flex
    post_image_dataset/x2flex/manifest.json --points 0.0577-0.1364:real --rollouts 250
    --eval_every 125 --opt_steps 16 --replay 64 --window 24 --margin 3 --lr 2e-4
    --lambda_lpips 0.1 --lambda_dc 1.0 --lambda_lpips_x1 0.05 --chain --term_log`.
- Question: B against control — does a ×2 stage trained on the real 2× image read closer
  to it, on lines and small glyphs, than one trained on the teacher's upscale? B+GAN
  against B — does the adversarial term sharpen lines beyond the target change? Level on
  both → the ×2 stage's width (fit probe: width 256 → 512 gave +0.8 dB).
- Read: the `test` and `text` rows, ×2 against the real 2× image (dB, LPIPS, dark
  lift) and pairwise 768-px crops `control | B | B+GAN`, with `original 2× | arm` beside
  them, labelled; the user's read decides. Bars: spread unknown.
- Preparation: built (Facts above), except the `exact` latents and the smoke, queued.
- Cost: 250 rollouts each at about 2.6 s (the smoke's rate) plus a 2.5-minute held-out
  build: about 15 minutes per arm, 45 for the batch.

### Facts (the batch, 2026-11-20)

- Read against the real 2× image after 250 rollouts, test 12 / text 5, one seed
  (`bench/pixel_readout/results/20261120-{1702-x2bctl,1709-x2b,1653-x2bgan}_1120_1650/`
  (local)); start = `x2real_1119_2242` as loaded:

  | | ×2 dB | ×2 LPIPS | dark lift | ×1 vs original dB |
  |---|---|---|---|---|
  | start | 29.77 / 25.71 | 0.196 / 0.176 | +7.91 / +14.03 | 32.88 / 27.83 |
  | control `x2bctl_1120_1650` | 29.77 / 25.58 | 0.196 / 0.172 | +8.00 / +14.42 | 32.80 / 27.65 |
  | B `x2b_1120_1650` | 29.77 / 25.45 | 0.181 / 0.174 | +7.48 / +14.27 | 32.42 / 27.23 |
  | B+GAN `x2bgan_1120_1650` | 29.77 / 25.45 | 0.181 / 0.173 | +7.51 / +14.32 | 32.42 / 27.22 |
  | chain (reference) | 30.06 / 25.98 | 0.171 / 0.114 | +5.83 / +12.07 | – |

- At rollout 125 every arm read worse on LPIPS (0.265 – 0.286 test) and came back by 250.
- B+GAN's per-step weight ran 3.4e-4 – 5.3e-4 (`--pegan_ratio 1.0`); the discriminator
  margin rose 5.3 → 7.6 over the run.
- Sheets (local): `~/x2pair_1120/armB/` (`original 2x | control | B`, 768-px crops at the
  window `x2_head.py` picked on the real image, labelled) and `~/x2pair_1120/armB_gan/`
  (`B | B+GAN`). Claude's read, labels visible, two sheets: control and B hard to tell
  apart; both smooth the paper grain the original carries on `test_01`.

- Two more B+GAN runs, B's recipe otherwise: `x2bgan8_1120_1715` (`--pegan_ratio 8`, weight
  2.5e-3 → 1.5e-3, weighted PE-GAN gradient over the head 1.0 – 2.0× rec2's, 0.11 – 0.50× the
  ×2 terms' summed norms) and `x2bganx_1120_1730` (`--pegan_param_ratio 1.0`: once per
  rollout, the PE-GAN gradient norm over the head set equal to that of rec2 + 0.1·LPIPS + dc;
  weight 4.1e-2 → 6.4e-3; disc margin 4.5 → 6.8). At 250 rollouts, test / text: `x2bgan8`
  29.71 / 25.41 dB, LPIPS 0.180 / 0.175, dark +7.52 / +14.30; `x2bganx` 29.63 / 25.39 dB,
  LPIPS 0.190 / 0.182, dark +7.42 / +14.20, ×1 vs original 32.43 / 27.26. At 125 every arm
  read worse on LPIPS, `x2bganx` most (0.295 / 0.290).
- Safe read set (the 6 `safe`-rated rows of the held-out artist sushispin, read with
  `--rollouts 0 --resume_x2`, `post_image_dataset/x2flex/manifest_safe_read.json`): dB /
  LPIPS / dark — control 27.74 / 0.256 / +7.07, B 27.73 / 0.250 / +6.05, `x2bgan8` 27.69 /
  0.247 / +6.08, `x2bganx` 27.66 / 0.256 / +6.05; teacher 27.67 / 0.205, chain 28.20 / 0.232.
  Sheets (local): `~/x2pair_1120/{safe_ctl_b,safe_b_gan8,safe_b_ganx,b_ganx}/`.
- Claude's read, labels visible, two sheets: B and both GAN runs near-identical; all
  softer than the original on line art.
- Every arm here ran 250 rollouts (4,000 updates, summed LR ≈ 0.4) from the converged
  `x2real_1119_2242`, re-warmed to 2e-4; the ×1 release trained 48,000 updates (≈ 4.8),
  the ×2 preview 16,000 (≈ 1.6).

- The user (2026-11-20), on the ×1 drift: separate ×1 and ×2 files; the ×2 file trains on
  the ×2 losses alone (`--x1_weight 0 --lambda_lpips_x1 0`). At 250 rollouts B with ×1 off
  `x2bo_1120_1815` reads 29.77 / 25.43 dB, LPIPS 0.180 / 0.174 (B with ×1 on 0.181 / 0.174),
  B+GAN ×8 with ×1 off `x2bgan8o_1120_1815` 0.182 / 0.176; safe 0.248 / 0.246.
- 1,000 rollouts (16,000 updates, as the preview), test / text, and safe:

  | | ×2 dB | ×2 LPIPS | dark lift | safe dB / LPIPS / dark |
  |---|---|---|---|---|
  | release `x2real_1119_2242` (`read_x2real_1120_1820`) | 29.77 / 25.71 | 0.196 / 0.176 | +7.92 / +14.03 | 27.76 / 0.258 / +6.89 |
  | B+GAN ×8, ×1 on `x2bgan8k_1120_1744` | 29.93 / 25.55 | 0.165 / 0.163 | +7.45 / +14.77 | 27.83 / 0.228 / +5.85 |
  | B, ×1 off `x2bok_1120_1840` | 29.98 / 25.55 | 0.163 / 0.162 | +7.42 / +14.85 | 27.85 / 0.228 / +5.85 |

  LPIPS by eval, `x2bok`: 0.280 / 0.227 / 0.180 / 0.163 (test, rollouts 250 / 500 / 750 /
  1000); `x2bgan8k`: 0.272 / 0.226 / 0.183 / 0.165. `x2bgan8k`'s ×1 vs original ended at
  32.64 / 27.41 (release 32.88 / 27.83). Disc margin at 1,000: 8.16.
- Sheets (local): `~/x2pair_1120/{release_b1k,b1k_gan1k,safe_release_b1k,safe_release_gan8k}/`.
- The user (2026-11-20), on the metrics: the ResShift output looks much cleaner than the
  heads while the numbers sit close.
- The user's read of the 1k sheets (2026-11-20): "그래 좀 더 낫다고 해야하나"; decision: "x1
  끄는쪽으로 가는게 적어도 같은 계산량에선 이득인듯". The ×2 file's recipe from here: `--x1_weight 0
  --lambda_lpips_x1 0`, the ×1 output served by the ×1 release file. Time per rollout: B
  ×1 on 1.00 s, ×1 off 0.92 s (250 rollouts), 0.82 s (1k); B+GAN ×8 1.92 s.
- Claude's read, labels visible, one sheet (`text_02`): B 1k's glyphs a little firmer than
  the release's; small glyphs smeared in both.

### Interpretation (the batch)

- B+GAN equals B on every number: at the weight the per-step rule set, the GAN term did
  not move the head, so this batch does not read the GAN. A GAN read needs a larger weight.
- B against control: test LPIPS 0.181 against 0.196 and dark lift 0.5 lower, text level;
  ×1 lost 0.4 dB more on B. Spread unknown; the user's read of the sheets decides.
- At 1,000 rollouts B alone equals B+GAN ×8: the gain over the release comes with the
  length on the real target, not with the PE-GAN term, which the discriminator out-ran in
  every run (margin 7 – 8). The ×1 loss in the ×2 file made no difference to ×2.
- Against the real 2× image dB favours blur and LPIPS is diluted by flat area; the
  teacher's crisp, re-drawn lines lose on both while reading cleaner. Sheets decide.

### Design: B at 4,000 rollouts, `x2b4k_1121_0950` (2026-11-21, queued on the user's word)

#### Facts

- Parent `x2bok_1120_1840` with `--rollouts` 1000 → 4000 and `--eval_every` 250 → 500,
  nothing else changed (resume `x2real_1119_2242`, ×1 loss off, real 2× target, LR 2e-4 on
  one cosine cycle over the run, so the schedule lengthens with it). 64,000 updates; the
  ×1 release trained 48,000.
- Records it rests on: `x2bok_1120_1840` test LPIPS 0.280 / 0.227 / 0.180 / 0.163 at
  rollouts 250 / 500 / 750 / 1000, 0.82 s per rollout.
- Read: the `test` 12 / `text` 5 rows and the safe set (`safe_x2b4k_1121_0950`), ×2 against
  the real 2× image (dB, LPIPS, dark lift), and `original 2× | B 1k | B 4k` crops by
  `x2_pair.py`, labelled; the user's read decides. Bars: spread unknown.
- Cost: 4,000 rollouts at 0.82 s plus nine evals, about 60 minutes.
- The user (2026-11-21), on Claude's proposal of this run, the discriminator probe below
  and a GAN arm after them: "ㅇㅋ 해보자".

#### Interpretation

- Question: does the ×2 output keep closing on the real 2× image with length, and what do
  its lines look like once the regression terms have converged? Still softer than the
  original with LPIPS flat → the remaining gap is the loss's, and the GAN arm continues
  from this head with this run as its control. Lines firm up → length first, GAN later.
- B's LPIPS was still falling at the end of the 1,000-rollout cosine cycle, so a GAN arm
  on the 1k head would be read against a moving control.

### Outcome of `x2b4k_1121_0950`

#### Facts

- 4,000 rollouts, 0.84 s per rollout, 58 minutes. Results (local)
  `bench/pixel_readout/results/20261121-1047-x2b4k_1121_0950/` and
  `…-safe_x2b4k_1121_0950/`. Against the real 2× image, test 12 / text 5, and the safe
  set (6), one seed:

  | | ×2 dB | ×2 LPIPS | dark lift | safe dB / LPIPS / dark |
  |---|---|---|---|---|
  | release `x2real_1119_2242` | 29.77 / 25.71 | 0.196 / 0.176 | +7.92 / +14.03 | 27.76 / 0.258 / +6.89 |
  | B 1k `x2bok_1120_1840` | 29.98 / 25.55 | 0.163 / 0.162 | +7.42 / +14.85 | 27.85 / 0.228 / +5.85 |
  | B 4k `x2b4k_1121_0950` | 30.45 / 25.92 | 0.141 / 0.137 | +6.52 / +14.00 | 28.20 / 0.204 / +5.44 |
  | chain | 30.06 / 25.98 | 0.171 / 0.114 | +5.83 / +12.07 | 28.20 / 0.232 |
  | teacher on VAE(x̂0) | 29.50 / 25.47 | 0.155 / 0.094 | +4.07 / +8.38 | 27.67 / 0.205 |

- By eval, test dB / LPIPS at rollouts 500 … 4000: 29.25 / 0.243, 29.46 / 0.272, 29.58 /
  0.220, 29.65 / 0.195, 29.79 / 0.178, 29.99 / 0.168, 30.31 / 0.144, 30.45 / 0.141.
- The file's ×1 output against the original image fell to 24.76 dB on test (release
  32.88): with the ×1 loss off the ×1 output of a ×2 file is not usable.
- Sheets (local): `~/x2pair_1121/b1k_b4k/` (`original 2× | B 1k | B 4k`, the 768-px
  crops of the runs' own sheets, 9 images, labelled).
- Claude's read, labels visible, two sheets (`text_02`, `test_01`): B 4k's glyphs and
  lines a little firmer than B 1k's; small glyphs smeared in both and short of the
  original; the original's grain on flat areas is in neither head's output.
- The user's read (2026-11-21): "확실히 결과가 나아지기는 했는데 아직 선이 또렷하지 않아서 gan은
  필요할걸로 생각돼".

#### Interpretation

- Length on the real target kept paying to 4,000 rollouts; the last two evals flatten
  with the cosine tail, so convergence and schedule are not separated.
- On `test` the head now reads above the chain and the teacher on dB and LPIPS; on `text`
  LPIPS it stays under both, which is where small glyphs are.

### Design: discriminator probe, `x2dprobe_1121_1015` (2026-11-21, queued behind `x2b4k_1121_0950`)

#### Facts

- Code: `bench/pixel_readout/x2_dprobe.py`, entered by `x2_head.py --dprobe N`. The head is
  frozen (`--resume_x2 x2b4k_1121_0950`, `--rollouts 0`). 32 train images (seeded sample of
  the `x2flex` train rows, `down` and `exact`) are rendered whole at the base point; one
  fresh `x2_pegan.PEDisc` per arm (the B+GAN runs' settings: taps after blocks 4 / 8 / 12,
  512-px paired crops, one crop per step, LR 1e-4, R1 / R2 0.1 at σ 0.01) trains 1,600
  steps on them and is read on every non-overlapping 512-px crop of the `test` 12 /
  `text` 5 images.
- Arms, by what stands on the fake side against the real 2× crop: `head` (the ×2 output),
  `chain` (ResShift ×2 of the ×1 release head's output), `null` (the real 2× image of
  another picture), `head_n0.03` and `head_n0.1` (`head` with Gaussian noise of that σ,
  [-1, 1] units, added to both crops in training and in the read).
- Read: held-out margin `D(real) − D(fake)` per arm and per tap, over all PE tokens and
  over the tokens whose 16-px cell is flat (range of the σ 2-px blurred luma under 4
  levels) or holds an edge (at least 24) in the real image; the training margin by row
  kind; the std of the luma's high-pass residual on flat cells for real / head / chain;
  margin maps on four held-out crops per arm. Bars: none; `null` is the meter's floor.
- Records it rests on: the discriminator margin 7 – 8 in every B+GAN run (above).
  Claude's read, labels visible, one sheet (`~/x2pair_1120/release_b1k/…test_00…`
  (local)): the original carries grain over the whole frame, both heads' outputs none.
- CPU check of the code on synthetic images (a smooth image against itself plus σ 0.02
  noise, 8 steps): `head` held margin 2.24, `null` −0.11, `head_n0.1` 0.02.
- Cost: about 12 minutes.

#### Interpretation

- Question: what does the discriminator separate the ×2 output from the real image on?
  If the noise arms take most of the margin away and flat cells read as high as edge
  cells, the cue is grain the head cannot produce from its features, and the GAN arm
  takes the discriminator with the noise level that leaves an edge margin. If the margin
  stays on edge cells under noise, the cue is the lines and the GAN arm keeps the
  discriminator and changes only its weight.
- `chain` against `head`: if the chain reads as far from the real image as the head, a
  discriminator on the real masters does not ask for what the user reads as clean in the
  ResShift output.
- The flat / edge split is weak on the deep taps (a PE token past block 8 sees the whole
  crop); the noise arms carry the read.

### Outcome of `x2dprobe_1121_1015`

#### Facts

- 10 minutes; results and margin maps (local)
  `bench/pixel_readout/results/20261121-1053-x2dprobe_1121_1015/`. Pool 32 train images
  (20 `down`, 12 `exact`), read on 17 held-out images (220 crops, 225,280 cells: 42.8 %
  flat, 31.9 % edge). Head = `x2b4k_1121_0950`. One seed per arm.

  | arm (fake side) | train margin, first → last 200 steps | held margin | tokens with real above | held margin flat / edge, tap 4 | train margin `down` / `exact` |
  |---|---|---|---|---|---|
  | `head` | 3.67 → 6.53 | 6.33 | 0.997 | 6.27 / 7.19 | 6.21 / 6.90 |
  | `chain` | 3.52 → 6.18 | 5.17 | 0.998 | 5.60 / 6.24 | 5.50 / 7.04 |
  | `null` | 0.01 → 0.03 | 0.05 | 0.536 | 0.03 / −0.03 | 0.01 / 0.03 |
  | `head_n0.03` | 4.13 → 8.22 | 7.97 | 0.998 | 8.30 / 9.58 | 7.81 / 8.82 |
  | `head_n0.1` | 3.68 → 8.38 | 8.30 | 0.988 | 8.04 / 9.87 | 7.99 / 9.18 |

- Std of the luma's high-pass residual on flat cells, 8-bit levels: real 0.87, head 0.48,
  chain 0.32.
- Claude's read of two margin maps, labels visible (`dprobe_head_test_13340817.png`,
  `dprobe_head_n0.1_test_10808923.png`): the margin is near-uniform over the crop at
  every tap, blank white background included.

#### Interpretation

- The grain reading is rejected: noise 4 to 15 times the real images' flat-cell residual
  on both crops leaves the discriminator separating 99 % of tokens. The cue survives
  noise of 13 levels, so it is not a low-amplitude texture.
- The margin's size is not comparable between the noise arms and the others: under noise
  the R1 / R2 term (a σ 0.01 perturbation) costs the discriminator less.
- The chain separates as the head does, so this discriminator does not place the ResShift
  output near the real masters either.
- The per-token logits act as one decision per crop: a PE token after block 4 already
  mixes the whole 512-px crop, so the flat / edge split and the maps do not localise the
  cue, and the generator's gradient from it is not local to lines.
- What the cue is stays open between the head's tone (dark pixels lifted 6 – 14 levels,
  shared with the chain) and its lines; the band probe below separates them.

### Design: band probe, `x2dband_1121_1125` (2026-11-21)

#### Facts

- `x2dprobe_1121_1015`'s command with `--dprobe_arms band`, nothing else changed (same
  pool, head, discriminator settings and seeds). The fake side is built from the real
  image and the head's output, split by a Gaussian of σ px at the 2× scale:
  `low<σ>real_high<σ>head` (the head's output with the real image's low band),
  `low<σ>head_high<σ>real` (the real image with the head's low band), σ 8 and 2, and
  `real_downup` (the real crop bicubic ×½ then ×2).
- Read: held margin and the share of tokens with real above, against `head` 6.33 / 0.997
  and `null` 0.05 / 0.536 of `x2dprobe_1121_1015`. Bars: none.
- Cost: about 10 minutes.

#### Interpretation

- Question: is the head separated from the real image by its low band (tone, colour) or
  its high band (lines, texture)? `low8head_high8real` near `null` and
  `low8real_high8head` near `head` → the lines; the GAN term then asks for what the user
  asks for, and the arm changes the weight and the discriminator's locality. The reverse
  → tone; the discriminator then needs the low band taken out of its input before a GAN
  arm says anything about lines.
- `real_downup` reads how separable a 1× image's content is at all, with no model in it.

### Outcome of `x2dband_1121_1125`

#### Facts

- 10.5 minutes; results (local) `bench/pixel_readout/results/20261121-1107-x2dband_1121_1125/`.
  Same pool, held-out crops, head and seeds as `x2dprobe_1121_1015`.

  | arm (fake side) | held margin | tokens with real above | train margin `down` / `exact` |
  |---|---|---|---|
  | `head` (`x2dprobe_1121_1015`) | 6.33 | 0.997 | 6.21 / 6.90 |
  | `low8real_high8head` | 6.40 | 0.996 | 6.18 / 6.94 |
  | `low2real_high2head` | 5.99 | 0.996 | 5.95 / 6.85 |
  | `low2head_high2real` | 4.52 | 0.996 | 4.57 / 5.22 |
  | `low8head_high8real` | 2.51 | 0.997 | 2.88 / 2.60 |
  | `real_downup` | 6.64 | 0.995 | 6.60 / 7.20 |
  | `null` (`x2dprobe_1121_1015`) | 0.05 | 0.536 | 0.01 / 0.03 |

#### Interpretation

- The margin follows the head's high band: with the real image's tone under it the head
  separates as before (6.40 against 6.33), and the band finer than σ 2 px carries most of
  it (5.99). The real image under the head's tone separates at well under half the margin
  (2.51), though still on 99.7 % of tokens: tone is a second, weaker cue.
- A real image taken down to 1× and back separates as the head does (6.64). What the
  discriminator reads is what a 1× image lacks at 2×, which is the sharpness the user
  asks for; it is the right teacher for the lines.
- With the first probe: the earlier B+GAN runs did not fail on a wrong cue. What is left
  is the term's weight and length (the largest, `x2bganx_1120_1730`, ran 250 rollouts on a
  head that was still 3,750 rollouts from where B alone ends) and the crop-level decision.

### Design: GAN micro arms on the B 4k head (2026-11-21, queued)

#### Facts

- The user (2026-11-21), on the B 4k sheets: "확실히 결과가 나아지기는 했는데 아직 선이 또렷하지
  않아서 gan은 필요할걸로 생각돼"; on the small glyphs: "애초에 anima가 그렇게 또렷한 일본어를
  적지를 못해서 생기는 문제같긴해".
- Control `x2gctl_1121_1120`: `x2b4k_1121_0950`'s command with `--resume_x2`
  `x2real_1119_2242` → `x2b4k_1121_0950`, `--rollouts` 4000 → 1000, `--eval_every` 500 →
  250, nothing else changed (a new cosine cycle at LR 2e-4 with the 200-step warmup).
- `x2gan1_1121_1120`: the control with `--pegan --pegan_param_ratio 1.0
  --pegan_lambda_max 1.0`. `x2gan4_1121_1120`: the same with ratio 4.0. The ratio sets the
  PE-GAN weight once per rollout so its gradient norm over the head is that multiple of
  rec2 + 0.1·LPIPS + dc's; the cap is raised from 0.1 so it does not bind at ratio 4
  (`x2bganx_1120_1730` at ratio 1: weight 4.1e-2 → 6.4e-3).
- Records they rest on: `x2bganx_1120_1730` (ratio 1, 250 rollouts from the release
  preview: LPIPS 0.190 / 0.182 against B's 0.181 / 0.174, sheets near-identical on
  Claude's labelled read, disc margin 4.5 → 6.8); the two probes above.
- Read: `test` 12 / `text` 5, ×2 against the real 2× image (dB, LPIPS, dark lift), the
  discriminator margin over the run, and `original 2× | control | arm` 768-px crops,
  labelled, one sheet set per arm; the user's read of lines and screentone decides. Small
  glyphs are not part of the read (Claude's proposal after the user's remark above).
  Bars: spread unknown.
- Cost: control about 14 minutes (0.84 s per rollout), each GAN arm about 32 (1.92 s,
  `x2bgan8k_1120_1744`'s rate): 78 minutes for the batch.

#### Interpretation

- Question: with the GAN term at a weight where it holds a full share of the update, on a
  head whose regression terms have converged, do the lines sharpen against the control?
  Yes at either ratio → a full-length run at that ratio. Level at both with the margin
  still at 6 – 8 → the discriminator's locality (512-px crops read as one decision) is
  the next variable, then the ×2 stage's width.
- Changed together in the GAN arms and not separated: the term and the 1.1 s per rollout
  it adds. The `exact` rows separate a little more than the `down` rows in both probes;
  a head that learns compression texture would show it on the crops.

### Outcome of `x2gctl_1121_1120` and `x2gan1_1121_1120`

#### Facts

- 1,000 rollouts each from `x2b4k_1121_0950`; control 0.83 s per rollout, `x2gan1` 1.71 s
  (discriminator update 42 ms). Against the real 2× image, test 12 / text 5, one seed:

  | | rollout | ×2 dB | ×2 LPIPS | dark lift |
  |---|---|---|---|---|
  | start (`x2b4k_1121_0950`) | 0 | 30.45 / 25.92 | 0.141 / 0.137 | +6.52 / +14.00 |
  | control | 500 | 29.75 / 25.17 | 0.181 / 0.168 | +5.57 / +12.42 |
  | `x2gan1` | 500 | 29.30 / 24.94 | 0.242 / 0.247 | +5.20 / +12.11 |
  | control | 1000 | 30.18 / 25.57 | 0.145 / 0.142 | +6.45 / +13.73 |
  | `x2gan1` | 1000 | 29.93 / 25.49 | 0.153 / 0.144 | +6.02 / +13.15 |

- `x2gan1`: discriminator margin 3.4 at rollout 50, 5.8 – 6.1 over rollouts 300 – 800,
  7.0 – 7.2 at 900 – 1000; weight 2.4e-2 → 4.4e-3 … 6.3e-3.
- Sheets (local): `~/x2pair_1121/gctl_gan1/` (`original 2× | control | GAN ratio 1`, the
  runs' own 768-px crops, 9 images, labelled).
- Claude's read, labels visible, one sheet (`test_01`): the GAN arm carries a grain-like
  texture in the shaded areas where the control is smooth and the original is grained;
  its lines a little darker than the control's, short of the original. The user has not
  read these sheets yet.
- Speed of the PE-GAN update (`pegan_speed.py`, `peganspeed_1121_1200` and
  `pegancheck_1121_1205`, 512-px crops, generator term with its backward plus one
  discriminator update): 70.1 ms as `x2gan1` ran it (four tower forwards in the update),
  44.6 ms with `gen_term`'s tokens reused and the two noised crops as one batch, 25.0 ms
  with the tower compiled (`--pegan_compile`; first update 18 – 48 s). On CPU the reuse
  matches the old path to six digits over three updates. Compiled against eager on one
  crop, bf16: tower tokens 1.1 – 1.5 % apart, the gradient at the fake crop at cosine
  0.996. `x2gan4_1121_1120` was requeued on the new path with `--pegan_compile`.

#### Interpretation

- The control did not return to its start in 1,000 rollouts of a re-warmed cycle, so arms
  are read against the control's end, not the B 4k head's.
- At ratio 1 the term moved the output this time: a texture on Claude's read, a lower dark
  lift, 0.25 dB and 0.008 LPIPS behind the control on test. Whether it reads as sharper
  lines is the user's read.
- The margin rose once the generator's rate decayed: the discriminator keeps training at
  a fixed rate while the head's follows the cosine.

### Outcome of `x2gan4_1121_1120` and the glyph read `glyph_1121_1125`

#### Facts

- `x2gan4_1121_1120` (ratio 4, the reused-token path with `--pegan_compile`): 1.34 s per
  rollout against `x2gan1`'s 1.71 and the control's 0.83, peak 10.8 GB, 25 minutes.
  Against the real 2× image, test 12 / text 5, one seed:

  | | rollout | ×2 dB | ×2 LPIPS | dark lift |
  |---|---|---|---|---|
  | control `x2gctl_1121_1120` | 1000 | 30.18 / 25.57 | 0.145 / 0.142 | +6.45 / +13.73 |
  | `x2gan1_1121_1120` | 1000 | 29.93 / 25.49 | 0.153 / 0.144 | +6.02 / +13.15 |
  | `x2gan4_1121_1120` | 750 | 29.32 / 25.05 | 0.226 / 0.227 | +5.36 / +12.63 |
  | `x2gan4_1121_1120` | 1000 | 29.50 / 25.23 | 0.186 / 0.180 | +5.50 / +12.28 |

- `x2gan4`: discriminator margin 3.6 at rollout 50, 5.5 – 5.8 over rollouts 150 – 800, 6.9 –
  7.1 at 900 – 1000; weight 9.8e-2 → 1.5e-2 … 1.8e-2.
- Sheets (local): `~/x2pair_1121/gctl_gan4/` (`original 2× | control | GAN ratio 4`) and
  `~/x2pair_1121/gan1_gan4/` (`original 2× | GAN ratio 1 | GAN ratio 4`), 9 images each,
  labelled.
- Claude's read, labels visible, one sheet (`gctl_gan4`, `test_01`): ratio 4 carries the
  same grain-like texture in the shaded areas as ratio 1; lines darker than the control's.
  The user has not read these sheets yet.
- `glyph_1121_1125` (`x2_head.py --glyph_sheets`, the ×1 release head `realr_1119_2200`,
  base point σ 0.097; results (local)
  `bench/pixel_readout/results/20261121-1230-glyph_1121_1125/`, one sheet per held-out
  image `original | VAE(latent) | VAE(x̂0) | ×1 head` at the ×2 sheets' window). dB
  against the original image, test 12 / text 5: VAE(latent) 40.42 / 35.46, VAE(x̂0)
  33.87 / 28.67, ×1 head 33.23 / 28.15.
- Claude's read, labels visible, one sheet (`…text_14_dan_5341366.png`): the body text is
  legible in VAE(latent) as in the original; in VAE(x̂0) the same glyphs have wrong
  strokes, and the ×1 head's follow VAE(x̂0)'s, a little softer.

#### Interpretation

- Ratio 4 pays more of the same price for the same change: 0.7 dB and 0.04 LPIPS behind
  the control at the end, still recovering when the cycle closed, dark lift about 1 level
  lower. Whether its lines read sharper than ratio 1's is the user's read.
- In both GAN runs the margin holds near 5.7 while the head's rate is high and climbs as
  the cosine decays; neither ratio closes it.
- Small glyphs: the VAE keeps them (35.5 dB on text), the DiT's one-step x̂0 at σ 0.097
  does not (28.7), and the head sits 0.5 dB under x̂0. The loss is in what Anima predicts,
  as the user said; a decoder reading its features cannot restore it.

### Interpretation (smoke)

- Against the real 2× image, bicubic has the highest dB of every column; LPIPS and dark
  lift order the columns differently (teacher best). dB alone does not rank arms here; the
  read rests on LPIPS, dark lift and the crops.
- The unopposed discriminator separates the head's output from the real crops within 50
  rollouts. Some of what it can key on is the masters' own compression, which the `exact`
  rows carry at 1:1; a B+GAN head that learns block or ringing texture would show it on
  the crops.
- The ×1 output lost about 0.5 dB against the original in 100 rollouts of ×2 training
  (Next arms, 4).

### Interpretation

- The target the ×2 stage has had is another model's upscale of the decoded image; the
  target a decoder would have is the 2× image the latent was encoded from. The ×1 output
  already moved to the original image in `realr_1119_2200` on the user's read.
- At σ 0.5 the turbo point's x̂0 and the master disagree, so a real target there would
  ghost; leaving the point out removes the question from this arm.

## ×2 preview — release

Moved to `reports/x2_preview_release_2026_11_19.md`: run 1 (`realr_1119_2200`) and run 2
(`x2real_1119_2242`), the render read (`x2bic_1119_2335`), the user's decision to ship the ×2 file
as a test file, and the deployment record.

## Next: the settled recipe as a full run, and the training code into the library

Decided by the user (2026-11-21): "다음 x2 plan에는 확정된 레시피로 x2 학습 돌리면서 x2 bench
코드가 상당히 복잡한데 라이브러리 승격 (이미 돼있으면 말고) 리팩터하는걸로 하자".

### Facts

- Already in the library (`library/anima/pixel_head.py`, 387 lines): `PixelHead`,
  `PixelHeadX2`, `load_pixel_head` (reads `x2_depth` files), `PixelTap`, the path and σ-range
  helpers; inference reaches them through `--decoder pixel_head` / `--pixel_scale`.
- Not in the library: training. `bench/pixel_readout/x2_head.py` is 1,412 lines with one
  `main()` that holds the data split, the teacher-forced rollout, the window and replay
  buffer, the loss terms, the held-out eval and the sheets as closures, beside the code of
  closed branches (arm A's teacher target, arm 3's critic `x2_dmd.py`, the ×1 loss inside a
  ×2 file) and of the probes (`--dprobe`, `--glyph_sheets`). The ×1 training and the shared
  helpers are `convnext_head.py` (965 lines). No `scripts/` entry or make target trains a
  head.
- The recipe so far: resume from the release ×1 head, the `x2flex` set, the real 2× image
  as the target, base points `0.0577-0.1364:real`, ×1 loss off, window 24 + margin 3,
  replay 64 × 16, LR 2e-4 on one cosine cycle, λ_lpips 0.1, λ_dc 1.0, 4,000 rollouts
  (`x2b4k_1121_0950`). Open in it: the GAN term and its weight (the micro arms above).

### Plan (Claude's draft)

1. Fix the recipe on the read of `x2gctl` / `x2gan1` / `x2gan4_1121_1120`.
2. Move the training loop into the library as modules a script composes: the item source
   (`x2flex` manifest rows), the rollout (features, x_t patches, VAE(x̂0)), the window and
   replay buffer, the loss terms (recipe loss, LPIPS, tone, PE-GAN), the held-out meters.
   A thin entry under `scripts/` with its flags in one place; the bench script keeps the
   sheets and the result envelope and imports the rest. Closed branches are not carried
   over: `x2_dmd.py` and the `--critic*`, arm A teacher-target and `--x1_weight` paths
   stay in the bench file or go to `_archive/`. Probes (`x2_dprobe.py`, `x2_overfit.py`,
   `feat_probe.py`) stay in `bench/` and import the library pieces.
3. Check of the move, before the full run: the moved loop on `x2bo_1120_1815`'s command
   (250 rollouts, same seeds) reads level with that run's 250-rollout eval (test LPIPS
   0.180, 29.77 dB); a unit test on the loss terms and the window margin
   (`CONTRIBUTING.md`, Tier 1.5).
4. The full run of the settled recipe through the new entry, read against
   `x2b4k_1121_0950` and the preview on `test` / `text` / the safe set and on renders
   (`p3_generated.py --x2`); on the user's read it replaces
   `anima_pixel_head_x2_preview.safetensors`, deployed as in "Deployment plan" below.

## Deployment plan: the file decides the scale

Decided by the user (2026-11-21), after asking whether a head file loaded with no ×1 / ×2
setting gives its own scale in principle: "ㅇㅋㅇㅋ 그걸 deploy plan으로 넣자".

### Facts

- The output scale is a property of the head's output layers, not of sampling: the DiT
  runs the same latent either way, the ×1 output layer writes 16×16 pixels per token and
  the ×2 stage 32×32.
- Now: a file stamped `x2_depth` loads as `PixelHeadX2` and `head_scales` gives it (1, 2);
  a ×1 file refuses scale 2 at load; a ×2 file at scale 1 passes. The scale is asked for
  everywhere, with 1 as the default: `inference.py --pixel_scale` (`PIXEL_SCALE=`),
  `GenerationRequest.pixel_scale`, the node's `scale` input (`"1"` / `"2"`).
- The preview file (`x2real_1119_2242`) trained both outputs. A file of the current recipe
  does not: `x2b4k_1121_0950`'s ×1 output reads 24.76 dB against the original (release
  32.88), and at scale 1 it would decode without a warning.
- The ×1-only parameters of `PixelHeadX2` are `base.norm_out` and `base.out`; with the ×1
  loss off they take no gradient and the optimizer's weight decay.

### Plan (Claude's draft; runs with step 4 of "Next" above)

1. The file: a ×2 file of the settled recipe is saved without `base.norm_out` / `base.out`
   and stamped with its scales (`scales = "2"`); training runs the head at `scale=2`.
   Files without the stamp keep today's reading (a ×1 file (1,), the preview (1, 2)).
2. Library: `load_pixel_head` builds the head the stamp describes and `head_scales` reads
   it; the default scale becomes the file's (its only scale; 1 for a two-scale file, as
   now) in `PixelTap.decode`, `--pixel_scale`, `PIXEL_SCALE=` and
   `GenerationRequest.pixel_scale`; an explicit scale the file does not hold refuses at
   load, in both directions. `tests/test_pixel_head.py` covers the stamp, the default and
   both refusals.
3. Node `ComfyUI-Anima-PixelDecoder`: `scale` gains `auto` as its default (saved
   workflows with `"1"` / `"2"` still load); `_vendor/` through `make vendor-sync`;
   version bump and README.
4. Catalog, Hub and docs: a catalog row for the ×2 file (none today), the Hub file in
   place of the preview on the user's read, the card's Use section,
   `docs/anima/inference/pixel_head.md`.

## Next arms

Claude's drafts, none queued. Micro arms on arm B's recipe, read as arm B is. Item 1 is
arm B's third arm (decided 2026-11-20); the rest follow arm B.

1. GAN on PE-Spatial features (the user, 2026-11-20: "우리 맨날 쓰는 pe spatial류를 gan에 활용해도
   될듯? fastframes류를 쓴다면"): FastFrames' form (heads on a frozen ViT at three depths,
   relativistic loss, R1 / R2; `plan.md`, "pixel losses on the head") on the local
   PE-Spatial-B16, real = the real 2× images. The weight is set from the gradient ratio on
   a smoke at the run's own settings, and the discriminator's logits are watched: arm 3's
   saturated within 100 rollouts.
2. The ×2 stage's width (1.6 M new parameters at width 256), if arm B reads level or the GAN
   arm sharpens too little.
3. The turbo point at ×2: postponed by the user (2026-11-20).
4. Keeping ×1: the ×1 loss weight, or a lower LR on the base, against the drift under ×2
   training. Read on the ×1 column of any arm above before a separate run.

Not taken up: the previous step's feature as a second input (+0.24 dB on a linear readout,
`reports/feat_probe_2026_11_20.md`).

## Open

- No seed twin, no spread, for any ×2 number.
- 16-window fit: a ceiling or slow convergence is not separated (the cosine schedule ends as
  the curve flattens); the 2-window numbers rest on one window per point.
- Only teacher-forced reads. No turbo or base rollouts through the ×2 head, no end-to-end time
  beside VAE → ResShift (teacher 3.16 s per image against the head's 12.9 ms).
- The dark-pixel lift is not split into softness and tone; `tone_probe.py` does not read the
  two-column sheets.
- `PixelHeadX2` and its loader are in `library/anima/pixel_head.py`; the ×2 training code is
  not ("Next: the settled recipe as a full run" above).
- ×4 not started.
- Arm B: whether masters are upscales or re-encoded (matters most on 896 / 768, near 1:1);
  a 2× source for the manga pages; a held-out set inside the 1024 set.
- `x2_dmd.py` and the `--critic*` flags stay in `x2_head.py`, off by default, after arm 3.

