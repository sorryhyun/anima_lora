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

## Arm B — the real 2× image as the ×2 target (decided 2026-11-20, not built)

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

### Design (Claude's draft; to be fixed in a run header before it is queued)

- Parent `x2ctl2_1120_0925`'s recipe (resume `x2real_1119_2242`, LR 2e-4, window 24, margin
  3, λ_lpips 0.1, λ_dc 1.0, λ_lpips_x1 0.05, replay 64 × 16), with the points reduced to
  the base range `0.0577-0.1364:real`, the train set reduced to the native set, and no
  manga pages (no 2× source checked for them).
- Control: that recipe with the ×2 target unchanged (the teacher on VAE(x̂0)).
- Arm: the ×2 target at the same windows changed to the real 2× image, recipe loss, LPIPS
  and the tone term against it; nothing else changed.
- Question: does a ×2 stage trained on the real 2× image read closer to it, on lines and
  small glyphs, than one trained on the teacher's upscale? Yes → the GAN arm on PE-Spatial
  features with the real 2× images as real. Level → the ×2 stage's width before any
  adversarial term (fit probe: width 256 → 512 gave +0.8 dB).
- Read: held-out images of the native set, ×2 against the real 2× image (dB, LPIPS, dark
  lift) and pairwise 768-px crops `control | arm`, with `original 2× | arm` beside them,
  labelled; the user's read decides. The held-out set needs more native-set images than
  those above. Bars: spread unknown.
- Cost: 250 rollouts each at about 1.5 s; the target build is CPU.

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

## Next arms

After arm B, Claude's drafts, none queued. Micro arms on arm B's recipe, read as arm B is.

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
- The ×2 stage lives in `bench/pixel_readout/x2_head.py::PixelHeadX2`; `library/anima/pixel_head.py`
  is untouched until an arm passes.
- ×4 not started.
- Arm B: whether masters are upscales or re-encoded; a 2× source for the manga pages; a
  held-out set inside the native set.
- `x2_dmd.py` and the `--critic*` flags stay in `x2_head.py`, off by default, after arm 3.

