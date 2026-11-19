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
