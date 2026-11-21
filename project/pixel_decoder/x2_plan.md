# ×2 output — plan (2026-11-19, updated 2026-11-21)

The P5 topic "×2 / ×4 output" of `plan.md`, as its own plan: decode and 2× upscale in one
readout of the DiT's last features, so a 1024-tier render leaves the head at 2048. Finished
arms are in `reports/`: arm A and its design `x2_arm_a_2026_11_19.md`, arms 1 + 2
`x2_lpips_dc_2026_11_19.md`, the preview release `x2_preview_release_2026_11_19.md`, arm 3
`x2_critic_2026_11_20.md`, the feature probe `feat_probe_2026_11_20.md`, the state after arm 3
`x2_state_2026_11_20.md`, arm B `x2_arm_b_2026_11_20.md`, B at 4,000 rollouts
`x2_b4k_2026_11_21.md`, the discriminator probes `x2_disc_probes_2026_11_21.md`, the GAN micro
arms and the glyph read `x2_gan_micro_2026_11_21.md`. Code:
`scripts/pixel_head/train_x2.py` on `library/anima/pixel_head_train.py` (training of the settled
recipe), `bench/pixel_readout/x2_head.py` (the earlier arms and the whole-image reads),
`x2_overfit.py` (fit probe), `x2_masters.py` / `x2_flex.py` (real 2× targets and the set),
`x2_pair.py` (pair sheets); results under
`bench/pixel_readout/results/` (local).

## State

Moved to `reports/x2_state_2026_11_20.md`: this section as it stood after arm 3 (arm A
`x2a_1119_1802` and the chain `x2chain_1119_1832`, the fit probes, the tone table, the user's reads
of 2026-11-19, arm 3, the real-pixel supply count, run 2 `x2real_1119_2242` and its render read
`x2bic_1119_2335`). The current recipe and the run in progress: "Next" below.

- The form: the release head `mixr_1119_1455` plus a second 2× stage (512 → 4 × 256 pixel
  shuffle, ConvNeXt blocks at 4-px spacing, an 8×8 RGB patch per position, 32×32 output pixels
  per token), default init, no skip. The ×1 output stays, so one file decodes at ×1 and ×2.
  Head time, ×1 and ×2 together: 12.9 ms on 80 × 52 tokens (2560 × 1664 output), eager.

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

## Arm B — the real 2× image as the ×2 target (2026-11-20 – 21)

Moved to `reports/x2_arm_b_2026_11_20.md`: the decision, the target supply (`x2masters_1120_1035`,
the `x2flex_1120_1700` set), the smoke `x2pgsm_1120_1555`, the batch `x2bctl_1120_1650` /
`x2b_1120_1650` / `x2bgan_1120_1650`, `x2bgan8_1120_1715`, `x2bganx_1120_1730`, ×1 off
`x2bo_1120_1815` / `x2bgan8o_1120_1815`, and 1,000 rollouts `x2bgan8k_1120_1744` / `x2bok_1120_1840`
(release read `read_x2real_1120_1820`). The user's decision (2026-11-20): "x1 끄는쪽으로 가는게 적어도
같은 계산량에선 이득인듯"; the ×2 file trains with `--x1_weight 0 --lambda_lpips_x1 0`.

Moved to `reports/x2_b4k_2026_11_21.md`: B at 4,000 rollouts `x2b4k_1121_0950` (safe read
`safe_x2b4k_1121_0950`), ×2 LPIPS 0.141 / 0.137 against the real 2× image (test 12 / text 5;
release 0.196 / 0.176), the file's ×1 output 24.76 dB on test. The user's read (2026-11-21): "확실히
결과가 나아지기는 했는데 아직 선이 또렷하지 않아서 gan은 필요할걸로 생각돼".

Moved to `reports/x2_disc_probes_2026_11_21.md`: the discriminator probe `x2dprobe_1121_1015`
(held margin `head` 6.33, `null` 0.05) and the band probe `x2dband_1121_1125`
(`low8real_high8head` 6.40, `low8head_high8real` 2.51, `real_downup` 6.64).

Moved to `reports/x2_gan_micro_2026_11_21.md`: the GAN micro arms on the B 4k head, control
`x2gctl_1121_1120`, `x2gan1_1121_1120` (ratio 1), `x2gan4_1121_1120` (ratio 4), test LPIPS at
rollout 1000 0.145 / 0.153 / 0.186; the PE-GAN update speed (`peganspeed_1121_1200`,
`pegancheck_1121_1205`: 70.1 ms → 25.0 ms with `--pegan_compile`); the glyph read
`glyph_1121_1125`. The recipe the user settled on these sheets: "Next" below.

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
- Not in the library when this was decided: training. `bench/pixel_readout/x2_head.py` is 1,412 lines with one
  `main()` that holds the data split, the teacher-forced rollout, the window and replay
  buffer, the loss terms, the held-out eval and the sheets as closures, beside the code of
  closed branches (arm A's teacher target, arm 3's critic `x2_dmd.py`, the ×1 loss inside a
  ×2 file) and of the probes (`--dprobe`, `--glyph_sheets`). The ×1 training and the shared
  helpers are `convnext_head.py` (965 lines). No `scripts/` entry or make target trains a
  head.
- The recipe so far: resume from the release ×1 head, the `x2flex` set, the real 2× image
  as the target, base points `0.0577-0.1364:real`, ×1 loss off, window 24 + margin 3,
  replay 64 × 16, LR 2e-4 on one cosine cycle, λ_lpips 0.1, λ_dc 1.0, 4,000 rollouts
  (`x2b4k_1121_0950`).
- The user (2026-11-21), after the `x2gctl` / `x2gan1` / `x2gan4_1121_1120` sheets: "gan으로
  레시피 확정이야 이거 이따가 돌릴게 2배정도가 제일 나은듯? gan2 이걸로 확정하자". The settled
  recipe: the lines above plus the PE-Spatial GAN term at `--pegan --pegan_param_ratio 2.0
  --pegan_lambda_max 1.0` (`--pegan_compile` for speed). Ratio 2 itself has not been run;
  ratios 1 and 4 have. The run is not queued: the user starts it later.

### Design: the full run, `x2bgan2_1121_1340` (2026-11-21, queued on the user's word)

#### Facts

- The user (2026-11-21): "let's start training b + gan for anima 2x pixel decoder but, how
  about turning gan on from 1k step among 4k?"; then, on the eval of the loaded head: "처음에
  eval하는건 빼도 될듯".
- Command: `x2b4k_1121_0950`'s (4,000 rollouts from `x2real_1119_2242`, `--eval_every 500`)
  with `--pegan --pegan_param_ratio 2.0 --pegan_lambda_max 1.0 --pegan_compile
  --pegan_start 1000 --skip_eval0`. Through `bench/pixel_readout/x2_head.py`, before the
  move into the library.
- `--pegan_start N` (new): before rollout N the term is out of the loss and the
  discriminator takes no update; both start at rollout N, the discriminator fresh, as in
  the micro arms at rollout 0. `--skip_eval0` (new) drops the held-out eval of the loaded
  head.
- Smoke `smoke_pegan_start` (100 rollouts, start 50, results (local)
  `bench/pixel_readout/results/20261121-1339-smoke_pegan_start/`): no PE-GAN keys in the
  log at rollout 50; at rollout 100 weight 0.158, margin 4.14, discriminator update 29 ms,
  peak 10.0 GB.
- The head the term starts on: `x2b4k_1121_0950`'s eval at rollout 1000 is 29.46 / 25.09
  dB, LPIPS 0.272 / 0.284 (test 12 / text 5), the highest LPIPS of that run's nine evals
  (0.196 / 0.176 at rollout 0, 0.141 / 0.137 at 4000). The head's rate at rollout 1000 is
  0.85 of its peak.
- Read: the evals at 500 and 1000 against `x2b4k_1121_0950`'s (same command and seed up to
  there), 1500 – 4000 against the same rows as the term's effect over length; the end as
  the micro arms were read (dB, LPIPS, dark lift, margin, `original 2× | B 4k | this run`
  768-px crops, labelled; the user's read of lines and screentone decides). Bars: spread
  unknown.
- Cost: 1,000 rollouts at 0.84 s and 3,000 at about 1.34 s, about 85 minutes with the evals.

#### Interpretation

- Question: does the term, added to B's own schedule from rollout 1000, end with sharper
  lines than B alone at the same length, and at what cost on LPIPS against the 0.141 /
  0.137 B reaches?
- At rollout 1000 the head is not converged, so this run reads the term on a moving head;
  the micro arms read it on a converged one. A start past rollout 2500 (B's LPIPS below
  0.18) would be the converged-head form at full length, if this run's lines read no
  sharper than the micro arms'.
- Not separated from the term: the discriminator's fixed rate against the head's cosine.
  In both micro arms the margin rose as the head's rate fell; here that stretch is longer.

### Outcome of `x2bgan2_1121_1340`

#### Facts

- 4,000 rollouts, 1.17 s per rollout over the run (0.84 before rollout 1000; discriminator
  update 27 ms per step), peak 11.4 GB, 78 minutes. Results (local)
  `bench/pixel_readout/results/20261121-1500-x2bgan2_1121_1340/` and
  `…-1505-safe_x2bgan2_1121_1340/`. Against the real 2× image, test 12 / text 5, one seed,
  beside `x2b4k_1121_0950` at the same rollout:

  | rollout | this run: dB | LPIPS | dark lift | B 4k: dB | LPIPS | dark lift |
  |---|---|---|---|---|---|---|
  | 500 | 29.21 / 25.01 | 0.249 / 0.251 | +9.16 / +16.25 | 29.25 / 24.99 | 0.243 / 0.246 | +8.66 / +16.15 |
  | 1000 | 29.47 / 25.10 | 0.259 / 0.271 | +5.05 / +12.92 | 29.46 / 25.09 | 0.272 / 0.284 | +6.96 / +14.23 |
  | 1500 | 29.29 / 25.05 | 0.266 / 0.273 | +6.78 / +14.11 | 29.58 / 25.15 | 0.220 / 0.227 | +6.48 / +13.62 |
  | 2000 | 29.32 / 25.03 | 0.255 / 0.265 | +5.36 / +12.75 | 29.65 / 25.16 | 0.195 / 0.193 | +6.68 / +13.91 |
  | 2500 | 29.45 / 25.18 | 0.211 / 0.217 | +6.46 / +14.02 | 29.79 / 25.25 | 0.177 / 0.175 | +6.11 / +13.19 |
  | 3000 | 29.67 / 25.32 | 0.176 / 0.167 | +6.04 / +13.39 | 29.99 / 25.44 | 0.168 / 0.164 | +5.94 / +12.71 |
  | 3500 | 29.69 / 25.37 | 0.166 / 0.153 | +5.25 / +12.54 | 30.31 / 25.77 | 0.144 / 0.139 | +6.13 / +13.25 |
  | 4000 | 29.82 / 25.52 | 0.158 / 0.145 | +5.47 / +12.69 | 30.45 / 25.92 | 0.141 / 0.137 | +6.52 / +14.00 |

- Rollouts 500 and 1000 are before the term: one command, one seed, two runs. Their gaps:
  0.04 / 0.02 dB, LPIPS 0.006 / 0.005, dark 0.50 / 0.10 at 500; 0.01 / 0.01 dB, LPIPS
  0.013 / 0.013, dark 1.91 / 1.31 at 1000.
- Safe set (6), `safe_x2bgan2_1121_1340`: 27.68 dB, LPIPS 0.219, dark +5.02 (B 4k
  `safe_x2b4k_1121_0950`: 28.20 / 0.204 / +5.44).
- Discriminator margin: 4.2 at rollout 1050, 6.4 at 1250, then 5.9 – 7.3 to the end (7.06
  at 4000). Weight: 0.151 at rollout 1050, 0.012 at 1250, 0.009 – 0.020 over 1250 – 3500,
  0.006 at 4000.
- The user (2026-11-21), at rollout 3650, on the margin: "흠 g margin이 쉽게 따라오질 못하는군
  원래 이랬지?". The records it was answered from: margin 7 – 8 in the 2026-11-20 B+GAN runs,
  5.5 – 7.2 in both micro arms, 6.33 for the frozen head in `x2dprobe_1121_1015`.
- Sheets (local): `~/x2pair_1121/b4k_bgan2/`, `original 2× | B 4k | B + GAN ratio 2 from
  1k`, the two runs' own 768-px crops, 9 images, labelled.
- Claude's read, labels visible, one sheet (`test_01`): the GAN column carries a grain in
  the shaded areas, as the original does and B 4k does not; its lines are thinner and
  darker than B 4k's.
- The user (2026-11-21), after that sheet was shown in the session: "I like this."

#### Interpretation

- Against B 4k at the end the run is 0.63 / 0.40 dB and 0.017 / 0.008 LPIPS behind, and
  about 1 level lower in dark lift. The LPIPS gap at 4000 is near the two runs' own gap at
  rollout 1000 (0.013) and the dB gap is well past it (0.01 – 0.04).
- From rollout 1500 to 2500 the run trails B's curve by 500 – 1,000 rollouts on LPIPS, and
  closes most of it by 3000 – 4000 as the rate decays.
- The margin sits where the frozen-head probe put it: the head does not lower it at any
  ratio tried (1, 2, 4), on a converged head or a moving one. The term still changes the
  output (grain, line weight), so it acts as a texture and edge prior without the
  discriminator being fooled.

### The move into the library, and its check `x2move_1121_1350`

#### Facts

- `library/anima/pixel_head_train.py` (the `x2flex` rows, the DiT rollout, the loss window
  and margins, the recipe / LPIPS / tone terms, the PE-Spatial discriminator, the trainer
  with the GAN start rollout, the meters, the file writer) and the entry
  `scripts/pixel_head/train_x2.py` (`X2TrainRequest`; `--config`, shipped defaults
  `configs/pixel_head/x2.toml`); CPU tests `tests/test_pixel_head_train.py` (7).
- Differences from the bench loop: the head runs at `scale=2` (no ×1 output, no ×1 term at
  weight 0, so the ×1 output layer takes no weight decay), no gradient log (the bench's
  draws from the replay stream every 50 rollouts), no teacher / chain columns, one eval set
  of numbers (×2 against the real 2× image).
- `x2move_1121_1350` (config `configs/x2move_1121_1350.toml`): `x2bo_1120_1815`'s recipe
  through the new entry. At rollout 125, test 12 / text 5: 29.55 / 25.29 dB, LPIPS 0.271 /
  0.258, dark +5.75 / +13.21; `x2bo_1120_1815` at 125: 29.54 / 25.28, 0.270 / 0.251,
  +5.29 / +12.35. 1.02 s per rollout at rollout 150.
- The user (2026-11-21): "250step까진 안해도 되지않아?"; the run was stopped after the
  rollout-125 eval, at about rollout 170. No head file or sheets from it.
- `bench/pixel_readout/`: seven finished probes moved to `_archive/bench/pixel_readout/`
  (local) on the user's word ("x2 현재시점에서 안쓰이거나 완료된건 archive로"); list in the
  bench `README.md`.

#### Interpretation

- The moved loop's gap to the bench run at 125 (0.01 dB, LPIPS 0.001 / 0.007, dark 0.46 /
  0.86) is inside the bench loop's own two-run gap at rollouts 500 and 1000 above: level.
  The GAN path of the new entry has run only in the unit test (a stub discriminator), not
  on the card.

### Plan (Claude's draft)

1. Done: the recipe, the full run (`x2bgan2_1121_1340`), the library entry and its check.
2. Done (2026-11-21): `x2_head.py` lost the closed branches (`--critic*`, `--dprobe`) and takes
   its loss, window, discriminator and sheet helpers from the library; `x2_dmd.py`,
   `x2_dprobe.py`, `x2_pegan.py` went to `_archive/bench/pixel_readout/` (local). It keeps
   the teacher-target and ×1-loss paths and the reads with the teacher / chain columns.
3. The read that is left for `x2bgan2_1121_1340`: renders (`p3_generated.py --x2`) beside
   B 4k and the preview. On the user's read it replaces
   `anima_pixel_head_x2_preview.safetensors`, deployed as in "Deployment plan" below.
4. The next ×2 run goes through `scripts/pixel_head/train_x2.py`; its first run with the
   term on is also the GAN path's first run on the card.

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

### Plan (Claude's draft; runs with step 3 of "Next" above)

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

### Deployed (2026-11-21)

#### Facts

- The user (2026-11-21): "최근에 학습된 x2bgan2 … 의 head_x2를 sorryhyun/anima-pixel-decoder 에
  anima_pixel_head_x2 로 올리자. 아니마 팻말 든 이미지를 리드미에 올리면 될듯? 올린 후 comfy 노드에서는
  배포 플랜대로 동작 변경"; then "리드미에선 use를 compatibility 위에 올려놔야 할듯". The render read
  of step 3 of "Next" was not run; the user's read of the pair sheets stands for it.
- The head files of `…-1500-x2bgan2_1121_1340` and `…-1505-safe_x2bgan2_1121_1340` hold the
  same tensors (the safe read rewrote the stamp). The deploy file
  `…-1500-x2bgan2_1121_1340/anima_pixel_head_x2.safetensors` (sha256 0f99a4e7…, 106 MB): no
  `base.norm_out` / `base.out`, stamp `scales = "2"`, `sigma`, `stream`, `run`, `init`,
  `points`, `x2_target`, `pegan`; its ×2 output equals the source file's on a random input.
- Library (plan items 1, 2): `drop_x1_output`, `default_scale`, `check_scale` in
  `library/anima/pixel_head.py`; `load_pixel_head` reads `scales`; `PixelTap.decode`,
  `--pixel_scale`, `PIXEL_SCALE=` and `GenerationRequest.pixel_scale` default to the file's
  scale; `resolve_pixel_head_path(None, 2)` takes the new catalog row `anima_pixel_head_x2`
  (same pack). `train_x2.py` drops the ×1 output layer at build and `save_x2_head` stamps
  `scales`. Tests: `test_pixel_head.py`, `test_pixel_head_train.py`, `test_downloads.py`, 30 pass
  on CPU.
- Card renders (`inference.py`, base 28 steps CFG 4 seed 42, the ANIMA sign prompt and its
  `monochrome, comic` variant): with `--pixel_head <deploy file>` and no `--pixel_scale` the log
  reads `scales (2,), decoding ×2` and the image is 2048². Claude's read of the crops: colour
  lines sharper than the VAE's bicubic ×2; on the monochrome shorts the screentone comes out as
  a wavy pattern instead of dots (on the card under Known limits).
- Hub `sorryhyun/anima-pixel-decoder`: `anima_pixel_head_x2.safetensors` (commit beb531f4);
  card with Use before Compatibility, a ×2 section with the samples, the preview moved to
  `old/anima_pixel_head_x2_preview.safetensors` (commit 96a559b7). Local copy of the card:
  `deploy/README.md`. The catalog fetch writes the same sha256.
- Node `~/ComfyUI-Anima-PixelDecoder` (plan item 3): `scale` = `auto` / `1` / `2`, `auto` by
  default (the file's scale); `_vendor/pixel_head.py` from `build_pixel_vendor`; 0.3.0; README ×2
  section. Pushed to GitHub and published to the registry as 0.3.0 (2026-11-21, on the user's
  "ㅇㅇ"). Then 0.3.1 (d02f904) without the `scale` input, on the user's "스케일 항목을 그냥 없애면
  안돼?": the file decides; a two-scale file (the old preview) decodes at ×1. Not run inside ComfyUI.

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
- The new entry's GAN path has not run on the card; the seed twin of a full ×2 run does not
  exist (the two-run gaps in "Outcome of `x2bgan2_1121_1340`" are from rollouts 500 and 1000).
- ×4 not started.
- Arm B: whether masters are upscales or re-encoded (matters most on 896 / 768, near 1:1);
  a 2× source for the manga pages; a held-out set inside the 1024 set.

