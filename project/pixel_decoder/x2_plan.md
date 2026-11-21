# ×2 output — plan (2026-11-19, updated 2026-11-21)

The P5 topic "×2 / ×4 output" of `plan.md`, as its own plan: decode and 2× upscale in one
readout of the DiT's last features, so a 1024-tier render leaves the head at 2048. Finished
arms are in `reports/`: arm A and its design `x2_arm_a_2026_11_19.md`, arms 1 + 2
`x2_lpips_dc_2026_11_19.md`, the preview release `x2_preview_release_2026_11_19.md`, arm 3
`x2_critic_2026_11_20.md`, the feature probe `feat_probe_2026_11_20.md`, the state after arm 3
`x2_state_2026_11_20.md`, arm B `x2_arm_b_2026_11_20.md`, B at 4,000 rollouts
`x2_b4k_2026_11_21.md`, the discriminator probes `x2_disc_probes_2026_11_21.md`, the GAN micro
arms and the glyph read `x2_gan_micro_2026_11_21.md`. Code:
`bench/pixel_readout/x2_head.py` (head and training), `x2_overfit.py` (fit probe), `x2_dmd.py`
(arm 3's critic), `x2_masters.py` (real 2× targets), `x2_pair.py` (pair sheets); results under
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

### Plan (Claude's draft)

1. The recipe is fixed (Facts above). The full run's shape was set by the user: the term
   from rollout 1000 of 4,000 (Design above).
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
4. The full run of the settled recipe (`x2bgan2_1121_1340`, started through the bench
   script on the user's word before the move), read against
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

