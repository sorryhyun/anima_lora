# ×2 output — plan (2026-11-19, updated 2026-11-22)

The P5 topic "×2 / ×4 output" of `plan.md`, as its own plan: decode and 2× upscale in one
readout of the DiT's last features, so a 1024-tier render leaves the head at 2048. This file
holds what is ahead; everything that ran is in `reports/` (index: `reports/README.md`):

- arm A `x2_arm_a_2026_11_19.md`, arms 1 + 2 `x2_lpips_dc_2026_11_19.md`, the preview release
  `x2_preview_release_2026_11_19.md`, arm 3 `x2_critic_2026_11_20.md`, the feature probe
  `feat_probe_2026_11_20.md`, the state after arm 3 and the finished-arms table
  `x2_state_2026_11_20.md`;
- arm B (the real 2× image as the target) `x2_arm_b_2026_11_20.md`, B at 4,000 rollouts
  `x2_b4k_2026_11_21.md`, the discriminator probes `x2_disc_probes_2026_11_21.md`, the GAN
  micro arms and the glyph read `x2_gan_micro_2026_11_21.md`;
- the settled recipe, its full run `x2bgan2_1121_1340`, the training code's move into the
  library, the deployment plan and the "Deployed" record `x2_full_run_deploy_2026_11_21.md`;
- the tiled pattern on that file and the GAN finetune arms `x2gft_1121_2225` /
  `x2gftw_1121_2300` / `x2gftl_1121_2300` `x2_tile_gft_2026_11_22.md`.

Code: `scripts/pixel_head/train_x2.py` on `library/anima/pixel_head_train.py` (training of the
settled recipe), `bench/pixel_readout/x2_head.py` (the earlier arms and the whole-image reads),
`x2_overfit.py` (fit probe), `x2_masters.py` / `x2_flex.py` (real 2× targets and the set),
`x2_pair.py` (pair sheets); results under `bench/pixel_readout/results/` and
`output/pixel_head/` (local).

## State

- The form: a ×1 head plus a second 2× stage (512 → 4 × 256 pixel shuffle, ConvNeXt blocks at
  4-px spacing, an 8×8 RGB patch per position, 32×32 output pixels per token). The shipped ×2
  file holds the ×2 output only; the file decides the scale.
- Shipped: Hub `anima_pixel_head_x2.safetensors` = `x2bgan2_1121_1340` (arm B's recipe with the
  PE-Spatial GAN term at ratio 2 from rollout 1000 of 4,000): 29.82 / 25.52 dB, LPIPS 0.158 /
  0.145 against the real 2× image (test 12 / text 5).
- Reported by the user on that file (2026-11-21): a systematic tiled pattern on fine texture,
  and a region of fine dense strokes coming out blurred.
- The GAN finetune heads (`x2gft`, `x2gftw`: ratio 2 on the converged B 4k head, 1,000
  rollouts) were not picked for the Hub; the shipped file stays through the sequence below.

## The sequence after the ×1 head is fixed (2026-11-22)

### Facts

- The user (2026-11-22): "x2는 어차피 x1에서 init한 형태이니까, x1를 먼저 5e-5로 돌려보고 x1를
  확정한다음 x2도 longer and better recipe (b only -> gan 마무리) 로 하면 좋겠어."
- No finetune head was picked for the Hub; `x2bgan2_1121_1340` stays the Hub file through this
  sequence.
- Order:
  1. ×1: 12,000 rollouts at LR 5e-5 (`plan.md`, "The long run at the lower LR"; designed as
     `x1ll_1122_0855`, postponed by the user, not run). The user's read fixes the ×1 head.
  2. ×2, B only: `scripts/pixel_head/train_x2.py` with `init` = the fixed ×1 head and
     `pegan_ratio` 0 (arm B: the real 2× image, base points σ 0.0577 – 0.1364, LPIPS 0.1,
     dc 1.0), longer than `x2b4k_1121_0950`'s 4,000 rollouts.
  3. ×2, GAN finish: the `x2gft` form on that head (`resume`, `pegan_ratio` 2, `pegan_start` 0,
     a new cosine cycle, 1,000 rollouts), with a seed twin.
  4. Read: the `text_04` place, the monochrome render and the fine-stroke renders, the new head
     beside the shipped file. The user's read decides the Hub file.
- Records the steps rest on: `x2b4k_1121_0950`'s ×2 LPIPS on test read 0.196 at rollout 0, 0.272
  at 1,000, 0.195 at 2,000 and 0.141 at 4,000 (still falling at the end of its cosine); `x1lowlr`
  against `realr` and `x1long` against `realr` on ×1 (`plan.md`, Outcome); `x2gftl`'s stripes at
  3,000 GAN rollouts against none at 1,000 (`reports/x2_tile_gft_2026_11_22.md`).
- Not set yet: step 2's length and LR. No ×2 run has been made at another LR or past 4,000
  rollouts; the values are written into the run's config when step 1 is read.

### Interpretation

- `x2b4k` spent half its run under its start, as `realr` did on ×1, and ended while still
  improving; the ×1 batch says a warm start gains from a lower LR and from length. Step 2 carries
  that over, and its config names the carry-over as unmeasured on ×2.
- The GAN stays a short finish on a converged head: that form has no pinwheel on two seeds, and
  length on it brings a regular pattern back.
- The ×1 head changes under step 2 (the base trains with the second stage), so the ×2 file's own
  ×1 column is read against the fixed ×1 head, as in "Keeping ×1" below.

## Next arms

Claude's drafts, none queued, behind the sequence above. Micro arms on arm B's recipe, read as
arm B is. The numbering is kept: the reports cite it.

1. Done: GAN on PE-Spatial features (`reports/x2_arm_b_2026_11_20.md`,
   `reports/x2_gan_micro_2026_11_21.md`, `reports/x2_full_run_deploy_2026_11_21.md`).
2. The ×2 stage's width (1.6 M new parameters at width 256), if arm B reads level or the GAN
   arm sharpens too little.
3. The turbo point at ×2: postponed by the user (2026-11-20).
4. Keeping ×1: the ×1 loss weight, or a lower LR on the base, against the drift under ×2
   training. Read on the ×1 column of any arm above before a separate run.

Not taken up: the previous step's feature as a second input (+0.24 dB on a linear readout,
`reports/feat_probe_2026_11_20.md`).

## Open

- One twin pair only (`x2gft` / `x2gftw`, the GAN finetune form); no spread for the other ×2
  recipes.
- 16-window fit: a ceiling or slow convergence is not separated (the cosine schedule ends as
  the curve flattens); the 2-window numbers rest on one window per point.
- Numbers are teacher-forced only. Base rollouts through the ×2 head exist as renders (the
  card's, the monochrome and fine-stroke ones), with no number on them; no turbo rollouts; no
  end-to-end time beside VAE → ResShift (teacher 3.16 s per image against the head's 12.9 ms).
- The dark-pixel lift is not split into softness and tone; `tone_probe.py` does not read the
  two-column sheets.
- No seed twin of a full ×2 run (the two-run gaps in
  `reports/x2_full_run_deploy_2026_11_21.md` are from rollouts 500 and 1000).
- ×4 not started.
- Arm B: whether masters are upscales or re-encoded (matters most on 896 / 768, near 1:1);
  a 2× source for the manga pages; a held-out set inside the 1024 set.
