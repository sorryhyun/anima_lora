# Turbo v4 plan (as of 2026-10-10)

The plan from the Phase 0d result to a 4-step v4 student. How our `[dmad]` path
differs from the official DMAD code is in `difference.md`. The phase history is in
`docs/proposal/turbo_dmad.md`.

## Where we are: Phase 0d

**Arm.** `anima_turbo_p0d_dmad_lasttap_w3` (job `20261010-105106-f394ed`). It is the
p0c recipe (raw signal `signal_rms` 0, λ_T = λ_R = 150, L_CDM 1.0, disc warmup 0,
seed 42) with two changes: `--dmad_feature_block_idx 27` (disc tapped at the last
block, as in the official H3 / Wan recipes) and `--dmad_window 3`. 500 steps,
15 s/step (p0c: 11.9), peak ~14.7 GiB. Rows:
`output/logs/anima_turbo_p0d_dmad_lasttap_w3.progress.jsonl`.

**Scalars against p0c (50-step bin means).**

| Steps | margin T (0d / p0c) | BCE T (0d / p0c) | raw g_T (0d / p0c) | `x_pred` std (0d / p0c) |
|---|---|---|---|---|
| 250–300 | 1.8 / 3.0 | 1.02 / 0.86 | 4.8e-4 / 8.3e-4 | 0.61 / 0.61 |
| 300–350 | 2.2 / 8.6 | 0.92 / 0.24 | 1.8e-3 / 8.2e-5 | 0.64 / 0.78 |
| 350–400 | 1.2 / 14.1 | 1.18 / 0.009 | 1.5e-3 / 3.2e-6 | 0.64 / 1.45 |
| 450–500 | 1.6 / 14.4 | 1.08 / 0.08 | 1.5e-3 / 2.0e-5 | 0.66 / 2.45 |

- **No saturation.** Disc margin peaks at 8.1 (step 160) and at 6.9 (step 185).
  Both times it returns to ~1 within 10 steps. The raw input gradient never falls
  below 2.1e-5, and `x_pred` std stays at 0.58–0.66.
- **The signal is strong and spiky.** The student `grad` averages 0.4–0.56 after
  step 300, against DP-DMD's ~0.2. There are seven spikes above 1 (the largest is
  4.87 at step 145); p0c had one by step 500. The disc absorbs them, but nothing
  caps them in raw mode.
- **Confound.** Tap and window changed together, so this run alone cannot say
  which one removed the collapse. The working read is the tap: middle-block
  features go flat once the disc separates, while last-block features still carry
  direction.

**Read (4 steps, cfg 1.0, `bench/turbo/real_prompts.txt`, seed 42, prompts
0 / 1 / 8 / 13).** Renders are in `output/tests/p0d_lasttap_w3_{250,500}`. They are
compared with `v11_init`, `p0b_dpdmd_500`, `p0c_dmad_cdm_{250,500}`.

- **500:** intact on all four prompts, where p0c 500 is prompt-independent stripe
  noise. Compositions sit near the init with small re-poses. Line and colour are
  as clean as p0b DP-DMD 500.
- **250:** re-poses the most of any arm (prompt 0 kneels with a wink; prompt 8
  legs up), like p0c 250. Hair strands are a little messier.
- No clear quality gain over p0b DP-DMD at grid scale. The win is stability.

**Uncommitted with this result.** Gap routing (`--dmad_gap_routing`, off by
default, ported from the H3 trainer; not run past step ~45), and the
`dmad.disc_warmup_steps` default lowered 50 → 0 (no official recipe warms the
critic).

## Steps

### 1. 1500-step stability rerun

The Phase 0d recipe at `--iterations 1500`. Phase 0 refuses `--resume`, and the
student's cosine schedule is sized to the iteration count, so it is a fresh run.

```bash
make turbo ARGS="--gan_loss_weight_gen 0 --iterations 1500 --save_every 250 \
  --validate_every_n_steps 0 --dmad --dmad_signal_rms 0 --dmad_disc_warmup_steps 0 \
  --dmad_feature_block_idx 27 --dmad_window 3 \
  --output_name anima_turbo_p0e_dmad_lasttap_w3_1500 --queue"
```

- ~6.3 h at 15 s/step.
- **Pass:** no saturated stretch (margin > 10 with raw g < 1e-5 for more than
  ~20 steps), `x_pred` std stays near 0.6, and the 250-step renders stay intact
  through 1500 (p0c was collapsed at 500 and 750).
- **If it fails:** cap the raw signal (a per-sample norm ceiling, or `signal_rms`)
  before touching the disc. The spikes are the visible trigger.
- **Optional, later:** a window-4 arm to separate the tap from the window.

### 2. r256 merged LoRA on the current artist images

An r256 LoRA trained on the current artist set, merged.

Open:
- Which artist set, and whether this is a new run or a merge of existing ones
  (`merge_loras.py` vs soup).
- Whether "merged" means baked into the DiT (`make merge`) so it can serve as the
  **teacher** in step 4, or kept as a LoRA used only to generate data in step 3.

### 3. Self-generated dataset

Generate images from prompts with the step 2 model, then run them through
`make preprocess` so they cache as an ordinary `data_dir`.

Open:
- Prompt source and count, sampler steps / CFG, resolution tiers.
- Captions: the generating prompt is the caption. TE caching reads only the
  caption beside the resized image, so the prompt must be written next to each
  image.
- Curation pass: whether to drop failed generations before caching.

### 4. 4-step v4

Distill a 4-step student on the step 3 dataset, with DMAD (the step 1 recipe) or
DP-DMD.

What the dataset does differs between the two:
- **DMAD:** head R scores the student against dataset latents, so the
  self-generated images are the "real" distribution the student is pulled toward.
  The teacher samples (head T) still come from the teacher DiT.
- **DP-DMD:** the DM term and L_CDM take only captions and the latent shape from
  the batch; the target distribution is the teacher's. Dataset latents reach the
  student only through the `[gan]` side term (`weight_gen` 0.03 in the shipped
  `turbo.toml`). So a self-generated dataset changes the student mostly through its
  prompts, **unless the teacher is the step 2 merged DiT**.

So the teacher choice in step 2 decides the method choice. With a base-DiT
teacher, the new data drives DMAD's head R but only DP-DMD's small GAN term. With
a merged-DiT teacher, both methods target it.

Compare at the same steps and seed on `bench/turbo/real_prompts.txt` plus a
diversity read (`val/div_*`), against v1.1 (`v11_init`).
