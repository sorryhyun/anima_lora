# Step 0: how deep the fit has to go, and what more pairs buy (2026-11-17)

Probes of where the anchor students' breaks come from, the full-dataset run `turbo_1117_0823`, its
rank 96, lower-LR and feature-matching variants, and the 7 of 28 target.

`teacher_handoff.md` left the anchor student (`2209`) with layouts near the teacher's and broken
objects, and the DM arm on steps 1–3 (`2221`) with a mixed finish. This report holds the probes
that say where the breaks come from and how far the step-0 fit is from removing them, the run on
the full-dataset cache (`turbo_1117_0823`) and its rank 96 twin (`turbo_1117_0918`).

Every probe is `bench/turbo/probe_handoff.py --dit models/diffusion_models/anima_merged15.safetensors
--ts_handoffs 1 --st_handoffs ,` on `judge_hard.txt` (group / café / guitarist), seeds 1–4,
768 × 1344, CPU-drawn noise, so the grids of different probes pair seed by seed. The anchor is
14 of 28 unless a section says otherwise.

The render reads are Claude's. The user's read is the gate.

## The breaks are step 0's residual: mix probe on `2209` and `2221`

### Facts

`--step0_mix a0.25,a0.5,a0.75,lp8,hp8` (arms as in `step0_fit.md`): step 0 is the student's own
velocity moved toward the anchor target by part of its residual, and the student draws steps 1–3.
Run `bench/turbo/results/20261117-0817-step0_mix_2209_2221/`, 4 min. Crops
`output/tests/turbo4/cmp/step0mix_2209_{bocchi,ba5}.jpg` (columns `S`, `M0.25`, `M0.5`, `L8`, `H8`,
`A`).

`2209` EMA 1024, mean of 12:

| step 0 | anchor MSE left | cos to the teacher's image | sameness (teacher 0.314) |
|---|---|---|---|
| `S` (the student's own) | 0.078 | 0.760 | 0.296 |
| `M0.25` | 0.044 | 0.880 | 0.264 |
| `H8` (the residual without its 8× band) | 0.036 | 0.902 | 0.264 |
| `L8` (the 8× band only) | 0.030 | 0.950 | 0.251 |
| `M0.5` | 0.019 | 0.940 | 0.255 |
| `M0.75` | 0.005 | 0.961 | 0.248 |
| `A>S@1` (the target itself) | 0 | 0.967 | 0.242 |
| `T>S@1` (the teacher's own step 0) | – | 0.941 | 0.303 |

- Guitarist: at `M0.25` the giant hand and twisted body of seed 1 and the doubled neck of seed 4 are
  gone and the figure is the teacher's. At `M0.5` the cosine is at the level of the teacher's own
  step 0. `L8` on seed 4 adds a floating hand.
- Group: the smeared face in the lower half of seed 1 is still there at `M0.25` and `M0.5`, and
  clean at `L8` and with the target.
- `L8` and `H8` each remove about half of the MSE and each recovers most of the cosine. The low band
  is 0.46 of `2209`'s error (0.59–0.68 on the merged14 students of `step0_fit.md`).
- With the target or the teacher's step 0 in front of them, the student's steps 1–3 finish at
  0.94–0.97.

`2221` EMA 250 against `2209`, its start: `S` cos 0.753 (0.760), `T>S@1` cos 0.935 (0.941), anchor
MSE 0.080 (0.078), sameness 0.287 (0.296). Step 0 and steps 1–3 both moved slightly down.

### Interpretation

- A quarter of the residual removes most of the guitarist's breaks. The group needs more.
- Both bands carry it. The regression has already taken more of the low band than the merged14
  students had.
- So the target for a fit is a held-out anchor MSE of 0.02–0.04 on these prompts, a third to a half
  of where the anchor students sit. The mix scales the whole residual; training does not have to
  lower it in that shape.
- The student's steps 1–3 are not what breaks.
- On `2221`, the DM terms trained steps 1–3 on the teacher-forced input, where they were already
  clean, and gave no signal on the inputs the student's own step 0 hands them.

## The random draw of `teacher_handoff.md` is closed

### Facts

That report's next arm drew anchor-only and DM-only iterations at random so the two gradients
would stop sharing a clip. On `2221` the pre-clip student gradient norm is 0.13 at the median and
0.19 at the 90th percentile of the 50 logged iterations, against the clip at 1.0. The anchor alone
(`2209`) runs at 1.2–1.3; the log does not split the terms. The cached anchor is 0.4 s of a 4.5 s
iteration.

### Interpretation

On `2221` the clip does not bind. The anchor's share at `div_weight` 0.1 is about 0.12, most of
`2221`'s norm. Drawn apart, an anchor iteration (norm 1 after the clip) would outweigh a DM
iteration (about 0.1) through AdamW's shared second moment. With the layouts held on `2221` the
draw was a cost knob by that report's own rule.

## Repeating pairs does not generalise: `2019` held out

### Facts

`2019` saw 128 pairs about four times and ends at a train anchor MSE of 0.036. Run
`bench/turbo/results/20261117-0822-handoff_m15_2019_p64/`.

| student | pairs | train anchor MSE | held-out anchor MSE | `S` cos |
|---|---|---|---|---|
| `2019` EMA 500 | 128, about 4 times each | 0.036 | 0.089 (0.072 / 0.089 / 0.107) | 0.733 |
| `2209` EMA 1024 | 1024, once each | 0.065 | 0.078 | 0.760 |

### Interpretation

The fit that comes from seeing a pair again stays on that pair. More distinct pairs is what moves
the held-out number.

## The full-dataset cache: `turbo_1117_0823`

### Facts

`2209` on `post_image_dataset/turbo_anchor_cache/m15_k14of28_pall` (3242 prompts × 2 ε = 6484
pairs), `grad_accum 2`, 6484 iterations: 3242 student updates, one pass (epoch 0 reads draw 0,
epoch 1 draw 1, so an update averages two prompts). Config `../configs/turbo_1117_0823.toml`,
2.5–2.9 it/s, 43 min. Probe `bench/turbo/results/20261117-0908-handoff_m15_0823_pall/`; sheets
`output/tests/turbo4/cmp/pall_{bocchi,ba5,cafe}.jpg` (rows teacher, `2209` `S`, `0823` `S`, `0823`
`A>S@1`).

Read fixed in the config header, against `2209` EMA 1024:

| | `2209` | `0823` EMA 6484 | read | |
|---|---|---|---|---|
| held-out anchor MSE | 0.078 | 0.070 (0.066 / 0.080 / 0.062) | 0.044 or below | not met |
| `S` cos | 0.760 | 0.801 (0.801 / 0.810 / 0.792) | toward 0.88 | a third of the way |
| `T>S@1` cos | 0.941 | 0.944 | held at 0.94 | held |
| sameness `S` (teacher 0.314) | 0.296 | 0.300 | – | |
| low-band share of the error | 0.46 | 0.43 | – | |

Train anchor MSE by tenth of the run: 0.088, 0.081, 0.077, 0.073, 0.070, 0.068, 0.065, 0.059,
0.062, 0.058. Every pair is seen once (the last value is held-out on the dataset's own captions);
the judge prompts read 0.070.

- Renders. Guitarist seed 1 is repaired (the teacher's pose, a whole guitar; a microphone
  floats without its stand). Seed 4's guitar is one piece with a stray strand at the headstock.
  Seed 2 still breaks (the body sits apart from the neck). The group's lower halves are where
  `2209` left them: the pastry with a face on seed 1, the tangled halo and hand on seed 2, layouts
  off the teacher's on seeds 3 and 4. The café keeps the teacher's blue aprons and garbled signs.
- With the target as step 0, `0823`'s steps 1–3 draw the teacher's image on all twelve
  (`A>S@1` cos 0.966).
- Held-out anchor MSE against cache size:

  | cache pairs | student updates | held-out anchor MSE |
  |---|---|---|
  | 128 (`2019`) | 500 | 0.089 |
  | 1024 (`2209`) | 512 | 0.078 |
  | 6484 (`0823`) | 3242 | 0.070 |

  About 0.01 for each 6–8× in pairs, and the last step also had six times the updates.

**Verdict: not passed on the read fixed before the run. `0823` is the best anchor student so far and
the start of the next arms.** The user's read of the sheets (2026-11-17): clear progress, at least
over Hub v3 on the v4 merged teacher.

### Interpretation

The slope is shallow. More data on this recipe does not reach 0.044.

## Rank 96 ties rank 64: `turbo_1117_0918`

### Facts

`0823` at rank 96 / alpha 220 (alpha / √rank kept at 22.5, the √r law), so the student starts from
the whole r96 delta instead of its r64 truncation. Same cache, seed and pair order. Config
`../configs/turbo_1117_0918.toml`, 2.5 it/s, 43 min. Probe
`bench/turbo/results/20261117-1004-handoff_m15_0918_r96/`; sheets
`output/tests/turbo4/cmp/r96_{bocchi,ba5,cafe}.jpg` (rows teacher, `0823` `S`, `0918` `S`).

| | `0823` (r64) | `0918` EMA 6484 (r96) | read | |
|---|---|---|---|---|
| held-out anchor MSE | 0.070 | 0.069 (0.066 / 0.079 / 0.063) | below 0.070, toward 0.044 | tie |
| `S` cos | 0.801 | 0.811 (0.831 / 0.812 / 0.788) | above 0.801 | +0.010, on the group |
| `T>S@1` cos | 0.944 | 0.948 | held at 0.94 | held |
| sameness `S` (teacher 0.314) | 0.300 | 0.299 | near the teacher's | same |
| low-band share of the error | 0.43 | 0.43 | – | |

Train anchor MSE by tenth of the run, `0918` / `0823`: 0.089 / 0.088, 0.085 / 0.081, 0.080 / 0.077,
0.075 / 0.073, 0.073 / 0.070, 0.070 / 0.068, 0.067 / 0.065, 0.061 / 0.059, 0.064 / 0.062,
0.059 / 0.058.

- The held-out anchor MSE is the same to 0.001.
- Rank 96 trails in train anchor MSE by 0.003–0.004 while the LR is high and by 0.001 at the end of
  the cosine.
- Renders: no gain. Guitarist seed 1 has a large stray hand at the top that `0823` did not draw;
  seed 2 still breaks and seed 4 is alike. In the group's lower halves the pastry with a face of
  seed 1 is gone and the girl beside it loses her face; seed 2 has a white blob on the lying girl.

Verdict: a tie with `0823`.

### Interpretation

- Rank is not what limits the fit.
- The pairs and their order are the same in both runs, so the train-MSE gap is the student, and it
  closes as the LR comes down.

## A lower student LR, short: `turbo_1117_1020` / `1021`

### Facts

`0823` at rank 64 on `m15_k14of28_p1024` (the p512 keys plus 512 more of `pall`, hardlinked; 2048
pairs, 2048 iterations, `grad_accum 2`), `student_lr` 5e-5 (`1020`) against 2e-5 (`1021`), nothing
else changed. Configs `../configs/turbo_1117_102{0,1}.toml`, probe
`bench/turbo/results/20261117-1049-handoff_lr_ab_1020_1021/`, crops
`output/tests/turbo4/cmp/lr_ab_p{1,2}.jpg` (teacher, `1020` `S`, `1021` `S`).

| final EMA | `1020` 5e-5 | `1021` 2e-5 | read (fixed in the config header) |
|---|---|---|---|
| held-out anchor MSE | 0.0741 | 0.0723 | 2e-5 below: by 0.002 |
| `S` cos | 0.785 | 0.772 | up: down |
| `T>S@1` cos | 0.944 | 0.938 | held at 0.94 |
| sameness `S` (teacher 0.314) | 0.285 | 0.260 | not down: down |

Train anchor MSE by tenth (the first half reads each pair once): 5e-5 0.089, 0.083, 0.079, 0.082,
0.081, 0.073, 0.068, 0.064, 0.066, 0.066; 2e-5 0.086, 0.075, 0.072, 0.072, 0.071, 0.067, 0.065,
0.062, 0.063, 0.066.

- Renders tie. Guitarist seed 1 grows a giant hand on both and seed 2 breaks on both; seed 4 keeps
  the guitar's neck at 2e-5 only. Café seed 1 is nearer the teacher at 2e-5, seed 3 has an odd arm
  there. The group's lower halves break alike; seed 3's layout drifts further at 2e-5.
- Verdict on the probe: not passed. The held-out MSE is level and `S` cos and sameness fall.
- The user's read of the renders (2026-11-17): 2e-5 shows promise. The LR stays open: 2e-5 on
  the full cache (`turbo_1117_1054`, `0823` with only the LR changed) is compared with `0823` before
  it is closed.

## A lower student LR, full length: `turbo_1117_1054`

### Facts

`0823` at `student_lr` 2e-5 (cosine to 2e-6), nothing else changed: the full cache, one pass, rank 64.
Config `../configs/turbo_1117_1054.toml`, probe
`bench/turbo/results/20261117-1204-handoff_m15_1054_lr2e5_pall/`, sheets
`output/tests/turbo4/cmp/lr2e5_full_p{0,1,2}.jpg` (teacher, `0823` `S`, `1054` `S`, seeds 1–4). 42 min.

| final EMA | `0823` 5e-5 | `1054` 2e-5 | read (fixed in the config header) |
|---|---|---|---|
| held-out anchor MSE | 0.0695 | 0.0674 | below 0.070: passed |
| `S` cos | 0.801 | 0.792 | not below 0.801: below |
| `T>S@1` cos | 0.944 | 0.937 | held at 0.94: just under |
| sameness `S` (teacher 0.314) | 0.300 | 0.284 | not down: down by 0.016 |
| anchor error low share | 0.429 | 0.413 | |

Per prompt the MSE falls on all three (0.066 / 0.080 / 0.062 → 0.064 / 0.078 / 0.060).

- Renders (my read; the verdict on them is the user's): close to `0823`. Guitarist seed 1 grows
  the giant hand the short runs had, which `0823` did not; seed 2's body is no better; seeds 3 and 4
  tie. The group's lower halves break alike. Café: seed 4 adds a sign with garbled text, the rest
  tie.
- Verdict on the probe: not passed. The held-out MSE passes; `S` cos, `T>S@1` and sameness fall.

### Interpretation

The short A/B's pattern again at full length: 2e-5 fits the anchor a little closer, and the images
move further from the teacher's and nearer each other.

## Feature matching: the floor probe

### Facts

`bench/turbo/probe_fm_floor.py` on 16 pairs of `m15_k14of28_p1024`, the v1.1 init as the student,
tap block 14 (middle), σ_k 0.75 (`bench/turbo/results/20261117-1052-fm_floor/`). Relative feature
L2, mean:

| arm | value |
|---|---|
| `own_det` (no-grad features of x_t twice) | below 1e-5 |
| `own_path` (training path, grad + block checkpoint, against no-grad) | below 1e-5 |
| `student` (x_s against x_t; anchor MSE 0.115) | 0.734 |
| `jitter` (v_target plus white noise of the student residual's rms) | 0.085 |

- The student's residual moves the teacher's features 8.6 times as far as white noise of the same
  rms. The per-pair ranking differs from the MSE's (pair 9: MSE 0.069, feature 1.06; pair 10:
  0.249, 0.71).
- At `fm_weight` 1.0 (the 64-iteration smoke) the feature gradient into `v` is 52–270 times the
  MSE's and the pre-clip norm 7–45 against the anchor's 1.2. 1.30 it/s against the anchor's 2.6,
  peak 12.7 GB.

### Interpretation

- No floor: the two forward paths agree.
- White noise is the easy control (patchify and the blocks damp its high band), so the 8.6× says the
  term is not a rescaled MSE, not yet that it reads the breaks.
- `fm_weight` 0.01 puts the feature and MSE gradients near par.

## Feature matching at 0.01: `turbo_1117_1105`

### Facts

`1020` with `dpdmd.fm_weight` 0.01 (tap block 14, σ_k 0.75), nothing else changed. Config
`../configs/turbo_1117_1105.toml`, probe `bench/turbo/results/20261117-1120-handoff_fm_1105/`, sheets
`output/tests/turbo4/cmp/fm_1105_p{0,1,2}.jpg` (`1020` `S` on top, `1105` `S` below, seeds 1–4).
24 min at 1.46 it/s.

| final EMA | `1020` | `1105` fm 0.01 | read (fixed in the config header) |
|---|---|---|---|
| held-out anchor MSE | 0.0741 | 0.0739 | not above 0.076: held |
| `S` cos | 0.785 | 0.786 | up: level |
| `T>S@1` cos | 0.944 | 0.943 | held at 0.94 |
| sameness `S` (teacher 0.314) | 0.285 | 0.279 | not down: down by 0.006 |
| anchor error low share | 0.449 | 0.447 | |

Every gap is under the `1020` / `1021` A/B spread (0.014 on `S` cos, 0.025 on sameness).

- `train/fm_grad_ratio` ran 1.2 over the first tenth and 1.5–1.75 after it: the weighted feature
  gradient into `v` sat at or above the MSE's for the whole run. The relative feature L2 fell 0.64
  → 0.55, the diversity term 0.089 → 0.067; grad norm 1.2–1.4 after the first tenth, no non-finite
  value.
- Renders near-identical to `1020`. Guitarist seed 1 keeps the giant hand, seed 3 is the same image,
  seed 4's guitar neck is a little tidier, seed 2 grows an extra arm behind the head. The group:
  seed 2's lower pink-haired face is cleaner, seed 4's two overlapping heads at bottom right stay.
- Verdict: not passed. Next arm `fm_weight` 0.1 (`turbo_1117_1123`).

### Interpretation

At par with the MSE's gradient the term barely moves the student. At 0.1 the ratio would be about
15 by this run's scale, under the smoke's 52–270 at 1.0.

## The 7 of 28 target, probed and parked

### Facts

`0823` against the anchor on the first 7 of 28 teacher steps (σ 1 → 0.9), no training on it:
`--anchor_k 7 --anchor_steps 28 --step0_mix a0.25,a0.5,a0.75`, run
`bench/turbo/results/20261117-0916-step0_mix_0823_on7of28/`.

| step 0 | MSE left to the 7 of 28 target | cos to the teacher's image |
|---|---|---|
| `S` | 0.037 (0.028 / 0.043 / 0.039) | 0.801 |
| `M0.25` | 0.021 | 0.863 |
| `M0.5` | 0.009 | 0.906 |
| `M0.75` | 0.002 | 0.933 |
| `A>S@1` | 0 | 0.944 |

- Half of its residual reads 0.906 where half of the 14 of 28 residual reads 0.940 (on `2209`), and
  its ceiling is 0.944 against 0.967. The low band is 0.70 of the error to it.
- Parked on the user's call. The 14 of 28 caches stay in use.

### Interpretation

- The shorter target is nearer in MSE and gives less for it.
- The 14 of 28 chord ends at σ 0.75, the first boundary of the 2-step grid, so a step 0 fitted to it
  is also the 2-step student's step 0 (`../idea.md` 3).
