# The turbo v1.1 delta on merged15: the best steps 1–3, an unfitted step 0 (2026-11-16)

Probes of the official v1.1 delta on merged15, Hub v3's training log, and the runs seeded from the
delta (`turbo_1116_1552`, `1558`, `1653`, `1730`) with the anchor-target probe.

The official turbo v1.1 delta (`models/extracted/anima_turbo_v1.1_delta_r96_asvd_adaln.safetensors`)
is the file Hub v3's run seeded its student and fake from. After the 1k failure (`m15_1k.md`) the
user's question was whether seeding from it, as v3 did, is what the merged-teacher runs miss. This
report holds the probes of the delta itself and what Hub v3's training log shows; the runs from this
init (`turbo_1116_1552`, `turbo_1116_1558`) are added when they finish.

## Probes, no training

### Facts

`bench/turbo/probe_handoff.py --dit models/diffusion_models/anima_merged15.safetensors --ts_handoffs 1
--st_handoffs ,` on `judge_hard.txt`, seeds 1–4: the delta alone
(`bench/turbo/results/20261116-1550-handoff_m15_v11_untrained/`) and with `--step0_lora` = Hub v3
(`20261116-1554-cross_v3s0_v11123_m15/`). The Hub v3 and `1355` rows are from
`20261116-1533-handoff_m15_1k/`, `0958` from `20261116-1054-handoff_m15_cdmcond/`.

| | sameness `S` | `S` cos | `T>S@1` cos | `T>S@1` sameness | anchor MSE |
|---|---|---|---|---|---|
| teacher m15 | 0.315 | | | | cond 0.165 |
| v1.1 delta | 0.396 (0.402 / 0.500 / 0.286) | 0.677 | 0.932 (0.914 / 0.946 / 0.936) | 0.301 | 0.138 (0.068 / 0.104 / 0.241) |
| Hub v3 | 0.296 | 0.635 | 0.805 | 0.260 | 0.062 |
| `0958` EMA 250 (cold) | 0.442 | 0.608 | 0.841 | | 0.138 |
| `1355` EMA 1k (cold) | 0.449 | 0.532 | 0.781 | 0.329 | 0.128 |

Step 0 and steps 1–3 drawn by different LoRAs:

| step 0 / steps 1–3 | sameness | cos to teacher |
|---|---|---|
| v1.1 / v1.1 | 0.396 | 0.677 |
| Hub v3 / Hub v3 | 0.296 | 0.635 |
| Hub v3 / v1.1 | 0.318 (0.289 / 0.333 / 0.331) | 0.742 (0.730 / 0.716 / 0.779) |
| anchor target / v1.1 (`A>S@1`) | 0.339 | 0.815 |
| anchor target / Hub v3 (`A>S@1`) | 0.277 | 0.752 |

- Given the teacher's step 0, v1.1's steps 1–3 finish at 0.932; Hub v3 reads 0.805, and the
  students trained on merged14 / merged15 0.745–0.841.
- v1.1's step-0 anchor MSE is 0.138, the level of the cold students.
- From the v1.1 delta to Hub v3 (10k iterations): anchor MSE 0.138 → 0.062, `T>S@1` cos
  0.932 → 0.805.
- Hub v3's step 0 with v1.1's steps 1–3 draws seeds at sameness 0.318 (teacher 0.315) and reads cos
  0.742, the closest 4-step result to the teacher. The anchor target as step 0 reads 0.815.

Renders, Claude's read of the guitarist grid of the first probe only: with the teacher's step 0 the
v1.1 column is hard to tell from the teacher's on all four seeds. Alone it keeps the jump and the
view from below with a figure that fills the frame; a monitor label on seed 2 and a corner mark on
seed 4 carry garbled text. The café and group grids and the cross-probe grids were not read.

### Interpretation

- v1.1's steps 1–3 keep the teacher's layout better than any student of this line.
- v1.1's step 0 is unfitted.
- Hub v3's 10k iterations fitted step 0 and lost steps 1–3.
- The two parts add up with no training: Hub v3's step 0 with v1.1's steps 1–3 draws seeds as
  varied as the teacher's.
- So merged15 is close enough to base for the delta to carry over, and the target for a student is
  v1.1's steps 1–3 with a fitted step 0 (`../plan.md`).

## Hub v3's training log

### Facts

`output/logs/turbo/20260828-024030` (base teacher, student and fake seeded from the v1.1 delta),
against `1355` (`output/logs/turbo/20261116-135557`) and the Hub v3 warm start with `dm_first_step`
off, `turbo_1113_1309`:

| | Hub v3's run | `1355` | `1309` |
|---|---|---|---|
| `grad_signal_rms`, first 300 it | 0.19–0.22 | 0.27–0.31 | 0.27 → 0.23 |
| `dm_rel_gap` | 0.16–0.19, flat to 6k | 0.24–0.26, 0.20 at 1k | 0.23 → 0.20 |
| `div_loss` (train) | 0.11 to 3,000 it, 0.078 at 6k | 0.195 → 0.107 | 0.07–0.09 |

- Hub v3's DM signal is flat from the first iteration, about 0.7 of what the merged-teacher runs
  see at peak LR.
- Hub v3's `div_loss` is flat for 3,000 iterations and falls as the cosine comes down.
- The LR was a cosine 5e-5 → 5e-6 over the configured 10k with a 200-step warmup, and the log
  ends at step 6,000. Whether the run stopped there is not known.

### Interpretation

- The DM term had little to move in v3's run.
- The anchor did its work late.

## Runs from the v1.1 init

### Facts

Both are `turbo_1116_0958` (merged15, static grid, 250 iterations, cosine, seed 42) with the student
seeded from the delta on Kaiming downs. The loop truncates the r96 file to rank 64 (0.914 of the ΔW
energy); the runs start from that truncation, not from the probed r96 row above.

- `turbo_1116_1552`: the shipped recipe, the fake seeded from the delta as well. About 25 min.
- `turbo_1116_1558`: `dpdmd.anchor_only`, with `cdm.weight`, `gan.weight_gen` and the fake warmup
  at 0 as in S-b. Steps 1–3 take no loss. About 8 min. Its first submission died at the first log
  line (`gan_w` unbound under `anchor_only` with a discriminator built); the log line is guarded
  now.

Probes `bench/turbo/results/20261116-1622-handoff_m15_v11init/` and
`20261116-1633-handoff_m15_v11init_anchoronly/`, EMA 250, group / café / guitarist in brackets:

| | sameness `S` | `S` cos | `T>S@1` cos | `T>S@1` sameness | anchor MSE |
|---|---|---|---|---|---|
| teacher m15 | 0.315 | | | | |
| start: v1.1 untrained (r96) | 0.396 (0.402 / 0.500 / 0.286) | 0.677 (0.714 / 0.711 / 0.605) | 0.932 (0.914 / 0.946 / 0.936) | 0.301 | 0.138 (0.068 / 0.104 / 0.241) |
| `1552`, shipped recipe | 0.383 (0.403 / 0.456 / 0.288) | 0.721 (0.727 / 0.761 / 0.675) | 0.922 (0.904 / 0.939 / 0.921) | 0.304 | 0.107 (0.064 / 0.080 / 0.177) |
| `1558`, anchor alone | 0.350 (0.351 / 0.406 / 0.293) | 0.766 (0.774 / 0.775 / 0.748) | 0.946 (0.938 / 0.958 / 0.942) | 0.308 | 0.082 (0.060 / 0.070 / 0.115) |
| Hub v3 step 0 / v1.1 steps 1–3 | 0.318 | 0.742 | | | 0.062 |
| Hub v3 untrained | 0.296 | 0.635 | 0.805 | 0.260 | 0.062 |
| `0958` (cold, shipped recipe) | 0.442 | 0.608 | 0.841 | | 0.138 |

Reads fixed in the config headers before the runs:

- `1552`, the init helps: holds. It is ahead of `0958` on all four numbers and in the renders.
- `1552`, training helps: three of four. Sameness, `S` cos and the anchor MSE are ahead of the
  start; `T>S@1` cos is 0.010 below it (0.922 against 0.932).
- **`1558`: passes.** The anchor MSE falls 0.138 → 0.082 with `T>S@1` cos at 0.946 and `T>S@1`
  sameness at the start's level (0.308 against 0.301). Every per-prompt number is ahead of the
  start except the guitarist's sameness (0.286 → 0.293).
- The anchor alone is ahead of the joint recipe on all four numbers from the same init at the
  same length, and its `S` cos (0.766) is above the untrained cross cell (0.742). Its sameness
  (0.350) and anchor MSE (0.082) are still short of Hub v3's (0.296, 0.062).

Training curves (`output/logs/turbo/20261116-155442`, `20261116-162245`):

- `1552`: `grad_signal_rms` 0.20–0.23 and `dm_rel_gap` 0.17–0.19 from the first iterations, the
  levels of v3's own run on base; the cold merged15 runs start at 0.27–0.31 and 0.24–0.26.
  `div_loss` 0.125 → 0.09.
- `1558`'s `div_loss` is still falling at the end: 0.093 / 0.078 / 0.067 over iterations
  50–100 / 100–200 / 200–250.

Renders. Judge renders (`inference.py`, merged15 int8, 4 steps, CFG 1, `judge_hard.txt` seeds 1–4,
768 × 1344): `output/tests/turbo4/{v11_m15,turbo_1116_1552,turbo_1116_1558}`. Grids
`output/tests/turbo4/cmp/v11init_{ba5,cafe,bocchi}.jpg`, rows teacher / Hub v3 untrained / v1.1
untrained / `1552` / `1558`. Claude's read, on the downscaled grids and three full-size tiles
(group seed 1, café seed 3, guitarist seed 2); the user's read is pending.

- v1.1 untrained is hazy and low in contrast, with large figures that fill the frame.
- `1552` against `0958`: the guitarist's distance and stage light vary by seed where `0958`
  draws one large figure on a plain stage; the café aprons are the teacher's light blue (`0958`:
  grey and brown) and `0958`'s speech bubbles are gone; the group has the sky and depth behind it.
- `1552`: crisp lines and saturated colour. The café sign is garbled on all four seeds ("CAFFE",
  "CAFF MILLENNIUM", "CAAE MILLENIUM"), and seed 3 fills its foreground with cups and boxes.
- `1558`: the layout of each seed is the nearest to the teacher's of the 4-step rows (the group's
  arrangement on seeds 2 and 4, the small figure of guitarist seed 1). The finish is softer than
  `1552`'s and objects break: the guitar body in the foreground of guitarist seed 2 is a shapeless
  dark mass, and the lower half of group seed 1 merges its figures. The café has less clutter than
  `1552`; its sign is unreadable on three seeds.
- With the teacher's step 0, `1558`'s steps 1–3 finish cleanly (the `T>S@1` column of its probe
  grid, guitarist). One grid.
- Hub v3 untrained keeps the cleanest finish and the only legible café sign of the 4-step rows.
- In the probe grids (CPU noise) guitarist seed 1 is broken in both trained students (a deformed
  hand at the top of the frame); v1.1 untrained draws a whole figure there.

### Interpretation

- `1552`'s 0.010 shortfall on `T>S@1` cos is not separated from the r64 truncation: the start was
  probed at r96 and the run began from the truncation.
- On `1558` the regression does not take the layout of steps 1–3.
- `1552` runs at Hub v3's equilibrium from the first iterations.
- `1558`'s soft finish and broken objects come with its own step 0, not from damaged steps 1–3.

## The anchor alone at 1k (`turbo_1116_1653`)

### Facts

`turbo_1116_1558` at 1000 iterations, the cosine over the whole run, saves every 250. 28.5 min
(job `20261116-165333-1112ba`). Probe `bench/turbo/results/20261116-1724-handoff_m15_anchor1k/`,
judge renders `output/tests/turbo4/turbo_1116_1653/`, grids
`output/tests/turbo4/cmp/anchor1k_{ba5,cafe,bocchi}.jpg` (rows teacher / Hub v3 untrained / `1552` /
`1558` / `1653`). EMA 1k against `1558` EMA 250, group / café / guitarist in brackets:

| | sameness `S` | `S` cos | `T>S@1` cos | `T>S@1` sameness | anchor MSE |
|---|---|---|---|---|---|
| teacher m15 | 0.315 (0.324 / 0.354 / 0.267) | | | | |
| `1558` EMA 250 | 0.350 (0.351 / 0.406 / 0.293) | 0.766 (0.774 / 0.775 / 0.748) | 0.946 | 0.308 | 0.082 (0.060 / 0.070 / 0.115) |
| `1653` EMA 1k | 0.365 (0.356 / 0.398 / 0.341) | 0.775 (0.771 / 0.789 / 0.765) | 0.945 | 0.312 | 0.068 (0.057 / 0.065 / 0.083) |
| anchor target as step 0 (`A>S@1`, `1653`'s steps 1–3) | 0.346 (0.309 / 0.375 / 0.355) | 0.818 (0.876 / 0.856 / 0.723) | | | 0 |
| Hub v3 untrained | 0.296 | 0.635 | 0.805 | 0.260 | 0.062 (0.060 / 0.078 / 0.049) |

Training `div_loss` by 100 iterations: 0.088 / 0.087 / 0.094 / 0.073 / 0.077 / 0.069 / 0.061 / 0.063 /
0.059 / 0.058. Flat for the first 300, falling as the cosine comes down, flat again over the last 400.

Read, against the conditions fixed in the config header:

- Length pays: two of three. The anchor MSE falls on every prompt, most on the guitarist
  (0.115 → 0.083), and `S` cos rises on the mean (the group is 0.003 lower). Sameness does not
  follow: 0.350 → 0.365, the guitarist 0.293 → 0.341.
- Steps 1–3 hold at length. `T>S@1` cos 0.945 against the start's 0.932; no leak.
- The finish does not come with the fit. Claude's read, on full-size crops of the places the
  header names (the user's read is pending):
  - group, lower half: seeds 1 and 2 still merge their figures (seed 1 a pastry fused into a face);
    seed 4 is whole in both runs;
  - guitarist: seed 2 is fixed (a smaller jumping figure with a whole guitar where `1558` drew a
    shapeless body); seed 1 is newly broken (the neck is cut from the body and a headstock floats
    beside it); seed 4's guitar body is garbled in both;
  - café: the aprons turn green on seeds 1 and 2, where `1558` drew the teacher's light blue; the
    sign is legible on seed 2 in both.
- Near its target on the cosine, far from it on the fit. `S` cos is 0.775 against 0.818 with the
  anchor target itself as step 0, and the sameness (0.365) sits at the target's (0.346), not at the
  teacher's (0.315). The anchor MSE 0.138 → 0.068 is about 0.3 of the residual's amplitude
  (`step0_fit.md`'s mix scale, MSE ∝ (1 − α)²), so a deeper fit than this has not been read.

## The anchor target: the 12-step rollout is the ceiling, not the chord

### Facts

Forward-only, on `1653` EMA 1k: `probe_handoff.py --anchor_k / --anchor_steps`, the `A>S@1` arm (the
target itself as step 0, `1653` drawing steps 1–3). Runs
`bench/turbo/results/20261116-{1728-anchor_target_14of28,1731-anchor_target_7of28,1733-anchor_target_3of12}/`,
2.5 min each; grids `output/tests/turbo4/cmp/anchor_target_{ba5,bocchi}.jpg` (teacher / 6 of 12 /
14 of 28 / 7 of 28 / `1653`'s own step 0). The teacher's image is 28 steps at CFG 4.

| anchor (k of steps → t_k) | `A>S@1` cos | `A>S@1` sameness | `1653`'s MSE to it | teacher cond MSE to it |
|---|---|---|---|---|
| 6 of 12 → 0.75 (trained on) | 0.818 (0.876 / 0.856 / 0.723) | 0.346 (0.309 / 0.375 / 0.355) | 0.068 | 0.165 |
| 3 of 12 → 0.9 | 0.810 (0.854 / 0.848 / 0.727) | 0.430 (0.394 / 0.466 / 0.428) | 0.057 | 0.159 |
| 14 of 28 → 0.75 | 0.967 (0.964 / 0.973 / 0.963) | 0.249 (0.264 / 0.295 / 0.188) | 0.084 | 0.140 |
| 7 of 28 → 0.9 | 0.945 (0.937 / 0.958 / 0.941) | 0.313 (0.327 / 0.337 / 0.275) | 0.046 | 0.102 |
| teacher's own path to 0.9 (`T>S@1`) | 0.945 | 0.312 | | |
| teacher | | 0.315 (0.324 / 0.354 / 0.267) | | |

- Both 12-step targets end at 0.81–0.82, whichever boundary they stop at; both 28-step targets are
  at the teacher's level or above. `7 of 28` is the teacher's own path to σ 0.9 and reproduces
  `T>S@1` to the third decimal.
- The 12-step targets draw seeds at sameness 0.346 and 0.430 against the teacher's 0.315. The
  28-step chord to σ 0.75 reads 0.249.
- As a step 0 for these steps 1–3, the chord to σ 0.75 reads 0.967 and the teacher's own state at
  σ 0.9 0.945.
- Claude's read of the two grids: the `14 of 28` column is the teacher's image on all eight tiles,
  and clean. The `6 of 12` column on the guitarist is a smaller, darker figure with the crowd
  filling the lower half (seeds 3 and 4).
- `S-a` (`_archive/anima_turbo/reports/step0.md`) read the chord as costing 0.02–0.07 on the
  merged14 students. Their steps 1–3 kept 0.70–0.80 of the teacher's step 0.
- Cost: the anchor is 2 teacher forwards per Euler step, so `14 of 28` is 29 forwards an iteration
  against 13, `7 of 28` is 15.

### Interpretation

- The 12-step rollout is what costs, not the chord.
- The 12-step targets draw seeds more alike, which is where `1653`'s sameness went with its fit.
  The 28-step chord to σ 0.75 draws them less alike than the teacher.
- The chord to σ 0.75 beats the teacher's own state at σ 0.9 as a step 0 for these steps 1–3.
- The guitarist's smaller, darker figure in the `6 of 12` column is the 0.723.
- With steps 1–3 that keep 0.945, the 12-step target is a larger loss than the chord.

## The anchor alone with `grad_accum 4` (`turbo_1116_1730`)

### Facts

`1558` with `optim.grad_accum = 4` at 1000 iterations: 250 updates on `1558`'s cosine, each the mean
of four per-iteration clipped gradients (four prompts, four ε). 28.3 min (job
`20261116-173048-d94ab2`). Probe `bench/turbo/results/20261116-1804-handoff_m15_anchor_accum4/`,
renders `output/tests/turbo4/turbo_1116_1730/`, grids
`output/tests/turbo4/cmp/anchor_accum4_{ba5,cafe,bocchi}.jpg` (teacher / `1558` / `1653` / `1730`).

| | updates × pairs | sameness `S` | `S` cos | `T>S@1` cos | anchor MSE |
|---|---|---|---|---|---|
| `1558` EMA | 250 × 1 | 0.350 (0.351 / 0.406 / 0.293) | 0.766 (0.774 / 0.775 / 0.748) | 0.946 | 0.082 (0.060 / 0.070 / 0.115) |
| `1653` EMA | 1000 × 1 | 0.365 (0.356 / 0.398 / 0.341) | 0.775 (0.771 / 0.789 / 0.765) | 0.945 | 0.068 (0.057 / 0.065 / 0.083) |
| `1730` EMA | 250 × 4 | 0.373 (0.360 / 0.413 / 0.346) | 0.770 (0.776 / 0.782 / 0.753) | 0.944 | 0.068 (0.055 / 0.065 / 0.084) |

Training `div_loss` by 200 iterations: 0.081 / 0.072 / 0.066 / 0.059 / 0.057.

Read, against the conditions fixed in the config header:

- Against `1558`, equal updates: the draw noise limits the fit. Anchor MSE 0.082 → 0.068 and `S`
  cos up on every prompt, `T>S@1` cos 0.944.
- Against `1653`, equal compute: a tie on the fit, slightly behind on the rest. The anchor MSE is
  the same to the third decimal (0.0679 against 0.0682); `S` cos is 0.005 lower and the sameness
  0.008 higher.
- Sameness 0.350 → 0.365 (`1653`) and 0.373 (`1730`), past the 12-step target's own 0.346.
- Renders (Claude's read, full-size crops; the user's read is pending): the group's lower half
  still merges on seeds 1 and 2 and the guitar body is garbled on guitarist seed 4, as in `1558`
  and `1653`. Guitarist seed 1 keeps a whole guitar where `1653` cut the neck. The café aprons keep
  the teacher's blue on seeds 1–3 where `1653` drew green on two.

### Interpretation

- Four pairs in one update and four updates of one pair land in the same place, so the fit follows
  the number of pairs seen.
- Sameness rises with the fit in both runs.

## What it means

- Seeding from the turbo delta is what the merged-teacher runs missed. The delta carries over to
  merged15, and from it the DM term starts at equilibrium instead of pulling a cold student across.
- The layout gain comes from the anchor, and the DM / CDM / GAN terms take some of it back while
  giving the crisp finish.
- Length alone does not bring the finish (`1653`), and the target the anchor regresses on is the
  12-step teacher, a different and less varied image than the 28-step one the student is judged
  against. The 28-step chord is the target to train on (`../anchor_only_recipe.md`).
- The fit follows the number of pairs seen. `grad_accum 4` reaches at 250 updates what 1000
  single-pair updates reach, and neither brings the finish (`turbo_1116_1730`).
