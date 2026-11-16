# The turbo v1.1 delta on merged15: the best steps 1–3, an unfitted step 0 (2026-11-16)

The official turbo v1.1 delta (`models/extracted/anima_turbo_v1.1_delta_r96_asvd_adaln.safetensors`)
is the file Hub v3's run seeded its student and fake from. After the 1k failure (`m15_1k.md`) the
user's question was whether seeding from it, as v3 did, is what the merged-teacher runs miss. This
report holds the probes of the delta itself and what Hub v3's training log shows; the runs from this
init (`turbo_1116_1552`, `turbo_1116_1558`) are added when they finish.

## Probes, no training

`bench/turbo/probe_handoff.py --dit models/diffusion_models/anima_merged15.safetensors --ts_handoffs 1
--st_handoffs ,` on `judge_hard.txt`, seeds 1–4: the delta alone
(`bench/turbo/results/20261116-1550-handoff_m15_v11_untrained/`) and with `--step0_lora` = Hub v3
(`20261116-1554-cross_v3s0_v11123_m15/`). The Hub v3 and `1355` rows are from
`20261116-1533-handoff_m15_1k/`, `0958` from `20261116-1054-handoff_m15_cdmcond/`.

| | sameness `S` | `S` cos | `T>S@1` cos | `T>S@1` sameness | anchor MSE |
|---|---|---|---|---|---|
| teacher m15 | 0.315 | | | | cond 0.165 |
| **v1.1 delta** | 0.396 (0.402 / 0.500 / 0.286) | **0.677** | **0.932** (0.914 / 0.946 / 0.936) | 0.301 | 0.138 (0.068 / 0.104 / 0.241) |
| Hub v3 | **0.296** | 0.635 | 0.805 | 0.260 | **0.062** |
| `0958` EMA 250 (cold) | 0.442 | 0.608 | 0.841 | | 0.138 |
| `1355` EMA 1k (cold) | 0.449 | 0.532 | 0.781 | 0.329 | 0.128 |

Step 0 and steps 1–3 drawn by different LoRAs:

| step 0 / steps 1–3 | sameness | cos to teacher |
|---|---|---|
| v1.1 / v1.1 | 0.396 | 0.677 |
| Hub v3 / Hub v3 | 0.296 | 0.635 |
| **Hub v3 / v1.1** | **0.318** (0.289 / 0.333 / 0.331) | **0.742** (0.730 / 0.716 / 0.779) |
| anchor target / v1.1 (`A>S@1`) | 0.339 | 0.815 |
| anchor target / Hub v3 (`A>S@1`) | 0.277 | 0.752 |

- **v1.1's steps 1–3 keep the teacher's layout better than any student of this line.** Given the
  teacher's step 0 they finish at 0.932; Hub v3 reads 0.805, and the students trained on merged14 /
  merged15 0.745–0.841.
- **v1.1's step 0 is unfitted:** anchor MSE 0.138, the level of the cold students.
- **Hub v3's 10k iterations fitted step 0 and lost steps 1–3:** anchor MSE 0.138 → 0.062, `T>S@1` cos
  0.932 → 0.805.
- **The two parts add up with no training.** Hub v3's step 0 with v1.1's steps 1–3 draws seeds as
  varied as the teacher's (0.318 against 0.315) and is the closest 4-step result to it (0.742).
  The anchor target as step 0 reads 0.815.
- So merged15 is close enough to base for the delta to carry over, and the target for a student is
  v1.1's steps 1–3 with a fitted step 0 (`../plan.md`).

Renders, Claude's read of the guitarist grid of the first probe only: with the teacher's step 0 the
v1.1 column is hard to tell from the teacher's on all four seeds. Alone it keeps the jump and the
view from below with a figure that fills the frame; a monitor label on seed 2 and a corner mark on
seed 4 carry garbled text. The café and group grids and the cross-probe grids were not read.

## Hub v3's training log

`output/logs/turbo/20260828-024030` (base teacher, student and fake seeded from the v1.1 delta),
against `1355` (`output/logs/turbo/20261116-135557`) and the Hub v3 warm start with `dm_first_step`
off, `turbo_1113_1309`:

| | Hub v3's run | `1355` | `1309` |
|---|---|---|---|
| `grad_signal_rms`, first 300 it | 0.19–0.22 | 0.27–0.31 | 0.27 → 0.23 |
| `dm_rel_gap` | 0.16–0.19, flat to 6k | 0.24–0.26, 0.20 at 1k | 0.23 → 0.20 |
| `div_loss` (train) | 0.11 to 3,000 it, 0.078 at 6k | 0.195 → 0.107 | 0.07–0.09 |

- **The DM term had little to move in v3's run:** its signal is flat from the first iteration, about
  0.7 of what the merged-teacher runs see at peak LR.
- **The anchor did its work late:** the loss is flat for 3,000 iterations and falls as the cosine
  comes down.
- **The LR was a cosine** 5e-5 → 5e-6 over the configured 10k with a 200-step warmup, and the log
  ends at step 6,000. Whether the run stopped there is not known.

## Runs from the v1.1 init

Both are `turbo_1116_0958` (merged15, static grid, 250 iterations, cosine, seed 42) with the student
seeded from the delta on Kaiming downs. The loop truncates the r96 file to rank 64 (0.914 of the ΔW
energy), so their start is slightly off the probed row above.

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
| **`1558`, anchor alone** | **0.350** (0.351 / 0.406 / 0.293) | **0.766** (0.774 / 0.775 / 0.748) | **0.946** (0.938 / 0.958 / 0.942) | 0.308 | **0.082** (0.060 / 0.070 / 0.115) |
| Hub v3 step 0 / v1.1 steps 1–3 | 0.318 | 0.742 | | | 0.062 |
| Hub v3 untrained | 0.296 | 0.635 | 0.805 | 0.260 | 0.062 |
| `0958` (cold, shipped recipe) | 0.442 | 0.608 | 0.841 | | 0.138 |

Reads fixed in the config headers before the runs:

- **`1552`, the init helps: holds.** It is ahead of `0958` on all four numbers and in the renders.
- **`1552`, training helps: three of four.** Sameness, `S` cos and the anchor MSE are ahead of the
  start; `T>S@1` cos is 0.010 below it (0.922 against 0.932). The start was probed at r96 and the
  run began from the r64 truncation, so the 0.010 is not separated from the truncation.
- **`1558`: passes.** The anchor MSE falls 0.138 → 0.082 with `T>S@1` cos at 0.946 and `T>S@1`
  sameness at the start's level (0.308 against 0.301). The regression does not take the layout of
  steps 1–3. Every per-prompt number is ahead of the start except the guitarist's sameness
  (0.286 → 0.293).
- **The anchor alone is ahead of the joint recipe** on all four numbers from the same init at the
  same length, and its `S` cos (0.766) is above the untrained cross cell (0.742). Its sameness
  (0.350) and anchor MSE (0.082) are still short of Hub v3's (0.296, 0.062).

Training curves (`output/logs/turbo/20261116-155442`, `20261116-162245`):

- **`1552` runs at Hub v3's equilibrium from the first iterations:** `grad_signal_rms` 0.20–0.23 and
  `dm_rel_gap` 0.17–0.19, the levels of v3's own run on base; the cold merged15 runs start at
  0.27–0.31 and 0.24–0.26. `div_loss` 0.125 → 0.09.
- **`1558`'s `div_loss` is still falling at the end:** 0.093 / 0.078 / 0.067 over iterations
  50–100 / 100–200 / 200–250.

### Renders

Judge renders (`inference.py`, merged15 int8, 4 steps, CFG 1, `judge_hard.txt` seeds 1–4,
768 × 1344): `output/tests/turbo4/{v11_m15,turbo_1116_1552,turbo_1116_1558}`. Grids
`output/tests/turbo4/cmp/v11init_{ba5,cafe,bocchi}.jpg`, rows teacher / Hub v3 untrained / v1.1
untrained / `1552` / `1558`. **Claude's read**, on the downscaled grids and three full-size tiles
(group seed 1, café seed 3, guitarist seed 2); the user's read is pending.

- **v1.1 untrained** is hazy and low in contrast, with large figures that fill the frame.
- **`1552` against `0958`:** the guitarist's distance and stage light vary by seed where `0958`
  draws one large figure on a plain stage; the café aprons are the teacher's light blue (`0958`:
  grey and brown) and `0958`'s speech bubbles are gone; the group has the sky and depth behind it.
- **`1552`:** crisp lines and saturated colour. The café sign is garbled on all four seeds ("CAFFE",
  "CAFF MILLENNIUM", "CAAE MILLENIUM"), and seed 3 fills its foreground with cups and boxes.
- **`1558`:** the layout of each seed is the nearest to the teacher's of the 4-step rows (the group's
  arrangement on seeds 2 and 4, the small figure of guitarist seed 1). The finish is softer than
  `1552`'s and objects break: the guitar body in the foreground of guitarist seed 2 is a shapeless
  dark mass, and the lower half of group seed 1 merges its figures. The café has less clutter than
  `1552`; its sign is unreadable on three seeds.
- **With the teacher's step 0, `1558`'s steps 1–3 finish cleanly** (the `T>S@1` column of its probe
  grid, guitarist). So the soft finish and the broken objects come with its own step 0, not from
  damaged steps 1–3. One grid.
- **Hub v3 untrained keeps the cleanest finish** and the only legible café sign of the 4-step rows.
- In the probe grids (CPU noise) guitarist seed 1 is broken in both trained students (a deformed
  hand at the top of the frame); v1.1 untrained draws a whole figure there.

## What it means

- **Seeding from the turbo delta is what the merged-teacher runs missed.** The delta carries over to
  merged15, and from it the DM term starts at equilibrium instead of pulling a cold student across.
- **The layout gain comes from the anchor, and the DM / CDM / GAN terms take some of it back** while
  giving the crisp finish. Which of them the finish needs, and whether a better-fitted step 0
  finishes cleanly without them, is open (`../plan.md`).
