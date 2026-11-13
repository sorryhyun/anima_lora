# Warm start from Hub v3: training on the m14 teacher makes v3 worse (2026-11-13)

`turbo_1113_1221` is the control `turbo_1113_1105` (the v1 recipe at 250 iterations: cosine
5e-5 → 5e-6, EMA 0.2 of the summed LR, `dm_first_step` on, seed 42) with the student seeded from
Hub v3 (`network.student_init_weights = models/loras/turbo_hub/anima_turbo_4step_v3.safetensors`;
base teacher, 10k it). The fake starts as in the control (`weight_svd`, 200 warmup steps). Config
`configs/turbo_1113_1221.toml`, job `20261113-122145-1a4adf`.

**Verdict: it passes the gate written in `plan.md`, and the run is still a regression.** The gate
compared the run with the control only. Hub v3 placed on the merged14 DiT with no training at all
beats it on every number and, in the user's read, on the renders. The largest gap is in diversity:
untrained v3 draws its seeds as differently as the teacher does (sameness 0.326 against the
teacher's 0.328), and 250 iterations raise it to 0.465, close to the control's 0.496. Training the student on the m14
teacher with the v1 recipe moves v3 toward the control rather than improving it, so the planned
1000-iteration run from this start is dropped.

## Evaluation

Same evaluation as the control:

- Judge renders: `inference.py` on `anima_merged14` (int8), 4 steps, CFG 1, Euler, shift 3,
  `bench/turbo/judge_hard.txt` (group / café / guitarist), seeds 1–4, 768 × 1344. These use CUDA
  noise.
- `bench/turbo/probe_handoff.py --ts_handoffs 1 --st_handoffs ,` on the same prompts and seeds, with
  CPU noise, so its images do not pair with the judge renders.

Rows of the grids `output/tests/turbo4/cmp/v3warm4_{ba5,cafe,bocchi}.jpg`: teacher (28 steps,
CFG 4), control EMA 250, `turbo_1113_1221` EMA 250, Hub v3 on merged14 untrained. Render dirs:
`output/tests/turbo4/{teacher_m14,turbo_1113_1105,turbo_1113_1221,hub_v3_m14}`.

Probe runs: `bench/turbo/results/20261113-1250-handoff_v3warm/` (this run),
`20261113-1216-cross_mfd0_ctrl123/` (control), `20261113-1210-handoff_v3_on_m14/` and
`20261113-1257-handoff_v3_on_m14_same/` (v3 untrained; the second rerun adds the sameness metric, which
the first predates).

## Numbers

Probe metrics, mean over the three prompts, with per-prompt values in group / café / guitarist
order:

| | Hub v3 on m14, untrained | `1221` EMA 250 | control EMA 250 | teacher |
|---|---|---|---|---|
| held-out anchor MSE | **0.063** (0.059 / 0.074 / 0.055) | 0.070 (0.062 / 0.074 / 0.075) | 0.149 | cond 0.171 |
| low-passed cos to teacher, `S` | **0.639** (0.660 / 0.657 / 0.599) | 0.593 (0.585 / 0.611 / 0.584) | 0.547 | – |
| same, `T>S@1` | **0.826** | 0.782 | 0.779 | – |
| same, `A>S@1` | **0.743** | 0.707 | 0.728 | – |
| sameness, `T>S@1` | **0.280** | 0.309 | 0.405 | – |
| sameness across seeds, `S` | **0.326** (0.342 / 0.367 / 0.270) | 0.465 (0.459 / 0.618 / 0.317) | 0.496 (0.502 / 0.441 / 0.545) | 0.328 |

Validation scalars from the training log (`output/logs/turbo/20261113-122201`), at 125 / 250:

| | `1221` | control |
|---|---|---|
| `val/div_xpred_ac_sim` | 0.908 / 0.870 | 0.948 / 0.909 |
| `val/div_ac_sim` | 0.583 / 0.483 | 0.661 / 0.547 |
| `val/fm_mse` | 0.265 / 0.247 | 0.299 / 0.242 |

## Gate (fixed in `plan.md` before the run)

| Condition | Result |
|---|---|
| Guitarist keeps the view from below on the seeds where the control draws the frontal figure | holds: jump, raised arms, lower camera, crowd silhouettes in front. No seed matches the teacher's strong foreshortening |
| Anchor MSE at EMA 250 below the control's 0.149 | holds: 0.070 |
| Sameness below the control's 0.496 | holds on the mean (0.465), but not per prompt: the guitarist falls to 0.317 and the café rises to 0.618 |
| Five-girl prompt is one scene | holds: one scene on all four seeds, with all five characters |

All four conditions hold, but the gate had no condition against the starting point.

## Renders

**Claude's read on downscaled grids, then the user's read.**

- **Guitarist:** the seeds' poses come almost unchanged from v3 (seed 1 one arm up, seed 4 both
  arms up). The 250 iterations kept v3's composition here.
- **Group:** v3 varies the arrangement between seeds and looks cluttered, close to a collage on
  some seeds. `1221` is one clean scene that keeps the prompt's positions, with a similar
  arrangement on every seed.
- **Café:** v3 varies the apron colour and the framing between seeds. `1221` draws blue aprons on
  every seed, which matches the café sameness rising to 0.618.
- **Logo and small text:** on the café renders a "ブルーアーカイブ" logo appears on the sign and a
  small copyright-like line in the bottom-left corner. Both are in Hub v3 untrained too (logo on
  seeds 1, 3 and 4, corner text on seeds 1, 2 and 4), so they come from v3 and not from this run.
- Claude first read `1221` as "v3, cleaned up". The user read it as a regression from v3, and the
  probe numbers above agree with the user.

## Which steps carry the regression

Two cross-handoff runs (`probe_handoff.py --step0_lora`) split the run's change into step 0 and
steps 1–3: `bench/turbo/results/20261113-1304-cross_v3s0_warm123/` (v3 draws step 0, `1221`
steps 1–3) and `20261113-1307-cross_warms0_v3123/` (the reverse).

Sameness across seeds (teacher 0.328), group / café / guitarist in brackets for the cross cells:

| Step 0 drawn by | steps 1–3: v3 | steps 1–3: `1221` |
|---|---|---|
| teacher (`T>S@1`) | 0.280 | 0.309 |
| v3 | 0.326 | 0.381 (0.398 / 0.456 / 0.290) |
| `1221` | 0.429 (0.437 / 0.563 / 0.288) | 0.465 |

Low-passed cosine to the teacher's image:

| Step 0 drawn by | steps 1–3: v3 | steps 1–3: `1221` |
|---|---|---|
| teacher (`T>S@1`) | 0.826 | 0.782 |
| v3 | 0.639 | 0.613 |
| `1221` | 0.637 | 0.593 |

- **Step 0 carries most of the diversity loss.** Swapping only step 0 to `1221` adds 0.08–0.10
  of sameness, while swapping only steps 1–3 adds 0.03–0.06. The café takes most of it
  (0.367 → 0.563 from step 0 alone).
- **Steps 1–3 carry the loss of teacher likeness.** Swapping step 0 moves the cosine by 0.00–0.02.
  Swapping steps 1–3 moves it by 0.03–0.04, including when the teacher draws step 0.
- So both parts of the recipe regress v3. Step 0 makes the seeds alike while keeping the anchor
  fit (MSE 0.063 → 0.070), and steps 1–3 move away from the teacher and make the seeds slightly
  more alike as well. In this run step 0 is trained by the anchor (`div_weight 0.1`) and by the DM
  term (`dm_first_step` on). v3 was trained without `dm_first_step`, and S-c.1 (the same switch on
  the `weight_svd` start) was read by the user as "improved, less diverse". Neither source of the
  step-0 change has been isolated yet.

## `dm_first_step` off from the same start: not the cause (`turbo_1113_1309`)

`turbo_1113_1221` with `dpdmd.dm_first_step = false`: step 0 is trained by the anchor alone, as v3
was. Config `configs/turbo_1113_1309.toml`, job `20261113-130933-4d08cb`. Probes:
`bench/turbo/results/20261113-1339-handoff_v3warm_nodm0/` and
`20261113-1341-cross_nodm0s0_v3123/` (`1309` draws step 0, v3 draws steps 1–3). Grids
`output/tests/turbo4/cmp/nodm0_{ba5,cafe,bocchi}.jpg`, with rows teacher / Hub v3 on m14 / `1221` /
`1309`.

Read fixed before the run (config header): step-0 sameness back near v3's 0.326 in the cross cell
means `dm_first_step` caused the diversity loss, and `T>S@1` cosine still below 0.826 means the
steps 1–3 DM term is next.

| | v3 untrained | `1221` | `1309` |
|---|---|---|---|
| sameness, own step 0 + v3 steps 1–3 | 0.326 | 0.429 | **0.405** (0.414 / 0.493 / 0.309) |
| sameness, `S` | 0.326 | 0.465 | 0.476 (0.421 / 0.567 / 0.442) |
| sameness, `T>S@1` | 0.280 | 0.309 | 0.333 |
| cos to teacher, `S` | 0.639 | 0.593 | 0.597 |
| cos to teacher, `T>S@1` | 0.826 | 0.782 | **0.773** |
| cos, own step 0 + v3 steps 1–3 | 0.639 | 0.637 | 0.643 |
| held-out anchor MSE | 0.063 | 0.070 | **0.063** (0.064 / 0.072 / 0.051) |

- **`dm_first_step` is not the main cause.** Turning it off takes back 0.024 of the 0.103 that
  step 0 added to the sameness. Step 0 trained by the anchor alone still makes the seeds alike.
- **The anchor MSE cannot see this.** `1309` keeps v3's anchor fit exactly (0.063) while its step 0
  raises the sameness by 0.08. The fit to the per-noise anchor target is unchanged, yet the
  seed-to-seed differences shrink.
- **Steps 1–3 regress as in `1221`** (`T>S@1` cos 0.773, sameness 0.333). The second condition
  holds, so the steps 1–3 DM term is next.
- With `dm_first_step` off, step 0 receives the anchor gradient and the change that DM updates at
  σ < 1 make to the shared LoRA at σ 1. The GAN also touches it. Which of these makes the seeds
  alike has not been measured.

Renders (Claude's read on downscaled grids; the user's read is pending):

- **Guitarist:** a wider shot with a small figure and the crowd in front, on all four seeds. Poses
  vary (guitar raised, both arms up), but the scale and framing are one, and v3's low camera with
  a large figure is gone. Sameness 0.442 against v3's 0.270 and `1221`'s 0.317.
- **Café:** closer to v3 than `1221`. Seed 1 has a dark apron, and the signs and menus vary. v3's
  logo is still on seed 3.
- **Group:** the arrangement varies more than in `1221`, closer to v3's busy layouts.
- Neither `1221` nor `1309` is better than v3 untrained on any prompt.

## What it means

- **This is not a loss from baking the soup.** merged14 is the teacher itself, and v3 placed on it
  sits closer to that teacher than any student distilled from it (anchor MSE 0.063 against
  0.15 for the m14 students at 250 to 1k iterations; `reports/step0_fit.md`).
- **The v1 recipe on the m14 teacher has its own attractor**, with or without `dm_first_step`. v3 starts at the teacher's
  sameness, so the seed collapse seen in every m14 student is not something 4-step sampling
  forces; the training puts it there. From the `weight_svd` start or from
  v3, 250 iterations move the student toward the same place: less teacher-like low-passed
  layout, café seeds that converge on one image, and a blander look. This matches the open item in
  `plan.md`: the plain DM term on steps 1–3 pulls the seeds together (control `T>S@1` sameness
  0.405 against the teacher's 0.328). This is the likely mechanism; no probe has isolated it.
- **The best 4-step result on m14 right now needs no training:** merged14 + Hub v3 at 1.0. The
  shipping target is one LoRA on base, so the candidate is the exact merge of the soup ΔW14 and v3
  (`scripts/toolkits/merge_loras.py anima_merged14.safetensors anima_turbo_4step_v3.safetensors
  --normalize off --adaln sum`), with v3's logo and corner text still in it.

## Dropped

- The 1000-iteration run from the v3 start on the v1 recipe (`plan.md` "Pass" branch). A longer
  run would most likely regress further.
