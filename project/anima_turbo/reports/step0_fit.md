# Step 0: how good the fit has to be, and Hub v3 as a fitted step 0 (2026-11-13)

`step0.md` left one question open: how low the anchor MSE has to go before the layout comes with
it. Three forward-only probes on `bench/turbo/probe_handoff.py` answer it. Same prompts
(`bench/turbo/judge_hard.txt`: group / café / guitarist), seeds 1–4, 768 × 1344, CPU-drawn noise.

**The render reads below are Claude's, on downscaled grids. The user's read is pending.**

## Step-0 mix probe

`--step0_mix` (new): step 0 is the student's own velocity `v_s` plus part of its residual to the
anchor target, `r = v_t − v_s`; the student draws steps 1–3. `M<α>`: `v_s + α·r`. `L<k>`: only the
k×-pooled band of `r` (area downsample, bilinear back up). `H8`: `r` without its 8× band.
Students: v1 1k EMA (`1112_1512_1k_ema`) and the new control's EMA 250 (`turbo_1113_1105`).
Run: `bench/turbo/results/20261113-1203-step0_mix/` (job `20261113-111726-67e57f`, 5.2 min).

Low-passed cosine to the teacher's image, mean of 12, and the anchor MSE left on step 0:

| Step 0 | v1 1k EMA | control EMA 250 | MSE left (v1 / control) |
|---|---|---|---|
| `T>S@1` | 0.756 | 0.779 | – |
| `A>S@1` (α 1) | 0.705 | 0.728 | 0 |
| `M0.75` | 0.686 | 0.727 | 0.009 / 0.009 |
| `M0.5` | 0.653 | 0.700 | 0.036 / 0.037 |
| `M0.25` | 0.611 | 0.631 | 0.082 / 0.084 |
| `L4` | 0.701 | 0.723 | 0.023 / 0.011 |
| `L8` | 0.684 | 0.720 | 0.039 / 0.023 |
| `L16` | 0.657 | 0.710 | 0.057 / 0.040 |
| `H8` | 0.597 | 0.592 | 0.094 / 0.114 |
| `S` (α 0) | 0.562 | 0.547 | 0.146 / 0.149 |

Held-out anchor MSE and the share of that error in the 8× band: teacher cond 0.171 (0.72), v1 1k
EMA 0.146 (0.59), control EMA 250 0.149 (0.68).

- **The layout is in the low band of the residual.** `L8` gives 85 % / 96 % of the way from `S` to
  `A>S@1`, `H8` 24 % / 25 %. `L16`, a 10 × 6 field per channel, still gives 66 % / 90 %. On the
  guitarist grids the `L` columns draw the smaller figure and the stage edge of `A>S@1`; `H8` draws
  the frame-filling figure of `S`.
- **A pooled anchor loss has little to redirect.** The low band is already 59–68 % of the students'
  error, and it is the part training lowered (0.123 → 0.087 / 0.102; the rest stayed at
  0.047–0.059). `idea.md` 4 is closed by this.
- **No threshold: the gain is close to proportional to α.** Half of the layout gain needs an MSE of
  about 0.04–0.08, that is 0.3–0.5 of where the 1k students sit.

## Hub v3 on its own teacher

`anima_turbo_4step_v3` (base teacher, 10k iterations at a constant 5e-5, warm start from the v1.1
delta, `dm_first_step` off) with `--dit anima-base-v1.0`.
Run: `bench/turbo/results/20261113-1206-handoff_hub_v3/` (job `20261113-112028-e7b1a6`, 2.5 min).

| | Hub v3 | m14 students (v1 1k / control 250) |
|---|---|---|
| held-out anchor MSE | **0.071** (0.045 / 0.097 / 0.070) | 0.146 / 0.149 |
| teacher cond velocity, same target | 0.234 (0.084 / 0.162 / 0.456) | 0.171 |
| ratio | 0.30 | 0.85 / 0.87 |
| `S` / `A>S@1` / `T>S@1` | 0.503 / 0.642 / 0.722 | – |

- **v3's step 0 is fitted**, to the level the mix probe names (0.3 of the reference), and its `S`
  column draws the jump from below on all four guitarist seeds. The judge renders agree
  (`output/tests/turbo4/hub_v3/`): view from below, a small full-body figure, a different pose per
  seed.
- So the step-0 regression does fit with enough training, on the same anchor (`k_anchor 6`,
  `div_weight 0.1`) and the same LR. `step0.md`'s "the fit follows the LR, not the length" holds
  from 250 to 1000 iterations and not at 10k.
- v3's `S` cosine is low (0.503): it draws the teacher's kind of layout, not the teacher's image
  of that noise. The cosine is not a quality score.

## Hub v3 on the merged14 teacher, untrained

The same LoRA on `anima_merged14`, against the merged14 anchor target.
Run: `bench/turbo/results/20261113-1210-handoff_v3_on_m14/` (job `20261113-120727-522366`).

| | Hub v3 on m14 | v1 1k EMA | control EMA 250 |
|---|---|---|---|
| held-out anchor MSE | **0.063** (0.059 / 0.074 / 0.055) | 0.146 | 0.149 |
| `S` | 0.639 | 0.562 | 0.547 |
| `A>S@1` / `T>S@1` | 0.743 / 0.826 | 0.705 / 0.756 | 0.728 / 0.779 |

- **v3's step 0 carries over to merged14.** With no training on that teacher it fits the merged14
  anchor 2.3 times better than any student trained on it, and the `S` column keeps the view from
  below and the jump on all four guitarist seeds.
- A student of this line warm-started from v3 (`network.student_init_weights`; same rank 64 /
  alpha 180 / adaln 16) starts with the step 0 that 10k iterations bought. What it has to keep
  through training, and what v3 brings that is not wanted (the panel split on multi-character
  prompts that `v4_preview` fixed), is the arm in `../plan.md`.
