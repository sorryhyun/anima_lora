# Plan: the 4-step line after `v4_preview`

Rewritten 2026-11-16. This file holds what is open; finished runs and probes are in `reports/` and
the run table (`configs/README.md`). The closed 2-step plan is
`91d476e0:scripts/distill_turbo/plan.md`.

## Where the line stands

- `v4_preview` (`1112_1512`, 1k EMA on the `1112_v1.md` recipe) is shipped. The recipe in
  `configs/methods/turbo.toml` has since taken the static grid and the cond-only L_CDM
  (`reports/grid_cdm.md`).
- Every student trained with that recipe on merged14 / merged15 draws one layout per prompt: from a
  cold start, from Hub v3, and at 1000 iterations (`reports/warm_v3.md`, `reports/m15_1k.md`).
- The best one-LoRA 4-step result is Hub v3 on merged15 with no training (sameness 0.296 against
  the teacher's 0.315).
- The official turbo v1.1 delta, the file Hub v3's run started from, has the best steps 1–3 of the
  line on merged15 with no training, and Hub v3's step 0 in front of them is the closest 4-step
  result to the teacher (`reports/v11_init.md`).

## Target: keep v1.1's steps 1–3, fit step 0

A student that keeps v1.1's steps 1–3 and takes step 0 to Hub v3's fit or below, in one LoRA. The
recipe is built for that, and every run is read on two numbers beside the renders
(`reports/v11_init.md` for the reference rows):

- **steps 1–3 kept:** `T>S@1` cos against v1.1's 0.932 (Hub v3 0.805);
- **step 0 fitted:** held-out anchor MSE from v1.1's 0.138 toward Hub v3's 0.062.

`S` is read against Hub v3's step 0 with v1.1's steps 1–3: sameness 0.318, cos 0.742. With the
anchor target itself as step 0 the cos is 0.815, which is the room a fitted step 0 has.

Runs start from the v1.1 delta on Kaiming downs. The loop truncates the r96 file to rank 64
(0.914 of the ΔW energy), so a run's own start is slightly off the probed row.

The first two runs are in `reports/v11_init.md`. The anchor alone (`turbo_1116_1558`, 250
iterations, 8 min) passed: anchor MSE 0.082, `T>S@1` cos 0.946, `S` 0.350 / 0.766. The shipped
recipe from the same init (`turbo_1116_1552`) is behind it on every number (0.107, 0.922,
0.383 / 0.721) and has the crisper finish. `1558` draws the teacher's layouts with a soft finish
and broken objects, which come with its own step 0: its steps 1–3 finish the teacher's step 0
cleanly. What is open, in order:

1. **Where the finish is lost and where `1552` gets it.** Cross probes, no training, 3 min each:
   `1558`'s step 0 with `1552`'s steps 1–3 and with v1.1's, and Hub v3's step 0 with `1558`'s
   steps 1–3. A clean render with `1552`'s steps 1–3 behind `1558`'s step 0 means the finish is a
   steps 1–3 matter and the two can be trained apart.
2. **The anchor alone at length.** `1558`'s anchor loss was still falling at 250 iterations. 1k
   is about 33 min. Read: anchor MSE toward Hub v3's 0.062, sameness toward the teacher's 0.315,
   `T>S@1` cos held, and whether a better-fitted step 0 finishes cleanly by itself.
3. **The anchor target cache**, if the length pays. The target depends only on (prompt, ε) and the
   frozen teacher, so a cache leaves one student forward and backward per iteration and lets the
   regression run on a real batch. Not built.
4. **The DM / CDM / GAN terms on steps 1–3 only, from `1558`** (`dm_first_step` off), if candidate 1
   puts the finish in steps 1–3: step 0 keeps the anchor and the terms that sharpen never reach
   it. Read on `T>S@1` cos and the renders.
5. **A term that holds steps 1–3** (the student's velocity on its own rollout states against the
   frozen init's) only if a longer anchor run starts to move `T>S@1` cos. Needs code.
6. **Without training:** Hub v3 on step 0 and v1.1 on steps 1–3 as two LoRAs across two samplers
   (sameness 0.318, cos 0.742). Not the one-LoRA ship.

## A/B protocol

1. **250 iterations**, compare EMA 250. An arm that wins at 250 needs a 1k check before it ships
   (`reports/m15_1k.md`).
2. **Renders are the gate.** 4 steps, cfg 1, `bench/turbo/judge_hard.txt` seeds 1–4 at 768 × 1344
   on the merged15 DiT (int8), beside the teacher (`output/tests/turbo4/teacher_m15/`). Grids go to
   `output/tests/turbo4/cmp/`.
3. **Probe beside the renders** (`bench/turbo/probe_handoff.py --dit
   models/diffusion_models/anima_merged15.safetensors --ts_handoffs 1 --st_handoffs ,`, about 3 min
   per student): held-out anchor MSE, `S` and `T>S@1` cos, and the sameness across seeds (m15
   teacher 0.315).
4. **A warm-started arm is read against its starting point, untrained**, and the read is fixed in
   the config header before the run.
5. One seed per arm so far on the static grid; the dynamic grid's two seeds were far apart
   (`turbo_1114_1745` in the run table).

## Open

- **The recipe on the base teacher.** Every student of the DM recipe collapsed on a merged teacher
  and Hub v3 was trained on base; the split is in `reports/m15_1k.md`. It matters again if
  candidate 4 brings the DM terms back.
- **Fake-lag probe**, before anything that touches the fake (student : fake 1 : 10, the VA-TDM
  teacher-regularised fake): freeze the student at a resume point, run about 40 more fake steps,
  and read how far the DM signal turns on fixed draws. Not built.
- **Café colour drift under cond-only L_CDM.** The aprons leave the teacher's blue on three of
  four seeds; a seed-43 twin of `0958` and a weight-matched arm (`cdm.weight` ≈ 1.7, the control's
  signal rms) would tell seed luck, magnitude and direction apart. Not run.
- **2 steps.** Reopening it waits on a step 0 that carries the layout on 4 steps (`idea.md` 3). A
  student that meets the target is that step 0.

## Dropped without a run

- **Adaptive clip:** the clip at 1.0 already acts on 98 % of the steps under AdamW, so every update
  has unit norm; a threshold at twice the median would give a spike twice the weight it has now.
  Under the cosine the one coupled burst of v1 was at steps 152–162.
- **A longer run from the `weight_svd` init:** v3 shows where it ends, at 16 h; a `weight_svd`
  student also stays inside its init subspace, and the working turbo ΔW sits outside it
  (`reports/warm_v3.md`).
