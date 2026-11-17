# Runs

One row per run. Configs are in `_archive/configs/anima_turbo/<id>.toml` (gitignored, with the
long-form run table in its `README.md`); the live one is in [../configs/](../configs/README.md).
Probe numbers are on EMA 250 unless a row says otherwise: sameness (lower is more varied), `S` cos,
`T>S@1` cos, anchor MSE. † = report archived in `_archive/anima_turbo/reports/`.

## merged14 teacher

| Id | Arm | Result | Report |
|---|---|---|---|
| `1112_0833` | 0, control (the `1112_v1.md` baseline), seed 42 | baseline | arm0† |
| `1112_0834` | 0, null twin, seed 43 | 9–10 % shared ΔW with the control | arm0† |
| `1112_0917` | GAN off | GAN is not the walk | arm0† |
| `1112_1044` | S-b, anchor alone | step 0 at the teacher's instantaneous level | step0† |
| `1112_1110` | S-b at student LR 2e-4 | worse | step0† |
| `1112_1143` | S-c.1, `dm_first_step` on | at the twin's level; user: improved, less diverse | step0† |
| `1112_1230` | S-b + `anchor_draws 4` | small consistent gain (anchor MSE 0.157 → 0.147) | step0† |
| `1112_1336` | S-b to 1k, constant LR | flat; length is not the lever | step0† |
| `1112_1410` | `1336` with a cosine tail from 750 | anchor MSE 0.123; the layout does not follow | step0† |
| `1112_1512` | v1: 1k, cosine, EMA, `dm_first_step` | shipped as Hub `v4_preview`; one frontal pose per prompt | [1112_v1](../1112_v1.md) |
| `turbo_1113_1105` | v1 recipe at 250 it, control for the later arms | anchor MSE 0.149 | |
| `turbo_1113_1104` | 1, MFD | failed: the seeds of a prompt draw one image | mfd† |
| `turbo_1113_1221` | warm start from Hub v3 | regresses from v3 untrained (sameness 0.326 → 0.465) | [warm_v3](warm_v3.md) |
| `turbo_1113_1309` | warm start, `dm_first_step` off | not the cause | [warm_v3](warm_v3.md) |
| `turbo_1113_1438` | warm start, fake seeded from v3 too | not the cause | [warm_v3](warm_v3.md) |
| `turbo_1114_1709` | 2, static grid (`dmd.dynamic_schedule = false`) | adopted on the user's read; sameness 0.496 → 0.445, `S` cos 0.547 → 0.567, `T>S@1` cos 0.779 → 0.823 | `bench/turbo/results/20261114-1739-cdm_signal_turbo_1114_1709/` |
| `turbo_1114_1745` | control at seed 43 | the dynamic grid is seed-unstable at 250 it (sameness 0.675 against 0.496) | `bench/turbo/results/20261114-1814-handoff_ctrl_twin/` |

## merged15 teacher (from 2026-11-16)

| Id | Arm | Result | Report |
|---|---|---|---|
| `turbo_1116_0957` | static-grid control | sameness 0.451 (teacher 0.315), `S` cos 0.547, `T>S@1` cos 0.777, anchor MSE 0.151 | |
| `turbo_1116_0958` | 3, cond-only L_CDM (`cdm.teacher = "cond"`) | adopted; every number up: sameness 0.442, `S` cos 0.608, `T>S@1` cos 0.841, anchor MSE 0.138 | `bench/turbo/results/20261116-1054-handoff_m15_cdmcond/` |
| `turbo_1116_1355` | `0958` at 1k (the shipped recipe) | failed: tighter framing, one layout per prompt | [m15_1k](m15_1k.md) |
| `turbo_1116_1552` | student and fake from the v1.1 delta | ahead of `0958` on every number (0.383 / 0.721 / 0.922 / 0.107) | [v11_init](v11_init.md) |
| `turbo_1116_1558` | anchor alone from the v1.1 init | anchor MSE 0.082, `T>S@1` cos 0.946; softer finish | [v11_init](v11_init.md) |
| `turbo_1116_1653` | `1558` at 1k | anchor MSE 0.068; the finish does not come | [v11_init](v11_init.md) |
| `turbo_1116_1730` | `1558` with `grad_accum 4` | ties `1653` at equal compute | [v11_init](v11_init.md) |
| `turbo_1116_1810` | anchor on the 28-step schedule (14 of 28) | adopted as the target to scale: sameness 0.291 (teacher 0.315), `S` cos 0.753 | [anchor_only_recipe](../anchor_only_recipe.md), `bench/turbo/results/20261116-{1827,1830,1833}-*` |
| `turbo_1116_2019` | anchor alone on the 64-prompt cache (128 pairs, about 4 times each) | train anchor MSE 0.036, held-out 0.089: repetition does not generalise | [step0_depth](step0_depth.md) |
| `turbo_1116_2209` | anchor alone on the 512-prompt cache, `grad_accum 2`, 512 updates | renders tie `2019` (p64); 0.4 s/it | [teacher_handoff](teacher_handoff.md) |
| `turbo_1116_2221` | DM / CDM / GAN on steps 1–3 from the teacher-forced handoff, from `2209` | layouts and step 0 held (anchor MSE 0.065, val flat); finish mixed: café outfits closer to the teacher, new garbled text boxes, bocchi seed-1 guitar loses its neck; 4.5 s/it | [teacher_handoff](teacher_handoff.md) |
| `turbo_1117_0823` | `2209` on the full-dataset cache (6484 pairs, one pass, `grad_accum 2`) | best anchor student: anchor MSE 0.070, `S` cos 0.801, `T>S@1` cos 0.944, sameness 0.300; short of the 0.044 read, guitarist seed 2 and the group still break | [step0_depth](step0_depth.md) |
| `turbo_1117_0918` | `0823` at rank 96 / alpha 220 (√r law) | ties `0823`: anchor MSE 0.069, `S` cos 0.811, `T>S@1` cos 0.948, sameness 0.299; renders no better; trails on the train loss at high LR, level at the end | [step0_depth](step0_depth.md) |
