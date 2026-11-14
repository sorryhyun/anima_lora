# Runs

One TOML per run, named by its run id (`mmdd_hhmm`, later `turbo_mmdd_hhmm`). Launch with
`make turbo ARGS="--config project/anima_turbo/configs/<id>.toml --queue"`. Paths in the table are
relative to `project/anima_turbo/`.

| Id | Arm | Change against control | State |
|---|---|---|---|
| `1112_0833` | 0, control | the baseline in `1112_v1.md`, seed 42 | done, rendered |
| `1112_0834` | 0, null twin | seed 43 | done, rendered |
| `1112_0917` | GAN-off probe (measurement only) | `gan.weight_gen = 0` | done: GAN is not the walk |
| `1112_1044` | S-b, anchor alone | `dpdmd.anchor_only`, no CDM / GAN / fake warmup | done: step 0 at the teacher's instantaneous level (`reports/step0.md`) |
| `1112_1110` | S-b2 | `1112_1044` at student LR 2e-4 | done: worse. User's read: 2e-4 steps are too large for the noise, 5e-5 steps too small to get past the instantaneous level in 250 it |
| `1112_1143` | S-c.1 | `dpdmd.dm_first_step` on (anchor kept) | done: probe `S` 0.445 / 0.441 (EMA 250 / 125), at the twin's level; guitarist pose moves to the jump, camera does not; raw 250 broken (`reports/step0.md`). User's read: improved, less diverse |
| `1112_1230` | S-c.2 probe | S-b + `dpdmd.anchor_draws 4` | done: small consistent gain, every paired number favours it (anchor MSE 0.157 → 0.147, `P>S@1` +0.01 / +0.03); passes by the repo gate rule, size thresholds in its header not met (`reports/step0.md`) |
| `1112_1336` | S-b long | S-b resumed (from step 125) to 1000 iterations, EMA span kept at 50 updates | done: held-out anchor MSE 0.165 / 0.143 / 0.155 / 0.140 at 250 / 500 / 750 / 1k, flat; length on the constant LR is not the lever (`reports/step0.md`) |
| `1112_1410` | S-b long, annealed tail | `1112_1336` from step 750 with `lr_schedule cosine` | done: 0.123 raw / 0.127 EMA; the anneal is what the 500-iteration students had. `P>S@1` 0.495 / 0.587, the layout does not come with it |
| `1112_1512` | v1 (`1112_v1.md`) | 1000 it, cosine, EMA 0.2 of summed LR, `dm_first_step` on, saves every 250 | done; 1k EMA shipped 11-12 as Hub `anima_turbo_4step_v4_preview`. Against Hub v3: the multi-character layout holds one scene; the solo guitarist draws one frontal pose on every seed, raw and EMA |
| `turbo_1113_1105` | control for the post-v1 arms | the v1 recipe at 250 it (cosine over the 250), saves every 125 | done, rendered; held-out anchor MSE 0.149 |
| `turbo_1113_1104` | 1, MFD | `turbo_1113_1105` + `mfd.dtau 0.1` / `max_steps 4` / `to_zero` | done: fails on the renders. Cleaner lines and all five characters on every seed, but the four seeds of a prompt draw one image and the sign text is worse (`reports/mfd.md`) |
| `turbo_1113_1221` | warm start from Hub v3 | `turbo_1113_1105` with the student seeded from Hub v3 | done: passes the `plan.md` gate against the control, but regresses from v3 untrained on merged14 on every number (sameness 0.326 → 0.465, `S` cos 0.639 → 0.593, anchor MSE 0.063 → 0.070) and in the user's read; the 1k run is dropped (`reports/warm_v3.md`) |
| `turbo_1113_1309` | warm start, `dm_first_step` off | `turbo_1113_1221` with `dpdmd.dm_first_step = false` | done: not the cause. Anchor MSE stays at v3's 0.063, but step 0 still raises the sameness (cross cell 0.405 against v3 0.326, `1221` 0.429); steps 1–3 regress as before (`T>S@1` cos 0.773) (`reports/warm_v3.md`) |
| `turbo_1113_1438` | warm start, fake seeded too | `turbo_1113_1221` with `network.fake_init_weights` = Hub v3 | done: not the cause. The DM gap starts lower (`dm_rel_gap` 0.214 against 0.292 in the first 25 it), but sameness is 0.483 (v3 0.326, `1221` 0.465); `S` cos 0.610, anchor MSE 0.070 (`reports/warm_v3.md`) |
