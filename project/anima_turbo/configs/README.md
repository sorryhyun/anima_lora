# Configs

The live run config only. Launch with
`make turbo ARGS="--config project/anima_turbo/configs/<id>.toml --queue"`; name a new run
`turbo_mmdd_hhmm`. Finished runs move to `_archive/configs/anima_turbo/`; their results are in
[../reports/runs.md](../reports/runs.md).

- `turbo_1116_1810.toml`: anchor alone from the v1.1 init on the 14 of 28 target, merged15 teacher
  (live anchor, no cache).
- `turbo_1116_2209.toml`: the anchor alone on the 512-prompt target cache, `grad_accum 2`.
- `turbo_1116_2221.toml`: DM / CDM / GAN on steps 1–3 from the teacher-forced handoff
  (`dpdmd.teacher_handoff`), warm-started from `2209`. The template for the next DM arm.
- `turbo_1117_0823.toml`: the anchor alone on the full-dataset cache (`m15_k14of28_pall`, one pass).
  The template for the next anchor arm.
- `turbo_1117_0918.toml`: `0823` at rank 96 / alpha 220 (the √r law).
- `turbo_1117_1020.toml` / `turbo_1117_1021.toml`: the student LR A/B on a short run, `0823` at rank 64 on
  the 1024-prompt subset of the full cache (`m15_k14of28_p1024`, hardlinked: the p512 keys plus 512
  more; 2048 iterations), `student_lr` 5e-5 against 2e-5.
- `turbo_1117_1105.toml`: paired feature matching (`dpdmd.fm_weight` 0.01) on the `1020` short run.
- `turbo_1117_1054.toml`: `0823` at `student_lr` 2e-5 on the full cache, on the user's read of the
  short A/B's renders.
- `turbo_1117_1123.toml`: `1105` at `dpdmd.fm_weight` 0.1.
- `turbo_1117_1403.toml` / `turbo_1117_1442.toml`: paired feature matching at 0.05 (middle tap) and at 15 on block 27.
- `turbo_1118_0953.toml`: `1442` + the paired GAN (plan.md arm 3), disc on block 27 of the FM forward, λ 0.5.
- `turbo_1118_1028.toml`: `0953` with the generator gradient held at 0.5x the FM gradient (`gan.paired_ratio`).
- `turbo_1118_1122.toml`: `1028` at `gan.paired_ratio` 1.0.
- `turbo_1118_1215.toml`: `1122` with `gan.disc_pool` 2.
- `turbo_1118_1302.toml`: `1122` with FastGen's conv head on taps 13 / 20 / 27 (`gan.disc_head = "conv"`, `gan.feature_blocks`, `disc_hidden` 256).
- `turbo_1118_1353.toml`: `1302` with `gan.r1_weight` 100.
