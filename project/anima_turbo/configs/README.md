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
