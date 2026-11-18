# Reports

Run table: [runs.md](runs.md). The merged14 arm reports (arm 0 and its twin, the step-0 arms,
MFD, the pre-arm-0 walk measurements) are in `_archive/anima_turbo/reports/`; their verdicts are
in the run table.

## The live line (merged15, v1.1 init)

- [v11_init.md](v11_init.md): the official turbo v1.1 delta on merged15 and the runs seeded from it.
  Its steps 1–3 read the highest `T>S@1` cos of the line (0.932). From it the shipped recipe beats
  the cold start, and the anchor alone beats both on the probe with a softer finish; then `1653`,
  `1730` and `1810` (the 14 of 28 target).
- [m15_1k.md](m15_1k.md): the shipped recipe at 1k on merged15: failed (tighter framing, one
  layout per prompt).
- [teacher_handoff.md](teacher_handoff.md): the anchor alone on a 512-prompt cache (`2209`), and
  the arm after it, DM on steps 1–3 from the input a fitted step 0 hands them (`2221`): step 0
  holds, the finish is mixed at 250.
- [step0_depth.md](step0_depth.md): the mix probe on step 0's residual to the anchor target
  (held-out anchor MSE 0.02–0.04 named as the depth); repeated pairs on the 64-prompt cache; the full-dataset cache
  (`turbo_1117_0823`): 0.070 and `S` cos 0.801; rank 96 (`turbo_1117_0918`) ties it; the 7 of 28
  target is parked.
- [paired_fm_gan.md](paired_fm_gan.md): paired feature matching on the σ 0.75 state (`1123`,
  `1403`, `1442`) and the paired GAN on top of `1442` (`0953`, `1028`); the line uses the GAN on
  the user's read of `1028`.
- [grid_cdm.md](grid_cdm.md): the static grid and the cond-only L_CDM, both adopted at 250
  iterations on the user's read and shipped in the recipe; one seed each.

## Step 0 on merged14

- [handoff.md](handoff.md): the handoff probe, per student step, of how much of the teacher's
  layout is lost; step 0 (σ 1 → 0.9) loses the most, and with the teacher drawing step 0 the
  student's steps 1–3 keep the layout in the renders.
- [step0_fit.md](step0_fit.md): layout recovery against the step-0 fit, and Hub v3's step 0; Hub v3
  (10k it) sits at 0.3 of the reference against 0.85 for the 1k students.
- [warm_v3.md](warm_v3.md): Hub v3 untrained on merged14 sits at the teacher's seed diversity, and
  the v1 recipe pulls a warm start from it back toward the control.
