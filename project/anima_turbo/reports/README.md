# Reports

Run table: [runs.md](runs.md). The merged14 arm reports (arm 0 and its twin, the step-0 arms,
MFD, the pre-arm-0 walk measurements) are in `_archive/anima_turbo/reports/`; their verdicts are
in the run table.

## The live line (merged15, v1.1 init)

- [v11_init.md](v11_init.md): the official turbo v1.1 delta on merged15. Its steps 1–3 are the
  best of the line (`T>S@1` cos 0.932) and its step 0 is unfitted. From it the shipped recipe beats
  the cold start, and the anchor alone beats both on the probe with a softer finish; `1653`,
  `1730` and `1810` (the 14 of 28 target) follow.
- [m15_1k.md](m15_1k.md): the shipped recipe at 1k on merged15 failed (tighter framing, one
  layout per prompt); the 250-iteration A/B does not predict 1k.
- [teacher_handoff.md](teacher_handoff.md): the anchor alone on a 512-prompt cache (`2209`) leaves
  the finish where the 1k runs left it; the arm after it trains DM on steps 1–3 from the input a
  fitted step 0 hands them (`2221`): step 0 holds, the finish is mixed at 250.
- [grid_cdm.md](grid_cdm.md): the static grid and the cond-only L_CDM, both adopted at 250
  iterations on the user's read and shipped in the recipe; one seed each.

## Why step 0

- [handoff.md](handoff.md): step 0 (σ 1 → 0.9) is the largest single loss of the teacher's
  layout; with the teacher drawing step 0, the student's steps 1–3 keep it.
- [step0_fit.md](step0_fit.md): the layout comes back in proportion to the step-0 fit; Hub v3
  (10k it) sits at 0.3 of the reference against 0.85 for the 1k students.
- [warm_v3.md](warm_v3.md): Hub v3 untrained on merged14 sits at the teacher's seed diversity, and
  the v1 recipe pulls a warm start from it back toward the control.
