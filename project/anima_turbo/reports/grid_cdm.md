# Static grid and cond-only L_CDM: both adopted at 250 iterations (2026-11-14, 2026-11-16)

Two arms on the v1 recipe at 250 iterations, each adopted on the user's read of the renders and
shipped in `configs/methods/turbo.toml` on 11-16. The 1000-iteration run of the two together did
not hold the gains (`m15_1k.md`). Moved here from `plan.md` on 2026-11-16; the run rows are in
`runs.md`.

## Static grid (`dmd.dynamic_schedule` off), 11-14

`turbo_1114_1709` against the control `turbo_1113_1105` and its seed-43 twin `turbo_1114_1745`, on
merged14. Grids `output/tests/turbo4/cmp/static_twin_*.jpg`.

| | static `1709` | dynamic, seed 42 / 43 |
|---|---|---|
| sameness `S` (teacher 0.324–0.328) | 0.445 | 0.496 / 0.675 |
| `S` cos | 0.567 | 0.547 / 0.453 |
| `T>S@1` cos | 0.823 | 0.779 / 0.745 |
| anchor MSE | 0.171 | 0.149 / 0.188 |

- The anchor MSE sits between the two dynamic seeds, and the café sign text is worse than the
  control's.
- One static seed, no twin. The dynamic grid is seed-unstable at 250 iterations (the two columns).
- The static grid gives up other step counts.
- CDM signal probe on the run's step-248 bundle: cond-only / CFG'd rms 0.58
  (`bench/turbo/results/20261114-1739-cdm_signal_turbo_1114_1709/`), which made the cond-only L_CDM
  a distinct arm.

## Cond-only L_CDM (`cdm.teacher = "cond"`), 11-16

The paper's form (arXiv:2605.06376 Eq. 9): the cond-only teacher against the fake in L_CDM, with CFG
reaching the student through the on-trajectory DM term only. `turbo_1116_0958` against the merged15
control `turbo_1116_0957` (static grid, weight 1.0; its curves overlap the merged14 static run
`1114_1709`). Probe `bench/turbo/results/20261116-1054-handoff_m15_cdmcond/`, grids
`output/tests/turbo4/cmp/cdmcond_*.jpg`.

| | control `0957` | cond-only `0958` |
|---|---|---|
| sameness `S` (m15 teacher 0.315) | 0.451 | 0.442 |
| `S` cos | 0.547 | 0.608 |
| `T>S@1` cos | 0.777 | 0.841 |
| anchor MSE | 0.151 | 0.138 |

- The CDM signal runs at 0.71 → 0.53 of the control's (probe: 0.58); every other curve is unchanged.
- Every probe number favours it on all three prompts.
- Renders: the guitarist jumps in a skirt seen from below as the teacher draws her (control: track
  pants, standing), the group gains camera variety, and the café aprons drift from the teacher's
  blue to grey and brown on three of four seeds.
- One seed, no twin. The code fallback stays `cfg`, so past run configs reproduce.
