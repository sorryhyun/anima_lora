# Turbo FEI band-deficit lever — CLOSED (dead twice)

Status: CLOSED 2026-06-20. The FEI band-deficit reweighting of the CFG-uplift
`δ_cfg` was killed once in the CA-era loop (2026-05, wrong distribution) and again
on-trajectory (2026-06, the right distribution, σ-matched). Both deaths point the
same way. **Do not re-propose FEI / band-split levers on the DP-DMD loop.**

The lever reweights `δ_cfg` in turbo distillation by where the student's FEI lags
the teacher's (FEI = 2-band split of unit-norm latent energy, `e_low+e_high≡1`):

```python
δ_cfg' = w_low · LP(δ_cfg) + w_high · HP(δ_cfg)
w_high = 1 + β · relu(e_high_T − e_high_S)
w_low  = 1 + β · relu(e_low_T  − e_low_S)
```

Because the two bands sum to 1 the two `relu`s can never both fire, so total CA
magnitude is bounded by `1 + β·max(Δ)` — feedback control on `δ_cfg`, not a target
on the image (unlike the direct FEI-statistic matching that destroyed image
quality, `project_fera_probe_2band_decision`). The mechanism was always "sound,
just mis-sited"; it is now also un-needed.

---

## Act 1 — CA-era falsification (2026-05): the live run inverted Phase 0

`item2_plan.md` / `proposal.md` §2. Phase 0 said GO; the live training run said
the validated lever isn't there — the active arm is the inverse one, at ~identity
strength.

Why Phase 0 looked clean. Probe on `anima_turbo_C` (n=90 = 30 artists × 3 seeds,
`bench/fera_artist/results/20260528-1902-turbo_C_phase0/`) at the student's 4-step
rollout stages: direction `student_over_low` everywhere (SNR peaked 2.25 at stage 2 /
t=0.75, sign 100%), so the `w_high` arm should fire and `w_low` stay inert. All four
decision thresholds cleared — a textbook GO.

How it was wired. A `ca_band` distill module + call site `distill.py:296–330`,
`beta=0.2, divisor=16, window=[0.30,0.95]`, trained into `anima_turbo_E_1k` /
`anima_turbo_F`. A diagnosis pass caught one σ-mismatch (measuring the gap at τ_ca,
the teacher's denoise of the renoised x_pred, injects a ~0.08 LP shift that masks the
≈0.012 lever and inverts the TB sign); the fix added an extra no-grad teacher forward
at the student's own `(x_t, t, c)` so both x0 estimates live at the same σ.

Why it falsified. Live band scalars on `anima_turbo_F`
(`output/logs/turbo/20260528-212337`):

| scalar | head → tail | reading |
|---|---|---|
| `band_w_high` | 1.000 → 1.000 | the Phase-0 arm is dead |
| `band_w_low`  | 1.005 → 1.011 | the arm that fires (≤4% reweight — near no-op) |
| `band_dh_pos` | ~0 | HF deficit ≈ never positive |
| `band_dl_pos` | 0.027 → 0.054 | LF deficit is the live signal, and growing |

Root cause — Phase 0 and the loss measure different things. Phase 0 probed the
student's own 4-step rollout states (where the integrated output is over-blurred →
`w_high`); the loss measures at `x_t = (1−t)·x₀ + t·ε`, a DMD2 renoise of real clean
data, where the student's one-shot x0 is merely noisier than the teacher's →
`w_low`. Opposite-sign gaps. The over-blur failure lives on the student's own
inference trajectory, which DMD2 single-call training never visits. The lesson: a
go/no-go probe must measure the *same quantity at the same distribution* the loss
will see. (Bumping β is the wrong response — it amplifies an unvalidated arm in a
plausibly counterproductive direction.) The stated revival condition was to measure
the band deficit *on the student's rollout states* — which only exist if training
visits them.

---

## Act 2 — on-trajectory revival (2026-06): legitimate reopen, clean null

The 2026-05-30 DP-DMD migration (commit `9410a3a`) met that condition as a side
effect: the student now rolls its genuine N-step trajectory in-loop and the DMD point
is a renoise of the student's own output (`distill.py:1006`), not real data.
`_archive/proposals/turbo_fei_band_on_trajectory.md` reopened the line — measurement
only.

Instrument. `bench/turbo/probe_fei_band.py` on the trained checkpoint
`anima_turbo_P_5k.safetensors` (student_steps=4, anchor 6/12, flow_shift=3,
teacher_cfg=4), n=30 × 2 ε = 60 paired samples. A trained checkpoint is the right
instrument — the zero-init student ≈ teacher for the first thousands of steps, so a
short run's gap is the same trivial one Act 1 warned about.

- Site A — trajectory LF gap at MATCHED σ. At student 4 / anchor 6/12 the
  student's post-step-0 state z1 (σ=0.9) coincides exactly with the teacher anchor's
  step-3 state, both integrated from the same ε → `|Δσ|=0`, the clean paired
  comparison the 2026-05 wiring lacked.
- Site B — x0-scale band deficit at the DMD point, swept over a fixed τ grid (the
  falsified headline was a *sign structure across the schedule*, so a τ-pooled mean
  would hide a flip).

Result — the falsified inverse persists, the trajectory is clean.

Site A (σ-matched): `gap_low = −0.00140 ± 0.00116`, `frac_over_low = 0.03`. The
student carries less LF than the teacher at σ=0.9 — it does not over-blur on
its own rollout. 97% of samples show no over-low.

Site B (per τ, n=60):

| τ | dh_pos (→ w_high / over-blur) | dl_pos (→ w_low) | frac dh | frac dl |
|---|---|---|---|---|
| 0.10 | 0.00000 | 0.00381 | 0.00 | 1.00 |
| 0.30 | 0.00000 | 0.01401 | 0.00 | 1.00 |
| 0.50 | 0.00000 | 0.02816 | 0.00 | 1.00 |
| 0.70 | 0.00113 | 0.03346 | 0.07 | 0.93 |
| 0.90 | 0.01922 | 0.03273 | 0.45 | 0.55 |

The over-blur arm `dh` is dead for τ≤0.5 and only fires at τ=0.9 (the noisy-renoise
end where the teacher's own 1-step x0 is noisy — the trivial artifact, not the lever).
The LF arm `dl` — the one that killed Act 1 — fires at 100% frac across low/mid τ.

---

## Verdict and why

CLOSE. There is no on-trajectory over-blur deficit for a `w_high` band lever to
act on. The most likely mechanism is the one the proposal pre-registered as the
meaningful null: the diversity anchor (introduced to de-collapse exactly the
mode-seeking behavior the old band lever chased) already fixed the over-blur. Site A's
no-over-low directly supports that — the DP-DMD anchor + `dm_x0_norm`
(`docs/methods/turbo.md`) absorbed the over-blur the lever was born from
([[project_turbo_alpha4_overdistill]], [[project_turbo_dmd_x0_norm_wins]]).

## What survives

- `bench/turbo/probe_fei_band.py` — the reusable Phase-0 instrument (Site A σ-matched
  by construction; envelope under `bench/turbo/results/`). Re-run on a future
  checkpoint only if the loop's diversity mechanism changes materially.
- The mechanism is sound, just mis-sited and un-needed. The DoG LP/HP split +
  bounded per-sample deficit is correct and reusable, but do not resurrect `ca_band.py`
  (`enabled=false`; β=0 is bit-identical to off and reclaims the ~14% wall-clock of the
  no-op teacher forward).

## Pointers

- `_archive/proposals/turbo_fei_band_on_trajectory.md` — the revival design (archived
  2026-06-20, closed by this finding); `item2_plan.md` / `proposal.md` §2 — the
  original CA-era wiring, marked falsified.
- Memory: `project_turbo_fei_gap_phase0` (annotated with the inversion).
- Context: `sigma_signal_where_anima_resolves.md` (the σ-window motivation),
  `project_turbo_dmd_x0_norm_wins` / `project_fera_probe_2band_decision` (the
  over-blur / seed-diversity history, why direct FEI matching fails).
- Runs: `20260528-212337` / `20260528-194036` (Act 1), `20260620-1327-phase0_n30`
  (Act 2).
