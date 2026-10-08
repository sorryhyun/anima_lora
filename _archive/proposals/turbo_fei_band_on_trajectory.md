# FEI band-deficit, revived on-trajectory — log-only Phase 0 inside the DP-DMD loop

Status: **CLOSED 2026-06-20** — Phase 0 run, gate hit CLOSE. See
`docs/findings/turbo_fei_band_revival_closed.md`. Measured on-trajectory
(`bench/turbo/probe_fei_band.py`, anima_turbo_P_5k, n=60 paired, Site A σ-matched
exactly): the student does **not** over-blur on its rollout (Site A gap_low
−0.0014, frac_over 0.03) and the over-blur arm `dh` is dead in the lever's
mid/low-τ region while the falsified LF arm `dl` fires at 100% frac — the inverse
persists at the right distribution. The diversity anchor already absorbed the
over-blur. The line is dead twice; do not re-propose FEI/band levers on turbo.

---

Phase 0 was pure logging (zero behavior change, zero extra forwards) and was
designed to be able to **close this line permanently** — that outcome is as
valuable as the lever.

Premise sources: `docs/findings/turbo_fei_band_deficit_falsified.md` (the
falsification this revives *on its own stated terms*), `docs/methods/turbo.md`
+ `docs/structure/turbo.md` (the loop that changed the premises).

## Why this is worth reopening — the falsification's precondition flipped

Item 2 (FEI band-deficit reweighting of the CFG-uplift) falsified in the CA-era
loop for a precisely diagnosed reason: **the loss measured at renoised *real* data,
where the only FEI gap is the trivial "one-shot x0 is noisier than the teacher's"**
(`w_low` fired weakly, the inverse of the validated arm), while the over-blur the
lever targeted lives on **the student's own rollout states — which DMD2 single-call
training never visited**. The finding's own revival condition:

> The right way to revive it is to measure the band deficit *on the student's
> rollout states*, which only exist if training visits them.

The 2026-05-30 DP-DMD migration (commit `9410a3a` — the same commit that deleted
`ca_band.py`) changed both facts **as a side effect**:

1. **Training now rolls the student through its genuine N-step rollout**
   (`scripts/distill_turbo/distill.py:828-897`) — the on-trajectory states exist
   in-loop, every step.
2. **The DMD point is now a renoise of the student's own output**, not of real
   data: `x_renoised_dm = renoise(x_pred.detach(), τ_dm, ε_dm)` (`distill.py:960`).
   The distribution the gradient applies at is the student distribution.

So the precondition the falsification said was unmeetable is now met by an
architecture change made for unrelated reasons. Nobody has re-measured. Phase 0 is
that re-measurement — *measurement only*, because the falsification's other lesson
is that wiring a lever before measuring at the right distribution is how this line
burned ~2 training runs last time.

There is also a live counter-hypothesis that makes the null outcome meaningful:
**the diversity anchor may already have fixed the over-blur** (it was introduced to
de-collapse exactly the mode-seeking behavior the old band lever chased at the
symptom level). If the gap is gone, close the line with a finding and stop carrying
it.

## Phase 0 — log-only band-gap telemetry (zero extra forwards)

Two measurement sites, both computable from tensors already in scope in the loop.
FEI = `library/runtime/fei.py:75::compute_fei_2band` on `[B,16,H,W]` fp32,
`σ_low = min(H_lat, W_lat)/16` (divisor 16 won the original Phase-0 SNR sweep —
keep it for comparability with the falsified run's numbers).

### Site A — trajectory latents at matched σ (the original Phase-0 quantity, now live)

The original GO probe measured FEI on **trajectory latents** `z_t`; the falsified
loss measured x0-estimates. Site A is the trajectory quantity, paired:

With the shipped defaults (`k_anchor=4`, `teacher_anchor_steps=8`,
`student_steps=2`, `flow_shift=3`), the teacher anchor endpoint and the student's
post-step-0 state land at the **same σ = 0.75**, integrated **from the same `ε`**
(`distill.py:805,811,832`):

```
teacher: ε ──(4 CFG Euler steps over {1.0, .954, .900, .833, .750})──► z_tk   @ σ=0.75
student: ε ──(1 step, head 0)───────────────────────────────────────► z1     @ σ=0.75
gap = e_low(z1) − e_low(z_tk)        # paired, same ε → low variance
```

`z_tk` is computed at `distill.py:811-817` (anchor rollout), `z1` at
`distill.py:832-840` (`x` after the step-0 Euler update). Both are live tensors;
the gap costs two Gaussian blurs. Sign convention matching the original probe:
**gap > 0 = `student_over_low` = over-blur** — the arm Phase 0 validated
(`Δ_low = −0.0118` at t=0.75 on turbo_C means teacher had *less* LF, i.e. student
over-low; re-derive the sign once against the probe code before trusting plots).

This is exactly the comparison the falsified design never had: same quantity
(trajectory FEI), same σ, same noise draw, student's own rollout.

### Site B — x0-scale gap at the DMD point (the lever's site)

If a lever ever gets wired, it acts on `grad_dm` at the DMD point — so also log
the deficit where the lever would live, σ-matched (the fix the falsified wiring
got right):

```
e_S = FEI(x_pred)                                   # student endpoint, already in scope
e_T = FEI(x_renoised_dm − τ_dm · v_real_cond_dm)    # teacher 1-step x0 est., v already computed (:961)
dh  = relu(e_high_T − e_high_S)                     # HF deficit → over-blur arm (w_high)
dl  = relu(e_low_T − e_low_S)                       # LF deficit → the arm that killed item 2
```

`v_real_cond_dm` is the CFG'd teacher velocity already computed for the DMD gap —
**no extra forward**. Because `e_low + e_high ≡ 1` (simplex), at most one arm is
positive per sample — the bounded-by-construction property the finding marked as
"sound, just mis-sited". Bin the logs by `τ_dm` (e.g. 5 buckets): the falsified
run's headline was a *sign structure* across the schedule, and a τ-pooled mean
would hide a sign flip. Note `x_pred`'s semantics depend on `grad_step` (true
endpoint under `all`/`last`-tail, one-step x0-pred at step *g* under `random`) —
log the mode alongside.

### Plumbing

- New TB scalars via `TurboMetrics` (`scripts/distill_turbo/metrics.py:58`):
  `band/traj_gap_low@0.75`, `band/dh_pos`, `band/dl_pos`, per-τ-bucket variants.
- Config: `[fei_log] enabled = false` in `configs/methods/turbo.toml` +
  `--fei_log` flag (`scripts/distill_turbo/config.py`). Off → byte-identical.
  On → a few separable convs per step, no sync added if accumulated GPU-resident
  like the existing metrics (single-flush pattern already in `TurboMetrics`).
- Site A requires `base_loss="dpdmd"` (needs the anchor rollout); Site B works
  under plain `dmd` too. If `k_anchor`/`teacher_anchor_steps` change the σ match,
  compare at the nearest teacher grid point and log the σ pair.

### Gate (pre-registered — decide the read BEFORE looking)

Attach the logging to the next routine turbo run (≥500 steps; no dedicated run
needed). Then:

- **`dh_pos` (HF deficit / over-blur) fires materially** — mean ≳ 0.01 in the
  mid/low-τ buckets, or Site A shows a persistent `student_over_low` gap at
  σ=0.75 → **Phase 1** (wire the `w_high` lever).
- **`dl_pos` dominates again, or both ≈ 0** → **CLOSE the line permanently.**
  Write the closing entry in `docs/findings/` (the anchor fixed over-blur, or the
  gap never lived at the new sites either) and stop re-proposing FEI levers on
  turbo. The 2026-05 falsification plus this null = the line is dead twice.
- Cross-check the verdict against **sample grids** at `--infer_steps 2 --cfg 1.0`:
  over-blur is visible. A scalar verdict that disagrees with the grids loses
  (the L2P / mod-guidance lesson: read the images, scalars are guides).

## Phase 1 — band-split the DMD gradient (only on a `dh_pos` GO)

Resurrect the DoG LP/HP split from `ca_band.py` (deleted in `9410a3a`; recover via
`git show 9410a3a^:scripts/distill_turbo/ca_band.py` — the finding explicitly kept
the mechanism as "sound, just mis-sited"). The lever moves from the dead `δ_cfg`
to the live DMD gradient, between `dm_x0_norm` and the surrogate assembly
(`distill.py:971-979`):

```python
grad_dm' = w_low · LP(grad_dm) + w_high · HP(grad_dm)
w_high   = 1 + β · dh_pos          # per-sample, from Site-B stats
w_low    = 1 + β · dl_pos          # simplex ⇒ at most one ≠ 1
```

- Split the **post-`dm_x0_norm`** gradient so the per-sample magnitude
  normalization and the bounded `≤ 1+β·Δ` amplification compose predictably;
  f-distill's `h` (a per-sample scalar, `distill.py:1029`) multiplies through
  either way.
- `β = 0.2`, τ-window from the Site-B bins (only reweight where the deficit
  actually fired — don't import the old `[0.30, 0.95]` window blind).
- `β = 0` must be bit-identical to current behavior (invariant test).

### Hard guardrails (from memory — these killed the neighbors)

- **Feedback control on the gradient only, never a target on the image.** Direct
  FEI-statistic matching destroyed quality.
- **Don't respond to a weak arm by raising β** — that is exactly the falsified
  move ("amplify an unvalidated arm"). If the arm is weak at the right
  distribution, the lever isn't there.
- **No fraction-of-Δ readouts** in the bench
  ([[project_spectral_fraction_metric_inverts]] — they reward no-ops).

### Decision gate (Phase 1)

A/B at fixed seed/data/iterations: `β=0` vs `β=0.2`, judged on grids (sharpness
without artifact-chasing), CMMD non-regression, and diversity (`diversity.py`) —
the old lever was "mildly antagonistic" to its own goal, so the bar is a visible
HF win, not a scalar nudge.

## Sequencing

```
Phase 0 logging (off-by-default flag, ride any ≥500-step run)
   ├─ dh_pos ≈ 0 / dl_pos again ──► CLOSE: docs/findings/ entry, line retired for good
   └─ dh_pos fires on-trajectory ──► Phase 1 band-split grad_dm (β=0 default)
                                        └─ Gate: grids + CMMD + diversity ──► ship / kill
```

## Contributing tier

Phase 0 is metrics-only (no numerics change; flag off = byte-identical) — invariant
test on the flag suffices. Phase 1 is numerics-changing → **Tier 1.5**: bench
script (`bench/turbo_fei_band/`, `bench/_common.py` envelope) + the `β=0`
bit-identity invariant test.

## References

- `docs/findings/turbo_fei_band_deficit_falsified.md` — the falsification, root
  cause (distribution + quantity mismatch), and the revival condition this
  proposal satisfies; original GO numbers (turbo_C rollout, d=16, `student_over_low`
  at every stage) to compare Site A against.
- `scripts/distill_turbo/distill.py:805-897` (rollout states), `:958-979` (DMD
  point + grad assembly), `:1036-1039` (surrogate).
- `library/runtime/fei.py:75` — `compute_fei_2band`; divisor-16 from the original
  Phase-0 SNR sweep.
- `git show 9410a3a^:scripts/distill_turbo/ca_band.py` — the kept-sound DoG split
  to resurrect.
- [[project_turbo_dmd_x0_norm_wins]], [[project_turbo_alpha4_overdistill]] — the
  over-blur / variance-inflation history this lever was born from.
