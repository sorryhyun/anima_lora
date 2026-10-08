# Dual-pool gradient routing — Phase 0 A/B

Wiring + smoke A/B for the proposal in
`_archive/proposals/turbo_dual_pool_grad_routing.md` (gitignored, local)
(PR #73). Split the turbo student into two always-on plain-LoRA pools — pool A
(diversity) sees **only** the step-0 diversity gradient, pool B (quality) **only**
the DMD/GAN/CDM refinement gradients — and check whether that parameter-level
separation buys diversity over the shipped single joint stack. Output stays a
plain LoRA (exact `ΔW_A + ΔW_B` concat).

## One variable

| Arm | config | student layout | merged rank |
|---|---|---|---|
| **A** | `arm_A_dual_32_32.toml` | dual: pool B r=32 (warm) + pool A r=32 (zero-init) | 64 |
| **B** (baseline) | `arm_B_single_r64.toml` | single r=64, joint grads (shipped) | 64 |

Both inherit the CDM + div0.1 superturbo recipe
(`configs/gui-methods/custom/superturbo.toml`) — warm start, GAN, `dynamic_schedule`,
`div_weight=0.1`, seed, data all fixed. Symmetric 32/32 splits the r=64 exactly, so
A vs B differs in **only** the routing.

> **NFE=2 caveat.** This recipe is `student_steps=2` (the superturbo NFE=2 line,
> marked closed with a critic collapse ~1750).
> Dual-pool is an objective-side lever (the one open bucket), so reopening here is
> defensible, but the diversity read happens where the critic is touchiest. Keep
> the run ≤750 iters; if the critic runs away before the gate, re-run at NFE=4
> (`student_steps=4`, drop `dynamic_schedule`) before drawing a verdict.

## Run

```bash
bash bench/turbo/dual_pool/run_phase0.sh        # queues both arms on the daemon
make run-status RUN=anima_superturbo_dualpool_A_v1
```

750 iterations, ckpt at 250 / 500 / 750. The routing itself is guarded by the
invariant test `tests/test_turbo_dual_pool.py` (grad lands on A alone / B alone;
merged concat reproduces `ΔW_A + ΔW_B`; div_scale dial) — run it first:

```bash
python -m pytest tests/test_turbo_dual_pool.py -q
```

## Reads (rank by rendered NFE=4 grids, NOT fm_mse)

- **Primary** — rendered 4-step grids at `--cfg 1.0` (`make gen`), fixed prompt
  set × seed sweep, human A/B on the 750 ckpts.
- **Diversity** (the axis this exists for) — cross-seed `ac_sim` + seed-grid spread
  per prompt. This is the loop's own `validate_every_n_steps=250` signal; `div_loss`
  is NOT it (it measures anchor-hitting, not diversity survival).
- **Quality guard** — glyph probe (`bench/turbo/glyph_scatter_probe.py`) +
  saturation vs baseline. Pool B must not regress with the DMD grad now denied A.
- **Bonus (unique to this design)** — render Arm A at `--div_scale 0 / 0.5 / 1.0`
  (the merge-time dial baked by `save_student`). If the dial visibly trades
  diversity against the init's mode, that is direct evidence the routing isolated
  the axis, independent of whether the arm wins outright. To emit a dialed
  checkpoint from a trained bundle, re-save with the dial (the loop saves at 1.0):

  ```python
  # after building turbo + loading the dual-pool student weights
  turbo.save_student("student_div0.5.safetensors", div_scale=0.5)
  ```

## Gate

- **Adopt** — Arm A ≥ baseline on rendered NFE=4 **AND** measurably better on
  cross-seed diversity **AND** no glyph/saturation regression.
- **Kill** — Arm A ties on diversity ⇒ the single student was never
  parameter-interference-bound (same verdict framework as `per_step_expert`);
  close the line, keep the doc + this dir as the record.
- **Not an auto-kill** — a diversity win with a quality loss: re-run at pool B=128
  (`student_rank` up, `div_pool_rank` unchanged) before deciding — capacity, not
  routing, may be binding.

## Runtime-verify (before trusting a full run)

The routing toggles `requires_grad` on the pools every iteration. The
flips are expected not to force a dynamo re-trace, and the proposal counts on
it — but the **first full compiled run must confirm** the recompile counter stays
flat (`make run-status` it/s should not tank after step 1). If it recompiles per
iteration, that is the compile-guard interaction to fix, not a routing bug.
