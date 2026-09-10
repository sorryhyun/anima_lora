# SCFM (velocity-space self-distillation) — CLOSED on Anima, removed from tree

Status: **CLOSED 2026-06-30.** The standalone SCFM objective (Shortcutting
Pre-trained Flow Matching, NeurIPS 2025) was built as a selectable
`base_loss="scfm"` turbo objective, trained across five runs, and falsified. Its
consistency term (Term B) is inert on Anima in both the paper-faithful
(`renoise`) and the off-trajectory (`rollout`) variants, so the objective reduces
to per-step flow-matching distillation onto the teacher's instantaneous field —
which renders blurry at 4 steps by construction. Code removed (loss loop,
config knobs, EMA plumbing, the `scfm` target, tests, `scfm.toml`); the design proposal
is archived at `_archive/proposals/turbo_scfm.md`. Do not re-propose a pure
velocity-fidelity / self-consistency distiller for Anima.

## What SCFM is, and what we hoped for

SCFM trains a single velocity field to be self-consistent across step sizes so
few-step Euler matches the teacher — no critic, no GAN, no diversity anchor. Two
terms (paper Eq. 13): Term A pins a renoised real latent's coarse-step
velocity to the CFG-guided teacher (the quality ceiling); Term B enforces "one
coarse Euler step == two finer sub-steps" on a stop-grad EMA copy of the student,
straightening the trajectory. Output is a plain LoRA, inferred at
`--infer_steps 4 --cfg 1.0` exactly like the DP-DMD student.

The motivation ([[project_turbo_teacher_gap_2026_06_29]]) was that DP-DMD
mode-collapses (pose-diversity loss, garbled text) — a fidelity objective that
just matches the teacher looked like the cheaper, structurally-right arm for the
gap we measured. SCFM is also ~6× cheaper per step than DP-DMD+GAN.

## The load-bearing risk fired exactly as pre-registered

The proposal's risk #1: *SCFM cannot exceed the teacher; if the naive few-step
teacher is itself collapsed, no velocity-fidelity method beats it at 4 steps.*
Phase-0 (no training) already measured this:

- Naive 4-step teacher (Arm 2) is dramatically worse than the DP-DMD student
  (Arm 3) — pale, washed-out, sketch-like. The 28→4-step gap is real
  discretization loss, recoverable only by distribution-matching.
- On-manifold consistency residual ≈ 0.044 (the base field is already ~95%
  straight at the student step sizes). Term B has almost nothing to optimize.

The straightness probe re-opened it (the cfg=4 transport is nearly straight and
non-crossing in the low-σ band — the reflow-friendly regime), so Phase 1 was built
and trained to settle Arm-1 reachability. It settled NO.

## The evidence — five runs, residual never leaves the base floor

`train/scfm_consistency_residual` is the gate: it must climb off the ~0.05 base
floor for Term B to be doing real work. It never did, across every configuration.

| run | term_b_point | k_ratio | lr | steps | residual trajectory | outcome |
|---|---|---|---|---|---|---|
| `anima_turbo_scfm` | renoise | 0.4 | 1e-5 | 1500 | flat ~0.05 | soft / Arm-2 |
| `anima_turbo_scfm_highlr` | renoise | 0.4 | 5e-5 | 1500 | flat ~0.05 | sharper (Term A only), still soft |
| `anima_scfm_accum` | rollout | 0.4 | 5e-5 (accum 2) | 1500 | 0.034 → 0.075 → 0.068 | inert |
| `anima_scfm3` | rollout | 0.4 | 5e-5 | 735† | 0.026 → 0.041 → 0.060 → 0.053 | inert |
| `anima_scfm2` | renoise | 0.9 | 2e-5 | 15000 | 0.042 → 0.072 → 0.047 → 0.073 | blurry + regressed |

† `anima_scfm3` was killed at ~735 steps (no checkpoint saved).

The 15k run (`anima_scfm2`) is the decisive one and it was the worst possible
config — `k_ratio=0.9` (90% Term A ⇒ near-pure per-step FM distillation onto the
teacher's instantaneous field = the washout) + `renoise` (Term B provably inert) +
`lr=2e-5` (below the documented 5e-5 sharpness win). It not only plateaued from
1.5k, it regressed past ~7.5k:

```
consistency_residual   0.042 → 0.072 → 0.047 → 0.073   (flat ~0.05 across 15k; Term B did nothing)
loss_a (teacher match) 0.017 → 0.008 → 0.034 → 0.081   (converged ~7.5k, then DRIFTED UP)
div_ac_sim (lower=better) 0.46@1.5k → 0.63 → 0.52       (most-diverse ckpt was 1500; got worse after)
```

## Why Term B is inert — the mechanism

Both Term-B variants measure ~0.05 = the base field's own natural consistency
level. The teacher's CFG=4 field is already self-consistent at the student step
sizes, on-manifold (`renoise`) and off-manifold (`rollout`, the states the
4-step Euler rollout actually visits). So the student learns a field that is
smoothly-consistent-but-wrong: it faithfully reproduces the teacher's
instantaneous velocity, whose 4-step Euler rollout *is* the washed-out Arm-2.

The blur is not a velocity-inconsistency the consistency term can fix — it is
the discretization/distribution gap, which SCFM deliberately omits. The only thing
that sharpens 4-step here is a distribution-matching term (GAN/DMD) — i.e. DP-DMD.
The kill criterion the proposal pre-registered (§9.5) fired: *"if it stays flat in
rollout mode too, the washout is not a velocity-inconsistency the student can fix
and the line is closed."*

## What did move (and why it doesn't save the line)

- lr 1e-5 → 5e-5 sharpened renders (off the washout toward Arm-3) and kept
  diversity (`div_ac` 0.34→0.30) — but the gain was Term A converging harder,
  not straightening. At best this gets you toward the teacher's 4-step field, not
  past it.
- k_ratio 0.4 kept diversity better than 0.9 (the SCFM selling point survives)
  but at the cost of sharpness — the diversity↔sharpness trade a fidelity-only
  objective cannot escape without distribution-matching.
- The compositional arm floated in the proposal (§7, "SCFM Term-B as an aux on top
  of DP-DMD") is also dead: Term B contributes nothing on Anima, so bolting it
  onto DP-DMD adds cost and no signal.

## Verdict

CLOSE. Pure velocity-fidelity / self-consistency distillation cannot beat
distribution-matching on Anima at 4 steps, because (a) the teacher's few-step field
is washed out and (b) its consistency residual is at the base floor everywhere, so
the straightening term has no purchase. The sharp-4-step frontier is back on
DP-DMD (`anima_turbo_R`) and its teacher-gap levers (`div_weight↑`, f-distill,
softrank — [[project_turbo_teacher_gap_2026_06_29]]).

## What survives in-tree

- `bench/turbo/probe_consistency_residual.py` — Eq.-11 residual scan of the base
  field (the on-manifold straightness instrument). Result:
  `bench/turbo/results/20260629-1440-scfm-consistency-residual/`.
- `bench/turbo/probe_teacher_straightness.py` — teacher-transport straightness /
  non-crossing probe. Result: `bench/turbo/results/20260629-1503-scfm-teacher-straightness/`.

These stay because they are the reusable evidence; re-run only if a future arm
needs to re-measure base-field geometry. Do not re-add the SCFM loss.

## Pointers

- `_archive/proposals/turbo_scfm.md` — the full design + the §9 progress log
  (Phase-0 probes, Phase-1 build, lr/k_ratio/rollout runs), archived 2026-06-30.
- [[project_scfm_paper_verdict]] — the running verdict memory (updated to CLOSED).
- [[project_turbo_teacher_gap_2026_06_29]], [[project_turbo_R_plateau]] — the gap
  SCFM was meant to attack; still owned by the DP-DMD arm.
- [[project_turbo_consistency_aux_shelved]] — the earlier shelving of a related
  consistency aux (same inert-EMA failure family).
