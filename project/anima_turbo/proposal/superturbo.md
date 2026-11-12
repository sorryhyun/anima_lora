# Superturbo: a 2-step student with the DM gradient through the rollout

Written 2026-11-10. Built as `dpdmd.rollout_grad` and run 2026-11-11; the 2-step line closed
2026-11-12 at the level of `1111_1115` step 250 (results: `project/anima_turbo/plan.md` at commit
`91d476e0`). The current loop is described in `docs/anima/methods/turbo.md`; the anchor-free
collapse that led here is in `project/anima_turbo/reports/mfd.md`.

A 2-step LoRA is a target of its own, not a by-product of the 4-step one. The loop as it
stands trains it badly for a structural reason, and the fix is specific to N = 2.

## The problem at N = 2

The static 2-step grid at shift 3 is σ 1 → 0.75 → 0.

- Step 0 is trained by the anchor only (`div_loss` against the teacher's mean velocity over
  [t_k, 1]). Step 1 takes every DM update and has to make the whole image in one jump from
  σ 0.75.
- The two steps share no gradient. `steps.student_rollout` detaches `x` after step 0
  unless `rollout_grad` keeps the graph (`if not keep: x = x.detach()` in `student_rollout`), so step 0 never learns where step 1 needs `x₁` to be.
  `detach_after_first = false` only moves the anchor loss into the final backward.
- `dm_first_step` does not help here. It asks step 0's own one-step x0 for a finished image,
  but step 1 sees that x0 at 25 % under 75 % noise, so only layout survives, and the anchor
  already supplies layout. At N = 2 it also halves step 1's DM updates. In the 2026-11-10
  renders, step 0 with DM (`anima_turbo_m14_mfd_k3anchor_dmfirst_nogan_nocdm_250_tafake`) and
  step 0 that DM never touched (`anima_turbo_m14_dpdmd_svd_500_tafake`) draw about the same
  blurred figure at 1 step. The two runs differ in more than this.

## Proposal

Keep the graph from step 0 into step 1 and put the DM (or MFD) signal on the final `x_θ`
only:

```
x₁  = ε − (1 − σ₁)·v₀(ε)             # grad kept
x_θ = x₁ − σ₁·v₁(x₁)
loss = (grad_signal · x_θ).mean() + div_weight · ‖v₀ − v_target‖²
```

- Step 0 now gets the gradient of the image the user actually samples, through step 1. It
  can move `x₁` off the teacher's σ 0.75 point to wherever step 1 finishes best, which is the
  only way the two steps can split the work.
- The anchor stays on with `k_anchor = 6` (teacher σ 0.75 on the 12-step grid, the 2-step
  first interval exactly). It is now the only thing holding composition diversity against
  the DM gradient on the same step.
- Training runs the inference path: static grid first. `dynamic_schedule` is the follow-up
  that tests how far the pair co-adapts to one grid.

Code: one knob, e.g. `dpdmd.rollout_grad` (default false), refused unless
`student_steps = 2`. Under it `student_rollout` skips the detach and the fresh leaf, and
the anchor loss rides the caller's backward (the `div_pending` path that `g = 0` already
uses). The fake update, `fake_transition` (source σ₁) and the metrics read the same `x_θ` as
now.

## Cost

- Two grad-bearing student forwards in one graph. On 16 GB this needs `--grad_ckpt`; the
  loop already sets the student view before the backward under it. Peak VRAM and s/step are
  to be read off the smoke, not estimated here.
- Teacher forwards per step do not change.

## Risks

- **Diversity.** DP-DMD stops the gradient after step 0 because the DM signal reaching it
  raised preference and lowered diversity (their Fig 5). This proposal removes that stop
  and relies on the anchor weight.
- **Co-adaptation.** Step 0 learns to serve this step 1 on this grid. A LoRA used at other
  step counts or samplers may degrade faster than the current one.
- **Gradient noise.** The signal reaches step 0 through step 1's input Jacobian.

## Arms

Same teacher (merged14, int8 GPTQ base, fp32 stream), init (`weight_svd`, r64 / α180,
adaln r16), seed, MFD dtau 0.1, GAN and CDM off, `fake_transition` on, static grid,
`student_steps = 2`, `k_anchor = 6`, 250 steps.

| Arm | Change |
|---|---|
| A | none (step 0 anchor only, DM on step 1) |
| B | `rollout_grad` on, `--grad_ckpt` |
| C (optional) | A with step 1 starting from σ 0.5 and the anchor matched to it (teacher step 9 of 12 at shift 3, 18 teacher forwards per step). Moves work to step 0 without touching the gradient path. Needs a student-grid knob apart from `flow_shift`, which the anchor grid shares |

Read, in this order:

1. Renders at 2 steps, cfg 1, the ANIMA-sign and nozomi prompts, seeds as in today's sheets,
   against the teacher at 28 steps cfg 4: composition spread across seeds first, then
   sharpness and stray content.
2. The 1-step render: where B moved `x₁` relative to A.
3. `val/div_xpred_ac_sim` and `val/div_ac_sim` at 50-step cadence.

B passes if its 2-step renders are at least as good as A's and its seeds are no more alike
than A's. If B is sharper but its seeds converge, the next arm is a higher `div_weight`, not
a longer run.
