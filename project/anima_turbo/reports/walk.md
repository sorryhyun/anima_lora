# The batch-1 walk: 2-step measurements and the 4-step ΔW overlap (2026-11-11 / 12)

Moved out of `plan.md` on 2026-11-12. These are the measurements the 4-step recipe and the A/B
protocol were fixed on, taken before arm 0. The arm 0 numbers (control / twin walk, the 4-step norm
profile, bursts) are in `arm0.md`. The closed 2-step plan with its run-by-run results is
`91d476e0:scripts/distill_turbo/plan.md`.

## What the 2-step line measured

On the 2-step runs `1111_1005`, `1111_1115`, `1111_1740`, `1111_2056` (all batch 1):

- **The per-step gradient is mostly noise.** Gradient probe (`bench/turbo/probe_grad_noise.py`,
  `1111_1115` step 375): of the direction energy of a student step, 77 % is between DM query draws
  (σ_a, ε_dm) of one sample, 20 % between samples, 3 % is the mean that accumulates.
- **The weights random-walk.** Under a constant LR, |ΔW| grows as √t and consecutive 125-step
  moves are orthogonal (cos ≤ 0.02) from the first 125 steps on.
- **LR scales the walk, not its onset.** 5e-5 → 2e-5 shrank every move by 2.56× and moved
  nothing else (Adam scales drift and noise together).
- **Lowering the noise does not raise the level.** `dm_draws 4`, `grad_accum 4` and exact
  checkpoint averages all landed on the level of `1111_1115` at step 250.
- **Removing the walk after the fact undoes the late drift.** Exact ΔW averages of the
  250 / 375 / 500 checkpoints, and the EMA of `1111_2056` (window 0.2), took the late
  benches / vehicles / brown hair back out. EMA window 0.1 (`1111_1115`) did not.
- Reading: the objective sets the level; the batch-1 walk only subtracts from it.

## The 4-step loop is in the same regime

`bench/turbo/dw_overlap.py`, cosine between the ΔW of two runs (weight_svd init, so ΔW starts
at 0). For two runs of one objective from one init it reads as the share of ΔW energy they
have in common.

| Pair | Steps | cos |
|---|---|---|
| `anima_turbo_m14_dpdmd_svd_500` vs `..._tafake` (4-step; same seed, init, data order; differ in `fake_transition` only, which left the losses identical) | 500 | **0.14** |
| `1111_1115` vs `1111_1740` (2-step; differ in `dm_draws`) | 125 / 250 / 375 / 500 | 0.19 / 0.17 / 0.16 / 0.17 |
| any other pair (different LR, or 2-step vs 4-step) | 500 | 0.09–0.11 |

- About 85 % of a 4-step student's ΔW is specific to its run, as in the 2-step line.
- The shared part is not concentrated: per module kind (adaln, self-attn, cross-attn, mlp) the
  4-step pair's cos is 0.11–0.19; per block group 0.10–0.21, highest in blocks 21–27.
- Short runs are no more deterministic: the 2-step pair already shares only 0.19 at step 125.
- Consequence for past reads: the 2026-11-10 `fake_transition` A/B (tafake sharper, plain
  drifting to brown hair and one centred sign; `../vatdm.md`) compared two such draws. Its render
  differences may be mostly run noise.

The 4-step |ΔW| trajectory came with arm 0 (`arm0.md`). A 4-step gradient probe does not exist: the
probe refuses anything but the 2-step superturbo config.

## Gradient norm on the 2-step runs: the clip is a normaliser

`grad_clip 1.0` acts on almost every step. Pre-clip student norms (TB `train/grad_norm_student`,
5-step interval means):

| Run | p5 | median | p95 | intervals above 1.0 |
|---|---|---|---|---|
| `1111_1005` | 1.74 | 3.20 | 10.1 | 100 % |
| `1111_1115` | 2.52 | 4.01 | 8.8 | 100 % |
| `1111_1740` | 2.42 | 3.74 | 7.4 | 100 % |
| `1111_2056` | 3.44 | 5.00 | 8.4 | 100 % |

The fake is the same (median 2.2–2.9, every interval above 1.0).

- Adam ignores a constant gradient scale, so a higher clip does not make steps larger; LR does.
  What the clip sets is how steps weigh against each other. At 1.0 every step enters at unit
  norm; without it a step weighs by its norm.
- On 2 steps the large norms were the noise. The DM norm grew with σ_a (median 1.24 below 0.25,
  4.93 above 0.75, 22–30 near 0.96–0.99), nearly all through the rollout path, and norm-weighting
  took the DM mean share from 4.3 % (unit-weighted) to 1.6 %.
- The 4-step profile (no rollout path) was read on arm 0's `grad_steps.jsonl`: flat below
  σ_a 0.75, pooled top / bottom quarter 1.43 (`arm0.md`). The rule fixed beforehand (≤ 1.5 → adaptive
  clip alone, above → adaptive clip plus σ_a normalisation) gives adaptive clip alone.
