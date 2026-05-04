# Transfer-hypothesis check
Source: `bench/dcw/results/20260504-1010-transfer-hyp/gaps_per_sample.npz`
n_steps=24, n_seeds_assumed=2

## Per-band early-vs-late correlation
Definition: r = corrcoef(gaps[:, :N/2].mean(1), gaps[:, N/2:].mean(1))

| Band | n_traj | r (all) | 95% CI | r (seed-avg) | 95% CI | Spearman ρ | Decision (seed-avg) |
|---|---|---|---|---|---|---|---|
| **LL** | 96 | +0.989 | [+0.98, +0.99] | +0.990 | [+0.98, +0.99] | +0.980 | v0a (online controller) |
| **LH** | 96 | +0.575 | [+0.42, +0.70] | +0.635 | [+0.43, +0.78] | +0.571 | v0 with early-step features |
| **HL** | 96 | +0.543 | [+0.38, +0.67] | +0.529 | [+0.29, +0.71] | +0.520 | v0 with early-step features |
| **HH** | 96 | +0.514 | [+0.35, +0.65] | +0.572 | [+0.34, +0.74] | +0.532 | v0 with early-step features |

## Per-seed splits (seed-coupling diagnostic)
If single-seed r ≈ all-trajectories r and seed-avg r is similar, seed coupling is weak and the all-trajectory r is trustworthy. If seed-avg r is much higher than per-seed r, per-prompt signal is real but seeds add noise, not coupling.

### LL
- seed 0: r = +0.991 (n=48, 95% CI [+0.98, +1.00])
- seed 1: r = +0.987 (n=48, 95% CI [+0.98, +0.99])

### LH
- seed 0: r = +0.648 (n=48, 95% CI [+0.45, +0.79])
- seed 1: r = +0.528 (n=48, 95% CI [+0.29, +0.71])

### HL
- seed 0: r = +0.564 (n=48, 95% CI [+0.33, +0.73])
- seed 1: r = +0.531 (n=48, 95% CI [+0.29, +0.71])

### HH
- seed 0: r = +0.482 (n=48, 95% CI [+0.23, +0.67])
- seed 1: r = +0.561 (n=48, 95% CI [+0.33, +0.73])
