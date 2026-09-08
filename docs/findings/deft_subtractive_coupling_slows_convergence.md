# DEFT findings — why the decomposition converges *slower* than LoRA

TL;DR. DEFT reaches the same optima as LoRA (identical reachable set) but
takes 2–8× more steps to get there, in every regime — easy, ill-conditioned,
extreme-underdetermined, and even its own home-field "remove a W₀ subspace"
target. The cause is mechanical and unavoidable: DEFT's shared-`P` makes `ΔW`
quadratic in `P`, and the resulting gradient `∂L/∂P` is dominated by a
`−W₀GᵀP` coupling term that carries the base weight's large, target-irrelevant
singular values. That term ill-conditions `P`'s optimization, forcing a small
stable LR. The OrthoInit control (same SVD warm start, no coupling) ties LoRA,
proving the coupling — not the warm start — is the culprit.

Bench (archived): `_archive/bench/deft/` — `probe_convergence.py` (the race),
`analyze_gradient.py` (the mechanism), `plan.md` (the gate), `impl/deft.py` (the
module). **Shelved at Phase 0**; Phase 1 not run. Memory: `project_deft_convergence_shelved`.

---

## 1. What was measured

`probe_convergence.py` — 24 real Anima DiT Linears, matched rank `r=16`, matched
param count (`r·(out+in)` for all arms), all arms `ΔW=0` at init. Fit a rank-`k`
target `T` in W₀'s top subspace (realistic; real ΔW lands there — chimera bench
`cap_top≈0.755`) under heavy-tailed input geometry `C` (Anima's outlier channels):
`L = ‖(ΔW−T)·C‖²_F / ‖T·C‖²_F`. Adam, best-of-LR-grid per arm.

Steps-to-threshold (rel-error ≤ 0.2; lower = faster):

| regime | LoRA | DEFT | OrthoInit |
|---|---|---|---|
| w0-aligned over-rank (realistic) | 23 | 181 (~8×) | 24 |
| + asymmetric LR (R-LR ×8) | 43 | 253 | — |
| hard: σ3.5 ill-cond, rank-256 target | 9 | 65 (~7×) | 9 |
| ablate: remove W₀ top (DEFT home field) | 11 | 21 (~2×) | 9 |
| random-subspace control | (none reach) | worst AUC | mid |

DEFT's best LR is always the smallest in the grid — it is unstable at larger
steps. That is the symptom; §2 is the cause.

## 2. Why — the gradient is dominated by a W₀-coupling term

LoRA is bilinear: `ΔW = B·A`. DEFT is quadratic in `P`:

```
ΔW = P·R − P·Pᵀ·W₀ = P·(R − Pᵀ·W₀)
              ▲ P appears twice — once linearly (PR), once quadratically (−PPᵀW₀)
```

With `G := ∂L/∂ΔW = 2(ΔW−T)·diag(c²)`, the closed-form gradients are
(verified against autograd to rel-err `4e-7`):

```
∂L/∂R = Pᵀ G
∂L/∂P = G·(R − PᵀW₀)ᵀ   −   W₀·Gᵀ·P
        └──── fit term ────┘   └ coupling ┘
```

- The fit term `G·downᵀ` is exactly LoRA's `∂L/∂up` analogue — the useful,
  loss-reducing direction.
- The coupling term `−W₀GᵀP` is unique to DEFT and carries the *base weight*
  `W₀`, whose singular values are large and have nothing to do with the target.

`analyze_gradient.py` measures `‖coupling‖ / ‖fit‖` in `∂L/∂P` along a real run
(median over 24 layers):

| step | 0 | 1 | 5 | 20 | 100 |
|---|---|---|---|---|---|
| ‖coupling‖/‖fit‖ | ∞ | 3.7 | 6.0 | 4.7 | 0.55 |

- Step 0: the fit term is exactly zero. Clean init sets `R = PᵀW₀`, so
  `down = R − PᵀW₀ = 0` and the fit term vanishes — 100% of `P`'s initial
  gradient is the W₀-coupling nuisance. P is shoved in a direction set by `W₀`
  before it ever "sees" the target. (LoRA's opposite: `up=0` so `∂L/∂down=0`, and
  `up` grows cleanly along `G·downᵀ` — the right direction from step 1.)
- Steps 1–20 (the decisive window): coupling still dominates ~4–6×. And the
  magnitude is inflated: `‖∂L/∂P‖_DEFT / ‖∂L/∂up‖_LoRA ≈ 4.8×` at init — P's
  update is ~5× larger than LoRA's, but mostly nuisance.
- Step 100: drops below 1 — once near the optimum `G` shrinks, so the coupling
  shrinks too and the fit term catches up. By then the slow early phase has
  already cost the step budget.

### Why that means slow

The coupling term's sensitivity to `P` scales with `W₀W₀ᵀ`, whose condition
number is huge (a wide pretrained spectrum). So `P`'s subproblem is badly
conditioned: a few directions have enormous curvature (the W₀-coupling ones) while
the loss-reducing directions have ordinary curvature. Any LR large enough to make
real progress on the latter makes the former oscillate/diverge → Adam is pinned to
the smallest stable LR (exactly what the grid shows) → slow descent everywhere
else. Quadratic-in-`P` also means the landscape is genuinely more non-convex than
LoRA's bilinear one.

## 3. The OrthoInit control isolates the cause

OrthoInit shares DEFT's SVD warm start (`P,Q` seeded from W₀'s top-r) but its
`ΔW = P·diag(λ)·Q` is bilinear — no W₀ in any gradient. It ties LoRA in
every regime (24/9/9 steps). So:

- the SVD warm start is free (neither helps nor hurts vs LoRA here), and
- the coupling is the entire penalty.

There is therefore nothing to salvage by tuning DEFT: the slowdown is the defining
feature (the live `−PPᵀW₀` coupling), not an init or LR detail. Removing the
coupling *is* OrthoInit, which we already ship.

## 4. Even the home-field target loses

The `ablate` target (`ΔW = −U_r Σ_r V_rᵀ`, i.e. "subtract W₀'s top component") is
exactly what `−PPᵀW₀` computes with `P=U_r, R→0`. DEFT still loses (21 vs 11):
`P=U_r` is already optimal, but DEFT must drive `R` from its init `PᵀW₀` down to
`0`, and during that transient `G≠0` keeps the coupling `−W₀GᵀP` pushing `P` off
`U_r`, which then needs re-correcting. LoRA just grows `up` from `0` — no fight.

## 5. Conclusion

- Convergence merit: falsified. No regime converges faster; the turbo
  leaf-fit motivation (faster student) dies here.
- Reachability merit: none possible (set ≡ LoRA).
- Inference: DEFT is mergeable (foldable like LoRA — `to_lora_weights()`),
  its one edge over ReFT, but irrelevant if it's never worth training.
- Recommendation: do not promote. If you want a W₀-aligned warm start, it's
  already OrthoInit (`use_ortho_init=true`). Subtractive weight decompositions
  have now failed Phase-0 twice on Anima (also HF-residual). For "additional
  stack" promotion, the only live candidate is the unrun ReFT bench
  (`bench/reft/plan.md`, a different rep-space slot — not a substitute).
