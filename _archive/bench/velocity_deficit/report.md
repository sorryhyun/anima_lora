# Velocity Deficit (paper 1798) — Phase-0 on Anima

**Verdict: the paper's premise does not transfer. The deficit Anima *does* have is
the irreducible-variance kind that SSC/MAFM cannot fix, and the paper's SSC
schedule is anti-aligned with Anima's deficit shape. Do not adopt SSC or MAFM
without a sampling A/B that beats this geometric prediction.**

Run: `results/20260629-1423-base/` — base DiT, 96 real latent×caption pairs ×
3 fresh-noise draws = 288 samples/σ, unguided single forwards.

## What the paper claims (1798)
FM's MSE objective learns `v=E[v_target|x_t]`, so by Jensen `‖v‖<E‖v_target‖` — a
**magnitude** deficit, harmful at the noise end (integration lag), benign at the
data end. Fix by scaling velocity up: SSC `γ(t)=1.1→1.0` (inference) or MAFM
magnitude loss (training, the "REPA-style" arm).

## What we measure on Anima
`ratio=‖v‖/‖target‖`, `cos(v,target)`, `proj_ratio=⟨v,t̂⟩/‖target‖=ratio·cos`.

| | paper (SiT) | Anima (this run) |
|---|---|---|
| deficit depth | large | mild, mean ratio **0.957** |
| σ-shape | monotonic, worst at noise end | **U-shaped**, worst at *both* ends (σ=0.05 → 0.88, σ=0.99 → 0.94), flat ~0.975 mid |
| nature | magnitude (cos≈1) | **directional**: `cos ≈ ratio` at every σ (0.95738 mean cos) |

## The load-bearing geometry
`ratio ≈ cos` across the whole σ range ⇒ `‖v‖ ≈ ‖target‖·cosθ` ⇒ the error
`(target−v)` is **orthogonal to the prediction v**. That is the textbook
signature of an MSE-optimal conditional mean: the Jensen gap has gone into an
*orthogonal residual* (conditional variance), not a *radial shrink*. There is
almost no pure radial under-scaling left to inject.

Consequence: scaling `‖v‖` up by γ (SSC's entire mechanism) does **not** recover
the lost variance — it lengthens a vector that is already the correct conditional
mean, overshooting along the mean direction. `proj_ratio_after_ssc` (orange,
post-SSC transport) confirms it never cleanly reaches 1.0 and *worsens* the data
end.

## Schedule is also wrong-shaped
`ideal_gamma=1/proj_ratio` is U-shaped (needs most correction at the data end,
σ→0, where ideal γ≈1.28). The paper's SSC γ is monotonic 1.0→1.1 and goes to
**1.0 exactly at the data end** — i.e. it applies the least correction where
Anima's deficit is deepest, and that data-end regime is the one the paper itself
says is *benign* (denoising) and should be left alone. The two schedules cross
near σ=0.45 and have opposite slope below it.

## Caveats / what this does NOT prove
- Unguided single forwards. Under CFG the velocity is already amplified at the
  noise end; SSC's noise-end boost is
  even more likely redundant there, but a guided run wasn't measured.
- This is conditional-mean *geometry*, not a rendered-quality A/B. The geometry
  predicts SSC won't help; a sampling A/B (CMMD) is the only thing that could
  overturn it. Given the orthogonal-residual result + prior
  [[project_sigma_reshape_no_win]] / [[project_repa_global_anchor_refuted]], the
  prior on that A/B is low.

## To probe a LoRA (the "MAFM like REPA" question)
`uv run python bench/velocity_deficit/probe_deficit.py --adapter <ckpt> --label <name>`
then diff the curve vs base. If a trained LoRA already lands `ratio`/`cos` closer
to 1, MAFM has nothing to add. Not yet run.
