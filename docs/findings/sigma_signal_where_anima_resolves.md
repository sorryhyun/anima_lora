# Where Anima's denoising signal actually lives: a σ ≈ 0.75 → 0.45 → 0 staircase

Two unrelated probes — a σ-sweep `x0_pred` reconstruction probe (since removed) and
`bench/fera_artist/probe_fei_trajectory.py` (low-frequency energy fraction along the
live CFG=4 sampler) — agree on the same σ structure for `anima-base-v1.0`:

- σ ≈ 0.75: `x0_pred` is already a recognizable picture (subject, layout,
  rough color blocks correct).
- **σ ≈ 0.45**: `x0_pred` is essentially the final image — the base reconstructs
  with normalized latent-MSE < 0.2 from this σ down.
- σ ∈ [0.45, 0]: every remaining change is refinement (fine details,
  high-frequency texture).

Per the FEI trajectory, the latent itself — not the model's prediction —
gains its low-frequency content over the same window: `e_low` rises from
0.30 at t=0.40 to 0.57 at t=0.10. So the "picture being added" lives almost
entirely in the σ < 0.45 tail. Above σ ≈ 0.55 the base already knows the
answer; the sampler is just stepping toward it.

![x0_pred across σ on a single sample, 3×4 grid (top-left = true x0,
then σ=0.05 to σ=0.95)](assets/sigma_signal_x0_vs_sigma_grid.png)

## Evidence 1 — `x0_pred = x_σ − σ·v` reconstruction

The σ-sweep probe (since removed) noised 16 real cached
latents with `(1−σ)·x0 + σ·ε` at σ ∈ {0.05, 0.15, …, 0.95}, ran a single
bare-DiT forward, and decoded `x0_pred`. Per-σ latent-MSE averaged across
samples + seeds (n=16 stems × 3 seeds, `20260528-1544-grid3-1024`):

| σ | mean lat-MSE | normalized |
|---|---|---|
| 0.05 | 0.0005 | 0.005 |
| 0.15 | 0.0019 | 0.019 |
| 0.25 | 0.0035 | 0.035 |
| 0.35 | 0.0058 | 0.058 |
| 0.45 | 0.0092 | 0.092 |
| 0.55 | 0.0142 | 0.142 |
| 0.65 | 0.0214 | 0.214 |
| 0.75 | 0.0331 | 0.331 |
| 0.85 | 0.0537 | 0.537 |
| 0.95 | 0.1003 | 1.000 |

The 20% normalized-error crossover sits at σ ≈ 0.55. Below that, `x0_pred`
matches the source well enough that an adapter has very little room to push
the prediction anywhere new. The visual grid above makes the qualitative
breakpoint even clearer: σ=0.45 is indistinguishable from σ=0.05 to the
eye; σ=0.75 is the same composition with smoothed-out details; σ=0.95 is
mostly a blur.

## Evidence 2 — FEI(z_t) along the live 28-step CFG=4 trajectory

`bench/fera_artist/probe_fei_trajectory.py` captures the low/high-band
energy of `z_t` itself (`compute_fei_2band` at div=4) at every step of a
real Anima sampler rollout. From `20260528-1523-trajectory` (10 prompts ×
6 artists, native buckets, 28 steps, flow_shift=3, CFG=4):

| t (sampler σ) | step | `e_low(z_t)` |
|---|---|---|
| 1.00 | 0 | 0.00016 |
| 0.80 | 12 | 0.01754 |
| 0.45 | 22 | 0.23842 |
| 0.40 | 23 | 0.30321 |
| 0.10 | 27 | 0.57371 |

`e_low` is the fraction of latent energy in the low-frequency band of a
DoG split at `σ_low = min(H_lat, W_lat) / 4`. It triples between t=0.40
and t=0.10 — i.e. the *latent's structure* is mostly being written into
that ~0.45→0.10 window, the same window where the visual `x0_pred` is
already done. Cross-checks against the training-time mixture probe
(`probe_fei_artist.py`, smoke `20260528-1456-smoke`): at matched t the two
probes agree (t=0.8 → 0.016 vs 0.018; t=0.4 → 0.291 vs 0.303), so this
isn't a sampling-vs-training artifact.

![per-artist FEI(e_low) along the 28-step trajectory; mean ±1σ band on
the left, std across artists on the right](assets/sigma_signal_fei_trajectory.png)

## Why both can be true at once

There's no contradiction between "the base reconstructs by σ=0.55" and
"low-freq energy keeps growing through σ<0.45." The base's `x0_pred`
subtracts `σ·v` from a still-noisy `z_σ` — the model knows the answer
long before the latent it's looking at contains the answer. The sampler
then spends the σ<0.45 tail copying that answer into the latent itself.

## Implications

1. The default sigmoid schedule mis-allocates capacity.
   `library/runtime/noise.py`'s `sigmoid` puts its bell at σ=0.5 — i.e.
   directly on the boundary where the base is already done. The probe
   reports 57.8% of sigmoid mass falls below σ=0.55 (the "no signal"
   region), versus 38.3% for a `μ=+0.5` logit-normal that biases toward
   the σ>0.55 region where the base is uncertain. NB this is a
   *training-time* capacity-allocation hypothesis and is still unconfirmed
   — the related *inference-time* idea (reshaping the sampler's σ schedule
   to densify one end at fixed NFE) was later refuted; see
   [[project_sigma_reshape_no_win]]. Different
   axis, but don't read this as an endorsed lever.

2. Adapter training capacity has two regimes to choose between:
   - σ > 0.55, where the base is genuinely uncertain about the answer
     itself (global structure / composition / pose) — this is where a
     LoRA can change what the model predicts.
   - σ < 0.45, where the answer is locked but the high-frequency
     texture is still being filled in. A LoRA targeting style/detail
     (a quality LoRA, a sharpener) needs mass here, not at σ=0.5.

   These are different jobs. The shipped default doesn't strongly
   commit to either.

3. The FEI router's training-vs-inference signal is consistent.
   `probe_fei_artist.py`'s training-mixture FEI and
   `probe_fei_trajectory.py`'s live-sampler FEI line up at matched t.
   The Hydra-content / freq-routed adapters can trust that the FEI
   features they see at training time are the FEI features they will
   see at inference — no separate trajectory calibration is needed for
   the router input.

## Caveats

- This is a "where is the base uncertain" diagnostic, not a quality
  predictor. Lower FM-MSE has historically not tracked CMMD on Anima
  (`project_fm_val_loss_uninformative`). A real schedule sweep needs
  CMMD-scored training to settle "optimal." This probe exists to inform
  the arms of that sweep.

- The reconstruction view is content-only. Style/identity that show
  up only as fine-grained low-σ texture won't change the normalized
  latent-MSE crossover much — that signal lives in the σ<0.45 tail by
  definition and is exactly what the `x0_pred` already-matches view
  can't see.

- The trajectory probe's t-axis runs `1.0 → 0.10`, not `1.0 → 0`.
  Anima's 28-step Euler with `flow_shift=3` stops at t≈0.10. The
  σ<0.10 sliver of refinement isn't covered here.

## Reproduce

The visual `x0_pred` grid + per-σ latent-MSE probe is no longer in-tree (it ran on
16 stems × 3 seeds at 1024). The FEI trajectory half survives:

```bash
# FEI along the live sampler trajectory (10 stems by default)
uv run python bench/fera_artist/probe_fei_trajectory.py \
  --k_per_artist 2 --max_artists 10 --infer_steps 28 --guidance_scale 4.0 \
  --label trajectory
```

Date: 2026-05-28. Surviving runs:
`bench/fera_artist/results/20260528-1523-trajectory/`,
`bench/fera_artist/results/20260528-1456-smoke/`.
