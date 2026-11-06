# Timestep Rank Masking (T-LoRA)

Timestep-dependent rank masking for LoRA training. Effective rank varies with the denoising step — low at high noise, full at low noise.

> For the structural walkthrough (rank schedule math, mask application inside the LoRA bottleneck, training-only semantics, shared GPU-resident tensor), see `docs/anima/structure/timestep-mask.md`. This doc is the usage / ops reference.

## Quick start

```bash
make lora ARGS="--network_args use_timestep_mask=true"   # off in configs/methods/lora.toml
make lora-gui GUI_PRESETS=tlora              # configs/gui-methods/tlora.toml (rank 32, weight_svd down-init)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_timestep_mask` | false (also in `lora.toml` since 2026-11-05; `tlora` GUI variant: true) | Enable timestep rank masking |
| `min_rank` | 1 | Minimum active rank (floor at the pure-noise end, σ=1) |
| `alpha_rank_scale` | 1.0 | Power-law exponent (1.0 = linear, >1 = steeper, <1 = flatter) |
| `network_dim` | — | Maximum rank (R_max), set by the method config |

## Inference

The mask is not stamped in the LoRA, and inference runs every rank at every σ: the
static merge cannot vary with σ, and the hook forward skips the mask in eval. The
T-LoRA paper (§3.1, arXiv 2507.05964) and its code apply the mask at inference too.
`inference.py --lora_timestep_mask [--lora_mask_min_rank 1 --lora_mask_alpha 1.0]`
does that on the P-GRAFT hooks (standard loop, not int8). On sweetonedollar, masked
training + masked inference tied unmasked training + full-rank inference 11–13 in a
blind A/B (s31); masked training + full-rank inference, what `lora.toml` shipped until
2026-11-05, was not tested against either (`docs/anima/findings/lora_recipe.md` § T-LoRA mask).

## Compatibility

Timestep masking composes with every adapter module type. The mask is applied at the bottleneck (after down-projection), so it is orthogonal to the module's outer parameterization:

| Module | Where mask is applied |
|--------|----------------------|
| LoRA | After `lora_down`, before dropout and `lora_up` |
| HydraLoRA | After shared `lora_down`; per-expert `lora_up` heads unaffected |

## Findings

The effect depends on where `min_rank` sits relative to the rank the model
would learn anyway. Scripts and results: `_archive/bench/timestep_mask/`
(`learned_rank.py` there is a generic ΔW effective-rank tool for any
LoRA-family checkpoint).

- **Floor ≥ natural learned rank — inert.** At `network_dim=48, min_rank=16`,
  mask on/off/σ-uniform all land at participation ratio ≈16 of 48
  (`results/20260607-*`). The data is the binding constraint; the band above
  the floor is idle capacity.
- **Floor below it — spectral regularizer.** At `network_dim=16, min_rank=1`
  (plain LoRA + REPA, single-artist subset), the mask raises effective rank —
  PR(median) 10.9 → 12.25, energy-weighted PR 5.2 → 9.8
  (`results/20260704-1741-learned-rank-dim16-minrank1`). Without the mask,
  high-energy modules concentrate into a few dominant directions; with it the
  top columns see gradient only on low-σ steps and the spectrum flattens.
- **Memorization — reduced, not removed.** Matched `sample_ratio=0.5` arms,
  `loss_gap.py` member-vs-same-artist-holdout gate
  (`bench/memorization/results/20260704-18*-sincos_half_*`): both arms show
  member-specific overfit; the mask trims AUC 0.82 → 0.77 (single seed, within
  noise), concentrated at σ=0.5 (0.86 → 0.71) where sigmoid-σ sampling peaks and
  the schedule halves the rank; σ=0.9 moves 0.62 → 0.68.
- σ-shape (sigmoid vs uniform sampler, `alpha_rank_scale`) had no measurable
  effect in either bench.

## Implementation

| File | Role |
|------|------|
| `networks/lora_anima/network.py` | `set_timestep_mask()` — computes rank, writes shared mask |
| `networks/lora_anima/network.py` | `clear_timestep_mask()` — removes mask (for inference) |
| `networks/lora_modules/lora.py` | Per-module mask application in each forward method |
| `train.py` | Calls `set_timestep_mask()` each step after noise sampling |
