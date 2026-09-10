# Timestep Rank Masking (T-LoRA)

Timestep-dependent rank masking for LoRA training. Effective rank varies with the denoising step — low at high noise, full at low noise.

> For the structural walkthrough (rank schedule math, mask application inside the LoRA bottleneck, training-only semantics, shared GPU-resident tensor), see `docs/structure/timestep-mask.md`. This doc is the usage / ops reference.

## Quick start

T-LoRA variants live in `configs/gui-methods/` (one file per variant, no toggle blocks):

```bash
make lora-gui GUI_PRESETS=tlora              # OrthoLoRA + timestep masking (rank 64)
```

Or toggle inside `configs/methods/lora.toml` by uncommenting the T-LoRA block and running `make lora`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_timestep_mask` | false | Enable timestep rank masking |
| `min_rank` | 1 | Minimum active rank (floor at the pure-noise end, σ=1) |
| `alpha_rank_scale` | 1.0 | Power-law exponent (1.0 = linear, >1 = steeper, <1 = flatter) |
| `network_dim` | — | Maximum rank (R_max), set by the method config |

## Compatibility

Timestep masking composes with every adapter module type. The mask is applied at the bottleneck (after down-projection), so it is orthogonal to the module's outer parameterization:

| Module | Where mask is applied |
|--------|----------------------|
| LoRA | After `lora_down`, before dropout and `lora_up` |
| OrthoLoRA (Cayley) | After `Q_eff` projection, multiplied with `lambda_layer` |
| HydraLoRA | After shared `lora_down`; per-expert `lora_up` heads unaffected |

The default block in `configs/methods/lora.toml` stacks LoRA + OrthoLoRA + T-LoRA together.

## Configs

`configs/methods/lora.toml` (T-LoRA toggle block) — OrthoLoRA (Cayley) + timestep masking, rank 64:

```toml
use_ortho = true
use_timestep_mask = true
min_rank = 1
alpha_rank_scale = 1.0
network_dim = 64
```

## Findings (bench-backed)

Does the mask earn its complexity? Benched twice; the answer depends on where
`min_rank` sits relative to the rank the model would use anyway. The bench
line is CLOSED — analytic objective met, scripts + results archived to
`_archive/bench/timestep_mask/` (2026-07-12; `learned_rank.py` there is a
generic ΔW effective-rank tool, reusable on any LoRA-family checkpoint).

Inert when the floor ≥ the natural learned rank (2026-06-07,
`_archive/bench/timestep_mask/results/20260607-*`): at `network_dim=48, min_rank=16`,
mask on/off/σ-uniform all land at participation ratio ≈16 of 48. The binding
constraint is the data, not the mask — the scheduled band above the floor is
idle capacity. Corollary: dim=48 is ~3× over-provisioned in that regime.

Active when the floor bites below it (2026-07-04,
`results/20260704-1741-learned-rank-dim16-minrank1`): at `network_dim=16,
min_rank=1` (plain LoRA + REPA, single-artist subset), the mask *raises*
effective rank — PR(median) 10.9 → 12.25, energy-weighted PR 5.2 → 9.8. The
plain run concentrates high-energy modules into a few dominant directions; the
mask flattens the spectrum (top columns see gradient only on low-σ steps). It
acts as a spectral regularizer, not a budget cut. σ-sampler shape (sigmoid vs
uniform) is irrelevant in both benches.

Memorization: mitigation, not a fix (2026-07-04,
`bench/memorization/results/20260704-18*-sincos_half_*`): matched
`sample_ratio=0.5` arms, `loss_gap.py` member-vs-same-artist-holdout gate.
Both arms flag member-specific overfit; the mask trims the AUC 0.82 → 0.77,
with the reduction concentrated at σ=0.5 (0.86 → 0.71) — exactly where
sigmoid-σ sampling mass peaks and the schedule halves the rank — while σ=0.9
ticks up slightly (0.62 → 0.68). Single seed, so the headline ΔAUC is within
noise; the σ=0.5 delta is the trustworthy part.

**Verdict**: at `min_rank=1` the mask is a mild net positive (spectral
flattening + the σ=0.5 memorization trim, slight observed improvement) — kept
opt-in, no default change. Don't tune `alpha_rank_scale` (σ-shape settled
irrelevant both times), and don't sweep `min_rank` for an "optimal" value:
with σ-shape inert the knob is just generic regularizer strength, and
spectrum metrics can't arbitrate quality on their own.

## Implementation

| File | Role |
|------|------|
| `networks/lora_anima/network.py` | `set_timestep_mask()` — computes rank, writes shared mask |
| `networks/lora_anima/network.py` | `clear_timestep_mask()` — removes mask (for inference) |
| `networks/lora_modules/lora.py` | Per-module mask application in each forward method |
| `train.py` | Calls `set_timestep_mask()` each step after noise sampling |

## Programmatic example

`examples/07_stack_ortho_init_tlora.py` builds a fresh OrthoInit + T-LoRA stack
from Python (no config file) and drives the mask via the one per-step hook
`apply_router_conditioning`, printing the live effective rank each step. It is the
runnable counterpart to the "T-LoRA is not a class, it's a buffer" note above —
see `examples/README.md` (row 07).
