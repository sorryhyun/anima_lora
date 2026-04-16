# Timestep Rank Masking (T-LoRA)

Timestep-dependent rank masking for LoRA training. The effective rank of the
low-rank adapter varies with the denoising timestep — low rank at high noise,
full rank at low noise — so early denoising learns coarse semantics and late
denoising learns fine detail.

## Quick start

```bash
make tlora      # OrthoLoRA + timestep masking (rank 64)
make tdora      # DoRA + timestep masking (rank 16)
```

Or add `use_timestep_mask = true` to any method config.

## How it works

### Rank schedule

At each training step the batch-averaged timestep `t` (normalized to [0, 1],
where 0 = pure noise and 1 = clean image) determines the active rank:

```
r(t) = floor((1 - t)^alpha * (R_max - R_min)) + R_min
```

- `R_max` = `network_dim` (e.g. 64)
- `R_min` = `min_rank` (e.g. 1)
- `alpha` = `alpha_rank_scale` (power-law exponent)

| `t` (noise level) | `alpha=1.0, dim=64` | Interpretation |
|--------------------|---------------------|----------------|
| 0.0 (pure noise) | r = 64 (full) | High-noise steps get full capacity |
| 0.25 | r = 49 | |
| 0.5 | r = 33 | Half rank |
| 0.75 | r = 17 | |
| 1.0 (clean) | r = 1 (minimum) | Low-noise refinement uses minimal rank |

Higher `alpha` concentrates more rank toward high-noise steps (steeper decay).
Lower `alpha` flattens the schedule.

### Mask application

A binary mask `[1, 1, ..., 1, 0, 0, ..., 0]` of shape `(1, R_max)` is
generated and applied element-wise after the `lora_down` projection:

```
lx = lora_down(x)        # (B, L, R_max)
lx = lx * mask           # zero out dimensions r..R_max
lx = lora_up(lx)         # project back to full dim
```

A single GPU-resident mask tensor is shared by reference across all ~200 LoRA
modules to avoid per-module CPU-to-GPU transfers.

### Training only

The mask is only applied when `self.training` is True. At inference, the full
rank is always used regardless of timestep.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_timestep_mask` | false | Enable timestep rank masking |
| `min_rank` | 1 | Minimum active rank (floor) |
| `alpha_rank_scale` | 1.0 | Power-law exponent (1.0 = linear, >1 = steeper, <1 = flatter) |
| `network_dim` | - | Maximum rank (R_max), set by the method config |

## Compatibility

Timestep masking composes with all adapter module types. The mask is applied
at the bottleneck (after down-projection), so it is orthogonal to the
module's outer parameterization:

| Module | Where mask is applied |
|--------|----------------------|
| **LoRA** | After `lora_down`, before dropout and `lora_up` |
| **DoRA** | Same as LoRA; magnitude `dora_scale` is separate |
| **OrthoLoRA** | After `q_layer`, multiplied with `lambda_layer` |
| **OrthoLoRA (Cayley)** | After `Q_eff` projection, multiplied with `lambda_layer` |
| **HydraLoRA** | After shared `lora_down`; per-expert `lora_up` heads unaffected |
| **ReFT** | Applied to both `rotated` and `source` tensors (min rank = 1) |

HydraLoRA + T-LoRA is the configured default for `make hydralora`.

## ReFT variant

ReFT modules receive a separate mask with their own dimension (`reft_dim`)
and a minimum of 1 active dimension:

```
r_reft(t) = floor((1 - t)^alpha * (reft_dim - 1)) + 1
```

## Configs

`configs/methods/tlora.toml` — OrthoLoRA (Cayley) + timestep masking, rank 64:
```toml
use_ortho_exp = true
use_timestep_mask = true
min_rank = 1
alpha_rank_scale = 1.0
network_dim = 64
```

`configs/methods/doratimestep.toml` — DoRA + timestep masking, rank 16:
```toml
use_dora = true
use_timestep_mask = true
min_rank = 1
alpha_rank_scale = 1.0
network_dim = 16
```

## Implementation

| File | Role |
|------|------|
| `networks/lora_anima.py` | `set_timestep_mask()` — computes rank, writes shared mask |
| `networks/lora_anima.py` | `set_reft_timestep_mask()` — same for ReFT modules |
| `networks/lora_anima.py` | `clear_timestep_mask()` — removes mask (for inference) |
| `networks/lora_modules.py` | Per-module mask application in each forward method |
| `train.py` | Calls `set_timestep_mask()` each step after noise sampling |
