# Timestep Rank Masking (T-LoRA)

Timestep-dependent rank masking for LoRA training. Effective rank varies with the denoising step — low at high noise, full at low noise.

> **For the structural walkthrough** (rank schedule math, mask application inside the LoRA bottleneck, training-only semantics, shared GPU-resident tensor), see **`docs/structure/timestep-mask.md`**. This doc is the usage / ops reference.

## Quick start

T-LoRA variants live in `configs/gui-methods/` (one file per variant, no toggle blocks):

```bash
make lora-gui GUI_PRESETS=tlora              # OrthoLoRA + timestep masking (rank 64)
make lora-gui GUI_PRESETS=tlora_ortho_reft   # OrthoLoRA + T-LoRA + ReFT stack (default)
```

Or toggle inside `configs/methods/lora.toml` by uncommenting the T-LoRA block and running `make lora`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_timestep_mask` | false | Enable timestep rank masking |
| `min_rank` | 1 | Minimum active rank (floor at clean end) |
| `alpha_rank_scale` | 1.0 | Power-law exponent (1.0 = linear, >1 = steeper, <1 = flatter) |
| `network_dim` | — | Maximum rank (R_max), set by the method config |

## Compatibility

Timestep masking composes with every adapter module type. The mask is applied at the bottleneck (after down-projection), so it is orthogonal to the module's outer parameterization:

| Module | Where mask is applied |
|--------|----------------------|
| **LoRA** | After `lora_down`, before dropout and `lora_up` |
| **DoRA** | Same as LoRA; magnitude `dora_scale` is separate |
| **OrthoLoRA (Cayley)** | After `Q_eff` projection, multiplied with `lambda_layer` |
| **HydraLoRA** | After shared `lora_down`; per-expert `lora_up` heads unaffected |
| **ReFT** | Separate mask with its own `reft_dim` and floor of 1 |

`configs/gui-methods/tlora_ortho_reft.toml` and the default block in `configs/methods/lora.toml` stack LoRA + OrthoLoRA + T-LoRA + ReFT together.

## Configs

`configs/methods/lora.toml` (T-LoRA toggle block) — OrthoLoRA (Cayley) + timestep masking, rank 64:

```toml
use_ortho = true
use_timestep_mask = true
min_rank = 1
alpha_rank_scale = 1.0
network_dim = 64
```

## Implementation

| File | Role |
|------|------|
| `networks/lora_anima/network.py` | `set_timestep_mask()` — computes rank, writes shared mask |
| `networks/lora_anima/network.py` | `set_reft_timestep_mask()` — same for ReFT modules |
| `networks/lora_anima/network.py` | `clear_timestep_mask()` — removes mask (for inference) |
| `networks/lora_modules/lora.py` | Per-module mask application in each forward method |
| `train.py` | Calls `set_timestep_mask()` each step after noise sampling |
