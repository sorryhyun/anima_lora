# OrthoLoRA (Exp): Cayley Parameterization + SVD-Informed Init

Experimental variant of OrthoLoRA that replaces soft orthogonality regularization with a hard Cayley constraint and initializes bases from the pretrained weight's SVD. Gated behind `use_ortho = true`.

Inspired by PSOFT (Wu et al., "Efficient Orthogonal Fine-Tuning with Principal Subspace Adaptation", ICLR 2026).

> **For the structural walkthrough** (frozen SVD bases, Cayley transform derivation, forward pass, why exact inverse beats Neumann, expressiveness trade-off), see **`docs/structure/ortholora.md`**. This doc is the usage / ops / benchmarking reference.

## Usage

In any method config (e.g. `configs/methods/lora.toml` or `configs/gui-methods/ortholora.toml`):

```toml
use_ortho = true
network_alpha = 64
network_dim = 64
```

Keep `alpha = dim` for neutral scaling.

For timestep-masked OrthoLoRA, uncomment the T-LoRA block in `configs/methods/lora.toml` or use the per-variant path:

```bash
make lora-gui GUI_PRESETS=tlora
```

## Trainable parameter count

Per module at rank r, dim d:

| Component | Standard OrthoLoRA | OrthoLoRA Exp |
|-----------|-------------------|---------------|
| Trainable | r(d_in + d_out) + r | 2r² + r |
| Frozen buffers | r(d_in + d_out) + r | r(d_in + d_out) |
| Optimizer states | ~ trainable | ~ trainable |

At r=64, d=3072: trainable params drop from ~394K to ~8.3K per module. Frozen buffer size is similar. Optimizer memory is significantly reduced.

## VRAM overhead

Compared to standard LoRA at the same rank, OrthoLoRA Exp uses slightly more activation memory (~150–200 MB at r=64 across 196 DiT modules) because `P_eff` and `Q_eff` are computed tensors saved in the autograd graph, not leaf parameters. Optimizer memory is lower (fewer trainable params). Net effect is modest.

## Save format

Training saves native keys (`S_p`, `S_q`, `lambda_layer`, `P_basis`, `Q_basis`). On save, these are automatically converted to standard LoRA (`lora_up.weight`, `lora_down.weight`) for ComfyUI compatibility. The conversion is exact — `DeltaW = P_eff @ diag(lambda) @ Q_eff` is rank r, factored directly without SVD.

## What to compare

When benchmarking against standard OrthoLoRA:

1. **Training loss curves** — does Cayley plateau higher? (subspace restriction)
2. **Generated image quality** — the real test for creative concept learning
3. **Orthogonality** — verify `P_eff^T @ P_eff ~ I` (should be exact, by construction)

## Differences from the PSOFT paper

| Aspect | PSOFT | Our implementation |
|--------|-------|-------------------|
| Bases | Frozen A', B' from symmetric SVD decomposition | Frozen P_basis, Q_basis from standard SVD |
| Rotation | Single r × r Cayley matrix R | Two r × r Cayley matrices (S_p, S_q) |
| Relaxation | Tunable alpha, beta vectors (2r params) | Tunable lambda vector (r params) |
| Inverse | Neumann series (K=5) | Exact solve (r × r is cheap) |
| Residual | Frozen W_res preserved | No residual — zero-init lambda instead |
| Target | NLP/CV classification | Creative visual concept learning (LoRA) |

## Files

| File | What changed |
|------|-------------|
| `networks/lora_modules/ortho.py` | `OrthoLoRAExpModule` class |
| `networks/lora_anima/` | Config parsing, module selection, save conversion |
| `configs/methods/lora.toml` | `use_ortho` toggle |
