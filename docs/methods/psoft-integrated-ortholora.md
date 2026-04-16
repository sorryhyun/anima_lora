# OrthoLoRA (Exp): Cayley Parameterization + SVD-Informed Init

Experimental variant of OrthoLoRA that replaces soft orthogonality
regularization with a hard Cayley constraint and initializes bases from
the pretrained weight's SVD. Gated behind `use_ortho_exp = true`.

Inspired by PSOFT (Wu et al., "Efficient Orthogonal Fine-Tuning with
Principal Subspace Adaptation", ICLR 2026).

## Motivation

Standard OrthoLoRA uses fully trainable `P` (d x r) and `Q` (r x d)
matrices with soft regularization (`||P^T P - I||^2 + ||Q Q^T - I||^2`)
to encourage orthonormality. Two weaknesses:

1. **Hyperparameter sensitivity** — `ortho_reg_weight` controls the
   trade-off between orthogonality and task loss. Too low: bases drift
   from orthonormality. Too high: constrains learning.
2. **Random initialization** — bases are initialized via QR of random
   Gaussian matrices, unrelated to the pretrained weight structure.

## Design

### Cayley parameterization

Instead of trainable (d x r) basis matrices, the module stores:

- **Frozen bases** `P_basis` (d x r) and `Q_basis` (r x d) — from SVD
  of the pretrained weight
- **Trainable skew-symmetric seeds** `S_p` and `S_q` (r x r) — fed
  through the Cayley transform to produce orthogonal rotation matrices

The Cayley transform maps a skew-symmetric matrix to an orthogonal
matrix:

```
A = S - S^T               (enforce skew-symmetry)
R = (I - A)(I + A)^{-1}   (guaranteed R^T R = I)
```

Effective bases are computed each forward pass:

```
P_eff = P_basis @ cayley(S_p)    # (d, r) — columns stay orthonormal
Q_eff = cayley(S_q) @ Q_basis    # (r, d) — rows stay orthonormal
```

At init `S_p = S_q = 0`, so `R = I` and `P_eff = P_basis`, `Q_eff = Q_basis`.
Combined with zero-init lambda, the LoRA output is exactly zero at init.

We use **exact inverse** via `torch.linalg.solve` instead of the Neumann
series approximation from the PSOFT paper. For r x r matrices (r = 4-64),
the exact solve is negligible cost and always correct — the Neumann series
only converges when `||S|| < 1` and silently diverges otherwise.

### SVD-informed initialization

`P_basis` and `Q_basis` are the top-r left and right singular vectors of
the pretrained weight. This aligns the initial basis with the weight's
principal subspace, so the Cayley rotation starts from a structurally
meaningful point.

### Trainable parameter count

Per module at rank r, dim d:

| Component | Standard OrthoLoRA | OrthoLoRA Exp |
|-----------|-------------------|---------------|
| Trainable | r(d_in + d_out) + r | 2r^2 + r |
| Frozen buffers | r(d_in + d_out) + r | r(d_in + d_out) |
| Optimizer states | ~ trainable | ~ trainable |

At r=64, d=3072: trainable params drop from ~394K to ~8.3K per module.
Frozen buffer size is similar. Optimizer memory is significantly reduced.

## Expressiveness trade-off

**This is the key thing being benchmarked.**

The Cayley rotation can only rotate `P_basis`'s columns within their
initial r-dimensional span — it cannot reach orthonormal bases outside
that subspace. This is effectively a principal-subspace restriction
(similar to PSOFT's design), despite the full-dim frozen bases.

For tasks that preserve pretrained representations (NLP fine-tuning),
this is well-motivated. For learning genuinely new visual concepts
(characters, styles) that may need components outside the top-r singular
vectors, it may be too restrictive.

Standard OrthoLoRA can learn **any** orthonormal basis via gradient
descent on the full (d x r) matrices (soft-regularized). OrthoLoRA Exp
trades that expressiveness for guaranteed orthogonality and fewer
trainable parameters.

## VRAM overhead

Compared to standard LoRA at the same rank, OrthoLoRA Exp uses slightly
more activation memory (~150-200MB at r=64 across 196 DiT modules)
because `P_eff` and `Q_eff` are computed tensors saved in the autograd
graph, not leaf parameters. Optimizer memory is lower (fewer trainable
params). Net effect is modest.

## Usage

In any method config (e.g. `configs/methods/lora.toml`):

```toml
use_ortho_exp = true
network_alpha = 64
network_dim = 64
```

`use_ortho_exp` takes priority over `use_ortho`. The `sig_type` and
`ortho_reg_weight` settings are ignored when `use_ortho_exp` is active.
Keep `alpha = dim` for neutral scaling.

For T-LoRA (with timestep masking), in `configs/methods/tlora.toml`:

```toml
use_ortho_exp = true
use_ortho = true          # ignored when use_ortho_exp is active
use_timestep_mask = true
```

Then train as usual:

```bash
make lora          # or make tlora for the timestep-masked variant
```

## Save format

Training saves native keys (`S_p`, `S_q`, `lambda_layer`, `P_basis`,
`Q_basis`). On save, these are automatically converted to standard LoRA
(`lora_up.weight`, `lora_down.weight`) for ComfyUI compatibility.
The conversion is exact — `DeltaW = P_eff @ diag(lambda) @ Q_eff` is rank r,
factored directly without SVD.

## What to compare

When benchmarking against standard OrthoLoRA:

1. **Training loss curves** — does Cayley plateau higher? (subspace restriction)
2. **Generated image quality** — the real test for creative concept learning
3. **Orthogonality** — verify `P_eff^T @ P_eff ~ I` (should be exact, by construction)

## Differences from the PSOFT paper

| Aspect | PSOFT | Our implementation |
|--------|-------|-------------------|
| Bases | Frozen A', B' from symmetric SVD decomposition | Frozen P_basis, Q_basis from standard SVD |
| Rotation | Single r x r Cayley matrix R | Two r x r Cayley matrices (S_p, S_q) |
| Relaxation | Tunable alpha, beta vectors (2r params) | Tunable lambda vector (r params) |
| Inverse | Neumann series (K=5) | Exact solve (r x r is cheap) |
| Residual | Frozen W_res preserved | No residual — zero-init lambda instead |
| Target | NLP/CV classification | Creative visual concept learning (LoRA) |

## Files

| File | What changed |
|------|-------------|
| `networks/lora_modules.py` | `OrthoLoRAExpModule` class |
| `networks/lora_anima.py` | Config parsing, module selection, save conversion |
| `configs/methods/tlora.toml` | `use_ortho_exp` toggle |
