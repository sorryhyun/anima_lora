# Plan: Cayley parameterization + SVD-informed init for OrthoLoRA

## Context

OrthoLoRA currently uses soft orthogonality regularization (`||P^TP - I||^2 + ||Q Q^T - I||^2`) with a
tunable penalty weight. This has two weaknesses: (1) P and Q can drift from orthogonality depending on the
reg weight hyperparameter, and (2) the bases are initialized from random Gaussian QR — unrelated to the
pretrained weight structure.

The PSOFT paper (ICLR 2026) introduces two ideas worth adopting:
- **Cayley parameterization**: Hard orthogonality constraint via `R = (I - S)(I + S)^{-1}` where S is
  skew-symmetric. Guarantees `R^T R = I` at every gradient step with zero hyperparameters.
- **SVD-informed initialization**: Initialize P, Q from the actual pretrained weight's top-r singular vectors
  instead of random bases.

We do NOT adopt PSOFT's principal-subspace restriction (freezing A', B', W_res). Our P and Q remain
trainable and full-dimensional, preserving expressiveness for learning new visual concepts.

## Changes

### 1. `networks/lora_modules.py` — OrthoLoRAModule

**Cayley parameterization** — Replace the free `p_layer` / `q_layer` Linear modules with Cayley-parameterized
orthogonal matrices. The trainable parameters become skew-symmetric matrices `S_p` and `S_q`, and the
actual orthogonal matrices are computed on-the-fly during forward:

```
R = (I - S)(I + S)^{-1}    where S = -S^T
```

Approximate `(I + S)^{-1}` via truncated Neumann series (K=5 terms):

```
(I + S)^{-1} ≈ sum_{k=0}^{K} (-S)^k
```

Concrete changes:
- Remove `p_layer` (nn.Linear, out_dim × rank) and `q_layer` (nn.Linear, rank × in_dim)
- Add `S_p` (nn.Parameter, rank × rank, skew-symmetric, for P) and `S_q` (nn.Parameter, rank × rank,
  skew-symmetric, for Q)
- Add frozen `P_basis` buffer (out_dim × rank) and `Q_basis` buffer (rank × in_dim) — the initial
  orthogonal bases that the Cayley rotation modifies:
  - `P_effective = P_basis @ cayley(S_p)` → (out_dim × rank), guaranteed orthonormal columns
  - `Q_effective = cayley(S_q) @ Q_basis` → (rank × in_dim), guaranteed orthonormal rows
- Keep `lambda_layer` (1 × rank) as the learnable singular value vector — unchanged
- Remove `base_p_weight`, `base_q_weight`, `base_lambda` frozen buffers (the residual subtraction for
  zero-init is no longer needed — Cayley starts at I, so P_effective = P_basis at init, and lambda starts
  at base values → delta = 0 naturally when lambda is initialized correctly)

Wait — we still need zero output at init. With Cayley: S_p = S_q = 0 → R = I → P_effective = P_basis,
Q_effective = Q_basis. So at init the trainable path equals the frozen base path exactly. We need the same
residual subtraction as before, OR we can initialize lambda to zero (so output is zero regardless of P, Q).

**Decision**: Initialize `lambda_layer` to zeros. At init, output = `P_eff @ diag(0) @ Q_eff = 0`.
During training, lambda grows from zero. This eliminates the need for frozen base copies entirely,
and is simpler than the current Marchenko-Pastur initialization + residual subtraction.

Actually — the current Marchenko-Pastur lambda init is intentional: it sets the initial scale to match
what a random Gaussian(0, 1/r) matrix's singular values would be, so the first gradient step is well-scaled.
Zero-init lambda means the first few steps produce negligible LoRA output. This is the same tradeoff as
standard LoRA's zero-init of lora_up. Let's go with zero-init lambda for simplicity — it's proven to work
in standard LoRA and PSOFT both use similar initialization.

**Forward pass** (simplified):
```python
def forward(self, x):
    org_out = self.org_forward(x)
    x_lora = x * self.inv_scale if self._has_channel_scale else x
    
    # Cayley-parameterized orthogonal matrices
    P = self.P_basis @ self._cayley(self.S_p)   # (out_dim, rank)
    Q = self._cayley(self.S_q) @ self.Q_basis   # (rank, in_dim)
    
    lx = F.linear(x_lora, Q)          # (*, rank)
    lx = lx * self.lambda_layer       # per-rank scaling
    # [timestep mask, dropout, rank dropout as before]
    out = F.linear(lx, P)             # (*, out_dim)
    
    return org_out + out * self.multiplier * self.scale
```

**`_cayley(S)` helper** — Neumann-approximated Cayley transform:
```python
def _cayley(self, S):
    # Enforce skew-symmetry
    A = S - S.T  # (rank, rank), guaranteed skew-symmetric
    # Neumann: (I + A)^{-1} ≈ I - A + A^2 - A^3 + ...
    power = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    result = power.clone()
    for _ in range(self.neumann_terms):
        power = -power @ A
        result = result + power
    return (torch.eye(A.shape[0], device=A.device, dtype=A.dtype) - A) @ result
```

**`regularization()` method** — Remove entirely. Cayley guarantees orthogonality structurally.
Keep the method signature returning `(0, 0)` tensors so callers don't break, or delete and remove
caller references. Prefer: return zeros so `get_ortho_regularization()` in lora_anima.py works unchanged.

**SVD-informed init** — New optional flag `svd_init`. When enabled:
- At `__init__`, takes `org_module.weight` → SVD → top-r left/right singular vectors
- `P_basis = U[:, :r]` (out_dim × rank), `Q_basis = V[:, :r].T` (rank × in_dim)
- When disabled, falls back to random QR init (current behavior)

**Parameter count change**:
- Before: `rank × out_dim + rank × in_dim + rank` (P_layer + Q_layer + lambda) + same for frozen bases
- After: `rank × rank + rank × rank + rank` (S_p + S_q + lambda) + `rank × out_dim + rank × in_dim` (frozen bases)
- Net trainable params: from `rank*(out+in) + rank` → `2*rank^2 + rank`. For rank=64, d=n=3072:
  before ~394K trainable, after ~8.3K trainable. Massive reduction. Frozen buffers are same size though.

### 2. `networks/lora_anima.py` — Network wiring

**Config parsing** (around line 375):
- Add `neumann_terms` kwarg (default 5)
- Add `svd_init` kwarg (default "true")
- Remove `sig_type` kwarg (no longer needed — lambda is zero-init)
- Keep `ortho_reg_weight` parsing but it becomes a no-op (Cayley is structurally orthogonal)

**Module instantiation** (around line 963):
- Pass `neumann_terms` and `svd_init` to OrthoLoRAModule constructor

**`get_ortho_regularization()`** (line 1245):
- OrthoLoRAModule.regularization() returns (0, 0) → method still works, just returns 0. No change needed.
- The `ortho_reg_weight` in train.py multiplies zero → no effect. Clean but harmless.

**Save conversion** (line 1593):
- Update the OrthoLoRA → standard LoRA conversion to handle new key names:
  - Old keys: `p_layer.weight`, `q_layer.weight`, `lambda_layer`, `base_p_weight`, `base_q_weight`, `base_lambda`
  - New keys: `S_p`, `S_q`, `lambda_layer`, `P_basis`, `Q_basis`
- Conversion logic: compute `P_eff = P_basis @ cayley(S_p)`, `Q_eff = cayley(S_q) @ Q_basis`,
  then `ΔW = P_eff @ diag(λ) @ Q_eff`. SVD of ΔW → lora_up/lora_down as before.
  Since ΔW is rank r (not rank 2r like before), this is actually simpler.

**Checkpoint loading** (around line 653):
- Detect new format via `"S_p" in key` (vs old `"q_layer" in key`)
- Support loading both old and new format checkpoints

### 3. `configs/methods/tlora.toml`

```diff
- sig_type = "last"
- ortho_reg_weight = 0.01
+ neumann_terms = 5
+ svd_init = true
```

### 4. `train.py` — post_process_loss

No changes needed. The ortho_reg pathway will compute 0 and add nothing to the loss.
Optionally: clean up the dead code path later, but not in this PR.

## Files to modify

1. `networks/lora_modules.py` — Rewrite OrthoLoRAModule (~180 lines)
2. `networks/lora_anima.py` — Config parsing, save conversion, checkpoint detection (~40 lines changed)
3. `configs/methods/tlora.toml` — Config keys (~2 lines)

## Verification

1. **Init sanity**: Instantiate OrthoLoRAModule, verify forward output is zero (lambda=0 → zero output)
2. **Orthogonality**: After a few training steps, verify `P_eff^T @ P_eff ≈ I` and `Q_eff @ Q_eff^T ≈ I`
3. **Training**: `make tlora` — should train without errors, loss should decrease
4. **Save/load**: Verify saved checkpoint converts to standard LoRA correctly, loads in ComfyUI
5. **Inference**: `make test` with the saved checkpoint — should produce coherent images
6. **Backward compat**: Old OrthoLoRA checkpoints (with `q_layer` keys) should still load correctly
