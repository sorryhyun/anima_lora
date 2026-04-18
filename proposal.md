# Proposal: make fp32-accumulation the default for LoRA / ReFT bottlenecks

## Summary

Remove the `lora_fp32_accumulation` flag and make fp32 compute the unconditional
behavior in `LoRAModule` / `HydraLoRAModule` forward paths. Apply the same
pattern to `ReFTModule`. Net effect: every low-rank-bottleneck matmul and the
ReFT `delta = source − rotated` subtraction run in fp32 regardless of autocast,
while stored parameters stay bf16.

This is a near-free precision gain — no persistent memory cost (temp
activations only), no change to saved weights, no change to inference (except
that inference already runs fp16/bf16 matmul on the same path).

## Motivation

Training the pipeline in bf16 is the right default: the base DiT, LoRA
weights, and saved checkpoints are all bf16. Mixed-precision autocast handles
most compute correctly. However, two spots in the network are numerically
sensitive enough that bf16 compute measurably degrades signal:

1. **LoRA bottleneck** — `y = up(down(x))` where the inner rank is small. Low
   rank means each element of the intermediate aggregates many bf16 products,
   and bf16's 7-bit mantissa accumulates fp-addition error faster than fp32's
   23-bit. This is exactly what the existing `lora_fp32_accumulation` flag
   addresses.

2. **ReFT subtraction** — `delta = source − rotated` where `source = Wh + b`
   and `rotated = Rh`. At init, `learned_source.weight` is cloned from
   `rotate_layer.weight`, so source ≡ rotated and delta = 0 exactly. Early in
   training, source and rotated drift but stay close, so delta is the
   difference of two similar-magnitude bf16 values — textbook catastrophic
   cancellation. In bf16, `(1.001 − 1.000)` can collapse to 0. The effect
   suppresses the learning signal most severely during the early optimization
   steps when it matters most.

Both cases are opt-in today (or absent, for ReFT). Making them unconditional
treats the policy as: **bf16 for storage, fp32 as a surgical accumulation
override where the math demands it.** That policy is already how
`OrthoLoRAExpModule` / `OrthoHydraLoRAExpModule` behave — they always run
compute in fp32 via `P_basis.dtype` because the Cayley `linalg.solve` needs
it. This proposal extends the same principle to the LoRA bottleneck and ReFT
subtraction.

## Cost analysis

Precision upgrade is near-free:

- **Parameter memory**: unchanged. Params stay bf16; we cast activations
  to fp32 via `.float()` inside forward only.
- **Optimizer state**: unchanged. bf16 master weights → AdamW8bit moments
  are whatever they were.
- **Activation memory**: fp32 activations are 2× bf16 but they're transient
  (freed after the forward, or recomputed under gradient checkpointing).
  The intermediates we cast are at rank dimension (LoRA) or reft_dim / h
  (ReFT) — none are the full `(B, L, embed_dim)` tensors.
- **Throughput**: fp32 GEMM is slightly slower than bf16 on modern GPUs, but
  the inner matmul is tiny (rank × embed_dim or reft_dim × embed_dim). The
  hit is below measurement noise on full-DiT training.

Compare against the cost of the precision bug: ReFT gradients near zero
during early training steps mean the LoReFT edit fails to bootstrap. Users
would likely not catch this from the loss curve alone — it looks like slow
convergence.

## Current state

### `LoRAModule.forward` (networks/lora_modules.py:217-252)

```python
if self.fp32_accumulation:
    lx = F.linear(x_lora.float(), self.lora_down.weight.float())
else:
    lx = self.lora_down(x_lora)
# ... rank dropout, timestep mask ...
if self.fp32_accumulation:
    lx = F.linear(lx, self.lora_up.weight.float())
    lx = (lx * self.multiplier * scale).to(org_forwarded.dtype)
    return org_forwarded + lx
else:
    lx = self.lora_up(lx)
    return org_forwarded + lx * self.multiplier * scale
```

### `HydraLoRAModule.forward` (networks/lora_modules.py:343, 374)

Same pattern — `if self.fp32_accumulation` wraps the two narrow matmuls.

### `ReFTModule.forward` (networks/lora_modules.py:722-748)

No fp32 accumulation path. Runs entirely under autocast, which demotes to
bf16 under mixed-precision training. The `delta = source - rotated`
subtraction therefore runs in bf16.

### `OrthoLoRAExpModule` / `OrthoHydraLoRAExpModule`

Already run fp32 unconditionally via `dtype = self.P_basis.dtype` (fp32
from SVD init). No change needed; the flag was a no-op here anyway.

### Config plumbing

- `--lora_fp32_accumulation` CLI flag (library/anima/training.py:167)
- `lora_fp32_accumulation = true` currently in `configs/methods/lora.toml`
- Read at `networks/lora_anima.py:1186-1189` and assigned to each LoRA
  module's `.fp32_accumulation` attribute.

## Proposed change

### 1. `LoRAModule` / `HydraLoRAModule`

Inline the fp32-cast path as the sole behavior. Delete the `if
self.fp32_accumulation` branch. Resulting forward does:

```python
# Bottleneck matmul is accumulated in fp32: rank is small so the inner
# sum burns bf16 mantissa quickly; promoting to fp32 costs transient
# activation memory only and recovers low-bit precision. Stored params
# stay bf16 (policy: bf16 storage, fp32 for surgical accumulation).
lx = F.linear(x_lora.float(), self.lora_down.weight.float())
# ... masks / dropout in fp32 ...
lx = F.linear(lx, self.lora_up.weight.float())
return org_forwarded + (lx * self.multiplier * scale).to(org_forwarded.dtype)
```

Remove the `fp32_accumulation` attribute from `BaseLoRAModule.__init__`.

### 2. `ReFTModule`

Mirror the same pattern. Cast `h` and the ReFT weights to fp32 for the
rotation + subtraction + back-projection, then downcast at the end:

```python
def forward(self, x):
    h = self.org_forward(x)
    if self.module_dropout is not None and self.training:
        if torch.rand(1) < self.module_dropout:
            return h

    # ReFT runs the rotation / subtraction / back-projection in fp32.
    # `delta = source - rotated` is catastrophic cancellation territory:
    # at init the two terms are equal by construction (learned_source is
    # cloned from rotate_layer), and early training keeps them close.
    # bf16 would collapse the learning signal; fp32 activations are free
    # here (rank-dim intermediates only, params stay bf16).
    W = self.rotate_layer.weight.float()
    h_f = h.float()
    rotated = F.linear(h_f, W)
    source = F.linear(
        h_f,
        self.learned_source.weight.float(),
        self.learned_source.bias.float(),
    )
    if self._timestep_mask is not None and self.training:
        rotated = rotated * self._timestep_mask
        source = source * self._timestep_mask
    delta = source - rotated
    if self.dropout is not None and self.training:
        delta = F.dropout(delta, p=self.dropout)
    edit = F.linear(delta, W.T)
    return h + (edit * self.multiplier * self.scale).to(h.dtype)
```

### 3. Remove flag plumbing

- Drop `--lora_fp32_accumulation` CLI arg from
  `library/anima/training.py:167`.
- Drop the read at `networks/lora_anima.py:1186-1189`.
- Drop the `lora_fp32_accumulation = true` line from
  `configs/methods/lora.toml`.
- Search for any other references (scripts, GUI config schema, etc.) and
  remove them.

### 4. Documentation

Add a one-paragraph note in `networks/lora_modules.py` above each affected
class's forward explaining *why* the fp32 cast exists. Keep the comments
short — the policy is the policy.

No user-facing doc change required: the previous flag was opt-in, so
existing configs that set it will now either be quietly ignored (need a
deprecation warning path) or fail on unknown-key validation. See
"Migration" below.

## Migration

`lora_fp32_accumulation` may appear in:

- User configs (`configs/methods/*.toml`) — only in `lora.toml` in-tree.
- Saved config snapshots (`.toml` emitted alongside checkpoints by
  `--no-config-snapshot`-off runs).
- External scripts in `anima_lora/scripts/`.
- GUI forms (`gui/`).

Options:

1. **Hard remove** — remove the argparse entry. Old configs that pass it
   will fail with "unknown key" from the config schema. Acceptable if
   config validation is already strict.
2. **Deprecation shim** — accept the flag, ignore it, warn once at load
   time. More forgiving for users with pinned configs; costs ~5 lines.

Recommend option 2 for one release cycle, then option 1.

## Risks

- **Throughput regression**: fp32 matmul is slower than bf16. Real-world
  impact is small (the inner dim is tiny) but should be measured before
  merging on a full 16GB training run.
- **Activation-memory pressure in gradient checkpointing**: if a checkpoint
  boundary sits around a LoRA module, the fp32 temporaries have to be
  recomputed in fp32 on backward. Still no persistent cost, but peak
  memory during recomputation rises modestly. Likely a non-issue —
  `blocks_to_swap` already moves most bulk — but worth watching during
  the measurement run.
- **Numerical behavioral change for existing LoRA-only runs**: users who
  had `lora_fp32_accumulation = false` (explicitly or by default) will
  see a different training trajectory. In practice the change is strictly
  a precision upgrade, so the loss should be equal or lower. Regression
  tests on a reference run are worthwhile.

## Rollout

1. Implement the module changes (LoRA, Hydra, ReFT) and update comments.
2. Remove the flag with a one-release deprecation warning.
3. Run a reference 3-epoch `tlora_rf` on a small subset and compare
   the loss curve against the pre-change bf16-subtraction baseline to
   confirm ReFT gradients flow sooner (sanity check, not a gate).
4. Delete the flag fully in the next release.
