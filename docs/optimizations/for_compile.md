# Changes from sd-scripts for torch.compile / dynamo

This document catalogues every change made to the `anima_lora` fork (relative to the original `sd-scripts` repo) that enables or supports `torch.compile` and PyTorch dynamo. Changes are grouped by file and subsystem.

---

## 1. Attention dispatch (`networks/attention.py`)

### 1.1 Flash Attention 4 graph breaks

FA4's CUTLASS/TVM kernels access raw DLPack data pointers, which fail with FakeTensors during dynamo tracing. Since FA4 is already a fused kernel that `torch.compile` cannot improve, we wrap it with `@torch.compiler.disable` to insert clean graph breaks while letting surrounding ops compile normally.

```python
# NEW
@torch.compiler.disable
def flash_attn_4_func(*args, **kwargs):
    out, _lse = _flash_attn_4_func_raw(*args, **kwargs)
    return out

@torch.compiler.disable
def flash_attn_4_varlen_func(*args, **kwargs):
    out, _lse = _flash_attn_4_varlen_func_raw(*args, **kwargs)
    return out
```

**sd-scripts**: No FA4 support at all.

> Note: FA4 is currently not the default attention backend — see [`fa4.md`](fa4.md) for why. The code paths remain in place for re-enabling.

### 1.2 Flex attention: NOT pre-compiled

When blocks are individually compiled (`static_token_count` mode), the outer `torch.compile` already traces into `flex_attention` and fuses it. Pre-compiling causes nested compilation that exhausts dynamo's recompile limit (`grad_mode` guard x mask variants) and falls back to the slow unfused path.

```python
# NEW — intentionally NOT compiled
compiled_flex_attention = _flex_attention  # raw, not torch.compile(...)
```

**sd-scripts**: No flex attention support.

### 1.3 Flex attention early-return path

New first-class `"flex"` attention mode with pre-computed `BlockMask` support for both cross-attention (KV trimming) and self-attention (static-shape padding). This avoids data-dependent control flow that would cause graph breaks.

### 1.4 New AttentionParams fields

| Field | Purpose |
|-------|---------|
| `softmax_scale` | Custom softmax scale passed through to all backends (avoids per-call branching) |
| `crossattn_block_mask` | Pre-computed BlockMask for cross-attention KV trimming (flex mode) |
| `selfattn_block_mask` | Pre-computed BlockMask for self-attention padding mask (flex, static-shape) |
| `crossattn_full_len` | Original KV length before bucketed trimming, for LSE sink correction (flash4) |

### 1.5 LSE sink correction for trimmed cross-attention (flash4)

When zero-padded KV positions are trimmed for efficiency, the softmax denominator must be corrected. New sigmoid-based correction:

```python
correction = torch.sigmoid(lse - math.log(n_pad))
x = out * correction.transpose(1, 2).unsqueeze(-1)
```

---

## 2. Model architecture (`library/anima/models.py`)

### 2.1 Removed `einops.rearrange`

`einops.rearrange` uses string-based symbolic shape parsing that is opaque to dynamo. All uses replaced with explicit tensor operations:

| Original (einops) | Replacement |
|---|---|
| `rearrange(t, "b ... (h d) -> b ... h d", h=..., d=...)` | `.unflatten(-1, (n_heads, head_dim))` |
| `rearrange(x, "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)", ...)` | `.unflatten().permute().reshape()` chain |
| `rearrange(em, "t h w d -> (t h w) 1 1 d")` | `.flatten(0, 2).unsqueeze(1).unsqueeze(1)` |
| `rearrange(shift, "b t d -> b t 1 1 d")` | `shift[:, :, None, None, :]` |

### 2.2 Removed `torch.autocast` context managers

Context managers introduce overhead and are difficult for dynamo to trace through. Removed from:

- **RMSNorm.forward**: replaced `with torch.autocast(...)` with direct `.float()` / `.to(x.dtype)` casts.
- **FinalLayer.forward**: removed `use_fp32` parameter and autocast wrapping entirely.

### 2.3 `.repeat()` → `.expand()`

`expand()` creates a view without allocating memory, while `repeat()` copies data. In `VideoRopePosition3DEmb.prepare_embedded_sequence`:

```python
# OLD
padding_mask.unsqueeze(1).repeat(1, n_heads, 1)
# NEW
padding_mask.unsqueeze(2).expand(-1, -1, n_heads)
```

### 2.4 KV bucket trimming constants

New `_KV_BUCKETS = (64, 128, 256, 512)` — cross-attention KV sequences are trimmed to the smallest bucket that fits, giving `torch.compile` at most 4 shape variants instead of one per unique caption length.

### 2.5 `set_static_token_count(count)`

API to enable static-shape training. When set, every forward pass pads visual token sequences to exactly `count` tokens (typically 4096), eliminating shape variation across bucket resolutions.

### 2.6 `compile_blocks(backend="inductor")`

Compiles each transformer block's `_forward` method individually:

```python
for block in self.blocks:
    block._forward = torch.compile(block._forward, backend=backend, dynamic=True)
```

**Critical:** compiles `_forward` (the actual attention/MLP), NOT `forward` (the checkpointing wrapper). The gradient checkpointing decorator (`unsloth_checkpoint`) uses `@torch._disable_dynamo`, which would cause an immediate graph break if `forward` itself were compiled — dynamo compiles nothing useful but still checks shape guards, causing recompile storms.

### 2.7 Static-shape padding in `forward_mini_train_dit`

When `static_token_count` is set:

1. Flatten 5D input `(B, T, H, W, D)` to `(B, seq_len, D)`.
2. Pad sequence dim to target length with zeros.
3. Reshape to fake-5D `(B, 1, target, 1, D)` — compatible with existing block code.
4. Pad RoPE embeddings and extra positional embeddings to match.
5. After all blocks: squeeze fake dims and strip padding to restore `(B, T, H, W, D)`.

### 2.8 Pre-computed BlockMask for flex attention

In static-shape mode, two BlockMasks are created before the block loop:

- **Cross-attention mask**: masks out zero-padded KV positions from bucketed trimming.
- **Self-attention mask**: masks out padding tokens introduced by `static_token_count`.

These are passed via `AttentionParams` so flex attention never needs to create masks inside the compiled region (which would cause data-dependent graph breaks).

---

## 3. Datasets (`library/datasets/`)

### 3.1 Constant-token buckets (`buckets.py`)

New `CONSTANT_TOKEN_BUCKETS` — 17 predefined `(W, H)` resolutions where `(W/16)·(H/16) ~ 4096` tokens with minimal padding (0.0%–1.6%). Used with `--static_token_count=4096` to make every forward pass shape-identical.

```python
CONSTANT_TOKEN_BUCKETS = [
    (1024, 1024),   # 4096 tokens, 0.0% pad
    (960, 1088),    # 4080 tokens, 0.4% pad
    (1088, 960),
    # ... 14 more landscape/portrait pairs
    (2048, 512),    # 4096 tokens, 0.0% pad
]
```

`BucketManager.make_buckets()` accepts `constant_token_buckets=True` to use these instead of dynamically generated resolutions.

### 3.2 Incomplete batch dropping (`base.py`)

Incomplete last batches are dropped (integer division instead of ceiling) to keep the batch dimension constant across epochs. This prevents `torch.compile` recompilation from a trailing partial batch.

```python
# When no sample_ratio: drop incomplete last batch
batch_count = len(bucket) // self.batch_size
```

Skipped when `sample_ratio < 1.0` (where every image matters more).

---

## 4. Training script (`train.py`)

### 4.1 Conditional block-level compilation

```python
if getattr(args, "static_token_count", None) is not None:
    model.set_static_token_count(args.static_token_count)
    if args.torch_compile:
        model.compile_blocks(args.dynamo_backend)
```

When `--static_token_count` is set, block-level compilation is used instead of full-graph compilation via Accelerator.

### 4.2 Dynamo backend routing (`library/train_util.py`)

```python
# Only enable Accelerator-level dynamo when static_token_count is NOT used
# (block-level compilation handles it separately)
dynamo_backend = "NO"
if args.torch_compile and not getattr(args, "static_token_count", None):
    dynamo_backend = args.dynamo_backend
```

**sd-scripts**: Always passes `dynamo_backend` to Accelerator when `torch_compile` is set.

### 4.3 Padding mask caching

Padding masks are cached by `(batch_size, h, w, dtype, device)` key to avoid re-allocation every step:

```python
padding_mask_key = (bs, h_latent, w_latent, weight_dtype, accelerator.device)
padding_mask = self._padding_mask_cache.get(padding_mask_key)
```

### 4.4 `constant_token_buckets` plumbed to dataset config

```python
constant_token_buckets=getattr(args, "static_token_count", None) is not None,
```

Passed through `library/config/` to `BucketManager.make_buckets()`.

---

## 5. LoRA networks (`networks/lora_anima/`)

### 5.1 `_orig_mod_` key stripping

`torch.compile` wraps modules in `_orig_mod` containers, inserting `_orig_mod.` or `_orig_mod_` into state-dict keys. Three locations handle this:

1. **`create_network_from_weights()`** — strips keys when loading external checkpoints.
2. **Module discovery loop** — strips `_orig_mod.` from module paths during LoRA target matching.
3. **`_strip_orig_mod_keys()` static method + `load_state_dict()` override** — ensures any state-dict loaded into the network is normalized.

```python
@staticmethod
def _strip_orig_mod_keys(state_dict):
    new_sd = {}
    for key, val in state_dict.items():
        new_key = re.sub(r"(?<=_)_orig_mod_", "", key)
        new_sd[new_key] = val
    return new_sd

def load_state_dict(self, state_dict, strict=True, **kwargs):
    state_dict = self._strip_orig_mod_keys(state_dict)
    return super().load_state_dict(state_dict, strict=strict, **kwargs)
```

**sd-scripts**: Zero `_orig_mod_` awareness — loading a checkpoint trained with `torch.compile` would fail.

### 5.2 Memory-saving down-projection autograd (`networks/lora_modules/custom_autograd.py`)

The LoRA down projection runs its matmul in fp32 for accumulation precision:

```python
lx = F.linear(x_lora.float(), self.lora_down.weight.float())
```

`F.linear`'s backward saves the exact forward input, so the `.float()` upcast of `x` is retained across the fwd→bwd window as an fp32 tensor (4 B / elem). At `static_token_count=4096` this is 32 MiB per 2048-wide Linear and 128 MiB for the 8192-wide MLP `layer2` input; accumulated across 28 DiT blocks × ~5–6 adapted Linears per block this was the largest single source of LoRA-side activation VRAM.

The fix is a targeted activation-recompute trick: a custom `torch.autograd.Function` that saves the low-precision `x` (bf16, 2 B / elem) and recomputes `x.float()` (or `(x * inv_scale).float()`) in backward. The fp32 bottleneck matmul is preserved in both directions, so gradients are bitwise-identical to the legacy path for deterministic kernels.

**Relevant to compile:** the feature uses **two separate `autograd.Function` subclasses** (scaled and unscaled), not one with an optional tensor. This keeps the graph shape fixed — no shape-dependent Python branches, no optional-tensor sentinels that could cause guard churn:

```python
class LoRADownProjectFn(torch.autograd.Function):       # no channel-scale
    @staticmethod
    def forward(ctx, x, weight):
        out = F.linear(x.float(), weight.float())
        ctx.save_for_backward(x, weight)                # bf16 x saved, not x.float()
        return out

class ScaledLoRADownProjectFn(torch.autograd.Function): # with channel-scale
    @staticmethod
    def forward(ctx, x, weight, inv_scale):
        x_work = x * inv_scale
        out = F.linear(x_work.float(), weight.float())
        ctx.save_for_backward(x, weight, inv_scale)
        return out

def lora_down_project(x, weight, inv_scale):            # dispatch at module init
    if inv_scale is None:
        return LoRADownProjectFn.apply(x, weight)
    return ScaledLoRADownProjectFn.apply(x, weight, inv_scale)
```

Each adapted LoRA module carries a boolean attribute `use_custom_down_autograd` set once by the network factory — Dynamo sees a static Python branch inside `forward`, not a runtime dispatch.

Wired through `LoRAModule`, `HydraLoRAModule`, `OrthoLoRAExpModule`, and `OrthoHydraLoRAExpModule`. The Ortho variants pass `Q_eff = R_q @ Q_basis` as the "weight" argument — autograd returns `grad_Q_eff`, which the existing graph propagates into `S_q` unchanged. ReFT (block-level intervention) and Conv2d LoRA are intentionally out of scope and take the legacy path.

**Measured (60-step A/B under `torch.compile`, default stack):** loss/average matched within 0.7 % (run-to-run noise), per-step loss statistically indistinguishable at z = +1.84, wall/step matched within 0.4 %, peak VRAM dropped ~4 GiB. The wall-clock parity is the compile-relevant signal: if Dynamo had broken the graph at each LoRA-patched Linear (once per `autograd.Function.apply`), kernel-launch overhead across 28 × ~5–6 sites would have erased the memory win. It didn't.

**sd-scripts**: No equivalent — plain `F.linear(x.float(), ...)` retains the fp32 cast unconditionally.

**Opt-in flag:** `use_custom_down_autograd = true` in `configs/methods/lora.toml` (or `--network_args use_custom_down_autograd=true`). Default off for now; ready to flip to default-on once a wider set of runs confirms the compile-graph behavior.

---

## 6. LoRA utils (`networks/lora_utils.py`)

Same `_orig_mod_` normalization applied during LoRA weight merging:

```python
# Strip _orig_mod_ from LoRA keys (inserted by torch.compile during training)
for k, v in lora_sd.items():
    normalized[k.replace("__orig_mod_", "_")] = v
```

---

## 7. Config (`library/train_util.py` dataset blueprint path)

`generate_dataset_group_by_blueprint()` accepts a new `constant_token_buckets: bool` parameter, forwarded to `dataset.make_buckets()`.

---

## 8. CLI arguments

### New in anima_lora

| Argument | File | Purpose |
|----------|------|---------|
| `--static_token_count N` | `library/anima/training.py` | Pad to N visual tokens; enables constant-shape buckets |
| `--trim_crossattn_kv` | `library/anima/training.py` | Enable bucketed KV trimming for cross-attention (no-op since FA4 removal) |

### Changed behavior

| Argument | sd-scripts | anima_lora |
|----------|-----------|------------|
| `--torch_compile` | Full-graph via Accelerator | Block-level if `static_token_count` set; Accelerator-level otherwise |
| `--dynamo_backend` | Always forwarded to Accelerator | Conditional: only forwarded when `static_token_count` is NOT used |

---

## Summary: the compilation strategy

The key insight is that a DiT training loop has three sources of shape dynamism that trigger `torch.compile` recompilation:

1. **Spatial resolution** — different bucket sizes produce different `(T, H, W)` token counts.
2. **Caption length** — variable text encoder output lengths for cross-attention KV.
3. **Batch size** — trailing incomplete batches at epoch boundaries.

The fork eliminates all three:

| Source | Solution | Files |
|--------|----------|-------|
| Spatial resolution | `CONSTANT_TOKEN_BUCKETS` + `static_token_count` padding | `buckets.py`, `library/anima/models.py` |
| Caption length | `_KV_BUCKETS` bucketed trimming (max 4 variants) | `library/anima/models.py`, `networks/attention.py` |
| Batch size | Drop incomplete last batches | `library/datasets/base.py` |

With shapes stabilized, `compile_blocks()` compiles each block's `_forward` with `dynamic=True` — the inductor backend generates optimized kernels once and reuses them for every step.
