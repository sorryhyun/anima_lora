# Changes from sd-scripts for torch.compile / dynamo

Changes in the `anima_lora` fork (relative to `sd-scripts`) that enable or support `torch.compile` and dynamo, grouped by file.

---

## 1. Attention dispatch (`networks/attention_dispatch.py`)

### 1.1 Flash Attention 4 (removed)

FA4 and its KV-trim + LSE-sink correction path were removed 2026-05-20 (with the
`crossattn_full_len` field and the `_KV_BUCKETS` constant); `flash_attn_4_func` /
`flash_attn_4_varlen_func` survive only as `None` stubs in `attention_dispatch.py`.
Cross-attention runs the full 512-length KV under FA2. Postmortem:
[`fa4.md`](fa4.md).

### 1.2 Flex attention: NOT pre-compiled

When blocks are individually compiled (`compile_blocks` / native-flatten mode), the outer `torch.compile` already traces into `flex_attention` and fuses it. Pre-compiling causes nested compilation that exhausts dynamo's recompile limit (`grad_mode` guard x mask variants) and falls back to the slow unfused path.

```python
# NEW — intentionally NOT compiled
compiled_flex_attention = _flex_attention  # raw, not torch.compile(...)
```

sd-scripts: No flex attention support.

### 1.3 Flex attention early-return path

New first-class `"flex"` attention mode with pre-computed `BlockMask` support for the cross-attention padding mask. This avoids data-dependent control flow that would cause graph breaks. (The self-attention `BlockMask` only ever served the retired static-pad path; native shapes have no padded self-attn KV, so `selfattn_block_mask` stays `None`.)

### 1.4 New AttentionParams fields

| Field | Purpose |
|-------|---------|
| `softmax_scale` | Custom softmax scale passed through to all backends (avoids per-call branching) |
| `crossattn_block_mask` | Pre-computed BlockMask for the cross-attention padding mask (flex mode) |
| `selfattn_block_mask` | Unused in native mode (no padded self-attn KV); stays `None` |

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

- RMSNorm.forward: replaced `with torch.autocast(...)` with direct `.float()` / `.to(x.dtype)` casts.
- FinalLayer.forward: removed `use_fp32` parameter and autocast wrapping entirely.

### 2.3 `.repeat()` → `.expand()`

`expand()` creates a view without allocating memory, while `repeat()` copies data. In `VideoRopePosition3DEmb.prepare_embedded_sequence`:

```python
# OLD
padding_mask.unsqueeze(1).repeat(1, n_heads, 1)
# NEW
padding_mask.unsqueeze(2).expand(-1, -1, n_heads)
```

### 2.5 `compile_blocks(backend="inductor")` — the single switch

`compile_blocks` is the one call that turns on `torch.compile`. It does two coupled things and raises the dynamo cache-size budget itself:

1. Native-shape flattening (`self._native_flatten = True`). The forward flattens each bucket's patch grid `(B, T, H, W, D)` to a fake-5D `(B, 1, seq_len, 1, D)` shape (`unflatten`-restored after the block loop). This keys the block graph on token count alone instead of guarding `H` and `W` separately (one graph per resolution). No padding, so flash self-attention sees no padded tokens. Bit-exact to the eager 5D path; eager (uncompiled) forwards leave the flag `False` and skip the reshape.

2. Per-block compile. Compiles each block's `_forward` method:
   ```python
   for block in self.blocks:
       block._forward = torch.compile(block._forward, backend=backend, dynamic=False)
   ```
   **Critical:** compiles `_forward` (the actual attention/MLP), NOT `forward` (the checkpointing wrapper). The gradient checkpointing decorator (`unsloth_checkpoint`) uses `@torch._disable_dynamo`, which would cause an immediate graph break if `forward` itself were compiled — dynamo compiles nothing useful but still checks shape guards, causing recompile storms.

The budget is `recompile_limit = max(current, 2*n + 8)` (via `pin_dynamo_limit`), where `n` is the number of token-count families — derived by `train.py::_derive_token_budget` from the buckets actually populated, defaulting to the 1024 tier's count. The `2*` covers fwd+bwd sharing the one `_forward` bytecode, the `+8` covers requires_grad / stride specializations. The `max()` lets a multi-resolution caller (e.g. a distill loop whose downsampled stages produce more distinct shapes) pre-raise the limit without `compile_blocks` lowering it.

### 2.6 Dynamic-seq marks disable inductor mix-order reduction

When `compile_dynamic_seq` is active (`train.py` auto-enables it with `torch_compile`, and the
bespoke distill loops force it via `ensure_dynamic_seq_for_freefit`),
`compile_blocks` marks only the seq axis dynamic and bounds it to the tier's
`seq_range` via a strict `mark_dynamic`. That strict bound collides with an
inductor fusion pass:

```python
# compile_blocks, when dynamic-seq marks are active (per-band mode: only if a band straddles 4096)
pin_inductor_flag("triton.mix_order_reduction", False)  # library/runtime/dynamo.py
```

Why. `mix_order_reduction` (torch 2.12, default-on) guards its profitability
check at the 4096 boundary at the compile hint — recording `Ge(seq, 4096)`
or its negation `seq <= 4095` depending on which batch traced first — and that
guard contradicts any strict `mark_dynamic` range straddling 4096, raising
`ConstraintViolationError` at guard build (live, or replayed from the
FxGraphCache artifact on a later lookup). The failure only surfaces once a
backward gains a seq-axis reduction — e.g. an adaln LoRA makes
shift/scale/gate require grad, so the modulation grads reduce over the seq axis
(broadcast-backward). It therefore hits *any* LoRA on a broadcast-consumed
Linear, not just adaln, and only under dynamic-seq.

Why `pin_inductor_flag` and not plain assignment. Inductor config
`user_override`s are thread-local ContextVars (torch 2.12). A plain
`config.triton.mix_order_reduction = False` only exists in the thread that ran
it; the grad-enabled step-0 compile (grad-ckpt recompute / AOT backward path)
schedules in a different context where the override is absent and the read falls
back to the entry's default — env-derived True — so the kill silently
reverted and the fusion still recorded the guard (step-0 crash under the
grad-ckpt presets). The pin sets the entry's
`.default` too (same pattern as `pin_dynamo_limit` for `recompile_limit`), which
every context reads. Poisoned per-signature cache dirs from crashed runs
self-heal: the config is part of the FxGraphCache key, so stale entries miss.
See also the caveat on `isolate_compile_cache` (stale-guard poisoning across
cache reuse). Discovered fixing the adaln training path (2026-07-15); ContextVar
regression root-caused 2026-07-17 — `docs/methods/adaln.md` §Path 2.

sd-scripts has no dynamic-seq path, free-fit bucketing or per-block compile. Why
the earlier static-pad and constant-token-bucket modes were removed:
[`../structure/anima-optimizations.md`](../structure/anima-optimizations.md) §3.

---

## 3. Datasets (`library/datasets/`)

### 3.1 Free-fit bucketing (`buckets.py`)

Each image keeps its native aspect ratio and lands its token count anywhere inside
its tier's `EDGE_TOKEN_BANDS` band; `make_buckets()` takes the on-disk cached
`(W,H)` as the bucket set. The many in-band token counts would each be a static
graph, so free-fit relies on `compile_dynamic_seq` (§2.6). Bands, tier choice and
the compile coupling: [`../structure/anima-optimizations.md`](../structure/anima-optimizations.md) §3.

### 3.2 Incomplete batch dropping (`base.py`)

Incomplete last batches are dropped (integer division instead of ceiling) to keep the batch dimension constant across epochs. This prevents `torch.compile` recompilation from a trailing partial batch.

```python
# When no sample_ratio: drop incomplete last batch
batch_count = len(bucket) // self.batch_size
```

Skipped when `sample_ratio < 1.0` (where every image matters more).

---

## 4. Training script (`train.py`)

### 4.1 Block-level compilation

```python
if args.torch_compile:
    model.compile_blocks(args.dynamo_backend, mode=getattr(args, "compile_inductor_mode", None))
```

`compile_blocks` is the only compile path: it enables native-shape flattening and compiles each block individually (never a full-graph compile of the DiT).

### 4.2 Dynamo backend routing (`library/runtime/accelerator.py`)

```python
# Always "NO": torch.compile is applied per-block by compile_blocks. Letting
# Accelerate full-compile on top would double-compile / graph-break.
dynamo_backend = "NO"
```

sd-scripts: Always passes `dynamo_backend` to Accelerator when `torch_compile` is set.

### 4.3 Padding mask caching

Padding masks are cached by `(batch_size, h, w, dtype, device)` key to avoid re-allocation every step:

```python
padding_mask_key = (bs, h_latent, w_latent, weight_dtype, accelerator.device)
padding_mask = self._padding_mask_cache.get(padding_mask_key)
```

---

## 5. LoRA networks (`networks/lora_anima/`)

### 5.1 `_orig_mod_` key stripping

`torch.compile` wraps modules in `_orig_mod` containers, inserting `_orig_mod.` or `_orig_mod_` into state-dict keys. Three locations handle this:

1. `create_network_from_weights()` — strips keys when loading external checkpoints.
2. Module discovery loop — strips `_orig_mod.` from module paths during LoRA target matching.
3. `_strip_orig_mod_keys()` static method + `load_state_dict()` override — ensures any state-dict loaded into the network is normalized.

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

sd-scripts: Zero `_orig_mod_` awareness — loading a checkpoint trained with `torch.compile` would fail.

### 5.2 Custom down-projection autograd (removed 2026-06-10)

`custom_autograd.py` and its fp32-bottleneck matmul policy were removed: under
`accelerator.autocast()` the fp32 matmuls never executed (bit-identical forward in
the `lora_fp32_bottleneck` bench). Training forwards now run the rank GEMMs in the
frozen Linear's output dtype (`org_forwarded.dtype`, `networks/lora_modules/base.py`);
inference paths keep fp32. Regression tests: `tests/test_lora_dtype_policy.py`.
`use_custom_down_autograd` is still accepted as a logged no-op so old snapshot TOMLs
replay.

The removal changed the AOT partitioner's saved-activation set and OOMed the
16 GB no-grad-ckpt run; the replacement is `activation_memory_budget` (base.toml,
set in `train.py` before `compile_blocks`, auto-skipped under
`gradient_checkpointing`). Root cause and measurements:
[`../findings/custom_autograd_removal_partitioner_oom.md`](../findings/custom_autograd_removal_partitioner_oom.md).

---

## 6. LoRA utils (`networks/lora_utils.py`)

Same `_orig_mod_` normalization applied during LoRA weight merging:

```python
# Strip _orig_mod_ from LoRA keys (inserted by torch.compile during training)
for k, v in lora_sd.items():
    normalized[k.replace("__orig_mod_", "_")] = v
```

---

## 7. CLI arguments

### Changed behavior

| Argument | sd-scripts | anima_lora |
|----------|-----------|------------|
| `--torch_compile` | Full-graph via Accelerator | Per-block via `compile_blocks` (native-shape flatten); never full-graph |
| `--dynamo_backend` | Always forwarded to Accelerator | Forwarded to `compile_blocks`; Accelerate's own dynamo stays `"NO"` |

---

## Summary: the compilation strategy

A DiT training loop has three sources of shape dynamism that trigger `torch.compile` recompilation:

1. Spatial resolution — different bucket sizes produce different `(T, H, W)` token counts.
2. Caption length — variable text encoder output lengths for cross-attention KV.
3. Batch size — trailing incomplete batches at epoch boundaries.

The fork bounds all three:

| Source | Solution | Files |
|--------|----------|-------|
| Spatial resolution | `compile_blocks` native-shape flatten (graph keys on token count) + `compile_dynamic_seq` (one graph per tier band) | `buckets.py`, `library/anima/models.py` |
| Caption length | Text encoder output zero-padded to a fixed 512-token KV (sink padding) | `library/anima/strategy.py`, `library/anima/models.py` |
| Batch size | Drop incomplete last batches | `library/datasets/base.py` |

`compile_blocks()` then compiles each block's `_forward` once per bounded band and reuses the kernels every step.
