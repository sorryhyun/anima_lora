# Changes from sd-scripts for torch.compile / dynamo

This document catalogues every change made to the `anima_lora` fork (relative to the original `sd-scripts` repo) that enables or supports `torch.compile` and PyTorch dynamo. Changes are grouped by file and subsystem.

---

## 1. Attention dispatch (`networks/attention_dispatch.py`)

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

sd-scripts: No FA4 support at all.

> Note: FA4 is currently not the default attention backend — see [`fa4.md`](fa4.md) for why. The code paths remain in place for re-enabling.

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

### 1.5 LSE sink correction for trimmed cross-attention (flash4 — removed)

The KV-trim + LSE-sigmoid correction path was bundled with FA4 and depended on FA4's `return_lse`. Both FA4 and the trim plumbing were removed (the `crossattn_full_len` field and the `_KV_BUCKETS` constant are gone as of 2026-05-20). See `docs/optimizations/fa4.md` for the postmortem; reviving it now means reimplementing the trim, not uncommenting it.

When the path was active, zero-padded KV positions were trimmed and the softmax denominator restored via:

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

### 2.4 KV bucket trimming constants (removed)

`_KV_BUCKETS` trimmed cross-attention KV sequences to the smallest fitting bucket, capping `torch.compile` shape variants. It was tied to the FA4-only trim path and was removed along with it (2026-05-20). Cross-attention now runs the full 512-length KV under FA2. See `docs/optimizations/fa4.md`.

### 2.5 `compile_blocks(backend="inductor")` — the single switch

`compile_blocks` is the one call that turns on `torch.compile`. It does two coupled things and raises the dynamo cache-size budget itself:

1. Native-shape flattening (`self._native_flatten = True`). The forward flattens each bucket's patch grid `(B, T, H, W, D)` to a fake-5D `(B, 1, seq_len, 1, D)` shape (`unflatten`-restored after the block loop). This keys the block graph on token count alone — the shipped `CONSTANT_TOKEN_BUCKETS` collapses to two token-count families (4032 and 4200) — instead of guarding `H` and `W` separately (one graph per resolution, 24 buckets). No padding, so flash self-attention sees no padded tokens. Bit-exact to the eager 5D path; eager (uncompiled) forwards leave the flag `False` and skip the reshape.

2. Per-block compile. Compiles each block's `_forward` method:
   ```python
   for block in self.blocks:
       block._forward = torch.compile(block._forward, backend=backend, dynamic=False)
   ```
   **Critical:** compiles `_forward` (the actual attention/MLP), NOT `forward` (the checkpointing wrapper). The gradient checkpointing decorator (`unsloth_checkpoint`) uses `@torch._disable_dynamo`, which would cause an immediate graph break if `forward` itself were compiled — dynamo compiles nothing useful but still checks shape guards, causing recompile storms.

The cache budget is `cache_size_limit = max(current, 2*n + 8)` where `n` is the number of token-count families (2): the `2*` covers fwd+bwd sharing the one `_forward` bytecode, the `+8` covers requires_grad / stride specializations. The `max()` lets a multi-resolution caller (e.g. a distill loop whose downsampled stages produce more distinct shapes) pre-raise the limit without `compile_blocks` lowering it.

There is no padded mode anymore.

### 2.6 Dynamic-seq marks disable inductor mix-order reduction

When `compile_dynamic_seq` is active (free-fit — §3.1 — auto-enables it, and the
bespoke distill loops force it via `ensure_dynamic_seq_for_freefit`),
`compile_blocks` marks only the seq axis dynamic and bounds it to the tier's
`seq_range` via a strict `mark_dynamic`. That strict bound collides with an
inductor fusion pass:

```python
# compile_blocks, when dynamic-seq marks are active
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
reverted and the fusion still recorded the guard. This shipped broken in
v1.14.0 (same commit as adaln default-on) and crashed community `make lora`
runs at step 0 under the grad-ckpt presets. The pin sets the entry's
`.default` too (same pattern as `pin_dynamo_limit` for `recompile_limit`), which
every context reads. Poisoned per-signature cache dirs from crashed runs
self-heal: the config is part of the FxGraphCache key, so stale entries miss.
See also the caveat on `isolate_compile_cache` (stale-guard poisoning across
cache reuse). Discovered fixing the adaln training path (2026-07-15); ContextVar
regression root-caused 2026-07-17 — `docs/methods/adaln.md` §Path 2.

> sd-scripts: no dynamic-seq path, no free-fit, no per-block compile — none
> of this machinery exists upstream. The legacy `set_static_token_count(count, pad=True)` path zero-padded every bucket up to a single shape, but it leaked padded tokens into flash self-attention (AdaLN shift + Q/K/V bias make zero-input padded rows emit non-trivial K/V; up to ~6.5% rel-L2 on the 4032 buckets) and couldn't even run the shipped table (4200 > the legacy 4096 target → truncation). It was removed 2026-05-24 along with `compile_core` / `--compile_mode full`, `static_token_count`, `static_pad`, and the flex self-attn pad-mask.

---

## 3. Datasets (`library/datasets/`)

### 3.1 Constant-token buckets (`buckets.py`)

`CONSTANT_TOKEN_BUCKETS` — 24 predefined `(W, H)` resolutions grouped into two token-count families, 4032 (= 63·64) and 4200 (= 60·70). Each resolution *exactly* fills its family's count, so there is zero intra-bucket padding by construction. Native shapes are the default mode: every forward runs at its real token count, so `compile_blocks`' flatten makes `torch.compile` trace one block graph per distinct count — just two for this table.

> Free-fit (opt-in `freefit=true`) is the alternative: keep native aspect ratio and land the token count *anywhere* inside a tier's band rather than snapping to these discrete buckets (`freefit_bucket`; `_archive/proposals/free_aspect_token_band_resize.md`). The off-table counts would explode the static graph count, so free-fit requires `compile_dynamic_seq` (auto-enabled when `freefit` + `torch_compile`), which marks only the seq axis dynamic and bounds it to the tier's `seq_range` — collapsing the whole band to one graph. Constant-token bucketing stays the default and stays frozen (the frozen top-5 aspect set `DCW_ASPECT_BUCKETS`, consumed by CNS calibration + mod-distill, is drawn from it); the two modes coexist per-dataset via the `freefit` flag.

```python
CONSTANT_TOKEN_BUCKETS = [
    # ---- 4032-token family (63*64) ----
    (1008, 1024),   # 63 x 64, ar 0.98 (nearest to square)
    (1024, 1008),   #          ar 1.02
    (896, 1152),    # 56 x 72, ar 0.78
    # ... 9 more landscape/portrait pairs
    (2016, 512),    # 32 x 126, ar 3.94
    # ---- 4200-token family (60*70) ----
    (960, 1120),    # 60 x 70, ar 0.86
    # ... 11 more landscape/portrait pairs
    (1920, 560),    # 35 x 120, ar 3.43
]
```

Two families instead of one because a single count's divisors near √N are sparse (4032 alone jumps aspect 1.29→1.75); interleaving 4032 and 4200 densely covers aspect space at the cost of one extra graph. Note this diverges from `DCW_ASPECT_BUCKETS`: the 832×1248 / 1248×832 HD pair (4056 tokens) is no longer a training bucket.

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

### 4.4 `constant_token_buckets` plumbed to dataset config

```python
constant_token_buckets=True,                          # native bucketing (default)
freefit=bool(getattr(args, "freefit", False)),        # opt-in free-aspect override
```

Passed through `library/config/` to `BucketManager.make_buckets()`. When `freefit=True` the bucket set becomes the actual on-disk cached `(W,H)` instead of the constant-token table (training auto-enables `compile_dynamic_seq`).

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

### 5.2 ~~Memory-saving down-projection autograd~~ (REMOVED 2026-06-10)

The custom down-projection `autograd.Function` (`custom_autograd.py`) and the
fp32-bottleneck matmul policy it serviced were removed. The training forward
ran under `accelerator.autocast()` (default `mixed_precision="bf16"`), and
autocast re-casts fp32 `F.linear` inputs back to bf16 — so the fp32 matmuls
never actually executed, the "fp32 activation retained for backward" the
Function was built to avoid was never retained, and the whole mechanism was
dead weight plus cast traffic (`x.float()` materialized a fp32 copy of every
adapted Linear's input that autocast immediately re-rounded — up to ~24%
module overhead on wide-input Linears). Measured in the `lora_fp32_bottleneck` bench (since removed):

* live-autocast path vs explicit bf16 GEMMs: bit-identical forward (max abs diff 0.0);
* cuBLAS bf16 GEMMs accumulate in fp32 internally, so the bf16 output sits at
  the rounding floor (one caveat: `allow_bf16_reduced_precision_reduction`,
  default True, costs ~1.4× error at k=8192);
* 200-step Adam probe: final ΔW deviation vs an fp64 run indistinguishable
  across fp32-bottleneck / bf16 / live-autocast regimes.

The training forwards now compute the rank GEMMs directly in the activation
dtype (`weight.to(x.dtype)` at the matmul boundary — the same cast autocast
performed, now explicit and autocast-independent). Inference paths
(HydraLoRAModule at eval, `ChimeraHydraInferenceModule`, EasyControl KV
prefill) kept their historical fp32 compute since the inference engine runs
without autocast. Regression tests: `tests/test_lora_dtype_policy.py`
(bitwise legacy parity under autocast + dtype honesty).

`use_custom_down_autograd` is still accepted everywhere it used to be (TOML
allowlist, factory, EasyControl, turbo CLI) but is a logged no-op, so old
snapshot TOMLs replay cleanly.

Post-removal addendum (2026-06-10, same day): the Function was numerically
dead weight but NOT memory-dead. Under `torch.compile` an `autograd.Function`
is traced as a HOP that pins the saved-for-backward set to its
`ctx.save_for_backward` choice ({x, weight}, casts recomputed in backward).
With plain traceable ops, AOT's min-cut partitioner chose to save ~0.8 GB more
intermediates per step — first-step OOM on a 16 GB card at 4200 tokens without
gradient checkpointing. The generic replacement is
`activation_memory_budget = 0.85` (base.toml → `torch._functorch.config.
activation_memory_budget`, set in `train.py` before `compile_blocks`):
measured identical step time (1.02 vs 1.01 s/it) and identical peak (~15.2 GB
total) to the custom-Function era on the 26-step probe. It is auto-skipped
under `gradient_checkpointing` — repartitioning makes checkpoint's recompute
pass select a different graph than forward (`CheckpointError`, torch #166926),
and ckpt already minimizes saved activations. Lesson: *numerically-inert ≠
memory-inert* — removing a custom Function changes partitioning even when its
math traces identically.

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

### Changed behavior

| Argument | sd-scripts | anima_lora |
|----------|-----------|------------|
| `--torch_compile` | Full-graph via Accelerator | Per-block via `compile_blocks` (native-shape flatten); never full-graph |
| `--dynamo_backend` | Always forwarded to Accelerator | Forwarded to `compile_blocks`; Accelerate's own dynamo stays `"NO"` |

---

## Summary: the compilation strategy

The key insight is that a DiT training loop has three sources of shape dynamism that trigger `torch.compile` recompilation:

1. Spatial resolution — different bucket sizes produce different `(T, H, W)` token counts.
2. Caption length — variable text encoder output lengths for cross-attention KV.
3. Batch size — trailing incomplete batches at epoch boundaries.

The fork eliminates all three:

| Source | Solution | Files |
|--------|----------|-------|
| Spatial resolution | `CONSTANT_TOKEN_BUCKETS` + `compile_blocks` native-shape flatten (graph keys on token count) | `buckets.py`, `library/anima/models.py` |
| Caption length | Text encoder output zero-padded to a fixed 512-token KV (sink padding) | `library/anima/strategy.py`, `library/anima/models.py` |
| Batch size | Drop incomplete last batches | `library/datasets/base.py` |

With shapes stabilized, `compile_blocks()` compiles each block's `_forward` with `dynamic=False` — the inductor backend generates optimized kernels once per token-count family (two) and reuses them for every step.
