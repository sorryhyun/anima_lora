# Efficient Cross-Attention for Padding Tokens

## Problem

The pretrained model uses 512-length cross-attention context. Padding positions (beyond
actual text tokens) are zeroed out but **must remain in softmax** — they act as attention
sinks. Masking them out produces black images.

Typical captions use 30–80 tokens, so ~85% of cross-attention KV positions are zero-padding.
Every DiT block computes full Q·K^T, softmax, and weighted-sum over all 512 positions.

## Key Insight

Zero-key padding positions have a **closed-form contribution** to attention. For any query
q_i and zero key k_pad = 0:

```
score(q_i, k_pad) = q_i · 0 / sqrt(d) = 0
exp(score) = exp(0) = 1
```

So each padding position contributes exactly `1` to the softmax denominator and `v_pad`
(which is also zero, since the V projection of a zero embedding is zero) to the numerator.

This means:

```
attn_out = Σ_real[ softmax_weight_r · v_r ]

where softmax_weight_r = exp(q · k_r / √d) / [ Σ_real exp(q · k_j / √d) + N_pad ]
                                                                              ^^^^^
                                              this is the only difference from standard attn
```

We can compute attention over **only real tokens** and correct the softmax denominator
by scaling with the LSE (logsumexp). Result is mathematically identical.

## Architecture (Cross-Attention in each DiT Block)

```
Block._forward() [anima_models.py:1104]
  └─ self.cross_attn(normalized_x, attn_params, crossattn_emb)
       └─ MultiHeadAttention.forward() [anima_models.py:373]
            └─ compute_qkv(x, context=crossattn_emb)   # K,V from crossattn_emb
                 k = k_proj(crossattn_emb)  # [B, 512, H, D]  ← padding waste here
                 v = v_proj(crossattn_emb)  # [B, 512, H, D]  ← and here
            └─ attention.attention(qkv, attn_params)
                 └─ flash4 / flash / flex / sdpa over full 512-length KV
```

## Implementation Plan

### Approach: LSE-Corrected Sparse Cross-Attention (Flash Attention 4)

Flash Attention returns `(out, lse)` where `lse` is the log-sum-exp of attention scores.
We use this to apply a post-hoc correction that exactly accounts for removed padding tokens.

**Math:**

Standard attention over real tokens only gives:
```
out_real = softmax(Q @ K_real^T / √d) @ V_real
         = Σ_r [ exp(s_r) / Z_real ] · v_r           where Z_real = Σ_r exp(s_j)

lse_real = log(Z_real)                                 ← returned by flash attention
```

What we want (pretrained behavior, including padding sinks):
```
out_correct = Σ_r [ exp(s_r) / (Z_real + N_pad) ] · v_r
            = out_real · Z_real / (Z_real + N_pad)
            = out_real · 1 / (1 + N_pad / Z_real)
            = out_real · 1 / (1 + N_pad · exp(-lse_real))
            = out_real · sigmoid(lse_real - log(N_pad))
```

One elementwise sigmoid + multiply. Exact.

## torch.compile Compatibility

### Constraint: Static KV shapes for compiled blocks

When `static_token_count` is set, each DiT block is individually compiled via
`torch.compile(block, backend="inductor")`. All inputs must have **static shapes** to
avoid Dynamo guard recompilation. Currently cross-attention KV is always `[B, 512, H, D]`.

Trimming KV to the exact `max(seqlens)` per batch would produce variable shapes
(47, 63, 81...) → guard fire → recompile every batch. **This defeats the compile strategy.**

### Solution: Bucketed trim lengths

Round `max(seqlens)` up to a fixed set of KV buckets:

```python
KV_BUCKETS = [64, 128, 256, 512]  # compile sees at most 4 distinct shapes

def bucket_kv_len(max_real_len: int) -> int:
    for b in KV_BUCKETS:
        if max_real_len <= b:
            return b
    return 512  # fallback: no trimming
```

Typical captions (30–80 tokens) → **128 bucket** → still **4x savings** over 512.
After warmup, Dynamo caches all 4 variants and never recompiles again.

### FA4 autograd.Function is compile-safe

FA4's `FlashAttnFunc` is a `torch.autograd.Function`. torch.compile treats `.apply()`
as an opaque node — no graph break. The post-hoc correction (`sigmoid` + multiply) is
standard PyTorch ops — fully traceable by Inductor.

Compilation chain inside a compiled block:
1. `FlashAttnFunc.apply(q, k, v, ..., return_lse=True)` → opaque autograd node
2. `correction = sigmoid(lse - log(n_pad))` → traced, fused by Inductor
3. `out *= correction` → traced, fused
4. Backward: autograd handles the chain; FA4 backward receives `(dout, dlse)` correctly

No graph breaks. No recompilation (with bucketed KV lengths).

### return_lse and gradient correctness

FA4's backward checks `ctx.return_lse` to decide whether to pass `dlse` to the kernel.
Since we compute `out * sigmoid(lse - ...)`, gradients flow through both `out` and `lse`.

- With `return_lse=True`: exact gradients, `dlse` propagated to Q/K/V
- With `return_lse=False`: `dlse` is dropped — loses second-order gradient term

The dropped term is small (correction factor is near-constant across queries for a given
caption length), so `return_lse=False` is a viable approximation for LoRA training.
**Recommend `return_lse=True` for correctness** — the kernel supports it, cost is minimal.

## Steps

### Step 1: Pass `crossattn_seqlens` through to attention

Currently `_preprocess_text_embeds` returns `(context, None)` — always discards seqlens.

**Change**: Compute and return actual seqlens.

```python
# anima_models.py, _preprocess_text_embeds()
# Instead of:
return context, None

# Return:
crossattn_seqlens = crossattn_mask.sum(dim=-1).to(torch.int32)  # [B]
return context, crossattn_seqlens
```

Add to `AttentionParams` dataclass:
```python
crossattn_seqlens: Optional[torch.Tensor] = None   # [B] real token counts for cross-attn
```

Populate in `forward_mini_train_dit()`:
```python
attn_params.crossattn_seqlens = crossattn_seqlens
```

### Step 2: Trim KV before projection (bucketed)

In `MultiHeadAttention.compute_qkv`, when processing cross-attention context, trim to
the nearest KV bucket above `max(seqlens)`:

```python
KV_BUCKETS = [64, 128, 256, 512]

# In compute_qkv, when context is crossattn_emb:
if attn_params.crossattn_seqlens is not None and context is not None:
    max_real_len = attn_params.crossattn_seqlens.max().item()
    # Round up to bucket boundary for compile-stable shapes
    trim_len = next((b for b in KV_BUCKETS if b >= max_real_len), context.shape[1])
    if trim_len < context.shape[1]:
        context = context[:, :trim_len]

k = self.k_proj(context)   # now [B, 128, H, D] typically
v = self.v_proj(context)   # same
```

This saves KV projection compute (~4x for 128 bucket vs 512) while keeping shapes
stable across batches with similar caption lengths.

### Step 3: LSE-corrected flash attention for cross-attention

Core change in `attention.py`. For flash4 cross-attention with trimmed KV:

```python
# attention.py — flash4 path
if attn_params.crossattn_seqlens is not None and Q_LEN != KV_LEN:
    full_kv_len = 512  # original padded length (from config or attn_params)
    n_pad = full_kv_len - KV_LEN  # scalar: padding tokens removed by trimming
    # Note: remaining positions between max(seqlens) and trim_len are still
    # zero-padded, so their exp(0)=1 contribution is already in the FA4 output.
    # We only correct for the positions we actually trimmed away.

    out, lse = _flash_attn_4_func_raw(q, k, v, softmax_scale=scale, return_lse=True)
    # lse: [B, H, Q_LEN]

    if n_pad > 0:
        # Sink correction: account for trimmed zero-key positions
        correction = torch.sigmoid(lse - math.log(n_pad))  # [B, H, Q_LEN]
        # out is [B, Q_LEN, H, D] for FA4 — need to align correction dims
        out = out * correction.transpose(1, 2).unsqueeze(-1)

    x = out
```

**Important**: `n_pad` here is `512 - trim_len` (a compile-time constant per bucket),
NOT `512 - max(seqlens)`. The positions between `max(seqlens)` and `trim_len` are still
zero-padded inside the trimmed KV — FA4 handles them normally. We only correct for
the positions we removed entirely (trim_len to 512).

This means `n_pad` is one of {448, 384, 256, 0} for buckets {64, 128, 256, 512} —
a small set of constants. `math.log(n_pad)` is computed once, not per-element.

**Fallback**: For backends without LSE (sdpa, xformers, sageattn), skip trimming and use
the full 512-length KV as before. No regression — just no speedup for those modes.

### Step 4: Expose LSE from flash wrappers

Current wrappers discard LSE:
```python
# attention.py lines 22-28
def flash_attn_4_func(*args, **kwargs):
    out, _lse = _flash_attn_4_func_raw(*args, **kwargs)
    return out
```

For the cross-attention path, call `_flash_attn_4_func_raw` directly with
`return_lse=True` to get `(out, lse)`. No need for a new wrapper — the raw function
is already imported.

### Step 5: Update the cached-adapter path

When `cache_llm_adapter_outputs=true`, cached `crossattn_emb` has no mask info.

**Option A** (preferred): Cache `crossattn_seqlen` in NPZ alongside `crossattn_emb`.
- In `strategy_anima.py` save path: `np.savez(..., crossattn_seqlen=seqlen)`
- In `load_outputs_npz`: load and return it

**Option B** (backward-compat fallback): Derive seqlens from cached embeddings on-the-fly:
```python
# Padding is zeroed, so:
crossattn_seqlens = (crossattn_emb.abs().sum(-1) > 0).sum(-1).to(torch.int32)  # [B]
```

Implement both: try loading from NPZ, fall back to Option B for old caches.

### Step 6: Plumb seqlens through train.py

```python
# train.py, cached adapter path (currently passes no mask):
model_pred = anima(
    noisy_model_input,
    timesteps,
    crossattn_emb,
    padding_mask=padding_mask,
    crossattn_seqlens=crossattn_seqlens,  # NEW
)
```

## Summary of files to modify

| File | Change |
|------|--------|
| `library/attention.py` | Add LSE-corrected cross-attention path for flash/flash4 (call raw func with `return_lse=True`, apply sigmoid correction) |
| `library/anima_models.py` | Return real crossattn_seqlens from `_preprocess_text_embeds`; add `crossattn_seqlens` field to `AttentionParams`; bucketed KV trim in `compute_qkv`; pass seqlens to attn_params |
| `library/strategy_anima.py` | Cache `crossattn_seqlen` in NPZ files; load with fallback |
| `train.py` | Pass `crossattn_seqlens` to model forward (cached adapter path) |

## Expected Speedup

For 512-length KV with ~60 real tokens on average (128 bucket):
- KV projection: **~4x** less compute (128 vs 512 tokens)
- Cross-attention matmul: Q·K^T goes from `[THW, 512]` to `[THW, 128]` — **~4x** less
- Sink correction: negligible (one sigmoid + multiply, fused by Inductor)
- Net wall-clock: cross-attention is ~30-40% of block time → expect **~15-20%** block speedup
- Entire training step: blocks dominate → estimate **~10-15%** overall speedup
- With bucket 64 (very short captions): up to **~20%** overall

## Risks / Things to Verify

1. **Numerical equivalence**: Test that `out * sigmoid(lse - log(N_pad))` matches full-padding
   attention output within bf16 precision. The sigmoid can lose precision when `lse` is very
   large or very small — check edge cases.

2. **LSE tensor layout**: Verify that FA4's LSE output shape is `[B, H, Q_LEN]` (not
   `[B, Q_LEN, H]`). Check whether it's in log base e or log base 2 (FA2 uses log2).
   If log2: use `1 / (1 + N_pad * 2^(-lse))` instead of sigmoid.

3. **torch.compile recompilation**: Verify that bucketed KV lengths (4 variants) actually
   stabilize after warmup. Monitor `torch._dynamo.utils.counters["frames"]["ok"]` to
   confirm no unexpected recompiles during training.

4. **Gradient correctness**: With `return_lse=True`, FA4's backward receives non-None `dlse`.
   Verify this works correctly with the backward kernel — the code path exists in
   `FlashAttnFunc.backward` but needs testing with actual gradient flow through the
   sigmoid correction.

5. **Cache regeneration**: Existing NPZ caches lack seqlens. The on-the-fly fallback
   (`abs().sum(-1) > 0`) handles this, but regenerating caches is cleaner long-term.

6. **Existing flex crossattn_block_mask**: The current flex path (lines 1626-1647) hard-masks
   padding positions (sets scores to -inf), which is NOT equivalent to the pretrained behavior
   (padding contributes exp(0)=1). Check if this has been causing issues in flex mode. If so,
   the flex path also needs sink correction (deferred — virtual sink token approach).

7. **LoRA on K/V projections**: LoRA adapters on `cross_attn.k_proj` / `cross_attn.v_proj`
   will now process trimmed context (~128 tokens instead of 512). This is correct — the
   adapter should only see real tokens. No change to LoRA logic needed.
