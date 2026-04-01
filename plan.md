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

We can compute attention over **only real tokens** and adjust the softmax denominator by
adding `N_pad` (the number of padding positions). Result is mathematically identical.

## Architecture (Cross-Attention in each DiT Block)

```
Block.forward() [anima_models.py:1039]
  └─ self.cross_attn(normalized_x, attn_params, crossattn_emb)
       └─ Attention.forward() [anima_models.py:376]
            └─ compute_qkv(x, context=crossattn_emb)   # K,V from crossattn_emb
                 k = k_proj(crossattn_emb)  # [B, 512, H, D]  ← padding waste here
                 v = v_proj(crossattn_emb)  # [B, 512, H, D]  ← and here
            └─ attention.attention(qkv, attn_params)
                 └─ flex_attention or sdpa over full 512-length KV
```

## Implementation Plan

### Approach: Sink-Corrected Sparse Cross-Attention

Modify the cross-attention path so that:
1. K, V are only projected from **real** (non-padding) tokens
2. Softmax denominator is corrected by adding `N_pad` per sample
3. Everything else stays unchanged

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

For the cached-adapter path (`crossattn_emb` passed directly, no mask available), we need
to either:
- (a) Also cache the mask / seqlen alongside crossattn_emb, or
- (b) Detect seqlen from the cached embedding (find last non-zero row)

Option (a) is cleaner — cache `t5_attn_mask` sum as a scalar alongside `crossattn_emb`.

### Step 2: Trim KV before projection (optional, for memory savings)

In `Attention.compute_qkv`, when context has known seqlens and they're uniform in the batch
(or we pad to max-real-length), trim context before K/V projection:

```python
# Before: k = k_proj(context)          # [B, 512, inner_dim]
# After:  k = k_proj(context[:, :max_real_len])  # [B, ~80, inner_dim]
```

This saves KV projection compute and memory proportional to the trimming ratio.

**Complication**: Batch samples may have different real lengths. Options:
- Pad to `max(seqlens)` in the batch (still saves ~80% vs 512 in most cases)
- Use NestedTensor / jagged tensors (complex, not compile-friendly)

Padding to `max(seqlens)` in batch is simplest and gives most of the win.

### Step 3: Sink-corrected flex attention

This is the core change. In `attention.py`, for cross-attention with `crossattn_seqlens`:

```python
# attention.py, flex attention path
if Q_LEN != KV_LEN and attn_params.crossattn_seqlens is not None:
    seqlens = attn_params.crossattn_seqlens  # [B], real token counts
    n_pad = full_kv_len - seqlens            # number of padding positions

    # Flex attention over trimmed KV (only real tokens)
    # score_mod adds log-space correction for padding sinks:
    #   softmax_i = exp(s_i) / [Σ exp(s_j) + N_pad]
    #             = exp(s_i) / [exp(logsumexp(s)) + N_pad]
    #
    # Equivalently, subtract log(1 + N_pad · exp(-logsumexp(s))) from each score.
    # But that requires knowing logsumexp(s) which isn't available in score_mod.
    #
    # Alternative: use a two-pass approach or post-softmax correction.
```

**The cleanest approach** — post-softmax denominator correction:

After computing standard softmax attention over real tokens only:
```
attn_real = softmax(Q @ K_real^T / √d) @ V_real     # standard attention
           = Σ_r [ exp(s_r) / Σ_r exp(s_j) ] · v_r

# What we want:
attn_correct = Σ_r [ exp(s_r) / (Σ_r exp(s_j) + N_pad) ] · v_r
             = attn_real · [ Σ_r exp(s_j) / (Σ_r exp(s_j) + N_pad) ]
             = attn_real · correction_factor
```

The correction factor is `1 / (1 + N_pad / Σ exp(s_j))` which requires `logsumexp`.

**Implementation with flex attention `score_mod`:**

We can't easily do post-hoc correction with flex attention. But we CAN use a simpler trick:

**Append a single "virtual sink" token** to the KV sequence with key=0 and value=0, and use
`score_mod` to give it a log-weight of `log(N_pad)` so it has the effect of N_pad zero keys:

```python
# Append one virtual sink token to K and V (zeros)
k_with_sink = F.pad(k_real, (0, 0, 0, 1))  # [B, H, real_len+1, D], last is 0
v_with_sink = F.pad(v_real, (0, 0, 0, 1))  # same

n_pad = attn_params.n_pad_tokens  # [B]

def score_mod(score, b, h, q_idx, kv_idx):
    # For the virtual sink token (last position), replace score (which is 0)
    # with log(N_pad) so exp(score) = N_pad, simulating N_pad zero-key positions
    is_sink = (kv_idx == real_kv_len)  # last position
    return torch.where(is_sink, torch.log(n_pad[b].float()), score)

block_mask = create_block_mask(...)  # mask: kv_idx <= real_len (inclusive of sink)
x = flex_attention(q, k_with_sink, v_with_sink, score_mod=score_mod, block_mask=block_mask)
```

This is **mathematically exact** and works within flex attention's `score_mod` API.

### Step 4: Update the cached-adapter path

When `cache_llm_adapter_outputs=true`, the cached `crossattn_emb` is passed directly
without masks. We need to also store and load `crossattn_seqlens`:

1. In `strategy_anima.py` caching: save `crossattn_seqlen` (scalar per sample) in NPZ
2. In `load_outputs_npz`: load it alongside crossattn_emb
3. In `train.py`: pass it through to the model

### Summary of files to modify

| File | Change |
|------|--------|
| `library/attention.py` | Add virtual-sink score_mod path for cross-attention |
| `library/anima_models.py` | Return real crossattn_seqlens from _preprocess_text_embeds; trim context to max(seqlens) before passing to blocks |
| `library/strategy_anima.py` | Cache crossattn_seqlen in NPZ files |
| `train.py` | Pass crossattn_seqlens to model forward |

### Expected Speedup

For 512-length KV with ~60 real tokens on average:
- KV projection: **~8x** less compute (60 vs 512 tokens)
- Cross-attention matmul: Q·K^T goes from `[THW, 512]` to `[THW, 61]` — **~8x** less
- Net wall-clock: cross-attention is ~30-40% of block time → expect **~20-25%** block speedup
- Entire training step: blocks dominate → estimate **~15-20%** overall speedup

### Risks / Things to Verify

1. **Numerical equivalence**: The virtual-sink trick must be tested to produce identical
   outputs (within fp precision) vs the full-padding baseline
2. **torch.compile compatibility**: `score_mod` with per-batch `n_pad` must work with
   compile + flex attention — test for graph breaks
3. **Gradient correctness**: Ensure backward pass through the virtual sink is correct
   (gradients to the sink token should be zero since v_sink=0)
4. **Cache regeneration**: Existing NPZ caches need re-generation to include seqlens
   (or compute seqlens on-the-fly from the cached embeddings)
