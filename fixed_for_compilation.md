# torch.compile Optimization Log

Goal: maximize torch.compile efficiency on 5060 Ti 16 GB.

## Results

| Configuration | Peak VRAM | Total Time | 2nd Epoch | Train Loss | Val Loss |
|---|---|---|---|---|---|
| FA2 + gradient checkpointing | 7.0 GB | 14:51 | 7:26 | 0.092 | 0.212 |
| Flex + compile + no gradient checkpointing | 15.2 GB | 8:19 | 4:09 | 0.090 | 0.201 |

~1.8x wall-clock speedup, slightly better loss.

## Applied Optimizations

### 1. `dynamic=False` in `compile_blocks()`

One-line change in `anima_models.py`. Since `static_token_count` guarantees
constant shapes and `bsz=1`, `dynamic=False` lets inductor generate
shape-specialized kernels with no guard overhead.

### 2. `attn_mode="flex"` for full compile fusion

Flash-attn uses `@torch.compiler.disable` which causes graph breaks at every
attention call. `flex_attention` traces through compile, giving one continuous
graph per block.

### 3. Fix flex attention recompilation storm

Two dynamo guard failures were causing combinatorial recompilation
(2 requires_grad states x 5 bucket seq_lens = 10, exceeding the recompile
limit of 8 and falling back to eager):

**`_sa_seq_len` closure guard** — `create_block_mask` stores `mask_mod` in the
`BlockMask`. Dynamo inspects the closure and creates a guard on the exact
Python int value (`== 4056`). Every different bucket seq_len triggers a
recompile. Fix: convert `_sa_seq_len` to a `torch.tensor` so dynamo tracks it
symbolically instead of guarding on the concrete value.

**`requires_grad` mismatch across blocks** — All blocks share the same
`_forward` code object and thus share dynamo's code cache. Block 0's input from
the frozen patch_embed has `requires_grad=False`; blocks 1+ get LoRA-enhanced
outputs with `requires_grad=True`. Fix: `x_B_T_H_W_D.requires_grad_()` at the
top of `_forward` to normalize all blocks to `requires_grad=True`.

## Implemented but Not Yet Tested

### `fp8_base_unet`

`FP8Linear` subclass + `_FP8LinearFunc` custom autograd (saves fp8 weight for
backward instead of transient bf16 copy) + `quantize_to_fp8()` utility.
Wired into `train.py:load_unet_lazily`, runs before LoRA `apply_to`.

**Known issue**: OOM during first compiled forward. Inductor schedules all
weight upcasts (`weight_fp8.to(bf16)`) simultaneously, creating ~130 MB of
transient bf16 copies per block. On 5060 Ti with only ~30-60 MB headroom,
this is fatal.

**Possible fixes** (in order of preference):

1. **Increase headroom first** — apply `static_token_count=3840` and/or
   increase `blocks_to_swap` before re-testing. The fp8 savings (~1.3 GB) far
   exceed the transient cost, but the first compiled step needs enough
   headroom for all transient copies.

2. **`torch.cuda.empty_cache()` before first step** — free fragmented reserved
   memory.

3. **Disable compile for fp8 linears** — `@torch.compiler.disable` on
   `_FP8LinearFunc.apply` so upcasts happen eagerly (freed one-at-a-time),
   at the cost of per-linear graph breaks.

4. **Native fp8 matmul via `torch._scaled_mm`** — SM120 supports fp8 matmul
   natively. Avoids upcasting entirely but requires scale management.

## Potential Future Work

### Reduce `static_token_count` to 3840

~230 MB activation savings + 12% self-attention compute reduction.
Bucket table change. Tradeoff: no 1024x1024 square bucket (nearest 960x1024).
