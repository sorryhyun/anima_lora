# Proposal: Full-model `torch.compile` for training

## Motivation

Embedding inversion (`make invert`) demonstrated that compiling the entire Anima DiT as one graph with `torch.compile(model)` uses **12.5 GB** peak VRAM — less than per-block compile + block_swap=4 at **15.1 GB**. The compiler can plan memory reuse across all 28 blocks, freeing early activations before later blocks run.

Training currently uses per-block compile (`compile_blocks`), which compiles `block._forward` independently for each block. The compiler can only optimize within a single block's scope — it cannot schedule memory reuse across blocks.

Since the data pipeline already enforces fully static shapes:
- **Patches**: all bucket resolutions → `(H/16)*(W/16) ≈ 4096` tokens, zero-padded to exactly 4096
- **Text**: max-padded to 512 (or bucketed 64/128/256/512 with KV trim)
- **Batch**: fixed batch size per rank

...the full model is a single static graph with no shape-dependent control flow. This is the ideal case for `torch.compile`.

## What changes

Replace `compile_blocks()` with `torch.compile()` on the whole model, as a new mode for the no-gradient-checkpointing path:

```
torch_compile = true
gradient_checkpointing = false
blocks_to_swap = 0          # full model on GPU
compile_mode = "full"        # new: "full" or "blocks" (default "blocks")
```

### Implementation sketch

In `AnimaTrainer.load_target_model` (train.py ~378):

```python
if args.torch_compile:
    if args.compile_mode == "full":
        # Full-model compile — no grad ckpt, no block swap
        assert not gradient_checkpointing, "full compile is incompatible with gradient checkpointing"
        assert not self.is_swapping_blocks, "full compile is incompatible with block swap"
        model = torch.compile(model, backend=args.dynamo_backend, dynamic=False)
    else:
        model.compile_blocks(args.dynamo_backend)
```

LoRA monkey-patching happens before compile, so the patched forwards are captured in the compiled graph. No changes needed to LoRA modules.

## Why per-block compile exists

Per-block compile was the pragmatic choice because:

1. **Gradient checkpointing compatibility**: `compile_blocks` compiles `block._forward` (the inner computation), not `block.forward` (the checkpointing wrapper). `unsloth_checkpoint` is decorated with `@torch._disable_dynamo`, which would cause immediate graph breaks if the outer forward were compiled. Full-model compile bypasses this entirely — no checkpointing means no wrapper.

2. **Compile time**: Per-block compiles one function reused 28 times. Full-model unrolls the block loop into one graph — longer initial compilation.

3. **Block swap compatibility**: The `wait_for_block` / `submit_move_blocks` calls are side effects inside the forward loop. Not compilable.

Full-model compile is only viable when both gradient checkpointing and block swap are off.

## Expected benefits

| | Per-block compile | Full-model compile |
|---|---|---|
| Compiler scope | 1 block | 28 blocks + patch_embed + final_layer |
| Cross-block memory reuse | No | Yes — compiler can free block N's activations before block N+K |
| Activation memory | ~28× one block's activations | Compiler-scheduled, potentially much less |
| Compile time | Fast (1 block, reused 28×) | Slower (one large graph) |
| Grad ckpt compatible | Yes | No |
| Block swap compatible | Yes | No |

Based on the inversion experiment, full-model compile could reduce training peak VRAM by **~2-3 GB** at the cost of a longer first step. This would put the "compile, no grad ckpt" row in the benchmark table closer to the grad-ckpt rows in VRAM while keeping the speed advantage.

## Risks and unknowns

1. **LoRA graph breaks**: LoRA modules are monkey-patched before compile, so dynamo traces through them. Standard LoRA/DoRA should be clean. T-LoRA's timestep-dependent rank masking and HydraLoRA's router dispatch may cause graph breaks — needs testing. Worst case: dynamo falls back to eager for those subgraphs, still functional but less optimized.

2. **Compilation time**: Full-model graph with 28 blocks + LoRA modules will take longer to compile (estimate: 2-5 minutes vs ~30s for per-block). Acceptable for multi-epoch training but painful for quick iterations. Could mitigate with `torch._inductor.config.cache_dir` persistence.

3. **Inductor memory planning**: The VRAM savings depend on Inductor's memory planner actually scheduling cross-block reuse. This is the expected behavior for a single static graph but hasn't been verified at this scale with LoRA.

4. **Recompilation on LoRA changes**: If LoRA multipliers or masks change mid-training (e.g., T-LoRA schedule), this could trigger recompilation. Need to ensure these are captured as tensor values, not Python scalars that become guards.

## Validation plan

1. Run the existing benchmark (rank=32, lr=5e-5, batch_size=2, 182 steps) with `compile_mode=full`, no grad ckpt, no block swap
2. Compare peak VRAM and wall time against the existing "FA2 + compile - grad ckpt" row (15.2 GB, 7:07)
3. Compare validation loss to ensure no numerical divergence
4. Test with LoRA, DoRA, and T-LoRA to identify graph break issues
