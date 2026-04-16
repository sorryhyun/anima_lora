# Flash Attention 4 — removed

This doc records why `attn_mode = "flash4"` and `trim_crossattn_kv` were removed from the training pipeline. The short version: **FA4 ran slower than FA2 on our targets**, and the cross-attention KV-trim trick that justified FA4's ergonomic complexity only worked under FA4, so it went with it.

## What was there

Two bundled features that shared a single dependency on FA4's `return_lse=True` path:

1. **FA4 forward kernel.** `flash_attn.cute.flash_attn_func` from a patched SM120 fork (`sorryhyun/flash-attention-sm120-fix`, based on `sisgrad`'s `dz/sm120_tma_optimized` branch). The stock `flash-attn` v4 does not ship SM120 (consumer Blackwell) kernels, so we pulled in a fork to get FA4 running on RTX 50-series cards at all.
2. **Cross-attention KV trim (`trim_crossattn_kv`).** Qwen3 text encoder outputs are zero-padded to 512 tokens, but real captions only fill 30–80. The trim path sliced the padded KV down to a bucketed length (`[64, 128, 256, 512]`), ran attention on the trimmed tensor, and then used FA4's returned log-sum-exp to apply a closed-form correction that rebuilt the contribution of the removed zero keys (they act as attention sinks in the pretrained softmax — dropping them without correction produces black images):

   ```
   out_corrected = out_trimmed * sigmoid(lse - log(N_pad))
   ```

   The correction is exact, not an approximation. It relied on FA4's LSE return, which FA2's Python interface doesn't expose, so the trim path was gated on `attn_mode == "flash4"`.

## Why it was removed

### FA4 was slower than FA2 in practice

On a single RTX 5060 Ti 16GB with LoRA rank 32, batch 2, 182 steps, FA4 was effectively a wash with FA2 — and often a regression once compile/static-token effects were accounted for. Representative numbers from the old benchmark table (before removal):

| Configuration | 2nd Epoch |
|---|---|
| FA2 + compile (static tokens) | 5:01 |
| FA4 + compile (static tokens) | 5:15 |

The expected advantage from FA4 (the fused CUTLASS/TVM kernel) did not materialize on consumer Blackwell. A few things contributed:

- **SM120 kernel maturity.** The SM120 port is a community fork, not a tuned upstream kernel. The TMA-optimized path worked, but it wasn't meaningfully ahead of FA2 in forward throughput at our shapes (batch 2, 4096 image tokens, 64–512 text tokens).
- **Compile interaction.** FA4's CUTLASS kernel accesses raw data pointers via DLPack, which fake-tensor tracing cannot see through. We had to wrap the kernel in `@torch.compiler.disable` and take graph breaks around every attention call. The surrounding fused region still got compiled, but the graph breaks added up across 32 blocks × 2 attention calls per block × every step.
- **Shape-dependent branching.** The KV trim path had a separate code branch for the LSE correction, which caused torch.compile recompilations at bucketed KV shapes (64/128/256/512) until we flattened the branch. Even after the fix, the compile graph was more fragile than the FA2 path.

The KV trim itself *did* reduce FLOPs — roughly 4× less cross-attention compute on short captions. But cross-attention is a small fraction of total compute (the 4096-token self-attention dominates), so the end-to-end win was small, and what little win there was got eaten by the slower FA4 forward and the compile friction.

### The trim was coupled to FA4 and couldn't be salvaged

We could in principle port the LSE-sigmoid correction to FA2 by monkey-patching `_flash_attn_forward` to return LSE. We chose not to:

- FA2's Python interface doesn't return LSE, and adding it means poking at the C++ bindings.
- Without FA4, the remaining speedup is tiny (cross-attention isn't the bottleneck).
- The trim path complicates every caller (`train.py` has to compute `max_crossattn_seqlen`, pass `crossattn_seqlens`, and handle postfix/prefix offset accounting) for a win we can no longer measure.

Keeping it as a flag that everyone sets to `true` in configs, with an opaque code path that depends on a dead backend, wasn't worth the maintenance cost.

## What's left

- **Default attention is FA2** (`attn_mode = "flash"`), via the upstream `flash-attn` 2.x wheel. This is what every provided config now uses.
- **Other backends still work:** `torch` (SDPA), `flex` (PyTorch flex attention), `sageattn`, `xformers`. None of them use trimming; all run the full 512-length cross-attention KV with attention-sink padding intact.
- **`trim_crossattn_kv` is a no-op.** The config flag still parses (and all stock configs set it to `false`), but the trim code in `library/anima_models.py` is commented out. If the flag is `true`, `train.py` will still compute `crossattn_seqlens` and pass it into the model, but the model will ignore it unless you're on `attn_mode="flex"` — in which case it builds a BlockMask that *masks out* the padding instead of treating it as a sink, which regresses quality. Leave it at `false`.

## If you want to bring FA4 back

Everything is commented rather than deleted, so re-enabling is mechanical:

1. **Dependency.** Uncomment the `flash-attn-4` line in `pyproject.toml`. On consumer Blackwell you still need the SM120 fork — the local `flash-attention-sm120/` source tree is kept for reference.
2. **Attention dispatch.** Uncomment the FA4 import block and the `flash4` branch in `networks/attention.py`. Both are bracketed with `# Flash Attention 4 ... is not supported yet` comments.
3. **Train path.** In `train.py`, `load_unet_lazily` currently raises on `attn_mode == "flash4"`; replace that with the original check against `_flash_attn_4_func_raw is not None`. Also restore the `args.fp8_base_unet` call (if you want fp8 too).
4. **KV trim.** Uncomment the `trim_crossattn_kv` branch in `library/anima_models.py` (the `if ... attn_mode == "flash4"` block). Flip `trim_crossattn_kv = true` in whichever configs you want the trim in, and switch those configs to `attn_mode = "flash4"`.
5. **GUI / configs.** Add `"flash4"` back to `_ATTN_MODES` in `gui/__init__.py` and the `--attn_mode` choices in `inference.py` and `library/anima_train_utils.py`. The FA4 VRAM presets (`FA4 8GB VRAM` / `FA4 16GB VRAM`) are also commented out in `gui/__init__.py`.

Before doing any of that, benchmark FA4 on your own hardware. If the kernel has matured upstream — or if SM120 TMA has landed in stock `flash-attn` — it may be worth reviving. On our targets in 2026-04 it wasn't.
