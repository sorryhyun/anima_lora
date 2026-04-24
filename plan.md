# torch.compile + CUDAGraph — plan

## Background: what's been done

Dynamo recompilation is down to ~0 on the hot path. `TORCH_LOGS=recompiles` trial (22 training steps, pre-validation) shows 0/0/0/0 across all original guard classes:

- `_static_pad_info[*]` guards (4/0, 4/1, 4/3) — eliminated by moving the un-pad tail into `_unpad_static_shape` (`library/anima/models.py:220`, `@torch.compiler.disable(recursive=True)`).
- `_warmup_active == True` (4/2) — replaced the bool + one-hot-mask split with a single always-applied `_expert_grad_mask` buffer (ones outside warmup collapses to identity).
- `___stack1` size mismatch (5/0, 5/1) — went away with 4/0.
- Python-level None↔Tensor mutations (`_timestep_mask`, `_sigma`) neutralized by registering them as non-persistent buffers; `_last_gate` left as an inline `STORE_ATTR`.

Remaining Dynamo recompiles: **7 × `GLOBAL_STATE changed: grad_mode`** at the train → validation boundary. Each compiled frame specializes once for train + once for eval; amortized over an epoch this is a fixed tax, not per-step churn. **Accepted.**

## Now: CUDAGraph re-capture across aspect-ratio buckets

Full-compile run with `mode="reduce-overhead"` logs:
```
CUDAGraph supports dynamic shapes by recording a new graph for each distinct input size.
We have observed 9 distinct sizes.
```

**Root cause.** The compiled frame starts at `DiT.forward(x_B_C_T_H_W, ...)` — raw latent, shape `(B, C, 1, H/16, W/16)`, which differs per bucket in `CONSTANT_TOKEN_BUCKETS` (17 entries). `static_token_count` pads *inside* the frame at `models.py:1623-1646`, but the entry point still varies. Result: one CUDAGraph per bucket shape (× 2 for grad_mode ≈ 34 captures in the worst case).

Capture is cheap (~1–3 s each) and replay is constant-cost, so steady-state throughput isn't hurt. Costs are (a) warmup time, (b) VRAM — each CUDAGraph carries its own memory pool, which can add up to several GB across 17 buckets.

### Phase F — constant-shape compile region

Split `forward_mini_train_dit` into three regions; compile only the middle.

#### 1. Pre-blocks (eager, `@torch.compiler.disable(recursive=True)`)

Everything whose shape depends on the bucket:

- `llm_adapter` call (if `t5_input_ids is not None` and `use_llm_adapter`)
- `prepare_embedded_sequence` → patchify + RoPE generation
- Static-pad: `x_B_T_H_W_D` → `(B, 1, target, 1, D)`; RoPE cos/sin padded to `target`
- `t_embedder` + pooled-text projection (already shape-constant, but belongs with eager setup)
- AdaLN mask construction (already constant via `torch.tensor(seq_len)` trick at `models.py:1744-1760`)
- Cross-attn BlockMask construction (flex mode)
- Compute `_static_pad_info` tuple

Returns a shape-invariant bundle: `(x_padded, t_embedding, adaln_lora, crossattn_emb, attn_params, block_kwargs, mod_delta, mod_schedule, mod_final_w, _static_pad_info)`.

#### 2. Compiled core — new method `_run_blocks`

```python
def _run_blocks(self, x_padded, t_embedding, crossattn_emb, attn_params,
                mod_delta, mod_schedule, **block_kwargs):
    x = x_padded.requires_grad_()
    for block_idx, block in enumerate(self.blocks):
        if self.blocks_to_swap:
            self.offloader.wait_for_block(block_idx)
        t_emb_block = t_embedding
        if mod_delta is not None and mod_schedule is not None:
            w_l = mod_schedule[block_idx]
            if w_l != 0.0:
                t_emb_block = t_embedding + (w_l * mod_delta).unsqueeze(1)
        x = block(x, t_emb_block, crossattn_emb, attn_params, **block_kwargs)
        if self.blocks_to_swap:
            self.offloader.submit_move_blocks(self.blocks, block_idx)
    return x
```

All inputs are constant-shape by construction:
- `x_padded: (B, 1, target, 1, D)` — from static-pad
- `t_embedding: (B, 1, D_t)` — scalar-ish
- `crossattn_emb: (B, max_text_len, D)` — already padded to `max_length` (text-encoder invariant)
- `attn_params.{self,cross}attn_block_mask` — constant-shape BlockMasks
- `block_kwargs["rope_cos_sin"]`: each `(target, 1, 1, D_head)` — padded to `target`
- `block_kwargs["adaln_lora_B_T_3D"]: (B, 1, 3, D)` — constant
- `mod_delta: (B, D_t)` or None; `mod_schedule: (num_blocks,)` tensor or None

#### 3. Post-blocks (eager)

- `_unpad_static_shape(x, _static_pad_info)` — already `@torch.compiler.disable`'d
- `final_layer(x, t_emb_final, adaln_lora_B_T_3D=...)` (includes `mod_final_w` path)
- `unpatchify`

### Wiring

Current: `train.py:1892-1915` calls `unet = torch.compile(unet, ...)`, which drags the entire `forward` (and thus `forward_mini_train_dit`) into Dynamo.

Two options:

- **(A) Narrow the wrap.** Replace `torch.compile(unet)` with `self._run_blocks = torch.compile(self._run_blocks, ...)` inside DiT, applied when `torch_compile` is requested. Drop the outer wrapper. Cleanest split — only the core is traced; pre/post are never Dynamo-visible.
- **(B) Keep the outer wrap, disable the wings.** Decorate the new `_pre_blocks` and `_post_blocks` helpers with `@torch.compiler.disable(recursive=True)`. Less invasive to `train.py`, but keeps the outer trace boundary where it is and relies on `disable` doing the right thing at method call sites.

Recommend **(A)**. It also gives us a named compiled callable we can assert on in tests.

### Risks / things to verify

- **Inference path** (`forward` at `models.py:1813`) routes to the same `forward_mini_train_dit`. The split must not break it. Both paths should hit the compiled `_run_blocks`.
- **Block-swap** (`self.offloader.wait_for_block` / `submit_move_blocks`) inside the loop already works under compile; it moves weights asynchronously and the compiled graph only depends on `x`. Should be unchanged, but verify on a `low_vram` preset.
- **Mod guidance schedule** — per-block `w_l` is a tensor index; if it's a Python float, it will guard. Convert `mod_schedule` to a tensor before passing in and index with `mod_schedule[block_idx]`.
- **ReFT** patches add `R^T·(ΔW·h + b)·scale` inside block `forward` via `forward_hook`s — shape-invariant, no change.
- **Postfix / HydraLoRA** adapter outputs — already shape-invariant (depend on block input + shared buffers, not bucket shape).
- **Spectrum inference** — uses `register_forward_pre_hook` on `final_layer`, which is now outside the compiled region. Skip-block logic should still work; may actually be cleaner since the hook no longer fires inside a traced frame.

### Validation

- `TORCH_LOGS=recompiles` on a 22-step trial: assert 0 new guard failures beyond the existing grad_mode tax.
- Log CUDAGraph captures (watch for the "observed N distinct sizes" warning, or `torch._inductor.utils.counters["inductor"]["cudagraph_skipped"]`): expect **1** graph instead of 9–17.
- `make test-unit` green (62/62).
- Bit-equivalence: hash block outputs on a single bucket before/after the refactor.
- VRAM peak before/after on a multi-bucket training run — expected drop proportional to `(buckets-1) × per-graph pool size`.

## Later / orthogonal

- **Phase E — diagnostics & regression test.** `make train-compile-debug` target (sets `TORCH_LOGS=recompiles,guards,graph_breaks`). Raise `torch._dynamo.config.recompile_limit` 8 → 16 with a log line when `args.torch_compile` is set. `tests/test_compile_recompiles.py` smoke-run asserting non-`grad_mode` guard counters are 0.
- **Online-softmax warning** at `torch/_inductor/lowering.py:7836` — try `torch._inductor.config.online_softmax = True` or `split_reductions = False`. Separate ticket.
- **Stretch: `[fullcompile]` preset.** Asserts `gradient_checkpointing = false`, `unsloth_offload_checkpointing = false`, `blocks_to_swap = 0`, `compile_mode = "full"`, `compile_inductor_mode = "reduce-overhead"`. `check_fullgraph_preconditions()` at startup fails loud if any method config re-enables those toggles.

## References

PyTorch 2.11 docs mirrored under `docs/pytorch_docs/`:
- `Dynamic Shapes` — `mark_dynamic`/`maybe_mark_dynamic`/`mark_static`, `set_stance`, `TORCH_COMPILE_DYNAMIC_SOURCES`.
- `Advanced Options to Control Dynamic Behavior` — PGO, `force_*_static_shapes` flags.
- `Dynamo Deep-Dive` — guard anatomy, resume functions, frame tracing policy.
