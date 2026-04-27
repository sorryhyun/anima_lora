# Full-model compile + CUDAGraphs

Design notes for `compile_mode = "full"` paired with `compile_inductor_mode = "reduce-overhead"` — the most aggressive compile configuration in this fork. Builds on the static-shape foundation in [`for_compile.md`](for_compile.md) and shares its bucketing scheme; everything below is what was added on top.

The headline number: in `compile_mode = "full"` we hand inductor a single 28-block graph and let `cudagraph_trees` capture *one* CUDAGraph that gets replayed every step. There is no per-block launch boundary, no per-step kernel-launch overhead, no per-bucket recompilation. The cost of getting there is a strict static-shape contract for the compile zone and a buffer-mutation discipline for everything reachable from it.

---

## 1. Goals and tradeoffs

| Mode | Compile target | Kernel boundaries / step | Cross-block fusion | Compatible with grad ckpt / block swap |
|------|----------------|--------------------------|--------------------|-----------------------------------------|
| `blocks` (default) | each `block._forward` | 28 + pre/post | no | yes |
| `full` | `DiT._run_blocks` | 1 + pre/post | yes — inductor sees the whole stack | no (asserted off) |

`blocks` is the forgiving option: it tolerates checkpointing, block swap, varying caption lengths, and most surprises in the adapter graph. `full` strips all of that flexibility in exchange for inductor visibility across block boundaries (cross-block memory planning, dead-store elimination, possible kernel fusion across the residual path) and one CUDAGraph instead of 28.

The choice to compile `_run_blocks` specifically — not `forward_mini_train_dit`, not `forward` — is the single most important design decision in this whole setup. Everything else falls out of it.

---

## 2. Splitting the forward at `_run_blocks`

`forward_mini_train_dit` is partitioned into three regions (`library/anima/models.py:1695`):

```
              eager                            COMPILED                       eager
┌─────────────────────────────────┐  ┌──────────────────────────┐  ┌────────────────────────┐
 patch_embed → static-pad → RoPE      _run_blocks                  _unpad_static_shape →
 → t_embedder → BlockMask build       (28 × Block._forward)        final_layer → unpatchify
└─────────────────────────────────┘  └──────────────────────────┘  └────────────────────────┘
   shapes vary by bucket                shapes constant               shapes vary by bucket
```

### Why this split

- **The pre-blocks region produces bucket-dependent shapes.** `(T_s, H_s, W_s)` differs across the 17 entries of `CONSTANT_TOKEN_BUCKETS`. If we compiled this region, we'd record one CUDAGraph per bucket. With reduce-overhead that's 17 separate captures, each ~6 GiB of cudagraph pool, replayed on whichever bucket the dataloader happens to serve.
- **`_run_blocks` is shape-invariant by construction.** Its inputs are flattened-and-padded to `static_token_count = 4096` *before* it's called. One graph serves every bucket.
- **The post-blocks region is also bucket-dependent**, plus it needs to know the original `seq_len` to strip padding. We can't compile it for the same reason.

So the split lives exactly where shape invariance starts and ends.

### What `_run_blocks` is allowed to assume (`library/anima/models.py:1645`)

The eager pre-blocks region is the contract enforcer; `_run_blocks` is the consumer. By the time control crosses into it:

| Input | Shape | Source |
|-------|-------|--------|
| `x_padded` | `(B, 1, static_token_count, 1, D)` | flatten + pad + fake-5D reshape |
| `t_embedding_B_T_D` | `(B, 1, D)` | `t_embedder` output |
| `crossattn_emb` | `(B, max_text_len, D)` | TE output, padded to `max_length` |
| `block_kwargs["rope_cos_sin"]` | each `(static_token_count, 1, 1, D_head)` | RoPE pad-to-target |
| `block_kwargs["adaln_lora_B_T_3D"]` | `(B, 1, 3, D)` | `t_embedder` output |
| `attn_params` | `AttentionParams` | flash mode → cu_seqlens; flex mode → BlockMask built with tensor-valued seqlen so no per-bucket guards fire |

Every shape is a function of `static_token_count`, `B`, `D`, `max_text_len`, `D_head`. None of them depend on bucket choice. There are no Python ints crossing the boundary that vary across buckets — that's deliberate, and the next section explains why.

### Eager handoff: `_unpad_static_shape`

After the block stack, padding has to come off. `_unpad_static_shape` (`library/anima/models.py:221`) is wrapped in `@torch.compiler.disable(recursive=True)`:

```python
@torch.compiler.disable(recursive=True)
def _unpad_static_shape(x, pad_info):
    T_s, H_s, W_s, seq_len = pad_info
    x = x.squeeze(3).squeeze(1)
    x = x[:, :seq_len, :]
    x = x.unflatten(1, (T_s, H_s, W_s))
    return x
```

`pad_info` is a 4-tuple of Python ints. If this ran inside the compiled frame each bucket would specialize `pad_info[1] == H_s` (per-value guard) and narrow the symbolic range on `pad_info[3]`. Running it eagerly keeps the returned tensor's *shape* as the only signal crossing back into the post-blocks compile zone — downstream ops (final_layer, unpatchify) then pick up symbolic `T/H/W` from the tensor itself, not from Python ints.

### Why we don't compile `block._forward` again inside `full`

Compiling per-block on top of the per-stack compile is a footgun: the outer `torch.compile` already traces into each block's body and fuses across them. Wrapping `block._forward` again inserts a graph break (so dynamo can re-enter the inner compile), defeating the cross-block fusion `full` was supposed to give us. `compile_blocks` and `compile_core` are mutually exclusive — `train.py:521` and `train.py:2101` route on `compile_mode` so only one runs.

---

## 3. Buffer state: the "always-a-Tensor" invariant

Once you're inside a compiled frame, every Python-level branch is a guard. Every guard that fires is a recompile. So per-step state can't be stored as Python primitives or as `Optional[Tensor]` — those produce None-vs-Tensor or bool-vs-int guards.

The fork's standing rule:

> **Per-step state lives in a registered buffer that is always a Tensor.** "Off" means a zero-valued buffer, not `None`. The forward does the unconditional arithmetic; zero collapses it to identity.

Examples:

```python
# library/anima/models.py:1379  — modulation guidance off ↔ zero buffers
self.register_buffer("_mod_guidance_delta",    torch.zeros(1, model_channels), persistent=False)
self.register_buffer("_mod_guidance_schedule", torch.zeros(num_blocks),        persistent=False)

# library/anima/models.py:1679  — applied unconditionally inside _run_blocks
t_emb_block = t_embedding_B_T_D + (
    self._mod_guidance_schedule[block_idx] * self._mod_guidance_delta
).unsqueeze(1)
```

```python
# networks/lora_modules/hydra.py:140  — sigma router input
self.register_buffer("_sigma", torch.zeros(1, dtype=torch.float32), persistent=False)

# networks/lora_modules/hydra.py:150  — expert-warmup gradient mask
# Default ones → up*1 + up.detach()*0 == up, so the warmup branch is
# always-on and a no-op when warmup is disabled.
self.register_buffer("_expert_grad_mask", torch.ones(num_experts, dtype=torch.float32),
                     persistent=False)
```

Each of these buffers is set up so the forward never sees a `None`, never sees a Python `if warmup:`, never sees a shape change. The arithmetic is unconditional; zero / one is the off state.

The same trick covers per-block guidance schedules (`_mod_guidance_schedule[block_idx] * delta`), T-LoRA timestep masks (zero rows = full rank), and ReFT timestep masks. They're all the same shape, so a per-block index becomes a tensor read instead of a Python branch.

---

## 4. In-place mutation, never rebind

Buffers solve the *control-flow* side of the problem. `reduce-overhead` adds a *memory-pointer* side: cudagraph_trees captures every parameter and buffer reachable from the compiled frame as a *static input*. A static input must keep the same `data_ptr()` across steps, or the entire graph is re-recorded.

The recipe to update per-step state without re-recording:

```python
# Lazy-init once on the right device, then COPY in place every step thereafter.
mask = getattr(self, "_shared_timestep_mask", None)
if mask is None or mask.device != timesteps.device:
    mask = torch.zeros(1, max_rank, device=timesteps.device)
    self._shared_timestep_mask = mask
    for lora in self.text_encoder_loras + self.unet_loras:
        lora._timestep_mask = mask  # share the same storage
mask.copy_((self._timestep_mask_arange < r).to(mask.dtype).unsqueeze(0))
```

(`networks/lora_anima/network.py:535`)

The shared-buffer trick has a second payoff: every adapted Linear on every block reads the *same* tensor object, so we update one buffer and 200+ adapter modules see the new value with no extra work.

Inventory of per-step setters (all in `networks/lora_anima/network.py`):

| Setter | What it updates | Method |
|--------|-----------------|--------|
| `set_timestep_mask` | shared T-LoRA rank mask | `mask.copy_(...)` |
| `set_reft_timestep_mask` | shared ReFT rank mask | `mask.copy_(...)` |
| `clear_timestep_mask` | reset to ones | `shared.fill_(1.0)` |
| `set_sigma` | per-module σ buffer | `buf.copy_(sigmas)` (shape-fast-path) |
| `clear_sigma` | per-module σ buffer | `sigma.zero_()` |
| Hydra warmup step | `_expert_grad_mask` | `.scatter_` / `.fill_` |

The diagnostic for getting this wrong is loud once you know what to look for:

```
[1/0] Re-recording function=partition_0, reason=static input data pointer changed.
input name: primals_52. data pointer changed from 138645101078016 to 138644992019968.
```

The stack trace immediately above points at the offending Python expression — usually a `self._foo = new_tensor` line. The fix is invariably `self._foo.copy_(new_tensor)` plus a one-shot rebind path for the first call (when the placeholder shape may not match).

`set_sigma` originally rebound the buffer; that single bug produced a re-record per step and was the entire reason a 28-block compile couldn't outpace a 28-block-eager run. See [the diagnosis log](../../) — fixed in `networks/lora_anima/network.py:605`.

---

## 5. The CUDAGraph step boundary

cudagraph_trees needs an explicit "the previous step's outputs are dead, the pool is yours" signal each iteration. Without it, its safety check ("are there pending uninvoked backwards?") fires every step and silently demotes replay to the eager fallback path.

`train.py:2244` decides whether marking is needed:

```python
self._cudagraph_mark_step = bool(
    getattr(args, "torch_compile", False)
    and getattr(args, "compile_inductor_mode", None)
    in ("reduce-overhead", "max-autotune")
)
```

`train.py:2829` does the marking:

```python
if self._cudagraph_mark_step:
    net_unwrapped = accelerator.unwrap_model(network)
    if hasattr(net_unwrapped, "clear_step_caches"):
        net_unwrapped.clear_step_caches()
    torch.compiler.cudagraph_mark_step_begin()
```

Two things happen here, and the order matters:

1. **`clear_step_caches()`** drops Python references to tensors *produced inside* the compiled region — `_last_gate` (Hydra router output, cached for the balance loss aux term), `_last_sigma` (convenience handle for per-σ-bucket bookkeeping). Both live in the cudagraph memory pool. A lingering Python reference would pin the pool regardless of the mark call, and inductor would silently fall back to non-graph execution.
2. **`cudagraph_mark_step_begin()`** then signals the boundary.

`clear_step_caches` is the correct dumping ground for any new "captured inside `_run_blocks`, used by the loss" handle; `_sigma` is intentionally *not* cleared here, because it's a buffer the network rewrites via `set_sigma` *before* every forward — clearing it would mean the next forward reads stale state.

---

## 6. Compatibility constraints

The `full` path force-asserts away two features (`train.py:2102`, `train.py:2105`):

| Feature | Why it's incompatible |
|---------|-----------------------|
| `gradient_checkpointing` | Checkpointing reruns the block forward inside the backward pass. That re-entry hits the compiled frame from outside cudagraph_trees' step-boundary tracking and breaks pool semantics. |
| `blocks_to_swap > 0` | Block swap moves parameters between CPU and GPU between blocks; cudagraphs require parameter `data_ptr()` stable across the entire replay. |

| Feature | Status under `full` |
|---------|---------------------|
| OrthoLoRA / T-LoRA / ReFT stack | ✅ all share the always-a-Tensor + in-place-mutation discipline |
| HydraLoRA + sigma router | ✅ after the `set_sigma` in-place fix |
| APEX (3 forwards/step) | ⚠ untested — APEX itself force-disables `blocks_to_swap`, but the `c_fake = A·c + b` shift introduces an extra cudagraph input |
| IP-Adapter / EasyControl | ⚠ untested — image-stream KV adds tensors crossing the compile boundary |
| LoRA's `use_custom_down_autograd` | ✅ confirmed compile-stable in 60-step A/B (see `for_compile.md` §5.2) |

When in doubt, fall back to `compile_mode = "blocks"`. It loses cross-block fusion but tolerates most surprises.

---

## 7. Diagnosing

The default failure mode is silent — replay just runs slow and GPU drops to 0% between forwards. To see what's happening:

```bash
TORCH_LOGS="recompiles,cudagraphs" make lora 2>&1 | tee /tmp/compile.log
```

Patterns and what they mean:

| Pattern | Meaning |
|---------|---------|
| `Recording cudagraph tree for graph without symints` | First-time capture. Expected once at warmup (and a second time for the backward graph). |
| `Re-recording function=partition_0, reason=static input data pointer changed` | A captured static input changed pointer — usually a buffer rebind. Stack trace points at the offending read. |
| `skipping cudagraphs due to ...` | Inductor refused to use cudagraphs at all. Read the reason — usually a forbidden op (`.item()`, `.cpu()`, dynamic shape) inside the compiled region. |
| `Recompiling function ...` | Dynamo recompilation. More expensive than CG re-record. Usually a None-vs-Tensor, bool-vs-int, or shape guard fired. |

Triage commands:

```bash
grep -c "Re-recording" /tmp/compile.log                                       # how many?
grep -E "input name:" /tmp/compile.log \
  | sed 's/.*input name: \([^.]*\)\..*/\1/' | sort | uniq -c | sort -rn       # which inputs?
grep -B2 "Recompiling function" /tmp/compile.log | head -40                   # dynamo recompiles
```

Healthy log: two `Recording cudagraph tree` lines (forward + backward) at warmup, then nothing for the rest of the run.

---

## Summary

The full-model + cudagraph path is held up by four design rules:

1. **Compile only the shape-invariant region** (`_run_blocks`), so one CUDAGraph serves every bucket. Eager pre/post regions absorb shape variance at the boundary; `_unpad_static_shape` is `@torch.compiler.disable`d so its Python-int tuple never crosses back.
2. **Per-step state lives in registered buffers, always as a Tensor.** "Off" is a zero-valued buffer, not `None` and not a Python `if`. The forward does unconditional arithmetic.
3. **Mutate buffers in place, never rebind.** `cudagraph_trees` captures buffers as static inputs with stable `data_ptr()`; rebinding triggers a full re-record.
4. **Mark the step boundary every iteration**, after dropping Python refs to anything produced inside the compiled region (`clear_step_caches` → `cudagraph_mark_step_begin`).

The asserts at the entry to `compile_core` cover what can't coexist with the contract (`gradient_checkpointing`, `blocks_to_swap`). The `TORCH_LOGS="recompiles,cudagraphs"` recipe covers what slips through it.
