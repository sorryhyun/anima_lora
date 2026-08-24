# Anima: anima_lora vs ComfyUI implementation differences

Two independent Anima DiT implementations live side-by-side in this workspace:

| label | path | role |
|---|---|---|
| **anima_lora** | `library/anima/models.py` (class `Anima`) | training + inference stack for LoRA / distillation / GRAFT / Spectrum |
| **comfy** | `comfy/comfy/ldm/cosmos/predict2.py` (class `MiniTrainDIT`) + `comfy/comfy/ldm/anima/model.py` (wrapper) | ComfyUI's vanilla Anima runtime |

They share the same model family lineage (`MiniTrainDIT`) and load the same base checkpoints for the transformer blocks, but the two forward paths have diverged in several behaviorally-visible ways. This doc catalogs those differences so you don't waste debugging cycles chasing phantom bugs when a workflow behaves differently between `inference.py` and ComfyUI.

This matters in particular for anything that hooks the forward path — notably `mod_guidance.py` in [ComfyUI-Spectrum-KSampler](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler) and the per-block mod-guidance scheduling documented in [`docs/inference/mod-guidance.md`](../inference/mod-guidance.md).

## TL;DR

| feature | anima_lora | comfy |
|---|---|---|
| `pooled_text_proj` MLP (distilled modulation-guidance head) | **present**, baked into `forward_mini_train_dit` | **absent entirely** |
| `torch.compile` on block forwards | `compile_blocks()` compiles each `block._forward` | not used |
| Constant-token bucketing (native 4032/4200 shapes) | `compile_blocks()` native-shape flatten | not supported |
| `crossattn_seqlens` / variable text length | computed from mask, used only for the flex block mask | not computed; always pad to 512 |
| Attention dispatch | unified `attention_dispatch.AttentionParams` (sdpa / flash / sageattn / flex; flash4 branch present but disabled) | `transformer_options` dict + ComfyUI's own attention |
| Custom block-swap / CPU offload | `enable_block_swap`, `ModelOffloader` | relies on ComfyUI's `model_management.py` |
| Gradient checkpointing variants | standard / CPU-offload / unsloth | standard only |
| Final-layer dtype cast | implicit (shared dtype assumed) | explicit `.to(crossattn_emb.dtype)` |
| Preprocess text embeds output | variable-length + `crossattn_seqlens` tensor | fixed-padded to 512 |

**The single most important difference for downstream work:** `pooled_text_proj` exists only in anima_lora. Everything else is cosmetic by comparison.

## 1. `pooled_text_proj` — exists only in anima_lora

The distilled modulation-guidance head is baked into anima_lora's `Anima.__init__` and invoked unconditionally during `forward_mini_train_dit`. ComfyUI's `MiniTrainDIT` and the Anima wrapper do not declare, reference, or load this module at all.

**anima_lora** — `library/anima/models.py:1368-1372`:

```python
self.pooled_text_proj = nn.Sequential(
    nn.Linear(crossattn_emb_channels, model_channels),
    nn.SiLU(),
    nn.Linear(model_channels, model_channels),
)
```

Zero-initialized output layer (`library/anima/models.py:1411-1413`) so it's a no-op at init, trained via `project/finished/mod_guidance/distill.py`.

Used in `forward_mini_train_dit` at `library/anima/models.py:1791-1801`:

```python
if not skip_pooled_text_proj:
    if pooled_text_override is not None:
        pooled_text = pooled_text_override
    elif crossattn_emb is not None:
        pooled_text = crossattn_emb.max(dim=1).values
    else:
        pooled_text = None
    if pooled_text is not None:
        t_embedding_B_T_D = t_embedding_B_T_D + self.pooled_text_proj(pooled_text).unsqueeze(1)
```

Explicitly excluded from LoRA targeting in `networks/lora_anima/config.py:162-165` (the `_DEFAULT_EXCLUDE` constant, appended to user-supplied excludes inside `LoRANetworkCfg.from_kwargs`):

```python
_DEFAULT_EXCLUDE = (
    r".*(_modulation|_norm|_embedder|final_layer|adaln_fused_down|adaln_up_|"
    r"pooled_text_proj).*"
)
```

Read that regex as the *default* target set, not the final one: since 2026-07-16
`train_adaln = true` in `configs/base.toml` rescues `adaln_up_` back out of it via
`include_patterns` on every LoRA-family run (see
[`../methods/adaln.md`](../methods/adaln.md)). `pooled_text_proj` — the subject of
this section — has no such escape hatch and stays untrained.

**comfy** — `grep -rn pooled_text_proj comfy/` returns no matches under `comfy/comfy/ldm/`. The vanilla ComfyUI forward at `comfy/comfy/ldm/cosmos/predict2.py:860-861` computes `t_embedding_norm` and passes directly to the block stack with no intermediate pooled-text addition:

```python
t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder[1](
    self.t_embedder[0](timesteps_B_T).to(x_B_T_H_W_D.dtype)
)
t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)
# ... straight into block loop; no pooled_text_proj step
```

### Why this matters — mod-guidance port semantics

The ComfyUI port of mod guidance (`mod_guidance.py` in [ComfyUI-Spectrum-KSampler](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler)) installs a `forward_hook` on `t_embedding_norm` that writes a precomputed combined tensor into the normalized embedding:

```python
self.cond_combined  = (proj_pos + delta).detach()
self.uncond_combined = (proj_neg + delta).detach()
```

**This is correct for ComfyUI** precisely because ComfyUI has no `pooled_text_proj` step. The hook has to supply both the base projection (`proj_pos` / `proj_neg`, which in anima_lora is auto-applied at line 1799) AND the guidance delta (`w·(proj_tag − proj_neg)`), because nothing downstream will add the base projection for it.

**Consequence:** if you ever port the ComfyUI hook semantics back into anima_lora's Python path, you need to subtract the base projection — otherwise you'd double-add it (anima_lora's `forward_mini_train_dit` already adds `proj(pool(crossattn))` at line 1799).

### Checkpoint compatibility

A `pooled_text_proj.safetensors` trained in anima_lora is not a state-dict subset of ComfyUI's `MiniTrainDIT`. It's shipped as a standalone weight file (`models/anima_mod_guidance/pooled_text_proj_0413.safetensors`) and loaded by the custom node's adapter-loader (`mod_guidance.py`), not by ComfyUI's model loader. LoRA checkpoints trained in anima_lora are safe in ComfyUI — they already exclude `pooled_text_proj` keys by construction (`networks/lora_anima/config.py` `_DEFAULT_EXCLUDE`).

## 2. Forward-path differences

### 2.1 Text-embedding preprocessing

Both codebases run a Qwen3 → `llm_adapter` → 1024-dim cross-attention path, but they differ in how the post-adapter sequence is shaped and masked.

**anima_lora** — `library/anima/models.py:1954+` (`_preprocess_text_embeds`) computes `crossattn_seqlens` from the target attention mask, returns `(context, seqlens)`. The seqlens tensor is threaded all the way into the flex-attention block mask at `forward_mini_train_dit:1840-1860`:

```python
def _crossattn_mask_mod(b, h, q_idx, kv_idx):
    return kv_idx < seqlens[b]

attn_params.crossattn_block_mask = attention_dispatch.create_block_mask(
    _crossattn_mask_mod, B, None, q_len, kv_len, device=x_B_T_H_W_D.device,
)
```

It previously carried bucketed KV trimming + sigmoid-based LSE correction (flash4-only), but that plumbing was removed with FA4 (2026-05-20); only the `flash4` branch stub remains in `attention_dispatch.py`. See `docs/optimizations/fa4.md` for why it was disabled.

**comfy** — `comfy/comfy/ldm/anima/model.py:193-214` pads the llm_adapter output to a fixed 512 tokens:

```python
out = self.llm_adapter(...)
return torch.nn.functional.pad(out, (0, 0, 0, 512 - out.shape[1]))
```

It does not compute per-sample seqlens, does not set up flex block masks, and does not apply LSE correction. Cross-attention just sees 512 KV positions every time, with the padding tail implicitly handled as attention sinks (matching the "do NOT mask padding" invariant from `CLAUDE.md`).

**Practical consequence.** Both paths produce the same image quality on normal prompts because the pretrained model was trained with max-padded text anyway (the padding positions act as attention sinks in cross-attention softmax — trimming or masking them produces black images, see `CLAUDE.md` § "Text encoder padding"). The anima_lora infrastructure for variable seqlens was originally there to feed the flash4 LSE-correction path; with flash4 disabled it now only drives the flex block mask, and is not a correctness requirement. **ComfyUI's simpler path is safe.**

### 2.2 Static-shape token bucketing

**anima_lora** — `compile_blocks()` (`library/anima/models.py`) enables native-shape flattening: the forward flattens `(B, T, H, W, D)` into a fake 5D shape of `(B, 1, seq_len, 1, D)`, restored after the block loop by `_unflatten_native_shape`. The shipped bucket table is two token-count families (4032 / 4200), each exactly filling its count, so `torch.compile` sees one block graph per token-count family (two total) with no padding. Eager forwards skip the flatten (bit-exact). The earlier `set_static_token_count(count, pad=True)` zero-padded every bucket to a single shape, but leaked padding into flash self-attention and couldn't run this table (4200 > 4096); it was removed 2026-05-24.

**comfy** — no static-shape mode. Processes variable `(B, T, H, W, D)` directly. Only padding is to patch boundaries via `comfy.ldm.common_dit.pad_to_patch_size()`.

**Practical consequence.** ComfyUI runs Anima eagerly with per-bucket shape changes, which is fine because it also doesn't use `torch.compile`. If you want to run anima_lora's compiled path in ComfyUI, you need to both wire up `compile_blocks` AND arrange the workflow's latent resolution to a compile-compatible bucket — which is realistically an "no, don't try it" situation.

### 2.3 Block forward signature

**anima_lora** (`library/anima/models.py:1179+`):

```python
def forward(self, x_B_T_H_W_D, emb_B_T_D, crossattn_emb,
            attn_params: attention_dispatch.AttentionParams,
            rope_cos_sin=None, adaln_lora_B_T_3D=None):
```

**comfy** (`predict2.py:456+`):

```python
def forward(self, x_B_T_H_W_D, emb_B_T_D, crossattn_emb,
            rope_emb_L_1_1_D=None, adaln_lora_B_T_3D=None,
            extra_per_block_pos_emb=None, transformer_options=None):
```

Two concrete divergences:

1. **anima_lora's `AttentionParams`** is a dataclass that encapsulates `attn_mode`, `softmax_scale`, and `crossattn_block_mask` in one object passed positionally. ComfyUI passes `transformer_options: dict` that ComfyUI's attention dispatch reads ad-hoc.
2. **RoPE shape.** anima_lora passes a `(cos, sin)` tuple computed per-forward. ComfyUI passes a single `rope_emb_L_1_1_D` already pre-unsqueezed. Semantically equivalent but not drop-in interchangeable.

**Practical consequence for the per-block mod-guidance hooks** (now shipped on both sides — see `docs/inference/mod-guidance.md`): in both codebases the `t_emb` argument is at **positional index 1**, so block-level pre-forward hooks can rewrite `args[1]` identically in both implementations. The two signatures diverge on args 3+, but the per-block scheduler only cares about index 1, so the hook factory is portable.

### 2.4 Final layer

**anima_lora** (`library/anima/models.py:1916`):

```python
x_B_T_H_W_O = self.final_layer(
    x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D
)
```

**comfy** (`predict2.py:897`):

```python
x_B_T_H_W_O = self.final_layer(
    x_B_T_H_W_D.to(crossattn_emb.dtype), t_embedding_B_T_D,
    adaln_lora_B_T_3D=adaln_lora_B_T_3D,
)
```

ComfyUI explicitly casts `x` to `crossattn_emb.dtype` before the final layer; anima_lora assumes dtype consistency and doesn't. This is a robustness choice, not a behavioral difference under normal dtype invariants — but if you mix dtypes across the stack in anima_lora, the final layer is where it'll blow up, not in ComfyUI.

## 3. Compilation / performance infrastructure

| capability | anima_lora | comfy |
|---|---|---|
| `torch.compile(block._forward)` | `compile_blocks()` at `library/anima/models.py:1435` | none |
| `@torch._disable_dynamo` on unsloth checkpoint wrapper | yes, `library/anima/models.py:215` | N/A |
| Gradient checkpointing — standard | yes | yes |
| Gradient checkpointing — CPU offload | yes (`enable_gradient_checkpointing(cpu_offload=True)`) | no |
| Gradient checkpointing — unsloth offload | yes (`enable_gradient_checkpointing(unsloth_offload=True)`) | no |
| Constant-token bucketing to stabilize compile cache | `compile_blocks()` native-shape flatten (native 4032/4200, two graphs) | no |
| Block swap / CPU offload inference | `enable_block_swap`, `ModelOffloader` at `library/anima/models.py:1571-1580` | relies on `comfy/model_management.py` LoRAM reservation |
| Switch offload mode between training/inference | `switch_block_swap_for_inference()` / `switch_block_swap_for_training()` | N/A |

anima_lora's entire performance stack is structured around "16GB VRAM must work for training and inference, and compile once across all bucket shapes." ComfyUI's stack is structured around "the runtime already handles VRAM via model_management, just make forward correct." Neither is wrong; they're solving different problems.

**Practical consequence for Spectrum** ([ComfyUI-Spectrum-KSampler](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler)): the Spectrum KSampler has to live with ComfyUI's eager forward — there's no compile boundary to respect, which makes the forward hook on `final_layer` trivially safe (the point of lifting it out of compile in `networks/spectrum.py` only applies on the anima_lora side).

## 4. LoRA loading

**anima_lora** uses the `networks/lora_anima/` package to build a LoRA network via monkey-patching target modules. Target selection uses the exclude pattern quoted above (in `config.py::_DEFAULT_EXCLUDE`) and `network_module` dispatch in configs. LoRA application is by runtime patching, not by weight merging.

**comfy** applies LoRAs via `comfy/lora.py` + `comfy/model_patcher.py`. Weight merging is the default; patch-based is also supported. ComfyUI's stock `LoraLoader` recognizes anima's `lora_unet_` key naming directly (it's the kohya-ss convention) — it strips the prefix and swaps underscores back to dots to map onto `diffusion_model.*` targets. No conversion step is needed.

**Practical consequence.** Plain LoRAs trained in anima_lora load fine in ComfyUI's stock LoRA loader. HydraLoRA / FeRA and prefix/postfix checkpoints carry extra keys (`router.*`, stacked `lora_ups.N.*`) that the stock loader silently drops — those need the `https://github.com/sorryhyun/ComfyUI-Anima_lora-Adapter` Anima Adapter Loader node.

## 5. Text encoder / conditioning interface

Both paths ultimately expect the same thing: a `(B, 512, 1024)` post-llm-adapter cross-attention input. The differences are in **how you get there**.

**anima_lora** — `library/anima/weights.py` loads a Qwen3 tokenizer + encoder, runs the llm_adapter outside the DiT (cached to disk via `scripts/preprocess/cache_text_embeddings.py`), and passes the resulting `crossattn_emb` directly into `forward_mini_train_dit` as `context`. `crossattn_seqlens` is either derived from the attention mask or passed explicitly.

**comfy** — ComfyUI's CLIP/text encoder framework wraps the Qwen3 + llm_adapter path in a CONDITIONING object that bypasses disk caching. The llm_adapter call is inside the Anima wrapper's `preprocess_text_embeds` (`comfy/comfy/ldm/anima/model.py:193+`), which is called during the ComfyUI sample loop rather than ahead-of-time.

**pooled-text extraction** (`max(dim=1).values` over `crossattn_emb`) happens in anima_lora at `library/anima/models.py:1795` as part of the `pooled_text_proj` flow. In ComfyUI it happens **only** inside the custom mod-guidance node (`mod_guidance.py`), because the vanilla forward has no reason to pool.

## 6. Timestep / noise schedule

Both run through `comfy/comfy/ldm/cosmos/predict2.py`-style `Timesteps` → SiLU → Linear. anima_lora samples timesteps via sigmoid-scaled gaussian for training (`sigmas = torch.sigmoid(args.sigmoid_scale * torch.randn(B, ...))` in `project/finished/mod_guidance/distill.py`) and uses `(1-σ)x + σ·noise` flow-matching noising. ComfyUI uses its own `comfy/model_sampling.py` schedule for sampling.

This doesn't affect per-step forward behavior — both implementations accept `timesteps` as a `[0, 1]` scalar tensor and produce equivalent noise predictions. The schedule choice lives above `forward`, in the sampler.

## 7. Block swap / VRAM management

Already covered in §3 as part of the performance table. Two concrete things worth calling out:

- **anima_lora's `ModelOffloader`** (`library/anima/models.py:1571+`) is a custom async-aware offloader with `wait_for_block` / `submit_move_blocks` hooks in the block loop at `library/anima/models.py:1677, 1694`. It runs inside the forward, not at the runtime layer.
- **ComfyUI's model management** is external: `comfy/model_management.py` decides what to keep in VRAM and swaps whole models when memory pressure exceeds thresholds. It does not do per-block swap inside a forward.

**Practical consequence.** If you train with `--blocks_to_swap 16` in anima_lora, the 16GB VRAM path works. If you run the same base model in ComfyUI with less than the required VRAM, ComfyUI will either OOM or swap the entire model between samples — there's no per-block offload equivalent.

## 8. Consequences by scenario

### Running an anima_lora-trained LoRA in ComfyUI

- ✅ LoRA weights load fine via ComfyUI's stock LoRA loader (kohya-ss `lora_unet_` keys are native).
- ✅ Base image quality matches (same underlying transformer blocks).
- ❌ Mod guidance via `pooled_text_proj_0413.safetensors` **does not load into ComfyUI's model** — it lives outside the base state dict. You must use the custom node from [ComfyUI-Spectrum-KSampler](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler) to activate it, which installs the `t_embedding_norm` hook out-of-band.
- ❌ Spectrum acceleration is not available unless you run the [ComfyUI-Spectrum-KSampler](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler) custom node.

### Running a ComfyUI-loaded model through `inference.py`

- You can't, directly. `inference.py` loads base weights from a `.safetensors` via `library.anima.weights.load_anima_model`, not from a ComfyUI model object. If you have a ComfyUI-compatible checkpoint on disk it will load fine into anima_lora's `Anima` class — the base weights match.
- `pooled_text_proj` will be zero-initialized (no-op) unless you also pass `--pooled_text_proj path/to/pooled_text_proj_0413.safetensors`.

### Hooking forward semantics (the mod-guidance case)

- anima_lora: the delta is applied inside the block loop in `forward_mini_train_dit` via `_mod_guidance_schedule` (per-block `w(l)`), with `final_layer` scheduled separately via `_mod_guidance_final_w`.
- comfy: the delta is applied via `register_forward_hook` on `t_embedding_norm` (for the base projection) plus `register_forward_pre_hook` on each block (for the scheduled steering delta). The hook has to supply **both** the base projection and the delta — anima_lora's line-1640 auto-addition doesn't exist here.
- **Do not copy** the ComfyUI hook's `(proj_pos + delta)` combined tensor back into anima_lora — it would double-add the base projection. See `docs/inference/mod-guidance.md` for the separation of `_mod_guidance_delta` (unit direction) from `_mod_guidance_schedule` (per-block `w`).

## Where the two diverge, and where they'll stay diverged

The transformer-block math is identical — that's the point of loading the same base weights. The divergence is entirely in the **wrapping layers**:

- Inference performance infrastructure (compile, offload, static shapes) — anima_lora only.
- Training infrastructure (gradient checkpointing variants, block swap training mode, distillation) — anima_lora only.
- `pooled_text_proj` modulation-guidance head and its surrounding data flow — anima_lora only, grafted on via the custom node in ComfyUI.
- Cross-attention masking / seqlens (flex-mode block mask only; flash4 LSE-correction path is dormant) — anima_lora only.
- Text-encoder integration model (disk-cached vs sampler-inline) — different but equivalent.

There is no reason to merge the two; they solve different problems. The point of this document is to make sure that when you're debugging "it works in `inference.py` but not in ComfyUI" (or vice versa), you can quickly check which of the divergences above might be responsible instead of assuming the underlying model differs.

## References

- `docs/inference/mod-guidance.md` — modulation-guidance mechanism, per-block schedule rationale, and distillation flow
- [ComfyUI-Spectrum-KSampler](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler) `mod_guidance.py` — ComfyUI port of mod guidance with the per-block hook mechanism
- `library/anima/models.py` — anima_lora's `Anima` class, forward path at `forward_mini_train_dit`
- `comfy/comfy/ldm/cosmos/predict2.py::MiniTrainDIT` — ComfyUI's vanilla forward
- `comfy/comfy/ldm/anima/model.py` — ComfyUI's Anima wrapper (`preprocess_text_embeds`)
- `networks/lora_anima/config.py` — `_DEFAULT_EXCLUDE` regex used in LoRA module targeting (`pooled_text_proj` exclusion)
