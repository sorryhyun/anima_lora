# Anima: anima_lora vs ComfyUI implementation differences

Two independent Anima DiT implementations live side-by-side in this workspace:

| label | path | role |
|---|---|---|
| **anima_lora** | `anima_lora/library/anima_models.py` (class `Anima`) | training + inference stack for LoRA / distillation / GRAFT / Spectrum |
| **comfy** | `comfy/comfy/ldm/cosmos/predict2.py` (class `MiniTrainDIT`) + `comfy/comfy/ldm/anima/model.py` (wrapper) | ComfyUI's vanilla Anima runtime |

They share the same model family lineage (`MiniTrainDIT`) and load the same base checkpoints for the transformer blocks, but the two forward paths have diverged in several behaviorally-visible ways. This doc catalogs those differences so you don't waste debugging cycles chasing phantom bugs when a workflow behaves differently between `inference.py` and ComfyUI.

This matters in particular for anything that hooks the forward path — notably [`custom_nodes/comfyui-spectrum/mod_guidance.py`](../custom_nodes/comfyui-spectrum/mod_guidance.py) and the per-block mod-guidance scheduling documented in [`docs/mod-guidance.md`](mod-guidance.md).

## TL;DR

| feature | anima_lora | comfy |
|---|---|---|
| `pooled_text_proj` MLP (distilled modulation-guidance head) | **present**, baked into `forward_mini_train_dit` | **absent entirely** |
| `torch.compile` on block forwards | `compile_blocks()` compiles each `block._forward` | not used |
| Static-shape bucketing (pad to 4096 tokens) | `set_static_token_count()` | not supported |
| `crossattn_seqlens` / variable text length | computed from mask, used for flex block mask + KV trim | not computed; always pad to 512 |
| Attention dispatch | unified `attention.AttentionParams` (sdpa / flash / flash4 / sageattn / flex) | `transformer_options` dict + ComfyUI's own attention |
| Flash4 LSE correction (trimmed KV softmax fix) | present | absent |
| Custom block-swap / CPU offload | `enable_block_swap`, `ModelOffloader` | relies on ComfyUI's `model_management.py` |
| Gradient checkpointing variants | standard / CPU-offload / unsloth | standard only |
| Final-layer dtype cast | implicit (shared dtype assumed) | explicit `.to(crossattn_emb.dtype)` |
| Preprocess text embeds output | variable-length + `crossattn_seqlens` tensor | fixed-padded to 512 |

**The single most important difference for downstream work:** `pooled_text_proj` exists only in anima_lora. Everything else is cosmetic by comparison.

## 1. `pooled_text_proj` — exists only in anima_lora

The distilled modulation-guidance head is baked into anima_lora's `Anima.__init__` and invoked unconditionally during `forward_mini_train_dit`. ComfyUI's `MiniTrainDIT` and the Anima wrapper do not declare, reference, or load this module at all.

**anima_lora** — `library/anima_models.py:1345-1349`:

```python
self.pooled_text_proj = nn.Sequential(
    nn.Linear(crossattn_emb_channels, model_channels),
    nn.SiLU(),
    nn.Linear(model_channels, model_channels),
)
```

Zero-initialized output layer (`anima_models.py:1361-1363`) so it's a no-op at init, trained via `scripts/distill_modulation.py`.

Used in `forward_mini_train_dit` at `anima_models.py:1628-1642`:

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

Explicitly excluded from LoRA targeting in `networks/lora_anima.py:66`:

```python
exclude_patterns.append(
    r".*(_modulation|_norm|_embedder|final_layer|adaln_fused_down|adaln_up_|pooled_text_proj).*"
)
```

**comfy** — `grep -rn pooled_text_proj comfy/` returns no matches under `comfy/comfy/ldm/`. The vanilla ComfyUI forward at `comfy/comfy/ldm/cosmos/predict2.py:860-861` computes `t_embedding_norm` and passes directly to the block stack with no intermediate pooled-text addition:

```python
t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder[1](
    self.t_embedder[0](timesteps_B_T).to(x_B_T_H_W_D.dtype)
)
t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)
# ... straight into block loop; no pooled_text_proj step
```

### Why this matters — mod-guidance port semantics

The ComfyUI port of mod guidance ([`custom_nodes/comfyui-spectrum/mod_guidance.py`](../custom_nodes/comfyui-spectrum/mod_guidance.py)) installs a `forward_hook` on `t_embedding_norm` that writes a precomputed combined tensor into the normalized embedding:

```python
self.cond_combined  = (proj_pos + delta).detach()
self.uncond_combined = (proj_neg + delta).detach()
```

**This is correct for ComfyUI** precisely because ComfyUI has no `pooled_text_proj` step. The hook has to supply both the base projection (`proj_pos` / `proj_neg`, which in anima_lora is auto-applied at line 1640) AND the guidance delta (`w·(proj_tag − proj_neg)`), because nothing downstream will add the base projection for it.

**Consequence:** if you ever port the ComfyUI hook semantics back into anima_lora's Python path, you need to subtract the base projection — otherwise you'd double-add it (anima_lora's `forward_mini_train_dit` already adds `proj(pool(crossattn))` at line 1640).

### Checkpoint compatibility

A `pooled_text_proj.safetensors` trained in anima_lora is not a state-dict subset of ComfyUI's `MiniTrainDIT`. It's shipped as a standalone weight file (`models/anima_mod_guidance/pooled_text_proj_0413.safetensors`) and loaded by the custom node's adapter-loader (`mod_guidance.py:82-113`), not by ComfyUI's model loader. LoRA checkpoints trained in anima_lora are safe in ComfyUI — they already exclude `pooled_text_proj` keys by construction (`lora_anima.py:66`).

## 2. Forward-path differences

### 2.1 Text-embedding preprocessing

Both codebases run a Qwen3 → `llm_adapter` → 1024-dim cross-attention path, but they differ in how the post-adapter sequence is shaped and masked.

**anima_lora** — `library/anima_models.py:1805+` (`_preprocess_text_embeds`) computes `crossattn_seqlens` from the target attention mask, returns `(context, seqlens)`. The seqlens tensor is threaded all the way into the flex-attention block mask at `forward_mini_train_dit:1682-1702`:

```python
def _crossattn_mask_mod(b, h, q_idx, kv_idx):
    return kv_idx < seqlens[b]

attn_params.crossattn_block_mask = attention.create_block_mask(
    _crossattn_mask_mod, B, None, q_len, kv_len, device=x_B_T_H_W_D.device,
)
```

It also supports bucketed KV trimming + sigmoid-based LSE correction for flash4 (`anima_models.py:1662-1679`, currently commented out but the correction is documented in `docs/mod-guidance.md` and `docs/fa4.md`).

**comfy** — `comfy/comfy/ldm/anima/model.py:193-214` pads the llm_adapter output to a fixed 512 tokens:

```python
out = self.llm_adapter(...)
return torch.nn.functional.pad(out, (0, 0, 0, 512 - out.shape[1]))
```

It does not compute per-sample seqlens, does not set up flex block masks, and does not apply LSE correction. Cross-attention just sees 512 KV positions every time, with the padding tail implicitly handled as attention sinks (matching the "do NOT mask padding" invariant from `anima_lora/CLAUDE.md`).

**Practical consequence.** Both paths produce the same image quality on normal prompts because the pretrained model was trained with max-padded text anyway (the padding positions act as attention sinks in cross-attention softmax — trimming or masking them produces black images, see `CLAUDE.md` § "Text encoder padding"). The anima_lora infrastructure for variable seqlens exists to enable the flash4 LSE correction path and future KV trimming experiments — it's not a correctness requirement. **ComfyUI's simpler path is safe.**

### 2.2 Static-shape token bucketing

**anima_lora** — `set_static_token_count(4096)` enables a transform (`anima_models.py:1593-1622`) that flattens `(B, T, H, W, D)` into a fake 5D shape of `(B, 1, target, 1, D)` and zero-pads to 4096 tokens. Together with bucket resolutions that satisfy `(H/16)·(W/16) ≈ 4096`, this gives `torch.compile` a single static shape across all aspect ratios — no recompilation across buckets.

**comfy** — no static-shape mode. Processes variable `(B, T, H, W, D)` directly. Only padding is to patch boundaries via `comfy.ldm.common_dit.pad_to_patch_size()`.

**Practical consequence.** ComfyUI runs Anima eagerly with per-bucket shape changes, which is fine because it also doesn't use `torch.compile`. If you want to run anima_lora's compiled path in ComfyUI, you need to both install `set_static_token_count` AND arrange the workflow's latent resolution to a compile-compatible bucket — which is realistically an "no, don't try it" situation.

### 2.3 Block forward signature

**anima_lora** (`anima_models.py:1158+`):

```python
def forward(self, x_B_T_H_W_D, emb_B_T_D, crossattn_emb,
            attn_params: attention.AttentionParams,
            rope_cos_sin=None, adaln_lora_B_T_3D=None):
```

**comfy** (`predict2.py:456+`):

```python
def forward(self, x_B_T_H_W_D, emb_B_T_D, crossattn_emb,
            rope_emb_L_1_1_D=None, adaln_lora_B_T_3D=None,
            extra_per_block_pos_emb=None, transformer_options=None):
```

Two concrete divergences:

1. **anima_lora's `AttentionParams`** is a dataclass that encapsulates `attn_mode`, `split_attn`, `softmax_scale`, `crossattn_block_mask`, and `crossattn_full_len` in one object passed positionally. ComfyUI passes `transformer_options: dict` that ComfyUI's attention dispatch reads ad-hoc.
2. **RoPE shape.** anima_lora passes a `(cos, sin)` tuple computed per-forward. ComfyUI passes a single `rope_emb_L_1_1_D` already pre-unsqueezed. Semantically equivalent but not drop-in interchangeable.

**Practical consequence for the per-block mod-guidance hooks** (now shipped on both sides — see `docs/mod-guidance.md`): in both codebases the `t_emb` argument is at **positional index 1**, so block-level pre-forward hooks can rewrite `args[1]` identically in both implementations. The two signatures diverge on args 3+, but the per-block scheduler only cares about index 1, so the hook factory is portable.

### 2.4 Final layer

**anima_lora** (`anima_models.py:1767`):

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
| `torch.compile(block._forward)` | `compile_blocks()` at `anima_models.py:1385` | none |
| `@torch._disable_dynamo` on unsloth checkpoint wrapper | yes, `anima_models.py:214` | N/A |
| Gradient checkpointing — standard | yes | yes |
| Gradient checkpointing — CPU offload | yes (`enable_gradient_checkpointing(cpu_offload=True)`) | no |
| Gradient checkpointing — unsloth offload | yes (`enable_gradient_checkpointing(unsloth_offload=True)`) | no |
| Static-shape to stabilize compile cache | `set_static_token_count(4096)` | no |
| Block swap / CPU offload inference | `enable_block_swap`, `ModelOffloader` at `anima_models.py:1493-1535` | relies on `comfy/model_management.py` LoRAM reservation |
| Switch offload mode between training/inference | `switch_block_swap_for_inference()` / `switch_block_swap_for_training()` | N/A |

anima_lora's entire performance stack is structured around "16GB VRAM must work for training and inference, and compile once across all bucket shapes." ComfyUI's stack is structured around "the runtime already handles VRAM via model_management, just make forward correct." Neither is wrong; they're solving different problems.

**Practical consequence for Spectrum** (`custom_nodes/comfyui-spectrum/`): the Spectrum KSampler has to live with ComfyUI's eager forward — there's no compile boundary to respect, which makes the forward hook on `final_layer` trivially safe (the point of lifting it out of compile in `networks/spectrum.py` only applies on the anima_lora side).

## 4. LoRA loading

**anima_lora** uses `networks/lora_anima.py` to build a LoRA network via monkey-patching target modules. Target selection uses the exclude pattern at line 66 (quoted above) and `network_module` dispatch in configs. LoRA application is by runtime patching, not by weight merging.

**comfy** applies LoRAs via `comfy/lora.py` + `comfy/model_patcher.py`. Weight merging is the default; patch-based is also supported. ComfyUI's LoRA key-mapping has to convert between anima_lora's key naming (e.g. `lora_unet_blocks_0_cross_attn_q_proj.lora_down.weight`) and ComfyUI's expected format — `scripts/convert_lora_to_comfy.py` handles this for you when exporting LoRAs for ComfyUI use.

**Practical consequence.** LoRAs trained in anima_lora load fine in ComfyUI after running `scripts/convert_lora_to_comfy.py`. LoRAs trained in ComfyUI (rare, since the training stack lives in anima_lora) would need the inverse conversion.

## 5. Text encoder / conditioning interface

Both paths ultimately expect the same thing: a `(B, 512, 1024)` post-llm-adapter cross-attention input. The differences are in **how you get there**.

**anima_lora** — `library/anima_utils.py` loads a Qwen3 tokenizer + encoder, runs the llm_adapter outside the DiT (cached to disk via `scripts/cache_text_embeddings.py`), and passes the resulting `crossattn_emb` directly into `forward_mini_train_dit` as `context`. `crossattn_seqlens` is either derived from the attention mask or passed explicitly.

**comfy** — ComfyUI's CLIP/text encoder framework wraps the Qwen3 + llm_adapter path in a CONDITIONING object that bypasses disk caching. The llm_adapter call is inside the Anima wrapper's `preprocess_text_embeds` (`comfy/comfy/ldm/anima/model.py:193+`), which is called during the ComfyUI sample loop rather than ahead-of-time.

**pooled-text extraction** (`max(dim=1).values` over `crossattn_emb`) happens in anima_lora at line 1636 as part of the `pooled_text_proj` flow. In ComfyUI it happens **only** inside the custom mod-guidance node (`mod_guidance.py:173`), because the vanilla forward has no reason to pool.

## 6. Timestep / noise schedule

Both run through `comfy/comfy/ldm/cosmos/predict2.py`-style `Timesteps` → SiLU → Linear. anima_lora samples timesteps via sigmoid-scaled gaussian for training (`sigmas = torch.sigmoid(args.sigmoid_scale * torch.randn(B))` in `scripts/distill_modulation.py:376`) and uses `(1-σ)x + σ·noise` flow-matching noising. ComfyUI uses its own `comfy/model_sampling.py` schedule for sampling.

This doesn't affect per-step forward behavior — both implementations accept `timesteps` as a `[0, 1]` scalar tensor and produce equivalent noise predictions. The schedule choice lives above `forward`, in the sampler.

## 7. Block swap / VRAM management

Already covered in §3 as part of the performance table. Two concrete things worth calling out:

- **anima_lora's `ModelOffloader`** (`anima_models.py:1493+`) is a custom async-aware offloader with `wait_for_block` / `submit_move_blocks` hooks in the block loop at `anima_models.py:1744-1757`. It runs inside the forward, not at the runtime layer.
- **ComfyUI's model management** is external: `comfy/model_management.py` decides what to keep in VRAM and swaps whole models when memory pressure exceeds thresholds. It does not do per-block swap inside a forward.

**Practical consequence.** If you train with `--blocks_to_swap 16` in anima_lora, the 16GB VRAM path works. If you run the same base model in ComfyUI with less than the required VRAM, ComfyUI will either OOM or swap the entire model between samples — there's no per-block offload equivalent.

## 8. Consequences by scenario

### Running an anima_lora-trained LoRA in ComfyUI

- ✅ LoRA weights load fine (after `scripts/convert_lora_to_comfy.py`).
- ✅ Base image quality matches (same underlying transformer blocks).
- ❌ Mod guidance via `pooled_text_proj_0413.safetensors` **does not load into ComfyUI's model** — it lives outside the base state dict. You must use the custom node (`custom_nodes/comfyui-spectrum/mod_guidance.py`) to activate it, which installs the `t_embedding_norm` hook out-of-band.
- ❌ Spectrum acceleration is not available unless you run the `comfyui-spectrum` custom node.

### Running a ComfyUI-loaded model through `inference.py`

- You can't, directly. `inference.py` loads base weights from a `.safetensors` via `anima_utils.load_anima_model`, not from a ComfyUI model object. If you have a ComfyUI-compatible checkpoint on disk it will load fine into anima_lora's `Anima` class — the base weights match.
- `pooled_text_proj` will be zero-initialized (no-op) unless you also pass `--pooled_text_proj path/to/pooled_text_proj_0413.safetensors`.

### Hooking forward semantics (the mod-guidance case)

- anima_lora: the delta is applied inside the block loop in `forward_mini_train_dit` via `_mod_guidance_schedule` (per-block `w(l)`), with `final_layer` scheduled separately via `_mod_guidance_final_w`.
- comfy: the delta is applied via `register_forward_hook` on `t_embedding_norm` (for the base projection) plus `register_forward_pre_hook` on each block (for the scheduled steering delta). The hook has to supply **both** the base projection and the delta — anima_lora's line-1640 auto-addition doesn't exist here.
- **Do not copy** the ComfyUI hook's `(proj_pos + delta)` combined tensor back into anima_lora — it would double-add the base projection. See `docs/mod-guidance.md` for the separation of `_mod_guidance_delta` (unit direction) from `_mod_guidance_schedule` (per-block `w`).

## Where the two diverge, and where they'll stay diverged

The transformer-block math is identical — that's the point of loading the same base weights. The divergence is entirely in the **wrapping layers**:

- Inference performance infrastructure (compile, offload, static shapes) — anima_lora only.
- Training infrastructure (gradient checkpointing variants, block swap training mode, distillation) — anima_lora only.
- `pooled_text_proj` modulation-guidance head and its surrounding data flow — anima_lora only, grafted on via the custom node in ComfyUI.
- Cross-attention masking / seqlens / flash4 LSE correction — anima_lora only.
- Text-encoder integration model (disk-cached vs sampler-inline) — different but equivalent.

There is no reason to merge the two; they solve different problems. The point of this document is to make sure that when you're debugging "it works in `inference.py` but not in ComfyUI" (or vice versa), you can quickly check which of the divergences above might be responsible instead of assuming the underlying model differs.

## References

- `docs/mod-guidance.md` — modulation-guidance mechanism, per-block schedule rationale, and distillation flow
- `custom_nodes/comfyui-spectrum/mod_guidance.py` — ComfyUI port of mod guidance with the per-block hook mechanism
- `library/anima_models.py` — anima_lora's `Anima` class, forward path at `forward_mini_train_dit`
- `comfy/comfy/ldm/cosmos/predict2.py::MiniTrainDIT` — ComfyUI's vanilla forward
- `comfy/comfy/ldm/anima/model.py` — ComfyUI's Anima wrapper (`preprocess_text_embeds`)
- `networks/lora_anima.py` — anima_lora LoRA targeting (`pooled_text_proj` exclusion at line 66)
- `scripts/convert_lora_to_comfy.py` — key-name translator for moving LoRAs between the two stacks
