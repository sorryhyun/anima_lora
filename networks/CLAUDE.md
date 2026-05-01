# networks/

Pluggable adapter implementations selected at runtime via the `network_module` config key. Each subdirectory is a self-contained adapter family; `attention_dispatch.py` is the shared backend router used by both training and inference.

## Layout

| Path | Role |
|------|------|
| `lora_anima/` | LoRA network creation, module targeting, timestep-masking orchestration. Split into `network.py`, `factory.py`, `loading.py`, `config.py`. |
| `lora_modules/` | Per-variant module implementations: `lora.py`, `ortho.py`, `hydra.py`, `reft.py`, plus `base.py` and `custom_autograd.py`. |
| `lora_save.py`, `lora_utils.py` | Save-time SVD distillation (OrthoLoRA → plain LoRA) and shared helpers. |
| `methods/postfix.py` | Continuous postfix tuning: learns N vectors appended to adapter cross-attention (modes: hidden, embedding, cfg, dual). |
| `methods/ip_adapter.py` | IP-Adapter: PE-Core-L14-336 vision encoder + Perceiver resampler + per-block `to_k_ip`/`to_v_ip`. |
| `methods/easycontrol.py` | EasyControl: per-block cond LoRA on self-attn (q/k/v/o) + FFN + scalar `b_cond` logit-bias gate; two-stream block forward at training, KV-cache prefill at inference. |
| `methods/apex.py` | APEX `ConditionShift` module (`c_fake = A·c + b`). |
| `attention_dispatch.py` | Unified `dispatch_attention()` — backend router (SDPA / xformers / FA2 / FA3 / sageattn / flex). |
| `spectrum.py` | Spectrum inference acceleration (Chebyshev feature forecasting). See root CLAUDE.md §Spectrum and `docs/methods/spectrum.md`. |

## LoRA variants

All live in `lora_modules/`. Stack freely via toggle flags in `configs/methods/lora.toml` — the default stacks LoRA + OrthoLoRA + T-LoRA + ReFT together.

- **LoRA** (`lora.py::LoRAModule`) — Classic low-rank: `y = x + (x @ down @ up) * scale * multiplier`.
- **OrthoLoRA** (`ortho.py::OrthoLoRAExpModule`, `OrthoHydraLoRAExpModule`) — SVD-based orthogonal parameterization with orthogonality regularization (linear layers only). Saved as plain LoRA via thin SVD on ΔW at save time. See `docs/methods/psoft-integrated-ortholora.md`.
- **T-LoRA** — Not a separate class. A `_timestep_mask` buffer on `LoRAModule` / `OrthoLoRAExpModule` (registered in `base.py`) is rebound to a shared live-updated mask by `lora_anima/network.py::LoRANetwork.set_timestep_mask`. Effective rank varies with denoising step via a power-law schedule. **Training-only** — inference runs full rank at every t (baking into DiT is bit-equivalent). See `docs/methods/timestep_mask.md`.
- **HydraLoRA** (`hydra.py`) — MoE-style multi-head routing: shared `lora_down` + per-expert `lora_up_i` heads, layer-local router on the adapted Linear's input. Requires `cache_llm_adapter_outputs=true`. Produces a `*_moe.safetensors` sibling for router-live inference. See `docs/methods/hydra-lora.md`.
- **ReFT** (`reft.py`) — Block-level residual-stream intervention (LoReFT, Wu et al. NeurIPS 2024). One `ReFTModule` per selected DiT block wraps the block's `forward` and adds `R^T·(ΔW·h + b)·scale` to the output; orthogonality regularized on `R`. Additive side-channel, composes with any LoRA variant, lives in the same `.safetensors`. Vanilla ComfyUI can't load ReFT (weight-patcher silently drops `reft_*` keys) — use the `AnimaAdapterLoader` custom node (`custom_nodes/comfyui-hydralora/`).

## Attention dispatch

`attention_dispatch.py::dispatch_attention()` routes to the active backend (torch SDPA, xformers, flash-attn v2/v3, sageattn, flex attention). **Tensor layout differs by backend** — BHLD for SDPA/sageattn, BLHD for xformers/flash-attn — so callers must hand tensors to the dispatcher in a known layout and the dispatcher transposes as needed. Check the backend branches before adding new attention call sites.

FA4 (flash-attention-sm120) was evaluated and is currently disabled — see `docs/optimizations/fa4.md`.

### Flash4 LSE correction

When cross-attention KV is trimmed (zero-padding removed for efficiency), the softmax denominator must be corrected. `attention_dispatch.py` applies a sigmoid-based LSE correction using `crossattn_full_len` to account for removed zero-key contributions. This pairs with the text-encoder-padding invariant in root CLAUDE.md — both must hold for cross-attention to produce correct output.

## Timestep masking — when to update what

T-LoRA's mask is a single CPU/GPU buffer shared across all adapted Linears, updated once per denoising step from `lora_anima/network.py`. Anything that calls into LoRA modules during a forward must have the mask set for the current `t` already — `factory.py` and `network.py` are the only places that should be poking `set_timestep_mask` / `clear_timestep_mask`. New adapter variants that want timestep awareness should reuse the same buffer pattern (register as a buffer in `base.py`, read it inside `forward`) rather than threading `t` through every call site.
