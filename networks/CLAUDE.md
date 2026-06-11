# networks/

Pluggable adapter implementations selected at runtime via the `network_module` config key (plus, for the LoRA family, the three-axis routing cfg). Each subdirectory is a self-contained adapter family; `attention_dispatch.py` is the shared backend router used by both training and inference.

## Layout

| Path | Role |
|------|------|
| `__init__.py` | `NetworkSpec` registry (`NETWORK_REGISTRY`) + the flat `NETWORK_KWARGS` TOML allowlist (one set, mirrors what `LoRANetworkCfg.from_kwargs` reads) + `resolve_network_spec()` (maps the three-axis cfg → a registry entry). **A new cfg+TOML net kwarg is inert and fails the config test until added to `NETWORK_KWARGS` here.** |
| `lora_anima/` | LoRA network creation, module targeting, timestep-masking orchestration, global routing. Split into `network.py` (assembly/runtime core), `network_metrics.py` (read-side metrics/diagnostics mixin — balance loss, router stats, ortho reg), `routers.py` (`GlobalRouter` / `FreqRouter` / `ContentRouter`, re-exported from `network.py` for back-compat), `factory.py`, `loading.py`, and `config.py`. |
| `lora_modules/` | Per-variant module implementations: `lora.py`, `ortho.py`, `hydra.py`, `stacked_experts.py`, `chimera.py`, `step_expert.py` (shared-down + K step-selected up-heads for the turbo DP-DMD student), plus `base.py` and `router_state.py` (shared σ/FEI/routing-weights buffer protocol for Hydra/OrthoHydra/StackedExperts — keeps the global-router grad path identical across all three; the per-module `set_sigma`/`set_fei`/`set_routing_weights`/`clear_*` method surface is the `RouterStateMixin` in that file, buffer-presence-guarded so a module inherits the full surface with absent buffers as safe no-ops). Training forwards compute their rank GEMMs in the **model compute dtype** (`org_forwarded.dtype` — NOT `x.dtype`; AdaLN LayerNorm hands fp32 under autocast(bf16). Bit-identical to the fp32-bottleneck path removed 2026-06-10 — see `bench/lora_fp32_bottleneck` + `tests/test_lora_dtype_policy.py`); inference paths keep fp32. The removed custom-autograd Function was also an implicit partitioner boundary — its memory role is now covered by `activation_memory_budget` (base.toml, 0.85; no-grad-ckpt runs only). Each module class owns its own save-pipeline hook (`distill_save_state_dict` / `build_moe_state_dict`) — the Cayley/SVD math and per-pool MoE layout live next to the variant that defined them. |
| `attn_fuse.py` | `AttnFuseSpec` + `iter_split_groups` + `match_fused_spec` — single source of truth for the runtime-fused `qkv_proj`/`kv_proj` ↔ on-disk split `q/k/v_proj` layout. Sits at the `networks/` top level so save (`lora_save.py`) and load (`lora_anima/loading.py`) both reach it without a cross-package import. |
| `lora_save.py`, `lora_utils.py` | Thin save-pipeline orchestrator + shared helpers. `lora_save.save_network_weights` calls each variant's `distill_save_state_dict` in fixed order, then dispatches to the matching `build_moe_state_dict`. Owns only the legacy sig-type OrthoLoRA distill (no live module class for it) and the variant-write sibling-file naming. |
| `methods/base.py` | Shared lifecycle base for the non-LoRA adapter networks (`easycontrol`, `soft_tokens`) — common `set_multiplier` / `is_mergeable` / `enable_gradient_checkpointing` trainer-facing protocol. |
| `protocol.py` | `typing.Protocol` description of the adapter-network surface every network duck-types to: `AdapterNetwork` (core trainer-facing lifecycle) + `RouterConditionableNetwork` (optional per-step routing setters, LoRA-family only). Not an enforced base — consumers keep `hasattr`-probing; the protocol is greppable docs + a contract test (`tests/test_adapter_protocol.py`, which also guards the inference↛training import boundary). |
| `methods/easycontrol.py` | EasyControl: per-block cond LoRA on self-attn (q/k/v/o) + FFN + scalar `b_cond` logit-bias gate; two-stream block forward at training, KV-cache prefill at inference. |
| `methods/turbo_dmd.py` | Turbo Anima DP-DMD distillation harness — owns student + fake `LoRANetwork` instances on one frozen DiT; output is a normal LoRA. See `docs/experimental/dpdmd.md`. |
| `methods/soft_tokens.py`, `methods/ip_adapter_pe_lora.py` | Soft tokens (SoftREPA parameterization) + the PE-LoRA delta path (`inject_pe_lora`), vendored into the Anima-Tagger ComfyUI node. (IP-Adapter, its other consumer, was downgraded to `bench/ip_adapter/`.) |
| `attention_dispatch.py` | Unified `dispatch_attention()` — backend router (SDPA / FA2 / FA3 / sageattn / flex). |
| `spectrum.py` | Spectrum inference acceleration (Chebyshev feature forecasting). See root CLAUDE.md §Spectrum and `docs/inference/spectrum.md`. |
| `spd.py` | Spectral Progressive Diffusion — training-free inference acceleration (grow spatial resolution along the trajectory, spectral noise-expansion handoff). Sampler-level runner registered like Spectrum. See `docs/inference/spd.md`. |
| `dcw.py` | DCW post-step correction for SNR-t bias on flow-matching DiTs at the sampler boundary. See `docs/inference/dcw.md`. |
| `calibration/` | Shipped calibration artifacts: `channel_stats.safetensors` + `cond_channel_stats.safetensors` (per-channel scaling, main + EasyControl cond stream — see `docs/optimizations/channel_scaling.md`; inert on frozen-basis ortho variants) + `cns_gamma.npz` (CNS γ schedule, also auto-downloaded from a release) + `dave_alpha.npz` (DAVE). |

## Three-axis routing surface (plan2)

As of commit `1dca212`, the LoRA-family routing flags collapsed into three orthogonal cfg axes consumed by `lora_anima/config.py::LoRANetworkCfg.from_kwargs` and dispatched by `__init__.py::resolve_network_spec`:

| Knob | Values | Meaning |
|---|---|---|
| `use_moe_style` | `False` / `"shared_A"` / `"independent_A"` | Expert layout — no experts, Hydra-style shared `lora_down`, or stacked per-expert `(lora_down, lora_up)`. |
| `route_per_layer` | `True` / `False` | Router location — per-Linear (Hydra default) or one network-level router. |
| `router_source` | `"none"` / `"input"` / `"sigma"` / `"fei"` / `"crossattn_emb"` | What signal the router reads — Linear input, σ-features, FEI on `z_t`, pooled cross-attention text features (the DiT's K/V), or no router. `"input"` requires `route_per_layer=True`; `"crossattn_emb"` requires `route_per_layer=False`. |

Variants that exist as cells in this matrix:

| Variant | `use_moe_style` | `route_per_layer` | `router_source` | Network module / path |
|---|---|---|---|---|
| Plain LoRA / OrthoLoRA / T-LoRA | `False` | — | `"none"` | `lora_anima` + `lora_modules/` (LoRA, ortho) |
| HydraLoRA (paper) | `"shared_A"` | `True` | `"input"` | `lora_anima` + `lora_modules/hydra.py` |
| σ-router on Hydra | `"shared_A"` | `True` | `"sigma"` | same |
| FEI-on-Hydra (lora.toml default) | `"shared_A"` | `True` | `"fei"` | same |
| **FeRA (author-faithful)** | `"independent_A"` | `False` | `"fei"` | `lora_anima` + `lora_modules/stacked_experts.py` + `GlobalRouter` |
| Text-routed Hydra / FeRA | `"shared_A"` / `"independent_A"` | `False` | `"crossattn_emb"` | `lora_anima` + `GlobalRouter` (pools + LN on the cross-attn text vector) |

The `"crossattn_emb"` cell routes the whole pool by **prompt content** (pooled post-LLM-adapter text features) rather than by σ/noise-frequency — the network-level `GlobalRouter` reads the same vector the DiT cross-attends to, fired per cond/uncond branch via `set_crossattn_routing` (train) / `set_hydra_crossattn` (inference). It is the non-chimera analogue of chimera's `content_router_source="crossattn_emb"` knob, broadcasting to the standard `_routing_weights` slot.

Pre-plan2 metadata stamps (`ss_use_hydra`, `ss_use_fei_router`, `ss_network_module = "networks.methods.fera"`) **no longer load** — the legacy fallback was removed in plan2 task #6. The new stamps are `ss_use_moe_style` / `ss_route_per_layer` / `ss_router_source`.

`ortho` stays a per-module bool — set `use_ortho=true` to get the PSOFT-style Cayley-rotated SVD parameterization (applies to OrthoLoRA, OrthoHydra, and `StackedExpertsLoRAModule`). `use_ortho_init=true` selects the sibling **OrthoInit** variant (`ortho_init` spec) — same top-r SVD seed but the bases are *trainable* (no Cayley, no frozen subspace), so ΔW is uncapped (full LoRA expressivity) with a W₀-aligned warm start. Mutually exclusive with `use_ortho`; the resolver still raises on the plain `use_moe_style` (Hydra/FeRA) combos, **but it now composes with `use_chimera_hydra`** — `use_ortho_init=true` swaps each chimera pool's frozen-basis+Cayley for trainable SVD-seeded bases (threaded via `cfg.use_ortho_init`; ΔW=0 at init still holds from the centered uniform gate, distills with R=I to the identical `*_chimera.safetensors` layout). Standalone OrthoInit distills to standard LoRA at save, so the on-disk/merge/inference path is identical to a distilled OrthoLoRA.

## LoRA variants

All live in `lora_modules/`. Stack freely via toggle flags in `configs/methods/lora.toml`.

- **LoRA** (`lora.py::LoRAModule`) — Classic low-rank: `y = x + (x @ down @ up) * scale * multiplier`.
- **OrthoLoRA** (`ortho.py::OrthoLoRAModule`, `OrthoHydraLoRAModule`) — SVD-based orthogonal parameterization with orthogonality regularization (linear layers only). Saved as plain LoRA via thin SVD on ΔW at save time. See `docs/methods/psoft-integrated-ortholora.md`.
- **OrthoInit** (`ortho.py::OrthoInitLoRAModule`) — top-r SVD of W₀ as *initialization only*: trainable `P_init`/`Q_init` (no Cayley, no frozen basis) + `lambda_layer` (λ=0 → ΔW=0 at init). Full LoRA expressivity (ΔW reaches any rank-r subspace) with a W₀-aligned warm start; the fix for "OrthoLoRA / T-LoRA-ortho feels too weak" (OrthoLoRA caps `colspace(ΔW) ⊆ top-r(W₀)`). Composes with the T-LoRA `_timestep_mask` (gates the singular values λ). Distills to standard LoRA (sqrt-split λ → down/up) at save.
- **T-LoRA** — Not a separate class. A `_timestep_mask` buffer on `LoRAModule` / `OrthoLoRAModule` (registered in `base.py`) is rebound to a shared live-updated mask by `lora_anima/network.py::LoRANetwork.set_timestep_mask`. Effective rank varies with denoising step via a power-law schedule. **Training-only** — inference runs full rank at every t (baking into DiT is bit-equivalent). See `docs/methods/timestep_mask.md`.
- **HydraLoRA** (`hydra.py`) — MoE-style multi-head routing: shared `lora_down` + per-expert `lora_up_i` heads, layer-local router on the adapted Linear's input (`router_source="input"`) or σ-features / FEI features (`"sigma"` / `"fei"`). With `route_per_layer=False` the per-layer router drops out for a network-level `GlobalRouter` fed σ-features, FEI, or pooled cross-attn text (`router_source="crossattn_emb"`). Requires `cache_llm_adapter_outputs=true`. Produces a `*_moe.safetensors` sibling for router-live inference. See `docs/methods/hydra-lora.md`.
- **Stacked experts / FeRA** (`stacked_experts.py::StackedExpertsLoRAModule`) — Independent-A layout: each expert owns its own `(lora_down, lora_up)`, stacked as `(E, …)` Parameters consumed in one `einsum`. Routed by `GlobalRouter` (one network-level router fed by FEI of `z_t`). Supports both free and PSOFT-style ortho parameterization. Independent-A did not beat shared-A FEI-routed Hydra on Anima, so this cell is unbenched/legacy; the live FEI-routing home is `docs/experimental/chimera-hydra.md`.

> **ReFT was removed from the live tree on 2026-06-08** and downgraded to a bench probe. The module, configs, docs, and a full re-integration map live in `bench/reft/` (`INTEGRATION.md` + `impl/`). Re-integrate only if the bench in `bench/reft/plan.md` shows it earns a niche.

## GlobalRouter (network-level routing)

`lora_anima/routers.py::GlobalRouter` (re-exported from `network.py` for back-compat, alongside `FreqRouter` / `ContentRouter` used by chimera) — `Linear(F_in → H) → ReLU → Linear(H → E) → softmax/τ`. Built when `cfg.route_per_layer=False` and `cfg.use_moe_style != False`. Final layer is zero-init so step-0 gates are uniform; warmup is the symmetry-breaker. Under `router_source="crossattn_emb"` the router is built with `apply_layer_norm=True` and `input_dim=CROSSATTN_EMB_DIM`; its `forward` RMS-pools a raw `(B, L, D)` text tensor over the sequence axis and LayerNorms (parameterless) before the MLP — no extra state_dict keys, on/off is deterministic from `router_source`.

Hook site: `LoRANetwork.set_fei(z_t)` runs the FEI computation (via `library/runtime/fei.py`) and the router once, then writes the resulting `(B, num_experts)` tensor by reference into each routing-aware module's `_routing_weights` buffer. One Python-level write propagates to every adapted Linear that step — that's the architectural commitment of the "global router" design and the failure mode to watch for (router collapse → every layer collapses together).

Training-loop call: `train.py` fires `network.set_fei(noisy_model_input)` at the per-step σ/FEI hook block when the cfg has `route_per_layer=False` and `router_source="fei"`. Inference: `library/inference/generation.py` mirrors the same call before each Euler step.

## Attn fuse spec (qkv/kv fuse↔split)

`attn_fuse.py::AttnFuseSpec` + `iter_split_groups` + `match_fused_spec` is the single source of truth for the runtime-fused `qkv_proj` (self-attn) / `kv_proj` (cross-attn) ↔ on-disk split `q/k/v_proj` layout. ComfyUI's cosmos backbone uses the split layout while Anima's training-side DiT uses the fused projections; save always writes split, load always re-fuses. Both `lora_save.py` and `loading.py` walk the same specs, so adding a new fused projection only needs one entry here.

## Attention dispatch

`attention_dispatch.py::dispatch_attention()` routes to the active backend (torch SDPA, flash-attn v2/v3, sageattn, flex attention). **Tensor layout differs by backend** — BHLD for SDPA/sageattn, BLHD for flash-attn — so callers must hand tensors to the dispatcher in a known layout and the dispatcher transposes as needed. Check the backend branches before adding new attention call sites.

FA4 (flash-attention-sm120) was evaluated and is currently disabled — see `docs/optimizations/fa4.md`. The KV-trim + LSE-correction path that depended on FA4 was removed (the `crossattn_full_len` field and `trim_crossattn_kv` flag are gone as of 2026-05-20); only the `flash4` branch stub remains in the dispatcher. See fa4.md for what re-enabling FA4 would entail.

## Timestep masking — when to update what

T-LoRA's mask is a single CPU/GPU buffer shared across all adapted Linears, updated once per denoising step from `lora_anima/network.py`. Anything that calls into LoRA modules during a forward must have the mask set for the current `t` already — `factory.py` and `network.py` are the only places that should be poking `set_timestep_mask` / `clear_timestep_mask`. New adapter variants that want timestep awareness should reuse the same buffer pattern (register as a buffer in `base.py`, read it inside `forward`) rather than threading `t` through every call site.
