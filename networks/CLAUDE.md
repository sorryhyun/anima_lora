# networks/

Pluggable adapter implementations selected at runtime via the `network_module` config
key (plus, for the LoRA family, the three-axis routing cfg). Each subdirectory is a
self-contained adapter family; `attention_dispatch.py` is the shared backend router used
by both training and inference.

## Layout

| Path | Role |
|------|------|
| `__init__.py` | `NetworkSpec` registry (`NETWORK_REGISTRY`) + the `NETWORK_KWARGS` TOML allowlist + `resolve_network_spec()` (maps the three-axis cfg → a registry entry). **`NETWORK_KWARGS` is auto-derived** (`_derive_network_kwargs` AST-scans `config.py` / `factory.py` / `__init__.py` for `kwargs.get("literal")` reads), so a new cfg+TOML net kwarg registers from its read alone — **just add the `kwargs.get("foo")` read, no separate allowlist edit** (recognizes the `kwargs.get` form only; a non-`kwargs.get` read won't be picked up). |
| `lora_anima/` | LoRA network creation, module targeting, timestep-masking orchestration, global routing. Split into `network.py` (assembly/runtime core), `network_metrics.py` (read-side metrics/diagnostics mixin — balance loss, router stats, ortho reg), `routers.py` (`GlobalRouter` / `FreqRouter` / `ContentRouter`, re-exported from `network.py` for back-compat), `factory.py`, `loading.py`, and `config.py`. |
| `lora_modules/` | Per-variant module implementations: `lora.py`, `ortho.py`, `hydra.py`, `stacked_experts.py`, `chimera.py`, `step_expert.py` (shared-down + K step-selected up-heads for the turbo DP-DMD student), plus `base.py` and `router_state.py` (shared σ/FEI/routing-weights buffer protocol for Hydra/OrthoHydra/StackedExperts — keeps the global-router grad path identical across all three; the per-module `set_sigma`/`set_fei`/`set_routing_weights`/`clear_*` method surface is the `RouterStateMixin` in that file, buffer-presence-guarded so a module inherits the full surface with absent buffers as safe no-ops). Training forwards compute their rank GEMMs in the **model compute dtype** (`org_forwarded.dtype` — NOT `x.dtype`; AdaLN LayerNorm hands fp32 under autocast(bf16). Bit-identical to the fp32-bottleneck path removed 2026-06-10 — see `tests/test_lora_dtype_policy.py` (the `lora_fp32_bottleneck` bench that proved it is also removed)); inference paths keep fp32. The removed custom-autograd Function was also an implicit partitioner boundary — its memory role is now covered by `activation_memory_budget` (base.toml, 0.99 — the settled knee, never 0.85; no-grad-ckpt runs only). Each module class owns its own save-pipeline hook (`distill_save_state_dict` / `build_moe_state_dict`) — the Cayley/SVD math and per-pool MoE layout live next to the variant that defined them. |
| `attn_fuse.py` | `AttnFuseSpec` + `iter_split_groups` + `match_fused_spec` — single source of truth for the runtime-fused `qkv_proj`/`kv_proj` ↔ on-disk split `q/k/v_proj` layout. Sits at the `networks/` top level so save (`lora_save.py`) and load (`lora_anima/loading.py`) both reach it without a cross-package import. |
| `lora_save.py`, `lora_utils.py` | Thin save-pipeline orchestrator + shared helpers. `lora_save.save_network_weights` calls each variant's `distill_save_state_dict` in fixed order, then dispatches to the matching `build_moe_state_dict`. Owns only the legacy sig-type OrthoLoRA distill (no live module class for it) and the variant-write sibling-file naming. |
| `methods/base.py` | Shared lifecycle base for the non-LoRA adapter networks (`easycontrol`, `soft_tokens`) — common `set_multiplier` / `is_mergeable` / `enable_gradient_checkpointing` trainer-facing protocol. |
| `protocol.py` | `typing.Protocol` description of the adapter-network surface every network duck-types to: `AdapterNetwork` (core trainer-facing lifecycle) + `RouterConditionableNetwork` (optional per-step routing setters, LoRA-family only). Not an enforced base — consumers keep `hasattr`-probing; the protocol is greppable docs + a contract test (`tests/test_adapter_protocol.py`, which also guards the inference↛training import boundary). |
| `methods/easycontrol.py` | EasyControl: per-block cond LoRA on self-attn (q/k/v/o) + FFN + scalar `b_cond` logit-bias gate; two-stream block forward at training, KV-cache prefill at inference. Opt-in target-stream adaln LoRA (`train_adaln`, cond-gated — method TOMLs pin it false against base.toml's LoRA-family default; see `docs/methods/adaln.md` §EasyControl). |
| `methods/turbo_dmd.py` | Turbo Anima DP-DMD distillation harness — owns student + fake `LoRANetwork` instances on one frozen DiT; output is a normal LoRA. See `docs/methods/turbo.md`. |
| `methods/soft_tokens.py`, `methods/ip_adapter_pe_lora.py` | Soft tokens (SoftREPA parameterization) + the PE-LoRA delta path (`inject_pe_lora`), vendored into the Anima-Tagger ComfyUI node. (IP-Adapter, its other consumer, was downgraded to `bench/ip_adapter/`.) |
| `register_injection.py` | `RegisterInjector` — shared DSR register-token machinery (`_run_blocks` wrap + mid-stack pre-hooks + rope extension + adoption metrics). Two owners: the standalone register method (`methods/register.py`) and the LoRA family (`num_registers > 0` in a lora TOML = full LoRA trained jointly with K learnable registers; param at top-level dot-free key `register_tokens`, own lr group at `unet_lr × register_lr_scale`, kept-live at inference — `is_mergeable()` False, merge refused, `load_dit_model` auto-detects the key and attaches dynamic hooks; REPA capture auto-trims the K trailing tokens). |
| `attention_dispatch.py` | Unified `dispatch_attention()` — backend router (SDPA / FA2 / FA3 / sageattn / flex). |
| `spectrum.py` | Spectrum inference acceleration (Chebyshev feature forecasting). See root CLAUDE.md §Spectrum and `docs/inference/spectrum.md`. |
| `spd.py` | Spectral Progressive Diffusion — training-free inference acceleration (grow spatial resolution along the trajectory, spectral noise-expansion handoff). Sampler-level runner registered like Spectrum. See `docs/inference/spd.md`. |
| `foveated.py` | Deferred-foveated merge — training-free inference acceleration (full grid above σ_c, then fovea tokens 1:1 + periphery 2×2-token groups merged via the `token_merger` forward kwarg; endogenous `combo` mask). Sampler-level runner registered like Spectrum/SPD. See `docs/inference/foveated.md`. |
| `calibration/` | Shipped calibration artifacts: `channel_stats.safetensors` + `cond_channel_stats.safetensors` (per-channel scaling, main + EasyControl cond stream — see `docs/optimizations/channel_scaling.md`; inert on frozen-basis ortho variants) + `cns_gamma.npz` (CNS γ schedule, also auto-downloaded from a release) + `dave_alpha.npz` (DAVE). |

## Three-axis routing surface (plan2)

As of commit `1dca212`, the LoRA-family routing flags collapsed into three orthogonal
cfg axes consumed by `lora_anima/config.py::LoRANetworkCfg.from_kwargs` and dispatched
by `__init__.py::resolve_network_spec`:

| Knob | Values | Meaning |
|---|---|---|
| `use_moe_style` | `False` / `"shared_A"` / `"independent_A"` | Expert layout — no experts, Hydra-style shared `lora_down`, or stacked per-expert `(lora_down, lora_up)`. |
| `route_per_layer` | `True` / `False` | Router location — per-Linear (Hydra default) or one network-level router. |
| `router_source` | `"none"` / `"input"` / `"sigma"` / `"fei"` / `"crossattn_emb"` | What signal the router reads — Linear input, σ-features, FEI on `z_t`, pooled cross-attention text features (the DiT's K/V), or no router. `"input"` requires `route_per_layer=True`; `"crossattn_emb"` requires `route_per_layer=False`. |

The matrix cells map to concrete variants: plain LoRA / OrthoLoRA / T-LoRA
(`use_moe_style=False`), HydraLoRA and its σ/FEI-routed forms (`"shared_A"` + per-layer
router — FEI-on-Hydra is the lora.toml default), FeRA (`"independent_A"` +
`GlobalRouter`), and the text-routed `"crossattn_emb"` cell. Pre-plan2 metadata stamps
(`ss_use_hydra`, `ss_use_fei_router`, `ss_network_module = "networks.methods.fera"`)
**no longer load**; the new stamps are `ss_use_moe_style` / `ss_route_per_layer` /
`ss_router_source`. `use_ortho` (Cayley/PSOFT) and `use_ortho_init` (trainable
SVD-seeded bases) are per-module bools, mutually exclusive.

**Load the `lora-routing` skill before adding/changing a variant or touching routing
code** — it holds the full variant matrix, per-variant module details
(LoRA/Ortho/OrthoInit/T-LoRA/Hydra/FeRA, ReFT's removal), ortho/ortho_init composition
rules, and the `GlobalRouter` mechanics (zero-init gates, `set_fei` reference-write into
every module's `_routing_weights` buffer, and the router-collapse failure mode).

## Attn fuse spec (qkv/kv fuse↔split)

`attn_fuse.py::AttnFuseSpec` + `iter_split_groups` + `match_fused_spec` is the single
source of truth for the runtime-fused `qkv_proj` (self-attn) / `kv_proj` (cross-attn) ↔
on-disk split `q/k/v_proj` layout. ComfyUI's cosmos backbone uses the split layout while
Anima's training-side DiT uses the fused projections; save always writes split, load
always re-fuses. Both `lora_save.py` and `loading.py` walk the same specs, so adding a
new fused projection only needs one entry here.

## Attention dispatch

`attention_dispatch.py::dispatch_attention()` routes to the active backend (torch SDPA,
flash-attn v2/v3, sageattn, flex attention). **Tensor layout differs by backend** — BHLD
for SDPA/sageattn, BLHD for flash-attn — so callers must hand tensors to the dispatcher
in a known layout and the dispatcher transposes as needed. Check the backend branches
before adding new attention call sites.

FA4 (flash-attention-sm120) was evaluated and is currently disabled — see
`docs/optimizations/fa4.md`. The KV-trim + LSE-correction path that depended on FA4 was
removed (the `crossattn_full_len` field and `trim_crossattn_kv` flag are gone as of
2026-05-20); only the `flash4` branch stub remains in the dispatcher. See fa4.md for
what re-enabling FA4 would entail.

## compile_blocks() and forward hooks

`compile_blocks()` compiles `block._forward`, **not** `block.__call__`
(`library/anima/models.py::compile_blocks`). Consequences for hook-based feature capture
(REPA, functional loss, probe tooling):

- `register_forward_hook` on a **block** survives compilation — `__call__`'s hook
  machinery runs eagerly around the compiled inner.
- Hooks on submodules *invoked inside* `_forward` are traced over under compile — don't
  rely on them firing.
- Under compile, captured block outputs arrive in **native-flatten layout `(B, 1, seq,
  1, D)`**; eager runs keep the 5D `(B, 1, H, W, D)` patch grid. Capture consumers must
  handle both.
- A hook that never fires silently turns the feature into a no-op — warn once at first
  consume if nothing was captured (pattern: `_warned_no_capture` in
  `library/training/repa.py`).

## Timestep masking — when to update what

T-LoRA's mask is a single CPU/GPU buffer shared across all adapted Linears, updated once
per denoising step from `lora_anima/network.py`. Anything that calls into LoRA modules
during a forward must have the mask set for the current `t` already — `factory.py` and
`network.py` are the only places that should be poking `set_timestep_mask` /
`clear_timestep_mask`. New adapter variants that want timestep awareness should reuse
the same buffer pattern (register as a buffer in `base.py`, read it inside `forward`)
rather than threading `t` through every call site.
