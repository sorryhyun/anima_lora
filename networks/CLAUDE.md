# networks/

Pluggable adapter implementations selected at runtime via the `network_module` config
key (plus, for the LoRA family, the three-axis routing cfg). Each subdirectory is a
self-contained adapter family; `attention_dispatch.py` is the shared backend router used
by both training and inference.

## Layout

| Path | Role |
|------|------|
| `__init__.py` | `NetworkSpec` registry (`NETWORK_REGISTRY`) + the `NETWORK_KWARGS` TOML allowlist + `resolve_network_spec()` (maps the three-axis cfg → a registry entry). **`NETWORK_KWARGS` is auto-derived** (`_derive_network_kwargs` AST-scans `config.py` / `factory.py` / `__init__.py`): a new net kwarg registers from its `kwargs.get("foo")` read alone — any other read form (`kwargs["foo"]`, a helper) is not picked up. |
| `lora_anima/` | LoRA network creation, module targeting, timestep-masking orchestration, global routing. Split into `network.py` (assembly/runtime core), `network_metrics.py` (read-side metrics/diagnostics mixin — balance loss, router stats, ortho reg), `routers.py` (`GlobalRouter` / `FreqRouter` / `ContentRouter`, also re-exported from `network.py`), `factory.py`, `loading.py`, and `config.py`. |
| `lora_modules/` | Per-variant modules: `lora.py`, `ortho.py`, `hydra.py`, `stacked_experts.py`, `chimera.py`, `step_expert.py` (shared down + K step-selected up-heads, turbo student), `base.py` (shared forward scaffold), `router_state.py` (σ/FEI/routing-weights buffers + `RouterStateMixin` setters). Training forwards compute rank GEMMs in the **model compute dtype** (`org_forwarded.dtype`, not `x.dtype` — AdaLN LayerNorm hands fp32 under autocast(bf16); pinned by `tests/test_lora_dtype_policy.py`); Hydra and `ChimeraHydraInferenceModule` compute in fp32 at inference. Rank-GEMM activation memory is governed by `activation_memory_budget` (base.toml 0.99 — the settled knee, never 0.85; no-grad-ckpt runs only). |
| `attn_fuse.py` | `AttnFuseSpec` + `iter_split_groups` + `match_fused_spec` — the fuse↔split layout contract (§Attn fuse spec). |
| `lora_save.py`, `lora_utils.py` | Save pipeline + shared helpers. `save_network_weights` calls each module class's `distill_save_state_dict` in a load-bearing order (see the `lora_save.py` docstring), then its `build_moe_state_dict`. |
| `methods/base.py` | `AdapterNetworkBase` — shared trainer-facing lifecycle for the non-LoRA networks (EasyControl, SoftTokens, Register). |
| `protocol.py` | `typing.Protocol` description of the duck-typed adapter surface: `AdapterNetwork` (core lifecycle) + `RouterConditionableNetwork` (per-step routing setters, LoRA family only). Not an enforced base — consumers `hasattr`-probe; guarded by `tests/test_adapter_protocol.py` (which also guards the inference↛training import boundary). |
| `methods/easycontrol.py` | EasyControl: per-block cond LoRA on self-attn (q/k/v/o) + FFN + scalar `b_cond` logit-bias gate; two-stream block forward at training, KV-cache prefill at inference. Opt-in target-stream adaln LoRA (`train_adaln`) — EasyControl TOMLs must pin it false, since base.toml turns it on for the LoRA family (`docs/methods/adaln.md` §EasyControl). |
| `methods/turbo_dmd.py` | Turbo Anima DP-DMD distillation harness — owns student + fake `LoRANetwork` instances on one frozen DiT; output is a normal LoRA. See `docs/methods/turbo.md`. |
| `methods/soft_tokens.py`, `methods/ip_adapter_pe_lora.py` | Soft tokens (SoftREPA parameterization) + the PE-LoRA delta path (`inject_pe_lora`), vendored into the Anima-Tagger ComfyUI node. |
| `register_injection.py` | `RegisterInjector` — DSR register-token injection (`_run_blocks` wrap + mid-stack pre-hooks + rope extension). Owners: `methods/register.py` (`--method register`) and the LoRA family (`num_registers > 0`: LoRA + K learnable registers under the top-level key `register_tokens`, lr `unet_lr × register_lr_scale`). Registers can't merge — the checkpoint stays kept-live (`is_mergeable()` False, merge refused); `load_dit_model` auto-detects the key; REPA capture trims the K trailing tokens. |
| `grad_basis.py` | Gradient-SVD `lora_down` init (`down_init="grad_svd"` / `"basis_file"`). The basis is **depth-baked** (`load_basis` refuses a block-count mismatch); `grad_svd` sketches in `train.py` before the network is built and is refused under block swap. `docs/methods/svd-down-lora.md` §Gradient-seeded siblings. |
| `attention_dispatch.py` | Unified `dispatch_attention()` — backend router (SDPA / FA2 / FA3 / sageattn / flex). |
| `spectrum.py` | Spectrum inference acceleration (Chebyshev feature forecasting). See `docs/inference/spectrum.md`. |
| `spd.py` | Spectral Progressive Diffusion — training-free inference acceleration (grow spatial resolution along the trajectory, spectral noise-expansion handoff). Sampler-level runner registered like Spectrum. See `docs/inference/spd.md`. |
| `foveated.py` | Deferred-foveated merge — training-free inference acceleration (full grid above σ_c, then fovea tokens 1:1 + periphery 2×2-token groups merged via the `token_merger` forward kwarg; endogenous `combo` mask). Sampler-level runner registered like Spectrum/SPD. See `docs/inference/foveated.md`. |
| `calibration/` | Shipped artifacts: `channel_stats.safetensors` + `cond_channel_stats.safetensors` (per-channel scaling, main + EasyControl cond stream; `docs/optimizations/channel_scaling.md`), `cns_gamma.npz` (CNS γ), `dave_alpha.npz` (DAVE). |

## Three-axis routing surface

The LoRA-family routing cfg is three orthogonal axes — `use_moe_style` /
`route_per_layer` / `router_source` — parsed by
`lora_anima/config.py::LoRANetworkCfg.from_kwargs` and dispatched by
`__init__.py::resolve_network_spec` to a `NETWORK_REGISTRY` entry. `use_ortho`
(Cayley/PSOFT) and `use_ortho_init` (trainable SVD-seeded bases) are per-module bools,
mutually exclusive.

**Load the `lora-routing` skill before adding/changing a variant or touching routing
code** — it holds the axis values and their constraints, the full variant matrix,
per-variant module details (LoRA/Ortho/OrthoInit/T-LoRA/Hydra/FeRA), the metadata stamps
(`ss_use_moe_style` / `ss_route_per_layer` / `ss_router_source`, and which unstamped
checkpoints don't load), ortho/ortho_init composition rules, and the `GlobalRouter`
mechanics (zero-init gates, `set_fei` reference-write into every module's
`_routing_weights` buffer, and the router-collapse failure mode).

## Attn fuse spec (qkv/kv fuse↔split)

The training DiT fuses `qkv_proj` (self-attn) / `kv_proj` (cross-attn); ComfyUI's cosmos
backbone uses split `q/k/v_proj`. Save always writes split, load always re-fuses. Both
`lora_save.py` and `loading.py` walk `ATTN_FUSE_SPECS`, so a new fused projection needs
one entry there.

## Attention dispatch

`attention_dispatch.py::dispatch_attention()` routes to the active backend (torch SDPA,
flash-attn v2/v3, sageattn, flex attention). **Tensor layout differs by backend** — BHLD
for SDPA/sageattn, BLHD for flash-attn. `dispatch_attention` takes `[B, L, H, D]` and
transposes per backend; check the backend branches before adding a call site.

FA4 (flash-attention-sm120) is disabled — `attn_mode="flash4"` raises in the dispatcher.
Re-enable notes: `docs/optimizations/fa4.md`.

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

## Timestep masking

T-LoRA's mask is one shared buffer per distinct module rank (a `reg_dims` override
yields mixed ranks), owned by
`lora_anima/network.py::LoRANetwork.set_timestep_mask` / `clear_timestep_mask` and fired
once per step from `library/training/forward/router_conditioning.py`. Anything that calls
into LoRA modules during a forward must have the mask set for the current `t` already.
New adapter variants that want timestep awareness should reuse the same buffer pattern
(register as a buffer in `base.py`, read it inside `forward`) rather than threading `t`
through every call site.
