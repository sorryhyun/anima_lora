---
name: lora-routing
description: The LoRA-family three-axis routing surface (use_moe_style / route_per_layer / router_source) — the variant matrix, per-variant module details (LoRA/Ortho/OrthoInit/T-LoRA/Hydra/FeRA), and GlobalRouter mechanics. Load before adding/changing a LoRA variant, touching routing code in networks/lora_anima/ or lora_modules/, or debugging router behavior.
---

# LoRA-family routing: three-axis surface, variants, GlobalRouter

## Variant matrix

The three axes (`use_moe_style` / `route_per_layer` / `router_source`) are consumed by `lora_anima/config.py::LoRANetworkCfg.from_kwargs` and dispatched by `networks/__init__.py::resolve_network_spec`. Legal values:

| Axis | Values |
|---|---|
| `use_moe_style` | `False` / `"shared_A"` / `"independent_A"` |
| `route_per_layer` | `true` / `false` |
| `router_source` | `"none"` / `"input"` / `"sigma"` / `"fei"` / `"crossattn_emb"` (empty / unset → `"none"`) |

Two constraints, both raising in `from_kwargs`: `"input"` requires `route_per_layer=True` (no per-Linear input signal reaches a network-level router); `"crossattn_emb"` requires `route_per_layer=False`.

Variants that exist as cells in this matrix:

| Variant | `use_moe_style` | `route_per_layer` | `router_source` | Network module / path |
|---|---|---|---|---|
| Plain LoRA / OrthoLoRA / T-LoRA | `False` | — | `"none"` | `lora_anima` + `lora_modules/` (LoRA, ortho) |
| HydraLoRA (paper) | `"shared_A"` | `True` | `"input"` | `lora_anima` + `lora_modules/hydra.py` |
| σ-router on Hydra | `"shared_A"` | `True` | `"sigma"` | same |
| FEI-on-Hydra (lora.toml default) | `"shared_A"` | `True` | `"fei"` | same |
| **FeRA (author-faithful)** | `"independent_A"` | `False` | `"fei"` | `lora_anima` + `lora_modules/stacked_experts.py` + `GlobalRouter` |
| Text-routed Hydra / FeRA | `"shared_A"` / `"independent_A"` | `False` | `"crossattn_emb"` | `lora_anima` + `GlobalRouter` (pools + LN on the cross-attn text vector) |

The `"crossattn_emb"` cell routes the whole pool by **prompt content** (pooled post-LLM-adapter text features) instead of σ/noise-frequency: the network-level `GlobalRouter` reads the same vector the DiT cross-attends to, fired per cond/uncond branch via `set_crossattn_routing` (train, `train.py`) / `set_hydra_crossattn` (inference, `library/inference/generation.py`), broadcasting to the standard `_routing_weights` slot.

Pre-plan2 metadata stamps (`ss_use_hydra`, `ss_use_fei_router`, `ss_network_module = "networks.methods.fera"`) **no longer load**; the stamps are now `ss_use_moe_style` / `ss_route_per_layer` / `ss_router_source`.

## Ortho / OrthoInit knobs

Both are per-module bools, **mutually exclusive**:

- `use_ortho=true` — PSOFT-style Cayley-rotated SVD parameterization (OrthoLoRA, OrthoHydra, `StackedExpertsLoRAModule`).
- `use_ortho_init=true` — the sibling **OrthoInit** variant (`ortho_init` spec): same top-r SVD seed, but the bases are *trainable* (no Cayley, no frozen subspace), so ΔW is uncapped with a W₀-aligned warm start.

The resolver raises on `use_ortho_init` + `use_moe_style` (Hydra/FeRA), **but it composes with `use_chimera_hydra`** — threaded via `cfg.use_ortho_init`, it swaps each chimera pool's frozen-basis+Cayley for trainable SVD-seeded bases (ΔW=0 at init still holds from the centered uniform gate; distills with R=I to the identical `*_chimera.safetensors` layout). Standalone OrthoInit distills to standard LoRA at save, so the on-disk/merge/inference path is identical to a distilled OrthoLoRA.

## LoRA variants

All live in `networks/lora_modules/`. Stack freely via toggle flags in `configs/methods/lora.toml`.

- **LoRA** (`lora.py::LoRAModule`) — Classic low-rank: `y = x + (x @ down @ up) * scale * multiplier`.
- **OrthoLoRA** (`ortho.py::OrthoLoRAModule`, `OrthoHydraLoRAModule`) — SVD-based orthogonal parameterization with orthogonality regularization (linear layers only). Caps `colspace(ΔW) ⊆ top-r(W₀)`. Saved as plain LoRA via thin SVD on ΔW at save time. See `docs/methods/svd-down-lora.md`.
- **OrthoInit** (`ortho.py::OrthoInitLoRAModule`) — top-r SVD of W₀ as *initialization only*: trainable `P_init`/`Q_init` (no Cayley, no frozen basis) + `lambda_layer` (λ=0 → ΔW=0 at init), so ΔW reaches any rank-r subspace. Composes with the T-LoRA `_timestep_mask` (gates the singular values λ). Distills to standard LoRA (sqrt-split λ → down/up) at save.
- **T-LoRA** — Not a separate class. A `_timestep_mask` buffer on `LoRAModule` / `OrthoLoRAModule` (registered in `base.py`) is rebound to a shared live-updated mask by `lora_anima/network.py::LoRANetwork.set_timestep_mask`. Effective rank varies with denoising step via a power-law schedule. **Training-only** — inference runs full rank at every t (baking into DiT is bit-equivalent). See `docs/methods/timestep_mask.md`.
- **HydraLoRA** (`hydra.py`) — MoE-style multi-head routing: shared `lora_down` + per-expert `lora_up_i` heads, layer-local router on the adapted Linear's input (`router_source="input"`) or σ-features / FEI features (`"sigma"` / `"fei"`). With `route_per_layer=False` the per-layer router drops out for a network-level `GlobalRouter` fed σ-features, FEI, or pooled cross-attn text (`router_source="crossattn_emb"`). Requires `cache_llm_adapter_outputs=true`. Produces a `*_moe.safetensors` sibling for router-live inference. See `docs/methods/hydra-lora.md`.
- **Stacked experts / FeRA** (`stacked_experts.py::StackedExpertsLoRAModule`) — Independent-A layout: each expert owns its own `(lora_down, lora_up)`, stacked as `(E, …)` Parameters consumed in one `einsum`. Routed by `GlobalRouter` (one network-level router fed by FEI of `z_t`). Supports both free and PSOFT-style ortho parameterization. Independent-A did not beat shared-A FEI-routed Hydra on Anima, so this cell is legacy; the live FEI-routing home is `docs/experimental/chimera-hydra.md`.

> **ReFT was removed from the live tree on 2026-06-08** and downgraded to a bench probe — module, configs, docs and a re-integration map live in `bench/reft/` (`INTEGRATION.md` + `impl/`).

## GlobalRouter (network-level routing)

`lora_anima/routers.py::GlobalRouter` (re-exported from `network.py` for back-compat, alongside `FreqRouter` / `ContentRouter` used by chimera) — `Linear(F_in → H) → ReLU → Linear(H → E) → softmax/τ`. Built when `cfg.route_per_layer=False` and `cfg.use_moe_style != False`. Final layer is zero-init so step-0 gates are uniform; warmup is the symmetry-breaker. Under `router_source="crossattn_emb"` the router is built with `apply_layer_norm=True` and `input_dim=CROSSATTN_EMB_DIM`; its `forward` RMS-pools a raw `(B, L, D)` text tensor over the sequence axis and LayerNorms (parameterless) before the MLP — no extra state_dict keys, on/off is deterministic from `router_source`.

Hook site: `LoRANetwork.set_fei(z_t)` runs the FEI computation (via `library/runtime/fei.py`) and the router once, then writes the resulting `(B, num_experts)` tensor by reference into each routing-aware module's `_routing_weights` buffer. One Python-level write propagates to every adapted Linear that step — hence the failure mode to watch for: **router collapse takes every layer down together**.

Training-loop call: `train.py` fires `network.set_fei(noisy_model_input)` at the per-step σ/FEI hook block when the cfg has `route_per_layer=False` and `router_source="fei"`. Inference: `library/inference/generation.py` mirrors the same call before each Euler step.
