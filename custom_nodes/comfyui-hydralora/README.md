# Anima Adapter Loader (ComfyUI)

A single ComfyUI node that loads Anima adapter checkpoints and dispatches their components through ComfyUI's patching system. Auto-detects what the file contains — any combination of plain LoRA, HydraLoRA (live per-Linear routing), LoReFT (residual-stream edits), and prefix / postfix / cond context splices — and routes each to its correct application path.

Algorithm-level notes live in the main `docs/methods/` tree (`hydra-lora.md`, `reft.md`, `postfix-sigma.md`, `prefix-tuning.md`). This README covers only what's ComfyUI-specific: detection, installation paths, and the node's changelog.

## Install

Drop `custom_nodes/comfyui-hydralora/` (this directory) into your ComfyUI `custom_nodes/`, restart ComfyUI. The node appears as **Anima Adapter Loader** in the loaders menu.

## The loader

One node, two independently-toggled sections with separate strength controls:

| Input | Purpose |
|-------|---------|
| `adapter_name` | safetensors file holding any mix of LoRA / Hydra / ReFT keys |
| `strength_lora` | scales LoRA + HydraLoRA delta (set 0 to disable both while keeping ReFT) |
| `strength_reft` | scales ReFT residual edit (set 0 to disable ReFT while keeping LoRA) |
| `postfix_name` | separate safetensors for prefix / postfix / cond context splicing |
| `postfix_strength` | scales the postfix / prefix delta |

The adapter section sniffs the safetensors header and routes each component independently — you get correct behavior whether the file contains plain LoRA, a `*_moe.safetensors` hydra checkpoint, a ReFT-only file, or any combination. The two strength sliders are useful for ablation ("is it the LoRA or the ReFT doing the anatomy fix?") and for dialing back either branch when one overshoots.

## How each component applies

**Plain LoRA** → `ModelPatcher.add_patches`, the standard ComfyUI weight-patch path.

**HydraLoRA** (live routing) → per-Linear `forward_hook` installed via `ModelPatcher.add_object_patch` on each adapted Linear's `_forward_hooks`. The hook replays `HydraLoRAModule.forward` exactly: rank-R `lora_down` projection, RMS pool over the sequence dim, optional sinusoidal(σ) concatenated onto the pooled vector, `Linear(rank + sigma_feature_dim, E)` router, softmax, gate-weighted expert `lora_up` blend. Routing is data-driven, so `strength_lora` is a single slider — per-expert controls would not be meaningful under live routing.

σ-conditional routing: a thin wrapper around `diffusion_model.forward` records the current `timesteps` into shared state on each denoising call; every hydra hook reads it to build the sinusoidal σ features. Detected automatically from `router.weight.shape[1] > rank`.

**ReFT** → per-block `forward_hook` installed via `ModelPatcher.add_object_patch` on `diffusion_model.blocks.<idx>._forward_hooks`. The hook adds `R^T · (ΔW · h + b) · scale · strength` to the block output.

**Prefix / postfix / cond** → `ModelPatcher.add_object_patch` on `diffusion_model.forward`, splicing learned vectors into the T5-compatible crossattn embedding *after* the LLM adapter + pad-to-512 step. Positive-batch rows only via `cond_or_uncond` from `transformer_options` (CFG-safe).

## Why forward hooks, not `forward` override

For both HydraLoRA and ReFT we install a `forward_hook` rather than overriding `block.forward` / `linear.forward`. Overriding `forward` strands weights on CPU under ComfyUI's cast-weights path: ComfyUI relies on walking the real `forward` to drive its `comfy_cast_weights` machinery, and replacing the method confused it — blocks ended up with `comfy_cast_weights=False` and their Linears stayed on CPU, producing a device mismatch at runtime. A hook leaves `forward` untouched, traces cleanly through `torch.compile`, and is properly reverted on `unpatch_model`.

## Code layout

| File | Role |
|------|------|
| `adapter.py` | LoRA / Hydra / ReFT loading, parsing, hook install |
| `postfix.py` | Prefix / postfix / cond context splicing |
| `nodes.py` | The `AnimaAdapterLoader` node |
| `__init__.py` | Re-exports `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` |

## Changelog

### 2.2.0 — 2026-05-02 — σ-band partition reconstruction + perf cleanup

Catches the node up to the training-side σ-band specialization (commit `bf37e3e`). When `specialize_experts_by_sigma_buckets = true` is on at training, the expert→band lookup buffer (`_expert_band`) is registered non-persistent, so it doesn't ride along in the safetensors and the hook had no way to re-derive it. Inference therefore ran soft routing across all `E` experts, silently ignoring the partition baked into the router weights.

Applied in `adapter.py`:

1. `load_adapter` now opens the safetensors metadata alongside the weights and propagates `ss_specialize_experts_by_sigma_buckets`, `ss_num_sigma_buckets`, and (optional) `ss_sigma_bucket_boundaries` into the parsed hydra bundle. Divisibility (`num_experts % num_buckets == 0`) is validated; mismatches log a warning and disable the partition rather than crashing.
2. `_make_hydra_hook` rebuilds `expert_band` from `num_sigma_buckets` using the **interleaved** `e mod B` rule, matching the training-side switch in `_register_sigma_band_partition`. Out-of-band expert logits are masked to `-inf` before softmax.
3. Custom σ-bucket edges (`ss_sigma_bucket_boundaries`, length `B+1`, monotone `0.0 → 1.0`) override the default uniform `linspace`, so checkpoints with capacity concentrated on a chosen σ regime — e.g. `[0.0, 0.5, 0.8, 1.0]` for late-step refinement — bucket samples the same way training did.
4. Hot-path fp32 casts (`.float()` on `lora_down`, `lora_ups`, `router_w`, `router_b`, `inv_scale`, and on `sigma`) are hoisted out of the per-call hydra hook into device-migration (one-shot) and a normalized `sigma_pre_hook` (once per denoising step). Eliminates the per-Linear-per-compile `DeviceCopy` warning torch.compile was emitting; behavior is unchanged.

### 2.1.1 — 2026-04-29 — CPU-stranding fix on lowvram path

Capturing σ via `add_object_patch("diffusion_model.forward", …)` stranded sub-Linears (e.g. cosmos `x_embedder.proj`) on CPU under ComfyUI's lowvram-aware load path — the same failure mode that retired the old `block.forward` override in favor of `_forward_hooks`. Replaced the wrapper with a forward pre-hook on `diffusion_model._forward_pre_hooks`; the hook records `args[1]` (timesteps) into the shared σ state read by each hydra hook, leaving `forward` untouched.

### 2.1.0 — 2026-04-21 — σ-input catch-up + plain-LoRA fall-through

Training had moved σ from an additive `sigma_mlp` bias on router logits to a direct router-input feature: `router = Linear(rank + sigma_feature_dim, E)` with sinusoidal(σ) concatenated onto the pooled rank-R vector (see `docs/methods/hydra-lora.md` §Fixes, 2026-04-20). The node hadn't been updated — it still looked for `sigma_mlp.*` keys and refused routers whose second dim wasn't exactly `rank`, so every σ-conditional hydra checkpoint skipped all hydra modules. In mixed checkpoints (`hydra_router_layers` = mlp only), the `elif` fall-through to plain LoRA also didn't fire, so cross_attn / self_attn adapters went unapplied too.

Applied in `adapter.py`:

1. `_parse_hydra` drops `sigma_mlp.*` parsing and filters to modules with `lora_ups` so plain-LoRA prefixes stop surfacing as `missing lora_down/lora_ups` skip warnings.
2. `_apply_hydra_live_to_model` derives `sigma_feature_dim = router_w.shape[1] - rank` (≥ 0) instead of refusing non-rank router inputs.
3. `_make_hydra_hook` concatenates sinusoidal(σ) onto the pooled rank-R router input (broadcast when σ is shape `(1,)` vs CFG-doubled batch); additive bias path removed.
4. `apply_adapter` runs the plain-LoRA path whenever `bundle["lora"]` is present, not only when hydra is absent. The two paths target disjoint prefixes (`_extract_lora_sd` skips `lora_ups.*`, `_parse_hydra` requires `lora_ups`), so coexistence is safe.

### 2.0.0 — 2026-04-20 — rank-R router rewiring

Live-routing hook updated to mirror the training-time forward exactly: RMS pool over the sequence dim of the post-`lora_down` rank-R signal, not mean-pool over the raw layer input. Corresponding training fix is in `docs/methods/hydra-lora.md` §Fixes (2026-04-20 entry) — pre-fix routers never learned, so old checkpoints are refused at load.
