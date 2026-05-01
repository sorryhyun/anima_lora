# custom_nodes/comfyui-hydralora/

ComfyUI **Anima Adapter Loader** node — single loader that dispatches Anima adapter checkpoints (any mix of LoRA / HydraLoRA / ReFT / prefix / postfix / cond) through ComfyUI's patching system. Exists because vanilla ComfyUI's weight-patcher silently drops non-LoRA keys (`reft_*`, `lora_ups`, postfix vectors), so a Hydra/ReFT/postfix checkpoint loaded with a stock LoRA loader produces wrong output with no warning.

Full user-facing docs and changelog live in `README.md`. This file is for code-level edits to the node.

## Files

| File | Role |
|------|------|
| `adapter.py` | LoRA / Hydra / ReFT key parsing, classification, hook install. |
| `postfix.py` | Prefix / postfix / cond context splicing on `diffusion_model.forward`. |
| `nodes.py` | The `AnimaAdapterLoader` ComfyUI node definition. |
| `__init__.py` | Re-exports `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`. |

## Application paths (which key goes where)

The node sniffs the safetensors header and routes each component independently — paths are disjoint and run in the same load:

| Component | Application path |
|-----------|-----------------|
| Plain LoRA | `ModelPatcher.add_patches` (standard ComfyUI weight patch). |
| HydraLoRA | Per-Linear `forward_hook` installed via `ModelPatcher.add_object_patch` on each adapted Linear's `_forward_hooks`. |
| ReFT | Per-block `forward_hook` installed via `ModelPatcher.add_object_patch` on `diffusion_model.blocks.<idx>._forward_hooks`. |
| Prefix / postfix / cond | `ModelPatcher.add_object_patch` on `diffusion_model.forward`, splicing learned vectors into the T5-compatible crossattn embedding **after** the LLM adapter + pad-to-512 step. CFG-safe via `cond_or_uncond` from `transformer_options`. |

## Critical invariant: forward_hook, never override `forward`

For Hydra and ReFT, install a `forward_hook` — do **not** replace `block.forward` / `linear.forward`. Overriding `forward` strands weights on CPU under ComfyUI's cast-weights path: ComfyUI walks the real `forward` to drive its `comfy_cast_weights` machinery, and replacing the method confuses it — blocks end up with `comfy_cast_weights=False` and their Linears stay on CPU, producing a device mismatch at runtime. A hook leaves `forward` untouched, traces cleanly through `torch.compile`, and is properly reverted on `unpatch_model`.

Prefix/postfix is the exception (it patches `diffusion_model.forward` itself), but that's the model-level forward, not a per-Linear / per-block one — same rule, different scope.

## σ-conditional Hydra routing

Routing is data-driven: `router = Linear(rank + sigma_feature_dim, E)` with sinusoidal(σ) concatenated onto the pooled rank-R router input. A thin wrapper around `diffusion_model.forward` records the current `timesteps` into shared state on each denoising call; every Hydra hook reads it to build the σ feature. Detected automatically from `router.weight.shape[1] > rank` — old `sigma_mlp.*` checkpoints are no longer supported (see README §2.1.0).

## Coexistence

Plain-LoRA and Hydra paths target disjoint key prefixes (`_extract_lora_sd` skips `lora_ups.*`, `_parse_hydra` requires `lora_ups`), so a mixed checkpoint where only some Linears are Hydra-routed runs both paths in the same load without conflict. Don't reintroduce mutual-exclusion checks.

## Publishing

This node ships as a ComfyUI Registry package — bump version in `pyproject.toml`, push to GitHub, then `comfy node publish --token $COMFY_REG`. The token is in `anima_lora/.env`.
