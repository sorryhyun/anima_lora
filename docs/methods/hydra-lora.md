# HydraLoRA: Multi-Style Routing via Layer-Local Experts

MoE-style multi-head LoRA with per-module routing. Targets multi-artist training in a single LoRA without style bleed — a standard LoRA trained on multiple artists blends all styles into one shared low-rank subspace, so distinct fingerprints are lost. HydraLoRA attaches several `lora_up` heads per adapted `Linear` and lets a learned router pick a per-sample mixture.

> **For the structural walkthrough** (architecture, forward pass, why RMS-over-rank-R, load-balancing formula, orthogonalized experts and the cold-start deadlock, composition matrix), see **`docs/structure/hydralora.md`**. This doc is the usage / ops / decision-log reference.

## File format

Training state dict (runtime form, used inside the trainer):

```
<prefix>.lora_down.weight        # (rank, in_dim)    shared
<prefix>.lora_up_weight          # (E, out, rank)    stacked per-expert
<prefix>.router.weight           # (E, rank)
<prefix>.router.bias             # (E,)
<prefix>.alpha                   # scalar
<prefix>.inv_scale               # (in_dim,)  optional; only when channel_scale is set
```

`save_weights` produces two files side by side:

1. **`anima_hydra.safetensors`** — standard LoRA (baked-down): expert ups are averaged to a single `lora_up.weight`, routers stripped. ComfyUI drop-in, but routing is lost so it's effectively a uniform-prior approximation.
2. **`anima_hydra_moe.safetensors`** — full multi-head format: per-expert `lora_ups.N.weight`, routers preserved, attention modules split into separate `q_proj`/`k_proj`/`v_proj` so the ComfyUI custom node can map them to the ComfyUI model's attention key names. Shared tensors (`lora_down`, `alpha`, `router.*`, `inv_scale`) are cloned into each split component.

## Inference

### CLI (router-live)

`inference.py` auto-detects moe files by safetensors-header sniff (`library/inference/models.py:_is_hydra_moe`). When detected, static merge is skipped and the network is attached as dynamic forward hooks — the training-time `HydraLoRAModule.forward` runs for every adapted layer on every denoising step, reproducing the trained router's per-sample, per-layer behavior.

Static merge and router-live are mutually exclusive: mixing hydra moe files with regular LoRA files in one `--lora_weight` list is refused. P-GRAFT composes cleanly — the cutoff step toggles `network.enabled` for both, and `HydraLoRAModule` honors the flag.

Use `make test-hydra` (or `python tasks.py test-hydra`) to run inference against the latest moe output.

### ComfyUI (live routing)

Use the **Anima Adapter Loader** node (`custom_nodes/comfyui-hydralora/`), which installs per-Linear forward hooks that reproduce `HydraLoRAModule.forward` exactly — including σ-conditional routing when the checkpoint's router input is wider than `rank`. See `custom_nodes/comfyui-hydralora/README.md` for installation, hook mechanics, and changelog.

## Orthogonalized experts — fallback behavior

The `OrthoHydraLoRAExpModule` default (both `use_ortho = true` and `use_hydra = true`) is the structural deadlock fix described in `docs/structure/hydralora.md` §5. One operational detail worth knowing:

**Fallback.** If `min(out_dim, in_dim) < num_experts · lora_dim` the disjoint SVD-slice partition can't fit, so `P_bases` degenerates to the legacy shared `P_basis` replicated `E` times (with a warning in the log). In that case all experts start identical (shared basis + zero `S_p` + zero `lambda_layer`); `expert_warmup_ratio` is the only symmetry-breaker — narrow-layer Hydra must not run with `expert_warmup_ratio = 0`. Implementation: `networks/lora_modules/ortho.py:OrthoHydraLoRAExpModule`.

## Composition with other variants

- **T-LoRA** — timestep rank masking applies to `lora_down` (shared across experts), so it composes directly. HydraLoRA + T-LoRA is the configured default.
- **DoRA** — currently not supported on HydraLoRA. Each expert would need its own magnitude vector, which is an unimplemented extension.
- **OrthoLoRA** — supported via `OrthoHydraLoRAExpModule` (`networks/lora_modules/ortho.py`). Cayley-parameterized orthogonal `S_p` becomes per-expert (`(num_experts, r, r)`), `S_q` stays shared (matching the shared `lora_down` story). Activated by setting both `use_ortho = true` and `use_hydra = true` — this is the configured default.
- **Spectrum** — composes cleanly. Cached steps skip all transformer blocks entirely (router included), so hydra just runs fewer times.
- **Modulation guidance** — orthogonal. Touches AdaLN only, outside the hydra-adapted Linears.

## Evolution: global → layer-local

The first HydraLoRA implementation used a single global router that read max-pooled `crossattn_emb` (the text conditioning, post-T5 projection) and broadcast one gate distribution to every adapted layer for every timestep. That design was motivated by a k-means / NMI analysis showing that max-pooled `crossattn_emb` clusters cleanly by artist (NMI ≈ 0.93). It worked enough to train, but had two problems:

1. **One-layer-wide routing.** The same gate was applied everywhere, so experts couldn't specialize per layer (e.g. an expert that's strong in early blocks but irrelevant in late blocks had no way to express that).
2. **Router decoupled from DiT input distribution.** The training signal for the router came from a fixed, pre-computed text embedding — not from what the adapted module actually sees at denoising time — so the routing decision was blind to noise level and image content.

The current layer-local design drops the text-space clustering path and reads each adapted layer's actual input. Old checkpoints with `_hydra_router.*` keys are refused at load time (see `networks/lora_anima/`) with an error message pointing at retraining.

## Configuration

The HydraLoRA toggle block in `configs/methods/lora.toml` (and the dedicated `configs/gui-methods/hydralora.toml` for GUI users) controls:

- `use_hydra = true` — switches `module_class` to `HydraLoRAModule`.
- `num_experts = 4` — default. Higher values give more specialization capacity at the cost of more `(out_dim * rank)` parameters per module.
- `balance_loss_weight = 0.001` — Switch Transformer load-balancing coefficient. Lowered from 0.01 because the original value dominated the weak router-gradient signal and pinned every router to uniform. Raise back toward 0.01 if you observe expert collapse after the rank-R router fix (see "Fixes" below); lower further (or zero) if specialization stays too weak.

HydraLoRA requires `cache_llm_adapter_outputs = true` (same as standard LoRA in this repo — unrelated to routing, but the cached crossattn is assumed by the surrounding training plumbing).

## Fixes

### 2026-04-20 — rank-R router rewiring (checkpoint-breaking)

Diagnostic on `anima-hydra-0420-644` (step 644) and prior checkpoints showed the router was inert: `‖router.weight‖` never moved from Kaiming init, median gate-marginal entropy sat at 1.0000, dominant-top1 fraction ≈ 2e-4. The network behaved as a single rank-R LoRA averaged across 4 expert heads, paying 4× the parameter / compute cost for no specialization.

Root cause: `_compute_gate` mean-pooled the raw `in_dim`-wide layer input over the ~4096-token sequence. Zero-mean activations cancel by √N, so the pooled vector had per-channel std ≈ 0.008, logit spread ≈ 0.01, softmax gates `[0.25 ± 0.002]` — effectively constant across samples, so the router gradient was vanishing and the balance loss (0.01 at the time) was dominant and squeezed everything to uniform.

Applied:

1. **Pool after `lora_down`.** `_compute_gate` now takes the rank-R `lx` (post `lora_down`) and RMS-pools it across the sequence dim. Content survives aggregation; no DC-bias outliers; router parameter count drops ~64× (e.g. `2048 × 4 → 32 × 4`).
2. **Gate computed before T-LoRA mask / dropout** so the gate is identical at train and inference time.
3. **Balance-loss weight pre-cut** from 0.01 → 0.001 in `lora.toml`, `gui-methods/hydralora.toml`, `gui-methods/hydralora_sigma.toml`, since with real router gradient restored the old weight would dominate.
4. **Old-shape router refused at load.** `create_network_from_weights` raises when `router.weight.shape[1] != rank`, with a retrain message — pre-fix routers never learned anything, so there's no salvage path.
5. **OrthoHydraLoRAExpModule mirrored** with the same change. Pool runs on the post-`Q_eff` `lx` but *before* λ scaling — λ is zero-init, so pooling post-λ would zero the router input at step 0 and freeze gradient.
6. **ComfyUI live-routing hook updated** to mirror the training-time forward exactly (rank-R RMS pool). See `custom_nodes/comfyui-hydralora/README.md` for node-side details.

Exit criteria for the first retrain: `‖router.weight‖` at final step > 1.5× init (init for `(E=4, rank=32)` @ std=0.01 ≈ 0.113); median normalized entropy ∈ [0.6, 0.95]; mean dominant-top1 > 0.2; zero dead experts; `make test-hydra` quality ≥ non-hydra LoRA baseline; ComfyUI `Anima Adapter Loader` visually matches CLI at `strength_lora=1.0`.
