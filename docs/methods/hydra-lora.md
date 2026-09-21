# HydraLoRA: Multi-Style Routing via Layer-Local Experts

MoE-style multi-head LoRA with per-module routing. Targets multi-artist training in a single LoRA without style bleed — a standard LoRA trained on multiple artists blends all styles into one shared low-rank subspace, so distinct fingerprints are lost. HydraLoRA attaches several `lora_up` heads per adapted `Linear` and lets a learned router pick a per-sample mixture.

Paper: [arXiv:2605.03252](https://arxiv.org/abs/2605.03252).

> For the structural walkthrough (architecture, forward pass, why RMS-over-rank-R, load-balancing formula, orthogonalized experts and the cold-start deadlock, composition matrix), see `docs/structure/hydralora.md`. This doc is the usage / ops / decision-log reference.

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

1. `anima_hydra.safetensors` — standard LoRA (baked-down): expert ups are averaged to a single `lora_up.weight`, routers stripped. ComfyUI drop-in, but routing is lost so it's effectively a uniform-prior approximation.
2. `anima_hydra_moe.safetensors` — full multi-head format: per-expert `lora_ups.N.weight`, routers preserved, attention modules split into separate `q_proj`/`k_proj`/`v_proj` so the ComfyUI custom node can map them to the ComfyUI model's attention key names. Shared tensors (`lora_down`, `alpha`, `router.*`, `inv_scale`) are cloned into each split component.

## Inference

### CLI (router-live)

`inference.py` auto-detects moe files by safetensors-header sniff (`library/inference/models.py:_is_hydra_moe`). When detected, static merge is skipped and the network is attached as dynamic forward hooks — the training-time `HydraLoRAModule.forward` runs for every adapted layer on every denoising step, reproducing the trained router's per-sample, per-layer behavior.

Static merge and router-live are mutually exclusive: mixing hydra moe files with regular LoRA files in one `--lora_weight` list is refused. P-GRAFT composes cleanly — the cutoff step toggles `network.enabled` for both, and `HydraLoRAModule` honors the flag.

Use `make test-hydra` (or `python tasks.py test-hydra`) to run inference against the latest moe output.

### ComfyUI (live routing)

Use the Anima Adapter Loader node (`https://github.com/sorryhyun/ComfyUI-Anima_lora-Adapter`), which installs per-Linear forward hooks that reproduce `HydraLoRAModule.forward` exactly — including σ-conditional routing when the checkpoint's router input is wider than `rank`. See `https://github.com/sorryhyun/ComfyUI-Anima_lora-Adapter` for installation, hook mechanics, and changelog.

## Composition with other variants

- T-LoRA — timestep rank masking applies to `lora_down` (shared across experts), so it composes directly; `configs/gui-methods/hydralora.toml` enables both.
- Spectrum — composes cleanly. Cached steps skip all transformer blocks entirely (router included), so hydra just runs fewer times.
- Modulation guidance — orthogonal. Touches AdaLN only, outside the hydra-adapted Linears.

## Checkpoints from the global router

The first HydraLoRA used one global router on max-pooled `crossattn_emb`, broadcast to every layer and timestep. The layer-local design reads each adapted layer's actual input instead; checkpoints with `_hydra_router.*` keys are refused at load with a retrain message.

## Configuration

`configs/gui-methods/hydralora.toml` is the shipped variant. HydraLoRA is one cell of the three-axis routing surface ([`../guidelines/training.md`](../guidelines/training.md#lora-family--the-three-axis-surface)):

- `use_moe_style = "shared_A"` + `route_per_layer = true` + `router_source` (`"sigma"` in the shipped variant) — selects `HydraLoRAModule`. The retired `use_hydra` key raises.
- `num_experts` — code default 4, shipped variant 6. More experts add `(out_dim * rank)` parameters per module.
- `balance_loss_weight` — Switch Transformer load-balancing coefficient (shipped variant 3e-7; the ceiling on Anima is ~5e-5). Raise it if experts collapse; lower it if specialization stays weak.

HydraLoRA requires `cache_llm_adapter_outputs = true` (same as standard LoRA in this repo).

### Hard σ-band partition

`specialize_experts_by_sigma_buckets = true` (with `num_sigma_buckets > 1` and `num_experts % num_sigma_buckets == 0`) partitions the E experts into B σ-bands. For a sample at σ in band b, only the in-band experts can win the gate (out-of-band logits masked to `-inf` before softmax). Soft routing still operates within a band.

- Layout: interleaved. Expert e belongs to band `e mod B` — see `networks/lora_modules/hydra.py::_register_sigma_band_partition`.
- Edges: optional. `sigma_bucket_boundaries = [0.0, 0.5, 0.8, 1.0]` (length B+1, strictly increasing, 0.0 → 1.0) overrides the default uniform `linspace(0, 1, B+1)`. Lets you concentrate capacity in a chosen σ regime — e.g. wide low-σ band, narrow high-σ band — while keeping equal experts per band. With variable bucket widths under uniform σ sampling, narrow buckets see fewer training samples per band; consider oversampling those σ ranges if you want their experts to converge as fast.

Both fields are stamped into safetensors metadata (`ss_specialize_experts_by_sigma_buckets`, `ss_num_sigma_buckets`, `ss_sigma_bucket_boundaries`) so inference (CLI + ComfyUI) reconstructs the partition exactly.

## Fixes

### 2026-04-20 — rank-R router rewiring (checkpoint-breaking)

Diagnostic on `anima-hydra-0420-644` (step 644) and prior checkpoints showed the router was inert: `‖router.weight‖` never moved from Kaiming init, median gate-marginal entropy sat at 1.0000, dominant-top1 fraction ≈ 2e-4. The network behaved as a single rank-R LoRA averaged across 4 expert heads, paying 4× the parameter / compute cost for no specialization.

Root cause: `_compute_gate` mean-pooled the raw `in_dim`-wide layer input over the ~4096-token sequence. Zero-mean activations cancel by √N, so the pooled vector had per-channel std ≈ 0.008, logit spread ≈ 0.01, softmax gates `[0.25 ± 0.002]` — effectively constant across samples, so the router gradient was vanishing and the balance loss (0.01 at the time) was dominant and squeezed everything to uniform.

Applied:

1. Pool after `lora_down`. `_compute_gate` now takes the rank-R `lx` (post `lora_down`) and RMS-pools it across the sequence dim. Content survives aggregation; no DC-bias outliers; router parameter count drops ~64× (e.g. `2048 × 4 → 32 × 4`).
2. Gate computed before T-LoRA mask / dropout so the gate is identical at train and inference time.
3. Balance-loss weight pre-cut from 0.01 → 0.001 in `lora.toml`, `gui-methods/hydralora.toml`, `gui-methods/hydralora_sigma.toml`, since with real router gradient restored the old weight would dominate.
4. **Old-shape router refused at load.** `create_network_from_weights` raises when `router.weight.shape[1] != rank`, with a retrain message — pre-fix routers never learned anything, so there's no salvage path.
5. OrthoHydraLoRAModule mirrored with the same change. Pool runs on the post-`Q_eff` `lx` but *before* λ scaling — λ is zero-init, so pooling post-λ would zero the router input at step 0 and freeze gradient.
6. ComfyUI live-routing hook updated to mirror the training-time forward exactly (rank-R RMS pool). See `https://github.com/sorryhyun/ComfyUI-Anima_lora-Adapter` for node-side details.

Exit criteria for the first retrain: `‖router.weight‖` at final step > 1.5× init (init for `(E=4, rank=32)` @ std=0.01 ≈ 0.113); median normalized entropy ∈ [0.6, 0.95]; mean dominant-top1 > 0.2; zero dead experts; `make test-hydra` quality ≥ non-hydra LoRA baseline; ComfyUI `Anima Adapter Loader` visually matches CLI at `strength_lora=1.0`.
