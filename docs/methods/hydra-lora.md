# HydraLoRA: Multi-Style Routing via Layer-Local Experts

MoE-style multi-head LoRA with per-module routing. Targets multi-artist
training in a single LoRA without style bleed.

## Motivation

A standard LoRA trained on multiple artists blends all styles into one shared
low-rank subspace, causing distinct artists to lose their visual identity.
HydraLoRA attaches several `lora_up` heads per adapted `Linear` and lets a
learned router pick a per-sample mixture of those heads, so distinct clusters
of samples can push different heads in different directions without
interfering with each other.

## Architecture

Each adapted `Linear` owns a `HydraLoRAModule` containing:

- **Shared `lora_down`** — a single `(rank, in_dim)` matrix, shared across
  experts. Captures common low-rank features.
- **Fused `lora_up_weight`** — a stacked parameter of shape
  `(num_experts, out_dim, rank)`. Each slice `[i]` is one expert head.
- **Layer-local `router`** — a `Linear(lora_dim, num_experts)` that reads the
  post-`lora_down` rank-R activations (not raw input, not the text embedding)
  and emits softmax gates over the experts for this specific layer and sample.

Forward (simplified, see `networks/lora_modules.py:HydraLoRAModule.forward`):

```python
lx       = self.lora_down(x)                           # (B, L, rank)

# RMS pool the rank-R signal over the sequence dim — see "Why RMS over
# rank-R" below. Computed before T-LoRA masking / dropout so the gate is
# identical at train and inference time.
pooled   = lx.reshape(B, -1, rank).pow(2).mean(dim=1).sqrt()  # (B, rank)
gate     = softmax(self.router(pooled), dim=-1)        # (B, num_experts)

combined = einsum("be,eod->bod", gate, lora_up_weight) # (B, out, rank) per-sample
out      = bmm(lx, combined.transpose(1, 2))           # (B, L, out)
delta    = out * multiplier * scale
```

Key properties:

- **Layer-local.** Each adapted module has its own router with its own weights.
  The same sample gets different gate distributions at different layers, so
  specialization is learned per-layer rather than as one global "style pick"
  applied uniformly everywhere.
- **Sample-dependent.** The effective `lora_up` varies per sample (via the
  batch-dim on `gate`). There is no single static `lora_up` that reproduces
  the trained behavior — averaging experts collapses the router to a uniform
  prior, which is not what was trained.
- **Small router overhead.** Per-module router params are `lora_dim *
  num_experts` plus bias. For `lora_dim=32, num_experts=4`, that's 132 params
  per module — ~64× smaller than the previous `in_dim`-wide router and
  negligible against the LoRA parameters themselves.

## Why RMS over rank-R (not mean pool over raw input)

Earlier revisions mean-pooled the raw layer input (`in_dim`-wide) before the
router. That choice was motivated by DC-bias outlier channels in DiT layer
inputs (peak/mean ratio 80–96×, documented in
`bench/channel_dominance_analysis.md`) — max pool would saturate softmax in
bf16, so mean pool was picked to average them out. But mean pool over a
~4096-token sequence cancels zero-mean activations by √N, collapsing per-
channel std to ~σ_x/√L ≈ 0.008; the only channels that survived were the
DC-bias outliers, which are layer-constant (not sample-dependent). The
router input was therefore near-identical across every sample, its gradient
was tiny, and the balance loss quietly pinned gates to uniform. End-to-end
measurement on live checkpoints (`anima-hydra-0420-644` and earlier):
median normalized entropy 1.0000 across 196 modules × 4 experts, 0/784 dead
experts, dominant-top1 fraction ≈ 2e-4, and `‖router.weight‖` pinned to its
Kaiming-init value at every step — confirming the router received no
effective gradient.

The current design pools the *post-`lora_down`* rank-R signal with RMS
(`sqrt(mean(x**2))`). RMS does not cancel under random signs, so sample-
level content survives aggregation over long sequences. Rank-R space is
bounded by `‖lora_down‖·‖x‖` and has no large DC-bias outliers, so softmax
stability in bf16 is not a concern. The router weight is still initialized
at `std=0.01` so starting gates are near-uniform and every expert receives
gradient at step 0.

## Load-balancing

Without a penalty, training collapses into using one or two experts — the
rest receive no gradient. HydraLoRA uses the Switch Transformer
load-balancing loss, averaged over all hydra modules:

```
L_balance = α · num_experts · Σᵢ (frac_i · gate_mean_i)
```

where `frac_i` is the fraction of samples (across the batch, at this layer)
whose dominant expert is `i`, and `gate_mean_i` is the mean gate value for
expert `i`. The network-level code sums this over every `HydraLoRAModule._last_gate`
cached during the forward pass, so the penalty integrates routing pressure
from every layer at once. Default `α = 0.01` in `configs/methods/hydralora.toml`.

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

1. **`anima_hydra.safetensors`** — standard LoRA (baked-down): expert ups are
   averaged to a single `lora_up.weight`, routers stripped. ComfyUI drop-in,
   but routing is lost so it's effectively a uniform-prior approximation.
2. **`anima_hydra_moe.safetensors`** — full multi-head format: per-expert
   `lora_ups.N.weight`, routers preserved, attention modules split into
   separate `q_proj`/`k_proj`/`v_proj` so the ComfyUI custom node can map
   them to the ComfyUI model's attention key names. Shared tensors
   (`lora_down`, `alpha`, `router.*`, `inv_scale`) are cloned into each
   split component.

## Inference

### CLI (router-live)

`inference.py` auto-detects moe files by safetensors-header sniff
(`library/inference_pipeline.py:_is_hydra_moe`). When detected, static merge
is skipped and the network is attached as dynamic forward hooks — the
training-time `HydraLoRAModule.forward` runs for every adapted layer on every
denoising step, reproducing the trained router's per-sample, per-layer
behavior.

Static merge and router-live are mutually exclusive: mixing hydra moe files
with regular LoRA files in one `--lora_weight` list is refused. P-GRAFT
composes cleanly — the cutoff step toggles `network.enabled` for both, and
`HydraLoRAModule` honors the flag.

Use `make test-hydra` (or `python tasks.py test-hydra`) to run inference
against the latest moe output.

### ComfyUI (live routing)

The `custom_nodes/comfyui-hydralora` **Anima Adapter Loader** node loads a
`*_moe.safetensors` and installs **per-Linear forward hooks** that
reproduce `HydraLoRAModule.forward` exactly: each adapted Linear computes
its own per-sample router gate from its own input and blends the
per-expert `lora_up` heads accordingly. A single `strength_lora` slider
scales the resulting delta; per-expert controls would not be meaningful
under live routing because the gate is data-driven.

σ-conditional router bias is supported when `sigma_mlp.*` keys are
present in the checkpoint. A thin wrapper around `diffusion_model.forward`
records the current `timesteps` into shared state on each denoising
call, and every hydra hook reads it to compute the σ-conditional bias
before softmax — matching the CLI's `set_sigma` path.

Implementation lives in `custom_nodes/comfyui-hydralora/adapter.py`
(`_apply_hydra_live_to_model`, `_make_hydra_hook`,
`_make_sigma_capture_wrapper`). Hooks are installed via
`ModelPatcher.add_object_patch` on each Linear's `_forward_hooks` (same
pattern as ReFT — overriding `forward` strands weights on CPU under
ComfyUI's cast-weights path).

## Composition with other variants

- **T-LoRA** — timestep rank masking applies to `lora_down` (shared across
  experts), so it composes directly. HydraLoRA + T-LoRA is the configured
  default.
- **DoRA** — currently not supported on HydraLoRA. Each expert would need
  its own magnitude vector, which is an unimplemented extension.
- **OrthoLoRA** — supported via `OrthoHydraLoRAExpModule`
  (`networks/lora_modules.py`). Cayley-parameterized orthogonal `S_p`
  becomes per-expert (`(num_experts, r, r)`), `S_q` stays shared (matching
  the shared `lora_down` story). Activated by setting both `use_ortho =
  true` and `use_hydra = true` — this is the configured default in
  `configs/methods/lora.toml` and `configs/gui-methods/hydralora.toml`.
- **Spectrum** — composes cleanly. Cached steps skip all transformer blocks
  entirely (router included), so hydra just runs fewer times.
- **Modulation guidance** — orthogonal. Touches AdaLN only, outside the
  hydra-adapted Linears.

## Evolution: global → layer-local

The first HydraLoRA implementation used a single global router that read
max-pooled `crossattn_emb` (the text conditioning, post-T5 projection) and
broadcast one gate distribution to every adapted layer for every timestep.
That design was motivated by a k-means / NMI analysis showing that
max-pooled `crossattn_emb` clusters cleanly by artist (NMI ≈ 0.93). It
worked enough to train, but had two problems:

1. **One-layer-wide routing.** The same gate was applied everywhere, so
   experts couldn't specialize per layer (e.g. an expert that's strong in
   early blocks but irrelevant in late blocks had no way to express that).
2. **Router decoupled from DiT input distribution.** The training signal for
   the router came from a fixed, pre-computed text embedding — not from what
   the adapted module actually sees at denoising time — so the routing
   decision was blind to noise level and image content.

The current layer-local design drops the text-space clustering path and
reads each adapted layer's actual input. Old checkpoints with `_hydra_router.*`
keys are refused at load time (see
`networks/lora_anima.py:create_network_from_weights_anima`) with an error
message pointing at retraining.

## Configuration

`configs/methods/hydralora.toml` controls:

- `use_hydra = true` — switches `module_class` to `HydraLoRAModule`.
- `num_experts = 4` — default. Higher values give more specialization capacity
  at the cost of more `(out_dim * rank)` parameters per module.
- `balance_loss_weight = 0.001` — Switch Transformer load-balancing
  coefficient. Lowered from 0.01 because the original value dominated the
  weak router-gradient signal and pinned every router to uniform. Raise back
  toward 0.01 if you observe expert collapse after the rank-R router fix
  (see "Fixes" below); lower further (or zero) if specialization stays too
  weak.

HydraLoRA requires `cache_llm_adapter_outputs = true` (same as standard LoRA
in this repo — unrelated to routing, but the cached crossattn is assumed by
the surrounding training plumbing).

## Fixes

### 2026-04-20 — rank-R router rewiring (checkpoint-breaking)

Diagnostic on `anima-hydra-0420-644` (step 644) and prior checkpoints showed
the router was inert: `‖router.weight‖` never moved from Kaiming init,
median gate-marginal entropy sat at 1.0000, dominant-top1 fraction ≈ 2e-4.
The network behaved as a single rank-R LoRA averaged across 4 expert heads,
paying 4× the parameter / compute cost for no specialization.

Root cause: `_compute_gate` mean-pooled the raw `in_dim`-wide layer input
over the ~4096-token sequence. Zero-mean activations cancel by √N, so the
pooled vector had per-channel std ≈ 0.008, logit spread ≈ 0.01, softmax
gates `[0.25 ± 0.002]` — effectively constant across samples, so the
router gradient was vanishing and the balance loss (0.01 at the time) was
dominant and squeezed everything to uniform.

Applied:

1. **Pool after `lora_down`.** `_compute_gate` now takes the rank-R `lx`
   (post `lora_down`) and RMS-pools it across the sequence dim. Content
   survives aggregation; no DC-bias outliers; router parameter count drops
   ~64× (e.g. `2048 × 4 → 32 × 4`).
2. **Gate computed before T-LoRA mask / dropout** so the gate is identical
   at train and inference time.
3. **Balance-loss weight pre-cut** from 0.01 → 0.001 in `lora.toml`,
   `gui-methods/hydralora.toml`, `gui-methods/hydralora_sigma.toml`, since
   with real router gradient restored the old weight would dominate.
4. **Old-shape router refused at load.** `create_network_from_weights`
   raises when `router.weight.shape[1] != rank`, with a retrain message —
   pre-fix routers never learned anything, so there's no salvage path. The
   ComfyUI `Anima Adapter Loader` skips-and-warns on the same mismatch.
5. **OrthoHydraLoRAExpModule mirrored** with the same change. Pool runs on
   the post-`Q_eff` `lx` but *before* λ scaling — λ is zero-init, so
   pooling post-λ would zero the router input at step 0 and freeze
   gradient.
6. **ComfyUI live-routing hook updated** to mirror the training-time
   forward exactly (rank-R RMS pool).

Exit criteria for the first retrain: `‖router.weight‖` at final step >
1.5× init (init for `(E=4, rank=32)` @ std=0.01 ≈ 0.113); median normalized
entropy ∈ [0.6, 0.95]; mean dominant-top1 > 0.2; zero dead experts;
`make test-hydra` quality ≥ non-hydra LoRA baseline; ComfyUI `Anima
Adapter Loader` visually matches CLI at `strength_lora=1.0`.
