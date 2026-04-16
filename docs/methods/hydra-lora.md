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
- **Layer-local `router`** — a `Linear(in_dim, num_experts)` that reads the
  current layer input (not the text embedding) and emits softmax gates over
  the experts for this specific layer and sample.

Forward (simplified, see `networks/lora_modules.py:HydraLoRAModule.forward`):

```python
# Mean pool over the sequence dim — see "Why mean pool" below.
pooled = x.reshape(B, -1, x.shape[-1]).mean(dim=1)     # (B, in_dim)
gate   = softmax(self.router(pooled), dim=-1)          # (B, num_experts)

lx       = self.lora_down(x)                           # (B, L, rank)
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
- **Small router overhead.** Per-module router params are `in_dim * num_experts`
  plus bias. For `in_dim=2048, num_experts=4`, that's ~8 K params per module.
  Negligible against the LoRA parameters themselves.

## Why mean pool (not max pool)

DiT layer inputs have DC-bias outlier channels with peak-to-mean ratios of
80–96× across most `self_attn`, `cross_attn.q`, and `mlp.layer1` modules
(documented in `bench/channel_dominance_analysis.md`). Max-pooling those
inputs would surface the outliers directly into the router logits and
saturate softmax in bf16 — the gate would effectively become a constant
one-hot function of the most active channel.

Mean pool averages the outliers out. It is also why the router weight is
initialized at `std=0.01` instead of Kaiming/Xavier: the starting gate has to
be near-uniform so all experts receive gradient, and the larger init can
still push logits past the bf16 softmax saturation threshold even after the
mean pool.

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
<prefix>.router.weight           # (E, in_dim)
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

### ComfyUI (manual blending)

The `custom_nodes/comfyui-hydralora` node loads a `*_moe.safetensors` and
exposes a weight slider per expert. It computes a weighted combination of
expert ups using the slider values, bakes that into a standard LoRA, and
hands it to ComfyUI's patcher. Slider weights are normalized to sum to 1 to
match the training-time softmax gate shape.

This is a **manual override**, not live routing. Live routing would need
forward hooks inside ComfyUI's model patcher — which the patcher doesn't
expose — so the node is intentionally limited to "pin the gate at a
user-chosen distribution, uniformly across all layers and samples." Use the
CLI path for true router-live inference.

## Composition with other variants

- **T-LoRA** — timestep rank masking applies to `lora_down` (shared across
  experts), so it composes directly. HydraLoRA + T-LoRA is the configured
  default.
- **DoRA** — currently not supported on HydraLoRA. Each expert would need
  its own magnitude vector, which is an unimplemented extension.
- **OrthoLoRA** — not supported. Orthogonal parameterization would need to
  hold per expert, multiplying init cost by `num_experts`.
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
- `balance_loss_weight = 0.01` — Switch Transformer load-balancing coefficient.
  Raise if you observe expert collapse (inspect `_last_gate` distributions);
  lower if expert differentiation is too weak.

HydraLoRA requires `cache_llm_adapter_outputs = true` (same as standard LoRA
in this repo — unrelated to routing, but the cached crossattn is assumed by
the surrounding training plumbing).
