# ReFT: Representation Fine-Tuning

Block-level residual-stream intervention on the DiT, following LoReFT (Wu et al., NeurIPS 2024). A small learned low-rank edit is added to the output hidden state of selected DiT blocks, instead of modifying any weights inside them. Composes additively with LoRA/T-LoRA/HydraLoRA in the same training run and the same `.safetensors` file.

## Why

The pretrained DiT's weights already know how to draw bodies, faces and hands. A rank-64 LoRA on top can improve style fidelity but sometimes reintroduces anatomy artifacts the base model had already solved — the low-rank weight edit fights the base weights everywhere it's attached. ReFT side-steps this: the block weights stay frozen, and a *single* additive correction is written onto the residual stream at each selected block. In practice this reads as "nudge the hidden state, don't rewrite the computation," and it has been effective at recovering anatomy fidelity that timestep-masked psoft-integrated OrthoLoRA alone would occasionally miss. See `docs/methods/timestep_mask.md` and `docs/methods/psoft-integrated-ortholora.md` for the adapters it composes with.

## How it works

For a selected DiT block with output `h = Block(x, emb, crossattn, …)` of shape `(B, T, H, W, D)`, ReFT replaces the block's forward with:

```
h      = Block(…)                          # frozen, unchanged
delta  = learned_source(h)                 # (…, reft_dim)
edit   = rotate_layer.T @ delta            # (…, D)  — project back to embed
h_new  = h + edit * (multiplier * scale)
scale  = reft_alpha / reft_dim
```

`rotate_layer` is a `Linear(embed_dim, reft_dim)` whose weight is QR-initialized so its rows start orthonormal, then regularized back toward orthogonality during training (see below). `learned_source` is a `Linear(embed_dim, reft_dim)` with both weight and bias zero-initialized, so the edit is exactly zero at step 0 and the block behaves identically to the base model until gradients flow.

Paper form vs. implementation. Wu et al. write the intervention as `(W·h + b) − R·h` inside an orthogonal subspace. That form is algebraically identical to `R^T·(ΔW·h + b)` under `ΔW = W − R`; parameterizing ΔW directly avoids the activation-level cancellation, so the module runs entirely in the ambient (bf16) dtype without fp32 up-casts in the bottleneck.

### Intervention point

LoRA attaches to every matching `Linear` inside a block (typically `self_attn.{q,k,v,o}`, `cross_attn.{q,k,v,o}`, `mlp.{layer1,layer2}`), so a LoRA with 28 blocks × ~8 Linears = ~224 adapted modules. ReFT attaches to the **block** itself — one module per selected block, wrapping the block's `forward`. The edit is applied once on the block output, not on each internal Linear. This matches the paper's parameter and activation budget.

Concretely, the wrapper is installed by swapping `block.forward` with `ReFTModule.forward`, which calls the original forward, computes the residual edit and returns the sum. Broadcasts cleanly over the `(B, T, H, W, D)` video-style layout because the bottleneck is last-dim-linear.

## Block selection: `reft_layers`

Not every block needs an edit. `reft_layers` in the method config picks which blocks get wrapped:

| Spec | Meaning |
|------|---------|
| `"all"` *(default)* | every block |
| `"first_N"` | first N blocks |
| `"last_N"` | last N blocks (task-signal heavy) |
| `"stride_K"` | every Kᵗʰ block starting from 0 |
| `"3,7,11,15"` | explicit list of block indices |

For the DiT used here (28 blocks), `"last_8"` is the configured default in `configs/methods/lora.toml`. Late blocks are where the strongest task-specific routing already happens in the base model, so small edits there tend to move the final image more than edits in the first few (high-noise, coarse-composition) blocks.

## Composition with the LoRA family

ReFT is an additive side-channel — it does not touch the weights LoRA patches into, and LoRA does not touch the residual stream ReFT edits. A single training run can enable any subset of:

- LoRA on the attached Linears (`network_dim`, `network_alpha`)
- OrthoLoRA or HydraLoRA re-parameterizations on top (`use_ortho`, `use_hydra`)
- Timestep rank masking (`use_timestep_mask`) — applied to both the LoRA bottleneck and, independently, to the ReFT bottleneck (see below)
- ReFT block-level edits (`add_reft`)

All of these coexist in the same `.safetensors` file. `configs/methods/lora.toml` turns LoRA + OrthoLoRA + timestep masking + ReFT all on; flip the individual toggles to test any subset. The GUI counterpart is `configs/gui-methods/tlora_ortho_reft.toml`.

### Timestep masking on ReFT

When `use_timestep_mask = true`, ReFT modules receive their **own** mask, separate from the LoRA mask, with its own dimension (`reft_dim`) and a floor of 1 active dim:

```
r_reft(t) = floor((1 − t)ᵅ · (reft_dim − 1)) + 1
```

The mask zeroes high-index columns of `delta` before projection back through `rotate_layer.T`, so higher-noise steps see a narrower intervention subspace and lower-noise steps see the full one. Same rationale as T-LoRA on the LoRA branch — fine-detail refinement wants full capacity, coarse high-noise steps don't. `networks/lora_anima/network.py:set_reft_timestep_mask` writes the mask each step; `networks/lora_modules/reft.py:ReFTModule.forward` applies it only when `self.training`.

## Regularization

Because we want `rotate_layer` to stay close to an orthogonal projection (so the intervention subspace is well-defined), each ReFT module exposes an orthogonality penalty:

```
L_ortho = || R · Rᵀ − I_{reft_dim} ||²_F
```

where `R = rotate_layer.weight`. The network-level `regularization()` averages this over all ReFT modules (and over LoRA regularizers if present). Plumbed into the training loss through the same path as OrthoLoRA's orthogonality term — see `networks/lora_anima/network.py` for the aggregation and `train.py` for where it's added.

## File format

ReFT weights live alongside LoRA weights in the same output `.safetensors`. One sub-state-dict per selected block:

```
reft_unet_blocks_<idx>.rotate_layer.weight      # (reft_dim, embed_dim)
reft_unet_blocks_<idx>.learned_source.weight    # (reft_dim, embed_dim)
reft_unet_blocks_<idx>.learned_source.bias      # (reft_dim,)
reft_unet_blocks_<idx>.alpha                    # scalar
```

Loader behavior on these keys is strict: any `reft_*` key must match `reft_unet_blocks_<idx>.*`. Older per-Linear ReFT wiring is refused at load time with an explicit error pointing at retraining (`networks/lora_anima/`).

Dim inference on load: `reft_dim` is read from `rotate_layer.weight.shape[0]`, block indices from the key prefix, and `reft_layers` is rebuilt from the set of present indices. Nothing else is needed from the original training config.

## Inference

### CLI (`inference.py`, `make test`)

No special flag. `reft_*` keys are detected in the adapter file and the matching `ReFTModule`s are constructed, installed onto the DiT blocks, and their trained weights loaded. The network is then treated like any other adapter — multiplier, P-GRAFT cutoff, etc., all apply to both the LoRA and ReFT branches together via `network.enabled`.

### ComfyUI — requires the custom node

**Vanilla ComfyUI cannot load ReFT.** The built-in LoRA patcher rewrites `Linear.weight` in place, which is fundamentally the wrong operation here — ReFT is an *activation-space* intervention, not a weight patch. The standard loader silently ignores `reft_*` keys.

Use the **Anima Adapter Loader** node (`custom_nodes/comfyui-hydralora/`), which installs per-block forward hooks for ReFT alongside LoRA / HydraLoRA / postfix. Separate `strength_lora` and `strength_reft` sliders allow independent ablation. See `custom_nodes/comfyui-hydralora/README.md` for the node's installation paths and changelog.

## Configuration

`configs/methods/lora.toml` defaults:

```toml
add_reft      = true
reft_dim      = 64
reft_alpha    = 64
reft_layers   = "last_8"
```

Notes:

- `reft_alpha` defaults to `network_alpha` if omitted.
- `reft_dim` is fully decoupled from `network_dim` — there's no reason they have to match. Start with `reft_dim` in the 32–64 range; the per-block parameter cost is `2 · reft_dim · embed_dim` plus a bias, which is small compared to a full-block LoRA bank.
- If you see the style transferring but anatomy still drifting, try `reft_layers = "all"` and a larger `reft_dim`. If training is already strong and you just want a light anatomy rescue, `"last_4"` or `"last_8"` with `reft_dim = 16–32` is cheap.

## Implementation

| File | Role |
|------|------|
| `networks/lora_modules/reft.py` | `ReFTModule` — intervention forward, orthogonality reg |
| `networks/lora_anima/` | Block selection (`_parse_reft_layers`), wrapping, loader |
| `networks/lora_anima/network.py` | `set_reft_timestep_mask()` — timestep masking on ReFT bottleneck |
| `library/anima/training.py` | CLI arg plumbing (`add_reft`, `reft_dim`, `reft_alpha`, `reft_layers`) |
| `train.py` | Calls `set_reft_timestep_mask()` each step after noise sampling |
| `custom_nodes/comfyui-hydralora/adapter.py` | ComfyUI loader with per-block forward-hook install |

## Reference

Wu et al., *ReFT: Representation Finetuning for Language Models*, NeurIPS 2024.
