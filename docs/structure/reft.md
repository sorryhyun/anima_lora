# ReFT: residual-stream intervention

A sibling adapter to LoRA that wraps the **block** instead of the linears inside it. ReFT adds one learned, low-rank edit to the output hidden state of each selected DiT block, leaving every weight inside the block untouched. It composes additively with any LoRA variant (plain / ortho / hydra / T-LoRA) and lives in the same `.safetensors` file.

Reference: Wu et al., *ReFT: Representation Finetuning for Language Models*, NeurIPS 2024 (LoReFT).

Recap from `lora.md`: LoRA attaches to every `Linear` in a DiT `Block` — `self_attn.{qkv,o}_proj`, `cross_attn.{q,kv,o}_proj`, `mlp.{layer1,layer2}`, plus the AdaLN heads. ~10 Linears × 28 blocks ≈ 280 LoRA modules. ReFT's attachment count is different: one module per *block*, not per Linear, and it does not modify any weight inside that block. From the DiT's perspective, the block's internal computation is unchanged — ReFT only rewrites what the block hands to the residual stream on its way out.

---

## 1. Why an activation-space adapter

LoRA is a **weight-space** edit: the adapted `Linear`'s forward runs with an effective weight $W_0 + \Delta W$. Every position in the sequence, every sample, every timestep sees the same delta applied to the same operator. If you're learning a new character or style, that's the right unit of edit. If you already have a LoRA doing style transfer and you notice anatomy has regressed, it's because that LoRA — necessarily — is also editing the weights that handle hands, faces, and bodies. A low-rank delta can't say "change this part of the computation but not that part."

ReFT flips the abstraction. Block internals stay frozen — the DiT's pretrained knowledge of anatomy, light, geometry is physically unchanged. What gets learned is a single residual edit **on the block's output**, read from that same output:

> "Run the block unchanged. Then look at what it produced, compute a small correction in a learned low-rank subspace, and add it back."

Because the correction is a function of the activation, not the input, it can be content-aware in a way a weight patch cannot. It can take effect on faces without affecting backgrounds (because the activation at face tokens differs from the activation at background tokens), with no spatial routing built into the adapter.

In practice it reads as a "light touch" operator. Empirically it tends to rescue anatomy fidelity that OrthoLoRA + T-LoRA alone occasionally drops.

---

## 2. The math

For a selected DiT block producing output $h \in \mathbb{R}^{B \times T \times H \times W \times D}$ (Anima's fake-5D layout, $D = 2048$), ReFT replaces the block's `forward` with:

$$
\begin{aligned}
h &= \text{Block}(x,\,e,\,c,\,\dots)\qquad\text{(frozen, unchanged)} \\[2pt]
\delta &= W_\text{src}\,h\ +\ b_\text{src}\qquad\quad \in \mathbb{R}^{\cdots \times d_\text{reft}} \\[2pt]
\text{edit} &= R^{\top}\,\delta \qquad\qquad\qquad\quad \in \mathbb{R}^{\cdots \times D} \\[2pt]
h_\text{new} &= h\ +\ \text{edit}\cdot(m\cdot s),\qquad s = \frac{\alpha_\text{reft}}{d_\text{reft}}
\end{aligned}
$$

Two trainable matrices per wrapped block:

- $R \in \mathbb{R}^{d_\text{reft}\times D}$ — `rotate_layer.weight`. QR-initialized from a random Gaussian so its rows start orthonormal, then **regularized** back toward orthogonality through training. $R$ picks the rank-$d_\text{reft}$ subspace in which the edit lives; $R^\top$ lifts the edit back to ambient dim.
- $W_\text{src}, b_\text{src}$ — `learned_source.weight` and `.bias`, **zero-initialized**. So $\delta = 0$ and $h_\text{new} = h$ at step 0, exactly.

### Paper form vs. implementation

Wu et al. write the intervention as $R^\top\big((W\,h + b) - R\,h\big)$ — a substitution "replace the component of $h$ inside $R$'s subspace with a learned linear function of $h$." Under $\Delta W = W - R$, that is algebraically identical to $R^\top(\Delta W\,h + b)$, which is what the code computes. Parameterizing $\Delta W$ directly avoids the activation-level cancellation of $R h$ against $W h$ — which would only be exact in fp32 — so the whole module runs in the ambient dtype (bf16 under mixed precision) with no fp32 up-cast in the bottleneck.

See `networks/lora_modules/reft.py:86–109` for the forward.

---

## 3. Where ReFT attaches

`reft_layers` in the method config picks which blocks get wrapped. The 28-block DiT gives several natural options:

| Spec               | Meaning                                                     |
| ------------------ | ----------------------------------------------------------- |
| `"all"`            | every block                                                 |
| `"first_N"`        | first N (coarse, composition-heavy)                         |
| `"last_N"` *(default `"last_8"`)* | last N (task-specific routing)               |
| `"stride_K"`       | every Kᵗʰ block starting at 0                               |
| `"3,7,11,15"`      | explicit list of block indices                              |

Late blocks in a DiT are where the strongest task-specific behavior lives — early blocks handle coarse composition, which the base model already does well. Editing the last ~8 block outputs moves the final image noticeably while keeping the parameter budget small.

### Attachment, again via monkey-patched `forward`

Same trick as plain LoRA (`lora.md` §3.2), one level up the object hierarchy:

```python
self.org_forward      = block.forward
block.forward         = self.forward   # ReFTModule.forward
del self.org_module
```

`ReFTModule.forward(*args, **kwargs)` calls the captured `org_forward` with all the block's usual arguments (x, timestep embedding, crossattn context, rope, …), takes its output $h$, adds the edit, returns the sum. The DiT code never learns ReFT exists — it just calls `block(x, ...)` and gets back a block output that happens to include a learned correction.

---

## 4. Composition with the LoRA family

ReFT is an **additive side-channel**. It does not touch the Linear weights LoRA patches into, and LoRA does not touch the residual-stream output ReFT edits. A single training run can enable any subset of:

- **LoRA** (`network_dim`, `network_alpha`) — weight patches on attention + MLP linears
- **OrthoLoRA** (`use_ortho`) or **HydraLoRA** (`use_hydra`) — re-parameterizations of those patches
- **T-LoRA** (`use_timestep_mask`) — applies to *both* the LoRA bottleneck and the ReFT bottleneck, independently (§5)
- **ReFT** (`add_reft`) — block-output edits

All of these coexist in the same `.safetensors`. `configs/methods/lora.toml` enables LoRA + OrthoLoRA + T-LoRA + ReFT together as the default stack.

### 5. Timestep masking on ReFT

When `use_timestep_mask = true`, ReFT receives its **own** mask with dim `reft_dim` and a floor of 1 active dim:

$$
r_\text{reft}(t)\ =\ \big\lfloor(1 - t)^{\alpha}\,(d_\text{reft} - 1)\big\rfloor + 1
$$

Applied to `delta` *before* projection back through $R^\top$ (`reft.py:102–103`):

```python
delta = F.linear(h, self.learned_source.weight, self.learned_source.bias)
if self._timestep_mask is not None and self.training:
    delta = delta * self._timestep_mask           # ← T-LoRA on the ReFT bottleneck
edit = F.linear(delta, self.rotate_layer.weight.T)
```

Same motivation as T-LoRA on LoRA — fine-detail refinement wants the full subspace, high-noise coarse steps don't need it. The ReFT mask is computed in `networks/lora_anima/network.py:577–600`, using the same GPU-resident shared-tensor pattern, and cleared at inference the same way.

---

## 6. Orthogonality regularization on $R$

We want $R$ to stay close to an orthogonal projection, so the "intervention subspace" it picks out is well-defined (not collapsing to a degenerate near-1D or rank-deficient subspace). The module exposes a penalty:

$$
\mathcal{L}_\text{ortho}\ =\ \big\|\,R\,R^{\top}\ -\ I_{d_\text{reft}}\,\big\|_F^2
$$

(`reft.py:111–115`). The network averages this over every ReFT module and adds it to the training loss through the same plumbing as OrthoLoRA's orthogonality term. The QR init means the penalty is already zero at step 0 — it only kicks in if training pulls the rows away from orthonormal.

---

## 7. File format

ReFT weights ride in the same output `.safetensors` as LoRA. One sub-state-dict per selected block:

```
reft_unet_blocks_<idx>.rotate_layer.weight      # (reft_dim, D)
reft_unet_blocks_<idx>.learned_source.weight    # (reft_dim, D)
reft_unet_blocks_<idx>.learned_source.bias      # (reft_dim,)
reft_unet_blocks_<idx>.alpha                    # scalar
```

Dim inference on load: `reft_dim` is read from `rotate_layer.weight.shape[0]`, block indices from the key prefix, and `reft_layers` is rebuilt from the set of present indices. Nothing else is needed from the training config.

Strict prefix policy: any `reft_*` key must match `reft_unet_blocks_<idx>.*`. Older per-Linear ReFT wirings are refused at load time with an explicit error pointing at retraining.

### Why ReFT cannot be merged into DiT weights

Plain LoRA merges losslessly because $\Delta W\,x$ is the same linear operator on every input — it can be folded into $W_0$ in-place. The ReFT delta **cannot**: $\text{edit} = R^\top(W_\text{src}\,h + b)$ is a function of $h$, the block's output. There is no weight matrix inside the block whose modification would produce the same edit — the edit is a genuine additive term on the activation, computed after the block runs. This has a practical consequence for ComfyUI (below).

---

## 8. Inference

### CLI (`inference.py`, `make test`)

No special flag. `reft_*` keys are detected in the adapter file, the matching `ReFTModule`s are constructed and installed onto the DiT blocks, their weights loaded. Everything else — `multiplier`, P-GRAFT cutoff, etc. — applies to the LoRA and ReFT branches together via `network.enabled`.

### ComfyUI — requires the custom node

**Vanilla ComfyUI cannot load ReFT.** The built-in LoRA weight-patcher rewrites `Linear.weight` in place, which is the wrong operation by construction — ReFT is an activation-space intervention, not a weight patch. The stock loader silently drops any `reft_*` keys and ships you a LoRA-only inference.

The **Anima Adapter Loader** node (`custom_nodes/comfyui-hydralora/`) installs per-block `forward_hook`s through `ModelPatcher.add_object_patch` to reproduce `ReFTModule.forward` exactly, in parallel with LoRA and HydraLoRA. Separate `strength_lora` and `strength_reft` sliders let you ablate the two branches independently at inference.

---

## 9. Configuration

`configs/methods/lora.toml` defaults:

```toml
add_reft      = true
reft_dim      = 64
reft_alpha    = 64       # scale = reft_alpha / reft_dim = 1 at default
reft_layers   = "last_8"
```

Notes:

- `reft_alpha` falls back to `network_alpha` when omitted.
- `reft_dim` is independent of `network_dim`. Per-block parameter cost is $2 \cdot d_\text{reft} \cdot D + d_\text{reft}$ ≈ 260K at $d_\text{reft} = 64,\ D = 2048$ — small compared to a full LoRA bank on the same block.
- Style transfer too strong but anatomy drifting? Try `reft_layers = "all"` with a larger `reft_dim`. Training is already converged and you want a light rescue? `"last_4"` or `"last_8"` at `reft_dim = 16–32` is cheap.

---

## 10. Minimal mental model

1. One `ReFTModule` wraps one DiT `Block`'s `forward`, not its internal Linears. Block weights stay frozen.
2. The edit is $R^\top(W_\text{src}\,h + b)$ added to the block's output. $W_\text{src}$ is zero-init, so step 0 is the base model exactly.
3. Orthogonality of $R$ is **regularized** (not structural, unlike OrthoLoRA's Cayley parameterization). QR init + Frobenius penalty.
4. Composes additively with every LoRA variant; T-LoRA masks the ReFT bottleneck with its own `reft_dim` schedule.
5. Cannot be merged into DiT weights — it's an activation-space intervention, so ComfyUI needs the custom node's forward-hook path.
