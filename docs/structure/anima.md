# Anima model

What the Anima diffusion model is and how a caption + image become a training step: the text-conditioning pipeline, the VAE, the DiT block stack, and the flow-matching loss at the end.

![Anima architecture](../structure_images/anima.png)

---

## 1. The model at a glance

Anima is a flow-matching DiT (Diffusion Transformer) operating in latent space:

```
              ┌──────────┐  ┌────┐  ┌────────────┐                      ┌─────────────┐
  caption ──▶ │ Qwen3-.6B│─▶│LLM │─▶│ crossattn_ │─ context (B,512,1024)│             │
              └──────────┘  │Adpt│  │    emb     │                     ▶│             │
              ┌──────────┐  └────┘  └────────────┘                      │   DiT       │── v̂
  caption ──▶ │ T5 tok.  │──────▶   t5_input_ids (target-side embed)    │  (28 blks)  │
              └──────────┘                                               │             │
  image/vid ─▶ VAE.encode ─▶ x₀ ──(+ noise σ)──────────────────────────▶ │             │
                                                                         │             │
  timestep t ─────────────────────────────────────────────────────────▶  └─────────────┘
```

The DiT's output $\hat v$ is the predicted velocity at the noisy point $x_t$ — a tensor of the same shape as the latent that, under rectified flow, points from the noise toward the clean data: $\hat v = v_\theta(x_t, t, c) \approx \varepsilon - x_0$. Integrating $\hat v$ along the trajectory $t: 1 \to 0$ at inference time is what turns pure noise into a latent the VAE can decode. Training teaches the DiT to make $\hat v$ match the true velocity; see §5 for the loss.

- DiT. `class Anima` in `library/anima/models.py`. 28 `Block`s with hidden dim `D = 2048`, 16 heads × 128 head-dim, MLP expansion ratio 4 → hidden `8192`. Cross-attention `context_dim = 1024`.
- Token budget. Free-fit bucketing: every training image keeps its native aspect ratio, and its patch grid lands wherever its resolution tier's token band allows. Each forward runs at the image's *real* token count — no padding tokens exist to leak into attention. How that stays compile-friendly is the subject of `anima-optimizations.md` §3.

---

## 2. Text conditioning

Text conditioning in Anima is not a single "encode and project" step — it's a small pipeline made of three pieces: a Qwen3 encoder, a T5 tokenizer (tokenizer only — T5 the model is never loaded), and a learned bridge called the LLMAdapter. The output of that pipeline, `crossattn_emb ∈ ℝ^{B×512×1024}`, is what every DiT block sees in cross-attention.

### 2.1 Two tokenizers, one encoder

Bundled under `library/anima/configs/`:

- `qwen3_06b/` — Qwen2Tokenizer for Qwen3-0.6B. Vocab 151,936, `hidden_size = 1024`.
- `t5_old/` — T5TokenizerFast, sentencepiece-based.

Both tokenize the same caption. Both pad unconditionally to `max_length = 512` with `padding="max_length"` (`library/anima/strategy.py`). That padding is load-bearing — see §2.4.

Why carry a T5 tokenizer without T5? Only its token IDs are used, as the target-side input to the LLMAdapter (§2.3). This saves the ~11 GB T5-XXL encoder and still gives the adapter a second, structurally different tokenization to cross-attend against.

Caveat — the T5 vocab is English-only, and CJK collapses to `<unk>`. `t5_old/` is
the `google/t5-v1_1-xxl` sentencepiece vocab (32,100, trained on C4). It has no CJK
coverage at all — not "poor" coverage, *none*:

```
'a cat sitting on a bench' -> ['▁','a','▁cat','▁sitting','▁on','▁','a','▁bench']
'안녕하세요'                -> ['▁', '<unk>']
'你好世界'                  -> ['▁', '<unk>']
'text that says "한글"'     -> ['▁text','▁that','▁says','▁"','<unk>','"']
```

Qwen3 (§2.2) tokenizes the same strings fine (byte-level BPE, 151k), so the *semantics*
still reach the adapter through the cross-attention context. What collapses is the
query stream: a 20-character Korean prompt gets 2 non-pad query slots instead of
~25, so the adapter has almost no positions to write distinct content into, and all
word-boundary / length structure is gone. The "second, structurally different
tokenization" benefit above is therefore English-only — for a CJK prompt the target
side contributes essentially nothing.

Known consequence for prompt following in CJK. Open question for CJK *glyph
rendering*: [`docs/findings/freetext_text_rendering.md`](../findings/freetext_text_rendering.md)
attributes Anima's inability to draw Korean to the DiT's visual glyph head ("Qwen3
already understands Korean semantically"), but that eval used an English control, which
differs from Korean in both the glyph prior and this `<unk>` collapse — so the two
were never separated. See also `anime_tools.captions.variants` (filler-token
selection already works around T5's English-centric vocab).

`AnimaTokenizeStrategy.tokenize()` returns four tensors, all shape `(B, 512)`:

```
[qwen3_input_ids, qwen3_attn_mask, t5_input_ids, t5_attn_mask]
```

### 2.2 Qwen3 forward

Qwen3-0.6B is loaded (`library/anima/weights.py`) as the `.model` of the causal LM (no LM head), bf16 by default. `AnimaTextEncodingStrategy.encode_tokens()` runs it and takes `last_hidden_state`:

$$
\text{prompt\_embeds} = \text{Qwen3}(\text{qwen3\_input\_ids})_{\text{last\_hidden}}
\ \in \mathbb{R}^{B\times 512\times 1024}
$$

Positions where the attention mask is `False` are zeroed:

```python
prompt_embeds[~qwen3_attn_mask.bool()] = 0
```

That gives us the source embeddings for the adapter.

### 2.3 LLMAdapter: bridging Qwen3 → DiT context

`LLMAdapter` (`library/anima/models.py`) is a 6-block transformer that cross-attends between the T5 token embeddings (queries / "target") and the Qwen3 hidden states (keys/values / "source"):

```
target_input_ids  ─▶ embed(.)  ─▶ in_proj(.)  ─▶ ┌──────────────┐
                                                 │  6 × Block   │
source Qwen3 embeds ───────────────────────────▶ │ (cross-attn) │ ─▶ out_proj(.) ─▶ crossattn_emb
                                                 └──────────────┘
```

All three dims (`source_dim`, `target_dim`, `model_dim`) are `1024`, 16 heads. Output shape: `(B, 512, 1024)`.

This output — not the raw Qwen3 hidden state — is what feeds the DiT cross-attention. The DiT's `cross_attn.kv_proj` projects that `1024 → 4096` (K + V fused), so no external projection is needed between the adapter and the DiT.

Why a bridge at all? The pretrained Anima was distilled against a T5-like condition stream; the adapter learns to synthesize that stream from Qwen3's cheaper-to-run hidden states.

### 2.4 Max-padded, attention-sink behavior

One non-negotiable invariant: both training and inference must pad to `max_length` and must **NOT** mask out padding via `crossattn_seqlens`.

The pretrained Anima learned to use zero-padding positions as attention sinks — the cross-attention softmax relies on them as a low-energy "nowhere to look" target. If you trim `crossattn_emb` down to the actual caption length, or apply an attention mask that removes the padding, the softmax denominator collapses, cross-attention saturates, and generations go black.

Practically:

- The tokenizer always pads to 512.
- Zero-masked positions in `prompt_embeds` are kept (they're just exact zeros, which is fine for cross-attention as sink tokens).

### 2.5 A second path: pooled-text modulation

Cross-attention is not the only way the caption reaches the DiT. A max-pooled summary of `crossattn_emb` is also added to the timestep embedding `t_emb` before AdaLN fans it out to every block — so the caption modulates self-attn and MLP gains, not just cross-attn alignment. See `modulation.md` for the full path and why it matters for modulation guidance.

---

## 3. VAE and latents

Anima uses the Qwen VAE (from the Qwen-Image family), 8× spatial compression, 16 latent channels. An input image of `H × W` pixels becomes a latent of shape `(16, H/8, W/8)`.

Latent caching is the second half of the offline pipeline: run the VAE over every training image once, write the latents to disk, free the VAE from VRAM. During training, only cached latents are loaded — the VAE does not need to be resident.

PatchEmbed inside the DiT then divides the latent spatially by 2 and maps channels `16 → 2048`, giving roughly `(H/16) × (W/16)` DiT tokens per frame.

Shape gotcha worth internalizing early: the DiT's forward takes a 5D latent `(B, C, T=1, H, W)` — a video-shaped layout with a singleton frame axis at dim 2. Everything around the DiT (VAE output, cached latents, the training inner loop) is 4D `(B, C, H, W)`. The boundary dance is always `unsqueeze(2)` going in and `squeeze(2)` coming out — never a bare `squeeze()`, which silently eats the batch dim when `B = 1`.

---

## 4. What one DiT block contains

One `Block` (`library/anima/models.py`) has three residual sub-layers, each gated by AdaLN-Zero modulation:

$$
\begin{aligned}
x &\leftarrow x + g_{\text{sa}} \cdot \text{SelfAttn}\!\big((1+s_{\text{sa}})\,\text{LN}(x) + b_{\text{sa}}\big) \\
x &\leftarrow x + g_{\text{ca}} \cdot \text{CrossAttn}\!\big((1+s_{\text{ca}})\,\text{LN}(x) + b_{\text{ca}},\ c\big) \\
x &\leftarrow x + g_{\text{mlp}} \cdot \text{MLP}\!\big((1+s_{\text{mlp}})\,\text{LN}(x) + b_{\text{mlp}}\big)
\end{aligned}
$$

Each `(shift, scale, gate)` triple is produced per sub-layer by a small head:

$$
(b_\star,\,s_\star,\,g_\star)\ =\ \text{split}_3\!\big(W_\star^{\text{adaLN}}\,\text{SiLU}(t_{\text{emb}})\big),
\quad W_\star^{\text{adaLN}} \in \mathbb{R}^{6D \times D}
$$

i.e. a `Linear(2048 → 6144)` that is then split into three `2048`-vectors.

The concrete Linear layers inside one block:

| Sub-layer   | Module       | Linear                                | Shape (in → out)   |
| ----------- | ------------ | ------------------------------------- | ------------------ |
| self-attn   | `self_attn`  | `qkv_proj` (fused Q,K,V)              | 2048 → 6144        |
|             |              | `output_proj`                         | 2048 → 2048        |
| cross-attn  | `cross_attn` | `q_proj`                              | 2048 → 2048        |
|             |              | `kv_proj` (fused K,V on 1024-dim ctx) | 1024 → 4096        |
|             |              | `output_proj`                         | 2048 → 2048        |
| MLP         | `mlp`        | `layer1`                              | 2048 → 8192        |
|             |              | `layer2`                              | 8192 → 2048        |
| AdaLN heads | `adaln_…[1]` | ×3 per sub-layer                      | 2048 → 6144        |

Across 28 blocks that is ~280 `Linear`s, plus the patch-embed / timestep-embed / final-layer heads outside the stack. Those ~280 Linears are what the LoRA family attaches to (`lora.md`).

One small inference-time hook lives here too: each block carries a `_xattn_gain` buffer (default 1.0 = identity) that multiplies into the cross-attn gate — this is what `--xattn_boost` sets to counteract the front-loaded decay of text drive across the trajectory.

---

## 5. The training step

Flow-matching with the rectified-flow parameterization. For a clean latent $x_0$ and independent Gaussian noise $\varepsilon \sim \mathcal{N}(0, I)$, sample $\sigma \in [0,1]$ and form:

$$
x_t = (1-\sigma)\,x_0 + \sigma\,\varepsilon
$$

(`library/runtime/noise.py`). The DiT predicts the straight-line velocity, and the target is:

$$
v^\star = \varepsilon - x_0,
\qquad
\hat v = v_\theta(x_t,\,t,\,c)
$$

(`train.py` — literally `target = noise - latents`). Loss is $\sigma$-weighted MSE:

$$
\mathcal{L}\ =\ \mathbb{E}_{x_0,\varepsilon,\sigma}\!\left[w(\sigma)\cdot \big\|\hat v - v^\star\big\|_2^2\right]
$$

with weighting chosen by `--weighting_scheme` (`library/runtime/noise.py`):

$$
w(\sigma) =
\begin{cases}
  \sigma^{-2} & \text{sigma\_sqrt} \\[2pt]
  \dfrac{2}{\pi\,(1 - 2\sigma + 2\sigma^2)} & \text{cosmap} \\[2pt]
  1 & \text{uniform}
\end{cases}
$$

Timesteps are sampled via logit-normal / mode / uniform (`compute_density_for_timestep_sampling`) and optionally restricted to a $[t_\text{min}, t_\text{max}]$ window.

The full step:

```
  x_0, c  ──▶ σ ~ p(σ),  ε ~ N(0,I)
            │
            ▼
         x_t = (1-σ)·x_0 + σ·ε
            │
            ▼
         v̂ = DiT(x_t, t, c)
            │
            ▼
         L  = w(σ)·‖v̂ − (ε − x_0)‖²
            │
            ▼        backward
         optimizer.step()
```

---

## 6. Minimal mental model

1. One caption feeds the DiT through two channels: token-level `crossattn_emb` (Qwen3 → LLMAdapter, always padded to 512 — the padding is a load-bearing attention sink) and a max-pooled summary riding the timestep embedding (`modulation.md`).
2. The DiT is 28 identical blocks of self-attn / cross-attn / MLP, each sub-layer gated by AdaLN-Zero from `t_emb`. The compute is ~280 `Linear`s — the attachment surface for every adapter in this repo.
3. Latents are 4D everywhere except inside the DiT, which wants 5D with a singleton at dim 2. `unsqueeze(2)` in, `squeeze(2)` out.
4. Training is rectified-flow velocity regression: noise a cached latent to a sampled σ, predict `ε − x₀`, σ-weighted MSE.
