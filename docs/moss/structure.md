# MOSS-Image model

What the MOSS-Image diffusion model is and how a caption + image become a training step and a sample: the text pass, the FLUX.2 VAE boundary, the single-stream DiT, the flow-matching loss, and the sampler. For setup, defaults and the GUI see [`docs/moss/README.md`](README.md); for the code map see `library/moss/CLAUDE.md`.

Almost nothing in [`anima.md`](../anima/structure/anima.md) carries over: different text encoder, different VAE, a single-stream block instead of self-attn + cross-attn, unpadded text, 4D latents.

---

## 1. The model at a glance

MOSS-Image (OpenMOSS, Apache-2.0) is a 5B flow-matching DiT on a frozen Qwen3.5-9B text encoder and the frozen FLUX.2 VAE. Image and text tokens share one sequence through every block:

```
              ┌─────────────┐  last_hidden   ┌──────────────┐
  caption ──▶ │ Qwen3.5-9B  │──(S, 4096)───▶ │ text_embedder│── text (S, 3072) ──┐
  (chat tpl)  └─────────────┘  true length   │ proj + 2 blk │                    │
                                             └──────────────┘                    ▼
                                                             ┌─────────────────────────────┐
  image ─▶ FLUX.2 VAE ─▶ (32,H/8,W/8) ─▶ patchify 2×2 + BN ─▶│ [ image (H·W) | text (S) ]  │
                                          (128, H/16, W/16)  │   42 single-stream blocks   │─▶ image rows ─▶ head ─▶ v̂
           + noise σ ─────────────────────────────────────▶  │   (shared AdaLN from t)     │
  timestep 1000·σ ─▶ time_embedding ─▶ shared_block_modulation ─▶ every block            │
                                                             └─────────────────────────────┘
```

- DiT. `MOSSImageTransformer` in `library/moss/model.py`, vendored from the model folder's `pipeline_moss_image.py` with state-dict names unchanged (nothing loads with `trust_remote_code`). 42 blocks, hidden `D = 3072`, 24 heads × 128 head-dim, FFN hidden `8192`. Depth is fixed — `num_layers` in `transformer/config.json`, not probed.
- Two DiT checkpoints, `base` and `sft` (`--model`). SFT ships only its DiT; TE, VAE and tokenizer always load from Base.
- Output. $\hat v = v_\theta(x_t, t, c) \approx \varepsilon - x_0$ on the image tokens only — text tokens run through every block but never reach the head.
- Token budget. Native aspect, both sides multiples of 16, area capped at `resolution²` (default 1024², never upscaled; `cache.py::target_size`). One image token per 16×16 pixels, so 1024² is 4096 image tokens. There are no buckets; training runs batch 1 at each image's own shape.

---

## 2. Text conditioning

### 2.1 Qwen3.5-9B, one prompt per forward

Each caption is wrapped in the Qwen chat template and encoded alone (`library/moss/loader.py::prompt_ids` / `encode_prompt`):

```
<|im_start|> user \n  {caption}  <|im_end|> \n <|im_start|> assistant \n <think> \n
└── prefix, 3 ──────┘             └── suffix, 7 ─────────────────────────────────────┘
```

`apply_chat_template(add_generation_prompt=True, enable_thinking=True)`, then the TE's `last_hidden_state` → `(S, 4096)` at the prompt's true length, capped at 4096 tokens. This is exactly `MOSSImagePipeline.encode_prompt`.

**Never pad.** Qwen3.5's gated-DeltaNet layers misbehave on padded batches, and RoPE puts the image at frame position = text length (§4.2), so padding would also move the image. Batched forwards pad *after* the TE and mask the padded keys instead (§4.3).

The 10 template tokens are not inert: they are MOSS's attention sink — an OFF-gate plus a per-head bias written into every image token. The uncond prompt `''` is these 10 tokens alone. Measured in [`moss/findings/attention_sink_chat_template.md`](findings/attention_sink_chat_template.md).

### 2.2 TE precision and memory

The bf16 TE (16.8 GB with its unused vision tower) is larger than a 16 GB card. `loader.py` keeps the 248k-row token embedding on the CPU, sizes an activation reserve to the longest prompt, and pages whole decoder layers from pinned memory when they do not fit. The default TE is **INT8** (`text_encoder = "int8"`, `library/moss/int8.py`, W8A8 ConvRot with SmoothQuant scales folded into the RMSNorm weights); `fp8` and `bf16` are selectable. Both quantized TEs put the decoder layers at 6.5 GiB, so nothing pages.

The TE cache stamps `encoder` (precision included) and `caption_format`, and re-encodes when either changes — cache, train and generate with the same `text_encoder` and caption format.

### 2.3 Caption format

`caption_format = "layout_json"` (default) converts the tag caption into a layout-JSON prompt at cache time (`anime_tools.captions.layout_json`). Such a cache also holds the plain tag caption as `embeds_plain`; training uses it on `plain_caption_rate` of steps (0.2). Caption variants (`embeds_v{k}`, i2i colorize) and caption dropout (the shipped `library/moss/asset/empty_prompt.safetensors`) are drawn per step by `train.py::choose_embeds`, each from its own RNG.

### 2.4 The text refiner

Inside the DiT, `text_embedder` (`_TextProjection`) is a `Linear(4096 → 3072)` followed by two unmodulated transformer blocks (`_TextBlock`: biased fused `qkv`, QK-norm, gated FFN) over the text alone, with the text's RoPE. After that the text joins the image sequence and has no weights of its own.

---

## 3. VAE and latents

The FLUX.2 VAE (`AutoencoderKLFlux2`, 0.25 GB) encodes at 8× spatial compression into 32 channels. The DiT boundary is `model.pack_latents` / `unpack_latents`:

$$
z_\text{DiT} = \frac{\text{patchify}_{2\times2}(z_\text{VAE}) - \mu_\text{BN}}{\sigma_\text{BN}}
\quad (B,32,H/8,W/8) \to (B,128,H/16,W/16)
$$

with the VAE's own BatchNorm running mean / var as the statistics. Encode takes the latent distribution's mode, as FLUX.2 encodes references.

- **Latents are 4D packed `(B, 128, H/16, W/16)`** everywhere, inside the DiT included — no frame axis, no `unsqueeze(2)`.
- `video_embedder` is a per-token `Linear(128 → 3072)`; the 2×2 patchify already happened at the VAE boundary.
- Decode runs in fp32 with the DiT off the card — at ~2 MP it does not fit beside it.

---

## 4. The DiT

### 4.1 Sequence

`embed_sample` builds one sample's sequence as `[image (H·W) | text (S)]` (i2i adds reference tokens between them, §4.5). All 42 blocks run over the whole sequence with joint self-attention; there is no cross-attention. After the last block, `features = x[:, :H·W]` goes to the head.

### 4.2 RoPE

`_RoPE` gives every token a 4-axis coordinate `(r, f, h, w)`, 16 frequency bands per axis (head-dim 128 = 4 axes × 16 bands × sin/cos), kept in fp32 (rebuilt on every `.to()`).

| Tokens | r | f | h, w |
|---|---|---|---|
| text, position i | 0 | i | 0, 0 |
| image | 0 | **S** (text length) | grid, centred on 0 |
| i2i reference | 1 | S | its grid stretched onto the image's |

So the image sits one frame after the last text token. Padding the text would shift `f` for the image — the second reason the TE is never padded.

### 4.3 What one block contains

`_Block` is one attention sub-layer and one FFN, each with AdaLN shift / scale / gate:

$$
\begin{aligned}
x &\leftarrow x + g_a \odot W_o\,\text{Attn}\big(\text{QKnorm}(W_{q,k,v}\,((1+s_a)\,\text{RMS}(x) + b_a)),\ \text{RoPE}\big) \\
x &\leftarrow x + g_f \odot W_\text{down}\,\text{SiTU}\big(W_\text{up\_gate}\,((1+s_f)\,\text{RMS}(x) + b_f)\big)
\end{aligned}
$$

with $\text{SiTU}(h, g) = 4\tanh(g/4)\,\sigma(g) \cdot 25\tanh(h/25)$ — a soft-capped gated activation on the two halves of the fused `ffn_up_gate` output.

| Sub-layer | Module | Shape (in → out) |
|---|---|---|
| attention | `proj.q`, `proj.k`, `proj.v` | 3072 → 3072 each |
| | `proj.o` | 3072 → 3072 |
| FFN | `proj.ffn_up_gate` (fused hidden + gate) | 3072 → 16384 |
| | `proj.ffn_down` | 8192 → 3072 |

All six are bias-free, ~113M parameters per block, ~4.8B across the 42 — the 5B. These 42 × 6 = 252 linears are the LoRA surface (§6).

**Modulation is shared.** One `time_embedding` MLP and one `shared_block_modulation = Linear(3072 → 6·3072)` produce a single `(B, 6, D)` set from the timestep, fed unchanged to all 42 blocks. There are no per-block AdaLN heads, and the caption does not modulate: unlike Anima's pooled-text path ([`modulation.md`](../anima/structure/modulation.md)), the gates are time-only and per channel. The head has its own `video_head_modulation` (shift, scale) before `final_linear_video` (3072 → 128).

Batched forwards (`forward_batched`) right-pad text to the longest prompt and pass a key mask to the text refiner and every block, so padding never reaches a valid token. The per-sample `forward` is the reference loop and stays op-for-op with the pipeline.

### 4.4 Hooks the repo adds

All additive; `None` / off is the reference model.

- `skip_layers` — per-sample block skips, so cond and the skip-22 uncond (§5.2) share one batch.
- `layer_hook(index, x, n_img)` — runs on each block's output, outside the compiled block. DAVE (`library/moss/dave.py`) and REPA use it.
- `return_features` / `head` — split so Spectrum can forecast the last block's image tokens and apply the head itself.

### 4.5 i2i references

`ref=` adds clean reference latents after the image tokens on RoPE's `r = 1` axis, never noised. `ref_timestep` gives them their own modulation set (a second row in `(B, 6, 2, D)`, `0` = clean); `ref_isolate` restricts reference queries to reference keys while image and text still read them. Both are stamped in the LoRA and applied by `lora.py::load_for_inference`. Tasks and recipes: [`moss/methods/i2i.md`](methods/i2i.md).

---

## 5. Training and sampling

### 5.1 The training step

Rectified flow, as the technical report's App. A.2 (`train.py::training_loss`):

$$
x_t = (1-\sigma)\,x_0 + \sigma\,\varepsilon,\qquad
v^\star = \varepsilon - x_0,\qquad
\mathcal{L} = \big\|\, v_\theta(x_t,\ 1000\sigma,\ c) - v^\star \big\|_2^2
$$

averaged over image tokens and channels, with **no σ / SNR weighting**. i2i recipes can weight the edited tokens (`region_weight`) and the copied ones (`rest_weight`) apart.

σ follows the final pretraining interval (`sample_sigma`): $Z \sim \mathcal{N}(\mu, s^2)$ with $\mu = -1.09$, $s = 0.95$; $S = 1 - \text{sigmoid}(Z)$ truncated to $S \le 0.95$ and rescaled onto the grid $\{1/1000, \dots, 1\}$ ($\sigma = \lceil 1000 S / 0.95 \rceil / 1000$), mixed with 1 % discrete uniform. Shift 1, no resolution shift — that interval trained at area ≤ 1024², the size the cache targets.

```
  cached z₀ (1,128,h,w), cached text (1,S,4096)
            │
            ▼
   σ ~ p(σ) on the 1/1000 grid,  ε ~ N(0,I)
            │
            ▼
   x_t = (1-σ)·z₀ + σ·ε
            │
            ▼
   v̂ = DiT(x_t, 1000·σ, text)        ← 42 blocks, LoRA on 252 linears
            │
            ▼
   L = mean ‖v̂ − (ε − z₀)‖²           ← image tokens only
```

Training reads only the cache (`post_image_dataset/moss[/<set>]`, two files per image), so neither the TE nor the VAE is loaded. On 16 GB the DiT runs compiled and fully resident on its int8 base, with the compiled blocks' saved activations capped by `activation_memory_budget = 0.3` instead of gradient checkpointing (`docs/moss/optimization/int8.md`). The bf16 plan swaps 36 blocks at budget 0.5; `scripts/moss/tune_swap.py` measures that pair for a machine.

### 5.2 Sampling

Defaults come from the reference pipeline (`requests.py`: `DEFAULT_STEPS` 50, `DEFAULT_CFG` 3, `DEFAULT_SHIFT` 5, `DEFAULT_UNCOND_SKIP` (22,)); read `pipeline_moss_image.py` before changing one.

- **σ grid** (`model.snapped_sigmas`): a shift-5 warped linspace, each σ snapped to the nearest σ of the 1000-step training grid and duplicates collapsed (so a high step count can return fewer steps). Timestep input is `1000·σ`.
- **CFG with a skipped uncond.** The uncond branch runs the negative prompt (default `''`, the bare template) with **block 22 skipped**; $\hat v = v_u + 3\,(v_c - v_u)$. Still two forwards per step, batched together via per-sample `skip_layers`.
- **Sampler.** `er_sde` (default here: the shared ER-SDE-Solver-3, `library.inference.sampling`, fed $\hat x_0 = x_t - \sigma \hat v$) or `euler`, the pipeline's ODE.
- Training-free extras on the same loop (`generate.py::denoise`): Spectrum (forecast the last block's features on most steps), DAVE (`--dave auto`, [`moss/inference/dave.md`](inference/dave.md)), CNS noise recolouring under ER-SDE.

---

## 6. Where LoRA attaches

`library/moss/lora.py` binds the shared `library.lines.adapters` (LoRA / LoKr / LoHa) to `layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}` — 252 linears. The fused `ffn_up_gate` gets one adapter over its full 16384-wide output, so the hidden and gate halves share the rank. `networks/` is bound to Anima module names and does not apply.

Adapters are held **outside** the blocks: the block swapper moves every `.weight` under a block between CPU and GPU, and an adapter registered inside would move with it. Shipped defaults (`configs/moss/lora.toml`): rank 32, alpha 128, Muon at lr 5e-5, kaiming `down` init — the reasoning is in [`moss/findings/muon_channel_init.md`](findings/muon_channel_init.md) and [`moss/findings/muon_no_channel_scaling.md`](findings/muon_no_channel_scaling.md); channel scaling and `dc_isolate` were removed ([`moss/channel_scaling.md`](channel_scaling.md)).

---

## 7. Minimal mental model

1. One caption → chat template → Qwen3.5-9B `last_hidden_state` at its true length, never padded. The 10 template tokens are the attention sink; `''` is the uncond.
2. The FLUX.2 latent is 2×2-patchified and BN-normalised into `(B, 128, H/16, W/16)` — 4D everywhere, one token per 16 px.
3. The DiT is 42 single-stream blocks over `[image | text]`, joint attention, all modulated by one time-only AdaLN set. Only image tokens reach the head.
4. Training is unweighted velocity MSE at logit-normal σ on the 1/1000 grid; sampling is shift 5 snapped to that grid, CFG 3 against an uncond with block 22 skipped.
