# Modulation Guidance

Quality steering via text-conditioned AdaLN modulation, based on [Starodubcev et al., "Rethinking Global Text Conditioning in Diffusion Transformers" (ICLR 2026)](https://arxiv.org/abs/2602.09268). Requires a short distillation step to train the `pooled_text_proj` MLP; thereafter inference-time steering is training-free.

## How it works

Anima's AdaLN modulation path is originally text-blind — the shift/scale/gate coefficients that control every sublayer are functions of timestep only. Text conditioning enters exclusively through cross-attention.

Modulation guidance adds a second text-conditioning channel by:

1. **Injecting a pooled text embedding into the modulation path** — a small learned MLP projects `max_pool(crossattn_emb)` into the timestep embedding space, making AdaLN coefficients text-aware.
2. **Applying guidance in modulation space at inference** — steering the AdaLN coefficients toward quality-positive and away from quality-negative directions, orthogonal to classifier-free guidance in noise space.

| Component | Text-dependent? | Notes |
|-----------|:-:|-------|
| Cross-attention KV | Yes | Qwen3 → LLMAdapter → 28 blocks |
| AdaLN shift/scale/gate | **Yes** (after training) | `t_embedder` + `pooled_text_proj` |
| CFG | Yes | Noise-space guidance (cond − uncond) |
| **Modulation guidance** | **Yes** | AdaLN-space guidance (pos − neg) |

## Architecture

### Pooled text projection

A 2-layer MLP injected into the model's forward path:

```python
# In Anima.__init__:
self.pooled_text_proj = nn.Sequential(
    nn.Linear(1024, model_channels),   # crossattn_emb dim → 2048
    nn.SiLU(),
    nn.Linear(model_channels, model_channels),
)
# Output layer is zero-initialized (no-op before distillation training)

# In Anima.forward, after t_embedding_norm:
pooled = crossattn_emb.max(dim=1).values   # (B, 1024)
t_embedding_B_T_D = t_embedding_B_T_D + self.pooled_text_proj(pooled).unsqueeze(1)
```

**Pooled text source** — max-pool from `crossattn_emb` (post-LLMAdapter), not raw Qwen3 outputs. The HydraLoRA routing analysis ([hydra-lora.md](../methods/hydra-lora.md)) empirically evaluated pooling strategies on this encoder with 1416 images across 37 artists:

| Strategy | Source | KMeans NMI |
|----------|--------|:---:|
| **Max pool** | **crossattn_emb** | **0.926** |
| Mean pool | crossattn_emb | 0.551 |
| Mean pool | prompt_embeds | 0.400 |
| EOS token | prompt_embeds | 0.170 |
| EOS token | crossattn_emb | 0.089 |

Max pool captures per-dimension peak activations from positions 16–50 (visual attribute tags), which fingerprint prompt-specific features. HydraLoRA already computes `crossattn_emb.max(dim=1).values` for expert routing — one pooling, two consumers.

### Injection point: after `t_embedding_norm`

The projection output is added to `t_embedding_B_T_D` **after** `t_embedding_norm`. Benchmarked against two alternatives:

| Injection point | MSE @ α=2.0 | MSE @ α=8.0 | Growth α=4→8 |
|-----------------|-------------|-------------|---------------|
| before_norm | 4.77e-4 | 1.08e-2 | 6.1x |
| **after_norm** | **4.76e-3** | **1.89e-1** | **7.2x** |
| adaln_lora | 4.29e-3 | 4.06e-2 | 2.6x |

`after_norm` has ~10x more sensitivity than `before_norm` (the norm re-centers perturbations) and ~4.7x the saturation headroom of `adaln_lora`.

### Inference guidance

At inference time, modulation guidance steers AdaLN coefficients using quality-axis prompts. The base projection is applied uniformly (training-consistent), but the steering delta is scheduled **per DiT block**:

```
base_emb     = t_embedding + proj(pool(main))                    # uniform, training-consistent
delta_unit   = proj(pool(p₊)) − proj(pool(p₋))                   # unit steering direction
emb_at_block[ℓ] = base_emb + w(ℓ) · delta_unit                   # scheduled per block
emb_at_final    = base_emb + w_final · delta_unit                # final_layer override
```

`delta_unit` and `w(ℓ)` are computed once at setup and reused across all denoising steps. They live on `anima._mod_guidance_delta` (unit direction, no `w` baked in), `anima._mod_guidance_schedule` (list of `w(ℓ)` of length `num_blocks`), and `anima._mod_guidance_final_w` (scalar).

**Why schedule instead of uniform?** Applying one global `w` to every block plus `final_layer` caused a drift failure mode on some LoRAs ("channel" collapsing to uniform pink at `w=3` while "sweetonedollar" stayed clean at the same `w`). Per-block functional-gap analysis on Anima's 28-block DiT showed early blocks 0–7 set coarse layout / tonal DC and are sensitive to DC blowout, while block 27 is a compensation/fix-up layer. Matches the Starodubcev et al. App. B–C analysis on FLUX (Strategy 4 — per-block `w`, not per-timestep).

### Profiles

Two named profiles cover the useful operating points — pick the safe one if a LoRA shows anatomy drift (missing fingers, extra digits) at the default.

| Profile | `start_layer` | `end_layer` | `final_w` | Use when |
|---|:-:|:-:|:-:|---|
| **`step_i8_skip27`** (default) | 8 | 27 | 0.0 | Best overall quality. Protects blocks 0–7 + 27. May occasionally show minor anatomy drift on drift-prone LoRAs. |
| **`step_i14`** (safe) | 14 | — | 0.0 | Reliably stays inside the trained manifold. Slightly less expressive — use when `step_i8_skip27` shows drift on the prompt at hand. |

Default guidance prompts use booru-style quality/score tags (in-distribution for Anima's text encoder):

| Aspect | p₊ | p₋ |
|--------|-----|-----|
| Quality+resolution | *"absurdres, masterpiece, score_9"* | *"worst quality, low quality, score_1"* |

Resolution tags (`absurdres`) are included because the quality and resolution directions correlate per-content (cosine 0.50), which works in our favor — quality guidance naturally pulls toward resolution too.

## Usage

### Inference

Default CLI invocation (reproduces the `step_i8_skip27` profile — recommended starting point):

```bash
python inference.py \
    --pooled_text_proj path/to/pooled_text_proj.safetensors \
    --mod_w 3.0 \
    --mod_pos_prompt "absurdres, masterpiece, score_9" \
    --mod_neg_prompt "worst quality, low quality, score_1" \
    # ... other args
```

To switch to the safe `step_i14` profile, add:

```bash
    --mod_start_layer 14 --mod_end_layer -1
```

To recover pre-0413 uniform behavior (not recommended — prone to pink-collapse on drift-prone LoRAs):

```bash
    --mod_start_layer 0 --mod_end_layer -1
```

| Flag | Default | Description |
|------|---------|-------------|
| `--pooled_text_proj` | — | Path to trained projection weights (.safetensors) |
| `--mod_w` | 3.0 | Peak guidance strength (`w` applied inside `[start_layer, end_layer)`) |
| `--mod_pos_prompt` | `"absurdres, masterpiece, score_9"` | Positive quality prompt |
| `--mod_neg_prompt` | `"worst quality, low quality, score_1"` | Negative quality prompt |
| `--mod_start_layer` | `8` | Inclusive first block that receives the steering delta. `0` = uniform (pre-0413). `8` = protect tonal-DC blocks 0–7. `14` = safe option. |
| `--mod_end_layer` | `27` | Exclusive last block + 1. `-1` = all remaining blocks. `27` skips Anima's final compensation block. |
| `--mod_taper` | `0` | Number of late slots inside `[start, end)` to scale by `--mod_taper_scale`. `0` disables taper. |
| `--mod_taper_scale` | `0.25` | Multiplier applied to tapered slots. |
| `--mod_final_w` | `0.0` | `w` passed to `final_layer`. `0.0` = don't disturb the output head. |

### Distillation training

The projection MLP must be trained via distillation before modulation guidance can be used. This trains only `pooled_text_proj` (~8M params) with the rest of the model frozen. **You normally never run this** — the distilled head ships as a release asset and the head is a one-shot artifact, re-distilled only when the base DiT changes. The loop lives in [`project/finished/mod_guidance/`](../../project/finished/mod_guidance/) (a finished line — the `distill-prep` / `distill-mod` targets were removed with the move; ops in its [`README.md`](../../project/finished/mod_guidance/README.md), verdicts in [`STATUS.md`](../../project/finished/mod_guidance/STATUS.md)).

**How distillation works** (paper-faithful, Starodubcev et al. §5 — *"we propagate the textual prompt solely through the pooled text embedding, using an unconditional prompt for T5"*):

1. **Teacher** forward — full model with the real `crossattn_emb` and `pooled_text_proj` disabled (original Anima behavior).
2. **Student** forward — `crossattn_emb` swapped for a cached `T5("")` unconditional baseline (same input Anima's own CFG-uncond branch uses), with the real pooled text injected via `pooled_text_proj`. This forces the projection to carry every bit of text information through the modulation path.
3. **Loss** — MSE between student and teacher noise predictions.

#### Step 1 — pre-stage

`prep.py` runs two phases:

- **Phase 1 (mandatory)** — encodes `T5("")` once into `post_image_dataset/lora/_anima_uncond_te.safetensors`. The training loop loads this as the student's unconditional crossattn input; without it the distill won't start. `make preprocess-te` already produces this sidecar for free — Phase 1 is the explicit re-stager after a model swap.
- **Phase 2 (optional, recommended)** — runs the frozen teacher (full CFG denoise from fresh noise, conditioned on each cached prompt) and writes clean latents under `post_image_dataset/distill_mod_synth/`. Training against these instead of real-image latents removes the real-vs-teacher distribution gap that floored the original val loss.

```bash
python -m project.finished.mod_guidance.prep                     # both phases (default)
python -m project.finished.mod_guidance.prep --skip_synth        # Phase 1 only (uncond sidecar)
python -m project.finished.mod_guidance.prep --skip_uncond       # Phase 2 only (assumes sidecar exists)
python -m project.finished.mod_guidance.prep --max_samples 16    # smoke-test the synth pass
```

Phase 2 defaults are tuned to Anima's production environment (`num_steps=20`, `cfg_scale=2.5`, `flow_shift=1.0`, top portrait bucket). Override `--buckets`, `--n_per_bucket`, `--num_steps`, `--cfg_scale` to taste.

#### Step 2 — distill

There is no preset merge chain here (bespoke single-GPU loop) — pass the two hardware knobs by hand: `--grad_ckpt` (+ unsloth CPU offload) and `--blocks_to_swap N`, the `low_vram` equivalents. See the [line README](../../project/finished/mod_guidance/README.md) for the full preset translation.

Invocation (current defaults shown):

```bash
python -m project.finished.mod_guidance.distill \
    --data_dir post_image_dataset/lora \
    --dit_path models/diffusion_models/anima-base-v1.0.safetensors \
    --output_path output/ckpt/pooled_text_proj.safetensors \
    --synth_data_dir post_image_dataset/distill_mod_synth \
    --iterations 5000 \
    --lr 2e-5 \
    --batch_size 1 \
    --warmup 0.02 \
    --attn_mode flash
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | `post_image_dataset/lora` | Directory with cached latents + TE sidecars. Also where the Phase 1 `_anima_uncond_te.safetensors` sidecar is looked up. |
| `--uncond_te_path` | `<data_dir>/_anima_uncond_te.safetensors` | Override path to the T5("") sidecar from `prep.py` Phase 1. |
| `--synth_data_dir` | `None` | Phase 2 synthetic-latent dir. When set, latents come from here (matched by stem + resolution); TE caches still come from `--data_dir`. |
| `--dit_path` | `models/diffusion_models/anima-base-v1.0.safetensors` | Base model. |
| `--output_path` | `output/ckpt/pooled_text_proj.safetensors` | Where to save the trained projection (`make test MOD=1` picks this path up automatically). |
| `--iterations` | 5000 | Training iterations. |
| `--lr` | 2e-5 | Peak learning rate (cosine to 10%). |
| `--warmup` | 0.02 | ≥1 = absolute steps; <1 = ratio of iterations. |
| `--batch_size` | 1 | Per-step batch size. |
| `--grad_accum` | 1 | Gradient accumulation steps (effective batch = `batch_size * grad_accum`). |
| `--grad_ckpt` / `--no_grad_ckpt` | on | Gradient checkpointing w/ unsloth CPU offload. Disable if you have VRAM headroom — faster iteration. |
| `--blocks_to_swap` | 0 | CPU-offload N transformer blocks. Only needed on very tight VRAM. |
| `--torch_compile` / `--no_compile` | on | Compile each `Block._forward` on native-shape buckets (one block graph per distinct token count, no static-pad flash leak). |
| `--validation_split` | 0.05 | Held-out fraction for val pass; set 0 to disable. |
| `--validate_every_n_steps` | 1000 | Val cadence. Best-val checkpoint replaces step-cadence saves when validation is on. |
| `--teacher_cache_K` | 6 | Number of pre-sampled sigma bins per sample. Each visit returns one of K deterministic (σ, noise) pairs, so the teacher forward is run at most K times per sample across the run. |
| `--no_teacher_cache` | off | A/B against the cache or recover the original continuous-sigma sampler. |
| `--prefill_teacher_cache` | off | Eagerly run every (sample, σ_idx) up front (~K·N teacher forwards) so training is teacher-free. |
| `--sample_ratio` | 1.0 | Fraction of (post-split) samples to keep per bucket; mirrors LoRA's per-subset `sample_ratio`. |

**VRAM notes.** The teacher forward runs under `torch.no_grad()` so it holds almost nothing; the student forward is what dominates peak VRAM (~12 GB on the default config). With `--no_grad_ckpt` you'll see VRAM swing between the weights-only baseline and that student peak — this is normal. Leave `--grad_ckpt` on (the default) only if the peak doesn't fit; if it does, `--no_grad_ckpt --blocks_to_swap 0` is faster.

**Teacher cache.** Because the K-grid pre-samples both σ and per-(sample, σ_idx) noise, every cache miss commits one deterministic teacher prediction that every later visit hits directly — no recomputation, identical (latents, noise, σ) inputs to the student whether the cache hit or missed. RAM footprint scales as `dataset_size × K × latent_bytes` (≈ few GB at default K=6 + Anima's 16×H×W bf16 latents); shrink K if RAM is tight.

Output: `output/ckpt/pooled_text_proj.safetensors`. Use it at inference via `--pooled_text_proj <path>` (or `make test MOD=1`, which auto-discovers it).

## Compatibility

| Feature | Interaction |
|---------|-------------|
| **T-LoRA** | Orthogonal — T-LoRA masks LoRA rank by timestep; modulation guidance steers AdaLN coefficients. Different parameter spaces. |
| **CFG** | Complementary — CFG in noise space, modulation guidance in AdaLN space. They stack. |
| **P-GRAFT** | Compatible — modulation guidance runs independently of LoRA presence. |
| **Spectrum** | Compatible — Spectrum skips blocks but still runs `t_embedder` + `final_layer`. Guidance delta applies to `emb_B_T_D` before blocks, carried through on cached steps. |
| **HydraLoRA** | Both have consumed pooled crossattn in the past, but HydraLoRA's current router reads the post-`lora_down` rank-R signal — so no shared pool any more. Orthogonal paths. |
| **LoRA training** | No conflict — LoRA explicitly excludes `pooled_text_proj` via the exclude pattern in `networks/lora_anima/factory.py`. |

## ComfyUI

Mod guidance ships inside the [ComfyUI-Spectrum-KSampler](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler) repo as a drop-in KSampler replacement. The adapter (`pooled_text_proj_0413.safetensors`, ~12MB) is auto-downloaded on first use from the anima_lora GitHub release page.

| Node | Inputs | When to use |
|---|---|---|
| **KSampler (Spectrum)** | `quality_tags`, `quality_neg`, `mod_w_profile` dropdown (+ SEA `refresh_ratio`, SMC) | Everyday use. Mod guidance folded into the unified sampler — pick a profile preset, `off` to disable. |
| **KSampler (Spectrum + Mod Guidance Advanced)** | full `start_layer` / `end_layer` / `taper` / `final_w` sliders + Spectrum knobs | Workflow tuning or per-LoRA experimentation. |
| **Anima Mod Guidance (model patch)** | `quality_tags`, `quality_neg`, `mod_w_profile` | Standalone `MODEL → MODEL` patcher; composes with any sampler. |

The `mod_w_profile` dropdown exposes the same two profiles documented above plus `uniform_w3` for reproducing pre-0413 behavior. Default is `step_i8_skip27`; switch to `step_i14` when a LoRA shows anatomy drift.

**`quality_neg` — decoupled steering negative.** The mod delta is `proj(quality_tags) − proj(quality_neg)`. Historically the node reused the **CFG** negative here, which is anti-correlated (cos ≈ −0.38) with the intended quality axis and weaker — the broad CFG negative is a velocity-space repulsion target, not a clean quality counter-pole. `quality_neg` gives the steering axis its own baseline (e.g. `worst quality, score_1`), matching the CLI's long-standing `--mod_pos_prompt` / `--mod_neg_prompt` split. **Empty = reuse the CFG negative** (legacy behavior, bit-for-bit). The CFG negative is unchanged either way.

## Design rationale

### Quality-axis separation in embedding space

`max_pool(crossattn_emb)` separates quality-positive from quality-negative prompts. Tested with 8 diverse content prompts:

| Metric | Value |
|--------|-------|
| Avg pos↔neg cosine distance (content-varied) | 0.038 |
| Avg same-quality cosine distance (different content) | 0.031 |
| Separation ratio | 1.22x |

### Quality direction consistency across content

The quality direction `max_pool(p₊) - max_pool(p₋)` is consistent across 8 diverse content types:

| Metric | Value |
|--------|-------|
| Average pairwise cosine similarity | 0.814 |
| Minimum pairwise cosine similarity | 0.770 |

All 28 pairwise similarities exceed 0.77 — a single global guidance direction generalizes across content.

### Modulation sensitivity

Perturbing `emb_B_T_D` with the quality direction produces smooth, monotonic changes in noise predictions. The high-noise regime (t=0.9) is most sensitive, consistent with the model relying more on modulation at early denoising steps.
