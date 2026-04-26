# IP-Adapter

Decoupled image cross-attention. A reference image is encoded by a frozen vision tower (PE-Core-L14-336 by default), reduced to K=16 compact tokens by a learned Perceiver resampler, then projected per DiT block into parallel `to_k_ip` / `to_v_ip` matrices. The patched cross-attention adds `scale * SDPA(text_q, ip_k, ip_v)` to the existing text cross-attention output. Matches the reference IP-Adapter (Ye et al. 2023) — a single global `scale`, no per-block learnable gate.

DiT is **frozen** — only the resampler and the per-block KV projections train. ~150M trainable params total at the default config (`K=16`, 28 blocks, hidden=2048).

Companion: `prefix-tuning.md` and `postfix-sigma.md` (text-side input-space conditioning); `mod-guidance.md` (per-block AdaLN path). IP-Adapter is the only one of these that hooks DiT cross-attention via a parallel attention pass — postfix/prefix concatenate into the existing text K/V.

---

## Design

```
reference image
   │
   ▼
PE-Core-L14-336    ──►  patch tokens  [B, ~257, 1024]   (frozen, eval, no_grad)
   │
   ▼
Perceiver resampler ──► IP tokens     [B, K=16, 1024]   (trainable)
   │
   │  ┌── per cross-attn block (28 of them) ──────────────────────────┐
   ▼  │                                                                │
   ┌──┴───────┐                          ┌───────────┐                │
   │ to_k_ip  │ ──► ip_k [B, K, 16, 128] │  k_norm   │ (cross_attn's  │
   │ Linear   │                          │ (shared)  │  RMSNorm)      │
   └──────────┘                          └───────────┘                │
   ┌──────────┐                          ┌───────────┐                │
   │ to_v_ip  │ ──► ip_v [B, K, 16, 128] │  v_norm   │ (Identity)     │
   │ Linear   │                          └───────────┘                │
   └──────────┘                                                        │
                                                                       │
   text_result = attention(q, k_text, v_text, ...)                    │
   ip_out      = SDPA(q, ip_k, ip_v)            ← decoupled, no mask   │
   out         = output_proj(text_result + scale * ip_out)            │
                                                                       │
   └───────────────────────────────────────────────────────────────────┘
```

### Key design choices

- **Decoupled SDPA, not concatenated KV.** Concatenating IP tokens into the text K/V (the "coupled" variant — what postfix already does) couples the softmax denominators and the IP path can't carry detail without stealing attention from text. Decoupled attention with its own softmax is the original IP-Adapter design and works much better for identity / style preservation.
- **No learned per-block gate.** Step-0 baseline equivalence comes from `to_v_ip` near-zero init (`std=1e-4`): `cross_attn.v_norm` is `nn.Identity`, so V's magnitude passes through, and `ip_out = SDPA(q, k, v)` is dominated by V — a tiny V means a tiny ip_out, regardless of what `k_norm` (RMSNorm) does to K's direction. Adding `scale * ip_out` to `text_result` at step 0 is therefore numerically negligible. An earlier variant of this network had a per-block learnable gate `α_i` initialized to zero (LoRA / ControlNet style). Empirically that **degenerate-converged**: with α and K/V both multiplicatively gating the residual, Adam preferred to keep α near zero (mean ~4e-5 after 4 epochs, sign-flipping noise) while letting K/V grow ~70× to compensate. Same product, but a stiff parameterization that bottlenecked training. Removing α and trusting the V init matches reference IP-Adapter (Ye et al. 2023) and lets K/V train at their natural Adam pace from step 0.
- **`to_k_ip` / `to_v_ip` init: `std=1e-4`.** This *is* the step-0 baseline mechanism — see the previous bullet. It also serves as symmetry-breaking for the K side (which gets RMSNorm'd to unit magnitude on the K direction, but the directions themselves come from this init).
- **Per-block KV cached once per batch.** `set_ip_tokens(...)` runs the resampler + 28 KV projections + RMSNorms once and stashes `[B, K, n_h, d_h]` tensors on each `cross_attn._ip_k_cached / _ip_v_cached`. The patched cross-attn forward is then a single SDPA call. Under gradient checkpointing the recomputed forward reads the same stashed K/V.
- **`B=1 → B=N` broadcast.** At inference, the cond and uncond CFG passes both use the same single ref image. `set_ip_tokens` accepts `[1, K, 1024]`; the patched forward `expand`s to match `q`'s batch dimension (free view, no copy).
- **Pre-cached PE features (default).** `make ip-adapter-cache` runs PE-Core once over `post_image_dataset/` and writes `{stem}_anima_pe.safetensors` sidecars (`[T_pe, d_enc]` bf16). At training time the dataset loads these into `batch["ip_features"]` and `train.py` skips loading the vision encoder entirely (saves ~600 MB VRAM) — same defaults as VAE latents and text encoder outputs. Set `ip_features_cache_to_disk = false` (and `cache_latents = false`) to fall back to the live-encoding path, which keeps PE-Core resident in bf16 and runs it on `batch["images"]` every step.

### Why PE-Core

PE-Core-L14-336 supports **dynamic resolution** out of the box — each ref image is resized to its closest patch-14 bucket (`scripts/img2emb/buckets.py:PE_CORE_L14_336_SPEC`, ~576 patch tokens, aspects 1:2 to 2:1). TIPSv2 also works (set `network_args = ["encoder=tipsv2"]`) but its bucket count is larger (~1024 tokens) and you need `make download-tipsv2` plus `trust_remote_code=True`.

---

## Training contract

1. `apply_to(unet=anima)` monkey-patches each `Block.cross_attn.forward` with a closure that captures `(orig_attn, ip_net, anima_attention)`. Lives on the instance, so the patch survives gradient-checkpointing reroll.
2. Per training batch, `train.py:_maybe_set_ip_tokens` resolves PE features (cached `batch["ip_features"]` by default, live PE on `batch["images"]` as fallback), feeds them through the resampler, and calls `network.set_ip_tokens(ip_tokens)`.
3. With probability `ip_image_drop_p` (default 0.1) the whole-batch image conditioning is **dropped** (`set_ip_tokens(None)`) — CFG dropout for the image branch, independent of caption dropout.
4. The DiT forward runs as normal. Inside each cross-attn, the patched forward computes the text path via the existing `attention.attention(...)` call, then adds `scale * SDPA(q, ip_k, ip_v)` before `output_proj`.
5. Backward flows through the IP path back through the resampler and per-block KV projections. DiT params have `requires_grad=False`.

Reference image and target image are the **same image** (sampled from `post_image_dataset/`). The model learns: "given image X as reference + caption Y, generate X." With image dropout it also learns "given caption Y alone, generate something that looks like Y" — preserving the base behavior.

### Caption dropout pitfall

Original IP-Adapter recipe (Ye et al. 2023) uses caption_dropout ≈ 0.05 trained on millions of image-text pairs. With Anima's `post_image_dataset` (small, ref==target), high caption dropout is a footgun: the model finds it easier to **memorize a representative image** to emit when the caption is missing than to learn a faithful image-conditional decoder, leading to mode collapse where every reference produces near-identical outputs. Empirically, `caption_dropout_rate=0.5` produced clean mode collapse in 2 epochs. **Recommended range: 0.10–0.20** for this dataset.

### Watching training (diagnostic log)

`train.py` enables runtime diagnostics on the network and logs a per-epoch summary. A healthy run looks like:

```
[IP-Adapter diag] params: ‖to_k_ip‖ min=… mean=… max=…  | ‖to_v_ip‖ min=… mean=… max=…
[IP-Adapter diag] runtime: ‖scale·ip_out‖/‖text_result‖ min=… mean=… max=…  (N=… calls)
[IP-Adapter diag]   block[0]:  ‖k‖=… ‖v‖=… ratio=…
[IP-Adapter diag]   block[14]: ‖k‖=… ‖v‖=… ratio=…
[IP-Adapter diag]   block[27]: ‖k‖=… ‖v‖=… ratio=…
```

What to expect / look for:
- **`‖to_v_ip‖`** is the signal-bearing side — `v_norm` is `nn.Identity`, so V's magnitude controls how much energy the IP path injects. Watch this trajectory rather than `‖to_k_ip‖`.
- **`‖to_k_ip‖`** is direction-only after `k_norm` (RMSNorm). It can grow to non-trivial Frobenius norm without changing per-head K direction much; treat it as informational.
- **runtime ratio** is `‖scale·ip_out‖/‖text_result‖` averaged over the epoch. At init it should be near 0 (V near zero from `ip_init_std=1e-4`). It typically rises to **~0.1–0.5** as K/V learn — the IP path becoming a meaningful fraction of the cross-attn output. Mean ratios > 1 sustained across blocks would suggest the IP path is overpowering text, which is unhealthy.
- **Block-to-block variation is expected.** The earlier-stack and middle blocks usually carry more IP signal than the final refinement blocks. Worry only if *every* block stays at ratio ≈ 0 deep into training — that means the IP path isn't being used at all.

Implementation: 0-d float32 / int64 tensors live on each `cross_attn` as `_ip_diag_ratio_sum` / `_ip_diag_count`, updated detached + on-device in the patched forward (no `.item()` host sync per step). `diagnostic_summary(reset=True)` is called from train.py at every epoch end, then it walks the modules to build the aggregate.

---

## Inference flow

```
load DiT  →  load IP-Adapter weights  →  apply_to(anima)   ← patches cross-attn
      │
      ▼
load PE-Core  →  encode REF_IMAGE  →  resampler  →  ip_tokens [1, K, 1024]
      │
      ▼
network.set_ip_tokens(ip_tokens)
      │
      ▼
denoising loop (cond + uncond CFG passes)
      ├─ cond:   text=positive_prompt + image=ref          ← IP active
      └─ uncond: text=negative_prompt + image=ref          ← IP active (same K/V)

      → guidance amplifies (text - null_text), IP stays in both → image cond not in CFG term
```

The IP path is **always on** during the denoising loop in v1 — image conditioning rides along but isn't part of the text-CFG steering. This is the simplest, most predictable behavior. A future `--ip_negative_drop` flag could clear IP for the uncond pass to do true image-CFG.

### How to drive `make test-ip`

Three usage patterns, with the prompt strategy that matches each:

| Pattern | Ref provides | Prompt should | Example |
|---|---|---|---|
| **Reproduction** | identity + content | empty / minimal — let the ref drive everything | sanity check that IP is wired in |
| **Style transfer** | style / aesthetic | the *new* content/composition you want, not a redescription of the ref | ref = Ghibli still, prompt = "a cat on a windowsill" |
| **Identity preservation** | character / face | new pose / scene / action | ref = character A, prompt = "drinking coffee at a cafe" |

The cardinal rule: **prompt ⊕ ref, not prompt ≈ ref**. Don't echo the ref's content into the prompt — it splits guidance and you get the worst of both channels.

Knob notes (specific to v1's "IP always on" CFG behavior):
- `guidance_scale` (text CFG) — consider lowering to 3–4 instead of the usual 5–7. Text gets amplified by CFG, IP doesn't, so a high text-CFG drowns out a working IP signal.
- `IP_SCALE` — if outputs ignore the ref, push to 1.5–2.0; if outputs lock onto the ref too hard, drop to 0.3–0.7.
- Number of denoising steps — leave normal. IP doesn't reduce the steps you need.

A debug ladder for a fresh checkpoint:

```bash
# 1. Pure reproduction — does IP do anything?
make test-ip REF_IMAGE=post_image_dataset/foo.png PROMPT="" IP_SCALE=1.0

# 2. Crank IP — confirm the path is alive at all
make test-ip REF_IMAGE=post_image_dataset/foo.png PROMPT="" IP_SCALE=2.0

# 3. Style transfer — different content, ref's style
make test-ip REF_IMAGE=post_image_dataset/foo.png \
    PROMPT="1girl, drinking coffee at a cafe" IP_SCALE=1.0
```

If step 1 returns the model's "default" image regardless of ref → IP path isn't carrying signal (check the `runtime ratio` in the diag log: ratio ≈ 0 across all blocks means K/V never learned to use the ref). If step 2 still returns the same image as step 1 → patched forward is broken or weights weren't loaded. If different refs at step 2 produce *the same* output → resampler has collapsed (run a resampler-diversity smoke test before training more).

---

## Quick start

### Train

```bash
make ip-adapter
```

Runs `train.py --method ip_adapter --preset default`. Override the preset like the LoRA targets (`PRESET=low_vram make ip-adapter`).

### Inference

```bash
# Default prompt is "double peace, v v," — minimal so the ref image drives style.
make test-ip REF_IMAGE=post_image_dataset/foo.png

# Custom content with the ref's style:
make test-ip REF_IMAGE=foo.png PROMPT="a girl drinking coffee at a cafe"

# Tune adapter strength:
make test-ip REF_IMAGE=foo.png PROMPT="..." IP_SCALE=0.7

# Negative prompt:
make test-ip REF_IMAGE=foo.png PROMPT="..." NEG="bad anatomy"
```

Outputs land under `output/tests/ip/`. Each generated PNG gets a paired `*_ref.png` copy of the reference for side-by-side review. The output resolution auto-snaps to the closest `CONSTANT_TOKEN_BUCKETS` entry to the ref's aspect ratio (`--ip_image_match_size`, set by the make target).

Direct CLI:

```bash
python inference.py \
    --dit models/diffusion_models/anima-preview3-base.safetensors \
    --text_encoder models/text_encoders/qwen_3_06b_base.safetensors \
    --vae models/vae/qwen_image_vae.safetensors \
    --ip_adapter_weight output/ckpt/anima_ip_adapter.safetensors \
    --ip_image foo.png \
    --ip_scale 0.8 \
    --ip_image_match_size \
    --prompt "your prompt"
```

`--ip_scale` overrides the saved `ss_ip_scale` from the checkpoint metadata; omit to use the trained default (typically 1.0).

---

## Config reference

`configs/methods/ip_adapter.toml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `network_dim` | 16 | K = number of IP tokens (resampler output slots) |
| `network_args.encoder` | `pe` | Vision encoder name (`pe`, `tipsv2`, `pe-g`) |
| `network_args.encoder_dim` | 1024 | Encoder hidden dim — must match the encoder |
| `network_args.resampler_layers` | 2 | Perceiver resampler depth |
| `network_args.resampler_heads` | 8 | Perceiver resampler attention heads |
| `network_args.ip_init_std` | 1e-4 | Init std for `to_k_ip` / `to_v_ip`. **This is what gives step-0 baseline equivalence.** Since `v_norm` is `nn.Identity`, near-zero `to_v_ip` produces near-zero `ip_out` regardless of `k_norm`. |
| `network_args.ip_scale` | 1.0 | Global scale on the IP attention contribution. Single knob; no per-block gate. |
| `learning_rate` | 1e-4 | Same LR for resampler and per-block KV projections (two param groups: `ip_resampler`, `ip_kv_proj`). |
| `cache_latents` | true | Inherited from `base.toml`; compatible with cached PE features. Set to `false` only when running the live-encoding fallback. |
| `cache_text_encoder_outputs` | true | Text path unchanged from LoRA training |
| `ip_features_cache_to_disk` | true | Reads `{stem}_anima_pe.safetensors` sidecars produced by `make ip-adapter-cache`. Missing cache = `FileNotFoundError`. Disable to fall back to live PE on `batch["images"]` (also set `cache_latents=false`). |
| `use_ip_adapter` | true | Method-forced; flips on `_maybe_set_ip_tokens` in train.py |
| `ip_image_drop_p` | 0.1 | Whole-batch image-conditioning dropout (CFG dropout) |
| `caption_dropout_rate` | 0.15–0.20 (recommended) | Text-side dropout. **Do not set this to 0.5+** on `post_image_dataset` — see "Caption dropout pitfall" above. Reasonable default is 0.10–0.20; the original IP-Adapter recipe used 0.05. |
| `blocks_to_swap` | 0 | DiT is frozen; no swapping needed |

---

## Compatibility

| Component | Compat | Notes |
|---|---|---|
| Cached latents | ✅ | Default. PE features are pre-cached alongside latents/TE; the vision encoder isn't loaded during training. Live-encoding fallback (`ip_features_cache_to_disk=false`) requires `cache_latents=false`. |
| Cached text encoder outputs | ✅ | Text path is unchanged — works exactly as in LoRA training. |
| `caption_shuffle_variants` | ✅ | Image conditioning is independent of caption shuffling. |
| Gradient checkpointing | ✅ | Patched cross-attn lives on the module instance; recompute reads the same stashed K/V. |
| `torch.compile` (`static_token_count=4096`) | ✅ | Patched forward is regular Python; SDPA inlines under compile. Constant-token bucketing applies as usual. |
| Block swapping | ✅ (irrelevant) | DiT is frozen; `blocks_to_swap=0` is the default. |
| Modulation guidance | ✅ orthogonal | Modulation = AdaLN path; IP-Adapter = parallel cross-attn. Stack freely. |
| LoRA stack | ⚠ untested | Should compose (LoRA targets `Linear`, IP-Adapter targets `cross_attn.forward`). v1 is adapter-only; LoRA-on-top is a future variant. |
| Spectrum inference | ⚠ probably ok | IP K/V are fixed per generation; cached steps skip cross-attn entirely so IP doesn't run. Untested. |
| Tiled diffusion | ⚠ untested | Each tile would attend the same IP K/V. Likely ok but unverified. |
| ComfyUI | ❌ today | No custom node yet. Plan: extend `custom_nodes/comfyui-hydralora` with an IP-Adapter branch. |

---

## Files

- `networks/ip_adapter_anima.py` — `IPAdapterNetwork`, the patched-forward closure, save/load, runtime diagnostics (`set_diagnostics_enabled` / `diagnostic_summary`).
- `library/vision/encoder.py` — PE-Core wrapper (`load_pe_encoder`, `encode_pe_from_imageminus1to1`). Used by both the cache script and the live-encoding fallback.
- `preprocess/cache_pe_encoder.py` — `make ip-adapter-cache` entry point. Writes `{stem}_anima_{encoder}.safetensors` sidecars next to each image.
- `library/datasets/base.py` — `_try_load_ip_features` reads sidecars in `__getitem__`; stacks into `batch["ip_features"]` per training bucket.
- `scripts/img2emb/resampler.py` — `PerceiverResampler` (reused).
- `scripts/img2emb/buckets.py` — per-encoder PE bucket spec for dynamic-resolution resize.
- `train.py` — `_maybe_set_ip_tokens` hook in `get_noise_pred_and_target`; calls `set_diagnostics_enabled` after the network is built; per-epoch `diagnostic_summary` log.
- `library/inference/generation.py` — `_setup_ip_adapter` in `generate()`; `--ip_image_match_size` aspect snap.
- `library/anima/training.py` — `--use_ip_adapter`, `--ip_image_drop_p`, `--ip_encoder`, `--ip_features_cache_to_disk` argparse.
- `configs/methods/ip_adapter.toml` — method config.
