# EasyControl — Phase 1

EasyControl-style image conditioning for Anima. Trains a per-block cond LoRA
on `self_attn` (q/k/v/o) and FFN (layer1/layer2), plus a per-block scalar
additive logit bias `b_cond` on the cond softmax positions. The reference
image is VAE-encoded and patch-embedded (via the DiT's frozen `x_embedder`)
into condition tokens that are run through every block via a separate cond
pre-pass; the resulting per-block `(K_c, V_c)` are concatenated onto target
self-attn keys and values for the rest of the DiT forward.

> **Status.** Phase 1 of the [proposal](../../easycontrol_proposal.md) — minimal
> training path. No KV-cache (Phase 2). No multi-condition (Phase 4).

## Architectural recap

For each Anima `Block`:

```
target tokens                              condition tokens (cached once per batch)
[B, S_t, D]                                [B, S_c, D]  (Phase 1: S_c = S_t, ref==target)
     │                                          │
     ▼                                          ▼
self_attn (frozen)                       self_attn (frozen) + cond_lora_qkv (trainable)
     │                                          │
     │                                          │  → cache (K_c, V_c) per block
     ▼                                          ▼
target_Q ⨯ [K_t; K_c]                    cond_Q ⨯ K_c           (no leak from target)
       softmax over [logits_t ; logits_c + b_cond]              softmax over logits_c
     │                                          │
     ▼                                          ▼
output_proj (frozen)                     output_proj + cond_lora_o (residual)
     │                                          │
     ▼                                          │   ← cross_attn skipped on cond
cross_attn (frozen, text)                       │
     │                                          ▼
     ▼                                  layer_norm_mlp + mlp + cond_lora_ffn(1,2)
mlp (frozen) + AdaLN gates                      │
     │                                          ▼
   block i+1 target                       block i+1 cond
```

The condition path runs eagerly (outside `_run_blocks`) because it doesn't
share AdaLN, RoPE, or the static-shape compile boundary with the target path.
At inference, the extended-key SDPA path is taken once per step on the target
side; in Phase 2 the `(K_c, V_c)` will be reused across all denoising steps so
the cond pre-pass collapses to a one-time setup.

## Step-0 baseline equivalence

The bench at `bench/active/easycontrol/step0_equivalence.py` settles which
init strategy makes the extended-self-attention forward pass identical to the
baseline DiT forward at step 0:

| Strategy                             | rel_l2 max | mean α  | Verdict     |
| ------------------------------------ | ---------- | ------- | ----------- |
| Zero-init V_c only                   | 0.50       | 0.50    | FAIL        |
| Zero-init both K_c and V_c           | 0.38       | 0.62    | FAIL        |
| Zero-init the cond embedder          | 0.38       | 0.62    | FAIL        |
| **`b_cond` = −10 (additive logit)**  | **6.4e-5** | **1.0** | **EXACT**   |
| `b_cond` = −30                       | 9e-17      | 1.0     | bit-exact   |
| Hard mask (cond logits = −∞)         | 0.0        | 1.0     | exact       |

Naive "zero-init V_c" leaves `α = Z_t / (Z_t + Z_c) ≈ 0.5`, which rescales the
target output by ½. The fix is an additive bias on the cond logits inside the
softmax — `b_cond` initialized to −10 makes the cond contribute ~e⁻¹⁰ ≈ 4.5e-5
of the total softmax mass, and is freely learnable.

`b_cond` is a per-block scalar `nn.Parameter`. Init −10 is hard-coded into
`configs/methods/easycontrol.toml` (`network_args = ["b_cond_init=-10.0", ...]`).

## Trainable parameters

For the default `r = 16`, `D = 2048`, `num_blocks = 28`, `mlp_ratio = 4.0`:

| Component         | Shape                          | Params |
| ----------------- | ------------------------------ | ------ |
| `cond_lora_qkv`   | 28 × (D→r→3D)                  | ~3.7 M |
| `cond_lora_o`     | 28 × (D→r→D)                   | ~1.8 M |
| `cond_lora_ffn1`  | 28 × (D→r→4D)                  | ~4.6 M |
| `cond_lora_ffn2`  | 28 × (4D→r→D)                  | ~4.6 M |
| `b_cond`          | (28,) scalars                  | 28     |
| **Total**         |                                | ~14.7 M |

Set `apply_ffn_lora=0` in `network_args` to drop the FFN LoRA — halves the
trainable count. (Phase 1 default is on per the published recipe.)

## Usage

### Training

```bash
make easycontrol                          # default preset
python tasks.py easycontrol               # cross-platform
make easycontrol PRESET=low_vram          # override hardware preset
```

Reuses the existing `cache_latents` output as the cond input — no separate
sidecar cache. Run `make preprocess` once if VAE latents aren't already
cached, then `make easycontrol`.

CFG dropout for image conditioning (independent of text):
- `easycontrol_drop_p = 0.1` (default) — per batch, drop the cond entirely.

### Inference

```bash
make test-easycontrol REF_IMAGE=post_image_dataset/foo.png \
                      PROMPT="a girl drinking coffee at a cafe"
```

Equivalents:

```bash
python tasks.py test-easycontrol post_image_dataset/foo.png \
                                 --prompt "a girl drinking coffee at a cafe"
```

Optional `EC_SCALE=0.8` to override the saved scale at test time.

## Phase 1 limitations

1. **No KV cache.** The cond pre-pass runs every step (training and inference).
   Phase 2 lifts this for inference: `precompute_cond_path()` runs once per
   `generate()` setup and the cached `(K_c, V_c)` are reused across all
   denoising steps.
2. **Plain LayerNorm on cond, not AdaLN.** The cond stream has no per-token
   timestep embedding to drive AdaLN scale/shift/gate. Phase 1 uses a plain
   transformer block forward (LayerNorm + residual). Step 0 baseline
   equivalence is preserved by `b_cond=-10` regardless. A future option:
   share target's `t_emb` with cond as a fixed input.
3. **No RoPE on cond.** Cond positions are independent of target. Phase 1
   skips RoPE on cond Q/K. Same step-0 caveat applies.
4. **Forced `blocks_to_swap=0`.** The cond pre-pass interleave with block
   swap is flagged as integration risk in the proposal. Phase 1 pins to 0,
   matching APEX and IP-Adapter's choice for the same reason.
5. **Forced torch SDPA on the patched extended-key path.** The additive
   logit bias on cond positions doesn't compose with flash/xformers/sageattn/
   flex (none accept a free-form additive bias on a subset of keys). Target
   self-attn falls back to torch SDPA when cond is set; baseline (no cond)
   still uses the configured `attn_mode`.

## Files

| Path                                          | Purpose                                          |
| --------------------------------------------- | ------------------------------------------------ |
| `networks/easycontrol_anima.py`               | `EasyControlNetwork` + patched self-attn closure |
| `configs/methods/easycontrol.toml`            | Method config (toggle-blocks-style sibling)      |
| `configs/gui-methods/easycontrol.toml`        | GUI-friendly self-contained variant              |
| `bench/active/easycontrol/step0_equivalence.py` | Verifies the `b_cond=-10` init recipe          |
| `easycontrol_proposal.md`                     | Full RFC with phase plan                         |
