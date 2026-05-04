# EasyControl

EasyControl-style image conditioning for Anima. Trains per-block cond LoRA on
`self_attn` (q/k/v/o) and FFN (layer1/layer2), plus a per-block scalar additive
logit bias `b_cond` on the cond softmax positions. The reference image is
VAE-encoded and patch-embedded (via the DiT's frozen `x_embedder`) into
condition tokens that flow through every block alongside the target stream;
target self-attention attends to a key set extended with the cond stream's
keys/values, and `b_cond` controls how much softmax mass the cond positions
can claim.

## Architecture

**Training** runs a **two-stream block forward** — target and cond inside each
`Block.forward` in one pass. There is no separate cond pre-pass and no
cross-block `K_c/V_c` cache during training; every block produces its own
cond_k/cond_v in the same scope where the target's extended self-attention
consumes them. This keeps the cond LoRA's gradient connected step-by-step
without a deferred-backward dance.

**Inference** prefills a per-block `(K_c, V_c)` cache once at setup and reuses
it across every denoising step and every CFG branch — the cond stream is
deterministic across timesteps (`cond_temb = t_embedder(0)`, no dependence on
the noisy target, frozen DiT + frozen LoRA), so re-running it is wasted
compute. See [Inference KV cache](#inference-kv-cache) below.

```
target stream (frozen DiT)              cond stream (frozen DiT + cond LoRA, t=0)
─────────────────────────               ─────────────────────────────────────────
AdaLN_self(t_emb=t)                     AdaLN_self(cond_temb=0)
self_attn.compute_qkv(                  self_attn.qkv_proj(cond_normed)
    target_normed, rope=target_rope)      + cond_lora_qkv(cond_normed)·scale
                                        q,k,v unbind → q_norm,k_norm,v_norm
                                        apply_rotary_pos_emb_qk(cond_rope)
        │                                       │
        ▼  ◄── target attends to ──┐            ▼
target_out = LSE-extended attn     │     cond_out = SDPA(cond_q, cond_k, cond_v)
   (target_q vs [target_k;cond_k], │     (own self-attn, S_c × S_c)
    b_cond bias on cond rows)      │            │
        │                          │            ▼
        ▼                          │     output_proj(cond_out)
output_proj(target_out)            │       + cond_lora_o(cond_out)·scale
+ gate · residual                  │     + cond_gate · residual
        │                          │            │
        ▼                          │     (cross_attn skipped on cond — official
AdaLN_cross + cross_attn(text)     │      drops it for the simple two-stream form)
+ gate · residual                  │            │
        │                          │            ▼
        ▼                          │     AdaLN_mlp(cond_temb=0)
AdaLN_mlp + mlp                    │     + mlp + cond_lora_ffn{1,2}·scale
+ gate · residual                  │     + cond_gate · residual
        │                          │            │
        └─►  next block            └─►    next block (cond_x flows
                                          block-by-block as an explicit
                                          checkpoint input/output, so
                                          autograd chains across blocks
                                          naturally)
```

Concrete details:

- **Cond stream uses the same DiT modules with `cond_temb = t_embedder(0)`.**
  AdaLN modulation, q/k/v projection, q_norm/k_norm/v_norm, output_proj,
  and MLP are all the same frozen modules target uses. The cond stream just
  gets its own AdaLN modulation params (computed from the t=0 embedding) and
  its own RoPE table (`pos_embedder` at cond's native shape).
- **Cond LoRA fires only on the cond stream.** Since target and cond are
  separate tensors, we just apply the LoRA delta to `cond_x` directly — no
  mask trick needed. Target's projections see the frozen weights only.
- **Cross-attention is skipped on cond.** Target gets text via cross-attn
  as usual; cond doesn't, matching the official's "simple two-stream"
  variant. (For future spatial conditioning where cond should be text-aware,
  the alternative is to route cond-q through the cross-attn alongside
  target-q with a sparse mask. Not done here.)
- **`_ExtendedSelfAttnLSEFunc`** runs target's attention over `[target_k;
  cond_k]` without materializing the full `(S_t + S_c)²` attention matrix:
  two memory-efficient flash-attention-2 forwards on the disjoint key tiles
  plus a Python LSE-arithmetic combine. Backward is custom (FA2's stock
  backward drops the gradient flowing through `softmax_lse`); see the
  Function's docstring in `networks/methods/easycontrol.py` for the math.
  Falls back to masked-SDPA (math kernel) when flash-attn is unavailable
  with a one-shot warning.
- **No deferred backward.** cond_x is an explicit checkpoint input and
  cond_x_out an explicit return value of each patched `Block.forward`.
  The autograd chain across blocks survives the per-block unsloth /
  cpu_offload / plain `torch_checkpoint` wrappers naturally —
  `accelerator.backward(loss)` is the only call needed.

## Step-0 baseline equivalence

The bench at `bench/easycontrol/step0_equivalence.py` settles which
init makes the extended self-attention match the no-cond baseline at step 0:

| Strategy                             | rel_l2 max | mean α  | Verdict     |
| ------------------------------------ | ---------- | ------- | ----------- |
| Zero-init V_c only                   | 0.50       | 0.50    | FAIL        |
| Zero-init both K_c and V_c           | 0.38       | 0.62    | FAIL        |
| Zero-init the cond embedder          | 0.38       | 0.62    | FAIL        |
| **`b_cond` = −10 (additive logit)**  | **6.4e-5** | **1.0** | **EXACT**   |
| `b_cond` = −30                       | 9e-17      | 1.0     | bit-exact   |
| Hard mask (cond logits = −∞)         | 0.0        | 1.0     | exact       |

Naive "zero-init V_c" leaves `α = Z_t / (Z_t + Z_c) ≈ 0.5`, which rescales
the target output by ½. The fix is an additive bias on the cond logits inside
the softmax — `b_cond` initialized to −10 makes cond contribute ~e⁻¹⁰ ≈ 4.5e-5
of the total softmax mass, and is freely learnable.

`b_cond` is a per-block scalar `nn.Parameter`. Init −10 is set via
`network_args = ["b_cond_init=-10.0", ...]` in `configs/methods/easycontrol.toml`.

The same script's **Section B** (`--skip_sweep`) verifies the equivalence
holds under the live two-stream layout — separate cond Q/K/V, cond's own
RoPE at smaller S_c, cond's own self-attention. Result: rel_l2_max = 2.6e-5
in fp32 / 8.0e-4 in bf16, α ≈ 1.0 — same EXACT verdict.

## Trainable parameters

For the default `r = 16`, `D = 2048`, `num_blocks = 28`, `mlp_ratio = 4.0`:

| Component         | Shape                          | Params  |
| ----------------- | ------------------------------ | ------- |
| `cond_lora_qkv`   | 28 × (D→r→3D)                  | ~3.7 M  |
| `cond_lora_o`     | 28 × (D→r→D)                   | ~1.8 M  |
| `cond_lora_ffn1`  | 28 × (D→r→4D)                  | ~4.6 M  |
| `cond_lora_ffn2`  | 28 × (4D→r→D)                  | ~4.6 M  |
| `b_cond`          | (28,) scalars                  | 28      |
| **Total**         |                                | ~14.7 M |

Set `apply_ffn_lora=0` in `network_args` to drop the FFN LoRA — halves the
trainable count.

## `cond_token_count`: cond static-pad budget

Cond is static-padded to `cond_token_count` so block compute sees a single
S_c across all batches and buckets (compile-friendly). The default is **4096**
to match Anima's `static_token_count = 4096` constant-token bucketing — for
the common ref==target setup (where cond is the SAME cached VAE latent used
by target), cond's native tokens are ≤ 4096 by bucket design and just pad to
4096 with no downsample needed.

| `cond_token_count` | What you get                                                 | Memory cost vs no-cond baseline |
| -----------------: | ------------------------------------------------------------ | -------------------------------: |
| 4096               | full target-resolution reference; ref==target lossless       | ~1.3 GiB                        |
| 2048               | ~2× downsampled reference                                    | ~0.7 GiB                        |
| 1024               | matches official EasyControl's 32×32 reference (cond_size=512) | ~0.4 GiB                      |

If you set `cond_token_count` lower than the cond latent's native token
count, `encode_cond_latent` raises — the caller must downsample explicitly
(or we add automatic latent-space resize as a future enhancement).

## Usage

### Training

```bash
make exp-easycontrol                          # default preset
python tasks.py exp-easycontrol               # cross-platform
make exp-easycontrol PRESET=low_vram          # override hardware preset
```

Reuses the existing `cache_latents` output as the cond input — no separate
sidecar cache. Run `make preprocess` once if VAE latents aren't already
cached, then `make exp-easycontrol`.

CFG dropout for image conditioning (independent of text):
- `easycontrol_drop_p = 0.1` (default) — per batch, drop the cond entirely.
  Patched `Block.forward` then falls through to the original baseline DiT
  behavior. Lets inference do image-CFG independently of text-CFG.

### Inference

```bash
make exp-test-easycontrol REF_IMAGE=post_image_dataset/foo.png \
                          PROMPT="a girl drinking coffee at a cafe"
```

Equivalents:

```bash
python tasks.py exp-test-easycontrol post_image_dataset/foo.png \
                                     --prompt "a girl drinking coffee at a cafe"
```

Optional `EC_SCALE=0.8` to override the saved scale at test time.

## Memory envelope

Measured peak GPU memory in the smoke bench (`bench/easycontrol/two_stream_smoke.py`,
gradient checkpointing on, target latent 64×64, batch 1, bf16):

| Configuration                            | Peak GPU memory |
| ---------------------------------------- | --------------: |
| Baseline DiT only (no cond)              | ~5.0 GiB        |
| Two-stream, `cond_token_count=1024`      | ~5.4 GiB        |
| Two-stream, `cond_token_count=4096`      | ~6.3 GiB        |

A real training step on 16 GiB GPUs (live observed) lands around **7.8 GiB**
at `cond_token_count=4096`. The Phase 1.5 design pinned ~1.4 GiB more on
top of this and did not fit on 16 GiB at constant-bucket S_c.

## Inference KV cache

The cond stream is deterministic across denoising steps:

- `cond_temb = t_embedder(zeros)` is the same on every step.
- `cond_x` evolves block-by-block but never reads the noisy target.
- DiT weights, cond LoRA weights, `b_cond`, RoPE table, and `cond_scale ·
  multiplier` are all fixed at inference.

So the per-block post-RoPE post-norm `(K_c, V_c)` tensors that
`_extended_target_attention` consumes from the cond stream depend only on the
reference latent. Computing them once and pinning them is bit-equivalent to
recomputing every step.

**Lifecycle.** `_setup_easycontrol` in `library/inference/generation.py`
calls:

```python
network.set_cond(cond_latent)        # encode reference, stage cond_x_in for block 0
network.precompute_cond_kv()         # walk cond stream once, fill _cond_kv_cache
```

After this, `EasyControlNetwork._cond_kv_cache` holds a
`list[(K_c_i, V_c_i)]` of length `num_blocks`. Each entry is a BSHD pair
`[B, S_c, n_heads, head_dim]` — the same layout `_extended_target_attention`
expects, post-`q_norm/k_norm/v_norm`, post-`apply_rotary_pos_emb_qk`.

**Patched Block.forward dispatch.** Three paths in priority order:

```
_cond_kv_cache is not None          → _target_only_with_cached_cond_kv
                                       (skip cond AdaLN/qkv/SDPA/MLP entirely;
                                        feed cached K_c/V_c into target's
                                        extended self-attn)
_cond_state    is not None          → _two_stream_inner (training path)
both None                           → original_forward (baseline DiT)
```

Cache batch broadcasting: the cache is primed at `B=1` (single reference);
when CFG runs the DiT at `B>1` (cond/uncond batched), `K_c/V_c` are expanded
on the batch dim automatically. CFG-via-two-separate-forwards (the current
default at `B=1` per branch) just reuses the cache directly.

**Memory.** At default `S_c = 4096`, `n_heads = 16`, `head_dim = 128`,
`num_blocks = 28`, bf16, batch 1:

```
2 (K + V) × 28 blocks × 4096 × 16 × 128 × 2 bytes ≈ 896 MiB
```

Lower `cond_token_count` scales the cache linearly (e.g. ~448 MiB at
`cond_token_count = 2048`). The startup log reports the actual size:

```
EasyControl: precomputed cond KV cache (28 blocks × 2 tensors, ~939 MB)
```

**Speedup envelope.** Per denoising step the cache eliminates, per block:
cond AdaLN, cond LayerNorm + `qkv_proj` + cond LoRA (qkv), the cond stream's
own `S_c × S_c` SDPA, cond `output_proj` + cond LoRA (o), cond MLP +
cond LoRA (ffn1/ffn2), and the cond residual writes. Target-side cost
collapses to `_extended_target_attention` (LSE-decomposed flash) + baseline
cross-attn + baseline MLP. Practical end-to-end speedup vs the no-cache path
scales with `S_c / S_t` and the FFN LoRA ratio; expect a meaningful drop in
per-step wall time at `cond_token_count = 4096`.

**Correctness.** The cache stores the exact tensors the two-stream path
would have produced (same modules, same scale, same RoPE). Setting
`network.clear_cond_kv_cache()` and re-running falls back to the two-stream
path bit-exactly.

**Cache invalidation.** `set_cond(new_latent)` clears the cache (stale until
`precompute_cond_kv` runs again). `set_cond(None)` / `clear_cond` /
`remove_from` also clear it. If you mutate `multiplier` or `cond_scale`
manually after caching, call `clear_cond_kv_cache()` and re-prime — the
cached K/V bake the effective scale at prime time.

**Custom node use.** ComfyUI's custom node should call the same two-line
sequence (`set_cond` then `precompute_cond_kv`) once per `(reference,
cond_scale)` change; subsequent KSampler steps use the cache automatically.

## Limitations

1. **`cond_token_count` is a manual budget.** If you pass a cond latent that
   would produce more tokens than `cond_token_count`, `encode_cond_latent`
   raises; the caller must downsample upstream. Automatic latent-space
   downsample (preserving aspect ratio) is a candidate follow-up.
2. **No spatial-control positional alignment.** Cond uses its own native
   RoPE positions (matches the official's "subject" mode). The official's
   `resize_position_encoding` interpolates cond positions into target's
   coordinate system for spatial control (depth maps, edges); reproducing
   that needs fractional positions, which Anima's `pos_embedder.seq[:H]`
   integer indexing doesn't support out of the box.
3. **`blocks_to_swap = 0` recommended.** The patched `Block.forward` does
   the cond compute inside the block's forward window, so block swap is
   structurally fine — but untested with EasyControl. Pinning to 0 for now;
   bf16 frozen DiT + cond LoRA fits without swapping anyway.
4. **Custom autograd Function inside `_ExtendedSelfAttnLSEFunc`.** The
   joint-softmax backward is implemented manually because FA2's stock
   backward drops the upstream gradient on `softmax_lse`. Verified against
   masked-SDPA reference within fp32 ulp on forward and all gradients
   (`bench/easycontrol/step1p5_lse_equivalence.py`). Falls back to
   masked-SDPA when flash-attn is unavailable.

## History

This file used to describe a Phase 1.5 design where cond ran a separate
*pre-pass* across all blocks before the target forward, caching per-block
`(K_c, V_c)` on each `block.self_attn` and replaying gradients through the
serial cond chain via a `backward_cond_path()` call after
`accelerator.backward`. That pinned ~1.4 GiB of state on 16 GiB GPUs and
relied on a fragile detach + `requires_grad_(True)` dance to keep unsloth's
per-block backward from re-traversing freed saved tensors.

The current design follows the official EasyControl reference's structure
(`EasyControl/train/src/transformer_flux.py`, `EasyControl/train/src/layers.py`)
— two streams, one block forward, no cross-block cache — and keeps Anima's
LSE-decomposed extended attention as the only memory optimization on top of
that structure. The published memory result is ~7.8 GiB total in actual
training (vs Phase 1.5's >16 GiB OOM at the same bucket).

## Files

| Path                                            | Purpose                                                |
| ----------------------------------------------- | ------------------------------------------------------ |
| `networks/methods/easycontrol.py`                 | `EasyControlNetwork` + patched `Block.forward` closure |
| `configs/methods/easycontrol.toml`              | Method config                                          |
| `configs/gui-methods/easycontrol.toml`          | GUI-friendly self-contained variant                    |
| `bench/easycontrol/step0_equivalence.py` | `b_cond=-10` init recipe + two-stream verification     |
| `bench/easycontrol/step1p5_lse_equivalence.py` | LSE-decomposed Function vs masked-SDPA reference |
| `bench/easycontrol/two_stream_smoke.py`  | End-to-end forward+backward smoke + peak memory        |
