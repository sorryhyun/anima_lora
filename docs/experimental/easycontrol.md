# EasyControl

EasyControl-style image conditioning for Anima. Trains per-block cond LoRA on
`self_attn` (q/k/v/o) and FFN (layer1/layer2), plus a per-block scalar additive
logit bias `b_cond` on the cond softmax positions. The reference image is
VAE-encoded and patch-embedded (via the DiT's frozen `x_embedder`) into
condition tokens that flow through every block alongside the target stream;
target self-attention attends to a key set extended with the cond stream's
keys/values, and `b_cond` controls how much softmax mass the cond positions
can claim.

![EasyControl two-stream shared attention](assets/easycontrol_attention.png)

Structurally this is the same shape as a unified multimodal transformer that
shares one attention between a clean subsequence and a noisy one under a
block-structured mask: the cond stream plays the clean/authoritative role
(attends only to itself, never reads the noisy target — which is exactly why it
is deterministic across timesteps and KV-cacheable at inference), and the target
stream full-attends over `[K_t; K_c]`. Two differences from the canonical
picture: the cond block is full/bidirectional (a clean image), not causal
triangular; and the hard attend/mask boundary on the target→cond columns is
replaced by a learnable scalar `b_cond` gate (init −10 → step-0 equivalence,
then learned upward). Regenerate the figure with
`uv run python docs/experimental/assets/easycontrol_attention.py`.

## Architecture

Training runs a two-stream block forward — target and cond inside each
`Block.forward` in one pass. There is no separate cond pre-pass and no
cross-block `K_c/V_c` cache during training; every block produces its own
cond_k/cond_v in the same scope where the target's extended self-attention
consumes them. This keeps the cond LoRA's gradient connected step-by-step
without a deferred-backward dance.

Inference prefills a per-block `(K_c, V_c)` cache once at setup and reuses
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

- Cond stream uses the same DiT modules with `cond_temb = t_embedder(0)`.
  AdaLN modulation, q/k/v projection, q_norm/k_norm/v_norm, output_proj,
  and MLP are all the same frozen modules target uses. The cond stream just
  gets its own AdaLN modulation params (computed from the t=0 embedding) and
  its own RoPE table (`pos_embedder` at cond's native shape).
- Cond LoRA fires only on the cond stream. Since target and cond are
  separate tensors, we just apply the LoRA delta to `cond_x` directly — no
  mask trick needed. Target's projections see the frozen weights only.
- Cross-attention is skipped on cond. Target gets text via cross-attn
  as usual; cond doesn't, matching the official's "simple two-stream"
  variant. (For future spatial conditioning where cond should be text-aware,
  the alternative is to route cond-q through the cross-attn alongside
  target-q with a sparse mask. Not done here.)
- `_ExtendedSelfAttnLSEFunc` runs target's attention over `[target_k;
  cond_k]` without materializing the full `(S_t + S_c)²` attention matrix:
  two memory-efficient flash-attention-2 forwards on the disjoint key tiles
  plus a Python LSE-arithmetic combine. Backward is custom (FA2's stock
  backward drops the gradient flowing through `softmax_lse`); see the
  Function's docstring in `networks/methods/easycontrol.py` for the math.
  Falls back to masked-SDPA (math kernel) when flash-attn is unavailable
  with a one-shot warning.
- No deferred backward. cond_x is an explicit checkpoint input and
  cond_x_out an explicit return value of each patched `Block.forward`.
  The autograd chain across blocks survives the per-block unsloth /
  cpu_offload / plain `torch_checkpoint` wrappers naturally —
  `accelerator.backward(loss)` is the only call needed.

## Step-0 baseline equivalence

A step-0 equivalence bench (since removed) settled which
init makes the extended self-attention match the no-cond baseline at step 0:

| Strategy                             | rel_l2 max | mean α  | Verdict     |
| ------------------------------------ | ---------- | ------- | ----------- |
| Zero-init V_c only                   | 0.50       | 0.50    | FAIL        |
| Zero-init both K_c and V_c           | 0.38       | 0.62    | FAIL        |
| Zero-init the cond embedder          | 0.38       | 0.62    | FAIL        |
| `b_cond` = −10 (additive logit)      | 6.4e-5     | 1.0     | EXACT       |
| `b_cond` = −30                       | 9e-17      | 1.0     | bit-exact   |
| Hard mask (cond logits = −∞)         | 0.0        | 1.0     | exact       |

Naive "zero-init V_c" leaves `α = Z_t / (Z_t + Z_c) ≈ 0.5`, which rescales
the target output by ½. The fix is an additive bias on the cond logits inside
the softmax — `b_cond` initialized to −10 makes cond contribute ~e⁻¹⁰ ≈ 4.5e-5
of the total softmax mass, and is freely learnable.

`b_cond` is a per-block scalar `nn.Parameter`. Init −10 is set via
`network_args = ["b_cond_init=-10.0", ...]` in `configs/easycontrol/easycontrol.toml`.

The same (since-removed) bench's Section B verifies the equivalence
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
| Total             |                                | ~14.7 M |

Set `apply_ffn_lora=0` in `network_args` to drop the FFN LoRA — halves the
trainable count.

Opt-in extra: `train_adaln = true` adds a target-stream adaln LoRA
(per-block deltas on the three `adaln_up_{br}` up-projections, 256→3·D;
+~4.3 M at the default `adaln_rank = 8`). It is cond-gated — applied only on
the two-stream / cached-cond-KV paths, so the no-cond fallback stays exact
baseline DiT. A t-only global per-σ modulation prior (e.g. colorize chroma
commitment), not a conditioning pathway. Sizing, rationale, and the base.toml
pin gotcha: `docs/methods/adaln.md` §EasyControl.

## Cond token count: native, no static padding

The cond stream runs at the cond latent's native token count — there is no
static-pad knob. Anima's native-shape bucketing already makes every forward run
at its real token count (one bucket per batch → uniform S_c within a batch), and
the DiT keys its compiled block graph on token count alone (two families: 4032 /
4200). `encode_cond_latent` just flattens the patch-embedded cond latent and
returns it at that count.

For the common ref==target setup the cond is the SAME cached VAE latent the
target uses, so its token count is identical to the target's and lands on one of
the two bucket families automatically. Static-padding to a fixed budget (the
removed `cond_token_count`, default 4096) was both unnecessary under native
bucketing and broken for the 4200-token family (4200 > 4096), and it leaked zero
tokens into the cond stream's self-attention and into target's LSE-extended
attention — the same padding-leak the DiT's static-pad path was removed to avoid.

Memory scales with the cond latent's resolution: a lower-resolution reference
produces fewer cond tokens and a smaller KV cache. To match the official
EasyControl's small (e.g. 32×32) reference, set `cond_res_scale` (below) or
downsample the cond image upstream.

## Position-Aware Interpolation (`cond_res_scale`)

The paper's Position-Aware Training Paradigm (§3.3) downsamples the control
condition to a fixed low resolution (512²) for efficiency, then Position-Aware
Interpolation (PAI) rescales the condition's position encodings back onto the
full-resolution target grid so spatial alignment is preserved. Anima implements
this as the `cond_res_scale` network arg (default `1.0` = off).

What it does. With `cond_res_scale = s` (0 < s < 1), `encode_cond_latent`:
1. downsamples the cond latent to ~`s×` per axis (`F.interpolate`, `area`
   filter, rounded to a patch-size multiple) — cond stream only; the target
   latent, the flow-matching loss, σ sampling, and cross-attn text are all
   untouched and run at full native resolution;
2. computes per-axis rescale factors `S_h = H_target/H_cond`, `S_w = W_target/W_cond`
   from the pre/post patch grids (the target grid = cond's full-res grid, since
   cond is the same spatial size as the target for both ref==target and
   colorize — so no caller plumbing is needed);
3. builds the cond RoPE at fractional positions `i·S_h` via
   `VideoRopePosition3DEmb.generate_embeddings_scaled` instead of the integer
   `seq[:H]`. Anima's RoPE is position-value-agnostic (frequencies are analytic,
   not table-indexed — the same path already feeds fractional temporal positions
   under FPS modulation), so fractional positions are exact.

Why it's cheap. The cond stream's own self-attention is `O(S_c²)`, so
quartering the token count (s ≈ 0.5) cuts it ~16×; the cond LoRA/MLP and the
`S_t×S_c` cond tile of target's extended attention drop ~4×. Realistic
end-to-end training speedup ≈ 25-30%, plus a smaller activation/KV footprint
(the [memory envelope](#memory-envelope) full→low-res row). The inference KV
cache shrinks by `1/s²` and inherits the scaled positions automatically — no
inference-side change.

When to use it.
- Subject / semantic control (identity, object): low-res cond is essentially
  free — the paper shows fidelity survives downscaling. Good `s` ≈ 0.5.
- Structure-critical tasks (colorize lineart, edges): downscaling discards
  high-frequency structure the cond is supposed to carry, and PAI realigns
  *positions* but cannot restore lost *information*. Keep `cond_res_scale = 1.0`
  (the shipped colorize default) or use a mild `s` ≥ 0.7 only if a bench shows
  boundary fidelity holds.

Compile. The cond stream carries its own dynamic-seq band. `compile_cond_stream`
derives it from the target's token band scaled by `s²` (`_cond_seq_range`), because
marking the cond seq axis with the target band raises
`ConstraintViolationError: <cond_tokens> not in range [lo, hi]` on the first step of
any `s < 1` run. On cond≠target subsets the cond image can also free-fit to its own
shape (even another tier), so the band self-widens at runtime with a one-line log and
a single dynamo recompile (`_fit_cond_seq_range`) rather than crashing.

Equivalence. At `cond_res_scale = 1.0` the downscale block is skipped and
the cond RoPE is the native-position table — bit-exact to the pre-PAI path, so
existing checkpoints are unaffected. The `b_cond` step-0 baseline equivalence is
undisturbed (PAI only moves cond positions, not the gate). Verified by a
PAI-equivalence bench (since removed).

## Variants

EasyControl is a family of control tasks: they all share the shipped network
(`networks/methods/easycontrol.py`) and the two-stream forward above, and differ
only in how the condition is built. A variant is selected by the
`EASYADAPTER` env var (unset → the default ref==target EasyControl); per-task
projects live under `easycontrol_adapters/`.

| Variant      | `EASYADAPTER` | Condition                                   | Config                                | Project                              |
| ------------ | ------------- | ------------------------------------------- | ------------------------------------- | ------------------------------------ |
| default      | (unset)       | the reference image itself (cond == target) | `configs/easycontrol/easycontrol.toml` | —                                    |
| colorize     | `colorize`    | synthetic mangafied B&W (XDoG + screentone) | `configs/easycontrol/colorize.toml`†  | `easycontrol_adapters/colorization/` |

† colorize is a self-contained descriptor (top-level `name` + `[staging]` /
`[preprocess]` / `[training]` knob tables + a `[general]`/`[[datasets]]` blueprint,
same shape as `near_twins.toml`). It trains the base `easycontrol` method with its
`[training]` table folded in as CLI overrides — there's no standalone `colorize`
method/dataset file anymore.

### colorize — manga / lineart → color

Trains the same extended-self-attention cond stream to colorize a B&W
screentoned page. Real manga has no color ground truth, so the pair is inverted:
the target is an existing color illustration (latents + captions reused from
the shared `post_image_dataset/lora/` cache — nothing re-encoded) and the
condition is a synthetic mangafied version of that same image (XDoG lineart
+ value-banded algorithmic screentone), cached to a parallel `cond_cache_dir`
(`post_image_dataset/easycontrol/colorize/cond/`). The text channel is reduced to
color-only captions (hair/eye/garment colors) in its own `text_cache_dir`
(`post_image_dataset/easycontrol/colorize/text/`), so the prompt carries the one variable
B&W can't — hue — giving a strong prompt→color binding. At inference an empty
prompt auto-colorizes; a color prompt (`pink hair, blue eyes`) steers.

```bash
make easycontrol-staging    EASYADAPTER=colorize   # stage 1: mangafy cond tree
make easycontrol-preprocess EASYADAPTER=colorize   # stages 2-3: VAE-encode cond + color text
make easycontrol            EASYADAPTER=colorize   # train (frozen DiT, adapter-only)
REF_IMAGE=page.png make test-easycontrol EASYADAPTER=colorize   # inference
```

In the GUI, the EasyControl experimental tab lists Colorize in its Variant
dropdown as a descriptor variant: a file-edited launcher (no in-GUI form — the
tab shows a pointer note to edit `configs/easycontrol/colorize.toml` directly, like
near_twins). Its Preprocess button runs staging + preprocess in one shot
(`--no-skip_mangafy`); its Train button trains the base `easycontrol` method with
the descriptor's blueprint + `[training]` overrides folded in (via the daemon, so
it survives the GUI closing).

Full design notes (caption policy, screentone bands, inference settings, Phase B)
live in `easycontrol_adapters/colorization/README.md`.

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
  Patched `Block.forward` then falls through to the original baseline DiT
  behavior. Lets inference do image-CFG independently of text-CFG.

REPA auxiliary loss (optional, `network_args = ["use_repa=true", ...]`) —
validated and shipped as the default for cond ≠ target tasks (sanitize /
near-twins, colorize):
- Relational (Gram) alignment of mid-block target-stream hiddens to cached
  PE-Spatial patch tokens of the clean target image — same machinery as the
  LoRA family's REPA v2 (`library/training/repa.py`; knobs `repa_weight` /
  `repa_layer` / `repa_encoder`, relational mode only). Because the DiT is
  frozen, the alignment gradient reaches the cond LoRA solely through the
  extended self-attention in blocks ≤ `repa_layer`, so the term acts as a
  conditioning-utilization pressure: the only way to satisfy it is to pull
  clean spatial structure from the reference. Wired for sanitize (near_twins),
  where the structural-consistency signal lands exactly on the edit region
  (see `configs/easycontrol/near_twins.toml`), and for colorize (region
  coherence pulled from the manga cond, see `configs/easycontrol/colorize.toml`).
  For ref==target subject control it would instead reward layout copying — use
  with care. Needs `{stem}_anima_pe_spatial.safetensors` sidecars (near_twins:
  `[preprocess] pe_encoder = "pe_spatial"` → re-run
  `make easycontrol-preprocess EASYADAPTER=near_twin`); the loader walks a
  fallback chain (`_repa_pe_sidecar_candidates`) so the colorize
  `text_cache_dir` redirect still finds the sidecars in the shared latent cache.
  On cond-dropped steps the term has no trainable path (frozen target stream) —
  keep `drop_p` low or zero when using it.
- **Launch sanity (load-bearing)**: confirm `repa/align_loss` appears in the
  progress jsonl from the first logged step, alongside `repa/active = 1.0`.
  `active=1.0` *without* `align_loss` is the silent-baseline failure signature
  (the pre-2026-06-12 dispatch bug, since fixed — the aux-loss `extra_forwards`
  dispatch ran only on the cached-crossattn branch, which EasyControl's
  in-model text path doesn't use). The operating-point validation is closed;
  the archived roadmap is `_archive/proposals/easycontrol_repa_operating_point.md`.

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

## Memory envelope

Measured peak GPU memory in the smoke bench (since removed;
gradient checkpointing on, target latent 64×64, batch 1, bf16):

| Configuration                            | Peak GPU memory |
| ---------------------------------------- | --------------: |
| Baseline DiT only (no cond)              | ~5.0 GiB        |
| Two-stream, low-res cond (~1024 tokens)  | ~5.4 GiB        |
| Two-stream, full-res cond (~4096 tokens) | ~6.3 GiB        |

A real training step on 16 GiB GPUs (live observed) lands around 7.8 GiB
for a full-resolution (ref==target) cond at constant-bucket S_c. The Phase 1.5
design pinned ~1.4 GiB more on top of this and did not fit on 16 GiB.
Cond memory now scales with the reference's native resolution — there is no
fixed token budget.

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

Lifecycle. `_setup_easycontrol` in `library/inference/generation.py`
calls:

```python
network.set_cond(cond_latent)        # encode reference, stage cond_x_in for block 0
network.precompute_cond_kv()         # walk cond stream once, fill _cond_kv_cache
```

After this, `EasyControlNetwork._cond_kv_cache` holds a
`list[(K_c_i, V_c_i)]` of length `num_blocks`. Each entry is a BSHD pair
`[B, S_c, n_heads, head_dim]` — the same layout `_extended_target_attention`
expects, post-`q_norm/k_norm/v_norm`, post-`apply_rotary_pos_emb_qk`.

Patched Block.forward dispatch. Three paths in priority order:

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

Memory. At a full-resolution `S_c = 4096`, `n_heads = 16`, `head_dim = 128`,
`num_blocks = 28`, bf16, batch 1:

```
2 (K + V) × 28 blocks × 4096 × 16 × 128 × 2 bytes ≈ 896 MiB
```

A lower-resolution reference scales the cache linearly with its native token
count (e.g. ~448 MiB at ~2048 cond tokens). The startup log reports the actual
size:

```
EasyControl: precomputed cond KV cache (28 blocks × 2 tensors, ~939 MB)
```

Speedup envelope. Per denoising step the cache eliminates, per block:
cond AdaLN, cond LayerNorm + `qkv_proj` + cond LoRA (qkv), the cond stream's
own `S_c × S_c` SDPA, cond `output_proj` + cond LoRA (o), cond MLP +
cond LoRA (ffn1/ffn2), and the cond residual writes. Target-side cost
collapses to `_extended_target_attention` (LSE-decomposed flash) + baseline
cross-attn + baseline MLP. Practical end-to-end speedup vs the no-cache path
scales with `S_c / S_t` and the FFN LoRA ratio; expect a meaningful drop in
per-step wall time for a full-resolution cond.

Correctness. The cache stores the exact tensors the two-stream path
would have produced (same modules, same scale, same RoPE). Setting
`network.clear_cond_kv_cache()` and re-running falls back to the two-stream
path bit-exactly.

Cache invalidation. `set_cond(new_latent)` clears the cache (stale until
`precompute_cond_kv` runs again). `set_cond(None)` / `clear_cond` /
`remove_from` also clear it. If you mutate `multiplier` or `cond_scale`
manually after caching, call `clear_cond_kv_cache()` and re-prime — the
cached K/V bake the effective scale at prime time.

Custom node use. ComfyUI's custom node should call the same two-line
sequence (`set_cond` then `precompute_cond_kv`) once per `(reference,
cond_scale)` change; subsequent KSampler steps use the cache automatically.

## Limitations

1. Cond runs at the reference's native resolution by default. Set
   `cond_res_scale < 1.0` to downsample the cond stream in latent space for
   faster/cheaper training (target + loss stay full-res) — see
   [Position-Aware Interpolation](#position-aware-interpolation-cond_res_scale)
   below. At the default `1.0` there is no token-count cap.
2. Spatial-control positional alignment is the PAI downscale, not arbitrary
   remapping. With `cond_res_scale < 1.0` the cond's RoPE positions are
   interpolated back onto the target grid (paper §3.3 PAI, `Pᵢ = i·S_h`) so a
   downsampled cond stays pixel-aligned with the target. This covers the
   common case where cond and target share content/coordinates (ref==target,
   colorize). Remapping a cond drawn in a different coordinate system onto
   the target (the official's full spatial-control story for cropped/offset
   conditions) uses the same `generate_embeddings_scaled` machinery but isn't
   wired through a per-condition offset/crop API here.
3. `blocks_to_swap = 0` recommended. The patched `Block.forward` does
   the cond compute inside the block's forward window, so block swap is
   structurally fine — but untested with EasyControl. Pinning to 0 for now;
   bf16 frozen DiT + cond LoRA fits without swapping anyway.
4. Custom autograd Function inside `_ExtendedSelfAttnLSEFunc`. The
   joint-softmax backward is implemented manually because FA2's stock
   backward drops the upstream gradient on `softmax_lse`. Verified against
   masked-SDPA reference within fp32 ulp on forward and all gradients
   (via an LSE-equivalence bench, since removed). Falls back to
   masked-SDPA when flash-attn is unavailable.

## History

This file used to describe a Phase 1.5 design where cond ran a separate
pre-pass across all blocks before the target forward, caching per-block
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
| `configs/easycontrol/easycontrol.toml`              | Method config (default ref==target)                    |
| `configs/gui-methods/easycontrol.toml`          | GUI-friendly self-contained variant                    |
| `configs/easycontrol/colorize.toml`             | Colorize descriptor — `name` + `[staging]`/`[preprocess]`/`[training]` tables + blueprint + `[variant]` GUI metadata (the single source of truth; folds onto the base easycontrol method) |
| `configs/easycontrol/near_twins.toml`           | Near-twins descriptor (same shape; text-removal control task)          |
| `easycontrol_adapters/colorization/`            | Colorize project — mangafy + `prep.py` + color-caption filter + README |
