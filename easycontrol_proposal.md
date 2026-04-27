# EasyControl for Anima — proposal

**Status:** draft / RFC
**Author:** —
**Last updated:** 2026-04-26

A proposal to add **EasyControl-style image conditioning** as a sibling to the existing `ip_adapter` method. EasyControl (Zhang et al., 2025) is a DiT-native image-conditioning recipe that has eclipsed IP-Adapter on FLUX-class models for spatial-aware reference, multi-condition stacking, and inference cost.

---

## TL;DR

- **Goal:** train an image-conditioning adapter on `(image, text)` pairs (same setup as `ip_adapter`) that preserves spatial detail / identity meaningfully better, while being **cheaper at inference** because the reference's KV is precomputed once and reused across denoising steps.
- **Mechanism:** encode the reference with the **VAE** (not CLIP/PE) into latent tokens, route them through the DiT's existing `self_attn` blocks via a **condition-only LoRA branch** with a **causal mask** so condition tokens never attend to noisy-target tokens. Target tokens get extended self-attention keys: `K = [K_target; K_cond]`, `V = [V_target; V_cond]`.
- **Why we want it:** VAE encoding preserves layout / identity natively (the IP-Adapter doc already calls out CLIP/PE features as "global semantics, weak on layout"); KV caching turns the per-step adapter cost into a near-free key extension; the condition-LoRA branch composes cleanly with the existing LoRA + OrthoLoRA + T-LoRA + ReFT stack.
- **Architectural mapping:** EasyControl was designed for FLUX's MMDiT joint attention. Anima has **separate** `self_attn` and `cross_attn` per block (`library/anima/models.py:983`). Condition tokens go through `self_attn` only (where target tokens already self-attend); `cross_attn` is reserved for text and stays untouched. Condition tokens also pass through the FFN. `S_c = 4096` (same bucketing as target).
- **Cost:** ~similar trainable param count to current IP-Adapter (LoRA branch ≈ resampler + 28 KV projections), but ~no per-step adapter compute at inference after the first step.

---

## Background

### What EasyControl actually does

```
                            target tokens                        condition tokens
                           [B, S_t, D]                          [B, S_c, D]
                                │                                     │
   block i           ┌──────────┴──────────┐               ┌─────────┴─────────┐
                     │  self_attn (LoRA)   │               │  self_attn (LoRA) │
                     └──────────┬──────────┘               └─────────┬─────────┘
                          target Q/K/V                          cond Q/K/V
                                │                                     │
                                └─────────┐         ┌────────────────┘
                                          │         │
                                          ▼         ▼
                                    extended KV: [K_t; K_c], [V_t; V_c]
                                          │
                          ┌───────────────┴───────────────┐
                          │ target_Q  attends to  [K_t; K_c]   ← target sees both
                          │ cond_Q    attends to  [K_c]        ← causal mask: cond never sees target
                          └───────────────────────────────┘
                                          │
                                  back to target_x, cond_x       (independent residual streams)
```

Two concrete consequences of the causal mask:
1. **`cond_x` at block i+1 depends only on `cond_x` at block i.** No coupling to the noisy target. So for a fixed reference image, the condition path is **deterministic across all denoising steps**.
2. → **KV-cache.** Run the condition path once. Save per-block `(K_c_i, V_c_i)`. Reuse for every step. The target path then pays only the cost of an extended-key SDPA, where the extra keys are read from cache.

A "Condition Injection LoRA" + "Position-Aware Training" round it out:
- Condition LoRA — ranks-r LoRA on each `self_attn` block's q/k/v/o (or just k/v) projections, applied **only when the input is condition tokens**. Enables stacking multiple condition types (e.g. canny + reference) with separate LoRA branches.
- Position-aware training — mix multiple resolutions during training so the model handles arbitrary aspect ratios. (Anima already buckets to constant-token; this is mostly free.)

---

## Why Anima wants this

We already have `networks/ip_adapter_anima.py` (decoupled cross-attention with PE-Core features, K=16 resampler tokens, per-block `to_k_ip` / `to_v_ip`). It works, but has known structural limitations called out in `docs/methods/ip-adapter.md`:

| Issue with current IP-Adapter | EasyControl-style fix |
|---|---|
| PE-Core features are global/semantic — weak on **spatial layout** and fine identity detail. The doc explicitly notes "CLIP features are semantic/global ... struggles with exact spatial layout, exact pose." | VAE-encoded condition is **in the same latent space the DiT already lives in** — full spatial structure preserved. |
| Resampler bottleneck (K=16 tokens) is a hard upper bound on detail you can carry. | Condition tokens = full bucketed VAE latent grid, ~hundreds–thousands of tokens. |
| Adapter pays per-step cross-attn compute. | Condition KV cached → ~free per-step. |
| `ip_scale` is a single global knob; one adapter trained for one role. | Multi-condition by design — multiple condition-LoRA branches stack. |
| No clean path to spatial-aligned controls (canny, depth) — that would need a separate ControlNet. | Spatial-aligned and reference-driven both fit the same architecture. |

These are exactly the things the IP-Adapter doc flags as "v1 limitations / future work."

---

## Mapping EasyControl onto Anima's architecture

EasyControl was published for FLUX / SD3, both **MMDiT** — text and target tokens are concatenated through one **joint** self-attention, and adding condition tokens means concatenating once more into that joint stream. Anima is **not MMDiT.** Each `Block` (`library/anima/models.py:983`) has:

- `self_attn` — image tokens self-attend (target only).
- `cross_attn` — image tokens (Q) attend to text tokens (K/V). Text is never in the same attention pass as image.

**Design (pinned):** condition tokens flow through every block's `self_attn` and FFN with a condition-only LoRA active; `cross_attn` is skipped (text is irrelevant to the reference's evolution). Per block:

- Target's `self_attn`: `Q_t @ [K_t; K_c]^T → softmax → @ [V_t; V_c]`
- Condition's `self_attn`: `Q_c @ K_c^T → softmax → @ V_c` (causal — does not see target)
- Condition tokens **skip `cross_attn` entirely**.
- Condition tokens **pass through FFN** with the condition-LoRA active.

This works because Anima's image tokens already only meet other image tokens inside `self_attn`, so adding condition tokens to that pass is a clean extension of "image tokens attending to image tokens." `cross_attn` stays untouched → no regression risk on text fidelity. The KV cache lives on each block's `self_attn` instance — same monkey-patch shape as the existing `ip_adapter` patch, just on a different submodule.

**Why not concat into `cross_attn` instead?** That would re-create the "coupled K/V" failure mode `docs/methods/ip-adapter.md` already calls out: condition and text would share a softmax denominator, and the condition path would steal attention mass from text. Worse here than for postfix because the condition has thousands of tokens, not a handful.

**Why FFN included.** Without FFN, condition tokens compute static attention features that depend only on shallow projections; the depth of the DiT doesn't transform them. With FFN, the condition path is genuinely a DiT forward with `cross_attn` skipped — full representational capacity. Doubles trainable param count vs. self-attn-only but matches the published EasyControl recipe.

---

## Concrete design

### Network module: `networks/easycontrol_anima.py`

Mirrors the structure of `networks/ip_adapter_anima.py` but with different hooks:

```python
class EasyControlNetwork(nn.Module):
    def __init__(self, *, num_blocks, hidden_size, num_heads,
                 cond_lora_dim, cond_lora_alpha, cond_token_count, ...):
        # Per-block LoRA on self_attn q/k/v/o (and optionally FFN).
        self.cond_lora_q = nn.ModuleList([LoRALinear(...) for _ in range(num_blocks)])
        self.cond_lora_k = nn.ModuleList([LoRALinear(...) for _ in range(num_blocks)])
        self.cond_lora_v = nn.ModuleList([LoRALinear(...) for _ in range(num_blocks)])
        self.cond_lora_o = nn.ModuleList([LoRALinear(...) for _ in range(num_blocks)])
        # Optional: a small projection from VAE latent channels into the DiT hidden
        # if you want a learnable condition embedder. Often unnecessary because the
        # patch-embedder of the DiT already does this — we can reuse it.

    def apply_to(self, unet):
        # Monkey-patch each Block.self_attn.forward — just like ip_adapter does
        # for cross_attn. Closure captures (orig_attn, network, block_idx).
        ...

    def encode_condition(self, vae_latents):
        # Patchify the VAE latent (reusing DiT's patch_embedder) into [B, S_c, D].
        # This produces M = constant condition-token count for the bucket.
        ...

    def precompute_cond_path(self, cond_tokens):
        # Run the condition tokens through each Block (self_attn with condition-
        # LoRA active, NO cross_attn, optional FFN). At each block, save
        # (K_c_i, V_c_i) on self_attn._cond_k_cached / _cond_v_cached.
        # This is the KV-cache step. Called once at inference per reference;
        # called every step at training.
        ...
```

### Patched `self_attn.forward`

Mirror the existing `ip_adapter` patch (`networks/ip_adapter_anima.py:530`):

```python
def patched_self_attn_forward(x, attn_params, ...):
    q, k, v = orig_attn.compute_qkv(x, ...)             # target Q/K/V
    cond_k = getattr(orig_attn, "_cond_k_cached", None)
    cond_v = getattr(orig_attn, "_cond_v_cached", None)
    if cond_k is not None:
        k_ext = torch.cat([k, cond_k], dim=1)           # [B, S_t + S_c, H, D]
        v_ext = torch.cat([v, cond_v], dim=1)
    else:
        k_ext, v_ext = k, v
    out = anima_attention.attention([q, k_ext, v_ext], attn_params=attn_params)
    return orig_attn.output_dropout(orig_attn.output_proj(out))
```

### Condition-pass forward

Run condition tokens through every block, **bypassing `cross_attn`**, with the condition-LoRA active on `self_attn`'s linear ops. Two ways to implement:

1. **Mode flag on the LoRA** (cleaner): the condition-LoRA's `forward` checks a thread-local / module-flag "is this condition input?" and only contributes when true. Then we run `block.self_attn(cond_tokens, ...)` directly, skip `cross_attn`, optionally `block.ffn(cond_tokens)`.
2. **Separate condition tower** (heavier): copy `self_attn` weights, attach LoRA, run a sibling stack. More memory, simpler logic.

We go with **(1)** — it's the EasyControl recipe and reuses the DiT weights at no memory cost.

### Causal mask — for free

When condition tokens go through their own forward pass, they only see `K_c`. There's no actual mask — they just don't have access to `K_t` because we never concatenate it for them. The "causal" in the paper is a presentation choice; the implementation is two separate forwards that share the LoRA but not the attention.

### KV-cache lifecycle

- **Training:** No cache. Each step recomputes the condition path under the same LoRA. Backward flows through both the target path and the condition path via the shared LoRA params.
- **Inference:** `precompute_cond_path()` runs once at `generate()` setup. Each block's `self_attn._cond_k_cached / _cond_v_cached` holds `[1, S_c, n_h, d_h]` (broadcast `B=1→N` for CFG, mirror of the existing `ip_adapter` trick).

### Trainable params

Roughly:
- Per-block LoRA on `self_attn` q/k/v/o (rank r) → `4 · 28 · 2 · D · r · bytes` ≈ ~50–100 M params at r=16, D=2048.
- Per-block LoRA on FFN (in_proj + out_proj at rank r) → ~doubles that.
- Per-block scalar `b_cond` logit bias → 28 params, negligible.

Total at r=16: ~120–200M trainable, comparable to the current IP-Adapter (~150M).

### Init for step-0 baseline equivalence

**The intuitive answer is wrong; the bench (`bench/easycontrol/step0_equivalence.py`) settles it.**

Naive intuition: "zero-init V_c, then the cond contribution is zero, so output ≈ baseline." This is **incorrect**. The decomposition is:

```
out_extended = α · out_baseline  +  softmax_c @ V_c
   where α = Z_t / (Z_t + Z_c),
         Z_t = Σ exp(L_t),  Z_c = Σ exp(L_c)
```

V_c = 0 zeroes the *second* term, but the *first* term is rescaled by α. Equivalence holds iff α = 1, which requires Z_c → 0, which requires the cond logits to be very negative (or masked).

The bench confirms this empirically (S_t = S_c = 4096, post-RMSNorm unit-magnitude K, fp32):

| Strategy | rel_l2 max | mean α | Verdict |
|---|---|---|---|
| Zero-init V_c only (K_c at standard init) | **0.50** | 0.50 | **FAIL** — output magnitude halved |
| Zero-init both K_c and V_c | 0.38 | 0.62 | FAIL — still huge |
| Zero-init the cond embedder (cond_tokens = 0) | 0.38 | 0.62 | FAIL — equivalent to zero K_c, V_c |
| Logit bias on cond keys, init = −10 | 6.4e-5 | 1.0000 | exact (visually identical) |
| Logit bias on cond keys, init = −30 | 9e-17 | 1.0000 | bit-exact |
| Hard mask cond positions | 0.0 | 1.0000 | exact (sanity check) |

So the actual recipe for step-0 baseline equivalence is:

**Add a learnable additive bias `b_cond` on the cond logits, initialized to ~−10 (or larger negative for stricter equivalence).** The bias gates softmax mass on cond positions:

```python
logits = Q @ [K_t; K_c]^T / √d
logits[..., S_t:] += b_cond     # b_cond starts at -10, freely learnable
```

At step 0, `exp(-10) ≈ 4.5e-5`, the cond positions hold negligible softmax mass, and α ≈ 1.0000. As training proceeds, `b_cond` is free to rise toward 0 (or above) as the model learns to use the condition. This is structurally similar to ControlNet's zero-convolutions, but applied as a **logit gate** rather than a value gate — which avoids the degenerate-gate convergence the IP-Adapter doc warns about (the gate isn't multiplicative on the contribution; it's an additive shift inside the softmax that lets the model *choose* to attend or not).

Implementation: a single per-block scalar (or per-head scalar) `nn.Parameter`, initialized to −10. Almost-zero param overhead, no architectural complexity.

(Why ControlNet-style zero-init doesn't work here: ControlNet's zero-conv zeroes a feature map that's then *added* to a residual stream. Adding zero is identity. Here we're inside a softmax — adding zero-valued K and V still leaks softmax mass because exp(0) ≠ 0.)

---

## Plumbing — concrete file plan

| Action | File | Why |
|---|---|---|
| **Add** | `networks/easycontrol_anima.py` | New `EasyControlNetwork`. Mirror the structure of `networks/ip_adapter_anima.py`. |
| **Add** | `configs/methods/easycontrol.toml` | New method config. Mirror `configs/methods/ip_adapter.toml`. |
| **Add** | `configs/gui-methods/easycontrol.toml` | GUI-friendly self-contained variant. |
| **Add** | `docs/methods/easycontrol.md` | Method doc — sibling of `ip-adapter.md`. |
| **Add** | `preprocess/cache_easycontrol_features.py` | Cache VAE-encoded reference latents as `{stem}_anima_easycontrol.safetensors` sidecars. (`make easycontrol-cache`.) Mirror `preprocess/cache_pe_encoder.py`. Can probably reuse VAE latent cache directly when ref==target, with a soft symlink — see "Open questions." |
| **Modify** | `library/anima/training.py` | Argparse: `--use_easycontrol`, `--easycontrol_drop_p`, `--easycontrol_features_cache_to_disk`. Mirror existing IP-Adapter flags. |
| **Modify** | `train.py` | Add `_maybe_set_easycontrol_tokens` analogous to `_maybe_set_ip_tokens`. Same CFG-dropout pattern. |
| **Modify** | `library/inference/generation.py` | Add `_setup_easycontrol` analogous to `_setup_ip_adapter`. |
| **Modify** | `Makefile` + `tasks.py` | `make easycontrol`, `make test-easycontrol`, `make easycontrol-cache`. |
| **Modify (later)** | `custom_nodes/comfyui-hydralora/` | Auto-detect EasyControl weights and apply (KV-cache path is friendlier for ComfyUI than IP-Adapter's per-step cross-attn). |
| **Untouched** | `networks/ip_adapter_anima.py` and family | Both methods coexist. EasyControl is a sibling, not a replacement. |

---

## Compatibility with Anima invariants

These are the project invariants from `CLAUDE.md` and where EasyControl lands on each:

| Invariant | EasyControl impact |
|---|---|
| **Constant-token bucketing (`(H/16)*(W/16) ~ 4096`).** | Target side stays at 4096. Condition uses the **same bucketing**, so `S_c = 4096`. Self-attn key length doubles to 8192. Single static shape per bucket; compile holds. |
| **Flash4 LSE correction (`networks/attention.py`).** | The LSE correction was for **trimmed text K/V padding**. Self-attn never had padding. Adding cond tokens to self-attn does **not** trigger LSE correction — they're real keys, not zero-padding. No change needed. |
| **DDP all_reduce on LoRA only.** | Condition-LoRA params register the same way LoRA does. `all_reduce_network()` covers them automatically. |
| **Lazy DiT loading.** | Condition path uses VAE — already in the early "load VAE → cache → free" phase. Cache `easycontrol_features` alongside VAE latents during the same pass. No new model in memory at training time once features are cached. |
| **`torch.compile` (full mode).** | Patched `self_attn.forward` is regular Python. SDPA inlines. With `S_c = 4096` pinned, the extended-key sequence length (8192) is a static shape. We need to verify the attention dispatch backends (`networks/attention.py`) handle non-square Q/K (Q=4096, K=8192). torch SDPA does. Flash-attn 2/3 do. xformers does. |
| **Block swapping.** | Condition path traverses every block; needs to interleave correctly with the swap-on / swap-off lifecycle. Likely solvable but **flagged as a known integration risk** — start with `blocks_to_swap=0` for v1. |
| **LoRA stack (LoRA + OrthoLoRA + T-LoRA + ReFT).** | Condition-LoRA targets `self_attn` projections; existing LoRAs may also target the same projections. They compose additively in the LoRA `up @ down` sense. Provided we **only activate the condition-LoRA on the condition input pass**, there's no interference with the standard LoRA on the target pass. (The mode-flag mechanism is what makes this clean.) |

---

## Phased implementation

### Phase 0 — design lock (this doc)
Pinned: extend `self_attn` (not `cross_attn`); `S_c = 4096` (same bucketing as target); condition path runs FFN. Step-0 baseline equivalence via per-block learnable logit bias `b_cond`, init = −10 (verified by `bench/easycontrol/step0_equivalence.py`).

### Phase 1 — minimal training path
- `networks/easycontrol_anima.py` with self-attn LoRA + FFN LoRA + per-block `b_cond` logit bias.
- `S_c = 4096`, ref==target (same as IP-Adapter v1's training setup).
- No KV-cache path. Recompute condition every step. Train and verify it learns.
- `configs/methods/easycontrol.toml`, `make easycontrol`, basic train loop integration.
- **Success criterion:** runs without OOM, loss decreases, sample reproductions show meaningful identity/layout transfer beyond what the current IP-Adapter does on the same data.

### Phase 1.5 — LSE-decomposed extended self-attn (memory cliff fix)

**Why this slot exists.** Phase 1 hit OOM on real hardware even with `--gradient_checkpointing` enabled, because grad-ckpt reduces *inter-op* memory but the extended-key self-attn allocates a peak *intra-op* tensor that grad-ckpt can't see. Until Phase 1.5 lands, Phase 1 is not actually trainable end-to-end on a single GPU at the published bucket.

**Where the cliff is.** The patched `self_attn.forward` adds a per-block scalar additive bias `b_cond[i]` on the cond positions. None of the alt attention backends (flash-attn v2/v3, xformers, sageattn, flex) accept a free-form additive bias on a key subset, so Phase 1 forces torch SDPA with an explicit `attn_mask`. With a non-`None` mask, torch SDPA falls off the flash and mem-efficient backends and dispatches to the **math kernel**, which materializes the full attention matrix:

```
attn_matrix shape : [B, n_h, S_t, S_t + S_c] = [B, 16, 4096, 8192]
≈ 1 GB / block / forward at bf16  (often 2 GB — math kernel computes in fp32 internally)
```

That peak repeats per block. grad-ckpt does not save it (it's allocated and freed inside one op). Plain LoRA training never sees this cliff because target self-attn stays at `[16, 4096, 4096]` and the flash kernel handles it in O(N) memory.

**The fix — LSE decomposition.** Because `b_cond[i]` is a **scalar** bias applied uniformly to all cond keys, the extended-key SDPA decomposes exactly into two ordinary memory-efficient SDPAs plus a small LSE-arithmetic combine. Per block:

```python
# Two ordinary flash/mem-efficient SDPAs — neither needs a mask.
out_t, lse_t = flash_attn(Q, K_t, V_t, return_lse=True)   # target self-attn
out_c, lse_c = flash_attn(Q, K_c, V_c, return_lse=True)   # cond as cross-attn

# Scalar bias on cond logits ↔ scalar shift of cond LSE:
#   exp(L_c + b) = exp(b) · exp(L_c)
#   ⇒ logsumexp(L_c + b) = b + logsumexp(L_c)
lse_c_adj = lse_c + b_cond[i]

# Combine via the standard logsumexp identity:
#   softmax over [L_t ; L_c+b] = α · softmax(L_t) + β · softmax(L_c+b)
# where α = exp(lse_t - total_lse), β = exp(lse_c_adj - total_lse).
total_lse = torch.logaddexp(lse_t, lse_c_adj)
alpha = (lse_t     - total_lse).exp().unsqueeze(-1)   # [B, n_h, S_t, 1]
beta  = (lse_c_adj - total_lse).exp().unsqueeze(-1)
out = alpha * out_t + beta * out_c
```

This is mathematically identical to the current path (each leg of the softmax is computed once, then renormalized against the joint partition function). It's the same trick used by ring-attention / sequence-parallel attention to combine tile-local attentions into a global one.

**Memory impact.**

| Path | peak attn-matrix shape | bytes per block (B=1, bf16) |
|---|---|---|
| Phase 1 (math kernel + mask)        | `[16, 4096, 8192]` | ~1 GB (≈2 GB at fp32 internal) |
| Phase 1.5 (two flash + LSE combine) | none materialized; intermediates O(S_t · n_h · d_h) | ~tens of MB |

Recovers ~1–2 GB peak per block, restores compatibility with the project's `attn_mode = "flash"` default, and unblocks `blocks_to_swap > 0` for cond pre-pass once the swap-interleave story is resolved.

**Plumbing.**
- `networks/attention.py` — extend the dispatch to surface `lse` from the flash backend (FA2 already returns LSE alongside out via `flash_attn_func(..., return_attn_probs=True)` or the underlying CUDA call). Either add a new entry-point `attention_with_lse(qkv, attn_params)` or have `attention()` return `(out, lse)` when an `lse=True` flag is passed. Keep the existing single-tensor return as the default.
- `networks/easycontrol_anima.py:_make_patched_self_attn_forward` — replace the "force torch SDPA + attn_mask" branch with the two-call + LSE-combine recipe above. Drop the math-kernel path entirely; if `lse` is unavailable (e.g. xformers backend), fall back to the current torch-SDPA-with-mask path with a one-line warning.
- `bench/easycontrol/step1p5_lse_equivalence.py` — bench that the new path matches the math-kernel path bit-for-bit (modulo flash-attn's expected fp32-accumulation ulp differences). Run on the same shapes as `step0_equivalence.py`.

**Success criterion.** Same loss curve as Phase 1's math-kernel path within fp32 ulp tolerance, peak VRAM during the extended self-attn drops to within ~10% of plain-LoRA's self-attn peak, and `make easycontrol` runs end-to-end on the standard hardware preset without OOM.

**Pre-req for Phase 2.** The KV-cache work in Phase 2 should land *on top of* Phase 1.5 — there's no point caching K_c/V_c if the per-step extended-attention itself can't fit.

### Phase 2 — KV cache at inference
- `precompute_cond_path()` once at `generate()` setup.
- Verify bit-for-bit equivalence vs Phase 1's recomputed path.
- Measure speedup. Expect ~2–5× faster than current IP-Adapter inference.

### Phase 3 — knobs and ergonomics
- `--easycontrol_scale`, CFG dropout, `easycontrol_image_match_size`.
- Diagnostic accumulator (per-block ratio of cond-attention contribution to target-attention contribution; mirror IP-Adapter's `[IP-Adapter diag]` log).
- Sample script analogous to `make test-ip`.

### Phase 4 — multi-condition
- Allow stacking multiple condition-LoRA branches (e.g. canny + reference) — separate sidecar caches, separate LoRA modules, separate cond-token streams.

### Phase 5 — ComfyUI custom node
- Extend `custom_nodes/comfyui-hydralora` to detect and apply EasyControl weights.

---

## Risks and open questions

1. **Step-0 baseline equivalence with extended self-attention keys.** ✅ **Resolved by `bench/easycontrol/step0_equivalence.py`.** Initial intuition ("zero-init V_c is enough") was wrong — the bench shows ~50% rel_l2 deviation because softmax mass leaks to cond positions and rescales the target output by the leaked-mass factor α. **Fix: a learnable additive logit bias on cond keys, initialized to −10**, which gives rel_l2 ≈ 6.4e-5 (visually identical to baseline) while remaining freely trainable. See the bench script and the "Init for step-0 baseline equivalence" section above for the full decomposition and table.
2. **Block swapping interleave.** Ordering `swap → forward target → swap → forward cond → swap` is awkward. v1 sets `blocks_to_swap=0` (same as current IP-Adapter and APEX). Worth investigating later but not for the first release.
3. **Cache invalidation when ref==target during training.** If we cache `easycontrol_features` per image and ref==target, this is just the VAE latent already cached for the target. We should reuse the existing VAE cache directly (a thin "give me the bucketed latent for image X at the condition's bucket" lookup) rather than write a duplicate sidecar. Saves disk space and avoids two-source-of-truth. *Open: do we need a separate sidecar at all in the ref==target setup?*
4. **Caption dropout pitfall (carried over from IP-Adapter).** The IP-Adapter doc specifically warns that high `caption_dropout_rate` on `post_image_dataset` causes mode collapse. Same warning applies here, possibly worse because the model now has higher-bandwidth access to the reference (full VAE tokens vs. 16 resampled tokens). Default `caption_dropout_rate=0.1`–`0.2`.
5. **CFG dropout shape.** Whole-batch image-condition dropout (set cond=None for the whole batch with prob `p`) is the IP-Adapter approach. EasyControl's published recipe uses the same. Carry it over verbatim.
6. **VAE latent channel count vs DiT hidden dim.** The DiT's patch embedder maps VAE latent patches → hidden dim. We can reuse the existing patch embedder weights (frozen) to embed the condition tokens — no new embedder needed, just call `unet.patch_embedder(cond_latent)`. This is the cleanest option and we'll start there.

---

## Comparison to existing methods (one-line each)

| Method | What it injects | Where | Encoder | Per-step cost | Status |
|---|---|---|---|---|---|
| `ip_adapter` | Compressed image tokens (K=16) | Parallel cross-attn | PE-Core / TIPSv2 | Every step | Implemented |
| `postfix` (cross-attn coupling variant) | N learned vectors | Concat into cross-attn text K/V | n/a (learned) | Every step | Implemented |
| `prefix` | N learned vectors | Concat into self-attn K/V (prefix) | n/a (learned) | Every step | Implemented |
| `controlnet` (hypothetical) | Spatial control feature map | Side branch summed into block outputs | Image-aligned | Every step | Not implemented |
| **`easycontrol` (proposed)** | **VAE-encoded reference (full latent grid)** | **Extend self-attn KV with condition stream** | **VAE (reused)** | **~Free after step 1 (KV cached)** | **Proposed** |

---

## What we don't propose (yet)

- **Multi-condition out of the gate.** Phase 1–3 are single-condition. Multi-condition (Phase 4) is real work — branch routing, per-condition dropout, scale composition.
- **Spatial-aligned controls (canny, depth).** EasyControl's paper shows these work in the same architecture, but for Anima they should be a separate ramp once the reference-image case is solid.
- **Adapter-only flag.** Current IP-Adapter freezes the DiT entirely. EasyControl's natural form *adapts* the DiT via condition-LoRA (which, by mode-flag design, only activates on cond input — so the target path is still effectively frozen). The pure-frozen-DiT framing carries over by construction. No special flag needed.
- **Replacing `ip_adapter`.** The two coexist. `ip_adapter` is cheaper to train and may stay better for pure style-transfer "vibes." EasyControl is what you reach for when you need spatial / identity fidelity.

---

## Open decisions before Phase 1 starts

1. Reuse existing VAE latent cache for ref==target, or write a parallel `easycontrol_features` sidecar?
2. Diagnostic schema — port the IP-Adapter `[IP-Adapter diag]` block verbatim, or design something more cond-specific (e.g. per-block `‖cond contribution‖ / ‖target contribution‖` from the SDPA logit decomposition, plus `b_cond` trajectory)?
3. Default LoRA rank for the condition branch (`r=16`? `r=32`?). Higher rank = more spatial detail capacity, more params. Worth a small ablation in Phase 1.
