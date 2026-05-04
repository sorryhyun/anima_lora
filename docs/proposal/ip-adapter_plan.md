# IP-Adapter plan — manifold-collapse fix via trainable PE-Core late layers

Source paper: arXiv 2605.00503v1 (EOSTok, ByteDance Seed). The mechanically
relevant finding for IP-Adapter is **implicit alignment** (Table 2: 1.75 →
1.02 rFID, 12.27 → 3.32 gFID by aligning *intermediate* encoder features
to a frozen VFM, while leaving the K query outputs unconstrained).

## Goal

Address the PR=6.2 effective-dimensionality collapse on Anima dataset
(see `memory/project_pe_feature_diagnostics.md`) by making PE-Core's late
layers adaptable to the manga/anime distribution while keeping them
semantically anchored to the frozen reference.

## What's already correct

The current IP-Adapter design (`networks/methods/ip_adapter.py`,
`docs/methods/ip-adapter.md`) already follows the paper's "no direct loss
on K outputs" rule: the K=16 IP tokens are supervised only through FM loss
via per-block `to_k_ip`/`to_v_ip` projections. Semantic anchoring comes
from upstream PE-Core, which is frozen. **Don't change this.**

## What's broken

PE-Core was trained on natural photos; Anima dataset is manga/anime.
Diagnostic measurements (memory note) put effective rank at PR=6.2 — the
features are pinned to a narrow subspace. Mean-centering helped (shifted
the subspace toward usable variance) but doesn't broaden it. The encoder
itself can't adapt to the data distribution because it's frozen.

The paper's prescription: make the encoder trainable, but regularize its
intermediate features against the frozen reference so it can't drift into
non-semantic noise.

## Architecture

```
image → PE-Core early layers (FROZEN, cached)
              ↓
         PE-Core late layers (FROZEN base + LoRA, trainable)
              ↓                         ↓
       layer-L_align hidden     layer-final patch tokens
              ↓                         ↓
   implicit-alignment loss       PerceiverResampler (existing)
   vs frozen PE-Core layer-L_align         ↓
                              K=16 IP tokens
                                          ↓
                          per-block to_k_ip, to_v_ip (existing)
                                          ↓
                       FM loss through frozen DiT (existing)
```

Three trainable groups:

1. **PE-Core late-layer LoRA** — small LoRA (rank ~16-32) on attention
   q/k/v/o (and optionally MLP) for the last N layers of PE-Core's vision
   tower (`library/models/pe.py`). New.
2. **Resampler + per-block IP projections** — unchanged from current
   design.
3. **Implicit-alignment projector** — small MLP `h_align`, maps PE-Core
   layer-L_align features (with LoRA active) to the same dim as frozen
   PE-Core layer-L_align features for cosine comparison. New, tiny.

## Training

Loss:

```
L = L_FM + λ_align · L_implicit + λ_drop · L_FM(image_dropped)
```

- `L_FM`: standard FM through frozen DiT. Unchanged.
- `L_implicit`: `1 − cos(h_align(PE_lora_at_L), PE_frozen_at_L))` per patch
  position, averaged. The frozen target is the *same image's* PE features
  at layer L_align without the LoRA active.
- `L_FM(image_dropped)`: existing CFG dropout via `image_drop_p`.
  Unchanged.
- `λ_align`: start 0.1. The trade-off is "let the encoder adapt" vs
  "keep it semantic." If `λ` is too high, manifold collapse persists; too
  low, encoder drifts.

## Caching strategy (the load-bearing detail)

The current pipeline pre-caches full PE-Core features as
`{stem}_anima_pe.safetensors`. A trainable LoRA breaks that — the cache
is invalidated the moment the LoRA changes.

**Solution: split the cache at the LoRA cut-point.**

Let the LoRA cover the **last N layers** of PE-Core (where N = something
like 6-8 of PE-Core-L14's 24 layers). Then:

- **Cache the activations entering layer (depth − N)** as
  `{stem}_anima_pe_pre.safetensors`. This is "PE-Core early layers
  output" — frozen, cacheable, shared across every training run.
- **Run only the last N layers + LoRA + resampler + IP projections
  online** during training. Forward cost is much smaller than the full
  PE-Core forward.
- **Implicit-alignment target** is the frozen PE-Core layer L_align
  output for the same image, also cached
  (`{stem}_anima_pe_align.safetensors`). L_align should be *inside* the
  LoRA region (e.g. one of the last N layers) — that's the layer whose
  features the LoRA is allowed to drift, regularized against the frozen
  reference at the same depth.

Cache invalidation: only when N or L_align changes. Both are fixed config
choices, not training-state-dependent.

VRAM cost increase: running last N layers online instead of feeding
cached features straight into resampler. Manageable; PE-Core-L is small
relative to DiT. Worth measuring before committing.

## Implementation steps

1. **Pick N and L_align.** Pre-experiment: probe PE-Core layer-by-layer on
   Anima dataset, measure where dataset-specific drift would help most
   (rank/PR per layer). Hypothesis: late layers (last 6-8) hold the
   dataset-specific bias; align target somewhere in middle of the LoRA
   region (L_align = depth − N + 2 or so).

2. **Wire LoRA into PE-Core.** `library/models/pe.py` is vendored from
   Meta's perception_models — small surface area. Add LoRA injection at
   attention modules of layers `[depth − N, depth)`. Pattern: replicate
   networks/lora_modules/ adapter style but scoped to this single model.
   Probably a new `networks/methods/ip_adapter_pe_lora.py` rather than
   reusing the generic LoRA path, because PE-Core layers don't follow the
   DiT module naming the LoRA dispatcher expects.

3. **Split the cache.** Update `preprocess/cache_pe_encoder.py`:
   - Add `--split_at_layer K` flag — runs PE-Core forward, dumps the
     activation entering layer K as `{stem}_anima_pe_pre.safetensors`.
   - Add `--align_at_layer L` flag — also dumps frozen layer-L output as
     `{stem}_anima_pe_align.safetensors`.
   - Default behavior unchanged (still produces the legacy
     `{stem}_anima_pe.safetensors` for non-LoRA training).
   - Add `make preprocess-pe-split` target.

4. **IP-Adapter network update.** Modify
   `networks/methods/ip_adapter.py`:
   - Accept the cached `*_pe_pre` features instead of `*_pe`.
   - Add a "tail" forward that runs PE-Core's last N layers with LoRA
     active, returning both the final patch tokens (for resampler) and the
     L_align output (for alignment loss).
   - Expose the alignment loss to the trainer via the `network.loss`
     hook pattern (see how REPA loss is wired — same shape).

5. **Config.** `configs/methods/ip_adapter.toml` gains:
   - `pe_lora_n_layers` (default 0 = disabled, falls back to current
     behavior)
   - `pe_lora_rank`, `pe_lora_alpha`
   - `pe_align_layer`, `pe_align_weight`
   - GUI variant in `configs/gui-methods/ip_adapter.toml` with the new
     keys exposed.

6. **Bench.** Extend `bench/ip_adapter/` (new if absent) with PR /
   effective-rank measurement, before vs after LoRA. The diagnostic that
   showed PR=6.2 is the same diagnostic that should show PR ≫ 6.2 after
   adaptation, if this works.

## Decisions to make

- **Which PE-Core variant.** Default L14-336 (current). G14-448 (`pe-g`)
  is bigger and would need more careful LoRA placement. Start with L14.
- **LoRA on attention only, or attention + MLP.** Attention-only is
  cheaper and matches the standard LoRA story. MLP adds capacity but
  doubles parameter count. Start attention-only.
- **`pe_align_weight` sweep range.** [0.01, 0.5]. Likely the most
  sensitive knob.
- **N (LoRA depth).** [4, 8] of PE-Core-L's 24 layers. More = more
  capacity to broaden manifold but also more drift risk.
- **Mean-centering interaction.** Existing mean-centering helped. Does
  it stack with implicit alignment, or is it now redundant? Keep as a
  separate config flag; ablate.
- **Augmentation revisit.** Memory note says "aug is feature-level
  near-no-op" — that was with frozen PE-Core. With trainable PE-Core,
  augmentation might re-engage because the encoder can now respond to
  it. Worth re-running aug ablation under the new design.

## Why no quantizer

Same logic as `img2emb_plan.md`: IBQ + APR is defending against
codebook collapse in AR next-token prediction. IP-Adapter has no AR head,
no vocabulary, no NTP loss. The K=16 IP tokens are continuous and consumed
by per-block parallel cross-attention.

Critically, the failure we observe (manifold collapse, PR=6.2) is *not*
the failure quantization addresses. Quantization would force the K
outputs onto a finite codebook of points within the collapsed subspace —
strictly worse than the current continuous regime, which can at least
encode position within the subspace at full precision. Discreteness is
not a missing ingredient here.

## Risks

- **Cache split adds friction.** Two new cache files per image, gated by
  config. Users will run training with the wrong cache loaded and get
  silent garbage. Add a dim/shape check on cache load.
- **Forward-cost increase from running last N layers online.** Measure.
  If it's >25% step-time hit on `low_vram` preset, reconsider N.
- **PE-Core LoRA could drift into "remembering training images" instead
  of broadening the manifold.** The implicit-alignment loss is the
  defense, but only at L_align. Worth a probe: train with very high
  `λ_align` (essentially freezing the encoder back) and verify gFID
  reverts to baseline — confirms the loss is doing the work.
- **Mean-centering may interfere with the cosine alignment** (cos sim
  invariant to scale, not to mean shift). Apply mean-centering downstream
  of the alignment target so the alignment compares centered-vs-centered
  features.
