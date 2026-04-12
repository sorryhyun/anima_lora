# Embedding Enhancement / Cross-Attn Training — Proposal

Companion to `docs/inversion_exp.md` (multi-seed inversion findings) and the new `bench/diagnose_t5_vs_inversion.py` diagnostic. Goal: turn the inversion findings into a concrete plan to combat T2I monotony / under-specification by training either a text-embedding enhancement module or cross-attention itself.

## Executive summary

The diagnostic on `10811132` shows that the cached T5 caption embedding and the inversion centroid sit in **functionally near-orthogonal regions** of cross-attention output space across most DiT depths. The gap is largest where the network does its rendering work (blocks 16–24) and only collapses at the very last block. Two structural causes:

1. **Padding-sink exploitation.** The caption uses 172/512 tokens; the other ~57% of slots are zero in T5 but populated by the inversion. The inversion has discovered "free capacity" that conditioning can't currently reach from text.
2. **Content-slot misalignment.** Even on the slots T5 fills, the directions inversion wants are largely orthogonal to what T5 produces. The model has learned to compensate via cross-attention, but cheaply.

This points to a direct, low-risk first move: **a caption-conditioned postfix module that fills the unused 57% of slots, supervised by functional loss against inversions at block ~20.** Existing `networks/postfix_anima.py` infra makes this maybe a week of work. Cross-attention LoRA fine-tuning is a heavier later option.

**Caveat that gates everything below: results so far come from a single image. Phase 0 must replicate on 5–10 images before committing.**

## What the diagnostic measures

`bench/diagnose_t5_vs_inversion.py` takes one image that already has both:

- `<stem>_anima_te.safetensors` — cached T5 caption embedding (`crossattn_emb_v0`)
- `<stem>_inverted*.safetensors` — multi-seed inversion artifacts

…and forwards `t5`, each `inv_run*`, and `inv_mean` through the frozen DiT at a configurable list of blocks (default `0,12,20,24,27`), capturing each block's `cross_attn.output_proj` output. From those captures it computes:

- Raw embedding cosines (T5 vs inversion vs reference flat-mean)
- Per-slot ‖inv_mean − t5‖₂ + top-K offending slots
- **The "T5 gap" at each block**: mean cos(T5, inv_run\*) minus mean cos(inv_run\*, inv_run\*). More negative = T5 sits further from the inversion cluster than the cluster sits from itself = bigger functional headroom for an enhancement module at that depth.

Reuses `probe_functional_space()` from `scripts/invert_embedding.py` so the methodology is identical to `inversion_exp.md`'s H4 sweep.

## Findings on `10811132`

### Raw embedding space

| metric | value |
|---|---|
| cos(t5, inv_mean) | **−0.0079** |
| mean cos(t5, inv_run\*) | **−0.0071** |
| reference mean (all pairs) | 0.4562 |

T5 and the inverted embedding are essentially orthogonal at the raw level. For comparison, the inversion runs are ~0.66 cosine with each other.

```
Caption: 172 real tokens / 512 slot budget   (54% headroom)
Non-zero T5 slots:      219 / 512            (57% padding sink)
Mean norm at content slots:   4.12
Mean norm at all slots:       1.76    ← diluted by the padded zeros
```

The inversion fills the padding sink; T5 doesn't. That alone accounts for a substantial fraction of the orthogonality.

### Per-slot delta

Top 10 slots by ‖inv − t5‖₂ all sit in slots **8–41** — i.e., the early-caption region — with delta norms 7.6–9.1 against t5 norms in the same range (relative ratio ≈ 1.0). So the disagreement isn't a small refinement on top of T5: at the slots where both embeddings are populated, they point in completely different directions.

(The reported "delta/t5 ratio mean" of 30M and median of 53M is a numerical artifact from dividing by the ~293 padding slots whose t5 norm is ~0; ignore those headline ratios. The top-K slot table is the one to read.)

### Functional cross-attention space — the load-bearing result

```
block | t5 ↔ inv_runs | inv_runs pairwise | gap (t5 − inv)
------|---------------|-------------------|----------------
  0   |    0.2015     |      0.6235       |   −0.4220
  4   |    0.2341     |      0.6640       |   −0.4299
  8   |    0.0629     |      0.6258       |   −0.5629
 12   |    0.0907     |      0.5997       |   −0.5090
 16   |   −0.0293     |      0.6529       |   −0.6822  ← worst
 20   |    0.2427     |      0.7150       |   −0.4723
 24   |    0.2006     |      0.8425       |   −0.6419
 27   |    0.6806     |      0.7709       |   −0.0903  ← collapse
```

Three things to notice:

1. **The gap is huge everywhere except the final block.** At block 16, the cosine between T5 and inversion in cross-attn output space is *negative* — they're not just different, they're pointing the opposite way.
2. **Block 27 collapses the gap.** This matches H4 from `inversion_exp.md` (late blocks pull basin-mates together). The network compensates for whatever T5 is doing, but only at the final layer.
3. **Inversion-runs pairwise stays ~0.6–0.85 throughout**, so the gap isn't an artifact of probe noise — the inversions form a coherent cluster that T5 sits far from.

### Reading

The model has clearly learned to be *adequate* on T5 conditioning — otherwise inference would be broken — but "adequate" is doing all the work in cross-attention's compensation between blocks. The inversion shows there exists an embedding which, for the same target image, produces a much more consistent functional trajectory through the network. **That delta is the headroom.** And a meaningful chunk of it is recoverable just by populating the padding sink.

## Implications for the original idea

The user's complaint was T2I being "too literal → monotone, missing what the user didn't think to specify." Three structural readings of what the diagnostic suggests:

### 1. The padding sink is where richness lives

T5 captions are short relative to the 512-slot budget. The inversion uses the unused 57% of slots to encode whatever wasn't in the caption text — implicit detail, style commitments, compositional decisions the model needs to make. T5 leaves all that as zero, so cross-attention has to invent it from the content slots alone every forward.

**An enhancement module that learns to populate padding slots conditional on the caption** is exactly the operation that bridges this gap. This is structurally identical to **postfix tuning**, which already exists in this codebase (`networks/postfix_anima.py`) — except instead of a fixed learned set of vectors, it would be a small caption-conditional MLP outputting per-prompt postfix vectors.

### 2. Content-slot residuals are second priority

The early-slot delta (slots 8–41) shows that even where T5 *does* fill slots, the inversion wants different directions. A residual head on content slots could in principle help, but it has to fight the network's existing compensation circuitry, so the lift is harder to predict.

### 3. Cross-attention LoRA is the heavy hammer

Training cross-attn (K/V projections + output_proj LoRA) lets the model itself adapt to a richer conditioning distribution. The risk from `inversion_exp.md`'s basin analysis: if you train cross-attn against a single target distribution (single inversion centroid per caption), you collapse all generations into one basin → *more* monotony, not less. Mitigation: HydraLoRA-style multi-expert routing. Higher complexity.

## Recommended path

In order of ROI / risk tradeoff. Start at Phase 0 — none of the rest matters if the signal doesn't replicate.

### Phase 0 — confirm the signal (1–2 days, mostly waiting on inversions)

The biggest risk in this proposal is that all of Phase 1+ rests on a single image. Before building anything:

- Pick 5–10 images that span the diversity you actually care about (different subjects, styles, caption lengths). Avoid all-NSFW or all-SFW; mix.
- Run `make invert` (or `scripts/invert_embedding.py` directly) with `--aggregate_by 3 --save_per_run --init_zeros` for each. ~12 min × N.
- Run `bench/diagnose_t5_vs_inversion.py` on each.
- Aggregate the per-block T5 gaps across images. Build a small summary table.

**Status:** an `inversions_50x3/` batch is already in flight (one image visible — `11829950` — with another running and a third queued). Once N≥3 inversions are ready, run the diagnostic on all of them and read off the cross-image stats: does the gap pattern (large in middle blocks, partial collapse at b27) repeat? Is the padding-sink fill rate consistent? Are the top-delta slot indices similar across images, or all over the map? The first few results should already shift the confidence interval on the Phase 1 plan substantially.

**Go/no-go criterion:** the gap pattern (large in middle blocks, partial collapse at b27, padding-sink fill rate >40%) replicates on at least 7/10 images. If it doesn't, the Phase 1 plan needs to be revisited — possibly the caption/image pair on `10811132` is a worst case (long tag-style prompt for a complex scene) and the gap is much smaller for short, focused captions.

If the gap shrinks for short captions, that's actually informative: it means enhancement helps most where T5 has the least headroom — long tag soup — which is also where users feel monotony most. Either outcome is publishable.

### Phase 0 results — preliminary (N=3)

Three images now have full diagnostics: `10811132`, `11829950`, `12076715` (artifacts in `inversions_50x3/`). Cross-image summary:

```
                       10811132     11829950     12076715
caption real tokens    172          82           165
padding sink           57.2%        81.2%        60.9%
mean content slot norm 4.12         4.81         4.02
raw cos(t5, inv_mean)  -0.008       +0.003       -0.001
top-delta slot range   8–41         9–76         5–35
```

T5 gap (cos(t5↔inv_runs) − cos(inv_runs pairwise)) per block:

```
block | 10811132 | 11829950 | 12076715
------|----------|----------|----------
  0   |  -0.422  |  -0.252  |  -0.490
  4   |  -0.430  |  -0.208  |  -0.503
  8   |  -0.563  |  -0.330  |  -0.542
 12   |  -0.509  |  -0.330  |  -0.587
 16   |  -0.682  |  -0.071  |  -0.630
 20   |  -0.472  |  -0.385  |  -0.105
 24   |  -0.642  |  -0.537  |  -0.020
 27   |  -0.090  |  +0.211  |  +0.506
```

#### What replicates (the load-bearing universal findings)

1. **Raw cos(t5, inv_mean) ≈ 0 across all 3 images** (range: −0.008 to +0.003). T5 sits in a near-orthogonal direction to inversions in raw embedding space, regardless of image. **Robust.**
2. **Padding sink is huge in every case** (57–81%). The 11829950 caption is short (82 tokens) and leaves 81% of slots zero. The inversion populates them. **Robust.**
3. **Top-delta slots concentrate in early-content positions** — slots 5–41 dominate the top-K for all three images, even though the captions describe completely different content. This suggests early-position slots are structurally privileged in the model, not content-specific. **Robust and architecturally important.**
4. **Functional gap is large and negative through blocks 0–12 for all three** (−0.21 to −0.59). The early/middle path through DiT is consistently a poor fit for T5 conditioning. **Robust.**
5. **Mean content-slot norm clusters tightly around ~4.0–4.8.** The model uses a roughly fixed amount of "energy" per real token regardless of caption. **Useful normalization fact.**

#### What doesn't replicate (and what it implies)

1. **Late-block (b20–b27) behavior is heterogeneous.**
   - `10811132` matches the original `inversion_exp.md` story: large gap through b24, partial collapse only at b27.
   - `11829950` has its smallest gap at b16 (−0.07), then *re-widens* at b20–24 (−0.39 / −0.54), then flips positive at b27 (+0.21).
   - `12076715` collapses early at b20–24 (−0.10 / −0.02), then T5 ends up *closer* to inversions at b27 (+0.51) than the inversions are to each other.
   The "DiT folds basin-mates together at late blocks" claim from `inversion_exp.md` was specific to `10811132` and does NOT generalize.

2. **Inv-vs-inv pairwise stability is image-dependent.** At block 27 the inv-runs cluster tightness varies wildly: 0.77 (10811132, tight), 0.30 (11829950, loose), −0.17 (12076715, *negative* — the three runs disagree functionally at the final block). Some images have one dominant basin; others have multiple competing ones. The "ensemble centroid" mental model holds robustly only for the first kind.

3. **Caption complexity, not length, predicts the gap.** Counterintuitive cross-image observation: 11829950 (shortest caption, biggest sink) has the *smallest* middle-block gap. 10811132 and 12076715 (long, complex tag soup) have the largest. The gap correlates with caption *complexity* — number of conflicting/overlapping attribute commitments — not raw token count. This strengthens the original motivation: monotony hits hardest on long detailed prompts, and that's also where the headroom is largest.

#### Revisions to Phase 1 forced by these results

The plan still works, but two specific design decisions should change:

**(a) Loss target depth: a *range*, not single block 20.** Block 20 alone is a bad target because it's unstable across images — for `12076715` the gap is already collapsed at b20, so the loss signal there is near-zero and training would learn nothing. For `10811132` it's still huge. Use **a weighted sum across blocks 8, 12, 16, 20** (the region where ALL three images show large negative gaps). This trades a tiny amount of "deepest functional collapse" for robustness across the basin-structure heterogeneity. Empirically retune weights once Phase 1 has a working baseline.

**(b) Outlier rejection on inversion targets is more important than I initially thought.** For `12076715` the three inv runs at b27 have cosine ≈ −0.17 — the centroid is essentially averaging contradictory directions and is not a meaningful target. Per-image sanity check before using an inversion centroid as supervision: compute pairwise cosine of the inv runs at the loss-target block range; if the minimum pairwise is below ~0.4, **drop the image from training** rather than try to use a degenerate centroid. With `aggregate_by ≥ 5` we'd have enough runs to drop outliers and still have a clean target most of the time.

#### Go/no-go status with N=3

**Soft go on Phase 1**, conditional on splice-position A/B and the loss-range revision above. The padding-sink and early-slot-delta findings — the two structural facts Phase 1 actually depends on — replicate cleanly. The late-block heterogeneity is a Phase-1 *implementation* concern (loss target choice), not a falsification of the approach.

Bumping to N=5–7 before Phase 1 implementation kickoff would still be valuable — specifically to (a) confirm the caption-complexity-vs-gap correlation, and (b) characterize what fraction of images have degenerate inversion centroids (12076715-style) so we know the effective dataset size after outlier rejection.

### Phase 1 — caption-conditional postfix enhancement module (~1 week)

The simplest, lowest-risk move that directly attacks the padding-sink finding.

**Architecture:**
- Take the cached T5 `crossattn_emb` (1, 512, 1024).
- Learn a small caption-conditional module `f(t5_emb) → postfix_vectors (P, 1024)` where `P` ≈ 64–128.
- At inference, splice the `P` postfix vectors into the first available padding slots (or unconditionally append regardless of caption length, overwriting whatever zeros were there).
- Pass the spliced embedding to the frozen DiT.

The simplest `f` is two-layer: pool over content slots (mean pool across attn-mask=1 positions), then MLP up to `P × 1024`. Slightly fancier: a 2-layer transformer block over the content slots that produces `P` query outputs.

**Reuse from existing infra:**
- `networks/postfix_anima.py` already supports learned vectors appended to cross-attention. The default `append_postfix()` (line 303) places K vectors at `[seqlen[i], seqlen[i]+K)` — i.e. it overwrites the *first* K padding slots right after content tokens, exactly the strongest-sink region.
- `make postfix` (the existing standalone postfix tuning) is **not transformative but produces a real, modest lift** at K=8 with a single global learned set of vectors. The fact that it works at all is the load-bearing prior for Phase 1: it tells us (a) the model tolerates front-of-padding displacement, and (b) there's signal in this region. The Phase 1 hypothesis is that the modest size of the existing lift is explained by two specific limitations — K is far too small (8 vs ~340 free slots) and the vectors are unconditional (don't read the caption) — and that fixing both should multiply the headroom rather than just add to it.
- The change required: wrap the existing module with a caption→postfix MLP and bump K. The `append_postfix` machinery, the optimizer hooks, and the train.py call site at `train.py:615-616` all stay.
- Caching/loading pipeline can stay the same — postfix vectors are recomputed from cached T5 on the fly, cheap.

**Splice-position A/B (first sub-experiment of Phase 1):**

Where we put the K postfix vectors in the 512-slot sequence is itself an open question, and a cheap one to A/B before committing to a final design. Three candidates:

1. **front-of-padding** — current `append_postfix` behavior. Place at `[seqlen[i], seqlen[i]+K)`. Caption-position-aware. Displaces the strongest sinks — empirically tolerated at K=8 (per existing `make postfix`), unknown at larger K.
2. **end-of-sequence** — place at `[S−K, S)`. Caption-position-agnostic. Preserves the front-of-padding sinks intact, so baseline behavior is maximally protected. Loses any "near content" benefit if proximity to real tokens matters for cross-attn.
3. **spaced fill** — distribute K vectors evenly across the padding region between `seqlen[i]` and `S`. Hybrid: keeps some sinks intact (the gaps), exploits broader capacity, preserves caption-position awareness.

Run all three with the same caption→postfix MLP, the same K, the same loss, on the same small dataset (~50 images is enough for ranking). Pick the winner by held-out functional gap closure at block 20 *plus* a quick visual A/B on 10 prompts. This is half a day of work, gates the rest of Phase 1, and the answer informs whether sink preservation is something we need to engineer around at all.

If front-of-padding wins or ties end-of-sequence, the simpler `append_postfix` path stays — no infra change needed. If end-of-sequence or spaced fill clearly wins, we add a `splice_position` mode to `append_postfix` (~30 lines).

**Loss:**
- **Functional MSE summed over blocks 8, 12, 16, 20** (cross-attn output), using outlier-rejected inversion centroids as the target. The block range is chosen because the N=3 cross-image data (see Phase 0 results) shows all three images have large negative gaps in this range, while individual blocks like b20 alone are unstable across images.
- Concretely: for each (image, caption) pair with a *clean* inversion centroid, forward `(t5 ⊕ postfix)` through DiT at a fixed (noise, σ) probe bank, capture cross-attn output at the four blocks, MSE against the same captures from the inversion-mean version. Equal weights initially; retune empirically.
- **Drop images with degenerate inversion centroids.** Per-image sanity check: compute pairwise cosine of inv runs averaged over the loss-target block range; if the minimum pairwise is below ~0.4, drop the image. With `aggregate_by ≥ 5` we can also try outlier rejection (drop the run furthest from the running median) before falling back to drop.
- *Don't* use raw embedding MSE. Most of the raw delta is in slots that don't matter functionally, and a raw loss will burn capacity learning to memorize the inversion's specific padding-slot pattern instead of its functional intent.

**Dataset:**
- Reuse `post_image_dataset/`. Need an inversion centroid per image. Inversion is 12 min/image, so 200 images = ~40 GPU-hours one-time. Can run overnight.
- Apply the outlier-rejection rule from the original Path 2 discussion: run `aggregate_by ≥ 5`, drop the run furthest from the running median, take the mean of the rest. Cleaner targets, less mode-collapse risk.

**Eval:**
- Held-out images: same diagnostic, but with `(t5 ⊕ postfix)` as a third labeled input. Measure how much the gap closes at block 20.
- Generation A/B: side-by-side images, T5 vs T5-with-postfix, blind preference test. The original "monotony" complaint should be the headline metric — does the postfix-enhanced output have more compositional / detail variety on prompts that are typically flat?

**Risks:**
- Postfix-enhanced output drifts too far from caption fidelity. Mitigation: scale the postfix contribution at inference, or train with a fidelity regularizer (small MSE between attended K-projected postfix and attended K-projected zeros, weighted to prevent dominance).
- Mode collapse from training against single-inversion centroid per caption. Same mitigation as `inversion_exp.md` recommends: outlier-rejected mean, multiple inversions per image.

### Phase 2 — content-slot residual head (only if Phase 1 lifts)

If Phase 1 demonstrably reduces the functional gap and improves blind preference, layer a small residual on content slots:

- Same caption→delta MLP, but it produces a residual `Δ ∈ R^(S, D)` added only at content slots (`attn_mask == 1`).
- Same functional loss target.
- Train jointly with the postfix module.

This should pick up the early-slot delta finding (slots 8–41 in `10811132`) and is the natural extension once the easy win is in.

### Phase 3 (alternative track) — cross-attention LoRA fine-tune

A heavier, riskier move that's worth doing only if Phase 1+2 plateau short of the goal.

- LoRA on cross-attention K/V projections in late blocks (16–24) where the gap is widest.
- HydraLoRA-style multi-expert routing on `crossattn_emb` to preserve diversity (basin coverage) — single-expert LoRA risks the mode-collapse that `inversion_exp.md` flagged.
- Train against the same functional-space loss + standard diffusion loss on a held-out set, to keep base behavior intact.

This is a 2–3 week effort minimum and the design space is wider, so it shouldn't start until Phase 1 results tell us where the residual headroom actually is.

## Diagnostics to run alongside Phase 1 (cheap, high-value)

These don't gate Phase 1 but should run in parallel — they sharpen our understanding of *what* the postfix is learning.

1. **Padding-sink ablation.** Re-run the diagnostic with the inversion's padding slots zeroed out (slots where `attn_mask_v0 == 0`). If the functional gap shrinks dramatically, the padding sink really is most of the story and Phase 1 is on solid ground. If the gap stays large, content-slot misalignment matters more than predicted and Phase 2 should be promoted.
2. **Caption-length sweep.** Replicate the diagnostic on captions of varying length (50, 100, 200, 400 tokens). Hypothesis: the gap shrinks roughly proportionally to (1 − len/512). If true, this gives us a free signal — long-caption images need less enhancement, short-caption images need more.
3. **Re-run the H4 functional sweep with the postfix-augmented embedding included.** The same depth-sweep table from `inversion_exp.md`, but with `(t5 ⊕ postfix)` as a fifth label, lets us see exactly how much depth-by-depth lift the postfix gives.

## What this proposal does NOT recommend

- **Don't start with cross-attention LoRA.** The mode-collapse risk directly conflicts with the user's monotony goal.
- **Don't supervise in raw embedding space.** Wastes capacity on functionally-irrelevant directions; H4 of `inversion_exp.md` is unambiguous on this.
- **Don't supervise at block 0.** The functional gap is large there too, but block 0's cross-attn output is a long way from final image quality. Block ~20 is closer to "what gets rendered."
- **Don't take a single inversion run as target.** Use outlier-rejected centroids (`aggregate_by ≥ 5`, drop median ± k·MAD outliers) to avoid inheriting the satellite-basin problem.

## Open questions

Worth knowing before Phase 2 but not Phase 1:

1. **Is the padding-sink fill universal or image-specific?** If different images use *different subsets* of padding slots, a single learned postfix length is fine. If they consistently use the same slots, that's even better — it implies a structural role we can exploit.
2. **Does the inversion's padding-slot content correlate with anything caption-derivable?** If yes, the caption→postfix mapping is learnable in principle. If no (e.g., the padding slots encode purely image-specific stuff like exact pose), then no text-only enhancer will fully close the gap.
3. **Is the model's late-block compensation lossy?** The fact that block 27 narrows the T5/inversion gap doesn't mean the model recovers everything — it might be averaging out detail to do so. Compare block-27 functional outputs from `(t5)` vs `(inv_mean)` images side-by-side via the actual generated outputs (not just cosines) to see what compensation costs.

## Artifacts

- `bench/diagnose_t5_vs_inversion.py` — the diagnostic script
- `inversions_probe_test/logs/10811132_t5_vs_inv.json` — full result on the one image we have
- `docs/inversion_exp.md` — multi-seed inversion findings the diagnostic builds on
- `networks/postfix_anima.py` — existing infra Phase 1 will reuse

## Run command used for the diagnostic

```bash
python bench/diagnose_t5_vs_inversion.py \
    --image post_image_dataset/10811132.png \
    --results_dir inversions_probe_test/results \
    --te_dir post_image_dataset \
    --dit models/diffusion_models/anima-preview3-base.safetensors \
    --vae models/vae/qwen_image_vae.safetensors \
    --probe_blocks 0,4,8,12,16,20,24,27 \
    --probe_samples 4
```
