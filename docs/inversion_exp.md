# Embedding Inversion — Multi-Run Experiment

Log of what we hypothesized and what we actually found while trying to turn multi-seed embedding inversion into a more robust "z_T-invariant semantic embedding." Companion to `inversion-idea.md` (the original pitch) and `bench/inversion_bench.md` (the preliminary bench).

The question driving this whole thread:

> If we invert the same image N times from different seeds, the inversions disagree per-token (0.52 cos in the bench, 0.33 here) but agree in bulk (0.99 flat cos). Is that disagreement a symmetry we can quotient out — and if so, would the quotiented mean be a cleaner "pure semantic" embedding?

## TL;DR

- The per-token disagreement is **not** a token-permutation symmetry. Hungarian alignment returns identity on every run.
- It is also **not** a continuous null space that DiT's cross-attention collapses at block 0. Block-0 functional cosine ≈ raw embedding cosine.
- But at **late DiT depth** (blocks ~20 onward) the three runs nearly collapse functionally — for the two that landed in the dominant basin. One run in three is consistently an outlier that stays different all the way through.
- **Mental model**: most inversions land in one basin; an occasional outlier lands elsewhere; the naive mean is an outlier-diluted ensemble centroid. That's why the bench observed "mean looks subjectively best" — it's not denoising a single basin, it's downweighting a stochastic outlier. This is weaker than "robust invariant embedding" but stronger than "pointless smoothing."
- Path 2 as originally framed (Hungarian + mean) is dead. Path 1 (caption→embedding distillation) is still viable and should target either raw-embedding mean (OK because late-block collapse happens anyway) or a functional-space loss (cleaner theoretically).

## Journey

### H1: "It's a token permutation symmetry"

**Thought.** Flat cosine ~0.996, per-token cosine ~0.526 looked like each run writing the same semantics into different sequence positions. If you permute token slots before averaging, per-token cosine should jump and the mean should become much sharper.

**Test.** Implemented `--aggregate_by N` in `scripts/invert_embedding.py` and an `align_and_aggregate()` helper that does Hungarian 1:1 assignment between each run and a reference run using row-wise cosine as similarity. Wired a toy test: 8-token embedding with slots 2↔5 swapped recovered the exact permutation (0.752 → 1.0 per-token cos). ✅ mechanism works.

**Result.** On real 50×3 inversions of `10811132.png`, Hungarian returned **identity on every run**: all 512 slots best-matched themselves. Per-token cosine: 0.333 before, 0.333 after. Zero movement.

```
  run 0: 512/512 slots identity, 0 moved
  run 1: 512/512 slots identity, 0 moved
  run 2: 512/512 slots identity, 0 moved
```

**Interpretation.** Slot identity is preserved across runs. The variance lives *inside* each slot — each slot's vector differs across runs, but not by enough for any other slot to be a better match. The disagreement is not permutation.

**H1 — rejected.**

### H2: "Cached TE init is anchoring the slots"

**Thought.** If the DiT has no intrinsic per-slot role, maybe what's pinning slots is that every run starts from the same cached `crossattn_emb_v0` (T5 output). Seeding with zeros should free runs to drift across slots, after which Hungarian would finally have something to match.

**Test.** Re-ran the same setup with `--init_zeros`.

**Result.** Per-token cosine **went up, not down**: 0.556 (vs 0.333 with cached init), at matched reconstruction loss (~0.082). Hungarian still returned identity. The zero-init runs were *more* consistent with each other per-slot, not less.

**Interpretation.** Two things: (a) the slot roles are not inherited from TE init — they are intrinsic to DiT (presumably via frozen K/V projections and positional embedding). (b) Cached TE init drops the optimizer in a crowded region where many nearby equivalent points exist, so stochastic gradient noise pushes different runs toward different nearby basins. Zero init leaves only one dominant descent direction, so all runs take a near-identical trajectory. Useful side note: **if you want reproducible inversions, zero init beats cached TE init**.

**H2 — rejected.** Hungarian remains a no-op regardless of init.

### H3: "It's a benign in-slot null space that cross-attention collapses"

**Thought.** If there's a continuous degeneracy in raw embedding space but it points in a direction that DiT's frozen cross-attention `K_proj` projects to zero, runs that disagree in embedding space would still be *functionally* identical — same attended values on the image side. In that case naive mean is already near-optimal and path 1 just needs to match in functional space.

**Test.** Added `--probe_functional` to `scripts/invert_embedding.py` and a standalone `bench/probe_saved_inversions.py` that reuses saved per-run embeddings so we don't re-pay the 12 minutes of inversion. The probe forwards each embedding through a fixed (noise, sigma) bank and captures `block[0].cross_attn.output_proj`'s output (the attended value projected back to the image side), then computes pairwise cosines at the functional level.

**Result.** At **block 0**:

| pair           | raw flat | functional flat | Δ |
|----------------|----------|-----------------|---|
| run0 ↔ run1    | 0.669    | 0.598           | −0.071 |
| run0 ↔ run2    | 0.682    | 0.786           | +0.104 |
| run1 ↔ run2    | 0.612    | 0.487           | −0.125 |
| run0 ↔ mean    | 0.888    | 0.925           | +0.037 |
| run1 ↔ mean    | 0.847    | 0.791           | −0.056 |
| run2 ↔ mean    | 0.893    | 0.892           | −0.001 |
| **mean**       | **0.765** | **0.746**       | −0.019 |

Functional cosine ≈ raw cosine (actually slightly lower on average). Cross-attention is *not* collapsing the embedding variance at block 0 — if anything, early blocks amplify some of the pairwise differences.

**H3 (at block 0) — rejected.** But only at block 0. The obvious next question: does variance collapse later in the network?

### H4: "Late DiT blocks collapse the variance even if block 0 doesn't"

**Test.** Extended `probe_functional_space()` to accept a list of block indices and capture all of them in a single forward pass. Sweep: `0, 4, 8, 12, 16, 20, 24, 27` (DiT has 28 blocks).

**Result.** The variance trajectory is *non-monotonic* and very different between the "dominant basin" pairs and the outlier:

```
raw flat cos (reference): 0.7652

block |  mean flat cos
 0    |  0.7463
 4    |  0.7734
 8    |  0.7471
12    |  0.7286   ← minimum
16    |  0.7663
20    |  0.8092
24    |  0.8954   ← peak
27    |  0.8396
```

Per-pair trajectories tell the real story:

```
pair          |  b00  |  b04  |  b08  |  b12  |  b16  |  b20  |  b24  |  b27
run0 ↔ run1   | 0.598 | 0.624 | 0.605 | 0.676 | 0.784 | 0.639 | 0.792 | 0.639
run0 ↔ run2   | 0.786 | 0.710 | 0.676 | 0.600 | 0.618 | 0.957 | 0.977 | 0.998
run0 ↔ mean   | 0.925 | 0.890 | 0.882 | 0.887 | 0.916 | 0.957 | 0.987 | 0.998
run1 ↔ run2   | 0.487 | 0.659 | 0.596 | 0.524 | 0.557 | 0.548 | 0.758 | 0.676
run1 ↔ mean   | 0.791 | 0.843 | 0.832 | 0.836 | 0.891 | 0.832 | 0.878 | 0.724
run2 ↔ mean   | 0.892 | 0.916 | 0.891 | 0.850 | 0.833 | 0.920 | 0.980 | 1.003
```

Two distinct behaviors:

1. **run0 and run2 converge hard with depth.** `run0↔run2` climbs from 0.786 at block 0 to **0.998 at block 27** — essentially the same functional output at the model's deepest point. `run0↔mean` and `run2↔mean` hit ~1.0 by block 24–27. The "late-block collapse" story is real for these two.

2. **run1 is a persistent outlier.** `run1↔run2` is 0.487 at block 0 and still only 0.676 at block 27. `run1↔mean` actually *drops* at the final block (0.891 at b16 → 0.724 at b27). run1 also had the highest per-run training loss (0.086 vs 0.081/0.081). It genuinely landed in a different basin from the start, and DiT doesn't fold it back in.

3. **Early-block dip (b08–b12).** Interesting side observation: functional variance temporarily *grows* relative to raw cosine in blocks 8–12, then collapses. Probably early blocks exploit run-specific embedding directions before later blocks wash them out.

### Ensemble-with-outlier mental model

The cleanest story that fits all four experiments:

- The loss landscape has **one dominant basin plus occasional satellite minima**. Most runs land in the dominant basin and behave near-identically at inference (cos ≈ 1.0 at deep blocks). Some minority fraction lands in a satellite basin (run1 here) and stays functionally distinct all the way through.
- The per-slot disagreement in raw embedding space (0.33–0.67 per-token cos) is real, but the subset that matters *functionally* is small — it's mostly collapsed by blocks 20+ for basin-mates.
- Naive mean is an **ensemble centroid** that downweights the outlier by 1/N. That's why it looks subjectively cleanest — it's not denoising within one basin, it's diluting an outlier's contribution.
- Path 2 (Hungarian-aligned mean) cannot help here because there's no permutation symmetry and no continuous null-space to fold. Path 2 is dead.
- Path 1 (caption→embedding distillation) is still viable. Raw-embedding MSE to `invert_mean` is defensible because late-block collapse means most of the raw variance is functionally irrelevant anyway. A functional-space loss (match DiT cross-attention output, not the embedding) is cleaner but adds wiring cost. Either works; the choice is pragmatic.

## What to try next

If we want to push further, in roughly decreasing ROI:

1. **More runs per image, with outlier rejection.** With N=5–7, the probability of one outlier dominating the mean drops. A trivial rejection rule (drop runs whose raw or functional cosine to the running mean is below median − k·MAD) should yield a tighter centroid than uniform mean.
2. **Test on multiple images.** The entire story above is from one image (`10811132`). It's entirely possible that different images have different basin structures. Replicate the sweep on 5–10 varied images before building anything durable on top.
3. **Late-block-only probe target for path 1.** If we do path 1, don't match in raw embedding or at block 0 — use block 20+ cross-attention output as the loss target. That's the subspace where basin-mates actually agree.
4. **Investigate run1's basin.** What makes one seed land in a different basin? Is it a structural property (geometry of the loss surface near the init) or stochastic (unlucky early sigma draws)? Re-running seed 1042 twice would disambiguate. If deterministic given the seed → structural; if not → stochastic.
5. **Quality-oriented metric for the bench.** The original bench concluded "mean looks best subjectively" but failed on the chosen MSE metric. Re-run with DINOv2 or LPIPS against the source image — if the ensemble story is right, mean should dominate on perceptual metrics.

## Artifacts

Code:
- `scripts/invert_embedding.py` — adds `--aggregate_by`, `--save_per_run`, `--probe_functional`, `--probe_blocks`, `--init_zeros`
- `bench/probe_saved_inversions.py` — probe-only runner that reuses saved per-run embeddings

Data (both in `inversions_probe_test/`):
- `results/10811132_inverted_run{0,1,2}.safetensors` — per-run inversions
- `results/10811132_inverted.safetensors` — aligned mean (identity perm, so = naive mean)
- `logs/10811132_alignment.json` — per-token cosines, permutations, and the full functional-probe sweep

Run commands used:
```bash
# Three zero-init 50-step inversions, save each, probe block 0
python scripts/invert_embedding.py \
    --dit models/diffusion_models/anima-preview3-base.safetensors \
    --vae models/vae/qwen_image_vae.safetensors \
    --attn_mode flash \
    --image post_image_dataset/10811132.png \
    --steps 50 --lr 0.005 \
    --aggregate_by 3 --save_per_run \
    --probe_functional --probe_samples 4 \
    --output_dir inversions_probe_test \
    --blocks_to_swap 0 --init_zeros

# Depth sweep (reuses saved embeddings, ~30s)
python bench/probe_saved_inversions.py \
    --results_dir inversions_probe_test/results \
    --logs_dir inversions_probe_test/logs \
    --image post_image_dataset/10811132.png \
    --dit models/diffusion_models/anima-preview3-base.safetensors \
    --vae models/vae/qwen_image_vae.safetensors \
    --probe_samples 4 \
    --probe_blocks 0,4,8,12,16,20,24,27
```
