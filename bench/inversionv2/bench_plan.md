# Tag-Conditional Slot Subspace Bench — Plan

Follow-up to `slot_subspace_analysis.py`. Goal: decompose per-slot T5 embedding variance into **"variance explained by known tag identity" + "residual"** so future inversion runs can bolt high-signal slots (rating / count-meta / artist) to data-derived prototypes instead of jittering them freely.

Destination script: `bench/inversionv2/tag_slot_analysis.py`
Output root: `bench/inversionv2/results/tag_slot/`

## Motivation — what we know so far

From `bench/inversion/results/slot_subspace/`:

- Content length p95 = 248, median 151. Current `--active_length=128` truncates past the median.
- Per-slot effective rank (slots 16–128 where `n_samples > 1500`): **k@95% ≈ 300 / 1024**. Compressibility is real but modest (~3.4×).
- Pooled basis fails: top-64 shared directions explain only 58% of typical slot energy → subspaces are position-specific.
- Inversion currently optimizes the full 1024-D per slot, so ~70% of its parameter budget can drift into directions T5 never uses. This is the observed "jitter."

The ~300-D per-slot rank is an *unconditional* measurement. Real captions in this dataset follow a rigid booru convention:

```
rating, count_meta(s), character(series), series, @artist, <alphabetical tags…>
```

…and the first ~5–7 tags are nearly deterministic. If we condition on **which tag** a slot holds, the residual variance should collapse dramatically. That's the hypothesis this bench tests.

## Dataset evidence (1987 captions in `post_image_dataset/`)

- **100% have @artist**, median tag-index 5 (p10=3, p90=6).
- **63 unique artists**; 54 have ≥5 images. Top 5: @sincos (243), @hews (126), @mikozin (97), @tottotonero (90), @sumiyao (amam) (81). Plenty of samples for per-artist SVD.
- `1girl` in 1800 captions at median tag-index 2; `1boy` at tag-index 1; `2girls` at 1; `solo` present in 700.
- Ratings: `explicit` (1121), `sensitive` (756), `absurdres` (90), `general` (20) at tag-index 0.
- **TE cache `*_v0..v7` variants are tag-shuffled copies** of the same caption (see `library/anima/strategy.py:342` `_generate_shuffled_captions` → `anima_smart_shuffle_caption`). Confirmed via distinct `t5_input_ids_v*` per variant. This multiplies effective sample count by 8× **and** lets us test position-invariance of tag directions directly.

## Tokenizer setup (no model load needed)

Crossattn slots are indexed by the **T5 tokenizer**, not Qwen3. The adapter maps Qwen3 hidden states onto T5 token positions, so `t5_input_ids_v{vi}` is authoritative for slot identity. Bundled at `library/anima/configs/t5_old/`. Load with:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("library/anima/configs/t5_old")
enc = tok(caption_text, return_offsets_mapping=True, truncation=True, max_length=512)
```

**Caption shuffle.** The cache stores 8 shuffled variants per caption (see `library/anima/training.py:30` `anima_smart_shuffle_caption`). The .txt sidecar is the ORIGINAL order — comparing `tok(txt)["input_ids"]` to `t5_input_ids_v{vi}` fails for all v>=0 because the suffix is shuffled. Do NOT try to reconstruct the shuffle — instead, **decode the cached ids and re-tokenize** to recover the shuffled caption string per variant. The SentencePiece round-trip `tok(tok.decode(ids))["input_ids"] == ids` is exact (verified on 10003461 v0). That re-tokenization is also what gives you the char→token offset mapping you need for tag→slot resolution.

**Sanity check required:** for each image × variant, assert that round-trip tokenization is exact. If any mismatches, fail loudly — it means the cached ids came from a different tokenizer revision than what's bundled.

## Phase 1 — Tag → token-slot mapping (shared infra)

The tokenizer returns character offsets per token. For each target tag `T` in a caption string:

1. Find `T`'s character range `[a, b)` in the caption (handle `, ` separators; be aware of substrings — `1girl` is also a substring of `2girls`, so match on comma-delimited tokens, not raw substring).
2. From `offset_mapping`, collect all token indices whose `(start, end)` falls inside `[a, b)`. That's the slot range for `T` in that caption.
3. A tag can span multiple tokens (e.g. `@sumiyao (amam)` is likely 4–6 tokens). For per-tag-class SVD we'll treat **the first token of the tag** as the canonical slot. Record the full range too for audit.

**Tags to track:**

- **rating**: always tag-index 0. Actual distribution in this dataset: `explicit` (1121), `sensitive` (756), `absurdres` (90), `general` (20). Zero `questionable` anywhere (likely dropped or re-mapped upstream) — do **not** include it as a class. Include `absurdres` and let the data decide whether it behaves like a rating.
- **count-meta**: `1girl`, `1boy`, `2girls`, `2boys`, `3girls`, `1other`, `solo`, `multiple_girls`, `multiple_boys` (track each separately; don't merge).
- **artist**: any tag starting with `@`. Each caption has exactly one.

**Output**: `phase1_positions.json` with per-tag-class lists of `(stem, variant_idx, first_token_slot, last_token_slot, n_tokens)` tuples. Plus a summary histogram: for each class, token-position distribution (mean / p10 / p50 / p90 / n_occurrences_total).

This phase is pure bookkeeping — no SVD. Cost: ~a few minutes of tokenizer calls. Outputs feed all subsequent phases.

## Phase 2 — Per-tag-class subspace SVD

For each target class `T`:

1. Collect every `crossattn_emb_v{vi}[first_token_slot_T, :]` across all (image, variant) occurrences from Phase 1.
2. Stack into `(N_occurrences, 1024)` float32. Non-centered SVD (for the same reason as the existing bench — we want total-energy subspace, not variance-around-mean).
3. Record:
   - Top-16 singular values
   - Effective rank k@80/90/95/99
   - **Mean vector**: if its norm is large and the top singular direction's VE is >50%, there's a canonical direction. Save the mean vector separately to `phase2_class_prototypes.safetensors` — it's directly usable as an inversion anchor.
   - Within-class pairwise cosine distribution (histogram) — how tight is the cluster?

**Key class list:**
- `rating=explicit`, `rating=sensitive`, `rating=general`, `rating=absurdres` (treat each rating as its own class)
- `1girl`, `1boy`, `2girls`, `solo`, `multiple_girls`
- (artists handled separately in Phase 3)

**Expected outcome shape:**
- If `1girl` has k@95% ≈ 3–10 and mean norm comparable to singular-value-1, the conditional subspace is **much** tighter than the 300-D unconditional slot rank. That's the headline.
- If k@95% is still ~100, tag identity doesn't collapse the slot much and the narrative shifts.

**Output**: `phase2_class_subspaces.json` (spectra, rank, cluster stats) + `phase2_class_prototypes.safetensors` (mean vectors).

## Phase 3 — Per-artist clustering

For each artist with ≥5 images:

1. Collect all `crossattn_emb_v{vi}[artist_slot, :]` for that artist (N_images × 8 variants).
2. Compute **within-artist mean cosine** (pairwise cosine of that artist's vectors, averaged).
3. Compute **between-artist mean cosine** (pairwise cosine of this artist's mean vs every other artist's mean).
4. Ratio = within / between. If >> 1, artist identity is cleanly separable at the artist slot.

Also run a single SVD across all artists' mean vectors to get a "style manifold" — the top principal components are the dominant style axes in the dataset. Report top-3 singular values and the artists whose means project most strongly onto each.

**Output**: `phase3_artist_clustering.json` with per-artist within/between cosines, cluster tightness ranking, and top style directions.

**Threshold for "useful":** if within-artist cosine > 0.7 and between-artist cosine < 0.3 for most artists, **we can use each artist's mean vector as a frozen inversion slot** (for any image by a known artist). That's a direct, tight inversion constraint on one slot — the biggest single expected win from this bench.

## Phase 4 — Position-invariance test

For each (caption, tag) pair where the tag appears in multiple variants:

1. Collect the 8 slot-vectors from `v0..v7` at whichever slot the tag lands in per variant.
2. Compute pairwise cosine between those 8 vectors.
3. Aggregate: median pairwise cosine across all (caption, tag) pairs, per tag class.

**Important caveat on shuffle scope.** `anima_smart_shuffle_caption` keeps the prefix through the first `@artist` tag **fixed across all 8 variants** — only tags *after* the artist are shuffled (section-wise around `On the .../In the ...` delimiters). So:

- rating / count-meta / character / series / artist → slots are IDENTICAL across v0..v7 per caption. Any cosine < 1.0 measures LLM-adapter noise from different suffix contexts, not position sensitivity. Still worth reporting, but interpret as a baseline.
- alphabetical suffix tags → slots genuinely vary across variants. These are the real position-invariance measurement.

The script should surface both, labelled.

Interpretation (for the suffix-tag case, which is where this test has real teeth):
- Median cos > 0.9 → tag direction is position-invariant within a caption. Strong signal: a tag-conditional projector only needs one vector per tag, reused regardless of slot.
- Median cos 0.5–0.9 → position-modulated but substantial shared component. A low-rank positional perturbation model (mean + position-dependent residual) works.
- Median cos < 0.5 → position dominates tag identity at this slot. Tag-conditional inversion constraint would be weak.

**Output**: `phase4_position_invariance.json` with per-class position-invariance summary.

## Implementation order & rough cost

| Phase | Cost | Dep |
|-------|------|-----|
| 1 — Tag→slot mapping | ~5 min tokenizer calls on 1987 × 8 = 15,896 strings | none |
| 2 — Per-class SVD | ~1 min (each SVD is small) | Phase 1 |
| 3 — Per-artist clustering | ~2 min (63 artists × pairwise) | Phase 1 |
| 4 — Position invariance | ~2 min (cheap pairwise per-caption) | Phase 1 |

Full run should be under 15 minutes wall-clock, dominated by TE cache loading (already ~3 min for the existing bench). Memory peak: ~4 GB for the full `(1987, 8, 512, 1024)` bf16 stack. If that's too much, process per-variant with lazy loading.

## CLI sketch

```
python bench/inversion/tag_slot_analysis.py \
    --image_dir post_image_dataset \
    --tokenizer library/anima/configs/qwen3_06b \
    --variants all             # or 0 to use only v0
    --min_artist_images 5
    --output_dir bench/inversion/results/tag_slot
    --max_images N             # for smoke tests
```

## Output file map

```
bench/inversion/results/tag_slot/
├── phase1_positions.json              # per-class slot histograms + (image, variant, slot) audit lists
├── phase2_class_subspaces.json        # per-class spectra, ranks, within-class cosine stats
├── phase2_class_prototypes.safetensors  # {class_name: mean_vector (1024,) bf16}
├── phase3_artist_clustering.json      # per-artist within/between cosines, style SVD
├── phase3_artist_prototypes.safetensors # {artist_name: mean_vector (1024,) bf16}
├── phase4_position_invariance.json    # per-class median pairwise cos across variants
└── summary.md                         # tight human-readable summary written by the script
```

The two `.safetensors` files are the load-bearing artifacts — everything else is analysis. They're directly consumable as inversion anchors in Phase 5 below.

## Phase 5 (follow-up, not part of this bench) — wire into `archive/inversion/invert_embedding.py`

Only start this after Phase 3 shows within/between cosine ratio strongly favors per-artist clusters. Add:

- `--tag_anchor path/to/phase2_class_prototypes.safetensors` — freezes specified slots to class-mean vectors.
- `--artist_anchor path/to/phase3_artist_prototypes.safetensors --artist_name @sincos` — freezes the @artist slot to the artist's mean vector.
- `--anchor_mode freeze|init|regularize` — hard-freeze (don't optimize), warm-init (optimize from prototype), or soft-pull toward prototype with a weighted L2.

The caption .txt for the inversion image is the authority on which slots get which anchors. For a caption whose rating = `explicit`, first meta = `1girl`, and artist = `@sincos`, the first ~5 slots get locked to known-good prototypes and only the remaining ~125 content slots are optimized. That directly removes the "jitter" in the high-signal positions.

## Caveats

1. **Tokenizer validity.** The Phase 1 assertion (tokenized IDs match cached `t5_input_ids_v{vi}`) is load-bearing. If the cached IDs were generated by a different tokenizer revision than what's bundled in `configs/qwen3_06b/`, positions will be wrong everywhere downstream. Fail loudly on mismatch; don't paper over it.
2. **Tag substring ambiguity.** `1girl` is a substring of `2girls`; `1boy` of `1boy...` variants. Match on comma-delimited tag tokens in the caption string, not raw `find`.
3. **`absurdres` at tag-index 0.** 90 captions have this as the rating slot. It's a quality tag, not a rating. Either it was misordered by the preprocessing pipeline, or this dataset uses it intentionally as a rating-slot stand-in. The bench should include it as its own class and let the subspace structure tell us whether it behaves like a rating or not.
4. **Phase 4 caveat on n=8.** 8 variants per caption is a small sample for pairwise cosine median; medians will be noisy per caption. Aggregate over the ~16k (caption, tag) pairs before drawing conclusions.
5. **Per-artist sample size imbalance.** @sincos has 243 × 8 = ~1944 samples; some artists have 5 × 8 = 40. Don't pool their SVDs — compare ranks separately or weight by sample count.
6. **Character tags are intentionally out of scope.** They're high-cardinality (near-unique per image) so per-class SVD isn't meaningful. If a future phase wants to cluster by character, it needs a separate path (e.g. character-mean embeddings from multi-image characters only).

## What "success" looks like

Minimum-viable success: Phase 2 shows `1girl` / `1boy` / `solo` each have k@95% < 20, mean-norm ≈ singular-value-1, and within-class cosine > 0.8 median. Then we have clean prototypes for those slots and the inversion jitter story is half-solved.

Full success: Phase 3 also shows artist within/between cosine ratio > 3 for ≥ 80% of artists with ≥5 images. Then the @artist slot is tightly identifiable and Phase 5's hard-freeze mode becomes viable as the default.

Null result: if tag identity doesn't collapse the slot subspace (k@95% > 50 conditional on class), then slots don't have tag-specific prototypes and the inversion has to keep optimizing those slots freely. That's still informative — it would mean the model spreads tag information across many slots rather than localizing it. In that case the right next step is a representation-similarity analysis (RSA) across slots conditioned on tag presence/absence, which is substantially more involved and should be scoped as its own bench.
