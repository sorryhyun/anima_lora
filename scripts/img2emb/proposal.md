# Img2emb training improvements

Two complementary changes to the resampler training regime. They compose —
K-cap cleans up *per-slot* geometry (no pad-leak), InfoNCE cleans up
*set-level* geometry (variant-invariance) — but either can ship
independently. Recommended order: **K-cap first, then InfoNCE**.

---

# Part 1 — K-slot hard cap (K=256)

## Problem

`crossattn_emb` targets are shape `(S=512, D=1024)` but ~70% of every row is
padding. Over the 1987-image cache:

| L range | count | cumulative |
|---:|---:|---:|
| ≤ 50 | 7 | 0.4% |
| ≤100 | 212 | 10.7% |
| ≤150 | 969 | 48.8% |
| ≤200 | 1646 | 82.9% |
| ≤250 | 1888 | 95.0% |
| ≤300 | 1954 | 98.4% |
| ≤400 | 1985 | 99.9% |
| max  | 410 | — |

Median L = 151, mean 156, max 410. Slots `[L:512]` are **exactly zero** in
the cached target (`scan_active_lengths` at `extract_features.py:252`
measures the non-zero prefix) — zero-padding is not an approximation, it's
a property of the text encoder output that the pretrained DiT depends on
(zero slots act as attention sinks, see `anima_lora/CLAUDE.md` → "Text
encoder padding").

Today the resampler predicts all 512 slots. The inactive tail is only pulled
toward zero by a weak penalty (`zero_pad_weight=0.01`). Final
`pad_residual_mean_abs = 0.036` vs. content std ~0.15 — roughly 25% of
content energy leaks into slots that should be dead. At inference the model
has no explicit signal for where content stops, so the leak is uniform
across the tail and every leaked slot becomes "fake content" that competes
with the DiT's real attention sinks.

## Proposal

Set the resampler's learned queries to `K=256`, zero-pad to 512 only at the
boundary to the DiT / phase-2 losses.

- Resampler output: `(B, K=256, D)` — all K slots are content, trained
  directly.
- Downstream input to DiT / phase-2 losses: `F.pad(out, (0, 0, 0, 512-K))`
  → `(B, 512, D)` with a hard-zero `[256:512]` tail by construction (192
  content + 320 pad by total slot count).
- Drop `zero_pad_weight` in phase 1.5 and `w_pad` in phase 2 (both become
  identically zero).
- Drop `pad_residual_mean_abs` from eval metrics for the same reason.

## Picking K

K must cover nearly every image's actual L or we lose real tail content.

- **K=192** — clips ~17% of the dataset. Too aggressive; the 150-200
  bucket straddles the cap and every 200-410 image loses tail.
- **K=256** — clips ~5%. Aligns with phase 1.5's own eval contract:
  `_strata` in `phase1_5_anchored.py:226-232` defines `content_all` as
  `0 ≤ s < 256`. Anything past 256 is already ignored by the quality
  metric, so the architecture is implicitly betting on a 256 cap.
- **K=320** — clips ~1.6%. Safer tail, linear cost in queries.
- K=448 or 512 — no truncation, but reduces to "keep status quo".

**Recommendation: K=256.** Matches the existing eval window, halves the
resampler's query-driven cross-attn cost, and loses content only on the
tag-heaviest 5% of captions (empirically the most noise-dominated slice).

Anchor compatibility: every `default_slot` in `anchors.yaml` is ≤ 2, and
`phase1_positions.json` slots are booru-tag positions (rating / people /
solo / artist) that sit at the head of the caption — no anchor slot ever
exceeds K=256 in the current cache, so `inject_spec_anchors` survives
unchanged.

## Implementation

Three files.

1. `scripts/img2emb/resampler.py` — no change (already parametric on
   `n_slots`).
2. `scripts/img2emb/anchors.py` — `AnchoredResampler.__init__` already
   accepts `n_slots`; no change.
3. `scripts/img2emb/phase1_5_anchored.py`
   - Add CLI flag `--n_slots` (default 256).
   - Pass through to `AnchoredResampler(..., n_slots=args.n_slots)`.
   - Supervise `tgt_b[:, :K]`; mask becomes
     `arange(K) < L.clamp_max(K)` — slots `≥ K` are dropped from the
     loss entirely, not penalized.
   - Zero-pad `pred` to 512 inside `_run_pred` before writing predictions
     out, so downstream consumers keep the 512-shaped tensor they expect.
   - Remove `args.zero_pad_weight` usage; pad term is 0 by construction.
   - Remove `pad_residual_mean_abs` from `_eval_per_variant`.
4. `scripts/img2emb/phase2_flow.py`
   - Treat the resampler as a `(B, K, D)` source. Pad to 512 at the
     boundary where `ctx` is fed to the DiT.
   - Remove `pad_tail_loss` and `--w_pad`.
5. `scripts/img2emb/data.py` — `_resampler_loss`: `zero_w=0` path already
   handled; ensure callers stop passing a non-zero weight.

Loss simplification after the change:

```python
# mask_b: (B, K), active where slot < min(L, K)
active = mask_b.unsqueeze(-1).float()
mse = ((pred - tgt) ** 2 * active).sum() / (active.sum() * D).clamp_min(1)
cos = (1 - F.cosine_similarity(pred, tgt, dim=-1)) * mask_b
cos_loss = cos.sum() / mask_b.float().sum().clamp_min(1)
total = mse + cos_w * cos_loss
```

No `pad_loss`, no `zero_pad_weight`.

## Tradeoffs

| | current (S=512 w/ pad loss) | length regressor | **K=256 hard cap** |
|---|---|---|---|
| pad residual | ~0.036, uniform leak | bounded by regressor R² on caption-level L | exactly 0 |
| new parameters | 0 | ~1M (pool→L head + gate) | 0 (fewer, actually) |
| queries cost | 512 | 512 | 256 (≈ 2× faster cross-attn) |
| long-caption handling | predicts full 512 | soft gate, bounded by head's training distribution | **hard clip at 256** — 5% of images lose tail content |
| supervision | weak (0.01 pad penalty) | explicit L regression | trivially exact |
| inference behavior | leaky tail, no L signal | gate with bounded R² | identical to real text encoder for L ≤ K |

The only real loss vs. the status quo is the ~5% long-L slice. Those are
the verbose tag-dump captions; most of their tail is redundant aliases and
low-information boilerplate. The compute win and the elimination of pad
leak are strict improvements on the other 95%.

---

# Part 2 — InfoNCE over shuffled caption variants

## Problem

Every image in the cache has `V=8` T5-embedded caption variants — same
tags, different shuffle orders — stored as `crossattn_emb_v0 …
crossattn_emb_v7`. Current training
(`_ResamplerTrainDataset.__getitem__` at `data.py:258-270`) uniformly
samples **one** variant per step and treats its `(S, D)` tensor as the
supervised target.

This is a one-to-many problem collapsed to one-to-one. Each step penalizes
the resampler for producing any of the other 7 valid embeddings for the
same image. Over training this biases the prediction toward the
variant-mean — which the phase-0 docstring already flags as a bad target
(`data.py:95-98`: "shrinks the norm under triangle inequality and sits
off the T5 manifold"). Random sampling mitigates the norm-shrink but not
the underlying signal: the gradient direction on any given step ignores
that the 7 un-sampled variants are equally correct answers.

Evidence the variants are **semantically redundant but positionally
divergent**:
- The phase-1.5 eval already computes `best_over_v` (per-slot max-cos
  across variants) and `mean_over_v` (mean-cos across variants) as sanity
  envelopes around the `vs_mean` metric — they consistently bracket it,
  confirming the variant scatter is non-trivial
  (`phase1_5_anchored.py:324-326, 336-340`).
- Shuffling reorders caption tokens before T5, so per-slot cos across
  variants is noisy (same content, different positions). Pooled cos
  across variants is the more stable signal.

## Proposal

Add a **multi-positive InfoNCE auxiliary loss** on pooled embeddings. The
V variants of one image are all positives; other images' variants are
negatives.

```python
# pred:        (B, K, D)   resampler output (K=256 under Part 1)
# tgt_pooled:  (B, V, D)   pre-pooled per-variant targets, cached once
# mask:        (B, K)      active-slot mask

pred_pool = _pool(pred, mask)                    # (B, D)
pred_n = F.normalize(pred_pool, dim=-1)

tgt_n = F.normalize(tgt_pooled, dim=-1)          # (B, V, D)
tgt_flat = tgt_n.reshape(B * V, D)               # all variants flattened

sim = (pred_n @ tgt_flat.T) / tau                # (B, B*V)
pos_mask = torch.eye(B, device=sim.device).repeat_interleave(V, dim=1).bool()

# SupCon-style multi-positive NCE
log_prob = sim - sim.logsumexp(dim=1, keepdim=True)
infonce = -(log_prob * pos_mask.float()).sum(dim=1) / V
infonce_loss = infonce.mean()
```

This composes with — does not replace — the existing MSE + cos losses.
MSE/cos still anchor the per-slot reconstruction against the sampled
variant; InfoNCE adds a set-level signal that says "your pooled
prediction should live in the cone spanned by this image's variants, and
nowhere near any other image's cone."

## Design choices

**Pooling.** Mean-pool over active slots, then L2-normalize.

```python
def _pool(x, mask):        # x: (B, K, D), mask: (B, K)
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1)
```

Slot-wise InfoNCE is tempting but the shuffled-caption setup breaks it:
variant `v_j` has tag X at slot 4, variant `v_k` has the same tag at
slot 17, so positional alignment is garbage. Mean-pool discards position
and keeps the semantic summary — exactly what we want for a set-level
objective. Length-normalized mean (divide by L, not K) avoids penalizing
long captions. Under Part 1, the denominator is `min(L, K=256)`.

**Temperature.** Start at `τ=0.07` (CLIP default). Expose as
`--infonce_tau`.

**Weight.** `--w_infonce` (default 0.1). The primary signal is still MSE
per-slot; InfoNCE is a regularizer that shapes the global embedding
geometry.

**Negatives.** All `V × (B-1)` other-image variants in the batch. With
`batch_size=48, V=8` that's 376 negatives per anchor — plenty of signal
without queues or momentum encoders. If it saturates, upgrade to a small
in-RAM memory bank keyed by image idx.

**Positive aggregation.** SupCon-style (sum over positives inside the
log, divided by count). Simpler alternatives considered:
- Hardest positive only (min over V): same spirit as the min-of-K MSE
  but on the NCE loss. Rejected: risks chasing outlier variants.
- Mean-pooled-over-V single positive: collapses the V signal before the
  loss. Rejected: defeats the point — if one pooled vector per image
  were enough we wouldn't need InfoNCE.

## Implementation

Minimal diff, four files. No new dependencies.

**1. `scripts/img2emb/extract_features.py`** — add a one-time artifact:
- Compute per-variant pooled targets at feature-extraction time (or as
  a small rebuild script). For each TE file, load all `crossattn_emb_v*`,
  zero-clamp `[L:]`, mean-pool over `[:L]`, store `(V, D)` fp32. (The
  target pool uses the full active length `L`, independent of K — the
  target is what we want the resampler's pool to converge *toward*.)
- Write `output/img2embs/features/target_pooled.safetensors` with a
  single `pooled` tensor shape `(N, V, D)`. At `N=1987, V=8, D=1024,
  fp32` ≈ 65 MB — trivial to hold in RAM.
- Sidecar in `active_lengths.json` or its own meta file recording `V`
  and pool method.

**2. `scripts/img2emb/data.py`**
- `load_cache` returns `tgt_pooled` alongside existing fields (lazy-load
  the safetensors).
- New helper `_infonce_loss(pred_pool, tgt_pooled_batch, tau)` — returns
  `(loss, metrics_dict)`.
- `_ResamplerTrainDataset.__getitem__` unchanged (still returns one
  variant for MSE). The InfoNCE path reads `tgt_pooled[full_idx]` in the
  main process using the batched `full_idx` tensor — no per-worker disk
  I/O.

**3. `scripts/img2emb/phase1_5_anchored.py`**
- CLI flags: `--w_infonce` (default 0.1), `--infonce_tau` (default
  0.07).
- In the training step: compute `pred_pool`, gather
  `tgt_pooled[full_idx]`, call `_infonce_loss`, add to `total_loss`.
- Log `infonce_loss` + `infonce_acc` (fraction of anchors whose top-1
  nearest variant belongs to the correct image — cheap recall@1 sanity).

**4. `scripts/img2emb/phase2_flow.py`**
- Same two flags, same wiring. Phase 2 already pools implicitly through
  the DiT's cross-attn but that's a different loss family — InfoNCE
  here still shapes the resampler's output geometry without touching
  the flow-matching term.

---

# Rollout

1. **Ship K-cap first** (smaller diff, cleaner problem statement).
   Re-run phase 1.5 with `--n_slots 256` and existing hyperparams.
   Compare `content_all / cos_median` against the current
   `siglip2_resampler_4layer_anchored.json` baseline (`vs_mean=0.627`,
   `best_over_v`, `mean_over_v`). If regression on `tail_64_256` is
   < 0.01 cos, ship it. Otherwise bump K to 320 and rerun.
2. **Propagate K-cap to phase 2** once phase 1.5 looks clean.
3. **Add InfoNCE artifact + plumbing.** Verify `target_pooled.safetensors`
   loads and pooled targets have sensible norms.
4. **Train phase 1.5 with `--w_infonce 0.1`** vs the K-cap-only
   baseline, same step budget, same seed. Primary metric:
   `best_over_v / cos_median` on `mid_8_64` and `tail_64_256` bands —
   where the one-variant-per-step bias should hurt most. Secondary:
   `infonce_acc` should climb above `1/batch_size` quickly.
5. If phase 1.5 improves, **propagate InfoNCE to phase 2** with the
   same weight. If it regresses at `w=0.1`, sweep `{0.02, 0.05, 0.2}`
   before abandoning.

---

# Tie-in with FSQ idea

The per-image variant scatter exposed by InfoNCE is also the natural
input signal for a future FSQ / VQ-VAE regularizer:
- High intra-image pooled variance → caption distribution is genuinely
  multi-modal; loosen commitment weight on that image.
- Low intra-image variance → tight cluster; candidate for codebook seed.

Doing InfoNCE first gives us (a) a calibrated pooled-embedding geometry
to quantize later, and (b) a diagnostic (`infonce_acc`) that tells us
whether images are separable in pooled space before we spend effort on
discretization.

---

# Open questions

**K-cap**
- Should `K` live in `active_lengths.json` metadata so downstream
  consumers auto-detect the cap, or treat it as a model-architecture
  constant baked into the safetensors metadata?
- If the dataset grows and `L_max` drifts past K, is the trigger a full
  retrain with larger K, or a dataset filter that drops long-L images?
  The ~5% that already exceed 256 today answer this implicitly — we
  already accept some truncation.

**InfoNCE**
- Should `pred_pool` be computed over **only the content tail**
  (excluding anchor slots), since anchor slots are hard-written from
  frozen prototypes and don't reflect the resampler's learned geometry?
  Probably yes — bias the pool toward the part the loss can actually
  steer. Requires passing `anchor_slot_mask` into `_pool`.
- Does it help to mix in a **slot-wise InfoNCE at a few reliable anchor
  slots** (rating, people_count) where positions *are* aligned across
  variants? Likely redundant with the CE head, but worth a probe.
- For phase 2: is the InfoNCE signal still informative once the DiT is
  in the loop, or does the flow-matching gradient already enforce
  enough global geometry? Easy ablation.
