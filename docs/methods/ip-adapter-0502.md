# IP-Adapter — 2026-05-02 status & next steps

Working notes from the debugging session on `make ip-adapter` producing bad
results. Pairs with `docs/methods/ip-adapter.md` (the architecture reference).

## What we found

**The `ip_init_std=1e-4` "step 0 ≈ baseline DiT" invariant was silently
broken.** `set_ip_tokens()` was running the IP K/V through
`cross_attn.k_norm` / `cross_attn.v_norm` (RMSNorm). RMSNorm rescales its
input to unit RMS regardless of magnitude, so the small projection output
got renormalized to ~1.0 and the IP path ran at full magnitude from step 1.

Evidence from run `20260502160758` (preset `tenth`, with the new tfevents
scalars wired up):

| step | `‖to_k_ip‖.mean` | `ip_text_ratio.mean` |
|---:|---:|---:|
| 2   | 0.23 | **2.29** |
| 54  | 0.89 | **5.26** |
| 80  | 0.98 | 0.93 |
| 318 | 1.92 | 0.96 |

Ratio of 2.3 at step 2 means the IP cross-attn output was already 2.3×
larger than text cross-attn. This explains the previous run's
(`20260502140913`) `epoch_baseline_no_ip_delta` being **negative every
epoch** (−0.006 to −0.011): turning IP off improved validation because it
removed a loud, mostly-meaningless signal that the model never fully
learned to use within the budget. The model was learning to *suppress* IP,
not use it.

## What's fixed

1. **K/V RMSNorm bypass on the IP path** — `set_ip_tokens()` no longer
   calls `cross_attn.k_norm` / `cross_attn.v_norm`. Step-0 ratio is now
   `~8e-4` instead of `~2.3` (verified with a numerical smoke test).
2. **Tensorboard scalars** — `IPAdapterNetwork.metrics()` emits at step
   cadence:
   - `ip_adapter/to_k_ip_norm/{mean,max}` (always on; pure weight reads)
   - `ip_adapter/to_v_ip_norm/{mean,max}` (always on)
   - `ip_adapter/ip_text_ratio/{mean,max}` (only while runtime
     diagnostics are on; gated by `ip_diagnostics_epochs`)
3. **`ip_diagnostics_epochs = 999`** in `configs/methods/ip_adapter.toml`
   so the ratio scalar streams the whole run instead of just epoch 1.

## Cleanup before the next run

- **Delete `output/ckpt/anima_ip_adapter.safetensors`** (May 2 15:31). It
  was trained against the buggy unit-RMS K/V, so loading it through the
  fixed code gives effectively-zero IP contribution.
- Kill the still-running `20260502160758` task — it's on the buggy
  trajectory.

## Run #1 — sanity check the fix (do this first)

```bash
make ip-adapter PRESET=tenth   # ~30 min, 8 epochs of 10% data
```

Pass criteria — read straight off tensorboard:

| Scalar | Expected at step ~10 | Expected by epoch end |
|---|---|---|
| `ip_adapter/ip_text_ratio/mean` | < 0.01 | growing slowly toward 0.05–0.3 |
| `ip_adapter/to_k_ip_norm/mean` | ~0.145 | growing monotonically (no spike-and-suppress) |
| `loss/validation/epoch_baseline_no_ip_delta` | (n/a, runs at epoch end) | **≥ 0 by epoch 2**, trending positive |

If `ip_text_ratio.mean` blows past 0.3 within the first epoch the IP path
is ramping too fast — drop `ip_scale` from 1.0 to 0.5 and re-run.

If the val Δ is still negative after 4 epochs **and** the ratio is healthy
(< 0.3), the bug isn't init scaling — it's a learning-signal problem.
Move to Run #2.

## Run #2 — full-budget training (only after #1 passes)

```bash
make ip-adapter   # default preset, full data
```

Bump these in `configs/methods/ip_adapter.toml` first:

- `max_train_epochs = 30` (was 8 — current budget is order-of-magnitude
  smaller than reference IP-Adapter recipes; raise until val Δ plateaus)
- `caption_dropout_rate = 0.1` (currently 0.0 — with text always present
  the IP path has no incentive to carry conditioning)
- `ip_image_drop_p = 0.1` (currently 0.05 — half the standard CFG dropout)

## Backstop ideas if Run #2 still doesn't converge

These are next-tier fixes, in priority order. Don't reach for them until
Run #2 has plateaued, otherwise we won't know which one mattered.

### A. Warm-start from text-side `kv_proj`

Stronger init than `std=1e-4`. Each block's `cross_attn` already has a
trained fused `kv_proj` weight (the text K/V producer). Split it and copy
the K-half to `to_k_ip` and V-half to `to_v_ip` — IP starts as "another
text-cross-attn that reads image tokens" instead of from random small
weights. The model already knows how to consume context tokens through
those weights, so the optimizer doesn't have to discover the basin.

Catch: this breaks "step 0 = baseline DiT" again, so we'd need a learned
per-block scalar gate (init 0) to keep early-step behavior stable. ~30 LoC,
add an `ip_init = "kv_clone" | "near_zero"` toml knob.

### B. Reference-image augmentation

The current self-paired pipeline (`ref == target`) plus PE-Core's
high-fidelity features (vs CLIP-L's lossy features) lets the model
memorize pixels rather than learn appearance. Standard fix: random
horizontal flip + color jitter + slight random crop on the reference image
only (target stays clean). Tencent's IP-Adapter relied on CLIP's
lossiness for this; PE-Core needs explicit aug.

Lives in `prime_for_forward()` of `IPAdapterMethodAdapter` — apply
augmentation to `images` before `encode_pe_from_imageminus1to1`. **Note:**
this only works in the live-encode path. The pre-cached PE features
(`{stem}_anima_pe.safetensors`) are deterministic per-image, so the
default `ip_features_cache_to_disk=true` path can't augment. Either flip
to live encoding (slower) or pre-cache N augmented variants per image.

### C. Trainability bench

Tiny-set overfitting check: 8–16 images, train for 200+ epochs, assert
`epoch_baseline_no_ip_delta > 0.005` at the end. If the model can't even
overfit a tiny set, the architecture has a bug; if it can, the failure is
purely data + budget. Wraps `train.py` with a fixed dataset and stricter
epochs. ~50 LoC at `scripts/bench_ip_adapter.py`.

### D. Shuffled-reference validation baseline

Add a third `ValidationBaseline` alongside `no_ip`: rotate batch[k]'s
reference to come from batch[(k+1) % B]. Three failure modes pin down
where the IP path is failing:

- shuffled ≈ no_ip and matched < no_ip → IP works, just undertrained
- shuffled ≈ matched (both worse than no_ip) → IP contributes biased noise
  unrelated to the reference (resampler collapse / wrong scale)
- shuffled ≫ matched but matched still worse than no_ip → IP binds to its
  reference but the binding is memorization, not generalization → need
  augmentation (B)

Lives in `IPAdapterMethodAdapter.validation_baselines()`.

## Reference points

- Architecture deep-dive: `docs/methods/ip-adapter.md`
- Code: `networks/methods/ip_adapter.py`
- Method config: `configs/methods/ip_adapter.toml`
- Trainer integration: `IPAdapterMethodAdapter` (same file as the network)
- Validation Δ infra: `train.py:2007-2120`,
  `library/training/method_adapter.py:156-164`
