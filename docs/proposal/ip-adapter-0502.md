# IP-Adapter — 2026-05-02 status & next steps

Working notes from the debugging session on `make ip-adapter` producing bad
results. Pairs with `docs/methods/ip-adapter.md` (the architecture reference).

## What we found

**The `ip_init_std=1e-4` "step 0 ≈ baseline DiT" invariant was structurally
unreachable**, not just silently broken — and the original diagnosis
("RMSNorm rescaled the IP K/V to unit RMS") was half wrong. The DiT's
`Attention` module (`library/anima/models.py:386`) defines:

```python
self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
self.v_norm = nn.Identity()
```

This is the standard QK-norm pattern (ViT-22B, Gemma, Qwen, Flux). V is
intentionally *not* normalized — V's magnitude linearly modulates the
attention output, and forcing it to unit-RMS would destroy the model's
ability to weight tokens. So:

- The K-side bypass in the original fix was real: `k_norm` was rescaling
  IP K to unit-RMS, and skipping it does drop K-side magnitude.
- The V-side "bypass" was a no-op the whole time. `v_norm = nn.Identity()`
  means V was *never* being normalized on either path. Small std=1e-4 on
  `to_v_ip` was the only thing keeping V small.

That's not enough. With `to_v_ip` at std=1e-4 init (weight Frobenius
~0.145) and the resampler producing unit-scale tokens, even one Adam step
on `to_v_ip` grows V's weight norm to ~0.26 and ip_out scales with it.
**Empirically the IP/text ratio was already 3.7 at step 2** in the
"fixed" run `20260502161937` (vs 2.29 in the buggy `20260502160758`) —
the K-only fix made the ratio *worse* because text_result happened to be
smaller in that batch, exposing how V was driving the contribution.

Evidence from run `20260502160758` (buggy: K rescaled to unit-RMS, V at
small init) vs `20260502161937` (K-bypass fix only, V unchanged):

| @ step 2 | `to_k_ip_norm/mean` | `to_v_ip_norm/mean` | `ip_text_ratio/mean` |
|---|---:|---:|---:|
| Buggy | 0.225 | 0.261 | **2.29** |
| K-only fix | 0.148 | 0.258 | **3.76** |

V-side norm essentially identical — confirming the fix only touched K.

`epoch_baseline_no_ip_delta` stayed negative (−0.006 to −0.011) in both
runs: turning IP off improved validation because it removed a loud,
mostly-meaningless signal. The model was learning to *suppress* IP, not
use it.

## What's fixed (proper fix)

1. **Per-block learned scalar gate, init 0** (`IPAdapterNetwork.ip_gate`).
   The patched cross-attn applies `text_result + gate * scale * ip_out`,
   so step 0 ≡ baseline DiT *by construction* regardless of K/V weight
   magnitudes. The optimizer learns when (and per which block) to open
   the IP path.
2. **K/V default Kaiming init** (no longer `std=1e-4`). With the gate
   handling step-0 baseline equivalence, K/V can be at "normal"
   magnitude so the IP path carries real signal as soon as the gate
   cracks open — instead of two parallel uphill climbs (gate AND K/V
   growing from near-zero). `ip_init_std` remains a toml knob for a
   defense-in-depth run.
3. **K-side RMSNorm bypass** retained — keeps the IP-side softmax near
   uniform over the K_ip slots so ip_out averages across slots rather
   than collapsing onto one. (V-side bypass dropped from the comments;
   `v_norm` is Identity anyway, nothing to bypass.)
4. **Optional `gate_lr` override** in `network_args` — overrides the
   global LR for the gate group only. Default `None` ⇒ gate uses the
   adapter LR. Run #1 evidence (below) showed the gate is the slowest
   thing in the system at lr=1e-4, so the toml ships with `gate_lr=1e-3`
   as the recommended starting point for any non-trivial run.
5. **Tensorboard scalars** — `IPAdapterNetwork.metrics()` emits at step
   cadence:
   - `ip_adapter/to_k_ip_norm/{mean,max}` (always on)
   - `ip_adapter/to_v_ip_norm/{mean,max}` (always on)
   - `ip_adapter/ip_gate/{mean,abs_mean,abs_max}` (always on — track gate
     opening)
   - `ip_adapter/ip_text_ratio/{mean,max}` (only while runtime
     diagnostics are on; gated by `ip_diagnostics_epochs`)
6. **`ip_diagnostics_epochs = 999`** in `configs/methods/ip_adapter.toml`
   so the ratio scalar streams the whole run.

Smoke-test verified: gate=0 produces `max |out_with_ip - out_baseline| =
0.0` exactly. gate=0.1 with default-init K/V produces a meaningful
contribution (0.46 in the smoke test) — confirming K/V are no longer
"asleep" at init.

## Cleanup before the next run

The buggy/K-only-fix runs and their checkpoints have already been
superseded by Run #1's (`20260502164913`) checkpoint at
`output/ckpt/anima_ip_adapter.safetensors` (May 2 17:22), which contains
the gate keys and was trained under the proper fix. Run #2 will warm-start
from this if `network_weights` is set, or train fresh otherwise — either
is fine.

If you want to retrain Run #2 from scratch (recommended, given Run #1's
gate barely moved): just delete the May 2 17:22 checkpoint or skip the
`network_weights` line.

## Run #1 — sanity check (`20260502164913`, completed)

```bash
make ip-adapter PRESET=tenth   # 8 epochs of 10% data, lr=1e-4, gate_lr=global
```

All three pass criteria green. Headline metric, **first time positive in
this codebase's history**:

| ep | `loss/validation/epoch_baseline_no_ip_delta` |
|---:|---:|
| 1 | +0.000154 |
| 2 | +0.000165 |
| 3 | +0.000253 |
| 4 | +0.000340 |
| 5 | +0.000339 |
| 6 | +0.000335 |
| 7 | +0.000285 |
| 8 | **+0.000449** |

Δ tripled from epoch 1 to epoch 8, with the largest improvement on the
σ=0.10 (high-noise / early-step) bin (Δ_σ0.10 = +0.00085 at ep8) — the
regime where image conditioning *should* help most. Sign convention:
positive Δ means "no_ip baseline > with_ip", i.e. removing IP makes
validation worse → the model is *using* IP, not suppressing it.

Step-cadence sanity:

| Scalar | step 2 | step 1872 (final) |
|---|---:|---:|
| `ip_text_ratio/mean` | 0.0025 | 0.26 (in target band 0.05–0.3) |
| `ip_text_ratio/max` | 0.007 | 1.50 (oscillates 0.4–1.6, never escapes) |
| `ip_gate/abs_max` | 0.0002 | 0.004 (barely off init=0) |
| `to_k_ip_norm/mean` | 26.13 | 26.08 (slight shrink) |
| `to_v_ip_norm/mean` | 26.13 | 26.50 (slight grow) |

**The gate barely opened.** Peak `abs_max` ~0.004, mean ~0.001. Despite
that, val Δ kept rising — even a sliver of IP signal helped. Most of the
contribution is coming from default-Kaiming-loud K/V modulated by a
near-zero gate; the "real" IP semantics learning hasn't kicked in yet at
this LR + budget.

This is the evidence that initially motivated bumping `gate_lr`. At
`lr=1e-4` the gate is the rate-limiter on the entire system. We later
reframed this — see "Diagnostic on Run #1" below — but the data still
holds.

## Diagnostic on Run #1: PE feature analysis

Run #1's "gate barely opens, val Δ still positive" pattern was the
prompt for a feature-only diagnostic before throwing more training at
the problem (`bench/ip_adapter/pe_feature_analysis.py`, see
`bench/ip_adapter/analysis.md` for the full write-up). Three measurements
on the 2407 cached PE features in `post_image_dataset/lora/`:

1. **Aug-invariance histogram** (hflip / crop(0.60) / jitter@0.2 vs
   random other image, mean-pooled cosine):
   - hflip 0.992, jitter 0.996, crop(0.60) 0.877; cross 0.686.
2. **Crop retrieval rank** (60% random crop, index 2407): recall@1 =
   0.22, median rank 26 / 2407.
3. **Effective rank**: participation ratio 6.2, 95% energy in 46 dims.

What this tells us:

- **Memorization is NOT the failure mode.** Crop recall@1 = 0.22 says
  the resampler can't shortcut via feature lookup-from-crop, even at a
  fairly aggressive 60% crop.
- **Capacity is fine.** 95%-energy rank 46 ≫ K=16 resampler tokens.
- **The wall is narrow signal on a collapsed manifold.** Cross-pair
  pool sim 0.69 + participation ratio 6.2 says the entire dataset's
  pooled PE features sit on a ~6-dim sub-space. The IP path's
  discriminative signal is a *small per-image delta on top of a strong
  shared-aesthetic background* — exactly the regime where a slow
  optimizer can't extract the signal from a high-baseline-correlation
  input distribution.

This reframes Run #1's "gate barely opens but val Δ positive" cleanly:
the signal is real (Δ positive) but small (gate doesn't need to open
much to capture it). Pure budget/LR levers can move the gate further
but don't address the underlying narrow-signal regime.

### Detour: reference-image aug was tried and reverted

The first reflex was the standard recipe — hflip + color jitter on the
reference image (option B from the original backstops). Implemented,
then reverted on a second look. The bench measured aug invariance at
the *pool* level (~0.99 cos for hflip and jitter), which is exactly
what makes them weak augmentations in feature space: the resampler
sees nearly the same pooled signature, so aug barely changes its input.
Aug also fights pixel memorization, which the bench just established
isn't our failure mode. And adding noise to a small discriminative
signal makes it harder to extract, not easier. See
`bench/ip_adapter/analysis.md` "Why we considered, then dropped..."
for the full reasoning trail. The aug code path is gone from the tree.

## Fix: dataset-mean PE centroid (implemented 2026-05-02)

The participation-ratio-6 collapse is what we want to subtract. New
artifacts:

- `IPAdapterNetwork.ip_centroid` — `[encoder_dim]` persistent buffer,
  init zero, subtracted from every PE token before the resampler in
  `encode_ip_tokens`. Buffer ⇒ round-trips through `state_dict` ⇒
  inference inherits the same shift the network was trained with. No
  extra deployment file needed; the trained `.safetensors` is
  self-contained.
- `IPAdapterNetwork.load_centroid_from_file(path)` — populates the
  buffer from a sidecar at first-time training only. Resume / inference
  ignore the path because the checkpoint already carries the buffer.
- `scripts/compute_pe_centroid.py` — streams the cached PE files,
  mean-pools each, averages across the dataset, writes a small `[D]`
  safetensors. ~1 second over 2407 files. Output:
  `post_image_dataset/ip_adapter/anima_pe_centroid_{encoder}.safetensors`.
  The centroid is a dataset artifact, not a checkpoint artifact, so it
  lives in its own dir alongside (not inside) the shared LoRA cache.
- `network_args` knob `ip_centroid_path=...` in the toml. Default points
  at the script's output path. Comment out to disable (buffer stays
  zero ⇒ no-op, equivalent to the pre-centroid behavior).

Centroid stats on this dataset: ‖centroid‖ ≈ 22.3, std ≈ 0.69, mean ≈
−0.11. Subtracting it makes the resampler's input zero-mean per dim;
per-element variance is unchanged (~0.69 std, same as before).

### Centroid vs `ip_gate` — orthogonal, both still needed

Tempting question: with the centroid handing the resampler the
discriminative delta directly, can we drop `ip_gate`? **No** — they
solve different problems:

- `ip_gate` enforces *step-0 output equivalence* (`text_result + 0 *
  scale * ip_out = text_result`). Centering doesn't make `ip_out` zero
  at step 0 — it makes the resampler's *input* zero-mean, but
  `to_v_ip` at default Kaiming still produces a random nonzero `ip_out`
  vector that the gate has to suppress.
- Centering accelerates the *learning dynamics on the resampler input*.
  It doesn't address output magnitude.

What the centroid *might* enable: relaxing the aggressive `gate_lr`. The
100× boost was needed because Run #1's gate was the rate-limiter on
slow-signal extraction. With the resampler already aligned with the
discriminative sub-space, the optimizer has clear gradient through the
IP path and the gate may converge at the global LR. Worth A/B-testing
once Run #2 results are in.

## Run #2 — full-budget training

```bash
make ip-adapter   # default preset, full data
```

The toml as it stands has the Run #2 settings applied:

- `caption_dropout_rate = 0.05` — forces ~5% text-free batches so the
  IP path has a learning signal it can't fake via captions.
- `gate_lr = 1e-2` — 100× the global LR. With the centroid in place
  this is probably overkill; consider dropping to global (1e-4) or 5e-4
  on a follow-up run if the gate runs away.
- `ip_centroid_path=post_image_dataset/ip_adapter/anima_pe_centroid_pe.safetensors`
  — pre-built sidecar, loaded into the persistent buffer at network
  construction.

Still worth checking before launch:

- `max_train_epochs` — Run #1 was 8 epochs of 10% data (~3700 step·images
  total). Reference IP-Adapter recipes use ~10× more. Bump to 30 if val Δ
  is still climbing at the end of the default run.
- `ip_image_drop_p` — currently 0.05; standard CFG-dropout recipes use 0.1.
  Helps inference do image-CFG independently of text-CFG.

Watch during training:

- `ip_gate/abs_max` should now reach ~0.05–0.1 within the first epoch
  comfortably (centroid + `gate_lr=1e-2`). If it stays under 0.01 for a
  full epoch, the gate gradient is still starving — most likely
  paired-but-different data is needed, not more LR (option D below).
- `ip_text_ratio/max` was oscillating 0.4–1.6 in Run #1 with the gate
  near zero. With the gate ~10× higher and centroid centering, expect
  ratio max to drift up. **If it climbs steadily past 2.0, the IP path
  is running away** — first try dropping `gate_lr` to 1e-3 (centroid
  may have done the work the boost was compensating for); if it still
  runs away, set `ip_scale = 0.5` and re-run.
- `to_k_ip_norm/mean` and `to_v_ip_norm/mean`. In Run #1 these barely
  moved (gate was suppressing all gradient). With the gate opening
  faster, K/V should start drifting visibly. Monotonic growth ⇒ healthy.
  Spike-and-suppress (sharp grow then sharp shrink) ⇒ gate opening too
  fast; either drop `gate_lr` or add `ip_init_std=1e-4`.

## Backstop ideas if Run #2 still doesn't converge

These are next-tier fixes, in priority order. Don't reach for them until
Run #2 has plateaued, otherwise we won't know which one mattered.

### A. Warm-start from text-side `kv_proj`

Stronger init than default Kaiming. Each block's `cross_attn` already has
a trained fused `kv_proj` weight (the text K/V producer). Split it and
copy the K-half to `to_k_ip` and V-half to `to_v_ip` — IP starts as
"another text-cross-attn that reads image tokens" instead of from random
weights. The model already knows how to consume context tokens through
those weights, so the optimizer doesn't have to discover the basin.

The per-block `ip_gate` (already in place) handles the "step 0 = baseline"
side, so this is now a clean copy: ~30 LoC, add an
`ip_init = "kv_clone" | "default"` toml knob.

### B. Reference-image augmentation — *retired, see "Detour" above*

The standard recipe (hflip + color jitter + small crop) was implemented
and reverted. Short version: pool-level PE invariance to hflip/jitter is
too high (~0.99 cos), so aug barely changes the resampler's input;
memorization isn't the failure mode anyway (recall@1 = 0.22); and adding
noise to a small discriminative signal makes it harder to extract. The
right lever for our actual problem (narrow signal on collapsed manifold)
is the centroid subtraction now in place. Re-bench at the per-token
level if you ever want to revisit aug — the pool-level number isn't
informative for the resampler's actual input.

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
- Feature analysis: `bench/ip_adapter/pe_feature_analysis.py` +
  `bench/ip_adapter/analysis.md`
- Centroid script: `scripts/compute_pe_centroid.py`
- Centroid sidecar: `post_image_dataset/ip_adapter/anima_pe_centroid_pe.safetensors`
