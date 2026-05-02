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

This is the evidence that motivated the `gate_lr=1e-3` default. At
lr=1e-4 the gate is the rate-limiter on the entire system.

## Run #2 — full-budget training

```bash
make ip-adapter   # default preset, full data
```

The toml as it stands has the Run #2 settings already applied:

- `caption_dropout_rate = 0.1` — forces ~10% text-free batches so the
  IP path has a learning signal it can't fake via captions.
- `gate_lr = 1e-3` — 10× the global LR, so the gate opens fast enough
  to actually exercise the IP path within the training budget.

Still worth checking before launch:

- `max_train_epochs` — Run #1 was 8 epochs of 10% data (~3700 step·images
  total). Reference IP-Adapter recipes use ~10× more. Bump to 30 if val Δ
  is still climbing at the end of the default run.
- `ip_image_drop_p` — currently 0.05; standard CFG-dropout recipes use 0.1.
  Helps inference do image-CFG independently of text-CFG.

Watch during training:

- `ip_gate/abs_max` should reach ~0.05–0.1 within the first epoch at
  `gate_lr=1e-3`. If it stays under 0.01 for a full epoch, the gate
  gradient is starving (likely a data-pairing problem, not LR).
- `ip_text_ratio/max` was oscillating 0.4–1.6 in Run #1 with the gate
  near zero. With the gate ~10× higher, expect ratio max to drift up.
  **If it climbs steadily past 2.0, the IP path is running away** — set
  `ip_scale = 0.5` (cuts effective ratio in half) and re-run. If it
  oscillates at a higher level but stays bounded (e.g. 1.5–2.5), that's
  fine; the optimizer is balancing.
- `to_k_ip_norm/mean` and `to_v_ip_norm/mean`. In Run #1 these barely
  moved (gate was suppressing all gradient). With the gate opening
  faster, K/V should start drifting visibly. Monotonic growth ⇒ healthy.
  Spike-and-suppress (sharp grow then sharp shrink) ⇒ gate opening too
  fast; either drop `gate_lr` to 5e-4 or add `ip_init_std=1e-4`.

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
