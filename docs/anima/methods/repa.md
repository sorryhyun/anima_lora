# REPA — relational alignment against PE-Spatial

Training-time auxiliary loss that aligns a mid-block DiT feature to cached
PE-Spatial patch tokens (REPA, Yu et al., arXiv:2410.06940). It is a
loss-level regularizer, not a new adapter — it rides the LoRA network kwargs
and composes with every variant in the LoRA family (the `lora.toml` default
stack included). The trained checkpoint inferences identically to a plain run of
the same variant; REPA adds no inference-time parameters.

## Shipped configuration

The knobs live in `configs/methods/lora.toml`. **REPA is off there since 2026-11-06**:
at alpha 512 / lr 1.25e-5 the LoRA recipe without it won blind on two artists, 32–16
([`../findings/lora_repa_off_wins_blind.md`](../findings/lora_repa_off_wins_blind.md)).
EasyControl task configs and `configs/soup/soup.toml` still turn it on. The values
below apply with `use_repa = true`:

| Knob | Default | Meaning |
|---|---|---|
| `use_repa` | `false` | master on/off |
| `repa_mode` | `"relational"` | Gram / relational-KD form (no head) |
| `repa_weight` | `0.05` | REPA settles at ~4% of total loss at plateau |
| `repa_layer` | `8` | DiT block tapped (of 28) |
| `repa_encoder` | `"pe_spatial"` | PE-Spatial-B target |
| `repa_target_dog` | `true` | REPA-DoG target band-pass (below) |

Relational aligns `MSE(G_dit, G_pe)` on Gram matrices `G = F̂F̂ᵀ` of
per-token L2-normalized features — only *which patches are alike* transfers
(layout, part structure, anatomy); any global domain direction (the encoder's
photo prior) cancels out of the pairwise structure by construction. There is no
projection head and no dimension match.

REPA-DoG applies a
difference-of-Gaussians band-pass of the target before standardize + per-token
L2-norm (`σ₁ = min(gh,gw)/16`, σ₂ off). It strips a broad low-frequency band
rather than just the DC term, which lifts target discriminability on all content
axes. DoG and `spatial_norm` are the same family (DoG at σ₁→0 is DC removal),
so they are mutually exclusive — DoG wins when on.

Both forms align a noisy-input mid-block feature to the clean-image
encoder target at every sampled σ (paper-faithful σ-conditioning), captured from
the primary forward — no second DiT forward, so no block-swap offloader desync.

`repa_mode = "absolute"` — per-token cosine through a 3-layer `REPAHead` MLP —
is also implemented; relational was preferred ~6:4 over it in A/B. The head is
training-only and stripped at save.

## Quick start

```bash
# One-time prereq: cache PE-Spatial sidecars into the LoRA cache dir
# ({stem}_anima_pe_spatial.safetensors — disjoint from the PE-Core caches CMMD reads)
make preprocess-pe-spatial

# REPA is off in configs/methods/lora.toml — turn it on per run:
make lora ARGS="--network_args use_repa=true"
```

Missing sidecars don't crash: a batch without `repa_pe_features` skips the term
for that step (and `train.py` prints the preprocess hint at startup). If PE
features load but the block hook never fires, the adapter warns once.

## Mechanics

- Capture: forward post-hook on `unet.blocks[repa_layer]`
  (`library/training/repa.py::REPAMethodAdapter`). The hook sits on
  `block.__call__`, *outside* the compiled `block._forward`, so it fires under
  `compile_blocks()`. Under native_flatten the captured shape is the fake-5D
  `(B, 1, seq, 1, D)` rather than eager `(B, 1, H, W, D)` — both flatten to the
  same row-major `(B, N_dit, D)`, so the reshape is layout-agnostic.
- Grid match: PE-Spatial tokens live on a per-aspect-bucket `(gh, gw)` grid
  (32×32 at square). The DiT side is adaptive-avg-pooled down to that grid so
  both sides are `N = gh·gw` tokens in the same row-major order. `(gh, gw)` is
  recovered from the encoder feature's own token count, disambiguated by the
  latent's orientation.
- DoG band-pass (`dog_standardize` in `library/vision/repa.py`):
  target-side only, relational mode only — reshape to the `(gh, gw)` grid,
  `Z − LP(σ₁)` low-band strip, standardize, flatten back for the per-token
  L2-norm + Gram match. `repa_dog_sigma1_div` / `repa_dog_sigma2_div` /
  `repa_dog_norm_std` tune the band; the σ clamp in `_guarded_blur` keeps a
  small divisor from over-padding the coarse PE grid.
- Numerics: Gram / cosine computed in fp32 (caches are bf16; low-norm cosine
  is precision-sensitive).
- Loss attach: scalar returned under `aux["repa"]`, weighted by
  `losses._repa_loss` in the stage-2 registry slot (same family as
  `hydra_balance` / `soft_tokens_contrastive`). Active iff the factory
  stamped `_repa_weight > 0`.
- Gradient reach: REPA gradient only flows into LoRA modules in blocks
  ≤ `repa_layer` (by design). Remember this when reading per-block deltas.
- Config plumbing: kwargs (`use_repa` / `repa_mode` / `repa_weight` /
  `repa_layer` / `repa_encoder` / `repa_target_dog` / `repa_dog_*` /
  `repa_spatial_norm` / `repa_timestep_weighting`) are parsed in
  `networks/lora_anima/factory.py` and stashed on the network. The
  `NETWORK_KWARGS` allowlist (`networks/__init__.py`) is AST-derived from literal
  `kwargs.get("<key>")` reads in the consumer modules — a new key read any other
  way is silently not forwarded.
- EasyControl: relational REPA is the EasyControl
  default for cond ≠ target tasks (sanitize/colorize). The aux-loss dispatch runs
  on both the cached-LLM-adapter and in-model (`crossattn_emb=None`) text paths;
  a launch check for `repa/active=1.0` *without* `repa/align_loss` catches a
  silently-skipped term. Mechanism in `docs/anima/experimental/easycontrol.md`.

## Known results

- v1 (global pooling, weight 0.5) improved anatomy but broke anime style — the
  vision encoder's photo prior leaked through. The relational form cancels that
  global direction.
- Relational REPA on plain LoRA lost blind to the same recipe without it, 32–16 on
  caststation + sweetonedollar at alpha 512 / lr 1.25e-5 (s37 / s38), with the fm loss
  unchanged; the LoRA default dropped it
  ([`../findings/lora_repa_off_wins_blind.md`](../findings/lora_repa_off_wins_blind.md)).
- FM val loss does not track REPA's effect on Anima; changes are read on sample
  grids, with CMMD against the run's own PE-Core features as the drift check.
- Relational REPA on the DP-DMD (turbo) student was net-negative (drift
  amplification, then over-alignment damage); turbo has no REPA path
  (`_archive/proposals/turbo_repa.md`, `_archive/bench/turbo_repa/`).

## References

- `library/training/repa.py` — adapter; `library/vision/repa.py` — relational loss + DoG band-pass (adapter module
  docstring covers grid / native_flatten mechanics in depth).
- `networks/lora_anima/factory.py` (kwargs + optional head attach),
  `library/training/losses.py::_repa_loss`,
  `library/datasets/base.py::_try_load_repa_pe` (sidecar loading),
  `library/vision/buckets.py::PE_SPATIAL_B16_512_SPEC`.
- Design history: `_archive/proposals/repa_v2_patchwise_pe_spatial.md`,
  `_archive/proposals/repa_phase1_operating_point.md`,
  `_archive/proposals/repa_dog_target.md`; v1 implementation in `_archive/repa/`;
  probes in `_archive/bench/repa/`.
- REPA: Yu et al., arXiv:2410.06940. Relational form cf. Park et al. 2019
  (relational KD). iREPA (target-standardization / encoder choice): Singh et
  al., arXiv:2512.10794. DoG target: arXiv:2603.14645.
