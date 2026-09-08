# REPA — relational alignment against PE-Spatial

Training-time auxiliary loss that aligns a mid-block DiT feature to cached
PE-Spatial patch tokens (REPA, Yu et al., arXiv:2410.06940). It is a
loss-level regularizer, not a new adapter — it rides the LoRA network kwargs
and composes with every variant in the LoRA family (the `lora.toml` default
stack included). The trained checkpoint inferences identically to a plain run of
the same variant; REPA adds no inference-time parameters.

## Shipped configuration

The stack settled to a single operating point (Phase 0/1 closed 2026-06; DoG
made default 2026-06-28). All defaults live in `configs/methods/lora.toml`:

| Knob | Default | Meaning |
|---|---|---|
| `use_repa` | `true` | master on/off |
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

REPA-DoG replaces the older `spatial_norm` DC removal with a
difference-of-Gaussians band-pass of the target before standardize + per-token
L2-norm (`σ₁ = min(gh,gw)/16`, σ₂ off). It strips a broad low-frequency band
rather than just the DC term, which lifts target discriminability on all content
axes. DoG and `spatial_norm` are the same family (DoG at σ₁→0 is DC removal),
so they are mutually exclusive — DoG wins when on.

Both forms align a noisy-input mid-block feature to the clean-image
encoder target at every sampled σ (paper-faithful σ-conditioning), captured from
the primary forward — no second DiT forward, so no block-swap offloader desync.

> An `absolute` arm (`repa_mode = "absolute"`) — per-token cosine through a
> 3-layer `REPAHead` MLP — remains implemented for reference. It lost the Phase-0
> A/B (relational preferred ~6:4) and is not the shipped path; the head is
> training-only and stripped at save. Don't use it without re-benching.

## Quick start

```bash
# One-time prereq: cache PE-Spatial sidecars into the LoRA cache dir
# ({stem}_anima_pe_spatial.safetensors — disjoint from the PE-Core caches CMMD reads)
make preprocess-pe-spatial

# REPA is on by default in configs/methods/lora.toml — just train:
make lora
```

Missing sidecars don't crash: a batch without `repa_pe_features` skips the term
for that step (and `train.py` prints the preprocess hint at startup). If PE
features load but the block hook never fires, the adapter warns once — REPA
being silently inert is a logged condition, not a quiet no-op.

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
- DoG band-pass (`dog_standardize` in `library/training/repa.py`):
  target-side only, relational mode only — reshape to the `(gh, gw)` grid,
  `Z − LP(σ₁)` low-band strip, standardize, flatten back for the per-token
  L2-norm + Gram match. `repa_dog_sigma1_div` / `repa_dog_sigma2_div` /
  `repa_dog_norm_std` tune the band; the σ clamp in `_guarded_blur` keeps a
  small divisor from over-padding the coarse PE grid.
- Numerics: Gram / cosine computed in fp32 (caches are bf16; low-norm cosine
  is precision-sensitive).
- Loss attach: scalar returned under `aux["repa"]`, weighted by
  `losses._repa_loss` in the stage-2 registry slot (same family as `fera_fecl`).
  Active iff the factory stamped `_repa_weight > 0`.
- Gradient reach: REPA gradient only flows into LoRA modules in blocks
  ≤ `repa_layer` (by design). Remember this when reading per-block deltas.
- Config plumbing: kwargs (`use_repa` / `repa_mode` / `repa_weight` /
  `repa_layer` / `repa_encoder` / `repa_target_dog` / `repa_dog_*` /
  `repa_spatial_norm` / `repa_timestep_weighting`) are parsed in
  `networks/lora_anima/factory.py` and stashed on the network; they are
  registered in the `NETWORK_KWARGS` allowlist (`networks/__init__.py`) — any new
  key must be added there or it's silently inert.
- EasyControl: relational REPA is validated and shipped as the EasyControl
  default for cond ≠ target tasks (sanitize/colorize). The aux-loss dispatch runs
  on both the cached-LLM-adapter and in-model (`crossattn_emb=None`) text paths;
  a launch check for `repa/active=1.0` *without* `repa/align_loss` catches a
  silently-skipped term. Mechanism in `docs/experimental/easycontrol.md`.

## Guardrails (from the v1 burn)

- Never re-run v1's operating point (global pooling + weight 0.5). v1's
  documented outcome: anatomy ↑ but anime style broken (vision-encoder
  photo-prior leak).
- **Style gate is non-negotiable** — any change that visibly moves style fails
  regardless of its anatomy delta.
- Judge on sample grids first, CMMD-vs-own-PE-Core as the drift tripwire.
  Never FM val loss (uninformative on Anima), never fraction-of-Δ readouts.
- REPA does not transfer to the DP-DMD (turbo) student — relational REPA on
  the distilled student was net-negative (drift amplification, then
  over-alignment damage even after the fix). Refuted and removed; don't
  re-propose (`_archive/proposals/turbo_repa.md`, `_archive/bench/turbo_repa/`).

## References

- `library/training/repa.py` — adapter + relational loss + DoG band-pass (module
  docstring covers grid / native_flatten mechanics in depth).
- `networks/lora_anima/factory.py` (kwargs + optional head attach),
  `library/training/losses.py::_repa_loss`,
  `library/datasets/base.py::_try_load_repa_pe` (sidecar loading),
  `library/vision/buckets.py::PE_SPATIAL_B16_512_SPEC`.
- Design history (archived on closure): `_archive/proposals/repa_v2_patchwise_pe_spatial.md`,
  `_archive/proposals/repa_phase1_operating_point.md`,
  `_archive/proposals/repa_dog_target.md`; v1 implementation in `_archive/repa/`;
  probes in `_archive/bench/repa/`.
- REPA: Yu et al., arXiv:2410.06940. Relational form cf. Park et al. 2019
  (relational KD). iREPA (target-standardization / encoder choice): Singh et
  al., arXiv:2512.10794. DoG target: arXiv:2603.14645.
