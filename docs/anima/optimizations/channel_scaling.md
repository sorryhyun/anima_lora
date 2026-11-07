# Channel Scaling — SmoothQuant-style LoRA pre-scaling

Per-channel input magnitude rebalancing for LoRA-family adapters — a training-time optimizer-geometry feature; after `bake_inv_scale` at save time it is invisible to inference. Absorbs a calibrated per-channel scale `s[c] = (mean|x[c]|)^α` into `lora_down` columns and applies `x / s` at forward, so the adapter output is unchanged at init but Adam's effective per-channel step no longer favors the DiT's DC-bias outlier channels.

> For the motivation (DC-bias outlier channels in the frozen Anima DiT, decomposition into "register-token sinks" vs "stable outlier features", and the GraLoRA alternative weighed against), see `_archive/bench/channel_stats/channel_dominance_analysis.md`. This doc is the usage reference.

## Quick start

On by default:

```toml
# configs/base.toml
channel_scaling_alpha = 0.5
```

Sole user knob:

- `0.0` — disabled. No `inv_scale` buffer; identical to a stock LoRA module.
- `0.5` — sqrt balance (SmoothQuant default; the shipped default).
- `1.0` — fully flatten per-channel input magnitude.

Two vendored calibrations ship in-tree (so deploys — including `custom_nodes/*/_vendor/` trees — work without a separate download):

- `networks/calibration/channel_stats.safetensors` (~3.5 MB) — main-stream LoRA family.
- `networks/calibration/cond_channel_stats.safetensors` (~2.2 MB) — EasyControl cond stream (see below).

No regeneration is needed unless the base DiT changes.

## Mechanism

At network build (`networks/lora_anima/factory.py::_load_channel_scales`):

1. Load the per-module `mean|x|` vector for every adapted Linear that has a calibration entry.
2. Compute `s = (mean|x|.clamp_min(1e-6))^α`, mean-normalize so `s.mean() == 1`.
3. Pass `s` into each `LoRAModule` / `HydraLoRAModule` / `StepExpertLoRAModule` constructor.
4. `BaseLoRAModule._register_channel_scale` mutates the down-projection weight (`down[:, c] *= s[c]`) in place and registers `inv_scale = 1 / s` as a persistent fp32 buffer.

At forward (`networks/lora_modules/{lora,hydra,step_expert}.py`):

```python
x_lora = self._rebalance(x)   # x * inv_scale, in the activation dtype
lx = F.linear(x_lora, self.lora_down.weight.to(x_lora.dtype))
```

The forward is bit-equivalent at init (`(x · s⁻¹) @ (down · s)ᵀ == x @ downᵀ`); the *only* effect is on optimization geometry, and that effect is Adam-specific. The per-column gradient is divided by `s[c]` (verified numerically: per-column grad-norm ratio matches `1/s` with corr 1.0000), and since Adam takes ~uniform per-element steps regardless of gradient scale, each channel's per-step output leverage changes from `mean|x[c]|` to `mean|x[c]|^(1-α)`. At α=0.5, dominance (max/mean leverage) on the shipped calibration drops from ~13–80× to ~3.8–9.3× (`mlp.layer1` worst); `cross_attn.kv` (T5-sourced) is already clean at 1.8× so the rebalance is a near no-op there. AdamW's decoupled weight decay is a uniform per-coordinate shrink, so the reparameterization doesn't interact with it.

`inv_scale` is calibration data — it carries no gradient and never updates during training. It rides through `state_dict` as a persistent buffer.

What it does **not** touch: the `lora_up` gradient. `∂L/∂up = Σ_t g_t h_tᵀ` depends only on the adapter latent `h = down·x`, which the reparameterization leaves bit-identical, so the up-side effect GraLoRA describes (the outlier channel's column of the full-fine-tune gradient entering every element of `up` through `down[:, c]`, then smeared over all input columns by `Δup·down`) is untouched. On MOSS-Image this up-side term is large and has its own fix (`dc_isolate`, on by default): `docs/moss/channel_scaling.md`.

Dead-channel caveat (standard SmoothQuant wart): channels with near-zero `mean|x|` get a gradient boost of up to `1/s ≈ 325` (main `mlp.layer2`) / 648 (cond stream). Adam's step does not depend on the gradient's size, so those `lora_down` columns random-walk at `lr/s` in raw coordinates. Their output is ≤ 1e-4 of the LoRA's, but on the 676-update soup shards they held 33–43 % of the saved ‖ΔW‖² (5 % on a 256-step single-artist run), which skews every tool that reads ‖ΔW‖ (`dw_cos.py` energy shares, `merge_loras.py --normalize global`, `match_dw`). `channel_scaling_floor = τ` (network arg, 0 = off) clamps `mean|x|` at `τ · median` before the power; τ = 0.01 bounds a dead channel's boost at 10× a median channel's and takes the dead-column share to 0.000.

## Liveness: which variants this actually affects

The scale is absorbed into the down-projection, and **whether that tensor is trainable decides everything** (2026-06-10 audit, verified numerically per variant). Every current LoRA-family variant carries a trainable down-projection, so channel scaling is live wherever it's wired:

| Variant | Absorbed into | Effect |
|---|---|---|
| LoRA | trainable `lora_down` | Live |
| HydraLoRA (shared-A) | trainable shared `lora_down` | Live |
| step_expert (turbo) | trainable per-step `lora_down` | Live |
| EasyControl cond LoRA (`_LoRAProj`) | trainable `lora_down` | Live |
| T-LoRA mask | — (rank axis) | Orthogonal; composes with any row above |

`configs/methods/lora.toml` (plain LoRA, `down_init = "weight_svd"`) and EasyControl are Live.

## EasyControl cond stream (`cond_channel_stats.safetensors`)

The EasyControl cond LoRA (`networks/methods/easycontrol.py::_LoRAProj`) sees clean reference latents, not the noisy main stream, so its activation profile diverges from the main calibration: cosine(main, cond) is median 0.93 but p10 0.71 / min 0.56, and `output/calibration/cond_stream_profile.json` shows reusing the main file would make `mlp.layer2` dominance *worse* (negative transfer efficiency — block-7 layer2: 28× raw → 104× with the transferred scale). Hence the separate cond-specific calibration, wired 2026-06-09 and loaded by `_load_cond_channel_scales` with the same α from `channel_scaling_alpha`. Regenerate with `scripts/calibration/cond_stream_profile.py`.

Unlike the LoRA family (see save/load below), the cond stream has no save-time bake — `inv_scale` ships in the EasyControl checkpoint as a persistent buffer. External consumers that aren't `inv_scale`-aware (e.g. the standalone `~/ComfyUI-EasyControl-KSamplerCompat` node) will mis-scale a channel-scaled checkpoint; in-repo inference pre-allocates and loads it correctly.

## Save / load

`inv_scale` is included in `state_dict` next to `lora_down.weight` / `lora_up.weight` / `alpha`. The fused-attn split/refuse round-trip (`networks/lora_anima/loading.py`) treats it as a shared tensor — q/k/v of the same Linear see the same input so they share `inv_scale`. Hydra handles it via its per-expert stacked layout.

At save time `bake_inv_scale` (`networks/lora_modules/lora.py`) folds `inv_scale` into `lora_down` and drops the key, so the on-disk checkpoint is a plain LoRA any consumer reproduces bitwise without knowing the convention; `LoRANetwork._reabsorb_baked_inv_scale` re-splits on resume. `merge_to` and `LoRAModule.get_weight` undo absorption before computing `up @ down`, so a baked DiT (`make merge`) is correct without the calibration file.

Round-trip is covered by `tests/test_per_channel_scaling_roundtrip.py`.

## Regenerating the calibration

```bash
python scripts/calibration/analyze_lora_input_channels.py --per_artist \
    --dit models/diffusion_models/anima-base-v1.0.safetensors \
    --dump_channel_stats networks/calibration/channel_stats.safetensors
```

The script registers `forward_pre_hook` on every `nn.Linear` in the DiT, accumulates per-input-channel `sum|x|` and token count over a small batch of cached samples at 5 flow-matching sigmas, then writes one `mean|x|` vector per LoRA-target Linear. 16 samples × 5 sigmas saturates the calibration in practice; `--per_artist` (71 samples on the current dataset) broadens coverage without changing per-group dominance numbers meaningfully. The σ grid and image content barely matter: calibration is σ-grid-insensitive and content-agnostic (`docs/anima/findings/channel_stats_content_independence.md`) — the dominance structure is weight/architecture-driven.

Only regenerate when:

- The base DiT weights change (different normalization → different DC-bias channels).
- The set of adapted Linears widens (new attention/MLP layers exposed by the trainer).

See `scripts/calibration/README.md` for the script flags and the output JSON layout.

## When this helps more

The bench analysis confirms the precondition (20–100× per-channel dominance, DC-bias not attention sinks) holds on the current Anima DiT, and the 2026-06-10 audit confirms the gradient geometry changes exactly as designed on every Live variant. For standard flow-matching LoRA training the sample-quality delta has not been A/B-measured. The one A/B that exists is on turbo (DP-DMD) distillation, where α=0.5 was clearly net-negative (melted faces, washed/oversaturated color vs a clean α=0 twin, 2026-06-16) — turbo therefore defaults to α=0.0. Recalibrating the scale on the turbo rollout does not rescue it (turbo's per-channel input profile matches the shipped calibration at cosine 0.996), and α=0.01 is a numerical no-op (dominance 6.14→6.01); the surviving explanation is that the Adam-geometry reparam is only safe under a stationary MSE target, not a co-training critic. The `down`-side leverage change is rank-independent, so GraLoRA's "helps more at high rank" prediction does not carry over. For the standard path the regime that should see the largest payoff is:

- Shared-A across experts (HydraLoRA) — the shared `lora_down` amplifies the bias K-fold.
- Long single-domain runs — the column-imbalanced gradient compounds over more steps.

## References

- Xiao et al., SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models, ICML 2023 — the original `s = mean|x|^α` absorption trick (this implementation borrows the parameterization, not the quantization goal).
- Dettmers et al., LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale, NeurIPS 2022 — the "outlier features" phenomenon the bench observes in the Anima DiT.
- Jung et al., GraLoRA: Granular Low-Rank Adaptation for Parameter-Efficient Fine-Tuning, NeurIPS 2025, arXiv:2505.20355 — the paper that prompted the input-channel analysis. Its mechanism is the `up` gradient (`∂L/∂B = ∂L/∂Y Xᵀ A`), which this reparameterization does not reach (see "What it does not touch" above); the `k×k`-block structure is the open alternative for that side, not a more invasive version of this one.

## Code

- `networks/lora_anima/factory.py::_load_channel_scales` — calibration load + α-exponentiation.
- `networks/lora_modules/base.py::_absorb_channel_scale` — in-place column scaling + `inv_scale` registration; `_rebalance` applies `x * inv_scale` at forward.
- `networks/methods/easycontrol.py::_load_cond_channel_scales` / `_LoRAProj` — cond-stream wiring.
- `networks/lora_anima/loading.py` — q/k/v split/refuse handling for `inv_scale`.
- `tests/test_per_channel_scaling_roundtrip.py` — save → load → rebuild forward-equality check.
- `scripts/calibration/` — calibration scripts (`analyze_lora_input_channels.py`, `cond_stream_profile.py`), README and dominance analysis.
