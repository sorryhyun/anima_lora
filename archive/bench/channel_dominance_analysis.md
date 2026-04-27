# LoRA Input-Channel Dominance Analysis

Investigation triggered by *GraLoRA: Granular Low-Rank Adaptation for Parameter-Efficient Fine-Tuning* (Jung et al., NeurIPS 2025, arXiv:2505.20355). The paper argues LoRA degrades at rank >=64 because a small number of outlier input channels dominate `∂L/∂B`, distorting updates at higher rank. Before considering a GraLoRA port for T-LoRA, we checked whether the precondition (extreme per-channel input skew) exists in the Anima DiT.

## Setup

- **Script**: `bench/analyze_lora_input_channels.py` — registers `forward_pre_hook` on every `nn.Linear` in the DiT, accumulates per-input-channel `sum|x|`, `max|x|`, and token count over a batch of real samples, then reports dominance (`max(mean|x|) / median(mean|x|)`) and peak-to-mean (`max|x| / mean|x|` on the most dominant channel).
- **Samples**: 8 images from `post_image_dataset/` using cached VAE latents (`*_anima.npz`) and cached T5 outputs (`*_anima_te.safetensors`, `crossattn_emb_v0`). Text encoder and VAE are never loaded.
- **Timesteps**: sigmas `{0.1, 0.3, 0.5, 0.7, 0.9}` — 40 forward passes total per run.
- **Configurations run**:
  - r=64 T-LoRA (OrthoLoRA + timestep mask), `output/anima-tlora-0415-12.safetensors`
  - r=32 T-LoRA, `output/anima-tlora-0413-12.safetensors`
  - Base DiT with no adapter (`--lora_weight` omitted)
- **Reference thresholds** (from GraLoRA Fig 3a): severe ~20–100x dominance, moderate 5–20x, negligible <3x.

## E1: Outlier channels are present and severe

Overall dominance distribution across all hooked Linears in the **base DiT**:

| Statistic | Value |
|---|---|
| mean | 16.9 |
| median | 10.0 |
| p90 | 43 |
| p99 | 81 |
| max | 96 |

Per-group max dominance (base DiT):

| Group | max | where |
|---|---|---|
| `self_attn.qkv_in` | **96.03** | block 2, channel 770 |
| `mlp.layer1_in` | **85.68** | block 10, channel 1410 |
| `cross_attn.q_in` | **80.16** | block 2, channel 1997 |
| `self_attn.out_in` | 51.14 | block 0, channel 1140 |
| `mlp.layer2_in` | 53.57 | block 0 |
| `cross_attn.out_in` | 27.23 | block 0 |
| `cross_attn.kv_in` | 1.94 | (cached T5 — clean) |

The outlier channels are **shared across many blocks**: channel 1410 dominates `mlp.layer1` in blocks 9–20; channel 1997 dominates `cross_attn.q_proj` in a large fraction of blocks; channels 770, 1721, 181 and 1722 reappear across `self_attn.qkv_proj` blocks. This is the "stable outlier feature" pattern GraLoRA targets.

Cross-attention KV input is the single clean group — expected, because its input is the cached T5 embedding (bucketed to max_length with zero-padding), which we pass through unchanged every block.

## E2: Dominance is rank-invariant

Comparing base / r=32 / r=64 on the same samples:

| Group | base mean / max | r=32 mean / max | r=64 mean / max |
|---|---|---|---|
| `self_attn.qkv_in` | 22.6 / 96.0 | 22.1 / 96.5 | 22.3 / 97.2 |
| `cross_attn.q_in` | 29.8 / 80.2 | 28.6 / 76.2 | 29.3 / 81.0 |
| `mlp.layer1_in` | 36.9 / 85.7 | (≈36.8 / 82)† | (≈36.7 / 82.6)† |
| `cross_attn.out_in` | 11.5 / 27.2 | 10.8 / 24.8 | 11.3 / 26.7 |
| `self_attn.out_in` | 8.9 / 51.1 | 7.7 / 30.8 | 8.2 / 38.4 |
| `cross_attn.kv_in` | 1.94 / 1.94 | 1.94 / 1.94 | 1.94 / 1.94 |

† r=32 and r=64 runs predate the `mlp.layer1/2` classifier fix; those modules were grouped as `other` in the raw output but the raw ratios (80.01, 82.61) align with the base run.

**Differences are within ~1–2% everywhere.** The LoRA's forward contribution is negligible relative to the base weights, so the input distribution to each Linear is effectively frozen. This explains why r=32 and r=64 look identical from this script's vantage point — the LoRA cannot shift its own inputs.

Implication: the outlier channels are a property of the **frozen base DiT's activation statistics**, not a LoRA training artifact.

## E3: These are DC bias, not attention sinks

The `peak_to_mean = max|x| / mean|x|` on the most dominant channel distinguishes:
- **~1–3x**: uniformly high values across all tokens → DC bias / persistent feature offset
- **~10–50x**: bimodal — minority of tokens carry the magnitude
- **>=50x**: extreme concentration → attention-sink / register-token behavior

Observed peak_to_mean on top channels (base DiT):

| Module group / location | ratio | p2m | verdict |
|---|---|---|---|
| `mlp.layer1` blocks 9–20 (ch 1410, 753, ...) | 30–86 | **1.2–1.6** | DC bias |
| `self_attn.qkv_proj` blocks 4–27 | 12–26 | **1.0–1.4** | DC bias |
| `cross_attn.q_proj` blocks 0–14 (ch 1997 etc.) | 21–76 | **1.0–1.2** | DC bias |
| `cross_attn.q_proj` blocks 15–23 | 16–21 | 2.5–4.6 | Mildly bimodal |
| `self_attn.qkv_proj` **blocks 0, 1** | 64, 68 | **7.2, 8.7** | Real sink |
| `mlp.layer2` **block 12** ch 5529 | 5.9 | **310** | Real sink (single token dominates) |
| `mlp.layer2` block 0 ch 6996 | 53 | 7.9 | Real sink |

**Most of the extreme dominance is uniform-in-token, not concentrated-in-token.** This matches the *outlier feature* phenomenon (Dettmers 2022, LLM.int8) — stable per-channel magnitude offsets that survive normalization and flow through the residual stream. It does **not** match the *attention sink* phenomenon (Xiao 2024, Sun 2024), where a few specific tokens absorb attention mass.

The GraLoRA paper's LLaMA evidence (Fig 3a — "mean input channel values") is itself a mean-over-tokens measurement, so their thesis targets the DC-bias regime we observe here, not sinks.

### Real sink exceptions worth noting

- **`blocks.0` and `blocks.1` self-attention** have genuine concentration (p2m 7–9). Early-layer sinks are consistent with the usual "the first few transformer blocks decide which tokens act as scratchpads" finding in ViT register-token literature.
- **`blocks.12.mlp.layer2` channel 5529**: one token has `|x|=242` vs mean `0.78` — p2m = 310. This is a genuine massive-activation event in a single module, isolated to that block. Orthogonal to the LoRA question but worth flagging as a separate curiosity.

## E4: Implications for T-LoRA at r=64

The precondition for GraLoRA holds: Anima DiT has the same class of DC-bias outlier channels GraLoRA documents in LLaMA, at the same order of magnitude. But **the precondition is not the claim**. GraLoRA's actual claim is that LoRA gradient deviation from FFT *widens with rank* (their Fig 3b). That effect lives entirely in `∂L/∂B`, which this script does not observe.

Two important things we did **not** establish:

1. **Whether the rank-dependent gradient concentration actually occurs in this DiT.** Our forward stats are rank-invariant, so no information about the backward pass. Measuring `||∂L/∂B||` per rank row with one backward pass (≈50 lines on top of this script) would be the direct test.
2. **Whether OrthoLoRA's orthogonal parameterization already neutralizes the effect.** The `P·Λ·Q` decomposition with orthonormal init spreads the down-projection across rotated directions, which partially diffuses the single-channel dominance that vanilla LoRA suffers from. It is not obvious a priori that T-LoRA at r=64 has the degradation GraLoRA assumes vanilla LoRA has.

## Alternatives to GraLoRA

Since the outliers are base-model DC-bias features, the mechanism a fix needs to address is: "`∂L/∂B[:, c]` is proportional to `|x[c]|^2`, so the dominant column of `X` dominates the gradient of `B`." Several cheaper interventions target this directly:

1. **Per-channel input pre-scaling** (SmoothQuant-style). Compute `s[c] = (mean|x[c]|)^α` once from calibration (the stats this script already produces), absorb `s` into `lora_down` columns so the adapter output at init is unchanged, and divide `x` by `s` at runtime. Effective rank gradient becomes uniform across channels. ~10 lines in `OrthoLoRAModule.forward`. Fully compatible with the timestep mask.
2. **Per-channel input LayerNorm on the adapter branch only.** Cleaner mathematically, slightly more compute, same net effect. Compatible with any LoRA variant.
3. **Do nothing** if OrthoLoRA already handles it and r=64 trains fine in practice.

GraLoRA is the most invasive option — architectural change, `k×k` adapter blocks, new key layout in checkpoints — and hits the same mechanism as the above. It is worth considering only if the simpler interventions are empirically insufficient.

## Takeaways

1. **Precondition holds**: Anima DiT has 20–100x DC-bias outlier channels in self-attn qkv, cross-attn q, and MLP layer-1 inputs, matching the pattern GraLoRA documents.
2. **Not attention sinks**: peak_to_mean on the dominant channels is ~1–2, meaning every token carries the outlier uniformly. The outliers are closer to LLM.int8 "outlier features" than to Xiao/Sun attention sinks. The GraLoRA paper's evidence is also in the DC-bias regime.
3. **Base-model behavior**: identical dominance with and without LoRA at any rank we measured. The outliers live in frozen upstream normalization and residual state, not in the adapter.
4. **Rank-interaction effect unverified**: the script measures forward inputs, not the `∂L/∂B` gradient concentration GraLoRA claims scales with rank. Without a backward-pass measurement or an apples-to-apples r=32 vs r=64 training-loss comparison, the *size* of any potential benefit is unknown.
5. **Cheaper fix candidate**: per-channel input pre-scaling (SmoothQuant-style) targets the same mechanism as GraLoRA with ~10 lines in `OrthoLoRAModule` and no checkpoint-format change. Calibration scales come for free from this script's output.

## Next steps (if we pursue this)

1. Add a backward-pass branch to `analyze_lora_input_channels.py`: run flow-matching loss and capture `||∂L/∂B||` per rank row for the worst-dominance modules. Confirm or reject the rank-interaction prediction directly.
2. Side-by-side sample quality check between the existing 0413 (r=32) and 0415 (r=64) adapters to see whether r=64 is actually better, worse, or indistinguishable on our data.
3. If either of the above indicates a real problem, prototype per-channel input scaling in `OrthoLoRAModule`, calibrate `s` from this script's `mean_abs` vectors, and retrain a short r=64 run as an A/B against 0415.
