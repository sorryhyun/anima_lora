# Transfer-hypothesis findings

**Status:** validated · **Date:** 2026-05-04 · **n:** 48 prompts × 2 seeds = 96 trajectories on base DiT, 24 inference steps · **Source bench:** `bench/dcw/results/20260504-1010-transfer-hyp/` · **Analysis script:** `../transfer_hypothesis_check.py`

## Headline

**LL early-vs-late per-sample correlation r = 0.989** [95% CI 0.98, 0.99] across 96 trajectories. Per-seed splits agree to 4 decimal places (seed-0 r=0.991, seed-1 r=0.987), so this is real prompt-conditional signal, not seed coupling.

Decisively above the proposal's r > 0.7 gate → **v0a (online controller) is the winning path**, no offline MLP needed.

Detail bands LH / HL / HH have r ≈ 0.5-0.6 (moderate temporal coherence) but they don't matter for DCW since it's LL-only.

See `transfer_check.md` for the full per-band, per-seed breakdown and `transfer_check.json` for the raw numbers.

## Three follow-up findings that sharpen the v0a design

### 1. Probe phase can be very short

r(K) saturates fast on the LL band (all 96 trajectories):

| K | σ at K-1 | r | covered fraction of trajectory |
|---|---|---|---|
| 1 | 1.000 | 0.358 | 4% |
| 2 | 0.958 | 0.872 | 8% |
| 3 | 0.917 | 0.936 | 13% |
| **4** | **0.875** | **0.959** | **17%** |
| 6 | 0.792 | 0.978 | 25% |
| 8 | 0.708 | 0.984 | 33% |

K=2 is already viable (r=0.87); the proposal's K=4 default is comfortably saturated. Diminishing returns past K=4.

### 2. A single per-prompt scalar α explains 94% of LL gap variance

Fit `g[i, t] ≈ α_i · μ(t)` (single amplitude per prompt against population-mean shape across trajectories): **R² = 0.942**.

The remaining 6% is per-prompt schedule shape, which v1's `(c_pool, σ_t)`-conditioned head would capture but isn't worth the complexity. **The v0a sketch's choice of one-scalar `α_prompt` (not a per-prompt schedule) is exactly right.**

K=4 estimate of α correlates 0.97 with the full-trajectory α — so v0a can lock in its amplitude with ~83% of the trajectory remaining for correction.

### 3. α range [−0.54, +4.91] explains the "scalar hurts flat styles" memory

Mean is 1.0 (by construction), std 0.97 — coefficient of variation = 1.0. About **8% of prompts have α < 0** — meaning for those prompts the scalar λ_LL = −0.015 actively *makes the gap worse* (their underlying gap has the opposite sign). This operationalizes the `project_dcw_when_to_use` failure mode.

v0a sign-flips the correction for those prompts automatically (`λ_LL = α_prompt · λ_scalar · (1−σ)` with α_prompt < 0 inverts the sign); the scalar can't.

## Implications for the proposal

The proposal's preferred path (base-model-scoped + v0a) is unambiguously the right one. **Effort estimate drops from 3-5 days to ~2 days.** Specifically:

Deliverables that can be deleted:
- **Step 4 (head module + train loop)** — no MLP, no training.
- **Step 6 (head training in the validation harness)** — no head to train.
- **Most of step 2** — only the population-mean `‖v_fwd‖(σ)` profile is needed, not full `(gaps, slopes, lambda_targets, pooled_embeds)` arrays.

What remains:
- A 24-float `‖v_fwd‖(σ)` reference profile per base DiT (~100 bytes).
- `OnlineDCWController` module (probe / fit / lambda_for).
- Inference-side wiring for `--dcw_online`, `--dcw_online_disable`, `--dcw_online_probe_steps`.
- The head-quality gate (still needed — perceptual A/B vs scalar on 4 LoRAs spanning style range, including a flat-style adapter where v0a's sign-flip on α<0 prompts is the headline test).

## Caveat worth checking before locking in v0a

The R² = 0.94 single-amplitude fit assumes *population-mean* μ(t) is a good template. The cross-LoRA hydra@{0.5, 1.0} runs showed the same μ(t) shape (linearity error <1% on late-half steps), but a meaningfully different LoRA family / training distribution could shift the shape.

Cheap to verify on the next 2-3 LoRAs: re-run `measure_bias.py --dump_per_sample_gaps --lora_weight <path>` and check (a) the early-late correlation still > 0.9, (b) the per-prompt α distribution still spans both signs, (c) the population-mean μ(t) shape still matches base ±5%. If any LoRA fails (a) or (c), v0a's reference profile may need to ship per-LoRA — back to the proposal's per-LoRA fallback branch. (b) is the validation that v0a actually does something the scalar can't.

## How to reproduce

```bash
# 1. Bench (≈35 min on 5060 Ti at n=48 × 2 seeds)
uv run python archive/dcw/measure_bias.py \
    --dit models/diffusion_models/anima-preview3-base.safetensors \
    --n_images 48 --n_seeds 2 --infer_steps 24 \
    --dump_per_sample_gaps \
    --label transfer-hyp

# 2. Analysis (seconds; reads gaps_per_sample.npz)
uv run python archive/dcw-learnable-calibrator/transfer_hypothesis_check.py \
    --npz bench/dcw/results/<TS>-transfer-hyp/gaps_per_sample.npz \
    --n_seeds 2 \
    --out_dir bench/dcw/results/<TS>-transfer-hyp/
```

For a LoRA replication add `--lora_weight <path> --lora_multiplier 1.0` to step 1.
