# DCW — Post-Step SNR-t Bias Correction

Training-free, sampler-level correction that closes the SNR-t bias of flow-matching DiTs by mixing each Euler step's `prev_sample` toward (or away from) the model's `x0_pred`.

Paper: [Elucidating the SNR-t Bias of Diffusion Probabilistic Models](https://arxiv.org/abs/2604.16044) (Yu et al., CVPR 2026)

**Read first:** `archive/dcw/findings.md`. The paper's bias direction does not reproduce on Anima — Anima's λ is **negative**, opposite the paper. Everything below assumes you've internalized that.

## Anima form

```
denoised   = latents − σ_i · v                       # x0_pred (FLUX velocity convention)
prev       = Euler/ER-SDE step                        # prev_sample
prev      += λ · (1 − σ_i) · (prev − denoised)        # DCW correction
```

Defaults: `λ = -0.010`, schedule `one_minus_sigma`. Both come from a perceptually-aligned wide sweep + closed-form fit on a narrow sweep — see `archive/dcw/findings.md` and `archive/dcw/plan.md §3`.

### Why λ < 0

Yu et al.'s Key Finding 2 (`||v_θ(x̂_t)|| > ||v_θ(x_t_fwd)||`) does **not** reproduce on Anima — the inequality is reversed at every late step, integrated signed gap −405.6 on the 24-step baseline. Paper-form positive λ widens `|gap|` on Anima; closing the gap requires negative λ. Speculative mechanism (manifold-mismatch readout) is in `archive/dcw/README.md §"Observed on Anima"`.

### Why `(1 − σ)` schedule

The bias is concentrated at low σ on Anima — `gap` is small around σ=0.5 and grows to ≈−64 by σ=0.04. The paper's `σ_i` decay would put correction in the wrong place; `const` overcorrects mid-trajectory and sign-flips the gap by step 15 (visible as over-smoothing). `(1 − σ)` weights late steps heaviest, matches the bias envelope, and dominated the 8-prompt visual panel.

## Quick start

```bash
make test-dcw                           # latest LoRA + DCW (defaults baked in)
make test-spectrum-dcw                  # Spectrum + DCW composed
```

Or add `--dcw` to any `inference.py` invocation:

```bash
python inference.py --dcw                                      \
    --dcw_lambda -0.010                                        \
    --dcw_schedule one_minus_sigma                             \
    ...  # other inference args
```

## CLI

| Flag | Default | Notes |
|------|---------|-------|
| `--dcw` | off | Enable post-step correction. |
| `--dcw_lambda` | `-0.010` | Negative on Anima — see findings. |
| `--dcw_schedule` | `one_minus_sigma` | One of `one_minus_sigma`, `sigma_i`, `const`, `none`. |

The final step (`σ_{i+1} == 0`) is always skipped — at that step `prev == x0_pred` exactly, so DCW would be a no-op modulo float rounding, and the `(1−σ)` weight is near 1 there so a numerical residual could otherwise nudge the final latent.

## Composition

DCW lives at the sampler boundary, not inside any module — it composes with everything below.

| Composes with | How |
|---|---|
| `--sampler er_sde` | Applied post-`er_sde.step`. |
| `--tiled_diffusion` | Applied to post-merge latents, not per-tile (tile boundaries should not see independent corrections). |
| `--spectrum` | Applied at the same sampler-step site on both actual-forward and cached-step branches. On cached steps, `x0_pred = latents − σ_i · noise_pred` carries Spectrum's prediction error; the correction is bias-agnostic so this is fine, but worth one ablation row in any tuning sweep. |
| `--lora_weight` / Hydra / OrthoLoRA / T-LoRA / ReFT / postfix | Orthogonal — no module patching, no extra weights. |

Untested at v0:
- CFG (the bench is conditional-only; one CFG-on baseline at integration time covers it).
- APEX (APEX trains around the bias; one ablation row when next APEX checkpoint is on hand).
- Stacked LoRA / OrthoLoRA / T-LoRA / ReFT (one row per family, not v0 blocking).

## Calibration recipe

The default `λ=-0.010` was derived from two independent estimates that agreed to 2 sig figs (perceptual winner of a wide sweep + closed-form fit on a narrow sweep). To re-tune for a different checkpoint / CFG-on / on a LoRA stack:

1. Run `archive/dcw/measure_bias.py --dcw_sweep --report_optimal_lambda` with at least 3 distinct λ values (baseline + 2 nonzero is enough).
2. Read `λ*` from the printed line — it computes the `(1−σ)`-weighted least-squares optimum from per-step response slopes:
   ```
   s_i  = ∂gap/∂λ                            (finite-diff from any 2 anchors)
   w_i  = (1 − σ_i)
   λ*   = − Σ w_i · g_i · s_i  /  Σ w_i · s_i²       (over i ≥ N/2)
   ```
3. Confirm with one more sweep at `{λ*−ε, λ*, λ*+ε}`.

## Overhead

Two pointwise ops per step (`denoised` is hoisted out of the ER-SDE branch — one extra subtract on the Euler branch — plus one fused `add+scale`). No DWT, no allocation, no cross-tile communication. Negligible vs the DiT forward.

## Limitations / open questions

- `s_i` flips sign mid-trajectory (positive through steps 12–21, strongly negative in the last 2–3). The `(1−σ)` schedule mostly compensates by under-weighting the mid range; a more concentrated schedule like `(1−σ)²` or step-clipped `max(0, 1−4σ)` would isolate the bias-closing region better. Tracked as plan deliverable #11 — bench, not v0 blocking.
- Cached-Spectrum `x0_pred` is biased by the Chebyshev forecaster's prediction error. Empirically should still help (correction is bias-agnostic). Worth one explicit ablation row.
- Sign-flip vs the original paper is unresolved. Three speculative mechanisms in `archive/dcw/README.md`; cleanest test (smaller / pixel-space DiT) is out of scope.
