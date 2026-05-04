# Per-LoRA DCW Calibration

**Status:** proposal (post LL-only finding 2026-05-03) · **Author scope:** reading A from the [DCW design discussion](../methods/dcw.md) · **Effort:** small (~½ day) · **Cost at inference:** zero · **Reference paper:** Yu et al., *Elucidating the SNR-t Bias of Diffusion Probabilistic Models* (arXiv:2604.16044)

## Summary

Auto-tune DCW per checkpoint instead of shipping a single global `(λ_LL, schedule_LL)`. The validation-time bias metric (`library/training/bias_metric.py`) already measures the per-step `||v_rev|| − ||v_fwd||` gap that DCW is designed to close. With a closed-form solver on the LL band's gap, training emits an **optimal LL-DCW recipe** as part of the LoRA's safetensors metadata. Inference reads it back automatically — `--dcw` works out of the box, and any CLI flag still wins over the saved recipe.

This is a **calibration-only** proposal. No new losses, no architectural changes, no extra training cost. The whole thing is one validation pass + a closed-form solve at end-of-training.

## Update vs the original proposal

The first draft assumed a 4-band recipe (`{LL, LH, HL, HH}` × `(λ, schedule)` = 8 knobs). The 2026-05-03 band-mask sweep showed this is over-parameterised: **LL is a causal lever, the other three bands are downstream symptoms**. Applying LL-only correction at step `i` propagates through the DiT's nonlinear forward and tightens all four band gaps at later steps (`docs/methods/dcw.md` §"LL-only correction"). HH-only correction is perceptually a no-op or actively harmful.

Recipe space therefore collapses to **2 numbers**: `(λ_LL, schedule_LL)`. Storage payload is trivial (a JSON-serialised dict of two scalars per checkpoint). Calibration runs on one band's anchors, not four.

## Why per-LoRA

The SNR-t gap factors into two sources (paper §1, [findings.md §6](../../archive/dcw/findings.md)):

| Source | LoRA-dependent? |
|---|---|
| (a) Solver discretization (Euler/ER-SDE) | No — pure math |
| (b) Network prediction error | Yes — LoRA *is* the network |

`(b)` is the per-LoRA part. Direction, magnitude, and σ-shape of the LL gap will drift with the adapter applied — the LoRA's training signal does not see SNR-t bias as an explicit objective, so its impact on the bias is incidental and adapter-specific. Open question §6 in `findings.md` flags this and never resolved it. We have the infrastructure to settle it.

The 2026-05-03 finding narrows the question. The bench on the base DiT showed LL gap = −317 with `(1−σ)`-shape and λ_LL ≈ −0.010 closes it to −225. **What we don't know:** whether LL gap shape and optimal λ_LL transfer across LoRAs, or whether each adapter shifts the LL signature meaningfully. That's exactly what per-LoRA calibration would settle.

## Mechanism

### 1. Per-band bias measurement during validation

Already wired in `archive/dcw/measure_bias.py` (offline bench) — port the **LL-band branch** to `library/training/bias_metric.py` (validation-time, runs under `--validation_bias_metric`).

`measure_bias_trajectory` returns a dict with `v_fwd`, `v_rev`, `gap`, `sigmas`. Add `gap_LL` (length `num_steps`) keyed by the Haar `LL` subband — same `haar_band_norms` helper, ~10 lines, no other bands needed for the recipe. (Optionally log all four bands for diagnostic, but only LL feeds the solver.)

### 2. Closed-form λ_LL solver

For schedule `w(σ)`, the optimal scalar that closes the late-half LL gap (under the linear-response assumption already used in `compute_optimal_lambda`) is:

```
λ_LL* = − Σ_{i ≥ N/2}  w(σ_i) · gap_LL(i) · s_LL(i)
        ────────────────────────────────────────────
              Σ_{i ≥ N/2}  w(σ_i) · s_LL(i)²
```

where `s_LL(i) = ∂gap_LL(i) / ∂λ_LL` is estimated from a 2-point finite-difference probe (one extra reverse rollout, λ_LL ∈ {0, −ε}). Total cost: ~one extra validation pass. The schedule choice is pre-committed to `one_minus_sigma` based on `findings.md §3` and the LL gap envelope (LL bias is monotone-negative across late steps, no sign flip → `(1 − σ)` matches without overshoot).

Output is a 2-scalar recipe:

```python
recipe = {"lambda_LL": -0.0150, "schedule_LL": "one_minus_sigma"}
```

(Numbers come out of the calibration. Example value reflects the global LL default from the 2026-05-03 magnitude sweep; per-LoRA calibration is expected to spread around this value.)

### 3. Save recipe into the LoRA's safetensors metadata

Safetensors files already carry a metadata dict (e.g. `ss_network_dim`, `ss_network_alpha` from the existing checkpoint flow). Add one more key:

```
ss_dcw_recipe = '{"lambda_LL": -0.0150, "schedule_LL": "one_minus_sigma"}'
```

Stored as a JSON string for forward-compat (older `inference.py` versions just ignore the key). Written by `library/training/checkpoint.py` on save; computed in the same end-of-training hook that already runs final validation.

A second key, `ss_dcw_calibration`, records the calibration provenance — git SHA, val sample count, baseline LL gap, residual LL gap after applying the recipe — so a downstream consumer can sanity-check and reproduce.

### 4. Inference reads the recipe automatically

`inference.py --dcw` picks up the recipe from the loaded LoRA(s):

- **Single LoRA:** use that LoRA's `ss_dcw_recipe` directly. Sets `--dcw_band_mask=LL`, `--dcw_lambda=lambda_LL`, `--dcw_schedule=schedule_LL`.
- **Stacked LoRAs:** there is no theoretical answer for combining recipes; do the simple thing — average `lambda_LL` across LoRAs, weight by their `--lora_weight` multipliers. Schedule mismatches: pick the most common (`one_minus_sigma` will dominate). Mark this regime with a stderr warning so users know they're in extrapolation territory.
- **No recipe in metadata:** fall back to the global LL default (post-magnitude-sweep value, see `docs/methods/dcw.md` §"LL-only correction").

CLI flags still win:
```
--dcw_lambda -0.005                # explicit scalar override
--dcw_band_mask LL                 # already needed by the band-mask landing change
--dcw_schedule one_minus_sigma     # explicit schedule override
--dcw_disable_per_lora_recipe      # ignore metadata, use globals
```

This keeps the calibration opt-in/opt-out and forward-compatible. A user who doesn't trust auto-calibration can disable it with one flag. The `--dcw_band_mask` flag from the LL-only landing change is reused — no new CLI surface for the per-LoRA path.

## Decision gates

Don't ship this without empirical evidence per-LoRA tuning matters. Concretely:

| Outcome | Action |
|---|---|
| Cross-LoRA `λ_LL*` variance > ±30% across ≥4 LoRAs | **Proceed.** LL signature is LoRA-specific; auto-calibration earns its keep. |
| Cross-LoRA `λ_LL*` clusters within ±15% | **Shelve.** Bias is base-DiT-dominated; ship the global LL default from the magnitude sweep and call it done — same perceptual win at zero training-side complexity. |
| Calibrated `λ_LL*` **worsens** late-half LL gap on ≥1 of 4 LoRAs | **Investigate.** Linear-response assumption breaking down for that LoRA; either shelve or move to a 3-point grid search instead of closed-form. |

Test set for the gate: 4 LoRAs spanning style range — flat (channel/caststation), detail-dense (general illustration), painterly, plus the base DiT itself as control. Each LoRA's calibration runs on the same 4-image × 2-seed bench config used in the 2026-05-03 sweep.

## Costs

| Pipeline | Added cost |
|---|---|
| Training (per validation pass) | +1 reverse rollout (one extra anchor at λ_LL=−ε) ≈ doubling val cost when `--validation_bias_metric` is on, **only on the final validation pass** (not every val). |
| Training (otherwise) | Zero. The recipe solver runs in seconds at end-of-training. |
| Storage | ~80 bytes of metadata per checkpoint (down from the original 200 bytes — single band, two scalars). |
| Inference | Zero. The metadata read is constant-time; the correction is the same DWT+iDWT+add already required by the LL-only landing change. |
| Cognitive | One extra knob (`--dcw_disable_per_lora_recipe`) for users who want the legacy behavior. |

The recipe-solve cost concentrates on whatever LoRA training run wants the calibration. Users training without `--validation_bias_metric` see zero overhead.

## What this is not

- **Not a training loss.** Reading B from the design discussion (`L_bias = |gap(i)|` added to the FM-MSE loss) requires backprop through reverse rollouts and is out of scope. If A succeeds, B becomes a follow-up; if A fails, B is unlikely to work either.
- **Not LoRA distillation of DCW.** Reading C (teacher = base DiT + DCW, student = LoRA without DCW, MSE-distill the trajectory) is a real engineering project on the order of APEX. Out of scope.
- **Not a wavelet adapter.** The per-band correction lives at the sampler boundary using a 4-line Haar DWT helper that's already in place. No new module, no `pytorch-wavelets` dep, no `networks/dcw.py`. The full paper-style adapter (Fig. 2 wavelet branch with iDWT in the model loop) remains shelved per `findings.md §4.2` until the calibration evidence justifies it.

## Risks and caveats

1. **Solver floor.** Even a perfectly-tuned recipe cannot drive the gap to zero — source (a) is solver-side. We can estimate the floor with a single 200-step rollout reference; if the floor is large vs. the achievable gap reduction, per-band tuning gives diminishing returns and may not be worth the metadata complexity.

2. **Val set size.** The calibration is fit on however many val samples the user trained with. A small val set means a noisy recipe that may overfit to those specific images. Recommend a minimum (≥8 images × 2 seeds × 12 steps) before writing the recipe; below that, fall back to defaults and emit a warning.

3. **CFG mismatch.** The bench measurement and the val measurement both run conditional-only (`guidance_scale=1`). Production inference uses CFG. CFG could amplify or dampen the bias, and the calibrated recipe might not be optimal at CFG-on. Mitigations:
   - Run the calibration at production CFG when feasible (slower; needs an unconditional embed cached).
   - Or accept that the recipe is "CFG=1 optimal, CFG-N approximate" and document the approximation.

4. **Stacked LoRAs.** The "average per-band λ weighted by multiplier" rule is a heuristic, not a theorem. It might be worse than no recipe in some compositions. The `--dcw_disable_per_lora_recipe` escape hatch is the safety net.

5. **Recipe drift across re-runs.** Two training runs of the same LoRA with different seeds may produce slightly different recipes. As long as the spread is within the ±15% noise band, this is fine; if not, the calibration is unstable and we should fall back to defaults regardless.

## Concrete deliverables

In order. Note that **deliverables 0a–0c (the LL-only landing change) are blockers** — the per-LoRA path consumes the `--dcw_band_mask=LL` infrastructure they install. Tracked in `docs/methods/dcw.md` §"Codebase changes needed".

0a. **Land LL-only DCW.** Extend `networks/dcw.py::apply_dcw` with `bands=`, port `haar_dwt_2d` / `haar_idwt_2d`, update `inference.py` CLI, thread `--dcw_band_mask` through `library/inference/generation.py`. Default: `LL`, λ from the magnitude sweep. *(Prerequisite. Per-LoRA proposal does not start until this is in.)*

0b. **Update findings.md.** Append §8 with the 2026-05-03 per-band finding so future calibration sessions don't re-litigate.

0c. ~~Run the magnitude sweep.~~ **Done 2026-05-03** (`bench/dcw/results/20260503-2124-ll-magnitude/`). New global default: `λ_LL = -0.015`, `schedule = one_minus_sigma`. See `docs/methods/dcw.md` §"Re-tuning λ for LL-only" for the per-step / per-band table.

Then the per-LoRA path proper:

1. **LL bias metric in training.** Port `haar_band_norms` (LL only) from `archive/dcw/measure_bias.py` to `library/training/bias_metric.py`. Extend the trainer's val logging to emit `loss/validation/bias_*/gap_LL` per-step (one extra scalar per existing logged step).

2. **Closed-form λ_LL* solver.** Refactor `archive/dcw/measure_bias.py:compute_optimal_lambda` into a reusable module under `library/inference/dcw_calibration.py`. New top-level fn: `solve_recipe_LL(gap_LL, sigmas, schedule="one_minus_sigma") -> dict` returning `{"lambda_LL": float, "schedule_LL": str}`.

3. **Recipe writer in checkpoint save.** End-of-training hook in `train.py` (call site: post-final-validation, pre-save). Runs the solver and adds `ss_dcw_recipe` + `ss_dcw_calibration` to safetensors metadata. Skipped silently when `--validation_bias_metric` was off.

4. **Recipe reader in inference.** `library/inference/adapters.py` extracts the recipe at LoRA load. `library/inference/generation.py` consumes `(λ_LL, schedule_LL)` and forces `band_mask=LL` (overridable on CLI). CLI flags wire through.

5. **Validation harness.** Bench script `bench/dcw/calibrate_per_lora.py` that loads N LoRAs, runs each through the recipe solver, dumps a comparison table of `λ_LL*` per LoRA. This is the artifact that decides the gate.

6. **Doc updates.** `docs/methods/dcw.md` gains a "Per-LoRA Calibration" section. This proposal moves to `docs/methods/lora-dcw.md` if the gate passes; gets archived under `archive/proposals/` if it fails.

Steps 1–4 are roughly half a day of work each. Step 5 (the gate) is the slow one — needs 4 trained LoRAs. With the recipe space collapsed to 2 scalars, the proposal stack is meaningfully simpler than the v0 sketch.

## Open questions

- Should the recipe be tied to the **method** (LoRA / OrthoLoRA / T-LoRA / HydraLoRA / ReFT) as well as the trained checkpoint? Plausible answer: no — the bias is downstream of the network forward, and method-specific structure should already be reflected in the per-LoRA `λ_LL*`. But worth one ablation row: does HydraLoRA's per-block expert routing produce a qualitatively different LL gap shape vs. classic LoRA at matched rank?
- Is LL correction stable when LoRA weight is below 1.0 (e.g. inference-time `--lora_weight 0.6`)? The recipe is calibrated at multiplier=1.0; a multiplier sweep at inference might benefit from a multiplier-aware rescaling. Out of scope for v0; flag for v1.
- Could `λ_LL*` be derived without a reverse rollout entirely — e.g. from the LoRA's weight-norm spectrum or from a per-block contribution analysis? Speculative; would change "calibration" to "prediction" and remove the rollout cost. Worth thinking about if cost becomes a bottleneck, which it currently isn't.
- Does **schedule** also drift per LoRA, or is `one_minus_sigma` universal? The 2026-05-03 finding pinned the schedule on the base DiT; per-LoRA the LL gap envelope might shift such that `(1−σ)²` or step-clipped variants do better on some adapters. Cheap to test (add a 4th item to the solver's schedule list); only worth doing if step-1 cross-LoRA results suggest the LL gap shape varies meaningfully, not just its magnitude.
