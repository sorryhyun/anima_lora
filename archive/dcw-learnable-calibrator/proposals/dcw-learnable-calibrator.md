# DCW Learnable Calibrator (LL-only, per-prompt)

**Status:** proposal — both empirical gates passed 2026-05-04 (transfer hypothesis: r=0.989 → v0a chosen; cross-LoRA invariance: max |Δ_LL| = 5.12% on 2 LoRA families → base-DiT-scoped) · **Author scope:** successor to [Per-LoRA DCW Calibration](./lora-dcw-proposal.md) · **Effort:** small (~2 days, all build) · **Cost at inference:** one closed-form fit at step K, then a scalar multiply per step · **Reference paper:** Yu et al., *Elucidating the SNR-t Bias of Diffusion Probabilistic Models* (arXiv:2604.16044)

## Summary

Replace the per-LoRA *scalar* `λ_LL` (the just-landed `ss_dcw_recipe` recipe) with an **online per-prompt amplitude** `α_prompt`, fit at inference time from K early-step probe observations against a base-DiT reference profile. Same DCW correction operator (`apply_dcw`, `bands={LL}`), same `(1−σ)` schedule, but the amplitude becomes a per-prompt scalar instead of a constant per checkpoint. The reference profile is ~100 bytes shipped alongside the base DiT; no MLP, no offline training step, no per-LoRA artifact under the cross-LoRA invariance hypothesis.

This is a **calibration-replacement** proposal. No change to the DCW operator, no change to LoRA training. The original v0 (offline MLP head conditioned on `c_pool`) was retired after the transfer-hypothesis check showed an early-vs-late per-sample LL gap correlation of r=0.989; v0a captures the same prompt-conditional signal with no learned weights.

## Why the scalar recipe is suboptimal

The 2026-05-04 band-variance bench (`bench/dcw/results/20260504-0930-band-variance/`) measured per-Haar-band cross-sample dispersion of the LL gap on the base DiT (n=8 images × 2 seeds × 24 steps, baseline λ=0):

| Band | Σ\|μ\| | mean σ | SNR=Σ\|μ\|/Σσ |
|---|---|---|---|
| **LL** | 362.1 | 15.19 | **0.99** |
| LH | 160.8 | 2.48 | 2.71 |
| HL | 166.5 | 2.36 | 2.94 |
| HH | 123.7 | 2.36 | 2.18 |

LL — the band the recipe optimizes — has SNR≈1: cross-sample dispersion is *the same size* as the mean signal. The LH/HL/HH detail bands are 2–3× more consistent. So a single λ_LL fit to the population mean is fitting a quantity that varies almost as much across prompts as the quantity itself.

This matches the qualitative observation already captured in memory ([`project_dcw_when_to_use`](../../../.claude/projects/-home-sorryhyun-anima-anima-lora/memory/project_dcw_when_to_use.md)): DCW helps complex images, hurts flat artist styles. A scalar can't gate on prompt intent. A per-prompt amplitude can.

The detail bands' high SNR justifies the existing LL-only scope: HL/LH/HH already transfer well across samples, so a global zero (or fixed constant) on those is fine. **All per-prompt capacity should be spent on LL.**

## Why now (and not as v0 of `lora-dcw`)

The per-LoRA scalar shipped because (a) it's cheap and (b) we hadn't yet measured cross-sample variance — only cross-LoRA variance was on the table. The 2026-05-04 finding moved the dominant uncertainty inside a single LoRA. The scalar recipe is not wasted: it remains the population-mean amplitude (`λ_scalar` in v0a's `λ_LL = α_prompt · λ_scalar · (1−σ)`), the probe-phase fallback during steps 0..K, and the natural fallback when no controller artifact is present.

## Cross-LoRA invariance finding (2026-05-04)

Validated against two LoRA families spanning {MoE, regular} × {multi-task, single-style}, plus an intra-Hydra multiplier sweep (full report: [`archive/dcw-learnable-calibrator/lora-transfer/findings.md`](../../archive/dcw-learnable-calibrator/lora-transfer/findings.md)). All configurations land cleanly in the proposal's top decision-gate branch — **max |Δ_LL| = 5.12%, well inside the ±15% bin**:

| config | LoRA | mult | LL signed gap | Δ_LL vs base | per-step LL shape r | branch |
|---|---|---|---|---|---|---|
| base | — | — | −355.49 | — | — | — |
| **artist@1.0** | `output/ckpt-artist/anima_lora_sincos.safetensors` (regular LoRA, artist style) | 1.0 | −356.33 | **−0.24%** | **r = 0.99999** | base-DiT scope |
| hydra@0.5 | `anima-hydra-0502-4812.safetensors` (HydraLoRA MoE) | 0.5 | −361.75 | −1.76% | r = 0.99999 | base-DiT scope |
| hydra@1.0 | `anima-hydra-0502-4812.safetensors` (HydraLoRA MoE) | 1.0 | −373.68 | −5.12% | r = 0.99992 | base-DiT scope |

LH / HL / HH bands shift by ≤1.5% on every config. Cross-sample LL SNR stays near 1.0 (base 0.99 → 0.99 ≤ snr ≤ 1.04 across all configs). Per-step LL gap *shape* correlation against base is r ≥ 0.99992 in all cases — five nines.

### Two sub-findings

1. **Multiplier linearity (HydraLoRA mult=0.5 vs interpolation between mult=0 and mult=1).** At mult=0.5 the LL gap sits within 0.80% of base-magnitude of the perfect linear interpolation. Late-half per-step max relative residual is 1.07%. **MoE routing is σ-conditioned, not multiplier-conditioned**: at fixed (σ, x_t) the same experts activate at any multiplier and `LoRA_out(mult) = mult · Σ_e route_e · ΔW_e · x` is exactly linear in mult by construction.

2. **Cross-family invariance (artist regular LoRA vs HydraLoRA at mult=1.0).** The artist LoRA perturbs the bias signature ~20× less than HydraLoRA (Δ_LL = −0.24% vs −5.12%). Probable mechanism: the artist LoRA shifts the *style decoder*, not the SNR-bias-generating attention path. This is the key result for the [`project_dcw_when_to_use`](../../../.claude/projects/-home-sorryhyun-anima-anima-lora/memory/project_dcw_when_to_use.md) failure mode — the base-DiT reference profile applies to flat-style adapters; v0a's per-prompt α fit handles individual flat-style prompts via sign-flip.

### Operational consequences

- **One reference profile per base DiT release** (`<base_dit_name>_dcw_vfwd.safetensors`, ~200 B), inherited by every LoRA. No per-LoRA artifact.
- **No multiplier-aware rescaling at inference.** Even mult=1.0 hydra is well inside the ±15% bin; lower multipliers interpolate linearly.
- **The per-LoRA fallback path stays in the resolution chain** but is expected to be unused — preserved for one-flag safety if a future LoRA family fails the gate.
- **Future cross-LoRA checks can drop multiplier sweeps** — one mult=1.0 measurement per LoRA is enough.

### What this finding does NOT cover yet

- **OrthoLoRA / T-LoRA / ReFT.** Different parameterizations might inject bias differently (e.g. T-LoRA's σ-conditioned rank could couple with the SNR-bias-generating attention path).
- **Painterly / detail-dense LoRAs.** Both checked LoRAs have very small bias-signature deltas; a LoRA designed to *enhance* detail could push the LL gap further (positive coupling instead of zero).
- **Image-conditioned methods (IP-Adapter / EasyControl).** Flagged for v2 since they change the network forward more meaningfully than weight-space adapters.
- **Multipliers > 1.0.** Sub-linearity at mult=0.5 was concave (0.8% above the line); at mult=1.5 it might bite harder.

The deliverable step 1 stands open for one more LoRA from a structurally different parameterization (T-LoRA / OrthoLoRA / ReFT) as a robustness margin, but the load-bearing question is settled.

## Transfer finding (v0a chosen, 2026-05-04)

The reverse trajectory is autoregressive: `x̂_{i+1} = x̂_i + (σ_{i+1} − σ_i) · v_rev(x̂_i, t_i)`. A sample whose early-step velocity deviates from the population mean carries that deviation into `x̂` for every subsequent step. The expected consequence: per-sample LL gap at early steps is highly correlated with per-sample LL gap at late steps. **Validated on n=48 prompts × 2 seeds = 96 trajectories from `post_image_dataset/lora/` on the base DiT** (`bench/dcw/results/20260504-1010-transfer-hyp/`; full analysis in [`archive/dcw-learnable-calibrator/transfer-analysis/findings.md`](../../archive/dcw-learnable-calibrator/transfer-analysis/findings.md)):

| Quantity | Value | Implication |
|---|---|---|
| Early-vs-late LL gap correlation `r` | **+0.989** [95% CI 0.98, 0.99] | Decisively > 0.7 → v0a (online controller) is the chosen path. |
| Spearman ρ (rank stability) | +0.980 | Trajectory ordering is preserved early → late, not just amplitude. |
| Per-seed split (seed-0 / seed-1) | 0.991 / 0.987 | Real prompt signal, not seed coupling. |
| Single-amplitude R² (`g[i,t] ≈ α_i · μ(t)`) | **0.942** | A single per-prompt scalar captures 94% of LL variance — v0a's one-scalar form is sufficient; v1's per-prompt schedule is not needed. |
| K=2 / K=3 / K=4 / K=6 r | 0.87 / 0.94 / 0.96 / 0.98 | K=4 default is comfortably saturated; K=2 already viable. |
| Per-prompt α range | [−0.54, +4.91] (mean 1.0, std 0.97) | ~8% of prompts have α<0 — the population-mean λ actively *hurts* them. v0a sign-flips the correction automatically; the scalar can't. |

The α<0 finding operationalizes the [`project_dcw_when_to_use`](../../../.claude/projects/-home-sorryhyun-anima-anima-lora/memory/project_dcw_when_to_use.md) memory: the scalar regresses on flat artist styles because their underlying gap has the opposite sign to the population mean.

### v0a controller (chosen architecture)

At inference, the controller runs in three phases:

1. **Steps 0..K (probe phase, K=4):** apply DCW with `λ_LL = λ_scalar` (the per-LoRA scalar recipe — fallback amplitude). Measure observed `‖v_rev‖_LL(σ_i)` per step. No extra DiT forwards — `v_rev` is already computed to take the step.
2. **Step K (fit phase):** subtract the 24-float base-DiT reference `‖v_fwd‖_LL(σ)` from the observed `‖v_rev‖_LL` to get per-step `g_obs(i)`; fit `α_prompt = Σ_{i<K} g_obs(i)·μ_g(σ_i) / Σ_{i<K} μ_g(σ_i)²` (closed-form least-squares against the population-mean gap profile `μ_g`). Microseconds.
3. **Steps K+1..N (run phase):** apply DCW with `λ_LL(σ_i) = α_prompt · λ_scalar · (1 − σ_i)`.

Reference artifact is just 24 floats per quantity (`vfwd_ref[i]` and `mu_g[i]`, ~200 B total) shipped as `<base_dit_name>_dcw_vfwd.safetensors`. No MLP, no per-LoRA artifact (under the cross-LoRA invariance hypothesis), no offline training step.

### Why the reference profile is `‖v_fwd‖`, not matched-x₀

At inference we have no x₀ for the running prompt, so we can't compute the matched gap directly. We compare observed `‖v_rev‖` against the population-mean `‖v_fwd‖`. This works iff cross-sample variance in `‖v_fwd‖(σ)` is small relative to per-prompt gap signal (so reference noise doesn't dominate the slope fit). To-be-confirmed by dumping `‖v_fwd‖_LL(σ)` mean+std from the same bench (cheap — re-uses existing infrastructure; the per-sample arrays are already in `gaps_per_sample.npz` for the gap side, and `‖v_fwd‖` is one extra column in `measure_bias.py`'s accumulator).

### Costs vs the original v0

| | v0 (MLP head, retired) | v0a (online controller, chosen) |
|---|---|---|
| Storage per base DiT | ~260 KB | ~200 bytes (2 × 24 floats) |
| Calibration cost | ≥100 prompts × 24 steps + ~10 s of head training | Same bench run; only the population-mean profiles are kept |
| Inference per step | 1 MLP forward | 0 (one scalar multiply per step + one closed-form solve at step K) |
| Inference one-time | ~5 ms head load | <1 ms (read ~200 B) |
| Per-prompt adaptation latency | Immediate | After step K (≈ 17% of trajectory at K=4) |
| Failure mode | Head overfits or generalizes poorly to OOD prompts | Probe steps run with scalar λ; if scalar is bad on this prompt, those K steps inherit that — same risk as today's recipe |

## Mechanism

### 1. Controller state

Per-prompt state across the denoising loop:
- `K` (probe-phase length, default 4 — see `K=4 r=0.96` in the transfer finding)
- `λ_scalar` (per-LoRA `ss_dcw_recipe` value, fallback amplitude during the probe phase)
- `vfwd_ref ∈ ℝ^{N=24}`, `mu_g ∈ ℝ^{N=24}` — base-DiT population-mean profiles, loaded once at inference start
- `vrev_observed ∈ ℝ^K` (filled during the probe phase)
- `α_prompt ∈ ℝ` — fit at step K, used thereafter

### 2. The closed-form fit at step K

Per-prompt LL gap is well-modelled by a single amplitude scalar against the population-mean shape (R² = 0.942 across n=96 trajectories):

```
g_obs(i) = ‖v_rev‖_LL(σ_i) − vfwd_ref[i]                # what the controller sees
g_obs(i) ≈ α_prompt · μ_g(σ_i)                          # the v0a model
α_prompt = Σ_{i<K} g_obs(i)·μ_g(σ_i) / Σ_{i<K} μ_g(σ_i)²    # least-squares closed-form
```

Apply at i ≥ K: `λ_LL(σ_i) = α_prompt · λ_scalar · (1 − σ_i)`. The scalar `λ_scalar` retains its role as the population-mean amplitude; `α_prompt` is the per-prompt rescaling. `α_prompt` is clipped to `±5` by default (the empirical range was [−0.54, +4.91] across 96 trajectories) so an outlier early-step can't blow the run.

### 3. Reference profile artifact

Per-base-DiT artifact `<base_dit_name>_dcw_vfwd.safetensors` (~200 B):
- `vfwd_ref` — population-mean `‖v_fwd‖_LL(σ_i)`, n_steps floats
- `mu_g` — population-mean LL gap per step, n_steps floats
- metadata: base DiT path, n_steps, flow_shift, calibration n, git SHA

Per the §"Cross-LoRA invariance" hypothesis the same artifact serves all LoRAs; the per-LoRA fallback is a sibling `<lora_name>_dcw_vfwd.safetensors` if any LoRA fails the invariance check.

### 4. Calibration set (one-shot, base-DiT-scoped)

Reuses `post_image_dataset/lora/` (TE-cached + VAE-cached). We only need the *population-mean* profiles, so the n requirement is far lower than v0's: ≥48 prompts × 1 seed is enough to pin the means to <2% (the n=48 × 2 transfer-hyp bench's mean shifted <3% from the n=8 base bench). Recommended n=64-100 for headroom; ~10-20 minutes on a 5060 Ti.

If the cross-LoRA invariance check fails for some adapter, per-LoRA calibration kicks in for that adapter using the LoRA's own training captions.

### 5. Inference integration

`inference.py --dcw` resolution order, extending the existing chain in `library/inference/dcw_calibration.py`:

1. **Per-LoRA controller** (sibling `<lora_name>_dcw_vfwd.safetensors`): use this LoRA's reference profile. Only present when the cross-LoRA invariance check failed for this adapter.
2. **Base-DiT controller** (`<base_dit_name>_dcw_vfwd.safetensors` shipped with the DiT): default path under the cross-LoRA invariance hypothesis.
3. **Per-LoRA scalar recipe** (`ss_dcw_recipe` metadata): use scalar (current behavior).
4. **Global default** (`DEFAULT_LAMBDA_LL`): use scalar.

CLI:

```
--dcw_online_disable             # ignore controller, use scalar fallback
--dcw_online_probe_steps K       # override K (default 4)
--dcw_online_alpha_clip 5.0      # |α_prompt| safety clip (default 5)
```

Stacked LoRAs: same heuristic as the scalar recipe — average reference profiles weighted by `--lora_weight` multipliers, OR (cheaper) pick the profile from the LoRA with the largest multiplier. v0: pick-largest, with stderr warning. The scalar-recipe averaging path stays available as a fallback when no controller artifact is present.

The DCW operator (`networks/dcw.py::apply_dcw`) does not change. Only the call site that today reads `lambda_LL` from the recipe gets two new branches: probe-phase (use `λ_scalar`, record `‖v_rev‖_LL`), then fit-phase (compute `α_prompt`), then run-phase (use `α_prompt · λ_scalar · (1−σ)`).

## Decision gates

### Cross-LoRA invariance gate — passed 2026-05-04

The branch table below was the gate's pre-commitment. Both observed LoRA families (HydraLoRA MoE at mult ∈ {0.5, 1.0} + regular artist LoRA at mult=1.0) landed in the top branch with max |Δ_LL| = 5.12% — see §"Cross-LoRA invariance finding". The proposal is therefore configured as **base-DiT-scoped, no multiplier rescaling**. The other two branches stay in the doc as the resolution chain's structural fallback (re-runs if a future LoRA family fails the gate).

| Outcome | Action |
|---|---|
| All LoRA × multiplier configurations match base-DiT SNR profile within ±15% AND λ_LL* within ±15% **(observed 2026-05-04)** | **Base-DiT-scoped calibration.** One reference profile per base DiT release; no per-LoRA artifact. Default proposal scope. |
| Cross-LoRA λ_LL* spread is 15–30% but multiplier-monotone (e.g. linear in mult) | **Base-DiT controller + multiplier-aware rescaling at inference.** Reference profile scaled by a 1-arg function of `--lora_weight`. ~10 lines of inference code, no extra calibration. |
| Cross-LoRA λ_LL* spread is >30% OR per-band SNR profile shifts qualitatively | **Per-LoRA fallback.** Calibrate each LoRA individually; ship as `<lora_name>_dcw_vfwd.safetensors`. The base-DiT controller still ships as the universal fallback. |

### Controller-quality gate (runs after, on the chosen scope)

Don't ship without empirical evidence the controller beats the scalar. Held-out 64-prompt eval split (held out *before* the calibration profile is computed):

| Metric | Threshold | Action |
|---|---|---|
| Mean residual integrated \|gap_LL\| (controller vs scalar) | ≥20% reduction | Ship the controller. Win. |
| Mean residual integrated \|gap_LL\| (controller vs scalar) | within ±10% | **Shelve.** Controller is just the scalar with extra steps; ship the scalar. |
| **Per-sample std** of integrated residual \|gap_LL\| (controller vs scalar) | ≥30% reduction | Strong ship signal — controller is doing the variance-reduction job, not just shifting the mean. The α<0 / sign-flip subset is where this is most visible. |
| Perceptual eval on 4 LoRAs spanning style range (flat/painterly/dense + base DiT control), 12 prompts each, side-by-side vs scalar | ≥60% prefer controller, no LoRA where controller loses badly (≤30%) | Required gate. The `project_dcw_when_to_use` evidence is exactly that the scalar regresses on flat styles — the controller must fix that (its α<0 sign-flip path is the mechanism), not introduce a new failure mode. |
| FID / CLIPScore on the eval split | Not used as gate | Memory'd as `project_fm_val_loss_uninformative` — automated quality metrics don't track quality on Anima. Run them as diagnostic but don't block on them. |

The "controller vs scalar" comparison must be at matched DCW operator (LL-only, `one_minus_sigma`); otherwise we're not isolating the controller's contribution.

## Costs

| Pipeline | Added cost vs scalar recipe |
|---|---|
| Calibration (base-DiT-scoped, default) | One bench run at ≥48 prompts (~10–20 min on 5060 Ti). No training step. **Done once per base DiT release**, not per LoRA. |
| Calibration (per-LoRA, fallback) | Same cost, multiplied by N adapters that fail the cross-LoRA invariance gate. Default: 0× (gate passes). |
| LoRA training | Zero. Calibration is decoupled from LoRA training. |
| Storage (base-DiT controller) | ~200 B (2 × 24 floats) shipped alongside the base DiT. Single artifact, all LoRAs inherit. |
| Storage (per-LoRA fallback) | ~200 B per fallback LoRA (`<name>_dcw_vfwd.safetensors`). Default: zero. |
| Inference (per step) | One scalar multiply. Effectively free. |
| Inference (one-time) | Read ~200 B at start; one closed-form solve at step K (microseconds). |
| Cognitive | One flag (`--dcw_online_disable`) for users who want the scalar behavior. |

## What this is not

- **Not a training loss.** The LoRA is not retrained. The reference profile is computed once per base DiT release; the per-prompt α is fit at inference. (Reading B from `docs/methods/dcw.md` — DCW as a training loss — remains out of scope.)
- **Not a replacement for the LL-only landing.** Builds on it. Without `--dcw_band_mask=LL` and `apply_dcw(..., bands={"LL"})` the controller has nothing to drive.
- **Not a multi-band controller.** LH/HL/HH stay at zero (or the scalar default) per the SNR≈3 finding above. Adding bands triples the artifact size and we have no evidence the detail-band gaps are prompt-dependent.
- **Not a new conditioning input.** The α fit reads only quantities the sampler already computes: per-step `‖v_rev‖_LL` (one DWT on the latent we already have) and a static reference profile. No `c_pool`, no vision tower, no IP-adapter-style image conditioning.
- **Not a wavelet adapter.** Same posture as the parent proposal: the per-band correction lives at the sampler boundary; no in-block wavelet branch.

## Risks and caveats

1. **Single-amplitude model leaves 6% of variance on the table.** R² = 0.942 means a single per-prompt scalar captures most but not all of the LL gap variance; the residual 6% is per-prompt schedule shape. If post-correction residual is still LL-variance-dominated, that's evidence v1 (per-prompt schedule via `(c_pool, σ_t)` head) would help. Settled empirically by the controller-quality gate.

2. **`‖v_fwd‖` reference cross-sample variance.** v0a's slope fit assumes the population-mean `‖v_fwd‖_LL(σ)` is a tight reference (i.e. cross-sample variance in `‖v_fwd‖` is much smaller than per-prompt gap signal, so the residual `g_obs ≈ ‖v_rev‖_obs − vfwd_ref` isolates the per-prompt amplitude). Plausible but **unverified** on Anima — needs one extra column in the calibration bench's accumulator. If `‖v_fwd‖` cross-sample SNR is < ~3, the fit is dominated by reference noise and v0a's α estimate is unreliable; revert to v0 (offline head) on that scope.

3. **Gap is a proxy.** Same caveat as the scalar recipe (and the parent proposal §"Risks"): minimizing measured LL gap is not the same as maximizing perceptual quality. The perceptual side-by-side gate is therefore non-negotiable.

4. **Probe phase runs uncorrected on bad-α prompts.** For prompts where `α_prompt` would be e.g. −0.5 (the population scalar would *hurt* them), the probe phase still applies the population scalar for K=4 steps before the controller corrects course. K=4 ≈ 17% of the trajectory at high σ, which is the regime where the bias is smallest, so the cost is bounded but nonzero. Mitigation: a lower `K` shrinks the uncorrected window (K=2 already gets r=0.87 — usable) but increases noise in the α fit. Settled by sweeping K on the calibration data in the validation harness.

5. **CFG mismatch.** Bench measures at CFG=1; production runs at CFG=N. Same caveat as the scalar recipe. The controller doesn't fix this — its reference profiles inherit the CFG-1-optimal posture. If CFG mismatch dominates residual error, the v2 path is a CFG-conditioned reference profile (cheap — just calibrate at CFG=N too); speculative until we have evidence.

6. **Stacked LoRA composition.** The "pick-largest-multiplier" heuristic is principled in the limit (when one LoRA dominates) but breaks when multipliers are balanced (e.g. style + character at 0.7/0.7). Document the limitation; recommend `--dcw_online_disable` (falls through to the scalar-recipe averaging path, which is also a heuristic but at least continuous in the multipliers). Under the cross-LoRA invariance hypothesis there's only one reference profile anyway, so this risk only materializes when the per-LoRA fallback fires.

7. **Reference-profile drift across base-DiT re-trains.** Each base DiT release needs a new calibration bench run and a new `_dcw_vfwd.safetensors`. As long as the per-step means are within ±5% across re-runs at the same n, the artifact is stable; if not, the bench is undersampled and n needs to grow. Cheap to verify (re-run with a different seed_base and diff the floats).

## Concrete deliverables

In order, with explicit blockers:

0. **Blockers (already in main):**
   - LL-only DCW (`networks/dcw.py::apply_dcw(..., bands=)`) — landed.
   - Per-LoRA scalar recipe (`library/inference/dcw_calibration.py`) — landed (commits `cce9ceb`, `620c9c9`, `96272c8`).
   - Cross-sample variance instrumentation (`archive/dcw/measure_bias.py` per-band std + SNR) — landed today.
   - **Transfer-hypothesis check** — done 2026-05-04 at n=48 × 2; r=0.989 → v0a chosen. Artifacts in `archive/dcw-learnable-calibrator/transfer-analysis/`. Per-sample dump flag `--dump_per_sample_gaps` landed in `archive/dcw/measure_bias.py`.
   - **Cross-LoRA invariance gate** — passed 2026-05-04 on HydraLoRA (mult ∈ {0.5, 1.0}) + regular artist LoRA (mult=1.0); max |Δ_LL| = 5.12%, all configs in the base-DiT-scope branch. Artifacts in `archive/dcw-learnable-calibrator/lora-transfer/` (`cross_lora_check.md` + `linearity_check.md`).

1. **Cross-LoRA robustness margin.** The gate passed on 2 LoRA families spanning {MoE, regular} × {multi-task, single-style}. One more run on a structurally different parameterization (T-LoRA / OrthoLoRA / ReFT) closes the §"Cross-LoRA invariance finding" §"What this finding does NOT cover yet" list. Single mult=1.0 measurement, ~30 min on 5060 Ti per LoRA. Re-runs `cross_lora_check.py` and updates the lora-transfer findings — non-blocking for the controller build but worth doing before §"Doc updates".

2. **Reference-profile exporter.** Extend `archive/dcw/measure_bias.py` (or a sibling script) to dump the 24-float `vfwd_ref` and `mu_g` arrays + cross-sample std for `‖v_fwd‖_LL` (the v0a feasibility verification — see §"Risks" #2) into a tiny safetensors per scope. Wire `--dump_reference_profile` analogously to today's `--dump_per_sample_gaps`. Output: `output/dcw_calibration/<scope>/dcw_vfwd.safetensors` + `manifest.json`. ~2 hours.

3. **Online controller module.** New file `library/inference/dcw_online.py`: defines `OnlineDCWController` with `probe_step(v_rev_LL, sigma)`, `fit(λ_scalar) -> α_prompt` (closed-form least-squares), `lambda_for(sigma) -> float`. Reads `<scope>_dcw_vfwd.safetensors`. Pure PyTorch, no training step at all. Includes `α_prompt` clipping + a debug logger that emits the fit residual per prompt (helpful for the validation harness). ~½ day.

4. **Inference resolution chain.** Extend `library/inference/dcw_calibration.py` resolve order + `generation.py` call site to load the controller and override the scalar recipe. Resolution order: per-LoRA controller (if step 1 forced fallback) → base-DiT controller → scalar recipe → global default. Add `--dcw_online_disable`, `--dcw_online_probe_steps`, `--dcw_online_alpha_clip` to `inference.py`. ~½ day.

5. **Validation harness.** Extend `bench/dcw/calibrate_per_lora.py` (the artifact from the parent proposal step 5) to also instantiate the controller and eval. Adds: per-sample residual variance metric, controller-vs-scalar A/B, K-sweep diagnostic (K ∈ {2, 3, 4, 6}), perceptual side-by-side scaffolding (PNG grid for 4 LoRAs × 12 prompts), explicit reporting of the α<0 / sign-flip subset (the operationalized `project_dcw_when_to_use` failure mode — these are the prompts the controller must fix). ~1 day.

6. **Controller-quality gate run.** Run the §"Controller-quality gate" on 4 LoRAs (flat, painterly, detail-dense, base-DiT control), each at multiplier=1.0. The slow step — needs the LoRAs trained and 4 perceptual side-by-sides. ~1 day end-to-end, dominated by side-by-side review.

7. **Doc updates.** If the gate passes: this proposal moves to `docs/methods/dcw-learnable-calibrator.md`, with the gate's results table inlined. `docs/methods/dcw.md` gains a "Per-prompt amplitude (online controller)" section and updates the inference resolution order. Parent `lora-dcw-proposal.md` gets a forward-link to this. If the gate fails: archived under `archive/proposals/`, `findings.md` gains a final section with the negative result, the scalar recipe stays as the production path.

Steps 2–5 are roughly half a day each. Steps 1 and 6 dominate wall time (cross-LoRA bench + perceptual review). Total: **~2 days** for v0a (down from 3-5 in the original v0 plan).

## Open questions

- **Is `‖v_fwd‖_LL(σ)` cross-sample-tight enough to serve as the inference-time reference?** v0a's α-fit subtracts a population-mean reference instead of the unobservable matched-x₀ forward branch. The structural argument (forward branch starts from independent noise, so cross-sample variance in `‖v_fwd‖` is small) is plausible but **unverified on Anima**. Settled by step 2's `--dump_reference_profile` output: we need cross-sample SNR(`‖v_fwd‖_LL`) ≳ 3 for the α fit to be signal-dominated. If it's lower, revert to v0 (offline head).

- **Does v0a's single-amplitude form leave the residual 6% of LL variance on the table in a way that hurts perceptually?** R² = 0.942 means a single per-prompt scalar captures most LL variance; the remaining 6% is per-prompt schedule shape. If post-correction residual is still LL-dominated, v1 (per-prompt schedule via `(c_pool, σ_t)` head) becomes justified. Settled by step 5's residual-variance metric. The K-sweep in step 5 is the cheap diagnostic — if K=8 wins meaningfully over K=4 on the residual, the per-prompt shape varies along the trajectory.

- **Cross-LoRA invariance robustness margin.** The gate passed on 2 LoRA families (HydraLoRA MoE + regular artist LoRA) — see §"Cross-LoRA invariance finding". Open sub-questions: does invariance hold for OrthoLoRA / T-LoRA / ReFT (structurally different parameterizations)? For *painterly* / detail-dense LoRAs specifically — both checked LoRAs perturb the bias signature very little; a LoRA designed to enhance detail could push the LL gap further. Image-conditioned methods (IP-Adapter / EasyControl) are flagged for v2 since they change the network forward more meaningfully than weight-space adapters. Step 1 of the deliverables is the cheap robustness margin run.

- **Optimal `K`.** Default K=4 lands at r=0.96 on base-DiT prompts; K=2 at r=0.87 is also viable. Lower K → DCW kicks in earlier on bad-α prompts (see §"Risks" #4) but the fit is noisier. K-sweep in step 5 picks the operating point empirically.
