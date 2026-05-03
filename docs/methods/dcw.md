# DCW — Post-Step SNR-t Bias Correction

Training-free, sampler-level correction that closes the SNR-t bias of flow-matching DiTs by mixing each Euler step's `prev_sample` toward (or away from) the model's `x0_pred`.

Paper: [Elucidating the SNR-t Bias of Diffusion Probabilistic Models](https://arxiv.org/abs/2604.16044) (Yu et al., CVPR 2026)

**Read first:** `archive/dcw/findings.md`. The paper's bias direction does not reproduce on Anima — Anima's λ is **negative**, opposite the paper. Everything below assumes you've internalized that.

## Anima form

```
denoised   = latents − σ_i · v                       # x0_pred (FLUX velocity convention)
prev       = Euler/ER-SDE step                        # prev_sample
diff       = prev − denoised
diff_LL    = haar_idwt(LL(diff), 0, 0, 0)             # LL-only band mask
prev      += λ · (1 − σ_i) · diff_LL                  # DCW correction
```

Defaults: `λ = -0.015`, schedule `one_minus_sigma`, band mask `LL`. The LL-only restriction is the empirical default — broadband correction worsens detail bands while LL-only improves all four (see §LL-only correction below). The closed-form solver and historical perceptual sweep both supported `λ ≈ -0.010` for the broadband variant; LL is a more responsive lever, and the magnitude sweep landed on `-0.015` as the conservative LL default. See `archive/dcw/findings.md` and `archive/dcw/plan.md §3` for derivation.

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
    --dcw_lambda -0.015                                        \
    --dcw_schedule one_minus_sigma                             \
    --dcw_band_mask LL                                         \
    ...  # other inference args
```

## CLI

| Flag | Default | Notes |
|------|---------|-------|
| `--dcw` | off | Enable post-step correction. |
| `--dcw_lambda` | `-0.015` | Negative on Anima — see findings. Tuned for `--dcw_band_mask LL`; use `-0.010` if you switch to `all`. |
| `--dcw_schedule` | `one_minus_sigma` | One of `one_minus_sigma`, `sigma_i`, `const`, `none`. |
| `--dcw_band_mask` | `LL` | Restrict correction to a subset of single-level Haar subbands. Format: `LL`, `HH`, `LH+HL+HH`, or `all`. LL-only is strictly better than `all` on Anima (see §LL-only correction). |

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

## When to use

DCW helps when the **target image is detail-dense** (busy compositions, intricate textures, complex backgrounds) — the late-step bias correction tightens edges and recovers fine structure. It is **not helpful — and can hurt — when the target is intentionally simple** (e.g. the flat, minimal style of channel/caststation-class artists). On those, the correction over-sharpens what should be deliberately smooth, and the baseline (no DCW) is preferable. Match the flag to the prompt, not the checkpoint.

## Limitations / open questions

- `s_i` flips sign mid-trajectory (positive through steps 12–21, strongly negative in the last 2–3). On the original total-gap measurement this looked like a schedule problem, but the per-band split below shows it was actually band-mixing: LL improvement vs detail-band degradation cancel under the wrong sign at mid-σ. Restricting to LL (next section) eliminates the apparent flip without needing a fancier schedule.
- Cached-Spectrum `x0_pred` is biased by the Chebyshev forecaster's prediction error. Empirically should still help (correction is bias-agnostic). Worth one explicit ablation row.
- Sign-flip vs the original paper is unresolved. Three speculative mechanisms in `archive/dcw/README.md`; cleanest test (smaller / pixel-space DiT) is out of scope.

## LL-only correction (2026-05-03 finding)

`bench/dcw/results/20260503-2102-band-mask-eyeball/` ran a per-Haar-subband sweep on the same 4-image / 2-seed bench. Headline:

| Config | late-half integrated \|gap\| | Δ vs baseline | per-band signed gap (LL / LH / HL / HH) |
|---|---|---|---|
| baseline | 330.1 | — | −317 / −165 / −165 / −127 |
| **`λ=-0.01_one_minus_sigma_LL`** | **235.7** | **−28.6%** | **−225 / −120 / −122 / −92** *(all bands improved)* |
| `λ=-0.01_one_minus_sigma_all` *(current default)* | 340.6 | **+3.2%** | −180 / −240 / −242 / −222 *(LL improved, detail bands worsened)* |
| `λ=-0.01_one_minus_sigma_HH` | 363.6 | +10.2% | −300 / −146 / −147 / **−287** *(HH sign-flipped)* |

The currently-shipped `_all` form makes the late-half gap **worse** than baseline. It only "wins" on max-|gap| at the very last step (σ=0.04), bought by worsening steps 12–22. The total-gap ranker mistook this for an improvement because the LH/HL/HH degradation hides under the σ-weighted summation in the `_all` config.

Restricting the correction to LL (one Haar subband) is **strictly better** by every metric we checked: lower late-half |gap|, no sign flips, all four per-band gaps improved vs baseline, and visually equivalent or slightly better on the 4-image panel. The mechanism: LL is an upstream causal lever — applying LL-only correction at step `i` propagates through the DiT's nonlinear forward and tightens all four band gaps at step `i+1` and after. Detail bands are downstream symptoms, not independent failures.

**HH-only is dead** in both schedules — perceptually indistinguishable from baseline (file-size delta < 1%) when not actively breaking things.

### Landed changes

The LL-only finding is wired through the inference path:

- `networks/dcw.py` — `apply_dcw` takes `bands: frozenset[str] = ALL_BANDS`; when the mask covers all four bands it falls through to the original `prev + s · (prev − x0_pred)` (bit-identical, no DWT round-trip), otherwise DWT-mask-iDWT the differential. `parse_band_mask` exposes the CLI string-form parser (`"LL"`, `"LH+HL+HH"`, `"all"`).
- `inference.py` — `--dcw_band_mask` (default `LL`); `--dcw_lambda` default raised in magnitude to `-0.015`.
- `library/inference/generation.py`, `networks/spectrum.py` — both `apply_dcw` call sites and the spectrum dispatcher thread the band mask through.

Follow-ups worth tracking:

- The `(1 − σ)` schedule is fine for LL-only — sign-flip count is 0 on every band, no need for `(1 − σ)²` or step-clipped variants from the previous "Limitations §1". Leave the schedule unchanged.
- LL-mode does not cost anything at inference: the DWT is one pass over a single-resolution latent (~16ch × H/2 × W/2 = a few MB), one mask, one iDWT, one add. Negligible vs the DiT forward, same overhead claim as before.

### Re-tuning λ for LL-only

LL is a more responsive lever than all-bands. `λ=-0.01_const_LL` already overshoots LL by 2× (sign-flips it from −317 → +659 and visibly breaks the image). With `one_minus_sigma`, the schedule's late-weighted shape keeps the correction in-bounds — a magnitude sweep on the same 4-image / 2-seed bench (`bench/dcw/results/20260503-2124-ll-magnitude/`) showed monotone improvement from −0.005 → −0.015 with no sign flips and no visible image damage:

| λ | late-half \|gap\| | Δ vs baseline | max \|gap\| |
|---|---|---|---|
| baseline | 330.1 | — | 64.0 |
| −0.005 | 281.5 | −14.7% | 53.0 |
| −0.010 | 235.7 | −28.6% | 42.1 |
| **−0.015** | **192.6** | **−41.7%** | **31.8** |

**New LL-only default: `λ = -0.015`, `schedule = one_minus_sigma`.** The closed-form solver predicts λ* ≈ −0.033 but that extrapolation crosses the nonlinear regime where `LL_const`-style overshoot kicks in (|λ · w(σ)| > ~0.01 late-step). −0.015 is conservative — closes 83% of the LL gap at the worst step (σ=0.04) and leaves headroom for per-LoRA calibration to push either direction.

Re-tuning recipe still works (see §"Calibration recipe"). Just pass `--dcw_band_masks LL --dcw_scalers <anchors>` and read `λ*` for the `one_minus_sigma:LL` group.
