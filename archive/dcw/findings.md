# DCW Findings — what the bench told us, what `plan.md` should change

Provenance: `bench/dcw/results/20260503-1720/` (sweep, 4 samples × 2 seeds × 24 steps, flow_shift=1.0, plain Euler, no CFG, no LoRA). Earlier baseline-only run: `20260420_223406+` cluster (28 steps × 6 × 3). Repo SHA at write time: `f9aa144`. Decode side-by-side images live under `20260503-1720/images/`.

This document is a delta on `plan.md`. Read it together; anywhere the two disagree, this file wins until the numbers change.

---

## 1. The headline result

**The paper's Fig. 1c does not reproduce on Anima — the sign is flipped.**

Yu et al. report `||v_θ(x̂_t, t)|| > ||v_θ(x_t_fwd, t)||` at every step (Key Finding 2). On Anima at production-matched inference settings the inequality is reversed at every step except step 0–1. Baseline integrated signed gap = **−405.6** (24-step run; the 28-step run gave −409.9). The shape is near-monotone-negative and accelerates in the last 6 steps (gap −22.9 at σ=0.25 → −64.0 at σ=0.04).

This isn't noise — it reproduces across multiple run configurations (`flow_shift=1.0` and `flow_shift=3.0` both negative in the late half; only `flow_shift=3` showed any positive-gap regime, and that was an early-step artifact of the shift schedule). The bias is real, large, and oppositely signed from the paper's setting.

**Mechanistic implication:** the paper's correction `prev += +λ · σ_i · (prev − x0_pred)` (Eq. 17 with σ-decay schedule, λ > 0) would push the trajectory *further from* the forward branch on Anima at every late step. Direction of correction must invert.

---

## 2. The sweep result

Pixel-mode DCW on a `(scaler × schedule)` grid; `scaler ∈ {-0.3, -0.1, -0.03, -0.01, 0, 0.01, 0.1}`, `schedule ∈ {const, one_minus_sigma}`. Twelve configs + baseline = 13.

### 2.1 Sweep ranker disagrees with the eye

The `integrated |gap|` ranker put **`λ=-0.01_const`** at #1 (|gap|=230, vs baseline 408). Inspecting the curve: that config sign-flips at step ~15 and ends at gap=**+13.4**. It's overcorrecting late steps into the paper's predicted regime — the score is low because positive overshoot cancels negative magnitude under summation, not because the bias is closed.

PNG file size confirms: baseline 1179 KB → `λ=-0.01_const` 808 KB. High-frequency content compressed away. Visual symptom is over-smoothing, exactly what a sign-flip on late-step gap would produce.

The user's perceptual winner — **`λ=-0.01_one_minus_sigma`** — ranks #3 by |gap| (427) but its curve geometry is qualitatively different:

| step | σ | baseline gap | `-0.01_const` | `-0.01_one_minus_sigma` |
|---|---|---|---|---|
| 18 | 0.25 | −22.9 | −20.1 | −29.2 |
| 21 | 0.13 | −38.1 | −15.4 | −36.6 |
| 22 | 0.08 | −48.8 | −6.9  | −38.8 |
| 23 | 0.04 | −64.0 | **+13.4** ← flipped | **−38.5** |

`one_minus_sigma` gives a 40 % peak |gap| reduction at the worst step *without flipping the sign*. Mid-trajectory (steps 0–15) is essentially untouched (Δgap ≤ 2.0). PNG size 1125 KB — high-freq content roughly preserved. Perception tracks curve shape, not raw integral.

### 2.2 Ranker that aligns with perception

`integrated |gap|` is the wrong objective. Two metrics that match the eye:

| metric | baseline | `-0.01_const` | `-0.01_one_minus_sigma` |
|---|---|---|---|
| late-6-step \|gap\| | 232 | 96 | 210 |
| max \|gap\| | 64 | 21 | **39** |
| sign-flip count, late half | 0 | 1 | **0** |
| `Σ \|gap\| · (1−σ) + 100 · #(gap_i > 0, late half)` | 232 | 196 | **210** ← within 7 % |

Recommend the script's `summary.json` adopt the last row (or equivalent late-weighted, sign-flip-penalised metric) as the primary ranker. Keep the current metric as a secondary column for compatibility with the paper's framing.

---

## 3. Why `one_minus_sigma` is the right shape

The bias on Anima is concentrated at low σ. Three intuitions, all consistent with the data:

1. **Manifold geometry.** At low σ, reverse trajectories are close to the learned data manifold and produce small residual velocities. Forward-noised real latents at the same σ sit slightly *off* the manifold (per-step independent ε_fwd, no trajectory continuity) and elicit larger "pull-back" velocities. The gap is a manifold-mismatch readout, peaking where the manifold matters most: late in denoising.
2. **Cumulative drift.** By step `i` the reverse branch has integrated `i` Euler steps of prediction error; the forward branch is recomputed from `x_0` each step. Late-step drift dominates.
3. **Sigma-scaling of the correction.** `(1−σ)` weight matches the empirically-observed `|gap(σ)|` envelope much better than σ-decay (paper's Eq. 20, peaks early). `const` distributes correction uniformly and corrupts mid-trajectory where there's no bias to fix.

This is the empirical answer to `plan.md` open question §5.1 ("FLUX uses a single `scaler`, not λ·σ_i; worth comparing"): on Anima at inference-matched settings, **neither the constant scaler nor the σ-decay schedule is right**. `(1−σ)` matches the bias envelope.

---

## 4. Revisions to `plan.md`

Section-by-section. Apply when the integration PR is opened.

### 4.1 §0 (translating the paper to Anima)

The flow-matching adaptation table is correct, but the schedule recommendation needs a footnote:

> | DCW correction (pixel) | `prev_sample + scaler · (prev_sample − x_0_pred)` |
>
> with `scaler = λ · (1 − σ_i)` (empirical `one_minus_sigma`, `λ ≈ -0.01` on Anima — see `findings.md` §3 for why this differs from the paper's `+λ · σ_i`).

### 4.2 §1 deliverables — re-prioritise

DWT-based modes (`low`, `high`, `dual`) drop from P0/P1 → P2 or shelved. Reasons:
- The bias envelope is captured by pixel-mode `(1−σ)` weighting; no evidence the wavelet decomposition gains anything beyond it.
- The "low-freq = shape, high-freq = detail" interpretation is pixel-space, not latent-space. We have no calibration for what a Haar DWT on 16-ch Qwen-VAE latents corresponds to perceptually. Plan.md §5.2 already hedged this; the empirical case never arrived.

Revised P0/P1:
1. `networks/dcw.py` with **pixel mode only**, default schedule `one_minus_sigma`. ~30 lines, not 120.
2. CLI flags: `--dcw`, `--dcw_lambda` (single scalar, default e.g. -0.01 after final sweep), `--dcw_schedule` (default `one_minus_sigma`, also `const`/`sigma_i`/`none` for ablation).
3. Plumbing into `generation.py` + `spectrum.py` + `er_sde` paths — unchanged.
4. Doc in `docs/methods/dcw.md` — lead with the Anima-specific sign and schedule, not the paper's.

### 4.3 §2.2 CLI defaults

Drop `--dcw_mode` argument entirely (pixel-only). Drop `--dcw_lambda_h` and `--dcw_wave`. `--dcw_lambda_l` becomes `--dcw_lambda`.

### 4.4 §2.5 dependencies

`pytorch-wavelets` and `PyWavelets` no longer required. Don't add them.

### 4.5 §3 validation plan

Replace the FID-lite plan with the perceptually-aligned ranker (above §2.2) plus a 16-prompt fixed-seed visual panel. The bench measurement is necessary; |gap| numbers alone are not sufficient.

### 4.6 §5 open questions — resolved / still open

| question | status |
|---|---|
| §5.1 FLUX scaler vs λ·σ_i | **Resolved**: neither — Anima needs `(1−σ)`. |
| §5.2 latent vs pixel for wavelet | **Mooted**: pixel-mode suffices; wavelet shelved. |
| §5.3 APEX + DCW double-correct | Still open. Needs ablation row when APEX checkpoint is on hand. |
| §5.4 postfix-σ + DCW | Still open, but expected non-interaction (different tensors). |

New open question:
- **§5.5 Sign-flip on Anima vs original paper.** Why does flow-matching on a DiT trained at this scale produce the inverted bias? Three candidate explanations in `bench/dcw/README.md` "Observed on Anima"; an ablation on a smaller / pixel-space model is the cleanest test, but probably out of scope.

---

## 5. Decision gates for proceeding

`plan.md` §4 outcome (1) is conditional on "|gap| closes *and* samples improve." Restate with the metric correction:

- **Proceed to integration if:** the narrowed sweep (λ ∈ {-0.02, -0.015, -0.01, -0.0075, -0.005, 0}, schedule=`one_minus_sigma`, n_images=6 × n_seeds=3 × 28 steps) produces a winner with (a) max |gap| ≤ 60 % of baseline, (b) zero late-half sign flips, (c) visibly improved 16-prompt panel.
- **Shelve if:** narrowed sweep fails (a) or (b), or panel shows no perceptual win.
- **Do *not* gate on integrated |gap|** — that metric is misaligned with perception in this regime, as §2.1 shows.

The current 4×2 sweep meets (a) and (b) for `λ=-0.01_one_minus_sigma` (max |gap| 39, late-half flips 0). Pending: (c) the panel, plus tighter sample count to confirm.

---

## 6. What this run did *not* tell us

Honest list:

- **Generalisation across LoRA stacks.** Bench was on the base DiT. The sign and shape of the gap may differ on a checkpoint with LoRA / OrthoLoRA / T-LoRA / ReFT applied. One ablation row each at integration time would settle it; doesn't block the v0 PR.
- **Interaction with CFG.** Bench is conditional-only (`guidance_scale=1`). Production inference uses CFG. If CFG amplifies or cancels the bias, the recommended λ may need rescaling. Run a CFG-on baseline at integration time.
- **APEX vs vanilla.** §5.3 above. APEX trains around the bias by construction; the gap on a 4-NFE APEX checkpoint may be qualitatively different.
- **Spectrum interaction.** §5.2 in plan.md; still untested.
- **Prompt sensitivity.** 4 captions across 4 images × 2 seeds is enough to see shape, not enough to claim universality. Visual panel before integration covers this.

---

## 7. TL;DR for future-me

- Anima's SNR-t bias is **opposite-signed** from Yu et al. and concentrated at **low σ** (late steps).
- Best Anima-specific DCW form: **pixel-mode**, `scaler = λ · (1 − σ_i)`, `λ ≈ -0.01`.
- The script's default `integrated |gap|` ranker overrates configs that overshoot. **Late-step `|gap|` weighted by `(1 − σ)` plus sign-flip penalty** is the perceptually-aligned ranker.
- `plan.md` simplifies considerably: pixel-only, one schedule, one scalar, no DWT deps.
- Decision is contingent on a 16-prompt visual panel; mechanistic case is solid, perceptual case is in progress.

---

## 8. Per-band finding (2026-05-03) — LL-only is the new default

`bench/dcw/results/20260503-2102-band-mask-eyeball/` ran a per-Haar-subband sweep on the same 4-image / 2-seed bench used in §1–§5. Headline:

| Config | late-half integrated \|gap\| | Δ vs baseline | per-band signed gap (LL / LH / HL / HH) |
|---|---|---|---|
| baseline | 330.1 | — | −317 / −165 / −165 / −127 |
| **`λ=-0.01_one_minus_sigma_LL`** | **235.7** | **−28.6%** | **−225 / −120 / −122 / −92** *(all bands improved)* |
| `λ=-0.01_one_minus_sigma_all` *(prior default)* | 340.6 | **+3.2%** | −180 / −240 / −242 / −222 *(LL improved, detail bands worsened)* |
| `λ=-0.01_one_minus_sigma_HH` | 363.6 | +10.2% | −300 / −146 / −147 / **−287** *(HH sign-flipped)* |

The previously-shipped `_all` form makes the late-half gap **worse** than baseline. It only "wins" on max-|gap| at the very last step (σ=0.04), bought by worsening steps 12–22 — the σ-weighted summation in the `_all` config hid LH/HL/HH degradation under the LL improvement. The total-gap ranker mistook this for an improvement.

Restricting the correction to LL is **strictly better** by every metric checked: lower late-half |gap|, no sign flips, all four per-band gaps improved vs baseline, and visually equivalent or slightly better on the 4-image panel. Mechanism: LL is an upstream causal lever — applying LL-only correction at step `i` propagates through the DiT's nonlinear forward and tightens all four band gaps at step `i+1` and after. Detail bands are downstream symptoms, not independent failures.

**HH-only is dead** in both schedules — perceptually indistinguishable from baseline (file-size delta < 1%) when not actively breaking things.

### Re-tuning λ for LL-only

LL is a more responsive lever than all-bands. `λ=-0.01_const_LL` already overshoots LL by 2× (sign-flips it from −317 → +659 and visibly breaks the image). With `one_minus_sigma`, the schedule's late-weighted shape keeps the correction in-bounds — a magnitude sweep on the same 4-image / 2-seed bench (`bench/dcw/results/20260503-2124-ll-magnitude/`) showed monotone improvement from −0.005 → −0.015 with no sign flips and no visible image damage:

| λ | late-half \|gap\| | Δ vs baseline | max \|gap\| |
|---|---|---|---|
| baseline | 330.1 | — | 64.0 |
| −0.005 | 281.5 | −14.7% | 53.0 |
| −0.010 | 235.7 | −28.6% | 42.1 |
| **−0.015** | **192.6** | **−41.7%** | **31.8** |

**Shipped LL-only default: `λ = -0.015`, `schedule = one_minus_sigma`, `band_mask = LL`.** The closed-form solver predicts λ* ≈ −0.033 but that extrapolation crosses the nonlinear regime where `LL_const`-style overshoot kicks in (|λ · w(σ)| > ~0.01 late-step). −0.015 is conservative — closes 83% of the LL gap at the worst step (σ=0.04) and leaves headroom for per-LoRA calibration to push either direction.

§§4.4 / 4.5 / 5.5 above all mooted the wavelet path on the basis of the broadband bias envelope; the LL band mask reinstates a single subband as a deliberate restriction *of* the pixel-mode correction, not a return to the paper's full DWT decomposition. Cross-reference: `docs/methods/dcw.md §LL-only correction` is the canonical user-facing write-up; the result envelope lives in `bench/dcw/results/20260503-2102-band-mask-eyeball/result.json` and `bench/dcw/results/20260503-2124-ll-magnitude/result.json`.
