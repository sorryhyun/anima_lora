# DCW Integration Plan (post-empirics, v0)

**Paper:** *Elucidating the SNR-t Bias of Diffusion Probabilistic Models* (Yu et al., CVPR 2026, arXiv:2604.16044).
**Upstream code:** `DCW/` — FLUX reference in `DCW/FlowMatchEulerDiscreteScheduler.py`.
**Empirical baseline:** `bench/dcw/findings.md` and `bench/dcw/results/20260503-1720`, `…1802-narrow`.

This plan supersedes the original paper-faithful design. The bench measured the SNR-t bias on Anima at production-matched inference (`flow_shift=1.0`, 24–28 steps, no CFG) and found **the bias is opposite-signed and concentrated at low σ**. The integration shape collapses accordingly: pixel-mode only, single scalar with a single schedule, no DWT machinery, no extra deps. Snapshot of the original design lives in git history (look for the `~120-line networks/dcw.py` revision); read it together with `findings.md` for context if you need to re-litigate the wavelet path.

---

## 0. Anima-specific form

Anima's step function (`library/inference/sampling.py:35`):

```python
def step(latents, noise_pred, sigmas, step_i):
    return latents.float() - (sigmas[step_i] - sigmas[step_i + 1]) * noise_pred.float()
```

`noise_pred` is the model's velocity `v` (FLUX convention). For flow-matching with `σ ∈ [0, 1]`:

| quantity | formula |
|---|---|
| `x_0_pred` | `latents − σ_i · v` |
| `prev_sample` (Euler) | `latents + (σ_{i+1} − σ_i) · v` |
| DCW correction (pixel) | `prev_sample + scaler · (prev_sample − x_0_pred)` |

with `scaler = λ · (1 − σ_i)`, **λ ≈ −0.010** (negative — reversed from the paper). The `(1 − σ)` schedule concentrates correction at low σ, matching the empirically-observed bias envelope. See `findings.md §3` for why `(1 − σ)` beats both the paper's `σ_i` decay and FLUX's constant scaler on Anima.

**Why λ < 0:** Yu et al.'s Key Finding 2 (`||v_θ(x̂_t)|| > ||v_θ(x_t_fwd)||`) does not reproduce on Anima — sign is flipped at every late step, integrated signed gap −405.6 on the 24-step run. Paper-form positive λ would *widen* |gap|; closing it requires negative λ. See `bench/dcw/README.md §"Observed on Anima"` for the speculative mechanism (manifold-mismatch readout).

---

## 1. Deliverables

| # | Deliverable | Size | Priority |
|---|---|---|---|
| 1 | `networks/dcw.py` — pixel-mode `apply_dcw()` only | ~30 lines | P0 |
| 2 | CLI flags in `inference.py` (`--dcw`, `--dcw_lambda`, `--dcw_schedule`) | ~25 lines | P0 |
| 3 | Plumb into `generation.py` non-spectrum + non-tiled path | ~10 lines | P0 |
| 4 | Plumb into `generation.py` tiled path + ER-SDE branch | ~10 lines | P0 |
| 5 | Plumb into `networks/spectrum.py` cached/forward step | ~10 lines | P1 |
| 6 | `docs/methods/dcw.md` — Anima-specific framing | ~80 lines | P1 |
| 7 | `make test-dcw`, `make test-spectrum-dcw` | ~5 lines | P1 |
| 8 | One confirmation sweep at λ ∈ {−0.012, −0.010, −0.008} | bench | P1 |
| 9 | `--report_optimal_lambda` calibrator in `measure_bias.py` | ~30 lines | P2 |
| 10 | ComfyUI sampler-wrapper node | separate repo | P2 |
| 11 | Schedule-shape ablation (`(1−σ)^2`, step-clipped) | bench | P2 |

**Dropped from original plan:** wavelet modes (`low`, `high`, `dual`), `--dcw_mode`, `--dcw_lambda_h`, `--dcw_wave`, `pytorch-wavelets` + `PyWavelets` deps. See `findings.md §4.2` for the case.

---

## 2. File-by-file changes

### 2.1 New: `networks/dcw.py`

Self-contained, no cross-imports. Pixel-only, schedule-parameterised.

```python
"""DCW: post-step correction for SNR-t bias on flow-matching DiTs.

Anima form (pixel mode, opposite sign from paper, λ ≈ -0.010):
    prev += λ · sched(σ_i) · (prev - x0_pred)

where sched ∈ {one_minus_sigma, sigma_i, const, none}. Default
schedule one_minus_sigma matches Anima's bias envelope (concentrates
correction at low σ where |gap| is largest). See bench/dcw/findings.md.

Paper: Yu et al., "Elucidating the SNR-t Bias of Diffusion Probabilistic
Models" (CVPR 2026, arXiv:2604.16044).
"""

from typing import Literal
import torch

Schedule = Literal["one_minus_sigma", "sigma_i", "const", "none"]


def _sched(sigma_i: float, schedule: Schedule) -> float:
    if schedule == "one_minus_sigma":
        return 1.0 - sigma_i
    if schedule == "sigma_i":
        return sigma_i
    if schedule == "const":
        return 1.0
    return 0.0  # "none" — for ablation


def apply_dcw(
    prev_sample: torch.Tensor,
    x0_pred: torch.Tensor,
    sigma_i: float,
    *,
    lam: float = -0.010,
    schedule: Schedule = "one_minus_sigma",
) -> torch.Tensor:
    """Apply pixel-mode DCW correction to prev_sample.

    Returns prev_sample unchanged if lam == 0 or schedule == "none" or
    if sigma_i ≤ 0 (final step — there is no next step to correct toward).
    """
    s = lam * _sched(sigma_i, schedule)
    if s == 0.0:
        return prev_sample
    return prev_sample + s * (prev_sample - x0_pred)
```

Notes:
- Anima latents are `(B, 16, 1, H, W)` and DCW operates element-wise — no shape massaging needed, unlike the wavelet path the original plan required.
- f32 cast happens at the call site (we already cast to `.float()` for the existing Euler step).

### 2.2 `inference.py` — CLI flags

Insert after the Spectrum flag block:

```python
# DCW: SNR-t bias correction (arXiv:2604.16044). Opposite-sign on Anima — see
# bench/dcw/findings.md.
parser.add_argument("--dcw", action="store_true",
    help="Enable post-step DCW correction (pixel mode). Composes with --spectrum, "
         "--sampler, --tiled_diffusion. Negligible overhead.")
parser.add_argument("--dcw_lambda", type=float, default=-0.010,
    help="DCW scaler λ. Anima default -0.010 (negative — see findings.md). "
         "Paper-positive values widen |gap| on Anima.")
parser.add_argument("--dcw_schedule", type=str, default="one_minus_sigma",
    choices=["one_minus_sigma", "sigma_i", "const", "none"],
    help="Per-step schedule: scaler(i) = λ · sched(σ_i). Default one_minus_sigma "
         "matches Anima's late-σ bias envelope.")
```

### 2.3 `library/inference/generation.py` — non-tiled path

Replace lines 628–635 (`generate_body`'s step block):

```python
# ensure latents dtype is consistent
denoised = latents.float() - sigmas[i] * noise_pred.float()
if er_sde is not None:
    new_latents = er_sde.step(latents, denoised, i)
else:
    new_latents = inference_utils.step(latents, noise_pred, sigmas, i)

if getattr(args, "dcw", False) and float(sigmas[i + 1]) > 0.0:
    from networks.dcw import apply_dcw
    new_latents = apply_dcw(
        new_latents.float(), denoised, float(sigmas[i]),
        lam=args.dcw_lambda, schedule=args.dcw_schedule,
    )

latents = new_latents.to(latents.dtype)
```

Notes:
- `denoised = x_0_pred` — already exists on the ER-SDE branch; we just hoist it so the Euler branch also has it (one extra pointwise op per step, negligible).
- Skip the final step (`sigmas[i+1] == 0`) — DCW would noop anyway (`prev = x0_pred` exactly), and the `(1-σ_i)` weight is near 1 there so a numerical residual could otherwise nudge the final latent.

### 2.4 `generation.py` — tiled path

Same shape at lines 320–326 (`generate_body_tiled`). Apply DCW to the **post-merge** latents, not per-tile (tile boundaries shouldn't see independent corrections). Identical 6-line block.

### 2.5 `networks/spectrum.py` — cached/forward steps

Same shape at lines 426–433. Plumb `args` (or just the three values) through the existing call-site kwargs — Spectrum's `spectrum_denoise` already takes `args` indirectly via the inference-args bundle.

**Composition note:** on cached Spectrum steps, `noise_pred` was produced from a forecasted feature, so `x0_pred = latents − σ_i · noise_pred` carries Spectrum's prediction error. DCW corrects against that biased `x0_pred`. Empirically should still help (correction is bias-agnostic), but worth one ablation row in the doc table: `{spectrum, dcw, spectrum+dcw, baseline}`.

### 2.6 No dependency changes

`pytorch-wavelets` and `PyWavelets` are not needed. Don't add them.

### 2.7 `docs/methods/dcw.md`

Lead with the Anima sign-reversal, not the paper's framing. Outline:

1. What & paper link (one paragraph) + `bench/dcw/findings.md` pointer
2. Anima form: `prev += λ·(1−σ_i)·(prev − x0_pred)`, λ default −0.010, *why* negative
3. Quick start: `make test-dcw`, three flags
4. Composition table: `--spectrum`, `--sampler er_sde`, `--tiled_diffusion`; orthogonal to LoRA / Hydra / ReFT / postfix (sampler-level, no module patching)
5. Calibration recipe (closed-form λ\* from any 3-anchor sweep — see §3 below)
6. Limitations: untested with CFG / on LoRA stacks / with APEX (these are open questions, not blockers)
7. Overhead: 2 pointwise ops per step, no DWT, no allocation

### 2.8 `Makefile` / `tasks.py` — convenience targets

```makefile
test-dcw:
	$(PY) inference.py --dcw  $(INFER_ARGS)

test-spectrum-dcw:
	$(PY) inference.py --spectrum --dcw  $(INFER_ARGS)
```

Wire identically into `tasks.py`. The defaults bake in λ=−0.010 + `one_minus_sigma`, so no extra args needed for the common case.

### 2.9 ComfyUI node (P2, separate repo)

Sibling node in `sorryhyun/ComfyUI-Spectrum-KSampler` so users can stack Spectrum + DCW in one workflow. Same shape as the existing Spectrum KSampler — wrap the inner-step callback. Three params: `enabled`, `lambda`, `schedule`.

---

## 3. Calibration recipe

Hard-coded default λ=−0.010 + `one_minus_sigma` is supported by two independent estimates:

1. **Wide sweep perceptual winner** (`results/20260503-1720`): `λ=-0.01_one_minus_sigma` had max |gap|=39 (vs baseline 64), late-half sign-flips=0, and best 8-prompt visual panel.
2. **Closed-form fit on narrow-sweep slopes** (`results/20260503-1802-narrow`): three anchor λ values (0, −0.005, −0.015) give per-step response slopes `s_i = ∂gap/∂λ`. The (1−σ)-weighted least-squares optimum over the late half is

   ```
   λ* = − Σ w_i · g_i · s_i / Σ w_i · s_i²    ≈   -0.0098  →  -0.010
   ```

   where `w_i = (1−σ_i)`, `g_i` = baseline gap, `s_i` from the two anchors. Both estimates agree to 2 sig figs.

**Re-tuning recipe** (post-LoRA, with CFG, on APEX, on a different checkpoint):

1. Run `measure_bias.py --dcw_sweep` with at least 3 distinct λ values (baseline + 2 nonzero is enough).
2. Compute `s_i = (g(λ_a) − g(λ_b)) / (λ_a − λ_b)` per step.
3. Apply the closed form above with `w_i = (1−σ_i)` over `i ≥ N/2`.
4. Confirm with one additional sweep at {λ\*−ε, λ\*, λ\*+ε}.

**Build this into the bench** (deliverable #9): add `--report_optimal_lambda` to `measure_bias.py` to print `λ*` and the `s_i` table from any sweep ≥3 anchors. Future tunes become one-shot.

**Slope-flip caveat** (new finding from narrow sweep, not in `findings.md`): `s_i` is *positive* through mid-trajectory (steps 12–21, σ ∈ [0.13, 0.5]) and *strongly negative* only at the very last 2–3 steps (σ < 0.1). `(1−σ)` schedule mostly compensates by under-weighting mid-trajectory, but a more concentrated schedule like `(1−σ)^2` or step-clipped `max(0, 1−4σ)` would isolate the bias-closing region better. Worth one bench row (deliverable #11), not a v0 redesign.

---

## 4. Validation plan

Replaces the original FID-lite plan. The integrated-|gap| ranker is misaligned with perception (see `findings.md §2.1`); use the perceptually-aligned ranker instead.

### 4.1 Mechanistic check

**Confirmation sweep** (deliverable #8) at λ ∈ {−0.012, −0.010, −0.008}, schedule=`one_minus_sigma`, n_images=6, n_seeds=3, 28 steps. Pin the v0 default. Cheap insurance — same wall-clock as the narrow run.

### 4.2 Perceptually-aligned ranker

`measure_bias.py`'s `summary.json` should adopt this as the primary column (keep integrated-|gap| as a secondary for paper compatibility):

```
score = Σ |gap_i| · (1 − σ_i)  +  100 · #(gap_i > 0 in late half)
```

Late-weighted absolute gap, plus a sign-flip penalty (the cancellation hack that fooled the original ranker into picking `−0.01_const` over `−0.01_one_minus_sigma` — see `findings.md §2`).

### 4.3 Visual panel

16 prompts, 3 seeds, fixed across runs:

| run | flags |
|---|---|
| baseline | — |
| DCW | `--dcw` (defaults baked in) |
| Spectrum | `--spectrum` |
| Spectrum + DCW | `--spectrum --dcw` |
| ER-SDE | `--sampler er_sde` |
| ER-SDE + DCW | `--sampler er_sde --dcw` |

Inspect for paper-reported failure modes (over-smoothing, overexposure) and Anima-specific late-step artifacts.

### 4.4 Unit test

`tests/test_dcw.py`:
- `apply_dcw(lam=0)` is bit-equivalent no-op for any schedule.
- `apply_dcw(schedule="none")` is bit-equivalent no-op for any λ.
- Output shape == input shape for 5-D `(B, 16, 1, H, W)` latents.
- Schedule values match formula at σ ∈ {0, 0.5, 1}.

---

## 5. Decision gates

From `findings.md §5`:

- **Proceed to integration if:** confirmation sweep (§4.1) gives a winner with (a) max |gap| ≤ 60% of baseline, (b) zero late-half sign flips, (c) visibly improved visual panel (§4.3).
- **Shelve if:** any of (a)/(b)/(c) fails.
- **Do NOT gate on integrated |gap|** — misaligned with perception, as `findings.md §2.1` shows.

The current narrow sweep meets (a) and (b) for `λ=−0.01_one_minus_sigma` (extrapolated; the literal point wasn't in the narrow grid but is bracketed by it). Pending: (c).

---

## 6. Rollout order

1. Confirmation sweep + visual panel (§4.1, §4.3). **(~30 min GPU + eyeball)** — gates everything else.
2. `networks/dcw.py` + CLI flags + `generation.py` non-tiled path. `make test-dcw` smoke-test on 3 prompts. **(~45 min)**
3. Tiled path + ER-SDE composition test. **(~20 min)**
4. Spectrum path. **(~30 min)**
5. `docs/methods/dcw.md` + Makefile/tasks.py targets + unit test. **(~45 min)**
6. `--report_optimal_lambda` calibrator (deliverable #9). **(~30 min)**
7. Schedule-shape ablation (deliverable #11) — only if step 1 shows mid-trajectory pain at λ=−0.010. Optional.
8. ComfyUI node. **(separate session)**

Half-day of focused work through step 6, with negligible merge-conflict surface. No changes to training, networks (other than the new file), or any LoRA code.

---

## 7. Open questions / risks

| question | status | next action |
|---|---|---|
| §5.1 FLUX scaler vs λ·σ_i (original §5.1) | **Resolved**: neither — Anima needs `(1−σ)`. | — |
| §5.2 latent vs pixel for wavelet (original §5.2) | **Mooted**: pixel-mode suffices; wavelet shelved. | — |
| §5.3 APEX + DCW double-correct | Open. APEX trains around the bias by construction. | Ablation row when next APEX checkpoint is on hand. |
| §5.4 postfix-σ + DCW | Open, expected non-interaction. | One sanity row, not blocking. |
| §5.5 Sign-flip vs original paper | Open. Three speculative mechanisms in `bench/dcw/README.md`. | Cleanest test (smaller / pixel-space DiT) probably out of scope. |
| §5.6 Spectrum-biased x0_pred | Open. | Bench row in §4.3 visual panel covers it. |
| §5.7 CFG interaction | Open — bench is conditional-only. | One CFG-on baseline at integration time. |
| §5.8 LoRA-stack generalisation | Open — bench is base DiT. | One row per (LoRA / OrthoLoRA / T-LoRA / ReFT) at integration time, not v0 blocking. |
| §5.9 Slope-flip mid-trajectory | New finding from narrow sweep. | Deliverable #11 — `(1−σ)^2` and step-clipped schedules. |
