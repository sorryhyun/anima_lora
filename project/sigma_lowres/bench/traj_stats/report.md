# traj_stats — trajectory-resolved latent statistics

## Phase 3 — compute-reuse oracle replay: CLOSED (line archived)

**Verdict: compute reuse dies at the pre-gauge gate** (2026-07-24,
`run_reuse_oracle.py`, run `results/20260724-0001-phase3-reuse-oracle/`;
offline replay of the 32 Phase 1 atlas generation traces — no renders, no
GPU). With tier routing already failed at 3a and channel truncation demoted
to measurement-only at Phase 2, this closes the last intervention item; the
proposal is archived (`_archive/proposals/traj_latent_stats.md`). Phases 0–2
(recorder / atlas / gauge) stay shipped as observability.

The proposal's detector rule — freeze a token's velocity once its code has
been stable m steps, below a σ threshold — replayed against recorded ground
truth (open-loop, i.e. the intervention's *best* case: real reuse adds
compounding drift on top):

| m | σ< | skip% | oracle% | false-freeze% | final-code mismatch% | stale p50 / p95 |
|---|-----|-------|---------|---------------|----------------------|------------------|
| 2 | 1.0 | 61.4 | 25.5 | 56.4 | 69.0 | 12.7 / 37.4 |
| 3 | 0.7 | 32.1 | 20.5 | 29.9 | 45.1 | 4.6 / 11.1 |
| 4 | 0.5 | 13.9 | 13.6 | 15.3 | 25.9 | 2.0 / 4.7 |

(staleness in σ→0 noise-floor units; full 3×3 grid in the result envelope)

Both ways it could fail, it fails:

1. **The headroom is small even with impossible-in-practice perfect
   detection.** A retrospective oracle freezing each token exactly at its
   true commit step skips only **25.5 %** of token-steps at σ<1.0 (**13.6 %**
   at σ<0.5). That already bounds query+MLP savings; real wall-clock savings
   are lower still, since frozen tokens must stay in the K/V stream
   (dropping them is foveation redux — the refuted mechanism).
2. **The realizable online detector is unreliable at every setting.**
   Stable-for-m does not predict stable-forever — codes flicker. At the only
   cell with tolerable staleness (m=4, σ<0.5, ~14 % skip ≈ the oracle
   ceiling), **26 % of frozen tokens end the trajectory in a different
   quantization cell** than the one frozen, and median staleness is already
   2× the σ→0 activity floor. This extends the Phase 1 addendum's
   "detect, don't predict": online code-stability detection isn't reliable
   either.

The aggregate front-loaded commit-CDF is real (Phase 1 stands), but — same
shape as the 3a tier-routing failure — it does not convert into a per-token
intervention basis. A ~10 % best-case wall-clock saving does not justify
per-step dynamic token subsets fighting `compile_dynamic_seq`, a gauge run,
and P4t-class process risk.

**Decode probe closed unrun**: measurement-only, no process risk, but its
payoff is bounded — it only informs latent-storage compression, and the VAE
latent caches total ~3.0 GB / 3008 files (the rest of
`post_image_dataset/lora/` is TE/PE caches the probe says nothing about).
Trivially reopenable: `analyze_subspace.py` + the PCA directions in the
Phase 1 `subspace_addendum.json` are the needed inputs.

## Phase 2 — the trajectory-intactness gauge

**Verdict: PASS** (calibration `results/20260723-2130-phase2-recal/`, traces
from `results/20260723-2120-phase2/`; 4 prompts × 2 seeds × 1024², 28-step
**euler** CFG 4 — Euler everywhere because the foveated runner forces it).

`gauge.py --baseline <dir> --candidate <dir>` compares matched
(prompt, seed) trace grids and emits the per-σ divergence profile D(X) as
curves (x̂₀ RMSE, code mismatch, ΔE, hf-ratio, commit-CDF shift) plus
structural detectors, then a verdict band. Calibration landed **one
exemplar in each band**:

| arm | verdict | expected | evidence |
|---|---|---|---|
| SMC-CFG | **intact** | intact | all distributional metrics ≈ 0 (max ΔE 0.02) |
| Spectrum | **perturbed** | perturbed | ΔE −0.65 localized in its forecast band; no structural damage |
| foveation σ_c=0.75 | **process-broken** | process-broken | commit-CDF hole 0.168 + hf blow-up ~30× over 14 knots |

Three findings that shape how the gauge must be read:

1. **P4t is rediscovered from traces alone — but the in-loop signature is a
   blow-up, not a flatline.** The proposal predicted the periphery `hf`
   trace flatlining; in the recorded (full-grid) latents the group-shared
   periphery is piecewise-constant, so cell boundaries dump Laplacian
   energy — hf_ratio jumps to ~30× at the σ_c crossing and stays ≥14×. The
   *flatline* only exists in the final image, after the bicubic readout.
   The commit-CDF hole (−0.17 around σ=0.80: tokens that had committed in
   baseline get yanked by the merge rewrite) is the second, independent
   break signature. The flatline detector is retained for future
   interventions that smooth in-loop.
2. **Quality-neutral ≠ process-transparent.** Spectrum's own bench is
   quality-neutral, but its Chebyshev-forecast steps are process-visible:
   per-token activity collapses on forecast knots (ΔE dips to −0.65),
   because information genuinely does not flow through the DiT on those
   steps. "Perturbed" is the *correct* reading, and the pre-registered
   expectation was updated accordingly (documented in
   `run_gauge_calibration.py`). The falsifier-3 requirement — separate
   foveation from Spectrum — is met with wide margin (hf blow-up 28×
   vs 0.34 max deviation; commit hole 0.168 vs 0.090).
3. **Verdicts are driven by distributional metrics only.** Any intervention
   touching the combine (SMC) or the step rule produces large *pointwise*
   divergence (SMC: 65 % code mismatch at 8-step smoke) through ordinary
   trajectory chaos while leaving process statistics intact. x̂₀ RMSE and
   code-mismatch curves are reported as descriptive context, never verdict
   inputs.

Recorder change for the gauge: the sidecar now stores `tok` (token-level
normalized x̂₀, f16) so D's RMSE term is exact rather than k=4-code-coarse.
Phase 0 gates re-verified after the change (`results/20260723-2131-tok-reverify-1024/`):
bit-exact PASS, overhead **1.35 %** at 1024² (budget 2 %); sidecar ~6 MB.
The foveated runner (`networks/foveated.py`) now records via the same
side channel as the other runners.

Verdict bands live in `gauge.BANDS` — bench-calibrated, provisional; re-run
`run_gauge_calibration.py --reuse <run_dir>` after any band/logic change
(offline, no rendering).

Repro:

    uv run python project/sigma_lowres/bench/traj_stats/run_gauge_calibration.py --label phase2
    uv run python project/sigma_lowres/bench/traj_stats/gauge.py --baseline <dir> --candidate <dir>

---

## Phase 1 — the anime-domain atlas

**Run**: `results/20260723-2100-phase1/` (`run_atlas.py --label phase1`).
Generation arm: 8 prompts (mignon prompt-set families detailed×4 / sparse×2 /
no_trigger×2) × 4 seeds, 1024², 28-step er_sde, CFG 4 → 32 traces. Inversion
arm: DirectEdit inversion (`invert(..., traj_recorder=...)`, guidance 1.0,
same 28-knot schedule) of 8 real cached-corpus images, one per artist,
1024-tier band. Static column: 512 corpus latents, same quantizer (k=4) +
normalization, no trajectory. All traces canonicalized to σ-descending knot
order; commit recomputed from stored codes in that shared order.

### (a) Commitment is front-loaded — the n=1 preview survives aggregation

Aggregate commit-CDF (mean [p25, p75] over 32 generation traces):

| σ | 0.96 | 0.86 | 0.80 | 0.72 | 0.625 | 0.50 | 0.333 |
|---|---|---|---|---|---|---|---|
| gen | 0.03 | 0.12 | 0.18 [.09,.19] | 0.24 | 0.34 | 0.47 [.40,.50] | 0.65 [.61,.68] |
| inv | 0.01 | 0.06 | 0.11 [.07,.16] | 0.18 | 0.30 | 0.48 [.44,.52] | 0.74 [.71,.75] |

Roughly half of all tokens have taken their final k=4 code by σ=0.5, ~two
thirds by σ=1/3, with a tight cross-seed/prompt spread — **not** uniform in σ,
so falsifier 1 does not fire and the late-step headroom for Phase 3
inference items is real. E(σ) confirms from the activity side: at τ=q95 the
active-token fraction falls 1.00 → 0.73 (σ=0.78) → 0.29 (σ=0.50) → 0.12
(σ=0.26). τ-sensitivity (q90/q99 band): the curve shifts level (E@σ=0.50 =
0.38/0.29/0.18) but not shape — every τ in the band shows the same
front-loaded decay. Per-family: sparse prompts commit *earlier* (0.28 by
σ=0.80 vs 0.14 for detailed / no_trigger — less text signal to integrate,
earlier lock); detailed and no_trigger are indistinguishable. Effective
guidance (`guide`) is even more front-loaded: post-combine ‖v_f − v_u‖
drops 2.14 → 0.44 → 0.26 over σ=1.0 → 0.80 → 0.33, consistent with the
cross-attn front-loading finding (the `crossattn_drive` bench, since
deleted).

### (b) Channel usage is skewed ~4× and stable across arms

`cbits(c, σ_min)` (generation, mean over 32): high channels 13 (1.50),
4 (1.40), 11 (1.19), 15 (1.18), 5 (1.10) vs idle channels 8 (0.37),
14 (0.53), 6 (0.57), 10 (0.70) — a ~4× per-channel entropy range at k=4.
The ordering is essentially the corpus ordering (Spearman vs static column
0.94; inversion arm 0.98). The skew exists already at σ≈0.9 and the channel
*profile* is frozen from σ≈0.92 down (per-knot gen↔inv Pearson ≥ 0.89 below
σ=0.92) — channels don't take turns, the domain just uses a fixed subset
hard. σ-scheduled channel truncation therefore has a target, but a
compute-irrelevant one (16 latent channels are mixed into 1024-dim tokens at
patch embed; see proposal Phase 3 scoping).

### (c) Generation matches inversion — corpus statistics transfer

The structural-disagreement falsifier does not fire:

| cross-arm check | value |
|---|---|
| commit-CDF max gap (gen vs inv means) | 0.086 |
| E(q95) mean abs gap | 0.094 |
| cbits channel-profile Pearson, median over knots | 0.90 (≥0.89 for σ ≤ 0.92) |
| cbits channel Spearman at σ_min | 0.95 |
| static vs gen final cbits (Pearson / Spearman) | 0.95 / 0.94 |
| static vs inv final cbits (Pearson / Spearman) | 0.99 / 0.98 |

The only divergence is the σ→1 extreme (channel-profile r = 0.42/0.60 at the
two highest knots), which is expected and mechanical: at σ≈1 generation's
x̂₀ is the model's prior guess from pure noise while inversion's is the
nearly-destroyed image estimate. Below σ≈0.92 the two processes are
statistically the same object, so corpus statistics license
generation-side claims (not just img2img/editing ones). Inversion is
mildly *more* front-loaded at low σ (0.74 vs 0.65 at σ=1/3) — real images
lock detail slightly earlier than the sampler does.

Two side observations worth keeping: (1) real-image trajectories end at
higher `hf` than generated ones (0.20 vs 0.13 token-Laplacian energy) — the
generated corpus is measurably smoother than the training corpus under
identical normalization, a usable gauge baseline; (2) the static column's
within-image redundancy is substantial (modal joint-code share 0.11, unique
code fraction 0.26 at k=4) and skewed per image — the entropy-aware tier
routing input exists.

**Verdict**: exploitable structure confirmed on both axes the proposal
gated on (front-loaded commit-CDF, skewed cbits) and statistics transfer
from corpus to generation. Phase 2 (the intactness gauge) proceeds;
Phase 3's committed-token compute-reuse item keeps its audition, channel
truncation is demoted to measurement-only.

### Addendum (2026-07-23) — channel *subspace* + boundary predictability

`analyze_subspace.py` (offline, reads existing traces + cached latents;
json in `results/20260723-2100-phase1/subspace_addendum.json`). Two
questions the marginal (axis-aligned) atlas stats couldn't answer:

**The domain lives in a fixed ~3-dim channel subspace, and inference never
leaves it.** Channel-covariance spectrum of token-level normalized x̂₀:

| population | effective rank | top-4 var | top-8 alignment to static |
|---|---|---|---|
| static corpus (488k tokens) | 3.38 / 16 | 0.90 | — |
| generated, final x̂₀ | 2.75 / 16 | 0.94 | cos ∠ = 1, 1, 1, 1, 1, .999, .996, .97 |
| generated, mid-trajectory σ=0.5 | 2.67 / 16 | — | same (≥ .97 all eight) |

The Phase 1 cbits skew was the axis-aligned shadow of this: the anime
domain uses a fixed ~3–4-dim subspace of the 16-dim VAE channel space, the
*same* subspace during generation (even mid-trajectory) as in the clean
corpus. Caveats: token-level (2×2-pooled) statistics — per-pixel tails are
suppressed; and "statistics live low-dim" ≠ "decode quality survives
projection" — that test is the Phase 3 subspace-truncated decode probe.

**Which tokens commit late is NOT predictable a priori; when they commit
is.** Per-token commit-knot vs final-hf Spearman: mean 0.17 (range
0.05–0.41 over 16 traces); vs early-σ guide: 0.14. The "detail commits
late" intuition is a weak per-token signal, while the σ-direction
(population commit-CDF) is tight and systematic. Design consequence for
Phase 3 compute-reuse: **detect, don't predict** — online code-stability
detection is free and reliable; pre-drawn masks (foveation's shape) are
refuted a second time by this data.

Repro:

    uv run python project/sigma_lowres/bench/traj_stats/analyze_subspace.py

Repro:

    uv run python project/sigma_lowres/bench/traj_stats/run_atlas.py --label phase1

---

## Phase 0 — passive recorder: bit-exactness + overhead

**Verdict: PASS** (run `results/20260723-2042-phase0/`, 1024², 28 steps,
er_sde, CFG 4.0, seed 42; spectrum smoke arm included).

Implements Phase 0 of `_archive/proposals/traj_latent_stats.md` (PR #74): the
`--traj_stats` recorder (`library/inference/traj_stats.py`), hooked into the
main inline loop, the tiled loop, and `spectrum_denoise` via
`SamplerSideChannels.traj_stats`. Invariant tests: `tests/test_traj_stats.py`.

| gate | result |
|---|---|
| determinism control (2× recorder-off, bit-identical) | PASS (main + spectrum) |
| bit-exactness (recorder on vs off) | PASS (main + spectrum, both recorder runs) |
| overhead ≤ 2 % at 1024² | PASS — **0.12 %** main (0.56 ms/step), 0.31 % spectrum |

Sidecar: 2.4 MB npz (28 steps × 4096 tokens × 6 traces, k=4).

## Performance notes (the two traps)

1. **Never pass `float(sigmas[i])` to `record()`** — the float() read is a
   stream sync that kills the loop's CPU run-ahead (~28 ms/step, 5.8 %
   overhead). The baseline loop never syncs per step (its `float(sigmas[i])`
   reads all sit behind short-circuited `None` checks). Hook sites pass the
   0-d tensor; conversion happens once at flush. Isolated `record()` cost is
   0.39 ms/step.
2. **`np.savez`, not `savez_compressed`** — flush runs inside the generation
   wall time; zlib on the ~4 MB payload cost >100 ms per generation.

## First trace (Phase 1 preview, single prompt/seed — not yet a claim)

- E(σ) (τ = σ→0 activity floor, provisional 95th-pct): 1.00 at σ≈0.95 →
  0.85 at σ=0.80 → 0.49 at σ=0.69 → 0.15 at σ=0.33 → 0.05 at the end.
- commit-CDF: 10 % of tokens committed by σ=0.80, 32 % by σ=0.55, 56 % by
  σ=0.33 — i.e. commitment is meaningfully front-loaded on this render,
  which is the exploitable-structure direction. Phase 1 (seed × prompt grid
  + inversion arm) decides whether it holds corpus-wide.

Repro:

    uv run python project/sigma_lowres/bench/traj_stats/run_bench.py --with_spectrum --label phase0
