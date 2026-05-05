# DCW v4 — target-window ablation + σ̂² shrinkage drop

**Status:** findings / decision (2026-05-05) · supplements [`dcw-learnable-calibrator-v4.md`](dcw-learnable-calibrator-v4.md) · **No retraining required to adopt the σ̂²-drop**; narrow-window deployment is gated on inference-side window plumbing (not yet wired).

## TL;DR

1. **Narrowing the supervision target from `7:` (full tail) to `7:14` lifts r_α by ~9% relative** (0.474 → 0.518 per-prompt CV r). The hot α-window across width-7 slices is **`7:14`, `14:21`, `21:`** — all clustered at r_α ≈ 0.52–0.55. Production `7:` (W=21) averages across the whole tail and dilutes per-window signal.
2. **Head supervision (`:7`) is strictly *worse* for α̂** (r_α=0.31). Falsifies the "supervise on early-step gap is more learnable" hypothesis at n=669 / 175-stem scale.
3. **σ̂² is *inverted***: best signal lives in the head window (`:7`, r_σ=0.44) — a region α̂ can't predict from. Current shared-target supervision is a structural compromise neither head wins.
4. **NLL is net-negative at every target window** — the σ̂² channel hurts more than it helps regardless of α-target choice. **Decision: drop σ̂²-based shrinkage at inference by default** (`--dcw_v4_disable_shrinkage` is now `True` by default; opt back in with `--no-dcw_v4_disable_shrinkage` once σ̂² passes Gate B).
5. **Landscape buckets fit ~50% better than portrait** at narrow target windows (`1248×832` r_seed=+0.70 vs `832×1248` r_seed=+0.46). Hidden in the production head's averaged 0.474. Partially re-opens [`project_dcw_bucket_prior_cosmetic`](../../.claude/projects/-home-sorryhyun-anima-anima-lora/memory/project_dcw_bucket_prior_cosmetic.md) for narrow-window heads.
6. **Inference dead-path bug fixed (separate finding).** A fresh `make dcw-train` revealed that `lambda_for_step` was producing λ=0 at every step regardless of α̂. Commit `9f02612` zeroed `bucket_prior_S_pop` (A2-cosmetic), but the inference formula still divided by `Σ(μ_g · S_pop)` → 0, the guard zeroed the entire correction. New formula `λ_i = alpha_gain · α̂ · μ_g[i] / Σ μ_g[i∈window]` (no S_pop dependency); a `--dcw_v4_alpha_gain` CLI flag (default 1e-2) carries the gap→λ scale. Window respects `target_start`/`target_end` from artifact metadata, so `tw7-14` heads no longer over-correct.
7. **Seed (intra-prompt) variance dominates the first ~4 steps then falls to a stable 13–14% floor.** Per-step decomposition on the clean 525-row pool: seed share = 73% at step 0, 34% at step 1, 22% at step 3, 18% at step 4, then 13–14% from step 6 onwards. The bench-1543 finding ("seed std ≈ between-prompt std") was specific to the pure-noise / early-step regime; by step 5 prompt variance is ~6× larger than seed. **Practical consequence: any feature or supervision that includes step 0 carries a 73% seed-noise burden; anything from step ≥4 inherits the same 13–14% floor as the production tail target.** v4's `k_warmup=7` warmup-observation window sits comfortably past the seed-dominated region — that design choice is now load-bearing-justified by data, not just intuition.

## Bench: k-supervision sweep

`bench/dcw/k_supervision_sweep.py` reuses the trainer's row loader, feature builder, `FusionHead`, and `train_one_fold` so the only thing varying is the **target window**. Observation `g_obs = v_rev_LL[:k_warmup=7]` is held fixed (matches the production inference contract and Spectrum's 7-step warmup, so observation is free-riding the warmup forward passes).

Full sweep result (8 windows × 8-fold prompt-stratified CV, n=669 rows / 175 stems):

| Window | W | r_α_mean | r_α_seed | r_σ | ΔNLL |
|--------|---|----------|----------|------|--------|
| `:7` (head)   | 7  | +0.308 | +0.317 | **+0.437** | −4.3% |
| `0:7`         | 7  | +0.296 | +0.310 | +0.427 | −3.7% |
| `7:14`        | 7  | +0.526 | +0.530 | +0.284 | **−26.9%** |
| `14:21`       | 7  | +0.522 | +0.522 | +0.273 | −20.6% |
| `21:`         | 7  | **+0.546** | +0.542 | +0.278 | −20.3% |
| `0:14`        | 14 | +0.409 | +0.418 | +0.322 | −8.9% |
| `7:` (prod)   | 21 | +0.474 | +0.485 | +0.329 | −21.8% |
| `14:`         | 14 | +0.505 | +0.509 | +0.300 | −21.1% |

Run dir: `bench/dcw/results/20260505-1009-k-sup-sweep-baseline/`.

**Reading.** Width-matched (W=7) windows isolate where the prompt+observation features carry α-signal. Head (`:7`, `0:7`) is dead; mid/late (`7:14`, `14:21`, `21:`) cluster tight at r ≈ 0.52–0.55. Production tail (`7:`, W=21) at 0.474 is mid-pack — averaging across the whole tail dilutes the per-window signal that mid-tail windows alone would expose. σ̂² inverts the pattern: best on `:7` (r_σ=0.44), worst where α̂ is best.

## Ablation: trainer with `--target_window 7:14`

The trainer (`scripts/dcw/train_fusion_head.py`) gained a `--target_window` flag (defaults to `k_warmup:n_steps` = current production behavior). Run with `--target_window 7:14 --label tw7-14` reproduces the bench number end-to-end and writes a real `fusion_head.safetensors` with `target_start=7`, `target_end=14` in metadata.

| | prod `7:` (W=21) | tw `7:14` (W=7) | Δ |
|---|---|---|---|
| r_α_mean (per-prompt) | +0.474 | **+0.518** | +0.04 (+9%) |
| r_α_seed (seed-cond) | +0.485 | +0.522 | +0.04 |
| r_σ (seed-var) | +0.329 | +0.261 | −0.07 |
| ΔNLL improvement | −21.8% | −20.9% | ~flat |

Per-aspect r_seed (from the `tw7-14` head):

| Aspect | n | r_seed |
|--------|---|--------|
| `1248×832` (HD landscape) | 217 | **+0.697** |
| `1152×896` | 105 | +0.598 |
| `832×1248` (HD portrait) | 137 | +0.458 |
| `768×1344` | 105 | +0.456 |
| `896×1152` | 105 | +0.407 |

Artifact: `output/dcw/20260505-1022-v4-fusion-head-tw7-14/`.

**Trade-off shape.** Narrowing the target tightens α-fit (point prediction gets ~9% better r) at the cost of σ̂² fit (per-prompt seed-variance prediction gets ~21% worse r). Net NLL is roughly unchanged because both shifts roughly cancel in the Gaussian NLL composite. The α-lift is real but small; deploying it requires inference-side window plumbing — see "Open" below.

## Decision: drop σ̂² shrinkage at inference by default

The σ̂² channel **fails Gate B at every target window** in the sweep — NLL improvement is negative across all 8 windows tested, meaning the head + σ̂² composite predicts worse than a constant `N(0, σ²_pop)` baseline despite α̂ alone having r ≈ 0.5. The σ̂² output is miscalibrated enough that applying its shrinkage `σ²_prior / (σ²_prior + σ̂²)` to α̂ at inference is net-harmful.

**Change landed:** `--dcw_v4_disable_shrinkage` is now a `BooleanOptionalAction` with `default=True`. Opt-in to shrinkage with `--no-dcw_v4_disable_shrinkage`. The redundant explicit flag in `cmd_test_dcw_v4` and `cmd_test_dcw_v4_spectrum` was removed (no behavior change at the make-task surface — those tasks already passed the flag explicitly; the change just makes "drop σ̂² for now" the harness-wide stance for any caller of `inference.py`).

**Reverse condition.** Re-enable once σ̂² passes Gate B (r_σ ≥ 0.4 *and* NLL improvement ≥ +15%). The path most likely to get there is **A3 multi-seed data collection** (200 prompts × 3+ seeds, aspect-balanced) + supervising σ̂² on a window where its signal lives (`:7` head window) separately from α̂'s (`7:14` mid window). That's a v5 architecture change, not a tuning fix.

## Seed vs prompt variance decomposition

Run `bench/dcw/plot_seed_band.py` on the clean pool (525 rows = 5 aspects × 35 prompts × 3 seeds) — analogous to `make dcw`'s `gap_curves.png` but separating intra-prompt (seed) from inter-prompt bands. Plot saved to `output/dcw/seed_band_curves.png`. Per-step seed share = `σ²_seed[i] / σ²_total[i]`:

| Step | Seed share | gap mean | Notes |
|--|--|--|--|
| 0 | **72.7%** | −2.2 | Pure noise; prompt conditioning hasn't engaged. |
| 1 | 34.3% | −42.8 | Drops sharply as text cross-attention bites. |
| 2 | 33.5% | −34.6 | |
| 3 | 22.0% | −27.9 | |
| 4 | 18.5% | −16.8 | First step under 20%. |
| 5 | 16.3% | −8.1 | |
| 6 | 14.6% | −2.6 | Reaches the steady floor. |
| 7–27 | 13–15% | rises to +21 | Stable plateau through end. |

Integrated tail seed share for various window starts (matches the trainer's `--target_window k:N` semantics):

| `target_start` | Integrated seed share |
|--|--|
| 0 | 30.2% |
| 1 | 17.7% |
| 4 | 13.9% |
| 7 (production) | 13.5% |
| 14 | 13.5% |
| 21 | 13.7% |

**Per-step share (30%) is higher than integrated share (13%) because seed noise is more step-uncorrelated than prompt structure.** Prompt drift correlates across steps (the AR-1-ish recursion), so when you integrate over 21 tail steps prompt signal accumulates coherently while seed noise partially cancels. The integration is itself a noise filter — which makes the v4 architecture's choice of integrated tail target a good fit for the data scale, not just a convenient parameterization.

**Practical takeaways:**

- Step 0 carries 73% seed noise — almost unusable as a standalone feature. The v4 head already wisely *observes* g_obs from `[0:7]` (where the LL norm is itself informative even if noisy) but never *supervises* on a single early step.
- Any `target_window` starting at `k ≥ 4` inherits the 13–14% floor; no benefit from waiting longer (e.g. `7:` and `14:` and `21:` all sit at 13.5–13.7% integrated). So the target-window choice from this doc's [bench section](#bench-k-supervision-sweep) (favoring `7:14`, `21:`) is orthogonal to the seed-variance-floor question — those windows pick up on *signal-shape concentration*, not on seed-noise reduction.
- `k_warmup=7` for the observation window is past the seed-dominated region (step ≤4) by 2 steps — robust to small seed-share fluctuations. Could probably drop to `k_warmup=5` without measurable seed-noise increase, saving 2 forward passes if Spectrum's warmup ever decouples.
- The bench-1543 ("seed std ≈ between-prompt std") finding was specific to the pure-noise / very-early step regime where seed dominates. Generalizing it to the whole trajectory was the framing error in `project_dcw_seed_variance_dominates`.

## Findings worth keeping

- **g_obs at n=175 carries little marginal signal.** Capacity ratio 2.08:1 vs c_pool but no CV uplift; bottleneck is supervision-side variance from single-seed labels, not architecture capacity. Already captured in [`project_dcw_seed_variance_dominates`](../../.claude/projects/-home-sorryhyun-anima-anima-lora/memory/project_dcw_seed_variance_dominates.md). The k-supervision sweep didn't displace this — the `7:14` lift comes from where the signal lives on the timestep axis, not from the observation channel finally working.
- **Aspect signal is real on narrow windows.** The 0.46–0.70 spread in per-aspect r_seed (and the n=217 advantage for HD landscape) are loud enough that a per-aspect residualization or aspect-conditioned head — both ruled "cosmetic" by [`project_dcw_bucket_prior_cosmetic`](../../.claude/projects/-home-sorryhyun-anima-anima-lora/memory/project_dcw_bucket_prior_cosmetic.md) on the aggregate `7:` head — should be re-tested at narrow windows. The previous ablation may have been signal-diluted by the wide target.
- **All net-negative NLLs come with positive r_α.** The head's *predictions* are useful; its *uncertainty* is not. Ship the α-channel, drop the σ̂²-channel until the data scale supports it.

## Open / not yet done

1. ~~**Inference-side window plumbing.**~~ **Done 2026-05-05** — `library/inference/dcw_v4.py::lambda_for_step` now reads `target_start` / `target_end` from artifact metadata (falls back to `(k_warmup, n_steps)` for legacy artifacts) and applies λ only on `[target_start, target_end)`. The `tw7-14` artifact is no longer research-only.

   Same patch also fixed a separate dead-path bug surfaced when the user ran a fresh `make dcw-train`: the pre-fix tail formula was `head_corr = −α̂ · μ_g[i] / Σ(μ_g · S_pop)`, but commit `9f02612` zeroed `bucket_prior_S_pop` in the artifact (A2-cosmetic finding), making the denominator 0 and the guard zero out the whole correction → λ = 0 at every step regardless of α̂. The new formula
   ```
   λ_i = alpha_gain · α̂ · μ_g[i] / Σ_{j∈[target_start, target_end)} μ_g[j]
   ```
   distributes α̂ proportionally to μ_g across the supervised window with no S_pop dependency. The gap-units → λ-gain-units conversion that S_pop carried is now an explicit `--dcw_v4_alpha_gain` flag (default 1e-2). With α̂≈100 and the current artifact's μ_g profile, default-gain λ ramps from ≈ +0.007 at step 7 to the +0.05 cap by step ≈16 — roughly matching v2 historical lam_scalar magnitudes.
2. **Per-step sweep.** `bench/dcw/k_supervision_sweep.py --per_step` runs CV for each individual step `t ∈ 0..N-1` and produces an `r_α(t)` curve. Not run yet (~28× the window-sweep cost). Worth running once before any v5 architecture change so the target-window choice is informed by the per-step signal shape, not just chunked windows.
3. **σ̂² rescue (split-window supervision).** Train α̂ on `7:14` and σ̂² on `:7` from the same head with separate target tensors. Keeps the two-trunk architecture from `9f02612` but feeds each trunk the window where its signal lives. Falls under v5 scope.
4. **A3 multi-seed collection.** The actual fix for both σ̂² (Gate B) and r_α-ceiling (single-seed label noise). Spec: 200 prompts × 3 seeds × 5 aspects ≈ 3000 rows, aspect-balanced. Not started.
5. **`alpha_gain` calibration.** Currently a hand-tuned scalar; the principled value is `tail_norm / Σ_{j∈window}(μ_g[j] · S_pop[j])` from a one-shot population-level S_pop measurement. Until that's done, treat the default 1e-2 as a placeholder and tune empirically per artifact.

## Reproduction

```bash
# Re-run the window sweep
uv run python bench/dcw/k_supervision_sweep.py --label baseline

# Re-train the narrow-window head
uv run python scripts/dcw/train_fusion_head.py --target_window 7:14 --label tw7-14

# Re-run with σ̂²-shrinkage explicitly enabled (to compare)
make exp-test-dcw-v4 -- --no-dcw_v4_disable_shrinkage
```

## Memory updates needed

- Update `project_dcw_v4_direction.md` to note the σ̂²-drop decision and the `7:14` target-window finding.
- Update `project_dcw_bucket_prior_cosmetic.md` with the caveat: cosmetic on aggregate full-tail head, but per-aspect spread is real on narrow target windows.
- New: `project_dcw_target_window_signal_shape.md` — pointer to this doc for the per-window r_α numbers and the α/σ̂² inversion finding.
