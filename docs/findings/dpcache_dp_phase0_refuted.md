# DPCache DP-planned Spectrum schedule — Phase 0 REFUTED, line closed

**Verdict (2026-07-24): falsifier #1 fired, decisively.** Globally-planned
skip-schedule placement adds nothing over Spectrum's content-blind growing
window at Anima's step counts — the window's realized schedule is not merely
competitive, it is **exactly locally optimal** under the true open-loop replay
cost. Nothing gets built; `--spectrum_schedule` stays `window` (default) +
`sea` (opt-in). Proposal: `dpcache_dp_schedule.md` (PR #74, never merged — not in tree). Bench: `_archive/bench/spectrum_dp/run_bench.py`, results in
`_archive/bench/spectrum_dp/results/20260724-0043-phase0/`.

## Setup

Per the proposal's Phase 0 — pure offline paper-check, no inference-path code.
12 prompts (the archived SEA-bench set), shipped geometry: 28 steps, CFG 4,
flow_shift 3, 1024², euler, `--compile_blocks`. All arms pinned to the SAME
live stop rule (`stop_at = n−3 = 25`, the discipline from the SEA P1
stop-mismatch artifact) and matched to the window's realized decision-region
refresh count **K = 7** (16/28 total forwards).

- **PACT** built open-loop per the paper: segment cost = cumulative rel-L1
  Chebyshev-forecast error over skipped steps, predictor conditioned on the
  warmup prefix + last two keys (a good match for Spectrum's forecaster — at
  `w=0.3` the prediction is 70% two-point Taylor). Pooled over prompts ×
  cond/uncond branches; DP over (prev-key, cur-key) states.
- **Scorer** (same for every arm): open-loop schedule replay — predictors
  update only at that schedule's actual steps, cached steps get forecast + real
  head replay + CFG combine, cost = SEA-filtered x̂₀ rel-L1 vs the full-compute
  x̂₀, summed. This is DP's home turf: no closed-loop drift, no per-prompt
  adaptivity penalty, cost model and scorer built from the same trajectories.

## Results (mean S over 12 prompts, lower = better)

| arm | S | note |
|---|---|---|
| **window** (shipped default) | **0.808 ± 0.201** | `AAAAAA.A.A.A.A..A..A..A..AAA` |
| oracle-from-window | 0.808 | **= window: all 84 single-move neighbors worse** |
| oracle-from-dp (local search) | 0.836 ± 0.175 | converges above window, 65 evals |
| sea (per-prompt δ, matched K) | 0.847 ± 0.209 | δ cv 0.016 — portability re-confirmed |
| dp / dp_tailfree | 0.876 ± 0.210 | win rate vs window 2/12, vs sea 4/12 |
| greedy_c1 (single-step costs) | 2.600 ± 0.545 | catastrophic — see below |

**Gate: FAIL** (dp must beat window and at least tie sea; it lost both).

## What the arms actually showed

1. **The growing window is a local optimum of the true cost.** Local search on
   the replay cost starting from the window mask finds no improving single
   move (85 evals, 3 passes). Starting from the DP mask it improves DP but
   converges *above* the window. At 28 steps with warmup 6 and stop 25, the
   free region is 19 slots × 7 forwards — the monotone growing cadence already
   sits at (a) local minimum of the placement landscape. There is no headroom
   for *any* fixed-schedule planner here, regardless of cost model.
2. **The tail-reallocation premise is dead (again).** The tail-free DP —
   freed from stop-forcing at matched *total* compute — voluntarily re-places
   3 forwards below σ=0.45, exactly reproducing the forced tail. Combined with
   the SEA P1 stop-mismatch finding, the "σ<0.45 tail carries ≈0% skip-cost,
   reallocate it" line from `seacache_sea_decision_metric.md` is now settled
   twice over: the SEA-weighted *input-distance* is ≈0 there, but the
   *injected x̂₀ error* of skipping those steps is not.
3. **Consecutive-skip compounding dominates placement quality.** `greedy_c1`
   (cache the steps with the cheapest single-step counterfactual cost — the
   no-segment-model control) is 3.2× worse than the window: single-step costs
   decrease monotonically in step, so greedy front-loads all forwards and dies
   on the long cached run. Any credible cost model here must price segments,
   and once it does, it rediscovers roughly the window's spacing — the DP's
   planned mask differs from the window by only ~2 step positions, and even
   those shifts *hurt*.
4. **SEA at matched K also sits slightly above the window** on this scorer
   (0.847 vs 0.808) — consistent with the shipped eyeball-near-tie verdict and
   with content-adaptivity being real but modest (~15% wiggle on a fixed
   σ-trend). No arm beat the blind cadence.

## Guards (don't re-propose)

- **No global skip-schedule planners** (DPCache / TeaCache-style placement
  optimization) for Spectrum at 24–30 step counts. The paper's headroom lives
  at T=50 with small K; at Anima's geometry the placement landscape has no
  exploitable structure the window doesn't already capture. A Phase-1
  closed-loop PACT could only score *worse* than this open-loop check.
- **No tail reallocation** via relaxed `stop_caching_step` — the planner keeps
  the tail when free to move it.
- **Single-step skip-cost rankings must not drive schedules** (greedy_c1
  disaster). Anything schedule-shaped needs segment-cumulative costs.
- The only surviving schedule axis is per-prompt/per-step *adaptivity* (SEA's
  lane, already shipped opt-in). The dp+sea hybrid died with its floor: the
  plan half contributes nothing over the window, so the hybrid reduces to
  window+sea, which is just `sea`.

## Reusable artifacts

- `_archive/bench/spectrum_dp/run_bench.py` — trajectory capture (both CFG branches +
  timesteps + x_t), disk-cached to `output/spectrum_dp_traj/` (~1GB/prompt;
  safe to delete, regenerates deterministically); open-loop schedule-replay
  scorer (`replay_cost`) that can score ANY future schedule mask against the
  shipped ones; batched `predict_many` for the Chebyshev forecaster; the
  PACT builder + DP planner.
- The scorer is the cheap first gate for any future when-to-skip idea: if a
  candidate schedule doesn't beat `window` on open-loop replay at matched K,
  it will not survive a real A/B.
