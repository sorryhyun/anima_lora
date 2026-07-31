# E1 [GATE] — debiased gaps: self-floors + draw-count extrapolation

| | |
|---|---|
| **Status** | **DONE 2026-07-29** |
| **Verdict** | Decision rule 1 fired — **token-count floor CONFIRMED debiased** (512 gap_∞ +0.304). The endpoint gap **is** the graph/Jacobian floor; the apparent target-content share was estimator bias. |
| **Runs** | `runs/20260728-2302-e1a-drawsweep/`, `runs/20260729-0014-e1b-debiased-map/`, `runs/20260729-0420-e1c-xzero-endpoint/` |
| **Instrument** | `bench/run_sigma_probe.py` — `--self_floor`, `--draw_sweep`, `--deterministic` (all landed here) |
| **Consumed by** | [E5](../e5/) (E1b paired-debiased is its 1024-tier source), [E8](../e8/), [E7](../e7/); `plot_debiased_map.py` (Fig 1c) |
| **In the paper** | §4.1 instrument validation; the gap-native restructure of `main.tex` is written in debiased units throughout |

**Question.** How much of every reported gap (incl. x-zero and endpoint)
survives when estimator variance is equalized out?

**Instrument change** (`run_sigma_probe.py`, landed):

1. `--self_floor`: for every arm (reenc + each demote/pi/yarn arm) run a
   **second** independent draw set `g_d′` (`seeds(arm_idx′)`) and record
   `cos_self_<key> = cos(g_d, g_d′)` per bin.
2. Report, alongside the existing gap, the **debiased cosine**
   `ĉ = cos(n̄, d̄) / sqrt(cos_floor_native · cos_self_d)` (split-half
   attenuation correction; both native estimates already exist) and
   `debiased_gap = 1 − ĉ`. Raw gaps kept for continuity.
3. `--draw_sweep 4,8,16,32,64`: endpoint-only mode (`--bins 0
   --endpoint_bin`), reduced probe set (N=12, redundancy-stratified),
   fit `gap(D) = gap_∞ + c/D` per route, report `gap_∞` with a bootstrap
   CI over images. Nested seeds so the D=64 run contains the D=32 draws
   (one pass, prefix sums — no extra forwards).

## E1(0) — retroactive gap-vs-D scan [DONE 2026-07-28, free]

Existing endpoint-bearing runs at D ∈ {4, 8, 16} already showed the
confound signature. 1024→896 (N=40): +0.100 / +0.035 / −0.016 at
D=4/8/16 — a clean c/D decay (fit on D=4,8: c ≈ 0.52, gap_∞ ≈ −0.03,
predicting +0.002 at D=16, matching the measurement). 896's "≈0 floor"
is ≈0 *because* D=16 pushed the bias below the band. 1024→768: +0.167 /
~+0.10 / +0.08–0.13 — shrinks 4→8, scatter ~0.05 across same-D runs; if
c ≈ 0.5 carries over, the paper's D=16 floor +0.127 contained ~0.03 of
estimator bias (true floor ~0.09). 1024→512 existed only at D=16 (no
trend; largest expected c → most unmeasured bias). The D=2/N=4 smokes
put reenc at −0.19 — a same-grid control 5× outside the band on draw
noise alone. **Verdict: confound live, observed, with a fitted c on one
route — E1 was not optional.**

Measured cross-run sensitivity (2026-07-28 smoke twins, D=2, same seeds
and inputs): two runs sharing the warm inductor kernel cache agree to
|Δcos| ≤ 0.015 (atomics-order noise); a run with a different kernel set
(cold-autotune first compile) lands up to |Δcos| ≈ 0.29–0.36 away. So
per-bin cosines at low D are *kernel-path chaotic* — never compare them
across processes; every reported gap/floor/debias pairing must stay
within one run, which the instrument already guarantees.

Note on what the per-bin SEM band can and cannot do: the band is
cross-image scatter of the *biased* estimator — it tightens with N
around a number whose bias only shrinks with D. It licenses the
[E3](../e3/) non-inferiority criterion as pure reanalysis of
`per_image.jsonl`, but it cannot bound the variance bias; only demoted
self-floors can.

## Pre-registered decision rule

- `gap_∞(512) ≥ 0.15` debiased → token-count floor confirmed; paper
  strengthens (report both raw and debiased).
- `gap_∞(768)` debiased ≤ reenc band → 768's "never safe" at high σ was
  estimator variance; safety map and abstract rewritten.
- Everything collapses into the reenc band → headline becomes the
  low/mid-σ result + claim-narrowing.

## Results (2026-07-29)

Runs live in `paper_bench/runs/` (gitignore-exempt, committable — future
paper-bench runs pass
`--results_root project/sigma_lowres/paper_bench/runs`). Instrument:
`--self_floor` + `--draw_sweep` + `--deterministic` landed, det-twins
bit-exact, stats-overlap cut wall ~2-3×.

- **(a) endpoint draw-sweep, N=12, D=4..64 nested**
  (`runs/20260728-2302-e1a-drawsweep/`). Debiased gap_∞:
  reenc −0.003 [−0.017,+0.008]; 896 +0.019 [+0.010,+0.030]; 768 +0.056
  [+0.043,+0.071]; **512 +0.304 [+0.197,+0.424], 12/12 images > 0.15 →
  decision rule 1 fires: token-count floor CONFIRMED debiased.** Rule 2
  does not fire (768 paired vs reenc +0.054±0.009 > margin), but the
  published 768 endpoint floor +0.127 is ~half estimator bias (debiased
  ~0.056) — floor-table magnitudes rewritten. Native floor extrapolates
  to 1.005 [0.994,1.016]: the draft's "endpoint floor ≈ 0.85" was pure
  draw noise (R1 vindicated on the native floor). Debiased fits are
  D-flat (|c| ≤ 0.05 for 512 vs raw c ≈ +0.29) — the attenuation
  correction works as designed.
- **(b) verdict grid 8×8+endpoint, N=40, `--self_floor`**
  (`runs/20260729-0014-e1b-debiased-map/`). Caveat first: at D=8/bin the
  *unpaired* debiased estimator overshoots (reenc bins to −0.4 where
  floors are small, σ≈0.19–0.44) — the readable object is the **paired
  per-image difference (arm − reenc)**, |Δ|>1.5 dropped.
  Paired-debiased map: 512 unsafe at every σ (+0.08..+0.60). 896 unsafe
  σ<0.5, ≈0 in σ∈[0.56,0.94] (formal 0.02-UB pass only at 0.688 —
  bin-level ε\* at N=40/D=8 is ~0.03–0.08, see [E8.1](../e8/)), **small
  real gap at the exact endpoint (+0.042±0.011)** that raw analysis
  missed. 768 ≈ 0 in σ∈[0.69,0.94] (means −0.03..+0.015) but clearly
  gapped at the endpoint (+0.092±0.012) and everywhere σ<0.6 — "never
  safe" softens to "no certifiable window at current instrument
  resolution; means ≈ 0 in [0.69,0.94]". Shipped 896@σ>0.5 map:
  re-confirmed debiased except the σ=1.0 endpoint itself.
- **(c) x-zero endpoint sweep, N=40, D=4..32, `--self_floor`**
  (`runs/20260729-0420-e1c-xzero-endpoint/`). Debiased graph-term
  gap_∞: 896 +0.034 [+0.017,+0.058]; 768 +0.074 [+0.053,+0.094]; 512
  +0.283 [+0.232,+0.332] — statistically equal to (a)'s full-endpoint
  gaps at every route. **The endpoint gap IS the graph/Jacobian floor:
  the target-content share R2 flagged (raw 768 0.127 vs x-zero 0.064)
  was estimator bias, not content.** The paper's original "any endpoint
  gap is the floor by construction" survives in debiased units;
  [E2](../e2/)'s α-sweep is demoted from gate-adjacent to cheap
  confirmation (predicted α-slope ≈ 0).

**Outcome: decision rule 1 fired → Branch A** (`../../paper_plan.md` §5);
the gap-native restructure was written into `paper/main.tex` in debiased
units.
