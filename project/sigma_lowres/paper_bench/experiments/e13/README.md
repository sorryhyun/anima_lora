# E13 — resolving both ends of the σ curve (segmented grid)

| | |
|---|---|
| **Status** | **DONE** 2026-08-01 — `runs/20260801-0125` (design B, 7.3 h). H1 falsified, H2 falsified (flattens), H3 confirmed, kill switch not triggered. See [Results](#results). |
| **Question** | Fig. 1a's curve is resolved by 8 uniform bins, but its two most load-bearing features — the σ→1 rise into the endpoint and the mid-σ peak's low-σ flank — each sit inside a *single* bin whose debiased read is at the estimator's resolution limit. What is the actual shape at both ends? |
| **Depends on** | [E1](../e1/) (paired debiased object, self-floors, the D→∞ endpoint anchors), [E8.1](../e8/) (ε\*), [E9](../e9/) (the I→0 mechanism this tests) |
| **Instrument** | `bench/run_sigma_probe.py` — needs segmented `--sigma_window` (below) |
| **In the paper** | Fig. 1a / Table 2 (the curve itself); §4.1 three reads; potentially §4.7 (E5's data-term fit is calibrated against this curve's shape) |

## Why — two under-resolved ends

The grid is 8 uniform bins on (0,1) plus an exact σ=1 endpoint bin
(`bin_sigmas`, `sigma_probe/kernel.py:52`). Both ends fail for the same
structural reason: **the feature is narrower than the bin**.

**High end.** The top bin spans [0.875, 1.0) and reports as "σ=0.94".
Its 8 stratified draws land at 0.875 … 0.984, and ‖ḡ(σ)‖ climbs 9.3 →
37.5 across that interval, so the accumulated bin gradient is dominated
by its top draws — the marker is neither the gap at 0.94 nor a flat
average. There is exactly **one sample** in the whole interval where the
rise happens, and the endpoint is a different probe mode (open markers).

**Low end.** The bottom bin spans [0, 0.125) and reports as "σ=0.06".
The curve then peaks at σ≈0.31 — which is exactly where the redraw floor
bottoms out (0.470) and ‖ḡ‖ is minimal (0.415), i.e. where the
attenuation correction is largest.

## What the free reanalysis already shows

Recomputed from `runs/20260729-0014-e1b-debiased-map/per_image.jsonl`
(reproduces Table 2 exactly; no GPU). Trimmed means, |Δ|>1.5 dropped.

| σ | route | raw paired | debiased paired | images with ĉ>1 |
|---|---|---|---|---|
| 0.31 | 896 | +0.083 ± 0.023 | **+0.208 ± 0.047** | 9/38 |
| 0.31 | 512 | +0.317 ± 0.033 | **+0.599 ± 0.073** | 2/38 |
| 0.81 | 768 | +0.094 ± 0.021 | +0.011 ± 0.022 | 36/39 |
| 0.94 | 768 | +0.061 ± 0.031 | +0.015 ± 0.034 | 28/38 |
| 0.94 | 512 | +0.258 ± 0.036 | +0.166 ± 0.061 | 13/36 |
| 1.00 | 768 | +0.114 ± 0.018 | +0.092 ± 0.012 | 3/36 |
| 1.00 | 512 | +0.326 ± 0.031 | +0.329 ± 0.030 | 0/36 |

Three things follow, none of which the current grid can settle:

1. **The σ→1 rise is real, not a debiasing artifact** — it survives in
   raw paired (arm − reenc) cosine units (768: +0.061 → +0.114; 512:
   +0.258 → +0.326). Good news for the figure; the *shape* of the
   approach is still unmeasured.
2. **The interior high-σ bins are at the resolution limit.** At
   σ ∈ [0.44, 0.94], 28–37 of 40 images have a negative debiased gap
   (ĉ>1, the correction over-shooting past the ceiling), versus 0–10 at
   the endpoint. The bins that would define the approach carry almost no
   signal above noise.
3. **The mid-σ peak may be manufactured by the correction.** In raw
   paired units 512 has a local *dip* at σ=0.31 (+0.317, between +0.445
   and +0.438); debiased it becomes the global *maximum* (+0.599). 896
   goes +0.083 → +0.208 at the same bin. This is where the reenc control
   itself is furthest off zero (unpaired debiased −0.11 at 0.31, −0.41 at
   0.19 — the overshoot §4.1 already flags). The pairing subtracts most of
   it, but a control that far out means the correction is outside its
   linear regime exactly at the peak.

(3) is the reason this is not a cosmetic run: **E5's held-out validation
fits the data term against this curve's shape**, including the mid-σ peak
and the 768 mid-σ dip that "sits outside the 95% band". If the peak's
location or height is partly estimator-driven, those numbers move.

Independent anchor: E1a's D→∞ endpoint extrapolation gives **+0.019 /
+0.056 / +0.304** (896/768/512) versus this grid's D=8 endpoint **+0.042 /
+0.092 / +0.329**. The *top* of the jump is itself ~2× inflated for the two
mild routes, so part of the visual severity is endpoint bias at D=8.

## Instrument change — segmented `--sigma_window`

Today `--sigma_window LO,HI` takes one interval and
`build_sigmas(bins, draws, endpoint, lo, hi)` returns a rectangular
`(bins, draws)` tensor. Everything downstream already derives from that
tensor — `total_draws = sigmas.numel()` (`run_sigma_probe.py:172`), the
seed-list length, `for b in range(n_bins)` in `grad_estimate_binned`, and
the reported `sigma_centers = sigmas.mean(dim=1)`. So:

**Supported by a ~15-line change**: multiple segments, each with its own
bin count, at a **global** draws-per-bin — concatenate one
`bin_sigmas(bins_k, draws, lo_k, hi_k)` per segment along dim 0. The
tensor stays rectangular; nothing else moves.

```
--sigma_window "0,0.1,5 : 0.1,0.9,8 : 0.9,1.0,5"      # LO,HI,BINS per segment
```
with `--bins` becoming the fallback for the single-segment form (back
compatible: `--sigma_window 0.5,1.0` keeps working). Validation: segments
sorted, non-overlapping, inside [0,1]; `args.bins` set to the total so
`result.json` and the arm-key seed-budget check
(`cli.py:432`) stay correct.

**Not supported without a ragged grid**: per-segment draws-per-bin. The
estimator iterates a rectangular `(bins, draws)`. Do not attempt it — vary
bin *density* per segment instead, and set `D` globally by the hardest
route.

Also owed: `plot_debiased_map.py` must stop assuming uniform bin width
(plot against `sigma_centers`, and draw the endpoint as a detached open
marker rather than a connected point — the current figure connects two
different probe modes with a line segment, which is most of why the last
leg reads as a cliff).

## Pre-registered predictions and decision rules

Frozen before the run.

**H1 (E9's mechanism).** E9 measured `I_768` going −0.31 in-window →
−0.014 at the endpoint, i.e. the gap is suppressed by negative B/C
interference that switches off as the data term dies. So gap(σ) over
[0.9, 1.0] should rise **smoothly and monotonically** to meet the
endpoint value.
- *Confirms*: the dense-window means rise monotonically and the top bin
  (σ≈0.99) lands within ε\* of the σ=1 endpoint read.
- *Falsifies*: a residual step at σ=1 larger than ε\* ⇒ the endpoint bin
  measures a different estimand than the σ→1⁻ limit, and every "the
  endpoint gap **is** the graph floor" claim (E1, §4.4) needs the
  limiting statement re-derived rather than assumed.

**H2 (peak reality).** If the mid-σ peak is real, the dense low-σ
segment resolves a smooth rise from σ=0 to a maximum, and the peak
location is stable against `D`.
- *Confirms*: peak location within one dense bin of 0.31 at both D
  settings.
- *Falsifies*: the peak flattens or moves as D rises ⇒ it tracks the
  floor minimum, not the physics; §4.7's data-term fit is refit against
  the corrected curve and the "768 dip outside the 95% band" claim is
  re-scored.

**H3 (896 reachability).** 896's entire σ→1 rise is 0.04, against
bin-level ε\* of 0.03–0.08 at N=40/D=8. Predicted: **not resolvable** on
this grid at any affordable D. If it isn't, say so in the paper as an
instrument limit rather than plotting an unresolved curve — the endpoint
sweeps needed D up to 64 to see this route at all.

**Kill switch.** If the reenc control's *unpaired* debiased gap stays
beyond ±0.15 in the dense low-σ segment at the higher D, the attenuation
correction is out of regime there and the low-σ portion of the map should
be published in raw paired units with the debiased read as an appendix
row — not the reverse.

## Cost

Calibrated on E1b (14 738 s actual; the model reproduces it to 4.1 h) and
checked against E9 (predicted 2.5 h, actual 3.8 h → ~1.5× overhead for
`--deterministic` plus arm-sum I/O). **Window width does not affect
cost** — σ is a scalar per draw. Cost is
`N × (bins+1) × D × arm-token-work`.

| design | bins | N | D | draws/img | est. wall (+overhead) |
|---|---|---|---|---|---|
| **A** dense both ends, full curve | 5+8+5 = 18 | 24 | 16 | 304 | 10.4 h (~15 h) |
| **B** dense both ends, lean | 4+6+4 = 14 | 24 | 12 | 180 | 6.1 h (~9 h) |
| **C** dense both ends, lean | 4+6+4 = 14 | 24 | 8 | 120 | 4.1 h (~6 h) |
| **D** high end only | 6 | 24 | 16 | 112 | 3.8 h (~5.7 h) |
| **E** low end only | 6 | 24 | 16 | 112 | 3.8 h (~5.7 h) |

**Recommendation: B.** It answers both ends in **one process**, which the
kernel-path rule requires for a single figure — D+E as separate runs cost
the same 7.6 h and produce two curves that may not be spliced. D=12 is the
compromise: D=8 is the resolution that already failed (finding 2 above),
D=16 doubles the bill for a variance gain that will not rescue 896 (H3)
anyway.

Splitting into D and E is only correct if the high end alone is wanted and
the low-σ question (H2) is deferred.

## Run commands

Instrument change first, then one submission. GPU work goes through the
daemon (a direct background GPU process is SIGKILLed by the harness
sandbox); at the time `make daemon-run` consumed `--label` from ARGS as
the *job* label, so the script never saw it — the run dir is timestamped.
(Fixed 2026-08-01: daemon-run flags are now prefix-scoped, `--label`
passes through.)

```bash
# design B
make daemon-run ARGS="project/sigma_lowres/bench/run_sigma_probe.py \
  --adapter output/ckpt/anima_soup_sincos.safetensors \
  --sigma_window '0,0.1,4 : 0.1,0.9,6 : 0.9,1.0,4' \
  --draws_per_bin 12 --endpoint_bin \
  --self_floor --deterministic \
  --demote_edges 896,768,512 --num_images 24 \
  --results_root project/sigma_lowres/paper_bench/runs --queue"

make daemon-wait JOB=<id>
```

Before burning GPU: a `--smoke` pass to confirm the segmented grid parses
and `sigma_centers` lands where intended (dense-end centers at
0.0125/0.0375/0.0625/0.0875 and 0.9125/0.9375/0.9625/0.9875, mid at
0.167…0.833, endpoint 1.0).

## Results

`runs/20260801-0125` — design B as specified (14 bins + endpoint, D=12, N=24,
896/768/512, `--self_floor --deterministic`), 7.3 h wall vs the 9 h estimate.
Figure: `runs/20260801-0125/gap_debiased_e13.png`.

**Scope warning, read first.** E13's probe set overlaps E1b's by **2 of 24
images**. Every *level* comparison against E1a/E1b — including this doc's
"the top of the jump is ~2× inflated at D=8" reanalysis note — is confounded
by probe set and cannot be settled here. E13 licenses **within-run shape
claims only**; those are clean, because one process measured every bin on the
same images, which is exactly why the kernel-path rule wanted one run.

Debiased paired means (Table 2 recipe), ±SEM:

| σ | 896 | 768 | 512 | reenc (unpaired) | ε\* |
|---|---|---|---|---|---|
| 0.0125 | +0.033 ± 0.008 | +0.064 ± 0.013 | +0.155 ± 0.033 | +0.038 | 0.022 |
| 0.0375 | +0.113 ± 0.011 | +0.175 ± 0.018 | +0.384 ± 0.028 | +0.018 | 0.030 |
| 0.0625 | +0.204 ± 0.024 | +0.273 ± 0.032 | +0.547 ± 0.041 | −0.008 | 0.053 |
| 0.0875 | +0.236 ± 0.033 | +0.316 ± 0.039 | +0.571 ± 0.040 | +0.011 | 0.064 |
| 0.1667 | +0.217 ± 0.043 | +0.319 ± 0.049 | +0.621 ± 0.086 | −0.064 | 0.080 |
| 0.3000 | +0.296 ± 0.036 | +0.266 ± 0.077 | +0.612 ± 0.059 | −0.136 | 0.097 |
| 0.4333 | +0.092 ± 0.026 | +0.231 ± 0.039 | +0.669 ± 0.061 | −0.166 | 0.065 |
| 0.5667 | +0.017 ± 0.010 | +0.112 ± 0.025 | +0.341 ± 0.058 | −0.106 | 0.042 |
| 0.7000 | +0.036 ± 0.016 | +0.038 ± 0.028 | +0.278 ± 0.032 | −0.118 | 0.046 |
| 0.8333 | −0.005 ± 0.009 | +0.053 ± 0.018 | +0.210 ± 0.047 | −0.083 | 0.030 |
| 0.9125 | +0.024 ± 0.022 | +0.063 ± 0.027 | +0.192 ± 0.043 | −0.067 | 0.044 |
| 0.9375 | +0.019 ± 0.010 | +0.055 ± 0.051 | +0.215 ± 0.069 | −0.067 | 0.084 |
| 0.9625 | −0.013 ± 0.015 | +0.020 ± 0.051 | +0.147 ± 0.049 | −0.054 | 0.080 |
| 0.9875 | −0.011 ± 0.023 | +0.084 ± 0.028 | +0.293 ± 0.060 | −0.025 | 0.045 |
| **1.00** | **+0.110 ± 0.041** | **+0.200 ± 0.059** | **+0.446 ± 0.069** | −0.048 | 0.098 |

### H1 — FALSIFIED. No smooth monotone approach; a residual step at σ=1.

Inside the dense high window the curve is **flat**, not rising. Per-image
paired against σ=0.9125, no route moves: 896 −0.037 (t=−0.9) … 512 +0.112
(t=+1.9) at σ=0.9875. E9's predicted monotone rise is absent.

The endpoint then sits **above** every window bin, per-image paired:

| route | Δ vs 0.9125 | vs 0.9375 | vs 0.9625 | vs 0.9875 |
|---|---|---|---|---|
| 896 | +0.086 (t=2.0) | +0.091 (t=2.2) | +0.123 (t=2.5) | +0.093 (t=2.1) |
| 768 | +0.143 (t=2.6) | +0.145 (t=2.9) | +0.180 (t=2.0) | +0.086 (t=1.6) |
| 512 | +0.255 (t=3.4) | +0.231 (t=3.0) | +0.299 (t=4.1) | +0.117 (t=2.4) |

The step is significant by the per-image paired test in **10 of 12** route×bin
comparisons (t ≥ 2.0). Against the endpoint's ε\* = 0.098 it is larger for 512
(all four bins) and 768 (three of four), but only one of four for 896 — the
paired test is the sharper instrument here, since it cancels image-level
variance. Either way the pre-registered falsification branch fires: the σ=1 bin
does not measure the σ→1⁻ limit, and §4.4's "the endpoint gap **is** the graph
floor" must be **re-derived rather than assumed**.

**But the mechanism is at least partly instrumental, and this is the load-bearing
caveat.** In *raw* paired units the same steps are +0.03 … +0.11 and **not one
is significant** (t = 0.60 … 1.76). The overshoot diagnostic says why: in the
dense high window 8–11 of 24 images (896) carry a negative debiased delta — the
correction over-shooting past the ceiling — versus **3/24 at the endpoint**
(512: 4/24 at σ=0.9125–0.9625 vs **0/24** at the endpoint). So the step is a
discontinuity between two
correction regimes, not a demonstrated physical one. The honest statement for
§4.4 is that **in raw paired units the entire high-σ region including the
endpoint is statistically flat at N=24/D=12** — the Fig. 1a cliff is not
reproduced as a resolved rise, and the debiased step rides on a correction that
is out of regime on the window side. Finding 2 of this doc reproduces at D=12.

### H2 — FALSIFIED (flattens). The peak is a plateau; but it is *not* a correction artifact.

The low end is now resolved and the news is clean: a smooth, strongly
significant monotone rise, **complete by σ≈0.09** — per-image paired, 896
+0.080 (t=6.3) then +0.091 (t=3.9) then +0.032 (t=1.2); 512 +0.229 (t=5.2),
+0.163 (t=5.4), +0.019 (t=0.6). The whole rise lives **inside E1b's first bin**
([0, 0.125) reported at σ=0.06), exactly as this doc predicted.

Above that there is no peak. Per-image paired against σ=0.30, the candidates
are indistinguishable: 768 gives 0.0875 +0.070 (t=0.9), 0.1667 +0.053 (t=0.6),
0.4333 −0.035 (t=−0.4); 512 gives −0.031 (t=−0.6), +0.046 (t=0.8), +0.040
(t=0.5). It is a **broad plateau over σ≈0.09–0.43** that falls off from σ=0.567
(512 −0.267, t=−3.6). 896 is the one route that declines earlier (0.4333 vs
0.30: −0.205, t=−4.1), so its plateau is ≈0.09–0.30.

Per the pre-registration, "flattens" ⇒ **§4.7's data-term fit is refit against
the corrected curve and the "768 dip outside the 95% band" claim is re-scored.**

The worry that motivated H2 — item (3), that the correction *manufactured* the
peak — is **not borne out**. The elevated mid-σ region survives in raw paired
units (512 raw +0.470/+0.486/+0.472/+0.353/+0.521 across 0.06–0.43). What
changes is the feature's *shape*: a plateau, not a peak at a locatable σ\*.

### H3 — CONFIRMED. 896's σ→1 approach is not resolvable.

All four dense high-σ bins sit inside ±ε\*: +0.024/+0.019/−0.013/−0.011 against
ε\* 0.044/0.084/0.080/0.045. As pre-registered, report this as an **instrument
limit** rather than plotting an unresolved curve. (896's *endpoint*, +0.110 ±
0.041, is resolved — only the approach is not.)

### Kill switch — NOT triggered.

The reenc control's unpaired debiased gap in the dense low-σ segment is
+0.038 / +0.018 / −0.008 / +0.011 — far inside ±0.15. The low-σ map is
published in **debiased units as primary**. This is a real improvement on E1b's
reanalysis (−0.41 at σ=0.19, −0.11 at 0.31): at D=12 the low-σ overshoot
largely vanishes, which is direct evidence it was a **D-resolution artifact**.

One caveat the kill switch does not cover: the control reaches −0.166 at
σ=0.4333 and −0.136 at 0.30 — outside the named dense segment, but sitting on
the **plateau's right edge**. The plateau's left half (σ≤0.17, control ≤0.064)
is the trustworthy part; the 0.30–0.43 shoulder carries a correction caveat and
should not be leaned on for a σ\* claim.

## The E5 refit — attempted 2026-08-01; E5's prediction survives, its *governor* does not

H2's falsification owed a refit of §4.7's data term. It was run. Headline:
**E5's predictive claim is NOT overturned by E13** — what breaks is the
ratio-governor comparison, for a normalization reason that is independent of
E13 and would bite any re-measurement.

`e5_holdout.py` / `e5_refit.py` gained `--tier1024_run` (default = the published
E1b run, so the shipped result is untouched) and were pointed at E13.

**Two bugs surfaced first, both fixed:**

1. **WLS had no bin-width term** (`w = 1/sem²`), which silently assumes every
   bin covers the same slice of the σ axis. True for a uniform grid, false for
   E13's: its end bins are 0.025 wide against 0.133 in the middle, *and* the
   dense low-σ bins carry the smallest SEMs — so **45% of the fit weight landed
   in σ<0.1, which is 10% of the axis** (E1b: 23.8%). This alone dragged
   p\* from 2.00 to **1.00**, the opposite grid boundary, which would have read
   as "E13 overturns the quadratic". It does not. Fixed via `bin_widths()`,
   normalized to mean 1 so it is **exactly inert on a uniform grid** — both
   scripts reproduce their published numbers bit-for-bit (verified).
2. `paired_stats` divided by `n` / `n−1` with no empty-bin guard (same defect
   as `plot_debiased_map.py`). Not hit at N=24, but latent for any thinner grid.

**With the weighting fixed, p\* returns to 2.00 — and the refit still fails:**

| | published (E1b) | refit (E13) |
|---|---|---|
| p\* | 2.00 | 2.00 |
| A_512 | 0.077 ± 0.010 | 0.149 ± 0.015 |
| A_896 | 0.008 ± 0.005 | **0.061 ± 0.006** |
| A_1120 (G9, untouched) | 0.007 ± 0.002 | 0.007 ± 0.002 |
| ratio-twin z | 0.14 | **8.17** |
| held-out 1024 RMSE | 0.092 | 0.354 (χ²/bin 151) |

### The prediction transfers across image sets — the governor does not

Fit on one run's images, predict route 768 as measured on the other's:

| fit on | predict | RMSE | oracle | verdict |
|---|---|---|---|---|
| E1b | E1b 768 | 0.093 | 0.105 | PASS |
| E1b | **E13 768** | **0.104** | 0.095 | **PASS** |
| E13 | **E1b 768** | **0.092** | 0.105 | **PASS** |
| E13 | E13 768 | 0.118 | 0.095 | PASS |

All four pass. **E5's "predicts routes it was not fit on at ~0.09 RMSE" is
intact under E13.** Do not record E13 as refuting it.

### Root cause of the z blow-up: A is not scale-invariant

E13's G(σ) runs **1.3–1.65× larger than E1b's** at mid-σ (different images →
different gradient norms). Since the design is x = m/G², a 1.4× G means x is
~2× smaller and the fitted A absorbs the difference. Refitting E13's curves on
E1b's G:

| route | A (E1b, own G) | A (E13, own G) | A (E13 on **E1b's G**) |
|---|---|---|---|
| 512 | 0.0765 | 0.1489 (1.95×) | **0.0849 (1.11×)** |
| 896 | 0.0076 | 0.0609 (8.01×) | **0.0280 (3.69×)** |

512's entire shift is G normalization (predicted G²-ratio 2.21× vs observed
1.95×). **Predictions are unaffected** because A·x preserves the product —
which is exactly why the transfer table above passes.

**The fragility is structural and pre-dates E13**: `m(σ)` comes from a *fixed*
run (G7) while `G(σ)` floats with whichever probe run supplies the curves. They
are not paired, so **A carries an unstated per-run normalization** and the
ratio-governor test — which compares raw A_896 against raw A_1120 — is not
scale-invariant. The published z = 0.14 rests on E1b's and G9's G happening to
sit on comparable scales. z = 8.17 reproduces at 8.2–8.7 under all three forms
(P/Q/X), confirming a normalization offset rather than a functional-form result.

**896's residual 3.69× is real curve change**, not normalization: E1b puts that
route's signal in the floor (F = +0.039, A ≈ 0) while E13 puts it in the slope
(A = 0.061, F = +0.007) — the low-σ rise E13 resolved.

The grid is not the driver. Restricting E13 to E1b-like coarseness barely moves
A (896: 0.0609 → 0.0737 mid+endpoint only; 512: 0.1489 → 0.1506), so the dense
ends contribute almost nothing to the amplitude.

### What would settle the curve-shape refit

**Re-run the segmented grid on E1b's exact 40-image probe list.** Then the
1024-tier legs are the *same images* as the published fit, G9 stays fixed, and
any movement in A/F **is** the resolution correction. The list is generated at
`e1b_probe_list.json` (40 images, from E1b's `result.json`; `--probe_list`
sets `num_images`, and `--self_floor` supports ≤50). With m, G and y all
describing the same images, A becomes comparable and the 896 F↔A
redistribution can be read as curve change rather than normalization.

**This rerun is [E14](../e14/)** (reserved 2026-08-01; consolidated same
day — there is no separate "e13b" submission). One process carries both the
probe-matched refit owed here and the E9-style B/C ledger arms
(`--repromote --keep_arm_sums`) that decompose the 896 low-σ plateau.
Command, instrument prep, pre-registered branches, and measured cost live in
E14's record, which supersedes the command formerly here.

Until then: §4.7 keeps the published fit — **it is not refuted**, the
prediction transfers (table above). E13's H1/H2/H3 verdicts stand on their own
as within-run shape claims and need no refit. The one thing that should be
written down regardless of the rerun is that **A carries a per-run G
normalization**, so the ratio-governor z is a scale-dependent statistic.

Artifacts: `runs/20260801-1034-e5-holdout-control2` (bit-identical control),
`-e13w` (width-fixed refit), `runs/20260801-1037-e5-refit-control` / `-e13`.

## What lands in the paper

Settled by the run; each item below is now **owed work**, not a prediction.

- **Fig. 1a redraw.** Done in-instrument (`plot_debiased_map.py` now plots
  against `sigma_centers` and detaches the σ=1 open marker). The cliff
  **resolves into a plateau-then-decline plus a detached endpoint**, not a
  smooth approach. Re-render the paper figure from this run.
- **Table 2** gains the dense-end rows above. The σ=0.06 and σ=0.94 rows were
  each hiding a feature narrower than their bin — the low-σ rise completes by
  σ≈0.09, and the high-σ window is flat.
- **§4.4 — re-derive, do not assume.** H1 falsified: the σ=1 read cannot be
  asserted as the σ→1⁻ limit. Preferred framing given the raw-units result:
  state the high-σ region as flat-within-resolution and the endpoint as a
  separate estimand, rather than claiming a measured rise in either direction.
- **§4.7 + E5 — re-score the shape claims, but the account stands.** H2's
  flattening fires the pre-registered consequence: the data term was fit
  against a peak at σ≈0.31 that is really a plateau over ≈0.09–0.43, so the
  "768 dip outside the 95% band" residual claim must be re-scored. **The refit
  attempt (below) found E5's *predictive* claim survives E13 intact** — this is
  a shape/residual re-score, not a refutation of the headline.
- **State that A carries a per-run G normalization** — the ratio-governor z is
  scale-dependent, which the refit attempt exposed and which holds regardless
  of E13.
- **896 as an instrument limit** (H3), stated as such — no unresolved curve.
- **Levels stay with E1a/E1b.** The 2/24 probe overlap means E13 must not be
  used to revise any endpoint *level*, including this doc's D=8 inflation note.
