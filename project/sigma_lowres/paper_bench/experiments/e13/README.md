# E13 — resolving both ends of the σ curve (segmented grid)

| | |
|---|---|
| **Status** | **PLANNED** — instrument change + one GPU run |
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
sandbox); `make daemon-run` consumes `--label` from ARGS as the *job*
label, so the script never sees it — the run dir is timestamped.

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

## What lands in the paper

- Fig. 1a redrawn against `sigma_centers` with the endpoint detached —
  the cliff either survives as a measured rise or resolves into a smooth
  approach.
- Table 2 gains the dense end rows; the σ→1 limit statement in §4.4
  becomes measured rather than inferred from a single bin.
- If H2 falsifies, §4.7's fit is refit and the E5 residual claims
  re-scored — flag this before running, since it touches a headline.
