# E4 — the end-to-end A/B

| | |
|---|---|
| **Status** | **DONE 2026-07-30** — exercise grid + 5th arm; residuals retired |
| **Verdict** | The projected "~14% ceiling" is now **measured: −14.6% wall / −15.1% FLOPs** for the shipped gate; sigma896's render deltas sit **inside the seed lottery**, and it is the only arm at-or-inside the yardstick on both corpora. No quality verdict is read from the exercise-N CMMD pass. |
| **Runs** | `runs/20260729-1537-e4-manifest/` (frozen manifest) · `runs/20260729-2148-e4-eval-sfw-s100{1,2,3}/`, `-cfg4-s1001/`, `runs/20260729-2137-e4-eval-s1001/` (evals) · `runs/20260729-2148-e4-yardstick/`, `runs/20260730-e4-yardstick-5arm/` · `runs/20260730-e4-fig-candidates{,-20steps,-30steps}/` |
| **Scripts** | `e4_manifest.py` (frozen design) · `e4_flops.py` (exact FLOPs) · `e4_render_eval.py` + `e4_fig_render.py` (renders/scoring/sheets) · `e4_seed_yardstick.py` (seed-noise yardstick) — committed as `4f538d7e`, figure sheets `9430c182` |
| **Companion** | [`claim_accumulated_bias.md`](claim_accumulated_bias.md) — what an in-band per-step verdict does and does not certify |
| **In the paper** | §5 Cost accounting (measured numbers), `tab:yardstick` + the Sample-level footprint paragraph; the "not yet run" claim is narrowed to the CMMD gate |

*Residual retirement (2026-07-30): the full-band CMMD rescoring and
val-loss/peak-mem additions formerly tracked as open work are closed
out — the 5-arm yardstick + narrowed §5 claim ("not yet run" limited to
the CMMD gate) stand as the shipped read, and the paper does not gate on
a full-band CMMD verdict. If the σ-gate-vs-896only tension (below) is
ever re-litigated, that rescoring is the instrument to revive.*

## Design realized

Frozen manifest `runs/20260729-1537-e4-manifest/` +
`launch_20260729_exercise.md` amendments: **4 arms** — native /
sigma896 (σ>0.5 gate + yarnsig) / **896only** (added: threshold 0 on the
safe route — isolates the σ-gate) / unsafe768 (threshold 0 on 1024→768,
the review's negative control, via the new `--sigma_lowres_route`,
commit `5b63ebb9`) — × **3 seeds** (1001–1003) × 2 artists (hews 60-stem
train / 8 ep; channel\_(caststation) 15-stem / 32 ep; 480 steps bs 1
each), `--deterministic --paired_step_rng`, stock lora recipe. 24
checkpoints. In-vivo CRN check: sigma896 demoted the **identical
244/480 step set on both artists** (σ stream is seed-keyed, not
data-keyed).

## Throughput — the paper's headline number is now measured

n=3 means; exact FLOPs via `token_step_hist` × FlopCounterMode
(`e4_flops.py`); wall tracks FLOPs ~1:1.

| arm | fwd PFLOPs | Δ | wall | Δ |
|---|---|---|---|---|
| native | 8.64 | — | 388 s | — |
| sigma896 | 7.35 | **−15.1%** | 331 s | **−14.6%** |
| 896only | 5.99 | −30.8% | 266 s | −31.4% |
| unsafe768 | 4.13 | −52.2% | 185 s | −52.3% |

The "projected ceiling of ~14%" reads as **measured −14.6% wall /
−15.1% FLOPs** for the shipped gate.

## Sample-level defensibility (seed-noise yardstick)

`runs/20260729-2148-e4-yardstick/`: within-seed cos(native~sigma896) vs
cross-seed cos(native~native) on the frozen (prompt, gen_seed) grid —
channel **0.9641 vs 0.9541 (inside the seed lottery)**; hews 0.9551 vs
0.9558 (boundary tie). Headline: *swapping σ>0.5 steps to the 896
sibling perturbs renders about as much as changing the training seed.*
Arm orderings shuffle between seeds — single-seed visual impressions
were substantially seed lottery.

## Negative control did its job by exposing the metric

At exercise N (9–12 SFW prompts, rating-mismatched pools) CMMD cannot
separate the known-bad route from anything (unsafe768 ≈ native on
channel, ≤ sigma896 on hews) → **no quality verdict is read from this
pass**; CMMD non-inferiority needs the full-band rescoring (retired
residual). Also banked: Δ(member−holdout) > 0 everywhere (no
memorization pathology); figure sheets committed (`9430c182`: 10607820
main, 14296235 + 8508115 appendix).

## Open tension for the gate story

sigma896~896only is the closest arm pair (hews s1001 cos 0.977) and
896only is another 16% cheaper — the σ-gate is endpoint-invisible at
this recipe, so its justification currently rests entirely on the
σ-resolved per-step certification
([`claim_accumulated_bias.md`](claim_accumulated_bias.md)'s
accumulated-bias question, now with an empirical handle: the
full-protocol quality read of sigma896 vs 896only).

## 2026-07-30 addendum — 5th arm `sigma768`

σ>0.5 gate + yarnsig on 1024→768: the off-map route under the paper's
own gate, deconfounding unsafe768's three-knob difference (jobs
20260730-0835\*, ~11 min for all 6). SFW evals rescored 5-arm in place
(renders resume-reused); yardstick rerun →
`runs/20260730-e4-yardstick-5arm/` (4-arm original kept). Means vs
native (hews / channel): yardstick 0.9558/0.9541, sigma896
**0.9551/0.9641**, 896only 0.9504/0.9500, sigma768 0.9503/0.9553 —
**sigma896 is the only arm at-or-inside the yardstick on both
corpora**; both controls fall below on ≥1 corpus (ordering consistent
with the map, margins small). Route-only pair sigma896~sigma768:
0.9488/0.9594. Paper §5 updated: measured −14.6%/−15.1% in Cost
accounting, new `tab:yardstick` + Sample-level footprint paragraph
(sigma768 shown as the off-map control instead of unsafe768), "not yet
run" claim narrowed to the CMMD gate; `bolya2025perception` added to the
bib. Fig candidate sheets with the sigma768 column:
`runs/20260730-e4-fig-candidates-20steps/` (arm-only addition — other
columns pixel-identical to the committed sheets).
