# paper_bench — completed experiments (discharge record)

Split out of `required_experiments.md` on 2026-07-29 so that file holds
only what still has to run. This file is the record of what already
landed, with results and outcomes. `paper_plan.md` stays the manuscript
plan.

---

## Review triage (2026-07-28) — verdict on the external review

Triage of the 2026-07-28 external review (ChatGPT), verified against the
actual instrument (`bench/run_sigma_probe.py`), the report
(`bench/report.md`), and `paper/main.tex`.

Verified correct (checked in code/repo, not taken on faith):

- **R1 — estimator variance confound is real and unaddressed.** In
  `run_sigma_probe.py` the floor is `cos(g_native_a, g_native_b)` (two
  independent draw sets, `seeds(0)`/`seeds(1)`); every demote arm gets
  **one** estimate and `gap = floor − ½[cos(a,d)+cos(b,d)]`. There is no
  demoted/demoted self-floor anywhere in the codebase. If per-draw
  gradient variance grows as token count falls (MSE averages over fewer
  elements), an iso-direction null still produces a positive gap that
  grows as the target grid shrinks — the same signature as "absolute
  token count sets the floor." Back-of-envelope with the measured
  endpoint floor ≈ 0.85 (run `20260727-2225`) and noise ∝ 1/tokens: a
  pure-variance null predicts spurious endpoint gaps ≈ 0.02 / 0.05 /
  0.15 for 896 / 768 / 512 vs measured −0.01 / 0.13 / 0.33. So the
  confound plausibly explains **~40% of the 768/512 floors**, not zero
  and not all. This is the one critique that genuinely gated the paper.
  Note the review under-claims in one spot: the x-zero probe is subject
  to the **same** confound (single demoted estimate), so x-zero is not a
  clean rescue of the graph term either — E1's debiasing had to be
  applied to x-zero too.
- **R2 — σ=1 is not graph-only.** `target = noise − lat`: at σ=1 the
  input is pure ε but the target still carries x per arm. The paper's own
  Table (floor table) showed it: 768 endpoint 0.127 vs x-zero 0.064 —
  **half** the 768 endpoint gap looked like target-content, yet the text
  said "any gap *is* the floor by construction" and only highlighted the
  512 route (where endpoint ≈ x-zero and the claim holds). *Resolved by
  E1(c): the apparent target-content share was estimator bias — see
  below.*
- **R3 — "never safe" is aggregation-dependent.** Confirmed in
  `bench/report.md` (pool4 addendum): pooled gap_768 ≈ 0 at σ ≥ 0.875,
  pooled gap_896 ≈ 0 at σ ≥ 0.625. The per-image and batch-SGD objects
  genuinely disagree at high σ; the safety map must state which object it
  is a map *of*, and the trainer claim should be conditioned on the real
  batch/accumulation size. *(Drives E3, still open.)*
- **R4 — 14% is a projected ceiling.** main.tex derived 0.86 from token
  ratios; the CMMD A/B is explicitly pending. Abstract/conclusion stated
  it as an outcome. *(Wording fixed in the Branch A rewrite; the A/B
  itself is E4, still open.)*
- **R5 — hygiene, all confirmed:** `.gitignore:35` ignores `results`
  globally (so the repro claim was false for the public repo;
  `paper_bench/runs/` is now gitignore-exempt and in-repo); pending
  markers in the manuscript; SwD bib listed a nonexistent author
  ("Khoroshikh", missing Drobyshevskiy/Kuznedelev) — **fixed 2026-07-28**
  against arXiv:2503.16397.

Partly right / softened rather than rerun:

- **Eq. 3 "derivation."** The G(σ) renormalization was already flagged
  post-hoc in the paper, but the abstract's "we first derive" and the
  "ratio sets amplitude / token count sets floor" language outran the
  evidence (2 ratio-matched pairs, 1 crossed pair). Now presented as a
  first-order *account* whose terms are individually evidenced; held-out
  prediction is E5.
- **Framing vs SPD/SwD.** The paper already conceded "the null's error is
  not its governor but its scope"; intro/related work now make explicit
  that neither SPD nor SwD *claims* naive gradient equivalence — we test
  a tempting extension, not their methods.

Overstated / optional:

- The 2-model × 2-adapter × 2-domain generalization matrix is the right
  ask for a strong venue but is not what makes the current claims true or
  false. One extra DiT + one full-FT probe arm is the 80/20 (E6).

---

## E1 [GATE] — debiased gaps: self-floors + draw-count extrapolation [DONE 2026-07-29]

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

**E1(0) — retroactive gap-vs-D scan [DONE 2026-07-28, free].** Existing
endpoint-bearing runs at D ∈ {4, 8, 16} already showed the confound
signature. 1024→896 (N=40): +0.100 / +0.035 / −0.016 at D=4/8/16 — a
clean c/D decay (fit on D=4,8: c ≈ 0.52, gap_∞ ≈ −0.03, predicting
+0.002 at D=16, matching the measurement). 896's "≈0 floor" is ≈0
*because* D=16 pushed the bias below the band. 1024→768: +0.167 / ~+0.10
/ +0.08–0.13 — shrinks 4→8, scatter ~0.05 across same-D runs; if
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
around a number whose bias only shrinks with D. It licenses the E3
non-inferiority criterion as pure reanalysis of `per_image.jsonl`, but
it cannot bound the variance bias; only demoted self-floors do.

**Pre-registered decision rule.**
- `gap_∞(512) ≥ 0.15` debiased → token-count floor confirmed; paper
  strengthens (report both raw and debiased).
- `gap_∞(768)` debiased ≤ reenc band → 768's "never safe" at high σ was
  estimator variance; safety map and abstract rewritten.
- Everything collapses into the reenc band → headline becomes the
  low/mid-σ result + claim-narrowing.

**RESULTS (2026-07-29; runs live in `paper_bench/runs/`
(gitignore-exempt, committable — future paper-bench runs pass
`--results_root project/sigma_lowres/paper_bench/runs`):
`20260728-2302-e1a-drawsweep`, `20260729-0014-e1b-debiased-map`,
`20260729-0420-e1c-xzero-endpoint`; instrument: `--self_floor` +
`--draw_sweep` + `--deterministic` landed, det-twins bit-exact,
stats-overlap cut wall ~2-3x).**

- **(a) endpoint draw-sweep, N=12, D=4..64 nested.** Debiased gap_∞:
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
- **(b) verdict grid 8×8+endpoint, N=40, --self_floor.** Caveat first:
  at D=8/bin the *unpaired* debiased estimator overshoots (reenc bins to
  −0.4 where floors are small, σ≈0.19–0.44) — the readable object is the
  **paired per-image difference (arm − reenc)**, |Δ|>1.5 dropped.
  Paired-debiased map: 512 unsafe at every σ (+0.08..+0.60). 896 unsafe
  σ<0.5, ≈0 in σ∈[0.56,0.94] (formal 0.02-UB pass only at 0.688 —
  bin-level ε* at N=40/D=8 is ~0.03–0.08, see E8.1), **small real gap at
  the exact endpoint (+0.042±0.011)** that raw analysis missed. 768 ≈ 0
  in σ∈[0.69,0.94] (means −0.03..+0.015) but clearly gapped at the
  endpoint (+0.092±0.012) and everywhere σ<0.6 — "never safe" softens to
  "no certifiable window at current instrument resolution; means ≈ 0 in
  [0.69,0.94]". Shipped 896@σ>0.5 map: re-confirmed debiased except the
  σ=1.0 endpoint itself.
- **(c) x-zero endpoint sweep, N=40, D=4..32, --self_floor.** Debiased
  graph-term gap_∞: 896 +0.034 [+0.017,+0.058]; 768 +0.074
  [+0.053,+0.094]; 512 +0.283 [+0.232,+0.332] — statistically equal to
  (a)'s full-endpoint gaps at every route. **The endpoint gap IS the
  graph/Jacobian floor: the target-content share R2 flagged (raw 768
  0.127 vs x-zero 0.064) was estimator bias, not content.** The paper's
  original "any endpoint gap is the floor by construction" survives in
  debiased units; E2's α-sweep is demoted from gate-adjacent to cheap
  confirmation (predicted α-slope ≈ 0).

**Outcome: decision rule 1 fired → Branch A** (paper_plan.md §5); the
gap-native restructure was written into `paper/main.tex` in debiased
units.

---

## E2 — target-strength sweep at the endpoint [DONE 2026-07-29]

`--target_alpha 0,0.25,0.5,0.75,1`: at σ=1, input = ε unchanged, target
= ε − α·x (per-arm x). Decomposes the endpoint gap into graph share
(α=0, ≡ x-zero-in-target-only) and target-content share (slope in α).
Post-E1(c) this was a cheap confirmation run (predicted slope ≈ 0).

**Instrument** (landed `7e6ed556`): `--target_alpha` in
`run_sigma_probe.py` — one full pass per α over every arm, draw seeds
shared across α (slope carries no draw noise), all α in ONE process
(kernel-path chaos rule). α=1 keys unsuffixed, others `@a<α>`;
envelope gains per-α aggregates + `alpha_slope_<arm>`.

**RESULTS** (run `20260729-0946-e2-target-alpha`: N=12, D=16,
endpoint-only, routes {768, 512} + reenc, `--self_floor`, wall 1.0 h):

- Paired debiased (arm − reenc, |Δ|>1.5 trimmed), endpoint bin:
  768: α0 +0.070±0.015, α0.25 +0.068±0.013, α1 +0.049±0.010;
  512: α0 +0.269±0.041, α0.25 +0.263±0.039, α1 +0.337±0.067;
  reenc envelope slope +0.003. **α-flat at the well-conditioned
  anchors → no resolvable target-content share; predicted slope ≈ 0
  CONFIRMED. The endpoint gap is the graph floor — E1(c)'s x-zero ≡
  endpoint equality reproduced along a second axis.** (Values also
  consistent with E1(a)'s N=40 draw-limit floors.)
- **Mid-α (0.5, 0.75) is unreadable by construction** — a bonus
  mechanistic result, not hidden noise: the native residual α·x − x̂
  passes near cancellation there, `gnorm_native` dips 60→33,
  `cos_floor` falls to 0.87, and the paired SEMs blow up ~7×. Small-G
  amplification (the paper's §input-branch mechanism) surfacing along
  the α axis. Consequence: the envelope's naive `alpha_slope_*` over
  all five α (+0.034 / +0.180) is inflated by these points — the
  verdict object is the anchor contrast above.
- Paper edit landed: the §floor `[pending]` marker replaced with the
  measured sweep + the mid-α G-dip note.
- Ops gotcha for future runs: `make daemon-run` consumes `--label`
  from ARGS as the *job* label (documented dispatcher behavior), so
  the script never saw it and the run dir was created label-less —
  renamed by hand afterward.

---

## E4 — the end-to-end A/B [CORE DISCHARGED 2026-07-30 — exercise grid; residuals stay in required_experiments.md]

**Design realized** (frozen manifest `runs/20260729-1537-e4-manifest/` +
`launch_20260729_exercise.md` amendments): **4 arms** — native /
sigma896 (σ>0.5 gate + yarnsig) / **896only** (added: threshold 0 on the
safe route — isolates the σ-gate) / unsafe768 (threshold 0 on 1024→768,
the review's negative control, via the new `--sigma_lowres_route`,
commit 5b63ebb9) — × **3 seeds** (1001–1003) × 2 artists (hews 60-stem
train / 8 ep; channel\_(caststation) 15-stem / 32 ep; 480 steps bs 1
each), `--deterministic --paired_step_rng`, stock lora recipe. 24
checkpoints. In-vivo CRN check: sigma896 demoted the **identical
244/480 step set on both artists** (σ stream is seed-keyed, not
data-keyed).

**Throughput — the paper's headline number is now measured** (n=3
means; exact FLOPs via `token_step_hist` × FlopCounterMode,
`e4_flops.py`; wall tracks FLOPs ~1:1):

| arm | fwd PFLOPs | Δ | wall | Δ |
|---|---|---|---|---|
| native | 8.64 | — | 388 s | — |
| sigma896 | 7.35 | **−15.1%** | 331 s | **−14.6%** |
| 896only | 5.99 | −30.8% | 266 s | −31.4% |
| unsafe768 | 4.13 | −52.2% | 185 s | −52.3% |

The "projected ceiling of ~14%" reads as **measured −14.6% wall /
−15.1% FLOPs** for the shipped gate.

**Sample-level defensibility (seed-noise yardstick,
`runs/20260729-2148-e4-yardstick/`)**: within-seed
cos(native~sigma896) vs cross-seed cos(native~native) on the frozen
(prompt, gen_seed) grid — channel **0.9641 vs 0.9541 (inside the seed
lottery)**; hews 0.9551 vs 0.9558 (boundary tie). Headline: *swapping
σ>0.5 steps to the 896 sibling perturbs renders about as much as
changing the training seed.* Arm orderings shuffle between seeds —
single-seed visual impressions were substantially seed lottery.

**Negative control did its job by exposing the metric**: at exercise N
(9–12 SFW prompts, rating-mismatched pools) CMMD cannot separate the
known-bad route from anything (unsafe768 ≈ native on channel, ≤
sigma896 on hews) → **no quality verdict is read from this pass**;
CMMD non-inferiority needs the full-band rescoring (residual). Also
banked: Δ(member−holdout) > 0 everywhere (no memorization pathology);
figure sheets committed (9430c182: 10607820 main, 14296235 + 8508115
appendix).

**Open tension for the gate story**: sigma896~896only is the closest
arm pair (hews s1001 cos 0.977) and 896only is another 16% cheaper —
the σ-gate is endpoint-invisible at this recipe, so its justification
currently rests entirely on the σ-resolved per-step certification
(`claim_accumulated_bias.md`'s accumulated-bias question, now with an
empirical handle: the full-protocol quality read of sigma896 vs
896only).

Eval/scoring stack committed as 4f538d7e (`e4_render_eval.py`,
`e4_flops.py`, `e4_seed_yardstick.py`); results in
`runs/20260729-2148-e4-eval-sfw-s100{1,2,3}/` + `-yardstick/`.

---

## E5 — Eq. 3 held-out validation [DONE 2026-07-29 — qualified PASS]

**Question.** Does the two-term account *predict* routes it was not fit
on? Fit A_e, Floor_e on {1024→896, 1024→512, 1280→1120}, fit governor
models A(ratio) and F(target tokens), predict {1024→768, 1280→1024}
from measured m̄(σ) (G7, route-uniform mean) and each run's own G(σ).
Analysis-only: `paper_bench/e5_holdout.py` (pass criteria pre-registered
in the script header before the numbers), run
`paper_bench/runs/20260729-1130-e5-holdout/` (result.json +
`e5_overlay.png`). Sources: E1b paired-debiased (1024 tier), G9 raw
paired (1280 tier — no self-floors exist there, D=4 caveat), G7 m(σ),
per-run gnorm.

**RESULTS — all four pre-registered gates fire; PASS with character:**

- **p\* = 2.00** (grid boundary — but that *is* the small-mismatch
  cosine-geometry limit the account pins a priori).
- **Ratio governor, amplitude-level:** A_512 0.077±0.010,
  A_896 0.0076±0.0053, A_1120 0.0068±0.0015 — the two ratio-0.875
  routes agree at **z = 0.14** despite 1.6× different target capacity.
  G9 showed the ratio governor at the σ\* level; this is the same
  verdict at the fitted-amplitude level.
- **Floor governor, interpolation-level:** fitted floors 512 +0.264,
  896 +0.039, 1120 +0.002; exp-law F(n) = 0.70·exp(−n/1041 tok) fit on
  512+896 predicts F(4825) = +0.007 (1120 fitted +0.002 ✓) and
  **F(2160) = +0.088 for 768 vs measured debiased endpoint
  +0.092±0.012** — the held-out floor is hit dead-on.
- **Held-out 1024→768:** RMSE 0.093 — *better than the oracle
  quadratic fit on the held-out data itself* (0.105); null committed
  region (σ > σ_eq: gap ≈ 0) RMSE 0.164 vs ours 0.096.
- **Held-out 1280→1024:** RMSE 0.092 vs oracle 0.056 (ratio 1.64,
  inside the 2× gate); null 0.104 vs ours 0.100 (marginal).

**Misses, recorded honestly (the "character"):**

1. **768 mid-σ window:** measured ≈ 0 in σ∈[0.56,0.94] vs predicted
   flat ~0.09–0.14 — the two-term form cannot dip below Floor_e. Same
   window-vs-endpoint anomaly E1(b) already recorded; E5 sharpens it
   into the account's one visible structural failure.
2. **1280→1024 peak overshoot** (predicted 0.37 vs measured 0.18 at
   σ=0.375): A(0.8) comes from linear-in-ratio interpolation resting on
   only two distinct ratio values; A(r) is evidently convex between
   0.5 and 0.875.
3. **χ²/bin 7.6 / 9.2 held-out (0.9–5.2 in-sample):** the prediction is
   NOT within instrument resolution anywhere — the claim licensed is
   "shape + magnitude class + governors at ~0.09 RMSE", never
   "predicts within ε\*".

**Post-hoc addendum (leave-896-out, NOT pre-registered; same session).**
896 was a fit route, so it had no held-out prediction; re-deriving the
governors from 512+1120 alone and predicting 896 gives: A(0.875) from
the ratio-twin 1120 alone = 0.0068 (vs 0.0076 fitted on 896 itself);
floor exp-law refit through 512+1120 (fragile leg: F_1120 =
0.002±0.014 enters a log fit) → **F(3012) = +0.019 vs E1a's measured
896 debiased floor +0.019 [+0.010,+0.030] — dead center. The floor law
is 2-for-2 on floors it never saw** (768 +0.088/+0.092, 896
+0.019/+0.019). Curve-level RMSE 0.072 vs oracle 0.064 (ratio 1.13),
χ²/bin 6.4 — same character: low-σ bins under-predicted (measured
+0.09..+0.21 vs predicted +0.02..+0.05 at σ ≤ 0.31), mid/high-σ
tracked, and the E1b σ=1 endpoint reads +0.042±0.011 vs +0.019
predicted — the endpoint-vs-window anomaly again, now visible on 896.

**Voice decision (paper_plan §2) resolved → Eq. 3 headlines as a
*predictive* first-order account with stated ~0.09 RMSE resolution;
"ratio sets amplitude / absolute size sets floor" upgrade from
"consistent with" to measured-governor statements (ratio-twin z=0.14;
floor exp-law hitting the held-out 768). The mid-σ 768 dip and the
A(r) convexity go into the account's stated limits.**

---

## E5 refit — three functional forms for the data-branch term [DONE 2026-07-29]

paper_plan §6 step 2 (analysis-only, no GPU): same fit → governors →
held-out pipeline as `e5_holdout.py` under three candidate forms for
S_e (shared additive floor in each): **P** power `A·m/G^p` (p shared,
scanned — the shipped form), **Q** derived small-perturbation quadratic
`A·(m/G)²`, **X** exact angular link `1−1/√(1+(c·m/G)²)` (exact 1−cos
of a pure orthogonal perturbation with ‖δg⊥‖ = c·m; small-κ limit = Q
with A = c²/2). Script `paper_bench/e5_refit.py`, run
`paper_bench/runs/20260729-1322-e5-refit/` (result.json + 3-panel
overlay `e5_refit.png`).

**RESULTS:**

- **P's scan lands on p\* = 2.00 exactly** — the "free" exponent picks
  the quadratic on its own; P and Q are near-duplicates everywhere.
- **In-sample 512** (the severe route, gap ~0.3): χ²/bin 3.5–3.9
  across all three forms (Q narrowly best) — the floor absorbs most of
  512, so the small-κ worry does not discriminate in-sample.
- **Held-out is decisive: X wins** — mean RMSE 0.071 vs 0.093 (P) /
  0.094 (Q). The entire margin is the 1280→1024 route (0.049 vs
  ~0.095): **the saturation of the exact link removes the peak
  overshoot E5 had booked as "A(r) evidently convex"** (miss #2). On
  768 all three tie (0.089–0.093) — the mid-σ dip is form-independent
  (it is the interaction term I_768 < 0, not a form artifact).
- **Form-invariant governors:** floor exp-law F(2160) = +0.087/+0.088
  in all three; ratio-twin z = 0.14 (P) / 0.44 (Q) / 0.15 (X).

**Ledger naming (consumed by main.tex §3.3/§4.6):** the **exact
angular link X headlines as the best empirical predictor** — its
empirical ingredient is the linear mismatch loading ‖δg⊥‖ = c·m, and
it is labeled empirical on that account; **Q is reported as the
derived small-perturbation form** and is X's small-κ limit. One
geometry at two orders: the derived quadratic is not a competitor to
the headline form but its local law, and E5 miss #2 (A(r) convexity)
is retired — the "convexity" was quadratic-form overshoot.

---

## E8.1 + E8.2 [DONE 2026-07-29 — written into main.tex]

From the gap-native restructure proposal (E8, 2026-07-28); part 3 (the
null→gap bridge) is still open and remains in `required_experiments.md`.

1. **ε\* — the minimum detectable gap.** The instrument's detectability
   threshold as a function of (N, D, floor cosine): below ε\*, a
   demotion is indistinguishable from redraw noise. Sources: E1
   self-floors (bias) + per-bin SEM (variance). "Safe" ≡ one-sided 95%
   CI of the debiased gap below ε\* — E3's non-inferiority criterion
   promoted from an ad-hoc reenc±0.04 band to the *definition*. Landed
   as main.tex §epsstar (Eq. epsstar / safe(ε)).
2. **The guarantee region.** The safety map restated as the (route, σ)
   region where debiased gap ≤ ε\* at one-sided 95% — E1(b)+E3 output
   verbatim. Wording is "statistical non-inferiority at instrument
   resolution", never a hard bound; per-example vs batch-aggregate map
   split retained (E3/R3).

---

## Manuscript status (theory→evidence→application + question-first reframe, written 2026-07-29)

Superseded the Branch-A spine the same day: `paper_plan.md` §3 spine
landed together with the question-first reframe (title now "When Does
Training on Downscaled Images Yield the Same Gradients?"; §1 leads with
the practical question, the map is presented as the answer). New
structure: §3 Theory (3.1 estimand — d_e vs reenc-excess Δ_e both
stated, aggregation operator part of the estimand, ε\* renamed "median
certification resolution" with the power footnote; 3.2 data/graph
branches — endpoint NOT data-free by construction, "data branch" rename
throughout; 3.3 four-term expansion d = S+F+I+R derived + A1–A4
reduction + exact angular link labeled empirical; 3.4 spectral null
with exact scope sentence; 3.5 seven discriminating predictions each
naming its probe), §4 Evidence (4.1 instrument+debiasing+"debiased
units only from here" + consolidated coverage statement; 4.2 phenomenon
+ Table-null boundary scoring; 4.3 endpoint≡x-zero≡α-flat; 4.4 data
branch + governors; 4.5 depth + RoPE/Resid waterfall; 4.6 the account
confronted — E5 held-out + three-form refit + 768 mid-σ dip as I_e<0
interaction signature delimiting the reduction's domain + E9 designated
+ claims-ledger table as the consolidated hedge), §5 Application
("the answer, in deployable form"), §6/§7 updated. Raw/historical
tables all moved to a dedicated appendix (incl. new raw-vs-debiased
endpoint revision-record table); per-sentence hedges replaced by the
ledger. E5 refit figure `figs/e5_refit.png` is the §4.6 figure
(`figs/e5_overlay.png` also staged). Abstract intentionally NOT
revised yet (user will revise after reading the rewrite; it still says
"input term"). Compiles clean under tectonic (0 overfull, no broken
refs, 22 pp).

Still open in the manuscript, each marked **[pending]** in place: E3
pooled-with-self-floors run, E4 A/B + `reenc_noise_floor.py` (δ_reenc
row + D(f) numbers), E7 membership probe, E8.3 overlay + t\*(δ)
figures, results tarball. (E2's marker cleared 2026-07-29 with the
measured sweep.)

Figures: Fig 1c regenerated in debiased units 2026-07-29
(`plot_debiased_map.py` → `figs/gap_debiased.png` — paired per-image
map with the bin-level ±ε\* band; RAPSD σ\* vlines removed, to be
addressed separately). Remaining raw figures are marked as raw in
captions; Fig-1 enlargement + waterfall + overlay figures owed with
E8.3 analysis.
