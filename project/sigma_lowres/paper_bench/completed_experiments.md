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

## Manuscript status (Branch A, written 2026-07-29)

E1(a,b,c) landed and rule 1 fired, so the paper_plan §2 spine is now the
manuscript: new §3 metrology (gap def + instrument + finite-draw-bias
subsection + ε\* and the safe(ε) definition), §4 account (softened,
"derive"→account; Table 1 + retrodict moved out), §5 Measurements (raw
tables kept as marked historical record, new debiased verdict-map +
debiased floor tables, floor waterfall ledger, x-zero≡endpoint equality,
PI/iso-severity/depth re-anchored with the one-signed-bias argument),
§6 "The null read in gap units" (Table 1 + scoring + E8.3 bridge stub),
§7 two-maps framing + projected-ceiling wording, limitations gain
ε\*-relative + debiased-coverage items, repro statement now matches what
is public (paper_bench/runs in-repo; raw-run tarball pending).
Abstract/intro/conclusion rewritten (narrowed headline, metrology as
contribution, 14% as projected ceiling). Compiles clean under tectonic
(0 overfull, no broken refs).

Still open in the manuscript, each marked **[pending]** in place: E2
α-sweep confirmation, E3 pooled-with-self-floors run, E4 A/B +
`reenc_noise_floor.py` (δ_reenc row + D(f) numbers), E7 membership
probe, E8.3 overlay + t\*(δ) figures, results tarball.

Figures: Fig 1c regenerated in debiased units 2026-07-29
(`plot_debiased_map.py` → `figs/gap_debiased.png` — paired per-image
map with the bin-level ±ε\* band; RAPSD σ\* vlines removed, to be
addressed separately). Remaining raw figures are marked as raw in
captions; Fig-1 enlargement + waterfall + overlay figures owed with
E8.3 analysis.
