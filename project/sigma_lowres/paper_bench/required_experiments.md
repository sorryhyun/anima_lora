# paper_bench — required experiments before the sigma_lowres paper ships

Triage of the 2026-07-28 external review (ChatGPT), verified against the
actual instrument (`bench/run_sigma_probe.py`), the report
(`bench/report.md`), and `paper/main.tex`. Each item below is marked with
what the review got right/wrong and what run actually discharges it.

Status legend: **[GATE]** = blocks the mechanistic claims as written;
**[FIX]** = paper/repo edit, no GPU; **[OWED]** = already owed by
`roadmap.md` (Phase 1b); **[STRETCH]** = raises acceptance ceiling, not
required for correctness.

---

## Verdict on the review

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
  and not all. This is the one critique that genuinely gates the paper.
  Note the review under-claims in one spot: the x-zero probe is subject
  to the **same** confound (single demoted estimate), so x-zero is not a
  clean rescue of the graph term either — E1's debiasing must be applied
  to x-zero too.
- **R2 — σ=1 is not graph-only.** `target = noise − lat`: at σ=1 the
  input is pure ε but the target still carries x per arm. The paper's own
  Table (floor table, main.tex ~line 693) shows it: 768 endpoint 0.127
  vs x-zero 0.064 — **half** the 768 endpoint gap is target-content, yet
  the text says "any gap *is* the floor by construction" and only
  highlights the 512 route (where endpoint ≈ x-zero and the claim holds).
- **R3 — "never safe" is aggregation-dependent.** Confirmed in
  `bench/report.md` (pool4 addendum): pooled gap_768 ≈ 0 at σ ≥ 0.875,
  pooled gap_896 ≈ 0 at σ ≥ 0.625. The per-image and batch-SGD objects
  genuinely disagree at high σ; the safety map must state which object it
  is a map *of*, and the trainer claim should be conditioned on the real
  batch/accumulation size.
- **R4 — 14% is a projected ceiling.** main.tex derives 0.86 from token
  ratios (~line 982); the CMMD A/B is explicitly pending (three
  `[pending: reenc_noise_floor.py]`-class markers at lines 447/1017/1125).
  Abstract/conclusion currently state it as an outcome.
- **R5 — hygiene, all confirmed:** `.gitignore:35` ignores `results`
  globally (so the repro claim is false for the public repo); pending
  markers in the manuscript; SwD bib listed a nonexistent author
  ("Khoroshikh", missing Drobyshevskiy/Kuznedelev) — **fixed 2026-07-28**
  against arXiv:2503.16397.

Partly right / soften rather than rerun:

- **Eq. 3 "derivation."** The G(σ) renormalization is already flagged
  post-hoc in the paper, but the abstract's "we first derive" and the
  "ratio sets amplitude / token count sets floor" language outrun the
  evidence (2 ratio-matched pairs, 1 crossed pair). Present as a
  first-order *account* whose terms are individually evidenced, and only
  attempt held-out prediction (E5) if the floors survive E1.
- **Framing vs SPD/SwD.** The paper already concedes "the null's error is
  not its governor but its scope"; a few sentences in intro/related work
  should make explicit that neither SPD nor SwD *claims* naive gradient
  equivalence — we test a tempting extension, not their methods.

Overstated / optional:

- The 2-model × 2-adapter × 2-domain generalization matrix is the right
  ask for a strong venue but is not what makes the current claims true or
  false. One extra DiT + one full-FT probe arm is the 80/20 (E6).

---

## E1 [GATE] — debiased gaps: self-floors + draw-count extrapolation

**Question.** How much of every reported gap (incl. x-zero and endpoint)
survives when estimator variance is equalized out?

**Instrument change** (`run_sigma_probe.py`):

1. `--self_floor`: for every arm (reenc + each demote/pi/yarn arm) run a
   **second** independent draw set `g_d′` (`seeds(arm_idx′)`) and record
   `cos_self_<key> = cos(g_d, g_d′)` per bin.
2. Report, alongside the existing gap, the **debiased cosine**
   `ĉ = cos(n̄, d̄) / sqrt(cos_floor_native · cos_self_d)` (split-half
   attenuation correction; both native estimates already exist) and
   `debiased_gap = 1 − ĉ`. Keep raw gaps for continuity.
3. `--draw_sweep 4,8,16,32,64`: endpoint-only mode (`--bins 0
   --endpoint_bin`), reduced probe set (N=12, redundancy-stratified),
   fit `gap(D) = gap_∞ + c/D` per route, report `gap_∞` with a bootstrap
   CI over images. Nested seeds so the D=64 run contains the D=32 draws
   (one pass, prefix sums — no extra forwards).

**E1(0) — retroactive gap-vs-D scan [DONE 2026-07-28, free].** Existing
endpoint-bearing runs at D ∈ {4, 8, 16} already show the confound
signature. 1024→896 (N=40): +0.100 / +0.035 / −0.016 at D=4/8/16 — a
clean c/D decay (fit on D=4,8: c ≈ 0.52, gap_∞ ≈ −0.03, predicting
+0.002 at D=16, matching the measurement). 896's "≈0 floor" is ≈0
*because* D=16 pushed the bias below the band. 1024→768: +0.167 / ~+0.10
/ +0.08–0.13 — shrinks 4→8, scatter ~0.05 across same-D runs; if
c ≈ 0.5 carries over, the paper's D=16 floor +0.127 contains ~0.03 of
estimator bias (true floor ~0.09). 1024→512 exists only at D=16 (no
trend; largest expected c → most unmeasured bias). The D=2/N=4 smokes
put reenc at −0.19 — a same-grid control 5× outside the band on draw
noise alone. Caveat: cross-run points differ in seeds/arm structure, so
this is indicative, not a substitute for (a). **Verdict: confound is
live, observed, with a fitted c on one route — E1 is not optional.**

Measured cross-run sensitivity (2026-07-28 smoke twins, D=2, same seeds
and inputs): two runs sharing the warm inductor kernel cache agree to
|Δcos| ≤ 0.015 (atomics-order noise); a run with a different kernel set
(cold-autotune first compile) lands up to |Δcos| ≈ 0.29–0.36 away. So
per-bin cosines at low D are *kernel-path chaotic* — never compare them
across processes; every reported gap/floor/debias pairing must stay
within one run, which the instrument already guarantees.

Note on what the existing per-bin SEM band can and cannot do: the band
is cross-image scatter of the *biased* estimator — it tightens with N
around a number whose bias only shrinks with D. It does license the E3
non-inferiority criterion as pure reanalysis of `per_image.jsonl`
(mean + 1.645·SEM vs reenc + margin, computable today), but it cannot
bound the variance bias; only demoted self-floors do.

**Runs.**
- (a) Endpoint draw-sweep, routes {896, 768, 512} + reenc, N=12, D=64
  once (prefix-analyzed for 4/8/16/32). ~3 arms × 2 estimates.
- (b) Repeat of the verdict σ-grid (8×8, N=40, routes 896/768/512) with
  `--self_floor` — this re-derives the **entire safety map** in debiased
  units.
- (c) x-zero endpoint with `--self_floor` (the graph-term claim rides on
  this one).

**Decision rule (pre-register before running).**
- `gap_∞(512) ≥ 0.15` debiased → token-count floor confirmed; paper
  strengthens (report both raw and debiased).
- `gap_∞(768)` debiased ≤ reenc band → 768's "never safe" at high σ was
  estimator variance; safety map and abstract rewritten, RoPE/depth
  decomposition re-checked in debiased units.
- Everything collapses into the reenc band → the paper's headline
  becomes the (still true, still novel) *low/mid-σ* result + the
  claim-narrowing the review suggests. Kill switch honesty: say so.

Everything downstream (E2–E5, the map, the trainer gate) consumes E1's
debiased numbers.

**E1 RESULTS (2026-07-29; runs live in `paper_bench/runs/`
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
  ~0.056) — floor-table magnitudes must be rewritten. Native floor
  extrapolates to 1.005 [0.994,1.016]: the draft's "endpoint floor
  ≈ 0.85" was pure draw noise (R1 vindicated on the native floor).
  Debiased fits are D-flat (|c| ≤ 0.05 for 512 vs raw c ≈ +0.29) — the
  attenuation correction works as designed.
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

## E2 — target-strength sweep at the endpoint (relabel the floor)

`--target_alpha 0,0.25,0.5,0.75,1`: at σ=1, input = ε unchanged, target
= ε − α·x (per-arm x). Decomposes the endpoint gap into graph share
(α=0, ≡ x-zero-in-target-only), target-content share (slope in α), and
interaction. N=12, D=16, routes {768, 512} (896 is ≈0 already). Cheap —
one afternoon run. Paper edits regardless of outcome: rename "endpoint
gap @ σ=1" → "high-noise endpoint gap"; x-zero is the graph-only control;
768's text stops claiming the plateau *is* the floor.

## E3 — aggregation-conditioned safety map

Mostly already measured (`--pool`); what's missing is the **framing** and
one run at the real operating point:

- One verdict-grid run with `--pool <actual train batch × accum>` (read
  from the shipped LoRA config) + `--self_floor` so pooled arms get
  self-floors too.
- Paper: publish **two maps** — per-example (worst case, what a
  batch-1 user sees) and batch-aggregate (what the shipped trainer
  consumes) — and define "safe" as a pre-specified non-inferiority test:
  one-sided 95% CI upper bound of the debiased gap below reenc + 0.02
  margin. The CI itself is free (per-bin mean + 1.645·SEM from existing
  `per_image.jsonl` rows — see E1(0) note); only the *debiasing* inside
  it waits on E1. This resolves the report-vs-paper 768 contradiction the review
  found instead of hiding it.

## E4 [OWED] — the end-to-end A/B (Phase 1b, already gated in roadmap.md)

Unchanged from `roadmap.md` but now with the review's negative control:
native baseline vs σ-conditional demotion (map from E1/E3, yarnsig
refinement per the 2026-07-27 PASS) vs **an unsafe-route arm**
(e.g. 1024→768 unconditional) as the negative control; identical data
order/noise/init, 3 seeds; measure realized wall-clock (incl. cache
load), examples/s, peak mem, CMMD, val loss, matched renders. Until this
lands, every "14%" in the paper reads "a projected ceiling of ~14%".
Also run `reenc_noise_floor.py` (script exists, queued) and clear the
three pending markers.

## E5 — Eq. 3 held-out validation, or demotion to "conceptual"

Only if E1 confirms the floors: fit A_e, Floor_e on {1024→896, 1024→512,
1280→1120}, predict {1024→768, 1280→1024} from measured m(σ), G(σ);
compare vs spectral-null and an unconstrained smooth fit. If it fails to
predict, keep Eq. 3 but present it as a decomposition, and change "ratio
sets / token count sets" to "consistent with". No new instrument needed —
this is analysis over existing + E1 rows.

## E6 [STRETCH] — one generalization arm each

- **One extra DiT** (any open flow-matching DiT with a different VAE),
  endpoint + 3-bin grid, routes {×0.875, ×0.5}, N=12: turns the case
  study into a phenomenon.
- **Full-FT probe** (all-param grads, N small, grad-ckpt): does the floor
  live in LoRA geometry or the model? One run answers it.
- Second LoRA checkpoint on Anima (different corpus): near-free, uses the
  existing instrument verbatim — largely subsumed by E7's controlled
  adapter if that runs.

## E7 — controlled-LoRA 2×2: data relationship to the adapter

**Question.** Does the safety map depend on the probe data's relationship
to the adapter — i.e. is the map a property of the (model, route) or of
the (model, route, *adapter's training distribution*)? The paper already
concedes operating-point dependence in Limitations; this measures it
instead of conceding it. It also cleans up a known wart: the current
verdict pool contains 2/40 stems from the probe adapter's own fine-tune
set (`report.md` in-pool overlap check, 2026-07-26) — membership was
never a controlled axis.

**Design.** Train a **controlled LoRA** with a frozen, manifest-recorded
fine-tune set (a dedicated `path_pattern` slice; standard `make lora`,
same recipe as the shipped probe adapter), holding out a matched split of
the same artists. Probe cells, N≈12 each, redundancy-matched across
cells:

| cell | membership | domain |
|---|---|---|
| S1 | trained-on (in the fine-tune manifest) | in-domain |
| S2 | held-out, same artists | in-domain |
| S3 | held-out, different artists / style cluster | near-OOD |
| S4 | (optional) far-OOD — outside the illustration corpus | OOD |

Run the E1-debiased instrument (endpoint + the 4-bin high-σ grid, routes
{896, 768}) per cell. Instrument delta: probe selection currently goes
through redundancy scoring with `--artists`/`--max_per_artist` only — add
a `--probe_list <file>` override (explicit stem list per cell) plus a
membership tag in `per_image.jsonl` rows, so cells are exact rather than
glob-approximate.

**Pre-registered readout.**
- Two-term prediction: the **floor** is a Jacobian/graph property →
  invariant across all four cells (strong, falsifiable). The **input
  branch** rides the residual and gradient norm, which plausibly shrink
  on trained-on images (memorized → small residual) → A_e and possibly
  the measured σ* may shift between S1 and S2–S4.
- S1 vs S2 shift ⇒ the map drifts *during* training as images become
  trained-on — directly relevant to deployment, since the trainer demotes
  its own fine-tune set. If this fires, add the trajectory variant:
  re-probe S1/S2 at an early-epoch checkpoint of the same controlled run
  (checkpoints are free — they already exist as `save_every` artifacts).
- No shift anywhere ⇒ one Limitations paragraph becomes a supported
  robustness claim; the safety map is adapter-agnostic on this model.

Cost: one standard LoRA train + ~4 probe runs at reduced grid; cheapest
item on the generalization axis and the only one that addresses a stated
limitation with existing hardware.

## E8 [FIX] — gap-native restructure: ε\* detectability threshold + null→gap bridge

**Proposal (2026-07-28).** Present the paper's spine in the representative
metric itself, three parts, all in debiased units:

1. **ε\* — the minimum detectable gap.** Derive the instrument's
   detectability threshold as a function of (N, D, floor cosine): below
   ε\*, a demotion is indistinguishable from redraw noise. Sources: E1
   self-floors (bias) + per-bin SEM (variance). "Safe" ≡ one-sided 95%
   CI of the debiased gap below ε\* — E3's non-inferiority criterion
   promoted from an ad-hoc reenc±0.04 band to the *definition*.
   Analysis over existing `per_image.jsonl` + E1 rows; no new GPU.
2. **The guarantee region.** Restate the safety map as the (route, σ)
   region where debiased gap ≤ ε\* at one-sided 95% — i.e. E1(b)+E3
   output verbatim, no new runs. Wording stays "statistical
   non-inferiority at instrument resolution", never a hard bound; keep
   the per-example vs batch-aggregate map split (E3/R3).
3. **Null→gap bridge (new analysis).** Convert each published tolerance
   δ into a predicted *gap curve*, not just a boundary: under the
   diagonal model on the measured P(f), compute the destroyed-band
   Bayes-residual mismatch m_null,e(σ; δ), map through the measured
   G(σ)^p with the route gain A calibrated on the one safe route, and
   overlay predicted vs measured curves per (δ, route). This is the
   Table-1 confrontation restated in gap units; it subsumes the
   continuous t\*(δ) sweep figure (family spread ≤ 0.13) and the
   δ_reenc-anchored row (`reenc_noise_floor.py`, owed in E4). Unit
   honesty: the null emits residual units only — the bridge (G^p,
   calibrated A) belongs to the two-term account, so the paper must say
   "the null read through our bridge", which also makes §3 load-bearing
   for the confrontation rather than decorative.

**Depends on:** E1 (all three parts consume debiased numbers); part 3
additionally on E4's `reenc_noise_floor` run for the anchored row.
**Cost:** analysis + figures only.

## Reproducibility deliverables (paper_bench/ contents) [FIX]

- `make_all.py` — single entry that regenerates every table/figure in
  `paper/` from `bench/results/` run dirs (table → run-id mapping frozen
  in a `MANIFEST.toml` with commit hashes, adapter path, seeds).
- Results archive: either carve `project/sigma_lowres/bench/results/`
  out of the global `results` ignore (`.gitignore:35`) for the verdict
  runs only, or publish a tarball (HF dataset) and link it — the
  reproducibility statement must match what's actually public.
- Manuscript pass: strip pending markers (after E4's floor run), shorten
  abstract, enlarge Fig. 1, drop colored link boxes, label every claim
  pre-registered / confirmatory / post-hoc (the freeze dates exist in
  `questions.md` — link the commits), and narrow the headline to
  "spectral sufficiency of the noisy input does not guarantee gradient
  equivalence under resolution substitution."

---

## Order of operations

E1(a) endpoint sweep first (cheapest decisive run — if the floors melt,
E2/E5 are moot and the rewrite is different) → E1(b,c) full debiased map
→ E2 + E3 in parallel (kick off E7's controlled-LoRA train here — it's
GPU-cheap and its probes need E1's debiased instrument anyway) → paper
edits + E8 (analysis-only; consumes E1's debiased numbers; E8.3's
anchored row waits on E4's `reenc_noise_floor` run) → E4 (Phase 1b as
owed) → E7 probes → E5 → E6 if targeting a top venue.
