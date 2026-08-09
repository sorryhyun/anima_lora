# E22 — does the cancellation hold per sample? (per-image g-ledger)

| | |
|---|---|
| **Status** | **DONE 2026-08-09 — verdict PER-SAMPLE HOLDS at σ = 0.7** (amendment 22.4, D = 96 single-σ rerun: 14/16 gated ≥ the 8-image floor, median ρ_i = −0.820, 92.9 % ≤ −0.7 — see 22.4 Results; licenses drafting E23a's pre-registration, single-σ scope caveat carried). The original 22.1 D = 24 run returned **INSTRUMENT-LIMITED** (all three σ void at the pre-registered reliability floor; recorded below, draws not escalated until the 22.4 amendment). Pre-registration committed 2026-08-08 before the 22.1 instrument existed (theory-first, mirroring 19.1/20/21); amendment 22.4 pre-registered 2026-08-08 after the 22.1 verdict, launched 2026-08-09. |
| **Question** | Every anti-alignment number in this line so far — E14's pooled ρ̄ ≈ −0.91, E21's cell-level LOCAL verdict — is a **cross-image mean** (40 images × 12 draws summed before any cosine). A training-facing correction acts on **one sample at one step**. Is the B/C cancellation a per-sample property (each image's data damage mirrored by its own graph response), or a population property that only emerges in the mean? This is the missing link ("estimand bridge") that E20.4 crashed into from the objective side — E22 closes it from the measurement side, **before** any correction is proposed. Secondary, observational: training "at 1024" is itself training on **preprocess-resized** pixels (source → bucket downscale; free-fit drove crop bias to ~0 but the downscale low-pass remains). Per-image records let us stratify the ledger by each image's realized resize factor — a first, hypothesis-generating look at whether that preprocessing bias is visible in LoRA-gradient space. |
| **Depends on** | [E21](../e21/) (LOCAL verdict — licenses factorized per-cell reads; adaln amplitude concentration — the candidate lever this experiment gates); [E20.4](../e20/) (derived data term fails at estimand level — the standing reason per-sample must be measured, not assumed); `run_sigma_probe.py` + `sigma_probe/` (the per-image arm loop this amends); E14/E19 ledger conventions. **The 19.3/19.4 stores cannot answer this**: they are cross-image sums, and `per_image.jsonl` carries only arm-vs-native cosines (`cos_<arm>`), never the arm×arm cross products that ρ_i needs. New GPU arm required. |
| **Instruments** | 22.0 `e22_corpus.py` (CPU corpus prep → `resize_factors.json` + `e22_probe_list.json`); 22.1 probe amendment `--per_image_ledger` (`sigma_probe/cli.py` + `stats.py` + driver; GPU, daemon job `20260808-163340-13b303` → `runs/20260808-1633-e221-per-image-ledger`); 22.2 `e22_per_image.py` (CPU digest + figure → `e22_per_image.json`, `e22_rho_i.png`); 22.3 applied — see Results; 22.4 amendment rerun (GPU, daemon job `20260809-003128-2c105b` → `runs/20260809-0031-e224-per-image-d96`; probe instrument unchanged as frozen; digest gained σ-subset + `--fig` handling only, regression-checked bit-identical on the 22.1 run → `e224_per_image.json`, `e224_rho_i.png`). |
| **In the paper** | The mechanism→prescription bridge for §5/discussion: PER-SAMPLE HOLDS licenses a training-facing lever (E23a); POOLED-ONLY kills the whole per-sample correction family in one pre-registered stroke and confines prescriptions to scheduler-side routing (the shipped σ-gated demotion recipe). Either outcome is a paper paragraph. *(Outcome: the HOLDS branch, at σ = 0.7 only — 22.4.)* |

**Numbering note**: e20's informally sketched "derive the amplitudes"
E21-idea remains unproposed (still blocked on 20.4); E22 is not that
either — E22 is measurement only, no objective term is derived or fit.

## Motivation (the debias horizon)

E21 established: the cancellation is locally enforced (every cell), the
phase-response *direction* is delocalized, and its *amplitude* is 86–87 %
adaln. The obvious next thought — correct the objective so training is
resolution-robust, or debias the resize baked in by preprocessing — runs
straight into a gap: **all evidence is pooled**. If ρ_i (per image) is
tight around −0.9, per-sample correction is on the table; if ρ_i is wide
or bimodal and only the mean is deep, any per-sample corrector amplifies
noise, and E20.4's estimand-level failure was the predictable symptom
rather than an implementation accident.

The preprocess-resize framing (secondary here, causal study deferred to
E23b): a "native 1024" training image whose source was 4 MP has already
been through the same class of operation whose second application
(1024→768) this line measures. The dataset's natural spread in realized
resize factor is a free observational axis over exactly that first
application — confounded by content (large sources are not random
images), so it can generate hypotheses, never causal claims.

## Frozen conventions (pre-registered — no tuning on outputs)

- **Run grid**: one route **1024→768** (largest signal), σ verdict bins
  **{0.4333, 0.5667, 0.7}** (mid-window, where E21 verified LOCAL); no
  endpoint bin. Arms: native a/b, reenc(+`__2`), 768(+`__2`),
  768rp(+`__2`). **No PI arms** (nothing here needs them; G11 untouched).
- **`draws_per_bin = 24`** (2× the 19.3 setting) — pre-registered SNR
  buy for the per-image estimand, fixed before any output is seen. No
  further draw escalation without an amendment to this file.
- **Corpus**: n = 16 images from the standard probe corpus, **stratified
  by realized resize factor** (source-pixel area / resized-pixel area,
  read from `image_dataset/` originals vs `post_image_dataset/resized/`,
  CPU): 8 **near-native** (smallest factors, target ≤ ~1.3) + 8 **heavy**
  (largest factors, target ≥ ~2). Rank-based deterministic selection,
  seed 42; the factor table is committed alongside the run
  (`resize_factors.json`). If the corpus cannot fill a stratum at the
  target boundary, take the extreme ranks and record the realized
  boundary — do not move the estimand.
- **Per-image estimand** (mirrors E21's slice-local convention with the
  image as the slice): per (image i, σ, granularity g):
  B_i = ḡ_rp,i − ḡ_reenc,i, C_i = ḡ_dem,i − ḡ_rp,i; ⊥ against the
  image's own native direction ĝ_i = (a_i + b_i)/|…|; second moments
  from **cross-set products only**; ref-noise subtraction from the
  image's own reenc set-diff; ρ_i = I_i / 2√(S_i F_i); rel gates from
  the image's cross-set cosines, **gate rel ≥ 0.5**, gated-out images
  reported by count, excluded from verdict statistics.
- **Granularity ladder** (fixed): (1) **global ρ_i** — the verdict
  quantity; (2) per-band ρ_i (E21's four bands) — secondary; (3)
  per-cell ρ_i — exploratory only, reported with gates, no verdict
  weight. The instrument stores scalar reductions per (image, σ, band ∪
  global ∪ cell), **not** per-image vectors (which would be ~100 GB).
- Pooled arm sums are also kept (`--keep_arm_sums`) as a same-run
  cross-check of the estimator, **noting** the stratified corpus makes
  pooled values non-comparable to E14/E21's (selection differs) — the
  cross-check is internal consistency, not replication.

## 22.1 — probe amendment (GPU, daemon job) — RAN

`run_sigma_probe.py --per_image_ledger`: inside the existing per-image
arm loop, before accumulation, compute the per-image scalar reductions
above and append them to `per_image_ledger.jsonl`. No new forwards — the
same arm gradients, reduced per image instead of only summed.

As implemented: flag + cross-flag validation in `sigma_probe/cli.py`
(requires `--repromote --self_floor` + the reenc control; refuses
pool / target-kappa / draw-sweep / x-zero / target-alpha modes); ledger
math in `sigma_probe/stats.py` (`image_ledger` + `build_ledger_slices` —
global row mirrors `vector_ledger.bc_ledger`, band/cell rows mirror
`e21_cells.cell_row` incl. the additive global-perp partition); the
driver retains each image's arm vectors non-streaming (8 lists × 3 bins
≈ 7.5 GB CPU RAM at this grid) and reduces them synchronously
(~10–20 s/image against the run's GPU-hours). Validated **before any
GPU** on synthetic arms against both committed implementations (global
row exact at output rounding, slice rows bit-exact, core cells resum to
the global row), then smoke-tested end-to-end on a 2-image daemon job.
The verdict run is one daemon job, 2.1 h wall, `--deterministic
--keep_arm_sums`, launched with the frozen grid/corpus unchanged.

## 22.2 — digest + figures (CPU) — RAN

`e22_per_image.py`: per-σ distributions of gated ρ_i (global; band);
stratum overlays (near-native vs heavy); reliability accounting; the
22.3 verdict logic incl. the reliability floor; pooled same-run
cross-check read from `<run>/ledger.json` (`vector_ledger.py
--data_ref reenc`). Digest `e22_per_image.json` in this dir +
`e22_rho_i.png`; `--figs_only` redraws from the committed digest.

## 22.3 — pre-registered readings (applied — outcome in Results)

| outcome (gated images, global ρ_i, every verdict σ) | verdict |
|---|---|
| median ρ_i ≤ −0.7 **and** ≥ 75 % of gated images ≤ −0.7 | **PER-SAMPLE HOLDS** — the cancellation is a per-sample mechanism; a per-sample training-facing lever may be pre-registered (E23a, adaln-targeted first per E21). |
| median ρ_i ≥ −0.5 **or** < 50 % ≤ −0.7 | **POOLED-ONLY** — cancellation is a population property; the entire per-sample correction family (including objective debiasing) is **dead in this line**, E20.4 retroactively explained; scheduler-side routing remains the only sanctioned prescription family. |
| in between / σ-dependent | **MIXED** — record per σ; no downstream claim without a follow-up pre-registration. |

- **Reliability floor**: if fewer than 8 images pass the rel gate at any
  verdict σ, that σ is **void** (instrument-limited, not evidence). If
  all three σ are void, the run verdict is **INSTRUMENT-LIMITED**: record
  it, stop, and do not escalate draws without an amendment.
- **Resize-factor read** (observational): pre-registered null = the two
  strata's ρ_i / |B_i|/G_i / h(B_i+C_i) distributions are
  indistinguishable within the twin-based noise. A stratum difference is
  **hypothesis-generating for E23b only** — the strata differ in content
  by construction, so no causal or debias claim is made here, only
  "visible / not visible in g-space".
- **Operating-point caveat** (standing): ρ_i is measured at one fixed
  adapter (the E14 checkpoint). E19.6 established operating-point
  invariance **pooled**; per-sample invariance across training is
  untested and NOT claimed — wording must say "per-image at this
  operating point".

## E23 sketch (NOT pre-registered here — gated on 22.3; 22.1's gate not met, **22.4's met at σ = 0.7**: E23a drafting is licensed, single-σ caveat carried)

- **E23a** (only if PER-SAMPLE HOLDS): adaln-targeted lever A/B on the
  shipped σ-gated demotion recipe — damp/gate the adaln branch for
  demoted samples (E21: 86–87 % of the phase-response amplitude lives
  there; cross-attn ≈ 0 so the text pathway is untouched). Tier 1.5:
  bench + invariant test required. Target is the **residual (B+C)**,
  never B or C alone (the network already cancels ~90 % of direction —
  correcting a leg double-counts).
- **E23b** (preprocess-resize, causal): transport the demotion estimand
  one rung up — images whose sources land natively in a higher tier,
  probed with `--tier 1536 --demote_edges 1024` (surface exists;
  feasibility check is part of E23b's own pre-registration), so
  "source→1024 preprocessing" becomes a measurable demote leg instead of
  an invisible prior. Cheap side axis: resample-kernel A/B at fixed
  source. Any mitigation (factor-conditioned adaln gate, supersample-
  matched reenc, sample reweighting) must cross the same estimand bridge
  and gets its own pre-registration.
- Anti-scope: no PI-RoPE-in-window revival (G11; closed), no
  ledger-derived objective term (E20.4; closed).

## Kill switches / honesty

- Per-image estimates are ~√40 noisier than the pooled line's; the
  design answer is the draw doubling + the reliability floor above, not
  post-hoc filtering. Gate counts are always reported.
- No constant is tuned on outputs; strata, gates, draw counts, and
  verdict thresholds are frozen above before the instrument exists.
- Wording: "per-image at this operating point, probe corpus" — never
  "per-user-sample at training time".
- The resize-factor read is labeled observational in every output.

## Results (2026-08-08)

Instrument: `--per_image_ledger` amendment to `run_sigma_probe.py`
(validated synthetically against the committed `vector_ledger.bc_ledger` +
`e21_cells.cell_row` before any GPU; smoke-tested on a 2-image daemon job).
Corpus: `e22_corpus.py` → `resize_factors.json` + `e22_probe_list.json`
(near-native stratum could NOT fill at the ≤ ~1.3 target — realized
boundary 1.7857, extreme ranks taken per the pre-registration; heavy
realized min 8.7351). Run: `runs/20260808-1633-e221-per-image-ledger`
(daemon job `20260808-163340-13b303`, 2.1 h wall, deterministic, 16 images
× 3 σ × 24 draws × 2 sets, arm sums kept). Digest: `e22_per_image.json` +
`e22_rho_i.png` (22.2).

### 22.3 verdict: **INSTRUMENT-LIMITED** (pre-registered reliability floor)

Gate passes (global rel ≥ 0.5, both legs) per σ: **3/16, 3/16, 5/16** —
all three verdict σ fall below the ≥ 8-image floor ⇒ every σ is void and
the run verdict is INSTRUMENT-LIMITED. Per the pre-registration this is
recorded as instrument, not evidence; the run stops here and **no draw
escalation happens without an amendment to this file** (that amendment
now exists — 22.4 below, pre-registered after this verdict). The failing leg is
overwhelmingly C (relC < 0.5 in 37/48 image-bins; relB in 23/48): at 24
draws the per-image graph-leg direction is not reproducible for most
images — quantitatively consistent with scaling the pooled leg
reliabilities (0.66–0.84 on this same run) down by the ~√20 per-image
draw deficit. The floor did exactly what it was frozen to do.

### Descriptive record (no verdict weight — every reading, gates reported)

- **All 48 (image × σ) global ρ_i are negative**; 46/48 ≤ −0.53; median
  −0.815. The 11 gated readings: median **−0.815**, range [−0.880,
  −0.531], every one ≤ −0.53 and 9/11 ≤ −0.70. No image anywhere in the
  corpus — either stratum, factor 0.93 → 39.0 — shows a positive or
  even near-zero ρ_i at any σ. Had the floor been met with these values,
  the reading would have been HOLDS-shaped at σ = 0.4333/0.7 and
  MIXED-shaped at 0.5667 (gated frac ≤ −0.7 = 1.0 / 0.67 / 0.8).
  *(22.4 later met the floor at σ = 0.7 and confirmed the HOLDS shape;
  0.4333/0.5667 remain untested at D = 96.)*
- **Pooled same-run cross-check** (`ledger.json`, reenc ref): ρ =
  −0.909 / −0.916 / −0.891 with relB/relC 0.66–0.84 — the estimator
  reproduces the deep constant on this stratified corpus (internal
  consistency; NOT comparable to E14/E21 selection).
- Bands (gated, small n): adaln ρ_i medians −0.82…−0.85 (n = 4–5);
  cross_attn −0.51…−0.67 (n = 2–3); self_attn 0 gated everywhere; mlp
  n ≤ 2. Cells: single-digit gated counts per image — recorded in the
  digest, no reading taken.
- **Resize-factor read (observational, pre-registered null)**: with 2–5
  gated images per σ the strata are not distinguishable (Mann-Whitney
  p ≥ 0.2 everywhere) — the null stands by default at this reliability;
  no hypothesis is generated for E23b from this run. The one suggestive
  ungated pattern (the factor-39 outlier image carries the two
  shallowest ρ_i of the corpus) is noted for completeness only.
  *(→ 22.4 at healthy counts: the null now stands on evidence, not by
  reliability default, and the factor-39 outlier pattern did not recur —
  it gates in at ρ_i = −0.785.)*

### Post-hoc band diagnostic (descriptive, free re-read of the committed jsonl)

Asked after the verdict ("is the unreliability carried by the text
pathway? would an unconditional rerun fix the gates?") — answered from
the stored per-band slice-local scalars, no new compute:

| band | G_l (median native grad mass) | median relC | median ρ_i |
|---|---|---|---|
| adaln | 0.045–0.116 | 0.35–0.42 | −0.82…−0.87 |
| self_attn | 0.014–0.024 | 0.16–0.23 | −0.64…−0.74 |
| mlp | 0.013–0.023 | 0.28–0.34 | −0.69…−0.75 |
| cross_attn | 0.0025–0.0058 | 0.31–0.38 | −0.47…−0.55 |

Two separate facts that must not be conflated: (a) cross-attn is where
ρ_i is **shallowest** (−0.5ish vs adaln's −0.85 — consistent with E21's
cross-attn ≈ 0 amplitude share of the phase response), and (b) cross-attn
carries **~2 orders of magnitude less gradient mass and C-leg noise**
than the image-pathway bands, so it contributes ~nothing to the global
rel failure — the least reliable band is self_attn (0/16 gated at every
σ), and the global ρ_i is already effectively "global minus cross-attn"
by mass. A no-prompt (uncond) rerun therefore **cannot rescue the
reliability floor**: the rel gate measures reproducibility across
independent noise-draw sets at a *fixed* caption — the caption is not a
draw-noise source, and the failing variance lives in the ε/σ draws
through adaln/self-attn/mlp. An uncond variant remains a legitimate
*separate* mechanism question (text-independence of the cancellation,
predicted "yes" by E21 + row (a)) and would need its own pre-registered
amendment; it inherits the same floor at D = 24.
**→ Diagnosis verified by 22.4**: the failing variance was draw-borne
as argued — D 24 → 96 alone lifted gates 5/16 → 14/16 (self_attn
0 → 11 gated), no caption change involved.

### What this buys the paper

Superseded 2026-08-09 by 22.4 (below), which met the floor at σ = 0.7
and returned PER-SAMPLE HOLDS — the §5 sentence now comes from 22.4.
What survives from this run: the measurement is uniformly signed (every
reading, gated or not, deep-negative), and E20.4's
retroactive-explanation clause is NOT triggered (that required
POOLED-ONLY, which this is not).

## 22.4 — amendment (pre-registered 2026-08-08): D = 96 single-σ rerun — **DONE 2026-08-09** (results below)

This is that amendment: the follow-up is **decided** (pinned 2026-08-08)
and frozen here before launch. Rationale from the measured reliabilities:
rel 0.5 on C needs roughly SNR² = 1 ⇒ ~4× the draws (D ≈ 96); dropping
to the single strongest σ (0.7 — best gate pass rate, 5/16, and strongest
pooled rel) prices the rerun at ≈ 0.9× 22.1's wall clock (~2 h, one
daemon job).

**Frozen grid (no tuning on outputs):**

- Same instrument, **no code change**: `run_sigma_probe.py
  --per_image_ledger --repromote --self_floor --keep_arm_sums
  --deterministic --seed 42`, same adapter (`anima_soup_sincos` — same
  operating point as 22.1), same route **1024→768**, same arms (native
  a/b, reenc(+`__2`), 768(+`__2`), 768rp(+`__2`)). **No PI arms** (G11
  untouched; per-image C_π is explicitly out of scope — it would need
  its own amendment *after* this one clears the floor).
- **Single σ bin centered 0.7, same 2/15 bin width as 22.1**:
  `--bins 1 --sigma_window 0.6333333333333333,0.7666666666666667`.
- **`--draws_per_bin 96`.** No further escalation inside this amendment:
  if the ≥ 8-image floor fails again at D = 96, the priced SNR model is
  wrong — the verdict is INSTRUMENT-LIMITED again, record and **diagnose
  before spending more** (a third run needs a new amendment with a
  stated reason the scaling failed).
- **Corpus unchanged**: the same 16 images, `e22_probe_list.json` /
  `resize_factors.json` as committed — no re-stratification.
- **Estimand, gates, floor unchanged**: per-image estimand as frozen
  above; rel gate ≥ 0.5 both legs; reliability floor ≥ 8 gated images at
  the (single) verdict σ.

**Pre-registered readings** — the 22.3 table applied at σ = 0.7 alone;
all wording is scoped **"per-image at σ = 0.7, this operating point"**:

- PER-SAMPLE HOLDS at σ = 0.7 licenses drafting E23a's pre-registration
  (adaln-targeted, per the E21 amplitude finding), carrying the single-σ
  scope caveat into E23a's own design.
- POOLED-ONLY at σ = 0.7 — the *best* per-image bin — kills the
  per-sample correction family in this line (single-σ caveat recorded,
  but no further per-image spend without new mechanism evidence).
- MIXED / floor-fail → record, stop, no downstream claim.
- Secondary resize-factor read re-applied under the same pre-registered
  null (observational only, as above); D = 96 may simply raise the gated
  count per stratum — still hypothesis-generating for E23b at best.

## 22.4 Results (2026-08-09)

Run: `runs/20260809-0031-e224-per-image-d96` (daemon job
`20260809-003128-2c105b`, 2.4 h wall, deterministic, 16 images × 1 σ ×
96 draws × 2 sets, arm sums kept) — the frozen grid exactly as
pre-registered above, probe instrument unchanged. Digest:
`e224_per_image.json` + `e224_rho_i.png` (22.2 script, σ-subset +
`--fig` handling added for the single-σ run; regression-checked
bit-identical against the committed 22.1 digest before use).

### 22.3 table at σ = 0.7: **PER-SAMPLE HOLDS**

- **Reliability floor cleared**: 14/16 images pass the rel ≥ 0.5 gate on
  both legs (floor ≥ 8). The priced SNR model was right — D 24 → 96
  lifted the gate pass rate from 5/16 to 14/16 (pooled same-run relB/relC
  0.953/0.949 vs 22.1's 0.66–0.84). The two ungated images (both heavy
  stratum, factors 9.5/13.0) miss narrowly (relB/relC 0.41–0.50) and are
  themselves deep-negative (ρ_i −0.773/−0.867).
- **Verdict quantities (gated)**: median ρ_i = **−0.8203**, IQR
  [−0.856, −0.770], range [−0.913, −0.655]; frac ≤ −0.7 = **0.929**
  (13/14) ≥ 0.75 ⇒ HOLDS. All 16 ρ_i in the corpus — gated or not — are
  ≤ −0.655; nothing is near zero or positive. Median per-image twin-based
  σ(ρ_i) = 0.027.
- **Pooled same-run cross-check** (`ledger.json`, reenc ref): ρ = −0.898
  — the per-image median (−0.820) sits slightly shallow of the pooled
  constant, as expected when the pooled sum further averages residual
  per-image noise.
- **Bands (secondary, gated)**: adaln **−0.846** (n = 14) ≈ global;
  mlp −0.703 (13); self_attn −0.624 (11 — was 0/16 gated at D = 24);
  cross_attn −0.550 (15). Depth ordering matches E21's amplitude
  concentration (adaln carries the phase response; cross-attn ≈ 0 share,
  shallowest ρ_i) and the 22.1 post-hoc band diagnostic.
- **Cells (exploratory, no verdict weight)**: 146–202 of 280 cells gated
  per image (vs single digits at D = 24); median of per-image cell
  medians −0.67 — recorded in the digest, no reading taken.
- **Resize-factor read (observational, pre-registered null)**: with 8
  near-native / 6 heavy gated the strata remain indistinguishable —
  Mann-Whitney p = 0.41 (ρ_i), 0.75 (B amplitude), 0.49 (h(B+C)). The
  null now stands **at healthy gated counts**, not by reliability
  default: no hypothesis is generated for E23b from g-space; the 22.1
  factor-39 outlier note does not recur (that image gates in at
  ρ_i = −0.785).

### What this buys the paper (supersedes the 22.1 paragraph's caveat)

The estimand bridge is closed at the pre-registered bar, single-σ scope:
**per-image at σ = 0.7, this operating point, the cancellation holds
sample-by-sample** (median −0.82, 93 % of gated images ≤ −0.7, every
image in the corpus deep-negative). The §5 sentence upgrades from
"consistent with a per-sample mechanism but instrument-limited" to a
positive claim at σ = 0.7, with 22.1's three-σ INSTRUMENT-LIMITED record
retained as the honest reliability account at D = 24. **E23a drafting is
licensed** (adaln-targeted per E21, targeting the residual B+C, never a
leg); its pre-registration must carry the single-σ caveat and remains a
separate document. E20.4's retroactive-explanation clause stays
untriggered (POOLED-ONLY never obtained). Per-image C_π stays out of
scope (would need its own amendment, as frozen above).

## Cost ladder (planned → actual)

| item | planned | actual |
|---|---|---|
| resize-factor table | CPU, minutes (header reads over the corpus) | seconds (`e22_corpus.py`) |
| 22.1 | **GPU ~1.5–2.5 h** (scale from 19.3's 5.6 h: ×16/40 images, ×3/5 bins, ×3/5 arm families, ×2 draws), one daemon job | **2.1 h** (7686 s) incl. compile warmup (25 token families, cold signature) + in-loop ledger CPU |
| 22.2 | CPU, ~minutes | seconds (+ ~2 min `vector_ledger.py` cross-check) |
| 22.3 | free (re-read of 22.2) | free — INSTRUMENT-LIMITED via the reliability floor |
| 22.4 | GPU ≈ 0.9× 22.1 (~2 h), one daemon job | **2.4 h** (8783 s — the 0.9× price underweighted draws: 96 vs 72 total draw-bins is 1.33× the forward work, partly offset by 1/3 the bins) + seconds of 22.2 digest |
