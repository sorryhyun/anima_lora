# E22 — does the cancellation hold per sample? (per-image g-ledger)

| | |
|---|---|
| **Status** | **PLANNED 2026-08-08** — this file is the pre-registration, committed **before** the 22.1 instrument amendment exists (theory-first, mirroring 19.1/20/21). |
| **Question** | Every anti-alignment number in this line so far — E14's pooled ρ̄ ≈ −0.91, E21's cell-level LOCAL verdict — is a **cross-image mean** (40 images × 12 draws summed before any cosine). A training-facing correction acts on **one sample at one step**. Is the B/C cancellation a per-sample property (each image's data damage mirrored by its own graph response), or a population property that only emerges in the mean? This is the missing link ("estimand bridge") that E20.4 crashed into from the objective side — E22 closes it from the measurement side, **before** any correction is proposed. Secondary, observational: training "at 1024" is itself training on **preprocess-resized** pixels (source → bucket downscale; free-fit drove crop bias to ~0 but the downscale low-pass remains). Per-image records let us stratify the ledger by each image's realized resize factor — a first, hypothesis-generating look at whether that preprocessing bias is visible in LoRA-gradient space. |
| **Depends on** | [E21](../e21/) (LOCAL verdict — licenses factorized per-cell reads; adaln amplitude concentration — the candidate lever this experiment gates); [E20.4](../e20/) (derived data term fails at estimand level — the standing reason per-sample must be measured, not assumed); `run_sigma_probe.py` + `sigma_probe/` (the per-image arm loop this amends); E14/E19 ledger conventions. **The 19.3/19.4 stores cannot answer this**: they are cross-image sums, and `per_image.jsonl` carries only arm-vs-native cosines (`cos_<arm>`), never the arm×arm cross products that ρ_i needs. New GPU arm required. |
| **Instruments** | 22.1 probe amendment `--per_image_ledger` (GPU, daemon): emit per-image debiased B/C scalar reductions inside the existing per-image loop; 22.2 `e22_per_image.py` (CPU digest + figures); 22.3 free re-reads of 22.2. |
| **In the paper** | The mechanism→prescription bridge for §5/discussion: PER-SAMPLE HOLDS licenses a training-facing lever (E23a); POOLED-ONLY kills the whole per-sample correction family in one pre-registered stroke and confines prescriptions to scheduler-side routing (the shipped σ-gated demotion recipe). Either outcome is a paper paragraph. |

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

## 22.1 — probe amendment (GPU, daemon job)

`run_sigma_probe.py --per_image_ledger`: inside the existing per-image
arm loop, before accumulation, compute the per-image scalar reductions
above and append them to `per_image_ledger.jsonl`. No new forwards — the
same arm gradients, reduced per image instead of only summed. Costed
below; runs as one daemon job.

## 22.2 — digest + figures (CPU)

`e22_per_image.py`: per-σ distributions of gated ρ_i (global; band);
stratum overlays (near-native vs heavy); reliability accounting; digest
`e22_per_image.json` in this dir + `e22_rho_i.png`.

## 22.3 — pre-registered readings

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

## E23 sketch (NOT pre-registered here — gated on 22.3)

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

## Cost ladder

| item | cost |
|---|---|
| resize-factor table | CPU, minutes (header reads over the corpus) |
| 22.1 | **GPU ~1.5–2.5 h** (scale from 19.3's 5.6 h: ×16/40 images, ×3/5 bins, ×3/5 arm families, ×2 draws), one daemon job |
| 22.2 | CPU, ~minutes |
| 22.3 | free (re-read of 22.2) |
