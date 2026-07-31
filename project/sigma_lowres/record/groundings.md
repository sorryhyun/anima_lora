# sigma_lowres — groundings: the evidence ledger for hypothesis.md

Each entry records one measurement: design, pre-registration status,
run pointer, result, and **what it grounds** in the account
(`hypothesis.md`). G1–G3 were pre-registered 2026-07-24 against the v1
two-term account (predictions committed before the runs finished); G5–G7
are the 2026-07-26 refinement probes; G8 is post-hoc (2026-07-26,
explicitly flagged). Instrument: `bench/run_sigma_probe.py` unless noted —
gap = split-half floor cosine − cross-grid cosine (cosine units, SEM
~0.02), reenc arm as the instrument-validity control (band ±0.04).

## Pre-registered outcome matrix (v1, frozen before G1/G2 landed)

| Endpoint (G2) | x-zero (G1) | Verdict |
|---|---|---|
| ≈ plateau | ≈ endpoint | **← MEASURED OUTCOME.** Two-term account holds, **graph-dominated** → Q2 per-module split (J-decomposition) is the mechanism probe; capacity governor favored |
| ≈ plateau | ≪ endpoint | Account holds, **content-dominated** → mechanism is resolution-conditioned prior error; content-side interventions open |
| ≈ 0 | (any) | **Account falsified** — input branch explains everything; spectral story revives with a slower decay constant |
| ≈ plateau | ≫ endpoint | Anomaly — account incomplete, reopen |

## G1 — x-zero probe (Test 1: isolates the J-term)

`run_sigma_probe.py --x_zero`: image zeroed in BOTH input and target on
every grid (input = σε, target = ε; captions + exact demoted latent shapes
kept). No content exists anywhere → any surviving gap is **pure
graph-shape sensitivity**. Run: 40 images, 4 σ-bins + σ=1 endpoint, edges
896/768/512, 8 draws/bin.

- Pre-registered read: xz_gap ≈ endpoint gap → Floor graph-dominated;
  xz_gap ≪ endpoint → Floor is content-correspondence in the residual.
- Secondary prediction: xz curve ~flat in σ (no content to fade). Caveat:
  low-σ bins are off-manifold (input σε is norm-shrunk) → σ=1 read primary.
- Anomaly bar: xz_gap substantially *above* endpoint would reopen the
  account.

**RESULT (2026-07-24, `results/20260724-2136-xzero/`): graph-dominated.**
σ centers 0.125/0.375/0.625/0.875/1.0; SEM in parentheses at the endpoint:

| edge | xz gap across σ | xz @ σ=1 | endpoint (G2) @ σ=1 |
|---|---|---|---|
| 896 | .012 .039 .014 .031 .004 | 0.004 (.018) | −0.009 (.042) |
| 768 | .062 .046 .033 .117 .064 | 0.064 (.028) | 0.127 (.054) |
| 512 | .188 .135 .158 .260 .299 | 0.299 (.040) | 0.326 (.059) |

- **512: xz ≈ endpoint** (0.30 vs 0.33) — Floor essentially all
  graph/function term; per-image data content contributes ~nothing.
- **768: xz ≈ half the endpoint** (0.06 vs 0.13, SEM-overlapping) — graph
  term is the bulk, possible minority content-correspondence share.
- **896: xz ≈ 0 everywhere**, matching endpoint ≈ 0.
- Flatness: no S1-like decline anywhere (512 mildly *rises*; low-σ bins
  are the off-manifold regime). Anomaly bar not triggered. Split-half
  0.74–0.88 (768's noisier 0.28 driven by the wide 0.875 bin; endpoint
  bin is the verdict bin).
- Bonus decomposition check vs Phase 0 at LOW σ: xz(512) ≈ 0.13–0.19 vs
  standard 0.35–0.47, xz(896) ≈ 0.01–0.04 vs standard 0.11–0.16 — the
  low-σ elevation in Phase 0 is S1 sitting on exactly this Floor, as the
  two-term account requires.
- Interpretation caveat (sharpens, doesn't weaken): with x = 0 the
  residual is ≈ −x̂_prior — the model's own grid-conditioned prior (xz ‖g‖
  at σ=1 is 39.9, large despite zero content). "Graph-dominated" means
  Jᵀx̂_prior mismatch — the network function across token counts,
  including its resolution-conditioned prior — NOT per-image content.

**Grounds**: Floor_e is J-side, not content (the G4 per-module split is
its decomposition; G6 later removes the x̂_prior factor too).

## G2 — σ=1 endpoint bin (Test 2: measures the Floor by construction)

`run_sigma_probe.py --bins 0 --endpoint_bin`: at σ = 1 the input is
exactly ε — the input-information term is zero by construction; any
measured gap IS the Floor. Standard arms (native ×2, reenc, 896/768/512),
40 images, 16 draws.

- Pre-registered prediction: gap(σ=1) matches the Phase-0 high-σ plateau
  per edge — 896 ∈ [0, 0.08], 768 ∈ [0.04, 0.15], 512 ∈ [0.2, 0.4], tier
  ordering preserved, gap_reenc within ±0.04.
- **Falsifier (the account's kill switch)**: endpoint gaps ≈ 0 for all
  edges → the σ=0.94 persistence was carried by the residual (1−σ) input
  signal, and a pure-input story revives.

**RESULT (2026-07-24, `results/20260724-2101-endpoint/`): prediction
PASSES on all three edges; falsifier not triggered.** cos_floor 0.86,
‖g‖ 63.9 (the σ→1 tail, as expected).

| edge | gap @ σ=1 (SEM) | predicted band | verdict |
|---|---|---|---|
| reenc | −0.038 (.032) | within ±0.04 | instrument valid |
| 896 | −0.009 (.042) | [0, 0.08] | ✓ ≈ 0 |
| 768 | +0.127 (.054) | [0.04, 0.15] | ✓ |
| 512 | +0.326 (.059) | [0.2, 0.4] | ✓ |

**Grounds**: Floor_e exists, is σ-independent (matches the high-σ
plateau), tier-ordered, and cannot be input-side; also exonerates the
latent-space quirk (reenc in band; input statistics at σ=1 are ~identical
Gaussians across arms).

## G3 — content-loss correlation (Test 3) + pooled addendum

From Phase-0 `per_image.jsonl`: correlate per-image high-σ gap with
content lost to demotion (latent down-up error; HF energy above demoted
Nyquist). S2 (content) predicts positive correlation; S3 (graph-only)
predicts none.

**RESULT (2026-07-24): cannot be measured at 8 draws.** The reliability
ceiling of the per-image high-σ gap — agreement between the two top
σ-bins across 40 images — is *negative* (r ≈ −0.09..−0.18 all edges):
the per-image gap is estimator noise with no stable image-level component
(same per-image-ranking failure as tier_routing 3a). All correlations
null (|ρ| ≤ 0.23, p > 0.15) **against a ~0 ceiling** — reads "instrument
blind", not "no effect".

**Group-level addendum (2026-07-25, `results/20260725-2155-pool4/`)**:
`--pool` mode re-asks at the granularity where per-image noise cancels —
per-image bin-gradients summed across images (redundancy-sorted strata of
4 + all-images aggregate) before cosines; see report.md
"Pooled-gradient addendum". Two reads: (1) pooled gap_896 collapses to
≈ 0 (within the pooled reenc control) at every bin σ ≥ 0.625 — the
per-image high-σ residual averages out, behaving as draw noise, not a
shared content-loss component; (2) the stratum-level redundancy →
pooled-gap trend is null (Spearman −0.07 / −0.05 across 10 strata,
redundancy 0.49–0.93) — the content covariate S2 needs is absent at the
group level too.

**Grounds**: closes the content-side (S2) reading from both directions;
the S2-vs-S3 verdict rests on G1, and the pooled view confirms what the
per-image instrument was too noisy to see isn't there.

## G4 — Q2 per-module / per-block split (Floor localization)

Runs `results/20260724-2306-endpoint-pg/` + `20260724-2343-xzero-pg/`
(`--per_group`); full record in **report.md "Phase Q2"**. Outcome: the
Floor localizes in **depth** (early blocks ~3× late, uniformly across
every module type within a block; content share is a late-block minority
effect); **RoPE refuted as a concentrated mechanism** (self-attn
up_q/up_k show zero excess over up_v); adaln among the highest with
ep−xz ≈ 0 (global-statistics shift, not content).

**Grounds**: the J-mismatch is an early-block representation property,
not a parameter circuit; "safe subset" is a depth band. **Origin-side
revision in G10** — the landing-side q/k-vs-v uniformity that refuted RoPE
here cannot localize *origin*; G10's intervention shows the 768 Floor is
RoPE-originated after all (propagation makes the landing uniform).

## G5 — 1280→1024 probe (route ordering breaks pure capacity)

Report.md "1280→1024" section, `results/20260726-2017/` (probe-local
1280 cache via `bench/prep_1280_probe.py`). Fits the two-term shape —
Floor_1280→1024 ≈ 0 (σ ≥ 0.875 and endpoint in-band, per-image and
pooled) under an S1 that decays *slower* than 1024→896's (crossover
~0.75 vs ~0.5) — but breaks the pure-capacity reading of route ordering:
capacity alone predicts σ\*(1280→1024) ≤ σ\*(1024→896); measured the
opposite. Ratio is likewise refuted as sole governor by the same data
(0.80 floors, 0.857 never does — because the latter's Floor > band).

**Grounds**: the hybrid split — S1 amplitude (A_e) tracks route severity
while the Floor is a separate graph/target-side term. Retrodicts all
three original routes (hypothesis.md retrodiction 6). Left open here,
discriminators in hypothesis.md: A's governor (iso-severity 1280→1120)
and Floor checkpoint-dependence through J (carving test).

## G6 — prior-distance probe (no 1280 discontinuity; prior ↮ Floor)

`bench/run_prior_distance.py`, `results/20260726-2120/` — base DiT only,
no adapter, no gradients: 16 images × 5 grids (native-1280 bucket + the
same image's 1024/896/768/512 demoted buckets) × 16 pure-noise σ=1
forwards, 4.5 min. Per grid, x̄ = E_ε[z − v̂] is the caption-conditioned
prior drift; adjacent pairs compared on the lo grid (area-downsample),
rel-L2 excess over a split-half floor (floors ≈ 0.02 everywhere,
cos ≈ 0.993–0.997):

| pair | 1280→1024 | 1024→896 | 896→768 | 768→512 |
|---|---|---|---|---|
| excess | .069 ± .007 | .069 ± .005 | .061 ± .006 | .104 ± .007 |

- **"Never learned 1280" conditioning REFUTED.** No discontinuity at the
  training-distribution edge: the prior distinguishes 1280-vs-1024
  exactly as much as heavily-trained pairs (~0.07 relative). The "a 1280
  tier has nothing to teach" corollary falls with it.
- **Prior distance DISSOCIATES from the Floor** — gradient Floors order
  ≈0 < 0.06–0.13 < ~0.3, yet prior distances are flat across the top
  three pairs. By elimination the Floor's route-ordering lives in the
  **J (graph) factor** — consistent with G4's depth localization.
  Checkpoint-dependence *through J* remains untested (carving test).
- Caveat: part of d is resampling artifact scaling with ratio harshness
  (the 0.67-ratio 768→512 pair jumps; top three ratios cluster at
  0.80–0.875), so the flat top-three is weak evidence alone. Load-bearing
  reads are the **absent discontinuity** and the **dissociation**,
  neither producible by the artifact.

**Grounds**: Floor_e is not carried by x̂_prior (closes G1's caveat);
prior-side checkpoint conditioning refuted.

## G7 — σ-resolved residual curve (route-UNIFORM m(σ))

`run_prior_distance.py --sigmas 0.125,0.375,0.625,0.875,1.0 --draws 8`,
`results/20260726-2133/` (16 min). Verdict object: the **mean residual**
`r̄ = E_ε[v̂(z_σ) − (ε − x)]` per grid/σ — the exact `r` in `g = Jᵀr`,
free of the content/resampling baseline (reported separately as
d_content ≈ 0.26–0.31/pair) — with a same-grid reenc control (≈ 0
everywhere, ≤ 0.02). Cross-grid excess over split-half floors:

| σ | 0.125 | 0.375 | 0.625 | 0.875 | 1.0 |
|---|---|---|---|---|---|
| 1280→1024 | .885 | .825 | .715 | .465 | .360 |
| 1024→896 | .862 | .808 | .706 | .459 | .378 |
| 896→768 | .860 | .814 | .715 | .466 | .393 |
| 768→512 | .897 | .872 | .767 | .508 | .435 |

- **OVERLAP (the pre-registered null for S1-severity)**: the three main
  routes coincide within ~±0.02 at every σ (768→512 only +0.04–0.06),
  while their gradient Floors span 0 → 0.3 and their σ\* spans 0.5 →
  nonexistent. The residual has a strong, **route-uniform** σ-shape
  (0.88 → 0.36, mirroring the U-shaped ‖r̄‖/gnorm) but essentially no
  route ordering. Paired per-image 1280→1024 − 1024→896: +0.023 → −0.018
  across σ — real but ~50× smaller than the curve itself.
- Fine print: at σ=1 a mild capacity ordering appears (.360 < .378 <
  .393 < .435); at low σ it inverts (+0.02) — both tiny against the
  route-uniform bulk.

**Grounds**: m(σ) is universal and monotone; everything route-specific
(A_e and Floor_e) is J-side. The σ\* ordering is NOT prediction-side — a
forward d(σ) explains the *shape* of the decline, never the *boundary
ordering*.

## G8 — gradient-norm renormalization check (POST-HOC, 2026-07-26)

Not pre-registered — added after noticing the gap curves peak at
σ ≈ 0.19–0.44 with a dip at the lowest bin, a shape the monotone-S1 v1
account doesn't produce. Story under test: gap is a cosine (mismatch
*fraction*), so gap ≈ mismatch/G with G = ‖g‖ U-shaped
(2.56 → 0.39 @ σ=0.31 → 9.26 in Phase 0). CPU-only re-analysis of
existing `per_image.jsonl` (Phase 0 `20260724-1237-phase0/` + Phase 1a
`20260724-1523-phase1a-t896/`): `bench/check_gap_renorm.py`.

- **Gap tracks 1/G across bins**: Spearman(bin-mean gap, 1/‖g‖) = +0.55
  / +0.90 / +0.67 for 896/768/512 (Phase 0), +0.81 / +0.48 (Phase 1a).
- **It holds within images**: per-image rank correlation of the two
  curves, mean +0.34..+0.47, 78–93% of images positive — a shape
  property, not an aggregation artifact.
- **Renormalizing removes the low-σ dip in every arm of both runs**: the
  absolute-mismatch proxy (gap − Floor)·‖g‖ flips bin 0.06 from below
  the peak to the maximum in all 5 arms when each edge's own high-σ
  plateau is the Floor (e.g. Phase-0 896: raw .110 < .162 → renorm
  0.147 > 0.058; with the G2 endpoint floors instead, 896 still flips
  outright and 768/512 land within noise of the peak). The restored
  numerator is low-σ-maximal, decaying into flat-within-noise by mid σ
  (mid-bin renorm differences ~0.01–0.03 ≈ SEM·G).
- **Limits**: the proxy is meaningless at σ ≳ 0.8 ((gap − Floor) ≈
  0 ± SEM × G² ≈ 86); the exponent p is not pinned (·G and ·G² both
  restore monotonicity; cosine geometry gives 2 in the small-orthogonal
  limit); and gap and G come from the same forwards, so a common cause
  fits as well as strict division. Consistency, not proof.
- Interpretive anatomy of G's U-shape (high-σ plan window ↔ image-scale
  residual; mid-σ refinement ↔ small residual; low-σ irreducible ε +
  shared LF): consistent with the independent cross-attn drive
  measurement (`docs/inference/xattn_boost.md`, drive floors below
  σ ≈ 0.85), but nothing measures plan-commitment and the gap in the
  same run.

**Metric policy note**: this does NOT change the claim/criterion metric.
All pre-registered bars (reenc band, endpoint Floors, σ\*) stay in gap
(cosine) units — direction is what the optimizer consumes (Adam
normalizes magnitude), and the renormalized proxy cannot carry a safety
bar in the σ > 0.5 region where routing decisions live. G8 explains the
*shape*; it is not a replacement observable.

**Grounds**: the G(σ)^p denominator in S1; retrodiction 2 (interior
peak/dip); part of retrodiction 1 (why σ\* is a ratio property).

## G9 — iso-severity probe 1280→1120 (A_e's governor: ratio, not capacity)

`results/20260726-2153/` (daemon job `20260726-215301-17601f`, 59 min):
24 images × {reenc, 1120, 1024} arms on the probe-local 1280 cache
(`prep_1280_probe.py`), bins 0.125/0.375/0.625/0.875 + σ=1 endpoint,
4 draws/bin. 1280→1120 is edge-ratio 0.875 — **matched to 1024→896** —
with target capacity 4825 tok vs 896's 3012 (1.6×); 1120 is a
synthetic-band non-tier grid (mechanism value only). Pre-registered
contrast (hypothesis.md open edge, frozen before the run): A ~ ratio →
σ\*(1280→1120) ≈ 0.5 (matches 1024→896); A ~ capacity → floors earlier.

| σ | 0.125 | 0.375 | 0.625 | 0.875 | 1.0 |
|---|---|---|---|---|---|
| reenc | +.047(.027) | −.005(.027) | +.048(.034) | +.020(.036) | −.012(.016) |
| **1280→1120** (r 0.875) | .129(.024) | .110(.027) | .054(.032) | −.012(.024) | −.007(.016) |
| 1280→1024 (r 0.80) | .170(.032) | .178(.034) | .111(.020) | .002(.030) | .061(.062) |
| 1024→896 (r 0.875, pool4 run, same bins/draws, n=40) | .132(.029) | .076(.020) | .077(.024) | .049(.038) | .100(.045)* |

\* pool4's 4-draw endpoint is noisy; 896's canonical 16-draw endpoint is
−0.009 (G2).

**VERDICT: A ~ ratio**, on two independent reads:

1. **σ\***: 1120 is clearly elevated at 0.375 (excess over reenc +0.115,
   3.0 SEM) and within the reenc control by 0.625 → σ\* ∈ (0.375,
   0.625) ≈ 0.5, matching the ratio-matched 1024→896. The same-native
   1024 arm (ratio 0.80) is still elevated at 0.625 (excess 0.063,
   1.6 SEM) and in-band at 0.875 — replicating G5's σ\* ∈ (0.625,
   0.875) at 4 draws.
2. **Curve coincidence (the stronger read)**: the two ratio-matched
   routes coincide within ~1.4 combined SEM at every bin despite the
   1.6× capacity difference, while the capacity-ordered same-native
   comparison (1120 vs 1024 arms) separates. Amplitude follows ratio.

- Floor_1120 ≈ 0 (endpoint −0.007 ± .016) — consistent with the Floor
  being governed by absolute target capacity / graph smoothness, which
  completes the hybrid's division of labor: **ratio sets A_e (S1
  amplitude); absolute target capacity sets Floor_e**. This also
  dissolves the apparent conflict with report.md's "ratio is refuted as
  the governor" — that verdict concerns the *Floor/σ\*-route-ordering*,
  not the S1 amplitude.
- Caveats: 4 draws/bin, n=24 (per-bin SEM ~0.03); the reenc control sits
  marginally outside the ±0.04 band at 0.125/0.625 (+0.047/+0.048,
  ~1.4 SEM) — the σ\*-in-(0.375, 0.625) read survives it, but the exact
  σ\* is not localized further (a `--sigma_window 0.375,0.625` run
  would pin it; mechanism value only).

**Grounds**: A_e ~ ratio (severity); with G5, the ratio/capacity
division of labor between A_e and Floor_e.

## G10 — PI-aligned RoPE probe (Floor decomposition: RoPE_e + Resid_e)

`run_sigma_probe.py --pi_align`, `results/20260727-1122` (40 images ×
16 draws, σ=1 endpoint-only, edges 896/768/512 + pi twins + reenc,
`--per_group`; smoke `20260727-1117`). DyPE-motivated (arXiv:2510.20766)
**origin-side intervention**: per demote edge an extra `<edge>pi` arm runs
the same demoted latent with RoPE at PI-stretched fractional positions
(patch `i` at `i · H_nat/H_dem` per axis, `generate_embeddings_scaled`),
making the demoted grid's relative phase geometry exactly match native.
Read committed before the run (conversation-level, not doc-frozen):
pi ≪ plain ⇒ RoPE share real; ≈ ⇒ Floor elsewhere (G4 confirmed from the
origin side); 896pi ≈ 0 as off-manifold control.

| edge | plain @ σ=1 (SEM) | pi (SEM) | paired Δ (SEM) | pi better |
|---|---|---|---|---|
| 896 | −0.021 (.041) | −0.040 (.040) | — | control ✓ |
| 768 | +0.080 (.048) | **−0.001 (.039)** | +0.081 (.031) | 78% |
| 512 | +0.320 (.058) | **+0.224 (.056)** | +0.096 (.039) | 70% |

- **768's Floor erased** to within the instrument band (median .059→.037);
  G2's Floor reproduced (+0.080 this pool vs canonical +0.127).
- **512's bulk survives** exact phase alignment (~30% shaved, 0/40 images
  in-band) — non-PE graph residue (softmax-over-N / normalization /
  capacity); caveat: 2× stretch is also the sparsest sampling of the
  trained coordinate range.
- **Landing ≠ origin**: plain-768 per-type landing is flat (q .089 /
  k .100 / v .090) yet the origin intervention removes the Floor — G4's
  RoPE refutation was a landing-side inference and is overturned for the
  mild route. The pi residue shows mild q>k>v (.040/.031/.021). Depth
  profile preserved under pi (early-block dominance in both components).
- Outlier 1/40 (`songchuan_li/dan_6583014`: 896pi +0.698) — scaled rope
  misbehaves on one grid; medians robust; inspect before trainer wiring.
- Caveats: single operating point; endpoint-only (σ-resolved pi curve
  owed); pi possibly off-manifold at low σ (smoke hint) — irrelevant to
  the σ>gate window but untested below it.

**Grounds**: Floor_e decomposes — **Floor_e = RoPE_e (coordinate-system
share, erasable by PI position alignment; the large majority of 768's,
~30% of 512's) + Resid_e (non-PE graph term; the true irreducible
Floor)**. Capacity governs Resid_e; the Phase-1a 896→768 verdict was
conditionally reopened for a PI-aligned 1024→768 high-σ route — closed
same day by G11.

## G11 — σ-resolved pi probe (route gate: PI-768 CLOSED; stretch off-manifold in-window)

`results/20260727-1234` (40 images × 8 draws/bin, bins 4 on σ∈(0.5,1.0) +
endpoint, arms reenc/768/768pi; 86 min). Gate pre-registered in
roadmap.md before the run: in-band by σ ≈ 0.6–0.7 → proceed; never
in-band → close. Full table in report.md "σ-resolved pi probe".

- **768pi WORSE than plain 768 through σ 0.56–0.81** (paired −0.05..−0.11,
  ≤3.7 SEM), better only at σ ≥ 0.94: the stretch is off-manifold once
  content is in the input — RoPE_e is erasable only in the
  noise-dominated regime. Training through pi-rope forwards would void
  the zero-adaptation substitution premise, so no trainer path.
- **S1(1024→768) fatal on its own**: plain 768 excess over reenc
  +0.09–0.22 across the window (~4 SEM at σ=0.94) — ratio 0.75, per
  A_e ~ ratio (G9).
- Tempering of G10's "erased": paired pi−reenc endpoint excess here
  +0.065 (.024); the morning absolute zero rode a −0.045 reenc draw.
  RoPE_e = large majority of Floor_768, not exactly all.

**Grounds**: the PI-768 route is closed (both the lever and the route);
G10's decomposition survives as mechanism (endpoint regime); safe set
unchanged. RoPE_e is a *noise-regime* coordinate artifact — its removal
does not transfer into the contentful window.
