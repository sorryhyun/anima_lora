# Open questions — companion to "When Does Training on Downscaled Images Yield the Same Gradients?"

*2026-07-30. Section/equation numbers refer to the current draft (main.pdf in
this directory). This note is written to accompany the paper: the effects
below are measured and replicated at our operating point, but several
ingredients of our account are calibrated rather than derived, and the
account fails in one diagnosable place. We are circulating this to ask two
things: (a) is there an existing theory, or a better hypothesis, for any of
the mechanisms below; (b) which additional measurement would you run first?*

*Update, same day: the first review round (`response.md`) and the
measurements it triggered answered Q4 outright and Q5's measured half —
those sections are reduced to their verdicts below, and two passages
(the Q1 "≤ 0.10" falsification bound, Q6's "batch-size-1 lower bound"
framing) are amended per the review. Full records: `response.md`,
`additional_question.md`, `action.md`.*

## What we take as established

- Training on downscaled ("demoted") latents perturbs the expected adapter
  gradient, and safety is a per-route noise-level boundary — a (route, σ)
  map, not a single crossover (§4.1). Input-level spectral sufficiency does
  not transfer to gradients (§4.2).
- The gap has a **σ-independent graph floor**, present at pure-noise input,
  ordered by absolute target capacity (endpoint ≡ x-zero ≡ α-flat probes,
  §4.3), plus a **data term**: a route-uniform residual mismatch m(σ) read
  over a U-shaped gradient norm, with amplitude governed by edge ratio
  (§4.4).
- At the endpoint the floor decomposes as **Floor_e = RoPE_e + Resid_e**
  via exact positional interpolation (§4.5): the 768 floor (≈0.07) is
  essentially all RoPE (Resid_768 ≈ 0); the 512 floor (≈0.30) splits into
  RoPE ≈ 0.10 + Resid ≈ 0.20; the 896 floor (0.02–0.04) is below the
  probe's paired resolution and remains undecomposed.
- The reduced two-term account (Eq. 6) predicts held-out routes at
  0.07–0.09 RMSE (§4.6) — but the 768 route's mid-σ window sits *outside*
  the bootstrap 95% band, in the direction no positive two-term form can
  produce (measured ≈ 0, below its own predicted floor ≈ 0.09).
- A σ-gated, frequency-selective ("YaRN-style banded") rotary alignment on
  demoted forwards is on-manifold in-window and improves demoted training
  (§5.2), whereas uniform positional interpolation is off-manifold with
  content in the input (worse than plain demotion through σ ∈ [0.56, 0.81]).
- *(added 2026-07-30)* The residual mismatch is route-uniform **in norm
  only**: across routes its directions are near-orthogonal at low σ and
  image-specific everywhere (E11 — old Q5's measured half).
- *(added 2026-07-30)* The endpoint target-content share is real and
  large (|t|/G ≈ 2.2) and lands ~parallel to ḡ_src, shortening under
  demotion — which is why the angular gap cannot see it (E10 — old Q4).

## Q1 — What is the σ-resolved composition of the graph term?

**Where the account is thin.** Eq. 6 carries the graph term as a single
σ-flat constant Floor_e, and its RoPE/Resid split is certified only at
σ = 1: the PI intervention that isolates RoPE_e is off-manifold in-window,
so we have no measurement of RoPE_e(σ), and no derivation that places a
positional term inside Δ_e(σ). The banded variant's own behavior (a low-σ
liability that the σ-gate had to remove) is direct evidence that positional
effects are *not* σ-uniform in-window — i.e., the endpoint split probably
does not extend as two flat constants.

**What a theory would look like.** An operator-level split of the graph
branch, ΔJ = ΔJ_rope + ΔJ_rest: the rotary phase mismatch enters attention
logits analytically, so a perturbative expansion in per-band phase error —
weighted by the residual's σ-dependent spectral content — could yield a
derived RoPE_e(σ) inside the four-term expansion (Eq. 5). We have not found
this done anywhere; pointers welcome.

**What we would measure.** (a) An *on-manifold partial-alignment dial*: sweep
the banded alignment's band threshold / strength as a family of partial
interventions per σ-bin and extrapolate to the full-alignment limit — a
RoPE_e(σ) estimate that never leaves the data manifold. (b) **Banded
alignment on 1024→512** — *amended 2026-07-30*: originally framed as a
falsification test ("can remove at most ≈ 0.10 of the 0.30 floor"), but
the review is right that PI/banded erasure measures F_rope + I_rope,rest,
not a pure RoPE share, so cross-terms void the bound. The run survives as
an **intervention-effect measurement** with the three-piece vector report
(F_rope, F_rest, I_rope,rest) — the first in-window RoPE-side datum on a
harsh route either way.

## Q2 — Why is the floor exponential in tokens, and what *is* Resid, mechanically?

**Where the account is thin.** The floor law F(n) = 0.70·e^(−n/1041 tok)
fits two anchors and predicted the third (Floor_1120 ≈ 0), but it is
calibrated, not derived — as is the band-counting sketch behind the
absolute-size governor. We claim no closed form (§3.3), and "Resid" is
currently a residue defined by subtraction, attributed only generically to
attention-over-N statistics, sequence-length-dependent normalization, and
coarse-graph capacity.

**What a theory would look like.** A first-principles account of the graph
share as a function of token count: signal-propagation / attention-entropy
arguments (softmax over N shifts logit statistics ∝ log N), an NTK-style
perturbation account of ΔJ under grid change, or a quantitative version of
rotary band decorrelation counting. Two specific targets: what sets
τ ≈ 1041 tokens, and is the exponential family even right (we have two
anchors and one out-of-sample check)?

**What we would measure.** Designated origin-side erasure probes in the PI
mold, one per candidate mechanism: an attention-temperature correction
(rescale logits by the log-N shift — the "PI of attention statistics") and
per-block ΔJ probes. *(The normalization-recalibration arm was dropped
2026-07-30: our norms are per-token — no explicit sequence-length
statistic to recalibrate.)* Whichever intervention erases part of Resid_e
names its mechanism; a null across both would point to capacity proper.
Per the review, the floor-law fit itself moves to κ_eff² units with a
three-family comparison (e^(−N/τ), e^(−√N/ℓ), N^(−p)), and the
discriminating run is a fixed-target / varied-source x-zero ladder (the
spectral-tail secant predicts source-N₀ dependence; target-only capacity
predicts none).

## Q3 — What makes the projected interaction negative — and where else?

**Where the account is thin.** The 768 mid-σ failure has a unique signature
in the four-term expansion: I_768(σ) < 0 (equivalently, graph-relative
stationarity fails there). The account *represents* the failure but does not
explain it — assumption (ii) is a smallness assumption with no theory of the
interaction's sign or magnitude. Why do the data-branch and graph-branch
perturbations anti-align on that route, in that window, and apparently not
elsewhere?

**What we would measure** *(amended 2026-07-30 — the review showed the
original scalar read Δ_demote − Δ_repromote − Floor_e actually estimates
I(σ) + [F(σ) − F(1)], conflating a negative interaction with a graph share
that collapses below its endpoint value)*: the demote–re-promote probe now
retains the mean gradient **vectors** and reads the interventional split
directly — B = ḡ_repromote − ḡ_native, C = ḡ_demote − ḡ_repromote, with
cross-draw-set-debiased S/F/I per (route, σ-bin). Pre-registered
two-branch resolution: I_768(σ) < 0 in-window (the review's
amplitude-matching account further predicts the window center sits where
|B⊥| ≈ |C⊥|), or F_768(σ) collapsing below F_768(1) (graph-share
σ-dependence). Implemented (E9); run pending.

## Q4 — ANSWERED 2026-07-30: parallel landing

The exact affine read t = ḡ(α=1) − ḡ(α=0) (zero draw noise at shared
seeds; E10, N=40 endpoint) resolved this: |t_src|/G ≈ 2.2 — J^⊤ does
**not** attenuate the target content — and δt = t_dem − t_src is
κ∥-dominated (aggregate −0.75/−1.18/−1.86 on 896/768/512 vs κ⊥
0.09/0.14/0.20, reproducible ≥ 0.995; reenc control at the noise floor).
Demotion *shortens* the target-content gradient along ĝ_src; the angular
gap is blind to that rescaling by construction, which is why the α-sweep's
scalar read was flat. Rotation appears only on the harshest route
(cos(t_512, t_src) = 0.74). Details: `additional_question.md`,
`action.md` §4.3.

## Q5 — What predicts the route-uniform residual mismatch?
*(measured half ANSWERED 2026-07-30 — the theory half is now sharper and
still open)*

**Answered by measurement (E11).** Route-uniformity is **norm-only**: the
per-route Δr̄ directions are near-orthogonal at low σ (corrected cos
≈ 0–0.08 non-adjacent), only weakly shared at high σ (+0.2–0.33), and
**image-specific everywhere** (cross-image direction consistency ≈ 0 at
every route and σ) — so a rank-one universal mismatch mode is excluded,
and a grid-conditional content prior ("small canvases are portraits") is
refuted as the carrier under full captions. Δr̄'s energy migrates to low
spatial frequencies as σ rises (low-third share ≈ 0.40 → 0.67).

**What a theory now owes (open).** A forward account of the DiT's
cross-band response that produces **one universal amplitude curve m(σ)
(0.89 → 0.36 relative L2 over σ) riding on image- and route-specific
directions** — uniform norms without a shared mode. The
posterior-covariance mechanism (D_z v* off-diagonals ∝ Cov(x_ω, x_ω′ | z))
survives only in a form with route-independent trace but image-dependent
eigenvectors; what statistic of D_z v̂ tests amplitude-universality
directly?

## Q6 — Which estimand does training actually care about?

**Where the account is thin.** The per-example and batch-aggregate
estimands genuinely disagree at high σ (pooled arms collapse the 896 excess
at σ ≥ 0.625 and 768 at σ ≥ 0.875). *Amended 2026-07-30: we no longer call
the per-example map a "batch-size-1 lower bound" — monotone improvement
with batch size requires an iid zero-mean disagreement model we have not
verified. The honest object is the decomposition E[d_B] ≈ intercept
(coherent drift) + (1/B)·covariance term.* There is no theory of when the
per-image residual behaves as cancelling draw noise versus shared bias —
and the two estimands disagree exactly where deployment decisions get made.

**What we would measure.** Pooled-arm self-floors (so the aggregate object
gets the same debiasing the per-example object has), an **a + b/B
batch-size fit** of the excess, a paired shadow-Adam replay (frozen
optimizer state applied to both gradient streams at the real batch/accum)
as the cheap pre-A/B instrument, and the long-horizon closure: whether an
integrated fixed-step training run tracks the per-example or the aggregate
map (§5.2's A/B is the existing instrument for this).

## If you only have bandwidth for one pointer

*(updated 2026-07-30)* With Q4 answered and Q5's measurement in, the open
core is Q3's vector-resolved probe (implemented, run pending) and **Q2 —
which is where an existing theory, if one exists in the long-context /
length-generalization literature that we have missed, would most change
the paper**. We would genuinely rather be scooped by a derivation of Q2
than keep the floor law empirical. The sharpest *new* theory target this
round produced is Q5's residue: universal amplitude on image-specific
directions.
