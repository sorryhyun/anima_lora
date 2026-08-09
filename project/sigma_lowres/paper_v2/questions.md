# paper_v2/questions.md — questions opened by the cancellation reframe

Numbering continues `../record/questions.md` (Q1–Q6 live there). This
file holds the paper-facing questions the reframe surfaced; each entry
states its evaluation unit and its decision gate against the revision
timeline (ICLR deadline ≈ 2026-10; ~2 months of runway from
2026-08-09).

## Q7 — Is the axis field's σ-rotation law-like, or only smooth?

**Status: RESOLVED (negative) 2026-08-09 — answered by
[E27](../paper_bench/experiments/e27/), the frozen E-record this entry
required (pre-registered before its instrument existed, same day).
Verdicts: 27.1 PLANE-MIXED, 27.2 LAW-WORSE (SLERP beat the geodesic
at every B target, median Δ = −0.166), 27.3 R LAW-WORSE, 27.4 NULL.
"Rotates smoothly" stays a description; see the resolution note at the
end of this entry.** The registered text below is kept as written;
per-item annotations mark what E27 settled.

### Setup (all measured, committed)

E24 STRUCTURED: at fixed σ the cancellation axis is one direction
across routes/stores/corpus (|cos| 0.87–0.99 across routes, ≈ 1.0
across stores); across σ it rotates smoothly — high neighbor cosines,
0.44 at the most distant gated pair (0.3 ↔ 0.7, 768). The current
state of the art for using this is a **per-σ-bin lookup with no
interpolation principle**: E25.0-2 NO-GAIN retired the one candidate
law tried so far (ĝ-frame planar co-rotation — the rotation matches
ĝ's rotation in *angle magnitude* but transport buys Δ ≈ 0). The one
measured hint that the rotation has a mechanism rather than being
merely smooth: E19.4 — at noise-dominated bins the anti-alignment is
partially RoPE-phase-mediated (PI-aligning phases rotates C,
cos(C, C_pi) 0.50–0.67 on 768, halving the anti-alignment).

### Question

Can R̂(σ) (and/or the B̂/Ĉ axis) be written as a low-parameter family —
fit on a subset of σ bins, predicting the *direction* at held-out
bins? I.e. does "rotates smoothly" upgrade from description to model?

### Evaluation unit (frozen intent — the wrong test is tempting)

**Direction cosines under leave-one-σ-out, never scalar map RMSE.**
The scalar estimand h(B+C) is direction-blind: no (route, σ) gap curve
can confirm or refute a direction law. Score debiased
cos(R̂_pred, R̂_measured) at each held-out bin against two baselines:

- (a) nearest-bin copy (what the E25a lookup implicitly does);
- (b) flanking-bin interpolate-then-normalize (SLERP between
  neighbors).

A law earns a claim **only if it beats (b)** — smoothness alone
already buys (b), so matching it demonstrates nothing.

### Candidate parametrizations (decreasing order of principle)

*Structural correction (2026-08-09, pre-run): candidates 1 and 2 are
**complements, not rivals** — a first-principles derivation of a
direction in 77.7M-dim adapter-gradient space from phase geometry is
not feasible (the map runs through Jᵀ, and E19.3 showed the result is
depth- and type-uniform, i.e. delocalized); the feasible quantitative
version of candidate 1 was always "candidate 2 supplies the plane,
the phase mechanism supplies the rate/generator". E27 tested exactly
that decomposition.*

1. **RoPE-phase-driven**: make E19.4 quantitative — predict the C
   rotation from the cross-grid phase-density change weighted by σ's
   noise level. Most principled; the only candidate that would carry
   mechanism content into paper 2. Needs derivation time before any
   fit. **[E27.4 NULL — refuted at its foothold.** V_phase =
   (C − C_pi)⊥ extracted from e194's stored π arms is the most
   reproducible direction in the read (rel 0.83–0.97) and yet lies
   almost entirely *outside* the axis-field plane (in-plane share
   0.01–0.09), aligned with neither the rotation generator nor the
   axis (|cos| ≤ 0.39 everywhere readable). The phase mechanism is a
   within-bin effect whose direction is unrelated to the σ-rotation —
   no derivation can rescue a generator that points elsewhere.]
2. **Fixed-plane geodesic**: E24's Gram top-2 eigenplane holds
   0.92–0.93 of the leg energy per condition. First check (cheap Gram
   re-read): is the top-2 plane itself σ-stable? If yes, a
   one-parameter geodesic θ(σ) in that fixed plane is a 2–3 parameter
   law. NB this is a *different* claim from the refuted ĝ-frame
   planarity — planar in its own eigenplane vs co-rotating with ĝ.
   **[E27.1 PLANE-MIXED + E27.2 LAW-WORSE — refuted at the
   2-parameter rung.** The σ-stability check came back negative the
   honest (LOO) way: held-out in-plane share median 0.769 (B), min
   0.336, collapsing at σ = 0.9625 — the pooled 0.92 included the fit
   conditions' own overfit. The geodesic lost to SLERP at all 12 B
   targets under both pre-named θ(σ) variants, and the fitted rate is
   fold-unstable (not affine in σ or ln σ). The field is *locally*
   2-D — each bin sits in its neighbors' span (SLERP median 0.96) —
   but the plane itself rolls with σ.]
3. **Lookup + SLERP** — this is baseline (b), the engineering floor,
   not a law. **[E27: not just the floor — the measured best at this
   resolution, now with numbers: median held-out |cos| 0.961 (B),
   0.962 (C), 0.855 (R̂).]**

### Data reality

Surviving direction-bearing stores (E24's three, one adapter): σ ∈
{0.3, 0.4333, 0.5667, 0.7, 0.8333, 0.9625, 1.0} on 768/896 — **7
bins, ~5 interior LOO targets**. The E14 15-bin grid kept no
`arm_sums/` (scalar ledger only), so the CPU read is capped at this
resolution. If the read is promising and needs density, a σ-dense
single-route `--keep_arm_sums` probe is the one GPU upgrade
(~0.57 GPU-h per condition, E26 pricing) — its own amendment, not
part of the CPU read.

*Correction (2026-08-09): this section under-counted the on-disk
assets — the e194 store also preserved the E19.4 **π arms**
(`768pi`/`896pi` + half-sets, ignored by E24/E25), which is what let
E27.4 extract V_phase as a pooled direction for zero GPU. Also
recorded for any future density amendment: at 7 bins, LOO plane reads
have limited power against smooth plane-wander (E27's synthetic gates
showed any smooth one-parameter curve reads near-planar at this bin
count) — E27's negative was decisive only because the measured shares
fell far below that bar.*

### Dead ends already adjudicated (do not re-enter)

- ĝ-frame transport / planar co-rotation: E25.0-2 NO-GAIN.
- Any objective-side use of a derived ledger term: E20.4 closed at
  estimand level. A rotation law is descriptive/lookup-side only.
- Per-sample anything: E22 → E23a gate unchanged.
- PI-RoPE as a *training* lever: G11 (off-manifold with content).
  PI as a *probe* of the phase mechanism is fine — that is E19.4.

### Payoffs and decision gate

- **This revision (gated)**: if a frozen read lands PASS before
  revision_plan §7 step 5 (geometry section drafting), the geometry
  section may carry one claim + one panel ("the rotation follows
  <law> at cos ≥ X vs SLERP baseline Y"), under the same scope rungs
  as the axis field. If the verdict is not in by step 5, the question
  transfers to paper 2 **intact — no placeholder language in the
  manuscript** ("rotates smoothly" stays a description).
  **[Executed 2026-08-09: the verdict landed *before* step 5 and is
  negative — same manuscript consequence as the timeout branch, but
  as a measured claim, not a gap: no law claim, no panel; "rotates
  smoothly" stays, and MAY now cite E27 as adjudicated (description
  is the verdict, not a placeholder).]**
- **E25a**: a passing law upgrades the lookup from measure-every-bin
  to interpolate-by-law, and may cover the 768/σ = 0.4333 reliability
  hole from its neighbors. **[E27.3: no upgrade — measure-every-bin +
  neighbor SLERP stands, quantified at median |cos| 0.855 for held-out
  R̂. Hole coverage: only the 896 read survives at that fold
  (SLERP 0.855) — plausible but route-crossed; don't over-claim.]**
- **Paper 2**: the mechanism bridge (why Jᵀ-born, why phase-mediated)
  — this question is its opening move either way. **[E27.4 sharpens
  the opening move: the cancellation axis and the phase-mediated
  component are *separate, mutually near-orthogonal directions* —
  the bridge must carry two objects, not one.]**

### Related non-question (recorded so it isn't re-proposed)

ρ(σ)-transfer in the scalar lane (upgrading the E20 canc form's
frozen ρ̄ to a fit-route-measured ρ(σ) curve) is *possible* and
CPU-cheap, but E20's diagnosis places the canc lane's failure in the
**amplitude law**, not in ρ — expected gain ≈ nil. Not pursued; if
ever run, it is an E20 lane amendment, not part of Q7.

### Resolution (2026-08-09 — [E27](../paper_bench/experiments/e27/))

**Q7 answers NO at this data resolution.** Both principled candidates
fell in one pre-registered CPU read (567 s, zero GPU): the geodesic
family at the 2-parameter rung (no fixed plane; SLERP won every
target), the phase-driven family at its foothold (the phase-borne
component is a reproducible direction *orthogonal* to the axis-field
geometry). New dead ends, recorded so they aren't re-entered:

- **Low-parameter σ-rotation laws on the 7-bin grid**: adjudicated,
  do not refit variants (a third θ(σ) parametrization is post-hoc
  multiplicity; E27's defect note explains why even a true geodesic
  could only have tied SLERP).
- **V_phase as rotation generator**: E27.4 NULL — any future phase
  derivation must first explain why its predicted direction is ⊥ the
  measured axis plane; "weight the phase density by σ" is dead as a
  rotation model.
- Still open (paper 2, not this revision): *what* the rolling plane
  and the off-plane V_phase direction are made of — that is a new
  question, not Q7. **Its strongest surviving form is now registered
  as [E28](../paper_bench/experiments/e28/)** (2026-08-09,
  run-deferred): the σ-rotation as the σ-indexed *conditioning frame*
  — B/C as pull-backs of a comparatively σ-fixed object through the
  σ-conditioned (adaln) network — discriminated by a
  frozen-conditioning probe (σ_cond pinned at 0.7 while the noising σ
  sweeps; CONDITIONING-CARRIED / STATISTICS-CARRIED thresholds frozen
  from the committed E24 table before the instrument flag exists).
  NB the *analysis-frame* version of "projected onto a rotating
  space" (ĝ-frame projection artifact) is already dead — E25.0-2.

What survives for the manuscript: the E25a lookup shape is settled
(per-σ-bin, neighbor-interpolated, quality now quantified), and the
geometry section's "rotates smoothly" sentence is an adjudicated
description backed by E27's SLERP numbers rather than an unprobed
observation.
