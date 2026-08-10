# E29 — native-block test (contiguity clustering on the committed across-σ tables)

| | |
|---|---|
| **Status** | **RESOLVED 2026-08-10 — 29-1 NATIVE-SMOOTH, 29-2 consistent on both adapters** (all five gates PASS before any verdict table was opened; see Results). Pre-registered same day at `1d86f73a` (roadmap §1(c), `paper_v2/roadmap.md`) — zero GPU: CPU-only, seconds, on committed JSON tables (the raw arm_sums are reclaimed — these tables are the only inputs that exist). Thresholds were frozen from committed/public numbers only; the verdict tables (`e26_grid_across_sigma.json` values) were **not read** before the freeze — only the file's key structure was inspected. |
| **Question** | Does the **native** axis field hide latent contiguous block structure across σ, or is it clean k = 1 (smooth rotation, as E24/E27 describe)? E28's 768-only read named a two-σ-block structure **under the pinned conditioning frame** (boundary between 0.5667 and 0.7 = at σ_cond). If the native field shows blocks too, roadmap (b)'s **intrinsic** branch (ANT/DeMe-style task blocks) gains a prior; if native is clean k = 1, the two-block object is **frozen-frame-only** and (b)'s **mismatch** branch (organizing variable = sign(σ_noise − σ_cond)) is favored. |
| **Inputs** | `../e26/e26_grid_across_sigma.json` — across-σ B̂/Ĉ/R̂ pair-cos tables for `flat` / `dirty` / `sincos_twin` (E24 estimand verbatim, R leg per `e28_read768.r_cos`; all family-B same-boot stores, so within-family vector reads are licensed under T0). `../e28/e28_read768.json` — the committed frozen-frame and twin tables used as calibration anchors and gates (public since the 28-B read). |
| **Explicitly NOT this** | Not (b)'s verdict: the σ_cond-relocation discriminator and the formal re-read of the frozen-frame table stay in the E28-F1 registration. Not a mechanism claim — this is a **prior-setting read** for (b); no rotation-law language (E27), no derived-account language (E20.4), nothing per-sample (E22 → 22.4 → E23a unchanged), nothing in paper 1 (revision_plan §8). |
| **In the paper** | Nothing in this revision. Paper 2: shapes the E28-F1 registration's prior only. |

## Frozen instrument

ANT-style contiguity-constrained clustering (Go et al. 2023's DP
formulation; exhaustive at 5 bins) on a symmetric 5×5 across-σ pair-cos
table over the ordered verdict bins {0.3, 0.4333, 0.5667, 0.7, 0.8333}:

- **Distance** d(i,j) = 1 − cos(i,j), **signed** cos (deviation from the
  roadmap sketch's "|cos|" wording, recorded: the sign structure *is* the
  named block signature — E28's cross-block pairs are ≈ 0 or negative —
  and folding to magnitude would erase it; an |cos| variant is reported
  descriptively). Debiased cos may exceed 1 ⇒ d may be slightly
  negative; used as computed, no clamp.
- **cost(P)** for a contiguous partition P = pooled mean of d over all
  within-segment pairs (singleton segments contribute none);
  cost(k) = min over all contiguous partitions into exactly k segments,
  k ∈ {1, 2, 3}. Ties broken by lexicographically-first boundary.
- **Selection (sequential elbow, frozen)**: k* = 1; upgrade to 2 iff
  cost(1) − cost(2) ≥ **τ = 0.30**; iff upgraded, upgrade to 3 iff
  cost(2) − cost(3) ≥ τ. Recorded limitation: the sequential rule can
  miss a 3-block structure whose best 2-split clears < τ; at 5 bins
  with the anchor contrasts below this is a corner case, accepted.
- **Diagnostic (no verdict weight)**: separation margin at the selected
  partition = min within-segment cos − max cross-segment cos.

**τ = 0.30 calibration (committed/public numbers only, computed at
freeze time)**: block anchors — frozen-frame B̂ gap(1→2) 0.576, R̂
0.397, Ĉ 0.500, all recovering the known {0.3, 0.4333, 0.5667} |
{0.7, 0.8333} boundary; smooth anchor — twin B̂ gap(1→2) 0.199. τ is
the midpoint of the two nearest anchors (0.199 / 0.397). Consequence
of using twin B̂ as a calibration anchor: **twin B̂ carries no verdict
weight below** (it is a calibration input); the verdict rests on twin
R̂, unread at this freeze. All anchor gap(2→3) values are 0.05–0.11 —
comfortably below τ, so no anchor selects k = 3.

## Validation gates (all must pass before any unread table is opened)

1. **Planted two-block synthetic** (within +0.6, cross −0.3), all four
   boundary positions: k* = 2 with the planted boundary recovered —
   and stable under every deterministic ±0.05 perturbation pattern of
   the 10 pairs (2¹⁰ sign combinations, no RNG).
2. **Smooth-rotation control**: exponential-decay tables
   cos(i,j) = ρ_adj^|i−j| for ρ_adj ∈ {0.5, 0.7, 0.85, 0.95}: k* = 1,
   stable under the same ±0.05 exhaustive perturbation. (Property
   noted at freeze: pure exponential decay yields gap(1→2) ≈ 0.13
   nearly independent of rate.)
3. **Committed positive control**: frozen-frame R̂ and B̂
   (`e28_read768.json`) → k* = 2, boundary {0.3, 0.4333, 0.5667} |
   {0.7, 0.8333} (verified at freeze; the gate reruns it).
4. **Committed smooth anchor**: twin B̂ (`e28_read768.json`) → k* = 1
   (verified at freeze; the gate reruns it).
5. **Provenance tie**: the e26 file's `sincos_twin.across_sigma_B`
   pairs must match e28's committed `twin_across_sigma_B` to ≤ 1e−3
   per pair (the E26 README recorded this cross-check PASS at 1e−4;
   the gate re-verifies it, tying the unread file to committed
   provenance without reading any new table).

## Pre-registered readings

**29-1 — primary verdict: `sincos_twin.across_sigma_R`** (the native
twin R̂ table — same operating point and boot family as the
frozen-frame run):

| outcome | verdict |
|---|---|
| k* = 1 | **NATIVE-SMOOTH** — the native field hides no block structure at this resolution of the instrument; the two-block object is frozen-frame-only; roadmap (b)'s **mismatch branch** is favored (prior only — (b) still runs its own discriminator). |
| k* ≥ 2 | **NATIVE-BLOCKS** — latent contiguous blocks exist in the native field; (b)'s **intrinsic branch** gains a prior; boundary recorded, and whether it matches the frozen-frame 0.5667 \| 0.7 boundary is noted. (d)'s 2-anchor lookup becomes the candidate E25a parameterization *only if (b) later lands intrinsic* — no license from this read alone. |

**29-2 — replication rows (recorded, no independent verdict weight)**:
`flat.across_sigma_R` and `dirty.across_sigma_R` under the identical
rule — consistent/inconsistent with 29-1 recorded per adapter. The
verdict rests on the twin because the frozen-frame comparison object is
a sincos-operating-point run; flat/dirty test adapter-generality of
whatever 29-1 shows.

**Descriptive rows (no verdict weight)**: B̂ and Ĉ tables for all three
adapters (twin B̂ excluded from any verdict use per the calibration
note); |cos|-variant classifications; separation margins; full
cost(k)/partition tables for every input.

## Results (2026-08-10) — **29-1 NATIVE-SMOOTH · 29-2 consistent (flat, dirty)**

Instrument `e29_cluster.py` → `e29_read.json`. All five gates PASS
before any unread table was opened (gate 1/2 exhaustive ±0.05 stable;
gate 3 recovers k = 2 at the known boundary on frozen-frame R̂ and B̂;
gate 4 twin B̂ k = 1; gate 5 provenance max dev 0.0).

- **29-1: NATIVE-SMOOTH** — twin R̂ selects **k\* = 1** (cost(1) 0.130,
  gap(1→2) 0.125 ≪ τ = 0.30). All 10 pairs positive, range
  0.57–1.38 (several debiased values > 1 — the truncation-domain
  regime; used as computed per the frozen no-clamp rule). The native
  residual direction is *more* σ-coherent than either leg (mean pair
  cos ≈ 0.87 vs twin B̂ cost(1) 0.345) — consistent with E25.0's
  descriptive R̂-axis-field read on family A.
- **29-2: consistent on both adapters** — flat R̂ k\* = 1 (gap 0.104),
  dirty R̂ k\* = 1 (gap 0.111). No adapter shows block structure.
- **Descriptive**: all six B̂/Ĉ tables (three adapters) are k\* = 1,
  gaps 0.13–0.20 — every native gap lands in the smooth-anchor band
  (≤ 0.199, the twin-B̂ calibration value) and none approaches τ; the
  |cos| variants agree everywhere (k\* = 1 on all nine native tables).
  No borderline case: the largest native gap is the calibration anchor
  itself.

**Consequence (per the frozen reading + roadmap §1(c))**: the native
field is clean k = 1 — smooth rotation, as E24/E27 describe — so the
E28 two-σ-block object is **frozen-frame-only**, and roadmap (b)'s
**mismatch branch** (organizing variable = sign(σ_noise − σ_cond),
intervention-induced) is favored going into the E28-F1 registration.
Prior only: (b) still owns the σ_cond-relocation discriminator, and the
(d) 2-anchor-lookup parameterization for E25a is now *disfavored*
(it required the intrinsic branch). Nothing here touches E27's
verdicts, the shipped σ-space lookup, or paper 1.

## Kill switches / honesty

- Read-only CPU analysis of committed JSON tables; nothing refit, no
  constant tuned on verdict data (τ fixed above from public anchors
  before any verdict value was read), no GPU, no store access.
- The frozen-frame tables are re-classified here **only as gates** —
  their formal re-read (replacing the eyeballed pair-sign verdict)
  belongs to the E28-F1 registration (roadmap (b)(i)), which also owns
  the σ_cond-relocation discriminator.
- Outputs (this dir): `e29_cluster.py` (instrument + gates + read),
  `e29_read.json` (the record).
