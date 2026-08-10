# E30 — base-frame expression gate (does the cancellation geometry exist in the shared base-weight frame?)

| | |
|---|---|
| **Status** | **PRE-REGISTERED 2026-08-10.** E30.0 (zero-GPU prior tier) may run immediately; **E30.1 spends GPU (≈ 4–5 h estimate) and requires an explicit go decision** — the accumulation instrument does not exist yet and no GPU is spent by this registration. Paper-2 material (`paper_v2/roadmap.md` — the line's forward doc; nothing here enters the paper-1 revision, per revision_plan §8). |
| **Question** | E26's 26.0-2 left the base-carriage question open at the *direction* level: cross-adapter raw cosines (0.27–0.35) are indistinguishable from the frame baseline (native ĝ cross-adapter 0.37–0.39) because gradients w.r.t. different adapters' parameters live in non-overlapping frames. The frame-free instrument is **base-weight space**: ∂L/∂W_base is one shared frame for every adapter of this base. Before any cross-adapter verdict can be designed (future E31), two facts must be measured at the sincos operating point: **(L1) does the cancellation geometry (deep ρ, I < 0, h-ordering, reliable legs) express in the base frame at all**, and **(L2) does the adaln-relevant weight slice carry it** (the tractable slice a universal lookup would use). If L1 fails, the geometry is intrinsically adapter-frame-tied and the universal-lookup payoff chain dies — itself a citable limitation closure. |
| **Licensed by** | E26 REPLICATES (the geometry exists on three adapters ⇒ the base-carriage question is well-posed); E29 NATIVE-SMOOTH (the native smooth field is the right comparison object — no hidden block confound); E21 (adaln carries 86–87 % of phase-response amplitude — the slice prior); E28 768-read PARTIAL + E29's mismatch-branch prior (conditioning pathway organizes the field). **Not gated on E28-F1**: the full-frame sketch hedges the slice choice, so this registration is robust to either F1 branch; the interpretation paragraph will cite F1 when it lands. |
| **Explicitly NOT this** | No cross-adapter read (that is E31, which needs its own registration, same-boot-family design per T0, and this gate passed first). No lookup construction, no training lever, nothing per-sample (E22 → 22.4 → E23a), nothing objective-side (E20.4), no paper-1 content. E30.0 carries **no verdict weight** (see its tier note). |
| **In the paper** | Nothing in this revision. Paper 2: the opening instrument of the base-carriage question (the limitation paragraph E26 stated open). |

## E30.0 — rotation-profile congruence (zero GPU, prior tier, **descriptive only**)

**Honesty note (why this tier cannot bear verdict weight):** its inputs
(the across-σ R̂/B̂/Ĉ tables for twin/flat/dirty) were fully published in
`../e29/e29_read.json` before this registration existed, so no threshold
can be frozen "before the read." The tier is recorded as a **prior**
for spending E30.1's GPU, not as a verdict.

- **Frame-safety rationale**: each adapter's across-σ cosine is a
  within-frame scalar (an angle in its own param space), so comparing
  *profiles* across adapters needs no frame alignment — unlike the
  direction comparison E26 could not make.
- **Read**: matched-pair Δcos between adapter pairs (twin↔flat,
  twin↔dirty, flat↔dirty) on the R̂ tables (B̂/Ĉ alongside), from
  `../e26/e26_grid_across_sigma.json`. **Truncation-cell rule**: any
  pair where either adapter's |cos| > 1 (debias truncation domain —
  twin R̂ has three such cells) is excluded from medians and listed.
- **Use**: congruent profiles (flat↔dirty especially — the two clean
  tables) support the base-carried-rotation prior going into E30.1;
  incongruence would have argued against spending the GPU. Instrument
  `e30_congruence.py` → `e30_congruence.json`.

## E30.1 — expression gate + sketch (one sincos run; **go required**)

### Instrument (to be built; diff reviewed against this section before submission)

One `run_sigma_probe.py` run accumulating, in the **same backward
pass** as the standard adapter-param arm sums:

1. **adaln-slice base gradient sums** — `requires_grad` enabled on the
   adaln/modulation weight tensors (exact tensor-name filter fixed at
   instrument time by a frozen name-pattern rule; the resolved list is
   recorded in the store manifest and amended into this README before
   submission). Stored **as sketches only** (below), norms streamed
   exactly.
2. **Full-base + complement sketches** — a fixed-seed **count-sketch**
   (feature hashing, k = 2¹⁸ buckets, ±1 signs; hash seed recorded in
   the manifest — the same seed defines the shared frame for any future
   E31 run) applied streaming per-tensor via gradient hooks
   (sketch-and-free; no full-model grad buffer ever materialized).
   Three sketch families stored per arm: full base, adaln slice,
   complement. Sketch storage is MB-scale; the store stays ordinary
   param-sums-sized.
3. Standard param-frame arm sums, unchanged (they gate the run and
   self-calibrate the sketch).

Count-sketch preserves inner products in expectation (all ledger reads
are Gram-based, so the E24 machinery applies verbatim on sketched
vectors); its accuracy is not assumed — it is **measured in-run** by
gate 2. **Exact slice vectors are not stored** (recorded forfeit: a
future exact-subset consumer — e.g. lookup construction — needs a
rerun).

**Kernel-path caveat (recorded up front)**: enabling `requires_grad` on
base tensors and adding hooks changes the autograd graph, so this run's
vectors may sit in their own kernel-path family even within the boot
(`project_crossboot_arm_store_break` logic). All E30.1 comparisons are
**within-run** by design; nothing here is compared vector-wise to any
other store. E31 must plan its own same-instrument, same-boot family.

### Protocol

e193-style, deltas only: `anima_soup_sincos` (E19.6 license), route
**768 only**, σ window `0.2333…,0.76667…,4 : 0.76667…,0.9,1` (bins
{0.3, 0.4333, 0.5667, 0.7, 0.8333} — the E26/E28-twin grid), 40-stem
e1b probe list, 12 draws/bin, `--repromote --keep_arm_sums
--self_floor --deterministic --seed 42`, fp32 sums, daemon-queued.
Cost estimate ≈ 4–5 GPU-h (768 5-bin baseline 3.3–3.8 h + wgrad
materialization for hooked tensors + sketch scatter; VRAM headroom to
be smoke-checked on 1 bin before the full submission).

### Validation gates (before any 30-A/30-B row is read)

1. **Param-frame ledger sanity**: the run's own adapter-param ledger
   passes the frozen E26 per-bin criteria (readable: rel_B ∧ rel_C ≥
   0.5; passing: I < 0, ρ ≤ −0.5, h(B+C) < min(h(B), h(C))) on ≥ 4/5
   bins — the geometry expresses normally in param frame despite the
   instrument (absolute criteria; no cross-run vector read).
2. **Sketch self-calibration**: sketched-vs-exact cosine on the
   param-frame vectors (where exact sums exist in-run), over all
   ledger-used pairs: max |Δcos| ≤ **0.02** (expected ≈ 0.002–0.005 at
   k = 2¹⁸; the tolerance is 4–10× slack). Fails ⇒ sketches unreadable,
   stop.
3. **Instrument diff review** against the seam note above (one
   accumulation site; the probe's noising/target/arm-selection paths
   untouched).

### Pre-registered readings

**30-A — expression gate (full-base sketch; the L1 verdict).** Apply
the gate-1 criteria to the **base-frame** ledger (legs ⊥ against the
base-frame ĝ of the same condition, machinery verbatim):

| outcome (5 bins, 768) | verdict |
|---|---|
| ≥ 4/5 bins readable AND every readable bin passes | **EXPRESSES** — the cancellation geometry lives in the shared frame; E31 (cross-adapter verdict: ĝ ceiling / negative-control floor / matched-σ axis read) is licensed for registration. |
| < half the bins readable | **NOT-EXPRESSED** — the geometry is adapter-frame-tied at this operating point; the universal-lookup chain dies; recorded as the limitation-paragraph closure. |
| in between | **PARTIAL** — surviving bins recorded; E31, if ever registered, restricted to them. |

**30-B — slice sufficiency (the L2 read; descriptive-to-gating).** Over
30-A-readable bins: (i) adaln-slice norm share s = ‖leg_adaln‖²/‖leg_full‖²
per leg per bin (streamed exact norms); (ii) across-σ rotation profile
of the slice vs the full sketch (frame-safe scalars, E29-instrument
classification alongside). **SUFFICIENT** iff median s ≥ 0.5 (both
legs) AND median matched-pair |Δcos| (slice vs full profile) ≤ 0.1;
else **INSUFFICIENT** (an E31 would then use the full sketch as its
frame and the slice claim is dropped — design constants 0.5/0.1 are
judgment values, recorded as such at freeze).

**30-C — descriptive rows (no verdict weight)**: base-frame ρ(σ) vs
this run's param-frame ρ(σ); base-frame across-σ profile vs param-frame
profile and vs the committed twin profile (scalar comparison only —
Δρ-band caveat applies cross-boot); complement-slice shares; E29 k\*
classification of the base-frame R̂ profile.

### Kill switches / honesty

- Registration spends zero GPU; 30.1 runs only on explicit go, smoke
  1-bin first for VRAM. If thresholds go stale before the run (e.g.
  instrument redesign), re-freeze via amendment — post-hoc
  renegotiation is not allowed.
- All 30.1 reads are within-run; no cross-store vector claim of any
  kind. Cross-adapter language stays out of every 30-A/30-B sentence —
  EXPRESSES licenses *registering* E31, nothing more.
- Pooled directions at one operating point; per-sample and
  objective-side family exclusions unchanged.
- Outputs (this dir): `e30_congruence.{py,json}` (30.0),
  `e30_read.{py,json}` (30.1, after go), instrument diff in the flag
  commit.
