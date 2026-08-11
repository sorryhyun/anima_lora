# E30 — base-frame expression gate (does the cancellation geometry exist in the shared base-weight frame?)

| | |
|---|---|
| **Status** | **PRE-REGISTERED 2026-08-10; E30.1 RUN + READ 2026-08-11.** 30-A **EXPRESSES** (5/5 bins, gate-1-caveated), 30-B **INSUFFICIENT** (adaln share ~0.37–0.40 < 0.5; profile congruent) — see "30.1 result" below. Gate 1 failed as frozen (registration defect — unattainable at the sincos operating point; instrument-perturbation intent measured satisfied at max\|Δ\| = 0.00115 vs the twin). Actual cost 15.4 GPU-h vs the 4–5 h estimate (deviation recorded). Paper-2 material (`paper_v2/roadmap.md` — the line's forward doc; nothing here enters the paper-1 revision, per revision_plan §8). |
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

**30.0 result (2026-08-10) — profiles congruent; prior supports
spending 30.1's GPU.** B̂ is the cleanest: median matched-pair |Δcos|
0.031–0.047 across all three adapter pairs (full 10 pairs each, no
truncation cells, max 0.17). Ĉ similar (medians 0.049–0.062). R̂:
flat↔dirty 0.085 over all 10 pairs; the twin rows exclude 3
truncation cells (all σ = 0.7-involving, the small-legs sincos
denominators) — twin↔flat 0.023, twin↔dirty 0.130 on the surviving 7.
Reading (prior-only, per the tier note): the across-σ rotation *rate
profile* is shared across the three adapters to roughly the
cross-boot scalar-drift band, consistent with base-carried rotation;
no adapter pair shows a structural profile break.

## E30.1 — expression gate + sketch (one sincos run; **go required**)

### Instrument amendment (2026-08-11, pre-submission — the promised resolved-list + deltas record)

Instrument built as `--base_sketch` on `run_sigma_probe.py`
(`bench/sigma_probe/base_sketch.py`; one `end_bin` seam in
`grad_estimate_binned`, off-path byte-identical; CPU invariant tests in
`tests/test_base_sketch.py`). Deltas vs the sketch-spec below, recorded
before submission:

- **Resolved adaln list (the frozen name-pattern rule
  `(^|\.)adaln_`)**: **114 tensors, 177,733,632 params** — per block
  `adaln_fused_down.1` + `adaln_up_{self_attn,cross_attn,mlp}` (×28)
  + the final layer's `adaln_modulation` pair; full name→numel map in
  `base_sketch/meta.json` of the store (and mirrored into the arm-sums
  manifest). `t_embedder` is deliberately outside the slice (the slice
  is the modulation band E21 measured, not the conditioning trunk).
- **Fourth sketch family `param`**: the exact per-bin LoRA flat vector
  sketched in-run through the same scatter path — gate 2 thereby
  certifies the *streaming accumulation path*, not just the hash
  family (smoke: cos(sketch(exact store), in-run sketch) = 1.000006).
- **Exact adaln upgrade**: the adaln slice is additionally accumulated
  exactly (fp16 CPU slots, ~356 MB × conditions; fp16 = the arm-store
  rounding precedent) and reduced at finalize to an **exact fp64
  cross-condition Gram + norms** (`base_sketch/adaln_exact.npz`), so
  every 30-B slice read is exact rather than sketch-estimated; the
  slice **vectors** are still discarded (the recorded forfeit stands).
  Complement family stored as full − adaln (exact by sketch
  linearity).
- **Hashing**: per-tensor multiplicative hashes from
  `blake2b(f"{seed}:{name}")`, k = 2^18, **seed = 3021** (manifest-
  recorded; any future E31 must reuse it verbatim).
- **VRAM record (16 GB card)**: smoke r1 OOM'd on allocator
  fragmentation (13.07 GiB allocated + 1.49 GiB reserved-unallocated);
  fixed by 2^22 hash chunking + `PYTORCH_CUDA_ALLOC_CONF=
  expandable_segments:True` (set by the flag before CUDA init); smoke
  r2 (2 images × 1 bin × 2 draws, full arm set) ran clean end-to-end —
  8 conditions, uniform hook fires (427 grad-receiving base tensors of
  551 per draw; the 124 no-grad tensors are paths this forward does
  not exercise and contribute exactly zero to every family).

### Instrument (as pre-registered; diff reviewed against this section before submission)

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
  `e30_read.{py,json}` (30.1, landed 2026-08-11), instrument diff in
  the flag commit.

### 30.1 result (2026-08-11) — run, gates, verdicts

Run `20260811-0821-e30-basesketch-768` (job 20260811-082111-7bde1a),
40/40 stems; store at `paper_bench/arm_sums/20260811-0821-e30-basesketch-768/`
with `base_sketch/` (four sketch families + exact adaln fp64
Gram/norms + meta). Param-frame ledger via `vector_ledger.py
--data_ref reenc` (ledger.json in the run dir); all reads in
`e30_read.{py,json}` — every estimand Gram-based (the sketch's design
property), via a coefficient engine over per-family 40×40 condition
Grams.

**Cost deviation recorded**: 15.4 GPU-h actual vs the 4–5 h estimate.
22.5–24.0 min/stem, flat from stem 1, vs 5.9 min/stem on the
un-instrumented E28 twin (~3.9×): the sketch scatter dominates the
step (memory-bound hash + `scatter_add_` — 100 % SM at ~230 W of the
300 W cap, no throttling). The 1-bin smoke certified VRAM and
correctness but was not used to project wall time; **E31 must budget
from the measured ~4× per-stem overhead**, not the registration
estimate.

**Gate 1 — FAILED AS FROZEN (1/5 bins); diagnostic intent measured
satisfied; registration defect recorded.** Only σ = 0.8333 passes —
the h-ordering leg fails elsewhere (param-frame h_B 0.030–0.198,
small sincos legs). The frozen pass condition was, however, never
attainable at this operating point: the un-instrumented sincos twin's
own columns in `../e26/e26_grid_read.json` — published 2026-08-10,
**before** this registration — show the identical 1/5 pattern (E26's
REPLICATES verdicts were flat/dirty; the twin was reference-only and
its criteria columns were never required to pass). A freeze error,
checkable at freeze time, not checked. The gate's stated purpose
("the geometry expresses normally in param frame despite the
instrument") is instead measured directly: this run reproduces the
twin's ledger to **max |Δ| = 0.00115** over all seven scalar columns
× 5 bins — the instrument leaves the param-frame geometry untouched
(the kernel-path caveat did not materialize at scalar level). Per the
no-renegotiation rule the gate is recorded FAILED, not waived; the
30-A/30-B verdicts below carry this caveat.

**Gate 2 — PASS.** Pairwise sketched-vs-exact max |Δcos| = **0.00166**
over all 140 within-bin condition pairs (tol 0.02; inside the
expected 0.002–0.005 band). Streaming-path certification: CPU
re-sketch of the exact arm store vs the in-run param family, min cos
**1.000000** over all 40 conditions. Bonus calibration: sketch/exact
adaln B-leg norm ratio 1.0000–1.0019 across bins (177.7 M-dim
vectors).

**Gate 3** — instrument diff review recorded at the 2026-08-11
amendment above (one `end_bin` seam, off-path byte-identical, CPU
invariant tests).

**30-A — EXPRESSES (gate-1-caveated).** 5/5 bins readable, 5/5 pass
(full-base sketch, bc_ledger machinery, data_ref reenc):

| σ | S | F | I | ρ | rel_B | rel_C | h(B) | h(C) | h(B+C) |
|---|---|---|---|---|---|---|---|---|---|
| 0.3 | +1.271 | +1.368 | −2.470 | −0.936 | 0.947 | 0.804 | 0.412 | 0.645 | 0.278 |
| 0.4333 | +0.237 | +0.259 | −0.457 | −0.923 | 0.850 | 0.541 | 0.184 | 0.340 | 0.169 |
| 0.5667 | +0.130 | +0.165 | −0.270 | −0.918 | 0.890 | 0.696 | 0.116 | 0.204 | 0.077 |
| 0.7 | +0.125 | +0.146 | −0.260 | −0.962 | 0.911 | 0.611 | 0.109 | 0.194 | 0.065 |
| 0.8333 | +0.102 | +0.115 | −0.205 | −0.947 | 0.941 | 0.696 | 0.103 | 0.127 | 0.042 |

**L1 answered: the cancellation geometry lives in the shared
base-weight frame** — deeper there than in param frame at every bin
(ρ −0.918…−0.962 vs −0.836…−0.921), and the h-ordering that fails on
sincos in param frame holds at every base-frame bin: the base legs
are not small (h_B 0.10–0.41), so the small-legs pathology is
param-frame-specific — an artifact of what the LoRA projection
discards, not of the geometry. Subject to the gate-1 caveat, **E31
(cross-adapter verdict) is licensed for registration** per the frozen
table.

**30-B — INSUFFICIENT.** Median s_B = **0.367**, s_C = **0.405**
(< 0.5 both legs; per-bin 0.36–0.42; the perp-sketch variant agrees,
0.36–0.42). Profile congruence alone passes — slice-vs-full R̂ median
|Δcos| 0.064 ≤ 0.1 (B̂ 0.021, Ĉ 0.017) — so the adaln slice
**rotates with** the full field but carries only ~37–40 % of leg
energy: E21's 86–87 % phase-response amplitude does **not** translate
into gradient-energy share. Per the registration: an E31 uses the
full sketch as its frame; the slice claim is dropped. Deviation
recorded: exact FULL norms were not stored (the "streamed exact
norms" cover the adaln slice only), so the share denominator is
sketch-estimated (~0.3 % norm error at k = 2¹⁸ — three orders under
the threshold margin); numerators are exact (adaln fp64 Gram).

**30-C (descriptive, no verdict weight).** Base-frame ρ(σ) deeper
than param-frame at every bin; the adaln and complement sub-frames
both show the same deep-ρ structure (−0.914…−0.972) — the
anti-correlation is not localized to either slice. Complement shares
0.58–0.64 (the two slices' shares sum to ≈ 1 as they must).
Base-vs-param R̂ profile median |Δcos| 0.127 (max 0.31);
base-vs-committed-twin R̂ 0.126 (scalar comparison only — cross-boot
caveat): the base-frame rotation profile is a relative of the
param-frame one, not a copy. E29 classification of the base-frame R̂:
**k\* = 1, single smooth family** (gap12 0.226 < τ = 0.30) —
consistent with E29's NATIVE-SMOOTH read.
