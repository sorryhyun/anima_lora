# roadmap — live follow-ups after the paper_v2 reframe

Created 2026-08-10. Successor to the frozen `../record/roadmap.md`
(record/ is provenance-only since 2026-08-01 and no longer maintained);
this file is the line's live forward-looking doc and lives here because
paper_v2/ is where the line actually works now. **Nothing in this file
is licensed to enter the paper_v2 tex** — `revision_plan.md`
(condensed 2026-08-11 to its resolved state; its §2 standing
constraints govern any future tex edit, verbatim plan at commit
`c1534b00`); every runnable item below still needs its own
pre-registration under `../paper_bench/experiments/` before GPU or
verdict. This doc only orders candidates and records rationale *before*
those registrations exist.

## 0. State this file starts from (2026-08-10)

- **E26 grid**: DONE + read — **flat REPLICATES, dirty REPLICATES**
  (5/5 bins readable and passing on both; frozen criteria applied by
  `e26_grid_read.py`, tables in `e26_grid_read.json` + the E26 README
  "Full-grid results"). Stores were `20260810-1114-e26-grid-flat` /
  `20260810-1519-e26-grid-dirty` (dirty's first submission crashed
  6.5 min in — root disk 100 % full → SIGBUS on store write; partial
  store deleted, resubmitted bit-identical argv in the same boot, so
  same-environment status vs the twin reference was preserved;
  deviation recorded in the E26 README), since reclaimed per §3.
- **E28 768-only**: 28-A READABLE, 28-B **PARTIAL** — two
  internally-coherent σ-blocks under the pinned frame, boundary between
  0.5667 and 0.7 (= at σ_cond); cancellation survives shallower. 896
  undecided. **E28-F1 (2026-08-11) resolved the block structure as
  MISMATCH-CARRIED — see §1(b).**
- **E25**: E25a frozen-restricted-only (σ = 0.7, per-bin, no span
  projector), E25b (explicit resolution conditioning) still a sketch;
  both need the E20.4-adjacency paragraph to freeze.
- **Environment wall**: cross-boot vector reads dead at 0.32–0.47
  (`project_crossboot_arm_store_break`); within-boot 0.96–1.02; scalars
  drift only Δρ ≈ 0.03–0.05. §4 below is the durable-use answer.
- Related-work additions (ANT / DeMe / PMA): **landed in the tex
  2026-08-11** (new "Noise levels as tasks" paragraph + PMA delta
  sentence, framing citations only per the no-mechanism guard); the
  deep-research brief for the searcher is `question.md`.
- **arm_sums reclamation executed 2026-08-10** — see §3 "Storage
  economics"; §1(c)'s inputs are now the committed
  `e26_grid_across_sigma.json` + `e28_read768.json` tables, not the
  stores.

## 1. Dependency spine

Only item (a) touches paper 1. Everything else is paper-2/paper-3
material in dependency order. **Live remainder after E28-F1
(2026-08-11)**: (a) is a tex edit only; (b)/(c) are resolved; (d) is
E25a-per-bin + E25b freeze decisions (2-anchor dead); (e) is the E30.1
expression gate (go required) → conditional E31.

**(a) E26 grid verdict → scope sentence** (revision_plan §7 step 7
of the frozen plan, commit `c1534b00`). **Verdict LANDED 2026-08-10:
flat REPLICATES, dirty REPLICATES.**
The stated prior (recorded here before any grid store was read:
base-carriage ⇒ shape-similar per-bin scalar profiles, depth ordering
with leg size; adapter-dependent σ-structure would count against it)
**held**: ρ(σ) deepens toward high σ on all three adapters with the
pointwise ordering sincos < flat < dirty, all bins readable, and no
adapter-dependent holes/blocks appeared — consistent with
base-carried organization / adapter-borne amplitude, still
descriptive (no mechanism run spent). The depth-scaling upgrade
stays unclaimed: the identity-consistency column lands in the
arithmetic branch at 9/10 adapter-bins (sole exception dirty
σ = 0.7, recorded). **(a) CLOSED 2026-08-11**: the scope sentence
landed in the tex with the full reframe execution (steps 3–9 of the
frozen plan all executed — see revision_plan §1 for the
realization record; the E26 rows are `tab:e26` in the appendix, the
one-operating-point limitation paragraph carries the REPLICATES
wording with the same-base qualifier and the 896 open cell).

**(b) E28-F1 — two-block objectification** — **RESOLVED 2026-08-11
([e28f1](../paper_bench/experiments/e28f1/)): F1-i TWO-BLOCK-FORMAL,
F1-B MISMATCH-CARRIED.** (i) The formal re-read (E29 instrument
verbatim) confirms the e28 object: all three legs k = 2 at
{0.3, 0.4333, 0.5667} | {0.7, 0.8333}. (ii) The discriminator
(σ_cond relocated to 0.4333, frozen-cond run + same-boot native twin,
7.6 GPU-h actual): **the boundary followed the pin** — R̂ k = 2 at
{0.3, 0.4333} | {0.5667, 0.7, 0.8333}, gap 0.404 ≫ τ. The organizing
variable is **sign(σ_noise − σ_cond)**: the two-block object is an
intervention-induced property of the frozen-frame probe, not latent
task blocks — E29's prior held. Matched-pair |Δcos| vs twin 0.216
(≈ e28's 0.217): same rotation-disruption magnitude at both pins.
Deviations recorded in the e28f1 README: gate 4 amended to
B-leg-primary (frozen C threshold was mis-calibrated — relC ≈ 0.55
at D = 12 ⇒ ±0.1 noise; e28's own gate C passed by overshoot), and
gate 5 reproduced the committed twin table at max |Δ| = 0.0
(family-B kernel path bit-stable). Paper-2 consequence: the
frozen-frame split is a conditioning-mismatch signature; no further
pin values without a new registration.

**(c) Native-block test** (zero GPU, CPU-only on committed tables —
`e26_grid_across_sigma.json` carries the across-σ B̂/Ĉ/R̂ tables for
flat/dirty/twin; the frozen-frame tables are in `e28_read768.json`):
the same clustering instrument on the **native** twin R̂ table (k = 1
vs 2) and on the E26 grid tables. If the native field hides latent
block structure, (b)'s intrinsic branch gains a prior; if native is
clean k = 1 (smooth rotation, as E24/E27 describe), the two-block object
is frozen-frame-only and (b)'s mismatch branch is favored.
**DONE 2026-08-10 — [E29](../paper_bench/experiments/e29/): 29-1
NATIVE-SMOOTH (twin R̂ k = 1, gap 0.125 ≪ τ = 0.30), 29-2 consistent
(flat/dirty R̂ k = 1; all nine native tables k = 1, largest gap = the
calibration anchor 0.199).** Consequence: the two-block object is
frozen-frame-only ⇒ (b)'s **mismatch branch favored** as prior; (d)'s
2-anchor lookup (intrinsic-branch-conditional) is disfavored. (b)'s
registration should freeze its clustering read on E29's instrument
(`e29_cluster.py`, τ = 0.30) rather than re-deriving one.

**(d) E25 freeze decisions**, gated on (b)/(c) — **(b)/(c) both landed
mismatch/smooth, so the 2-anchor lookup is DEAD** (it required the
intrinsic branch); E25a stays on per-bin parameterization:
- ~~blocks intrinsic → **2-anchor lookup** becomes the cheap E25a
  parameterization candidate (one direction per block + boundary,
  replacing per-bin interpolation forced by E25.0-2 NO-GAIN)~~;
- ~~E25b (resolution conditioning into adaln) freezes independently but
  its rationale paragraph should cite whichever of (b)'s branches
  lands — both are "conditioning input organizes the field" facts.
  Tier 1.5: bench + invariant test, CMMD non-regression,
  wall-clock-at-fixed-steps pitch.~~ **FROZEN 2026-08-11 →
  [e25b](../paper_bench/experiments/e25b/)** citing F1's
  MISMATCH-CARRIED as the rationale fact; Stage 1 (~10 GPU-h) needs an
  explicit go and by frozen submission order waits for E30.1's 30-B
  (context row, not a gate); the Tier 1.5 packaging moved to its
  Stage-2 amendment.

**(e) Base-carriage instrument line (E30 → E31)** — the frame-free
answer to E26's 26.0-2 open question ("property of the model vs of the
adapter"), registered 2026-08-10 as
[E30](../paper_bench/experiments/e30/): base-weight space is the one
shared frame across adapters. **E30.0 DONE (prior-only): rotation
profiles congruent across all three adapters** (B̂ median matched-pair
|Δcos| 0.03–0.05; frame-safe scalars, values were already public so no
verdict weight). E30.1 (≈ 4–5 GPU-h, go required) = the expression
gate: dual-frame accumulation (param + adaln-slice + full-base
count-sketch, k = 2¹⁸ fixed hash seed) in one sincos run, all reads
within-run. EXPRESSES ⇒ E31 (cross-adapter matched-σ axis verdict
with ĝ ceiling + negative-control floor, same-boot same-instrument
family) becomes registrable; NOT-EXPRESSED ⇒ the universal-lookup
chain dies and the limitation paragraph closes. Payoff if shared:
"cancellation axis is a property of the base" (paper-2 headline
candidate) + E25a's lookup collapses to once-per-base.

## 2. Application shelf (paper-3 candidates; outlook only, no scope)

- **A. σ-gated feature-forecast certification.** Port the line's
  certification template (substitute cheap compute → measure the
  *induced deviation* below sample level → per-σ gate) to inference
  feature caching/forecasting (TaylorSeer / TeaCache / MagCache
  family), whose gates today are tuned thresholds validated only at
  sample level. The exact B/C object does **not** port (no residual at
  inference); the seam analog is downstream-Jacobian pull-through of
  the substituted-feature error. ~~Inherited testable prediction from
  (b): if blocks are intrinsic, forecast error spikes when the cache
  horizon crosses the block boundary ⇒ σ-gated cache horizons with a
  measured gate.~~ **Dead with (b)'s resolution** (F1-B
  MISMATCH-CARRIED — no intrinsic blocks, no boundary for a cache
  horizon to cross). The cheapest probe stands on its own merits
  without the inherited prediction: per-σ feature-drift profile along
  Anima sampling trajectories (block-output cos / Taylor residual vs
  step gap) — CPU-to-1-GPU-h class.
- **B. Gradient-certified micro-conditioning.** If E25b lands, the
  generalization is "conditioning-injection absorbs a compute-graph
  substitution, certified at the gradient level": resolution-aware
  conditioning for mixed-resolution inference (the learnable
  counterpart of PMA's training-free phase fix), and a cache-age /
  step-gap embedding as the forecast-error absorber for A.
- ~~**C. 2-anchor lookup** (see 1(d)) — an efficiency consequence, only
  if blocks are intrinsic.~~ **DEAD 2026-08-11** — F1-B landed
  MISMATCH-CARRIED, the intrinsic branch it required does not exist.
- **D. Quant reference perturbations + learned gap certifier**
  (proposal: [`quant_probe_gap_predictor.md`](quant_probe_gap_predictor.md),
  DRAFT 2026-08-13). Fake-quant probes as controlled reference
  perturbations — Q-data/Q-graph direction fingerprints (a_e
  universality; floor decomposed into generic-fidelity vs
  grid-specific), a bits-dial amplitude response (empirical Jacobian
  gain), δ-from-dtype (parameter-free spectral rival) — feeding a small
  coefficient-level predictor of h-unit gap scalars that certifies
  routing windows for new setups without an arm campaign. Successor to
  the E25 lever line's harness (the lever itself closed at 25e);
  offline-certifier-only per the E20.4 guard; scalar targets only per
  the T0 wall; generalization scope gated on (e)'s E31 outcome. Every
  stage needs its own registration before GPU.

## 3. arm_sums — durable-use plan (the environment wall, §0)

What the 2026-08-10 break actually killed, stated precisely: **only
cross-boot-family *direction* comparisons.** Two uses of arm_sums
stores remain fully alive and are the reason to keep paying their disk
cost:

1. **Re-analysis**: any new CPU-side estimator can be re-run on any old
   store forever (E24/E25.0/E27 were exactly this on e193/e194).
   Scalars (ρ, G, h, S/F/I) are boot-portable within Δρ ≈ 0.03–0.05.
2. **Within-family vector reads**: stores from the same boot remain
   mutually vector-comparable (e193↔e194 at 0.999; the e28 pair; the
   E26 grid + twin family).

The plan, in tiers:

- **T0 — boot-family bookkeeping (IMPLEMENTED 2026-08-10).** Every
  store manifest carries a **boot fingerprint**, stamped inside
  `ArmSumAccumulator.finalize` (no entry point can skip it): boot
  epoch from `/proc/stat` btime, driver, torch/CUDA, and the
  inductor/Triton cache dirs (T2 forward-compat). `bench/_common.py::
  boot_fingerprint()` is the helper; every `result.json` `env` block
  now carries boot_epoch + driver too. Reader side:
  `vector_ledger.py::assert_same_family(*stores)` blocks cross-family
  vector reads (unfingerprinted stores never match;
  `allow_cross_boot=True` for deliberately-cross-boot estimands like
  e28_gate2_diag). All ten existing manifests were backfilled from
  `journalctl --list-boots`, which **corrected the family table**:
  **A = boot 2026-08-06 18:25** (e193, e194 — both inside it),
  **B = boot 2026-08-09 19:32** (e28 pair, g2diag, e26 grid + twin).
  The e221 smoke (boot 2026-08-08 12:34) and the e260 smokes (boot
  2026-08-09 11:14, which died at 19:12 — **not** family B as this
  file previously claimed) sit in their own dead singleton families;
  their committed reads were within-pair (same boot), so nothing
  already read is invalidated, but no future vector read may pair
  them with B. Any cross-family estimand must bundle a same-boot
  native reference (the E28 amendment pattern) or be seed-twin by
  design (the E26 pattern). This paragraph is the single home of the
  rule; experiment docs may point here instead of restating it.
- **T1 — environment canary (cheap, makes membership measurable).**
  Freeze a standardized mini-probe (single bin σ = 0.7 / 768, small
  image subset, small D — minutes of GPU) run at store-creation time;
  its pooled B̂/Ĉ against the family's committed canary certifies
  membership at the ≥ 0.95 band instead of trusting uptime
  bookkeeping. One canary spec + one committed canary per family.
  Needs a small registration (spec freeze) before first use.
- **T2 — environment pinning (experimental; would make stores
  boot-portable going forward).** Persist the kernel-path caches
  across reboots: point `TORCHINDUCTOR_CACHE_DIR` / Triton cache at a
  persistent path instead of `/tmp`, plus deterministic algo-selection
  flags where applicable. **Not assumed to work** — the break's
  mechanism is localized to the reboot pair + fresh `/tmp` re-autotune
  but not proven reducible to it. Certification test (pre-register
  before relying on it): deliberately reboot with pinned caches, rerun
  the g2diag-style single-bin native probe, PASS iff B/C/ĝ vs the
  pre-reboot twin land in the same-boot ≥ 0.95 band. PASS ⇒ new
  stores are cross-boot comparable (fingerprint then keys on the
  cache-dir hash instead of boot epoch); FAIL ⇒ T0/T1 stay the
  permanent law and fp64-accumulation forwards remain the only
  unexplored (expensive) candidate.
- **Storage economics — EXECUTED 2026-08-10 (user decision).** All
  raw arm_sums except the g2diag canary candidate were reclaimed
  (~86 GB; disk 96 % → 71 %). Before deletion, every vector table the
  pending analyses need was committed: the grid family's across-σ
  B̂/Ĉ/R̂ tables (flat, dirty, e28 native twin) landed in
  `../paper_bench/experiments/e26/e26_grid_across_sigma.json`
  (E24 estimand verbatim; twin B-leg cross-checked PASS against
  `e28_read768.json`; no verdict applied — §1(c) input only). The e28
  frozen store's tables were already in `e28_read768.json`; family A
  (e193/e194) and the smokes had no pending vector consumer. Store
  `manifest.json`s retained as provenance; scalars/ledgers survive in
  committed JSONs. **Kept**: `20260810-0214-e28-g2diag-native07`
  (2.4 GB) — family B's only T1 canary candidate. Consequences: §3
  use (1) (open-ended re-analysis on old stores) is forfeited for
  reclaimed stores — any new estimator must be applied at read time
  going forward; the standing policy is **stores live until their
  registered read commits its tables, then raw sums are reclaimed**.
  T0 stays (fingerprints license the twin reads while stores are
  alive); the T1 canary becomes the only cross-family bridge.
  **Same day**: stores were centralized under
  `../paper_bench/arm_sums/<run-name>/` (one root for the T0/T1/
  reclamation lifecycle; per-run `bench/results/<run>/arm_sums` paths
  are now symlinks, `run_sigma_probe.py` writes there directly —
  `../paper_bench/arm_sums/README.md` is the store-root contract).
- **e28f1 pair (2026-08-11)**: the two F1-ii stores
  (`20260810-2301-e28f1-cond0433-768` + `20260811-0247-e28f1-native-twin-768`,
  ~24 GB, boot family B) have their tables committed in
  `e28f1_read.json` and are **eligible for reclamation** under the
  standing policy — not yet reclaimed.

## 4. Not-doing list (so this file doesn't re-open closed doors)

- No PI/RoPE revival (G11; PMA's mechanism *explains* the closed
  verdict, it does not reopen it — the training path has no
  mixed-scale attention).
- No per-sample estimands (E22 → 22.4 → E23a gate unchanged).
- No derived-account language (E20.4), no rotation-law language (E27),
  no mechanism-bridge language in paper 1 (revision_plan §2 guards —
  adjudicated).
- Feature forecasting inside ordinary LoRA training: N/A (training
  steps have no trajectory adjacency); the only in-repo trajectory
  loops are distillation (turbo/RSD), where the view/ckpt recompute
  hazard makes seam substitution unattractive.
