# roadmap — live follow-ups after the paper_v2 reframe

Created 2026-08-10. Successor to the frozen `../record/roadmap.md`
(record/ is provenance-only since 2026-08-01 and no longer maintained);
this file is the line's live forward-looking doc and lives here because
paper_v2/ is where the line actually works now. **Nothing in this file
is licensed to enter the paper_v2 tex** — `revision_plan.md` §7/§8
govern the revision; every runnable item below still needs its own
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
  undecided.
- **E25**: E25a frozen-restricted-only (σ = 0.7, per-bin, no span
  projector), E25b (explicit resolution conditioning) still a sketch;
  both need the E20.4-adjacency paragraph to freeze.
- **Environment wall**: cross-boot vector reads dead at 0.32–0.47
  (`project_crossboot_arm_store_break`); within-boot 0.96–1.02; scalars
  drift only Δρ ≈ 0.03–0.05. §4 below is the durable-use answer.
- Related-work additions (ANT / DeMe / PMA) are already planned into the
  revision: `revision_plan.md` §10 last subsection; the deep-research
  brief for the searcher is `question.md`.
- **arm_sums reclamation executed 2026-08-10** — see §3 "Storage
  economics"; §1(c)'s inputs are now the committed
  `e26_grid_across_sigma.json` + `e28_read768.json` tables, not the
  stores.

## 1. Dependency spine

Only item (a) touches paper 1. Everything else is paper-2/paper-3
material in dependency order:

**(a) E26 grid verdict → scope sentence** (`revision_plan.md` §7 step
7). **Verdict LANDED 2026-08-10: flat REPLICATES, dirty REPLICATES.**
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
σ = 0.7, recorded). What remains of (a) is a **tex edit only** —
§7 step 7's scope sentence takes the REPLICATES wording (three
adapters of this base at 768 across the window; same-base qualifier;
896 an open cell) and waits on the tex steps before it.

**(b) E28-F1 — two-block objectification** (paper-2 opener, needs its
own registration): (i) replace the eyeballed pair-sign read with an
interval-clustering criterion — ANT-style contiguity-constrained
clustering (Go et al. 2023's DP formulation; trivial at 5 bins) on the
frozen-frame R̂ |cos| table, k ∈ {1, 2, 3}, cost-gap threshold frozen
before the read. (ii) The discriminator: **relocate σ_cond** (0.4333,
the half-window value the E28 registration itself names as the
amendment). Boundary follows the pin ⇒ the organizing variable is
sign(σ_noise − σ_cond) (intervention-induced mismatch structure);
boundary stays ⇒ intrinsic task blocks (the ANT/DeMe reading). One
extra frozen-conditioning run + twin ≈ 7–8 GPU-h at e28 prices.

**(c) Native-block test** (zero GPU, CPU-only on committed tables —
`e26_grid_across_sigma.json` carries the across-σ B̂/Ĉ/R̂ tables for
flat/dirty/twin; the frozen-frame tables are in `e28_read768.json`):
the same clustering instrument on the **native** twin R̂ table (k = 1
vs 2) and on the E26 grid tables. If the native field hides latent
block structure, (b)'s intrinsic branch gains a prior; if native is
clean k = 1 (smooth rotation, as E24/E27 describe), the two-block object
is frozen-frame-only and (b)'s mismatch branch is favored.

**(d) E25 freeze decisions**, gated on (b)/(c):
- blocks intrinsic → **2-anchor lookup** becomes the cheap E25a
  parameterization candidate (one direction per block + boundary,
  replacing per-bin interpolation forced by E25.0-2 NO-GAIN);
- E25b (resolution conditioning into adaln) freezes independently but
  its rationale paragraph should cite whichever of (b)'s branches
  lands — both are "conditioning input organizes the field" facts.
  Tier 1.5: bench + invariant test, CMMD non-regression,
  wall-clock-at-fixed-steps pitch.

## 2. Application shelf (paper-3 candidates; outlook only, no scope)

- **A. σ-gated feature-forecast certification.** Port the line's
  certification template (substitute cheap compute → measure the
  *induced deviation* below sample level → per-σ gate) to inference
  feature caching/forecasting (TaylorSeer / TeaCache / MagCache
  family), whose gates today are tuned thresholds validated only at
  sample level. The exact B/C object does **not** port (no residual at
  inference); the seam analog is downstream-Jacobian pull-through of
  the substituted-feature error. Inherited testable prediction from
  (b): if blocks are intrinsic, forecast error spikes when the cache
  horizon crosses the block boundary ⇒ σ-gated cache horizons with a
  measured gate. Cheapest probe: per-σ feature-drift profile along
  Anima sampling trajectories (block-output cos / Taylor residual vs
  step gap) — CPU-to-1-GPU-h class.
- **B. Gradient-certified micro-conditioning.** If E25b lands, the
  generalization is "conditioning-injection absorbs a compute-graph
  substitution, certified at the gradient level": resolution-aware
  conditioning for mixed-resolution inference (the learnable
  counterpart of PMA's training-free phase fix), and a cache-age /
  step-gap embedding as the forecast-error absorber for A.
- **C. 2-anchor lookup** (see 1(d)) — an efficiency consequence, only
  if blocks are intrinsic.

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

## 4. Not-doing list (so this file doesn't re-open closed doors)

- No PI/RoPE revival (G11; PMA's mechanism *explains* the closed
  verdict, it does not reopen it — the training path has no
  mixed-scale attention).
- No per-sample estimands (E22 → 22.4 → E23a gate unchanged).
- No derived-account language (E20.4), no rotation-law language (E27),
  no mechanism-bridge language in paper 1 (revision_plan §8 —
  adjudicated).
- Feature forecasting inside ordinary LoRA training: N/A (training
  steps have no trajectory adjacency); the only in-repo trajectory
  loops are distillation (turbo/RSD), where the view/ckpt recompute
  hazard makes seam substitution unattractive.
