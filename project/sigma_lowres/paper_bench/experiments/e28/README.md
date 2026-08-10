# E28 — frozen-conditioning axis probe: is the σ-rotation the conditioning frame, or the noise statistics?

| | |
|---|---|
| **Status** | **PRE-REGISTERED 2026-08-09 — RUN DEFERRED to paper 2.** Thresholds frozen below from committed numbers only; the instrument flag (`--cond_sigma`) does **not** exist yet and no GPU is spent by this registration. Running requires an explicit go decision (≈ 5.7 GPU-h full / ≈ 2.9 GPU-h staged) — this record exists so the strongest surviving mechanism hypothesis after [E27](../e27/) does not evaporate, and so its reading rules predate any instrument or run. **GO 2026-08-09** (explicit user decision, same day): `--cond_sigma` landed as the one-argument swap the seam note demands (validation gate 3 review in the flag commit); 768 stage submitted per the frozen staging. σ grid realized as the segmented window `0.2333…,0.7667…,4 : 0.7667…,0.9,1` — centers {0.3, 0.4333, 0.5667, 0.7, 0.8333}, the 0.8333 bin's draw values verified bit-identical to e194's construction. **768 stage DONE 2026-08-10 — gate 2 FAIL as frozen, diagnosed as a cross-environment instrument break (NOT the flag: a same-boot native diag matches the frozen run at 0.96–0.99); see the 768-stage record below.** **Native-twin amendment executed same day (user go): `20260810-0658-e28-native-twin-768` (3.8 h, same boot verified) — gate 2 re-evaluated cross-set against it: PASS (B +0.9502 / C +1.0872 / ĝ +0.9668 at σ = 0.7; B margin razor-thin, recorded). 28-C(ii) matched-σ rows (descriptive): away from σ_cond the frozen field is near-orthogonal to native (B/C 0.02–0.11 at σ ≤ 0.5667; 0.68–0.73 at 0.8333) while staying internally reliable — the pin replaces distant-σ geometry rather than perturbing it. 28-B not yet read (staging order); 896 decision pending.** |
| **Question** | E27 closed Q7's law candidates: no fixed plane, no 2-parameter rate, and the phase-borne component V_phase is a reproducible direction *orthogonal* to the whole axis-field plane. The surviving "strong version" (recorded in Q7's resolution): **B and C are pull-backs of a comparatively σ-fixed upstream object through the σ-conditioned network — the "rotating space" is the σ-indexed conditioning (adaln) frame, not anything positional.** Three committed facts point this way rather than merely permitting it: E21 (adaln carries 86–87 % of the phase-response amplitude), E19.3 (the anti-alignment is depth- and type-uniform — a global frame inherited through the chain rule), E27.4 (the rotation is not carried by RoPE/positional geometry). Discriminator: **freeze the conditioning** — pin the σ fed to the DiT forward at σ_cond while the *noising* σ sweeps the grid — and ask whether the axis field stops rotating. |
| **Explicitly NOT this** | The ĝ-frame projection story ("the axis rotates because the anchor it is perp'd against rotates") — that is E25.0-2's estimand, adjudicated NO-GAIN; E28 intervenes on the network's conditioning input, not on the analysis frame. Also not a PI/RoPE revival (RoPE and grid geometry are untouched — G11 irrelevant here), and not a training lever: conditioning-freeze is a *probe* intervention, exactly as PI-align was for the phase mechanism in E19.4. |
| **Depends on** | [E24](../e24/)/[E27](../e27/) machinery + committed native tables (the comparison target); e193 protocol actuals (cost/store sizing); `sigma_probe/kernel.py` (the seam); E19.6 (operating-point invariance licenses running at the same `anima_soup_sincos` operating point). |
| **Instruments** | `run_sigma_probe.py --repromote --keep_arm_sums --cond_sigma 0.7` (flag to be added; see seam note), then the E24/E27 CPU machinery verbatim on the new store. |
| **In the paper** | Nothing in this revision (guard: no rotation-law or mechanism-bridge language, per revision_plan §8). Paper 2: this is the opening probe of the mechanism bridge, replacing the phase-density derivation E27 killed. |

## The seam (recorded so the flag is built at the right joint)

`sigma_probe/kernel.py::estimate` builds `noisy = (1 − σ)x + σε` from
`sview` and calls `bundle.anima(noisy_5d, sigma_b, ca, …)` — `sigma_b`
is the conditioning input (timestep embedding → adaln). The
`--cond_sigma` flag must replace **only the `sigma_b` passed to the
DiT forward** with the pinned value, leaving untouched: the noising
`sview`, the flow-matching target (ε − x, conditioning-independent),
the σ-gated rope handle (`set_sigma` keeps the *noising* σ — RoPE is
explicitly not intervened on), and every arm-selection/seed path.
The diff must be reviewable as "one argument swapped at one call
site"; anything wider re-opens this registration.

## Frozen protocol

- Same probe protocol as e193 bit-for-bit where applicable: the E14
  40-image probe list, same seed, 12 draws/bin, `--repromote
  --keep_arm_sums --self_floor --deterministic`, operating point
  `anima_soup_sincos` (E19.6 license), routes 896 + 768.
- σ (noising) grid = the E24 verdict bins **{0.3, 0.4333, 0.5667,
  0.7, 0.8333}** bit-exactly (endpoint bins excluded — E27 measured
  the endpoint region as the least plane-shared; adding it back is an
  amendment, not a default).
- **σ_cond = 0.7**, frozen: the cross-store-replicated bin (axis
  agreement ≈ 1.0 across e193/e194), interior to the window.
- **Staging, frozen**: 768-only first (5 conditions, ≈ 2.9 GPU-h) —
  if the σ = 0.7 consistency gate (below) fails there, stop; 896
  completes the store only after the gate passes.
- Analysis: E24/E27 machinery verbatim (legs, ⊥ against the
  condition's own ĝ from the *same frozen-conditioning* arms, rel
  gate 0.5, debias conventions, chain orientation). Cross-run
  comparisons against the committed native stores use the E24
  cross-store convention (draw noise independent across runs).

## Validation gates (before any verdict row is read)

1. The E24 instrument-validation suite reruns clean on the new store
   (synthetic + within-condition scalar reproduction).
2. **σ_cond-bin consistency (free control)**: at noising σ = 0.7 the
   frozen-conditioning run is protocol-identical to a native run —
   its B̂/Ĉ must match the committed e193/e194 σ = 0.7 axis at the
   cross-store band (|cos| ≥ 0.95, both legs). Fails ⇒ the flag is
   not the one-argument swap it claims to be; fix before reading
   anything else.
3. The flag's diff is committed and reviewed against the seam note
   before the run is submitted.

## Pre-registered readings

**28-A — readability gate (precedence first).** Frozen-conditioning
at distant noising σ is off-manifold for the network (a (z, σ_cond)
pair it never trained on — the same caveat family as G11's PI
qualifier). If **> half of the off-σ_cond conditions fail the rel
gate** (either leg), the verdict is **INCONCLUSIVE-OFF-MANIFOLD**:
record which bins survive, read nothing else as verdict-bearing, and
note that a conditioning-freeze at smaller σ-distance (σ_cond per
half-window) would be the amendment.

**28-B — the discriminator (B table, gated conditions; C reported).**
Native committed reference, across-σ family: median |cos| 0.791, min
0.442 (E24). Precedence: 28-A → CONDITIONING-CARRIED →
STATISTICS-CARRIED → PARTIAL.

| outcome (frozen-conditioning across-σ pairs) | verdict |
|---|---|
| the family passes the SHARED-AXIS criterion the native field failed: median \|cos\| ≥ 0.7 **and** min \|cos\| ≥ 0.5 | **CONDITIONING-CARRIED** — the σ-rotation is the conditioning frame; the axis field is (approximately) one direction once the frame is pinned. Paper 2's mechanism bridge gets its object; any future law lives in conditioning space, not σ space. |
| per-matched-pair \|Δcos\| vs the native table has median ≤ 0.05 **and** the family still fails SHARED-AXIS (min < 0.5) | **STATISTICS-CARRIED** — the rotation survives a pinned frame; it lives in the noise-level statistics of the input distribution. The conditioning-frame hypothesis dies; the E27-era description stands unchanged. |
| in between | **PARTIAL** — record *which* pairs flatten (prediction under partial conditioning-carriage: flattening concentrated at pairs nearest σ_cond); any downstream use needs a follow-up registration naming the surviving structure. |

**28-C — descriptive rows (no verdict weight).** (i) Does the
*cancellation* survive the frozen frame: ρ(σ) and h-unit magnitudes
vs native per bin — whether the network still magnitude-matches B
with pinned conditioning is mechanism-relevant either way. (ii)
Per-bin cos(B̂_frozen(σ), B̂_native(σ)): how far the frozen field sits
from the native one at matched noising σ. (iii) The R̂ mirror of the
28-B table.

## Kill switches / honesty

- Pooled directions at one operating point; nothing per-sample
  (E22 → 22.4 → E23a unchanged); nothing objective-side (E20.4
  closed); no training-lever claim — probe intervention only.
- This registration spends **zero** GPU; the run is a paper-2
  decision. If the flag lands but thresholds here have gone stale
  (e.g. stores reclaimed), re-verify sources before submitting — the
  thresholds themselves are not renegotiable post-hoc.
- Anti-scope: no σ_cond sweep (one pinned value; a sweep is an
  amendment with its own multiplicity accounting); no endpoint bins;
  no PI arms; no per-block/per-type slicing (that is a *second*
  follow-up if 28-B lands CONDITIONING-CARRIED — E19.3's machinery
  would apply, but it is not licensed here).
- Wording guard for any eventual write-up: CONDITIONING-CARRIED does
  **not** license "derived" language for the account (E20.4) and does
  not touch the E27 verdicts — the σ-space lookup remains the
  shipped read regardless.

## 768-stage record (2026-08-10) — gate 2 FAIL, diagnosed as instrument-environment, run PAUSED pending amendment

Store: `bench/results/20260809-2216-e28-cond07-768/arm_sums` (12 GB,
manifest records `cond_sigma: 0.7`). Instruments (this dir):
`e28_gate2.py` → `e28_gate2.json`, `e28_gate2_diag.py` →
`e28_gate2_diag.json`.

- **Gate 1 PASS**: E24 synthetic suite clean; the store's
  within-condition scalars reproduce through the independent
  `bc_ledger` path on all 5 bins.
- **Gate 2 FAIL as frozen**: vs committed e193/e194 at σ = 0.7/768,
  cos_B ≈ 0.33 / cos_C ≈ 0.43 / ĝ ≈ 0.43 against the ≥ 0.95 band
  (committed native↔native baseline 0.999/1.02/0.997). Both native
  stores agree about where e28 sits. A seed-layout subtlety the
  registration missed (e28 shares the reenc-reference noise
  realization with e193 at exactly the gate bin — same seed, same
  draw offsets 36–47, same arm_idx 2) is corrected for in the gate
  read (shared-ref term +0.0006, immaterial).
- **Diagnosis — the failure is NOT the flag.** A same-boot single-bin
  NATIVE run (`bench/results/20260810-0214-e28-g2diag-native07`, no
  `--cond_sigma`, draws 0–11 ⇒ seed-independent of every counterpart
  arm) separates the causes: **diag ↔ e28 (same boot, conditioning
  on/off): B 0.978 / C 0.991 / ĝ 0.964** — the freeze at its own
  σ_cond bin is nearly a no-op, exactly the gate-2 premise; **diag ↔
  e193/e194 (native ↔ native, across the 2026-08-09 reboots): B 0.32
  / C 0.41 / ĝ 0.47** — the committed stores are not
  vector-comparable to *any* run made in the current environment.
  Environment forensics: adapter/DiT/VAE/latent/TE caches all predate
  e193 unchanged; torch 2.12.0+cu132 and driver 610.43.02 unchanged
  (apt history clean); the only break is the reboot pair
  2026-08-09 19:14/19:32 with a fresh `/tmp` inductor re-autotune.
  The magnitude matches the line's recorded paired-run chaos floor
  (~0.41), and the `--deterministic` help text warned a cold cache
  "can still shift results by ~0.3 at D = 2" — now measured at
  **0.32–0.47 on pooled 40-image D = 12 debiased legs**, i.e. the
  pooled axis-field directions carry a large kernel-path-dependent
  component. Within-path readings are unaffected (e193↔e194 same-boot
  0.999; diag↔e28 same-boot 0.96–0.99; e221/e224 cross-corpus
  transfers in E23.0 spanned the Aug 7→8 boots and worked — that
  cache apparently survived those reboots).
- **Frozen-protocol consequence**: 896 NOT submitted; 28-A/28-B not
  read (the 28-A preview rows inside `e28_gate2.json` are context
  only). The e28 store itself is internally valid (gate 1; and the
  boot is self-consistent by diag↔e28) — nothing needs re-running on
  the frozen-conditioning side.
- **Line-wide caveat (bigger than E28)**: every future cross-run
  vector read against the committed Aug-7 arm stores (e.g. the
  E23-named D ≥ 48 σ = 0.7 top-up for a restricted E25a) inherits
  this wall. Cross-run estimands must bundle a same-environment
  native reference — or use seed-twin arms in one run.
- **Amendment required to proceed (user decision — extra GPU beyond
  the registered ladder)**: re-measure the native comparison table in
  the current environment as a **seed-twin** of the e28 grid (same
  5-bin window, same seed ⇒ shared draw noise cancels in every
  comparison — strictly stronger than the original cross-store
  design): 768 native twin ≈ 2.9 h; then gate 2 re-evaluates against
  it, and the 28-B STATISTICS-CARRIED branch's |Δcos| table uses the
  twin table. 896 completion (+ its native twin) ≈ 5.8 h further if
  the gate passes. The 28-B thresholds themselves are unchanged —
  only the identity of the native reference is amended, forced by the
  measured cross-environment irreproducibility.

## Cost ladder (planned → actual)

| item | GPU | note |
|---|---|---|
| flag + diff review | none | one argument swap at one call site (commit `7f30101f`) |
| 768 stage | ≈ 2.9 h → **3.3 h** | 5 bins × 1 route, arm sums 12 GB fp32 |
| gate-2 diagnosis | **+1.0 h** | same-boot native 0.7-bin run (unplanned; forced by the reboot-pair environment break) |
| 896 completion | ≈ 2.9 h | **blocked** — pending the native-twin amendment above |
| CPU read | ~10 min | E24/E27 machinery verbatim (unrun) |
