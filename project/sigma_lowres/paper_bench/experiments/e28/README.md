# E28 — frozen-conditioning axis probe: is the σ-rotation the conditioning frame, or the noise statistics?

| | |
|---|---|
| **Status** | **PRE-REGISTERED 2026-08-09 — RUN DEFERRED to paper 2.** Thresholds frozen below from committed numbers only; the instrument flag (`--cond_sigma`) does **not** exist yet and no GPU is spent by this registration. Running requires an explicit go decision (≈ 5.7 GPU-h full / ≈ 2.9 GPU-h staged) — this record exists so the strongest surviving mechanism hypothesis after [E27](../e27/) does not evaporate, and so its reading rules predate any instrument or run. |
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

## Cost ladder (planned)

| item | GPU | note |
|---|---|---|
| flag + diff review | none | one argument swap at one call site |
| 768 stage | ≈ 2.9 h | 5 bins × 1 route, arm sums ≈ 9 GB (e193 actuals scaled) |
| 896 completion | ≈ 2.9 h | gated on validation gate 2 passing at the 768 stage |
| CPU read | ~10 min | E24/E27 machinery verbatim |
