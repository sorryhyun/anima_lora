# E28-F1 — two-block objectification: formal re-read + the σ_cond-relocation discriminator

| | |
|---|---|
| **Status** | **PRE-REGISTERED 2026-08-10 (roadmap (b), user go same day).** Thresholds frozen below before any new store exists; the clustering instrument is [E29](../e29/)'s, frozen there (τ = 0.30) and reused verbatim per roadmap §1(c)'s consequence — nothing re-derived here. Two parts: **F1-i** (zero GPU — formal clustering re-read of the committed E28 frozen-frame tables; outcome already public via E29 gate 3, so this row carries *formalization* weight only, no new information) and **F1-ii** (the discriminator — one frozen-conditioning run at the relocated pin + a same-boot native seed-twin, ≈ 7 GPU-h at e28 actuals). **F1-i RESOLVED same day: TWO-BLOCK-FORMAL** — all three legs (R̂ gap12 0.396 / B̂ 0.576 / Ĉ 0.500) select k\* = 2 at {0.3, 0.4333, 0.5667} \| {0.7, 0.8333}, E29 gates 1–5 PASS (`e28f1_reread.json`); the eyeballed read is replaced and F1-ii cleared to submit. **F1-ii SUBMITTED same day, boot family B** (same boot as the e28 pair/E26 grid — epoch 1786271541): daemon jobs `20260810-230028-9081ae` (cond0433) → `20260810-230033-e2b9f5` (native twin), back-to-back per the frozen order. NB the `--sigma_window` segmented spec is **one shell token** (spaces + `:` inside a single quoted argument) — the first submission attempt split it into three tokens and died at argparse (jobs `-225935`/`-225945`, error state, no GPU spent). |
| **Question** | E28's 768-read (PARTIAL) named a surviving structure by eye: under a pinned conditioning frame (σ_cond = 0.7) the axis field splits into two internally-coherent σ-blocks with the boundary between 0.5667 and 0.7 — exactly at the pin. Two hypotheses own that shape: **mismatch** — the organizing variable is sign(σ_noise − σ_cond), an intervention-induced structure that will *follow the pin* when σ_cond relocates; **intrinsic** — the blocks are task blocks in the ANT/DeMe sense (they were always there; the pin merely unmasked them) and the boundary *stays* at 0.5667 \| 0.7 regardless of pin position. [E29](../e29/) set the prior: all nine native tables are clean k = 1 (NATIVE-SMOOTH), so the two-block object is frozen-frame-only and the **mismatch branch is favored** — prior only; this registration owns the verdict, and the read below is prediction-symmetric. |
| **Explicitly NOT this** | Not a mechanism-account claim (E20.4 wording guard unchanged; no "derived" language). Not a rotation law (E27 closed). Not a σ_cond sweep — exactly one relocation value, the one E28's own registration named as the amendment; a sweep is a further amendment with its own multiplicity accounting. Not per-sample (E22 → 22.4 → E23a unchanged), not per-block/per-type slicing, nothing in paper 1 (revision_plan §8). 896 stays out of scope (the E28 896 cell remains undecided and is not touched here). |
| **Depends on** | [E28](../e28/) (the frozen-frame object + `--cond_sigma` seam, commit `7f30101f`, gate-3 review recorded there); [E29](../e29/) (`e29_cluster.py` — the frozen instrument: signed-cos distance, pooled-mean cost, sequential elbow τ = 0.30, lexicographic tie-break; its gates rerun here); e28 committed tables (`../e28/e28_read768.json`); E24 machinery (`build_cond`/`cross_cos`/`bc_ledger`); T0 boot-family law (`paper_v2/roadmap.md` §3). |
| **Instruments** | F1-i: `e28f1_reread.py` → `e28f1_reread.json` (imports `e29_cluster`, CPU seconds). F1-ii runs: `run_sigma_probe.py` with the e28 argv verbatim except `--cond_sigma 0.43333333333333335` and labels (`e28f1-cond0433-768`, `e28f1-native-twin-768`); read: `e28f1_read.py` — E24/e28-read machinery verbatim on the new pair + the E29 classifier. |
| **In the paper** | Nothing in this revision (revision_plan §8). Paper 2: this is the (b) opener — the object the mechanism bridge gets to talk about, whichever branch lands. |

## Frozen protocol (F1-ii runs)

- Same probe protocol as the e28 768 stage bit-for-bit: E14 40-image
  probe list, seed 42, 12 draws/bin, `--repromote --keep_arm_sums
  --self_floor --deterministic`, operating point `anima_soup_sincos`
  (E19.6 license), route 768 only, segmented σ window
  `0.23333333333333334,0.7666666666666667,4 : 0.7666666666666667,0.9,1`
  (centers {0.3, 0.4333, 0.5667, 0.7, 0.8333}).
- **σ_cond = 0.43333333333333335**, frozen — the exact nominal center
  of bin 1 (the "half-window" value E28's registration named and
  roadmap (b)(ii) fixed at 0.4333). One relocation value; no sweep.
- **Native seed-twin in the same boot**, identical argv minus
  `--cond_sigma` (the E28 amendment pattern, mandatory from the start
  here — the environment wall makes the committed twin's *vectors*
  unusable and the old stores are reclaimed). If a reboot intervenes
  before both stores land, the incomplete pair is discarded and both
  runs resubmit in one boot; no cross-family vector read is licensed
  (T0).
- Submission order: frozen-conditioning run first, twin second,
  back-to-back on the daemon queue.
- Analysis frame: all verdict reads are **within-run** (cluster the
  frozen-cond store's own across-σ table) or **within-pair same-boot**
  (gates, matched-σ rows) — nothing compares against family-A/B
  vectors.

## Validation gates (all must pass before the F1-B verdict row is read)

1. **Instrument gates**: `e29_cluster.py`'s five gates rerun clean
   (synthetic planted-boundary + smooth control + committed anchors).
2. **Store integrity**: within-condition scalar reproduction through
   the independent `bc_ledger` path on all 5 bins of both new stores
   (the e28 gate-1 convention).
3. **Same-family**: `vector_ledger.assert_same_family` passes on the
   new pair (T0 fingerprints; no `allow_cross_boot`).
4. **σ_cond-bin consistency (the e28 gate-2 analog, relocated)**: at
   noising bin s0.4333 the frozen-conditioning run is
   protocol-near-identical to native ⇒ its B̂/Ĉ vs the twin must land
   `|cos| ≥ 0.95` on both legs (ĝ reported). Fails ⇒ instrument break;
   read nothing else.
5. **Twin reproduction (context gate, family-B-conditional)**: the new
   twin's across-σ B̂ pair table vs the committed
   `e28_read768.twin_across_sigma_B` — if the new pair is still boot
   family B, require max per-pair |Δ| ≤ 0.05; if a reboot forced a new
   family, this downgrades to a recorded context row (the wall), not a
   gate.

## Pre-registered readings

**F1-i — formal re-read (zero GPU, runs at freeze).** Apply the E29
classifier to the committed E28 frozen-frame tables
(`c_rows.R_across_sigma_frozen` primary; `frozen_across_sigma_B`/`_C`
descriptive). Expected and already public via E29 gate 3: R̂ k\* = 2 at
{0.3, 0.4333, 0.5667} | {0.7, 0.8333}. This row **replaces the
eyeballed pair-sign read** as the formal statement of the two-block
object; if it somehow fails to reproduce, F1-ii does not submit until
the discrepancy is resolved.

**F1-A — readability gate (precedence first, 28-A verbatim at the new
pin).** If > half of the 4 off-σ_cond conditions fail the rel gate
(either leg, 0.5), verdict **INCONCLUSIVE-OFF-MANIFOLD**: record which
bins survive, read nothing else as verdict-bearing. (Max σ-distance
from the pin is 0.4 — the same as e28's, so 28-A's READABLE outcome is
the expectation, not a guarantee.)

**F1-B — the discriminator.** E29 classifier (signed cos, τ = 0.30,
k ∈ {1, 2, 3}) on the **frozen-conditioning(0.4333) store's own across-σ
R̂ table** (all 10 pairs required, e28 `r_cos` construction; |cos|
variant reported descriptively). The four possible k = 2 cuts are
distinct, so the branches cannot overlap:

| outcome (R̂ table, new pin) | verdict |
|---|---|
| k\* = 2 with the cut adjacent to the pin bin — `s0.3 \| s0.4333` or `s0.4333 \| s0.5667` | **MISMATCH-CARRIED** — the boundary follows the pin; the organizing variable is sign(σ_noise − σ_cond), an intervention-induced structure. The two-block object is a property of the frozen-frame *probe*, not of the field; paper-2 wording treats it as a conditioning-mismatch signature. (Sharper e28-analog sub-prediction, descriptive only: the pin bin joins the **upper** block ⇒ cut `s0.3 \| s0.4333`.) |
| k\* = 2 with the cut at `s0.5667 \| s0.7` | **INTRINSIC-BLOCKS** — the boundary stays where E28 found it despite the pin moving 2 bins: latent task blocks (ANT/DeMe reading) that the pin unmasks. Roadmap (d)'s 2-anchor lookup for E25a is re-licensed as a candidate (it was disfavored on E29's prior). |
| k\* = 1 | **NOT-REPLICATED** — no block structure at the relocated pin; the e28 two-block object was specific to σ_cond = 0.7. Neither branch gains; any follow-up names this pin-dependence as its object and registers separately. |
| k\* = 3, or k\* = 2 at `s0.7 \| s0.8333` | **OTHER-STRUCTURE** — record the partition + margins; no branch verdict; follow-up requires its own registration. |

Precedence: gates → F1-A → F1-B. No other row carries verdict weight.

**F1-C — descriptive rows (no verdict weight).** (i) B̂ and Ĉ
frozen-frame cluster classifications at the new pin (same instrument).
(ii) 28-C(ii) analog: per-bin matched-σ cos(B̂_frozen, B̂_twin) — does
the "pin replaces distant-σ geometry, near-no-op at own bin" shape
replicate at 0.4333. (iii) Matched-pair |Δcos| of the frozen B̂ table
vs the twin B̂ table (the 28-B STATISTICS-CARRIED estimand, reported
for continuity). (iv) ρ(σ) + rel legs per bin, both stores — does the
cancellation again survive shallower. (v) New-twin vs committed-twin
across-σ table deltas (gate 5's numbers, kept as the family-B
replication record).

## Kill switches / honesty

- Pooled directions at one operating point; probe intervention only —
  no training lever, nothing objective-side.
- One relocation value; the multiplicity of "two pin values total"
  (0.7 from e28, 0.4333 here) is inherent to the discriminator design
  and is not extended without a new registration.
- F1-i's outcome is already public (E29 gate 3 used the same table +
  instrument as a gate); it is registered as formalization, not
  discovery, and cannot be cited as an independent replication.
- E29's mismatch prior is recorded in the Question row; the F1-B table
  is symmetric and the prior confers no threshold advantage.
- Storage policy per roadmap §3: the two new stores (~24 GB) live
  until `e28f1_read.py` commits its tables into `e28f1_read.json`,
  then raw arm_sums are reclaimed (manifests retained).
- If the thresholds here go stale before submission (reboot mid-pair,
  store loss), the runs resubmit; the thresholds themselves are not
  renegotiable post-hoc.

## Cost ladder (planned)

| item | GPU | note |
|---|---|---|
| F1-i formal re-read | none | CPU seconds on committed JSON |
| frozen-cond run (pin 0.4333) | ≈ 3.3 h | e28 768-stage actual |
| native seed-twin | ≈ 3.8 h | e28 twin actual |
| CPU read | ~min | `e28f1_read.py` |
