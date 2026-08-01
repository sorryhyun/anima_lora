# E14 — low-σ vector ledger: decomposing the 896 bump

| | |
|---|---|
| **Status** | **PROPOSED 2026-08-01** — reserved as ledger arms on E13's probe-matched **e13b** rerun (one process, per the kernel-path rule; command below). Numbering note: this record takes the E14 slot; the prior E14 (MLMC pricing) is now [E15](../e15/), the prior E15 (placement) is now [E16](../e16/). |
| **Question** | The measured 1024→896 curve carries a low-σ plateau (≈+0.20 over σ≈0.05–0.30 on E13's grid) that the account's prediction misses by ~4× — the one panel of the head-to-head figure where ours does not beat the spectral account (`runs/20260801-1118-fig-accounts-e13`). Three explanations produce the same scalar signature: (a) a genuinely large data term the A(ratio) governor starves, (b) negative B/C interference shaping the σ≈0.4 cliff, (c) attenuation-correction inflation at mid-σ. Which is it? |
| **Depends on** | [E9](../e9/) (the B/C instrument, `vector_ledger.py`, the unit-honesty rule, the amplitude-matching localization it validated on 768 — but its grid was σ ∈ [0.5, 1.0]: it stops exactly where the bump begins); [E13](../e13/) (segmented `--sigma_window`, `e1b_probe_list.json`, the 2/24-overlap scope rule that forbids a standalone ledger run on a fresh probe set); [E5](../e5/) (the governor under test; the per-run G normalization of A) |
| **Instrument** | `bench/run_sigma_probe.py --repromote --keep_arm_sums --self_floor` on E13's segmented grid; analysis `../../vector_ledger.py` (shared with E9/E10) |
| **In the paper** | the 896 panel of Fig. 1 (`accounts_headtohead`); §4.6's reduction-domain paragraph (does the 768 failure have a sibling?); the A(ratio) governor's standing (§4.7); potentially the 1120 ratio-twin claim (§4.5) |

## Why the scalar curve cannot answer this

`gap(σ)` is one number per bin; causes (a)–(c) all read as "elevated
plateau". The ledger factors it: **B = ḡ_rp − ḡ₀** (data branch, native
graph — a same-grid contrast, so finite-draw bias cancels to first
order), **C = ḡ_dem − ḡ_rp** (graph branch, demoted data), with
cross-set-debiased S/F/I, ρ = cos(B⊥, C⊥), |B⊥|/|C⊥|, and the exact
counterfactual angles h(B), h(C), h(B+C) (= the realized gap, by
construction). Each cause lands in a different column:

| cause | signature at σ ∈ [0.05, 0.45] |
|---|---|
| (a) data term, governor starved | h(B) ≈ measured plateau; I ≈ 0; C⊥ small — the route's true loading, no A-transport, no per-run-G normalization |
| (b) interference cliff | \|C⊥\|/\|B⊥\| crosses ~1 at the cliff and I swings negative there — E9's validated localization signature (768: crossing 0.98 at σ=0.688 = window center) |
| (c) estimator inflation | h(B+C) from arm means sits well below the scalar debiased plateau; the per-bin excess is the correction layer |
| (d) low-σ graph share (assumption-iii failure) | F_896(σ) elevated below σ≈0.5 — the branch-(ii) alternative, untested at low σ; E9 already found F strongly σ-dependent in-window |

## Pre-registered decision rules

Frozen before the run. The outcomes are not exclusive; shares are read
per bin, magnitudes **only** via h(·) (E9's unit-honesty rule: at
plateau magnitudes κ ≈ 0.7–0.9, S/F/I is out of its truncation domain —
sign, decomposition, and localization only).

- **H-a** (data term): h(B) ≥ 0.7 × measured plateau in the clean
  segment (σ ≤ 0.17, where E13's control is inside ±0.07) with I within
  ±0.02 there ⇒ the two-term *form* is fine and the A(ratio) governor is
  starving 896 ⇒ fires **phase 2** (below): the 1120 twin must be
  re-measured in vector units before the ratio governor is used again.
- **H-b** (interference cliff): |B⊥|/|C⊥| crosses 1 within one bin of
  the measured cliff (σ≈0.4–0.5) and I < 0 beyond it ⇒ the 896 shoulder
  is the 768 failure's sibling ⇒ §4.6's domain paragraph widens to both
  routes; the figure's 896 panel gets the same "reduction's domain"
  annotation as 768.
- **H-c** (estimator): h(B+C) < 0.6 × the scalar debiased plateau at
  matched bins ⇒ the plateau is partly correction inflation ⇒ per E13's
  kill-switch protocol, that σ-region publishes raw-paired as primary
  with the debiased read as the appendix row.
- **H-d** (graph share): F_896(σ ≤ 0.45) > 2 × F_896(1.0) resolved by
  the cross-set read ⇒ assumption (iii) fails outside the endpoint's
  protection ⇒ the "σ-independent floor" claim gets an explicit low-σ
  domain bound.
- **Kill switch** (inherited from E13): if the reenc control's unpaired
  debiased gap exceeds ±0.15 anywhere in the dense low segment, the
  affected bins are excluded from H-a/H-c reads.

## Design — ride the e13b rerun, don't run standalone

A standalone ledger run on a fresh probe set would recreate exactly the
cross-run level-comparison trap E13 documented (2/24 overlap). Instead
the reservation is: **e13b (the probe-matched rerun E13 already owes)
gains `--repromote --keep_arm_sums`**, so one process on E1b's 40-image
list yields (i) the probe-matched curve refit E13 owes §4.7, and
(ii) this ledger, on the same images, same G, same kernel path.

```bash
make daemon-run ARGS="project/sigma_lowres/bench/run_sigma_probe.py \
  --adapter output/ckpt/anima_soup_sincos.safetensors \
  --sigma_window '0,0.1,4 : 0.1,0.9,6 : 0.9,1.0,4' \
  --draws_per_bin 12 --endpoint_bin --self_floor --deterministic \
  --repromote --keep_arm_sums \
  --demote_edges 896,768,512 \
  --probe_list project/sigma_lowres/paper_bench/experiments/e13/e1b_probe_list.json \
  --results_root project/sigma_lowres/paper_bench/runs \
  --label e13b-probematched-ledger --queue"
```

**Cost.** e13b alone ≈ 12 h (E13's calibration, 7.3 h × 40/24). The
repromote arms are native-grid forwards; E9's calibrated overhead for
`--repromote --keep_arm_sums --deterministic` was ~1.5× ⇒ **~18 h**
total. Cheaper variant if the queue is tight: `--repromote` on 896 only
(the bump route) ≈ +20% instead of +50% — but the cross-route ledger is
what lets H-b use E9's universality result, so all three routes is the
recommendation.

**Smoke first.** `--repromote` (E9-era) and the segmented
`--sigma_window` (E13-era) have never run together; a `--smoke` pass
must confirm they compose (the window only builds the σ tensor, so they
should, but the arm-key seed-budget check in `cli.py` is the thing to
watch).

**Storage.** Arm-sum vectors at 15 bins × ~9 arms × 40 images — same
order as E9's store × ~4. Stays under the gitignored `bench/results/`
(vector stores never ship in `runs/`).

## Phase 2 (conditional on H-a): the twin, in vector units

If the plateau is a real data term, the ratio governor's defense — the
1120 twin — must be re-measured on the same instrument: the E9-style
probe on 1280-tier natives (route 1280→1120, matched bins). If
h(B)_1120 ≪ h(B)_896 at matched bins in one unit system, "ratio sets the
amplitude" is falsified in debiased vector units and §4.5/§4.7 are
rewritten; if they agree, the published governor stands and the E1b-era
fit's failure was plumbing (weights + grid), not physics. Separate run
(~4–6 h); not scheduled until H-a fires.

## What lands in the paper

- H-a ⇒ governor asterisk in §4.7 + phase-2 twin run before any
  A(ratio) reuse; the 896 panel keeps the measured curve with the
  starved prediction explained.
- H-b ⇒ §4.6 domain paragraph covers 896's shoulder; one mechanism, two
  routes — coherence, not damage.
- H-c ⇒ raw-paired primary for the affected bins (E13 protocol).
- H-d ⇒ explicit domain bound on assumption (iii).
- Either way: the e13b half of the run discharges E13's owed §4.7
  refit, so this reservation supersedes the plain e13b command in E13's
  record.
