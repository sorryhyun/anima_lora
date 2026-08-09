# E14 — low-σ vector ledger: decomposing the 896 bump (+ the probe-matched refit E13 owes)

| | |
|---|---|
| **Status** | **DONE 2026-08-02** — run `runs/20260801-2304-e14-ledger-probematched` (job `20260801-230338-a2211a`, 40/40 images, 21.7 h, full 15-bin grid fp32; the grid decision resolved full-grid after disk was freed to 84 GB). Ledgers: `ledger.json` (reenc ref, CPU) + `ledger_native.json` (native ref, GPU path — validated 719/720 fields identical vs CPU). **Verdicts below** ([Results](#results-2026-08-02)): kill switch clean; **H-a FAILS** (no phase-2 twin), **H-b FIRES**, H-c fires σ≤0.062 only, **H-d fires substantively**. Headline: the 896 bump is a **two-large-opposing-terms regime** (ρ ≈ −0.7…−0.9 at every σ), not a starved data term. Numbering note: this record takes the E14 slot; the prior E14 (MLMC pricing) is now [E15](../../../record/e15/), the prior E15 (placement) is now [E16](../e16/). **Consolidated 2026-08-01**: the "e13b" probe-matched rerun E13 owed is not a separate submission — this run carries it; E13's record now points here. |
| **Question** | The measured 1024→896 curve carries a low-σ plateau (≈+0.20 over σ≈0.05–0.30 on E13's grid) that the account's prediction misses by ~4× — the one panel of the head-to-head figure where ours does not beat the spectral account (`runs/20260801-1118-fig-accounts-e13`). Three explanations produce the same scalar signature: (a) a genuinely large data term the A(ratio) governor starves, (b) negative B/C interference shaping the σ≈0.4 cliff, (c) attenuation-correction inflation at mid-σ. Which is it? |
| **Depends on** | [E9](../e9/) (the B/C instrument, `vector_ledger.py`, the unit-honesty rule, the amplitude-matching localization it validated on 768 — but its grid was σ ∈ [0.5, 1.0]: it stops exactly where the bump begins); [E13](../e13/) (segmented `--sigma_window`, `e1b_probe_list.json`, the 2/24-overlap scope rule that forbids a standalone ledger run on a fresh probe set, and the owed §4.7 probe-matched refit this run discharges); [E5](../e5/) (the governor under test; the per-run G normalization of A) |
| **Instrument** | `bench/run_sigma_probe.py --repromote --keep_arm_sums --self_floor --deterministic` on E13's segmented grid, E1b's 40-image probe list; analysis `../../vector_ledger.py` (shared with E9/E10). Instrument deltas landed 2026-08-01 (streaming arm retirement — **required**, see below; `--arm_sums_dtype`; `--partitioner_aggressive` now opt-in). |
| **In the paper** | the 896 panel of Fig. 1 (`accounts_headtohead`); §4.6's reduction-domain paragraph (does the 768 failure have a sibling?); the A(ratio) governor's standing (§4.7 — this run's 1024-tier legs are the probe-matched refit); potentially the 1120 ratio-twin claim (§4.5) |

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

## Design — one process, probe-matched

A standalone ledger run on a fresh probe set would recreate exactly the
cross-run level-comparison trap E13 documented (2/24 overlap). Instead
**one process on E1b's 40-image list** yields (i) the probe-matched
curve refit E13 owes §4.7 (same images ⇒ same G ⇒ A comparable, so the
896 F↔A redistribution reads as curve change, not normalization), and
(ii) this ledger, on the same images, same G, same kernel path.

```bash
# NB: submitted via `python -m anima_daemon submit` with `--` because at
# launch time `make daemon-run` stole `--label` for the job display name
# (unlabeled run dir; cost one false start). Fixed same day — daemon-run's
# flags are now scoped to the prefix before the script path, so the
# daemon-run form works too.
uv run python -m anima_daemon submit --label e14-ledger-probematched -- \
  project/sigma_lowres/bench/run_sigma_probe.py \
  --adapter output/ckpt/anima_soup_sincos.safetensors \
  --sigma_window '0,0.1,4 : 0.1,0.9,6 : 0.9,1.0,4' \
  --draws_per_bin 12 --endpoint_bin --self_floor --deterministic \
  --repromote --keep_arm_sums \
  --demote_edges 896,768,512 \
  --probe_list project/sigma_lowres/paper_bench/experiments/e13/e1b_probe_list.json \
  --results_root project/sigma_lowres/paper_bench/runs \
  --label e14-ledger-probematched
```

Config rationale (all measured, runs under
`bench/results/20260801-{1226,1425,1515}`):

- **`--deterministic` stays.** The nondet twin study (T5a/T5b, identical
  args, D=12) put the atomics floor at |Δ| 0.03–0.14 on per-image raw
  gaps over σ≈0.04–0.43 — *the payload bins*, where ‖ḡ‖ bottoms out and
  the same amplification that makes ĉ large magnifies atomics noise —
  versus 0.008 at the endpoint. (The old "|Δcos| ≤ 0.015" nondet number
  was endpoint-only.) Propagated to N=40 bin means that is ~5–12% wider
  SEMs exactly where H-a–H-d read; det costs ~10% wall. Bad trade.
- **No draw batching.** B=2 native OOMs at budgets 0.99/0.95/0.90 (fits
  at 0.85 + `expandable_segments`, but the GPU is already 100%-utilized
  at B=1, so batching was a wash paid for with recompute). Dead lever
  for this arm mix; `--draw_batch_tokens` remains for small-grid-heavy
  configs.
- **Partitioner aggressive recompute OFF** (new probe default;
  `--partitioner_aggressive` restores ≤E13 behavior). Issue-58 bench:
  −2.25 GB for +12.6% s/it — a VRAM trade the B=1 probe (~11.5 GB peak)
  doesn't need. Measured +7% on the probe (T2 vs T5).
- **Budget 0.99, B=1** — the proven envelope; 0.99↔1.0 is ~inert.

## Instrument prep + calibration (2026-08-01)

- **Streaming arm retirement (required).** The probe used to hold every
  arm's per-bin CPU fp32 vectors until image end; at 77.7M adapter params
  (311 MB/vector) the repromote × self-floor × 15-bin arm set is 16
  lists ≈ **75 GB against 46 GB RAM** — the first e13b smoke died to the
  kernel OOM-killer, not the GPU (E13's own 10-list set ≈ 47 GB was
  already riding swap; E9 survived repromote only because its grid was
  5 bins). Arms are now archived to `arm_sums` + freed by the stats
  worker right after their stats job (FIFO ⇒ safe), keeping only a/b +
  in-flight arms resident (~28 GB). `--pool`/`--target_kappa` keep the
  old whole-set path.
- **`--arm_sums_dtype {fp32,fp16}`** (default fp32 — this run ships
  fp32 per 2026-08-01 decision). The store is 16 keys × 15 bins ×
  311 MB ≈ **75 GB fp32**; all 240 memmaps materialize during image 1,
  so a too-full disk fails in the first ~30 min, not at hour 15. Check
  ≥78 GB free before submitting. fp16 (~37 GB, ~1e-3 accumulation
  rounding) is the escape hatch.
- **Main-thread stalls removed**: `flat_grad` now cats on-GPU + one D2H
  copy (was 560 small sync copies per bin); the 4.7 GB/arm store write
  runs on the stats worker, off the GPU-driving thread.
- **Composition validated** (formerly the owed smoke): `--repromote` ×
  segmented `--sigma_window` × `--keep_arm_sums` × `--self_floor` ran
  end-to-end (smoke + 2-image T-runs); the `cli.py` arm-key seed-budget
  check passes with margin (8 blocks × 1000 + 180 draws = 8180 < 10000).
- **Timing (2-image probe-pair runs, 16 arm passes, D=12)**: nondet +
  aggressive-off = 24.8 min/img; twins agree within 0.7%. Det production
  estimate ~27–29 min/img.

## Cost

Measured basis (above), det, full 15-bin grid, N=40: **~19 h**. The
repromote arms are ~6/16 of the passes; dropping them to 896-only would
save ~25% but forfeits the cross-route ledger H-b leans on — not
recommended.

### The one open decision — the dense-high segment (resolved: full grid)

Dropping `0.9,1.0,4` (11 bins + endpoint instead of 15) saves **~27% ⇒
~14 h**. For: E13 settled everything that window existed for (H1
falsified within-run, H3 confirmed, and the refit showed the dense ends
contribute ~nothing to A); no E14 hypothesis reads σ ∈ (0.9, 1); H-d
needs only the endpoint bin, which stays. Against: the grid would no
longer exactly match E13's, adding a second difference axis to the
"movement in A/F = probe-set correction" attribution (blunted by E13's
own finding that the grid barely moves A), and the high-σ raw-flatness
claim stays at N=24. Resolution: the full grid ran (disk freed to
84 GB), keeping the exact E13 grid match.

## Storage

Arm-sum store ≈ 75 GB fp32 under `<run_dir>/arm_sums/`. The run dir
lands in the committable `paper_bench/runs/` tree, but **vector stores
never ship** — this run's `arm_sums/` was analyzed and then deleted
(downstream users, e.g. E21, forward-depend on the digests only).
Scalar deliverables (`result.json`, `per_image.jsonl`) commit as usual.

## Results (2026-08-02)

Run `runs/20260801-2304-e14-ledger-probematched`; scalar curves from
`per_image.jsonl` (N=40, nan-guarded — 2 images produced 3 NaN cells via
self-floor≈0 blowups), vector reads from `ledger.json` (reenc ref) /
`ledger_native.json` (native ref). Kill switch: **not tripped** (dense-low
reenc control ≤ |0.045|, bound ±0.15); clean segment = all five σ ≤ 0.17
bins (|control| ≤ 0.07).

| rule | verdict | read |
|---|---|---|
| **H-a** (data term, governor starved) | **FAILS** | The I ≈ 0 / C⊥-small signature is decisively violated: I = −0.10…−0.33 across the clean segment and \|B⊥\|/\|C⊥\| ≈ 1.0–1.2 (C⊥ is *not* small). Native-ref h(B) alone is ≈ 104–112 % of the plateau, but it is immediately half-cancelled by C — this is not "the route's true loading, no interference". **Phase 2 (1120 twin) is not triggered.** |
| **H-b** (interference cliff) | **FIRES** | \|B⊥\|/\|C⊥\| crosses 1 between σ = 0.433 and 0.567 — within one bin of the measured cliff (scalar debiased sign flip between 0.30 and 0.433) — with I < 0 everywhere beyond it and ρ ≈ −0.88…−0.96 through the shoulder. The 896 panel gets the "reduction's domain" annotation. (The "768 sibling / widen §4.6 to both routes" corollary initially drawn here fell on probe-matched re-read — E19 19.0: no in-window 768 crossing on this grid; the crossing↔window localization holds on 896 only.) |
| **H-c** (estimator inflation) | fires **σ ≤ 0.062 only** | h(B+C) < 0.6 × scalar debiased plateau at 0.013/0.037/0.062 (0.010/0.058/0.132 vs plateaus 0.057/0.149/0.238); NOT at the plateau peak (0.087, 0.167). Practical impact small — raw ≈ debiased in those bins anyway (the excess is largely mean-of-gaps vs gap-of-means estimand mismatch); per E13 protocol the affected bins publish raw-paired primary. |
| **H-d** (graph share) | **FIRES** (substantively) | F_896 is strongly σ-dependent below 0.45: 0.02 → 0.19 (σ=0.087–0.167) → 0.86 (σ=0.30) vs F(1.0) ≈ 0. The registered >2× threshold is trivially met because the endpoint F ≈ 0 — the honest statement is that assumption (iii) fails at low σ and the "σ-independent floor" claim needs an explicit low-σ domain bound. |

**Headline mechanism**: at every σ, B⊥ and C⊥ are strongly anti-aligned
(ρ ≈ −0.7…−0.9, cross-set debiased, reproducible: relB/relC mostly
0.6–0.99; even the huge σ=0.30 bin — S 0.90, F 0.86, I −1.55 — replicates
at relB 0.92/relC 0.72). Both interventions are individually far larger
than the realized gap; the measured curve is the *residual of a near-
cancellation*. The plateau exists because cancellation is incomplete where
\|B⊥\|/\|C⊥\| > 1 (σ ≈ 0.04–0.30) and the cliff is the crossing back
through 1. This is why the spectral-account prediction misses by ~4×: it
transports amplitude without the interference structure. One mechanism,
measured probe-matched on 896 (crossing ≈ 0.5); E9's 768 crossing
(0.688) did not reproduce on this grid (E19 19.0 — no in-window 768
crossing). The anti-alignment itself was later confirmed not a
shared-arm artifact (E19 19.0) and per-sample at σ = 0.7 (E22.4:
median ρ_i −0.820).

The probe-matched half (E13's owed §4.7 refit) reads from the same run;
A/F redistribution vs the published E1b fit is now attributable to the
probe-set correction alone (same images, same G, same kernel path).

## Phase 2 (conditional on H-a): the twin, in vector units — **NOT FIRED** (H-a failed)

If the plateau is a real data term, the ratio governor's defense — the
1120 twin — must be re-measured on the same instrument: the E9-style
probe on 1280-tier natives (route 1280→1120, matched bins). If
h(B)_1120 ≪ h(B)_896 at matched bins in one unit system, "ratio sets the
amplitude" is falsified in debiased vector units and §4.5/§4.7 are
rewritten; if they agree, the published governor stands and the E1b-era
fit's failure was plumbing (weights + grid), not physics. Separate run
(~4–6 h); never scheduled — H-a failed.

## What lands in the paper

- H-a **failed** ⇒ no governor asterisk and no phase-2 twin; the 896
  panel keeps the measured curve, with the ~4× miss explained by the
  cancellation mechanism above.
- H-b fired ⇒ §4.6 domain paragraph covers 896's shoulder — coherence,
  not damage (896 only; the two-route extension fell probe-matched,
  E19 19.0).
- H-c ⇒ raw-paired primary for the affected bins (E13 protocol).
- H-d ⇒ explicit domain bound on assumption (iii).
- Either way: the probe-matched half of the run discharges E13's owed
  §4.7 refit (E13's record points here), so any movement in A/F against
  the published E1b fit reads as the resolution/probe-set correction.
