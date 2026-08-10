# E26 — cross-adapter cancellation geometry (the E7 pair)

| | |
|---|---|
| **Status** | **RESOLVED 2026-08-10** — E26.0 smoke: both adapters PASS (2026-08-09). Full grid: **flat REPLICATES · dirty REPLICATES** (5/5 bins readable and passing on both). Records: `e260_smoke.json`, `e26_grid_read.{py,json}`, `e26_grid_across_sigma.{py,json}`. |
| **Question** | Does the B–C cancellation geometry (near-cancellation, deep ρ, negative I, reliable pooled residual direction) exist on LoRA adapters other than the line's operating point (`anima_soup_sincos`)? **Answer: yes — on both E7 adapters, at every bin of the 768 window.** |
| **Adapters** | `output/paper/e7/anima_soup_e7_flat.safetensors` · `anima_soup_e7_dirty.safetensors` — verbatim shipped recipe, dim 32/alpha 128 ⇒ parameter space identical to sincos, designed style axis, artists disjoint from sincos. (s1001–s1003 seed siblings stay an unrun pre-declared extension tier.) E7's own probe runs were not reusable (no `arm_sums/`) and were not read here. |
| **Scope consequence** | (`revision_plan.md` §3/§5, lands in tex at §7 step 7) The geometry claims' breadth extends from one operating point to **three adapters of this base model (disjoint data, designed style axes) at 768 across the window** — same-base qualifier stays (the cross-adapter direction read is frame-confounded, see 26.0-2); **896 replication remains the stated open cell**. |

**Provenance / freeze discipline.** The smoke was pre-registered and
its thresholds frozen from committed sincos rows before any
E7-adapter gradient existed (`54edadbc`); the full-grid amendment —
grid cells, in-session reference, per-bin REPLICATES thresholds — was
frozen before any grid gradient existed (`fa55b2a4`). This README was
condensed to the resolved state after the read; the verbatim frozen
text, including every contingency branch that never fired
(INCONCLUSIVE + D = 24 top-up remedy, single-PASS / dual-FAIL smoke
decision arms, PARTIAL/FAILS grid verdict wordings, the
reboot-interleave clause), lives at those two commits. No unfired
branch is restated here.

## Protocol (as run)

Two tiers: **E26.0 smoke** (one condition — σ = 0.7 / route 1024→768,
the line's best-behaved bin — go/no-go for the grid) → **full grid**
(licensed by the smoke's dual PASS).

Both tiers are verbatim e193
(`bench/results/20260807-0745-e193-depth-ledger/result.json` is the
arg authority), deltas only:

| knob | e193 | E26.0 smoke | E26 grid |
|---|---|---|---|
| `--adapter` | `anima_soup_sincos` | the E7 checkpoint | same |
| `--sigma_window` | `0.23333…,0.76667…,4` | `0.63333…,0.76667…,1` (center 0.7) | `0.23333…,0.76667…,4 : 0.76667…,0.9,1` (bins {0.3, 0.4333, 0.5667, 0.7, 0.8333}) |
| `--demote_edges` | `896,768` | `768` | `768` |
| `--endpoint_bin` | on | off | off |
| `--label` | `e193-depth-ledger` | `e260-smoke-{flat,dirty}` | `e26-grid-{flat,dirty}` |

Everything else matched: `--num_images 40 --probe_list
project/sigma_lowres/paper_bench/experiments/e13/e1b_probe_list.json
--draws_per_bin 12 --repromote --keep_arm_sums --self_floor
--deterministic --seed 42`, tier 1024, fp32 sums, no grad-ckpt, no
compile; one probe process per adapter (kernel-path rule),
daemon-queued. Probe data held fixed across all adapters (the e1b
40-stem list): the adapter is the only moving part.

**The grid is 768-only, the E28-twin window** — the exact grid of
`20260810-0658-e28-native-twin-768`, measured 3.8 GPU-h/adapter.
Deviation from the plan's §5 sketch, recorded at the freeze: "896 at
the verdict bins" was dropped — one probe run cannot carry per-route
bin sets, a separate 896 run forfeits the shared native/reenc arms
(cost ≈ full grid), and per the 2026-08-10 environment amendment its
cells would need their own same-environment sincos reference (the
twin covers 768 only). 768 carries the paper's window/dip claims and
the larger residual; 896 stays the open cell.

**In-session sincos reference = the twin store** (same boot family
2026-08-09 19:32, same grid, same estimand path). Committed e193/e221
rows are context, never verdict denominators (environment amendment).

**Criteria applied** (frozen at `fa55b2a4`; per adapter, over the 5
bins): a bin is *readable* iff rel_cos_B ≥ 0.5 ∧ rel_cos_C ≥ 0.5;
**REPLICATES** iff ≥ 4/5 bins readable AND at every readable bin
I < 0, ρ ≤ −0.5, h(B+C) < min(h(B), h(C)). The smoke used the same
three criteria at its single bin (deliberately generous vs sincos's
ρ ≈ −0.89: the smoke asked "is the cancellation present," not "is it
sincos-deep"). **Identity-consistency column** (pre-registered):
per bin, ρ_implied = (h²(B+C) − h²(B) − h²(C)) / (2·h(B)·h(C)) next
to measured ρ — the "enforcement depth scales with the perturbation"
upgrade is claimed only where measured deepening exceeds what
ρ_implied already forces from the magnitudes.

**Validation gates**: passed before any new read — unmodified
`vector_ledger.py --data_ref reenc` reproduces the committed e193
global 768/σ = 0.7 row and the e221 rows from their stores to max
dev 0.0.

### Reference rows (committed, quoted before running)

sincos @ σ = 0.7 / 768, `data_ref = reenc`:

| source | D | draws | S | F | I | ρ | rel_B | rel_C | h(B) | h(C) | h(B+C) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e193 `ledger_depth.json` global | 40 | 12 | 0.070 | 0.119 | −0.163 | −0.890 | 0.879 | 0.852 | 0.0570 | 0.1767 | 0.0440 |
| e221 `ledger.json` | 16 | 24 | 0.060 | 0.116 | −0.148 | −0.891 | 0.838 | 0.801 | 0.0548 | 0.1623 | 0.0498 |

The signature is protocol-robust across D/draws; h(B+C)/h(C) ≈
0.25–0.31 (the cancellation removes ~70–75 % of the graph term's
angular cost) and h(B+C) < h(B) < h(C).

## Full-grid results (2026-08-10) — **flat REPLICATES · dirty REPLICATES**

Read: `e26_grid_read.py` → `e26_grid_read.json` (this dir), frozen
criteria applied to `vector_ledger.py --data_ref reenc` ledgers
(unmodified instrument) on both grid stores; in-session sincos
reference = the e28 native twin's ledger (same boot family).

- **flat: REPLICATES** — 5/5 bins readable (relB 0.92–0.98, relC
  0.82–0.92), all 5 pass (I < 0, ρ −0.890…−0.948, h(B+C) < min(h(B),
  h(C)) everywhere).
- **dirty: REPLICATES** — 5/5 readable (relB 0.93–0.97, relC
  0.86–0.90), all 5 pass (ρ −0.923…−0.980). The smoke's G-inflation
  caveat did not express at grid level: every bin's h-ordering holds
  outright.
- **ρ(σ) shape is shared, depth orders with the perturbation at every
  bin**: all three adapters deepen toward high σ (twin −0.86…−0.92,
  flat −0.89…−0.95, dirty −0.92…−0.98) with the pointwise ordering
  sincos < flat < dirty — the roadmap §1(a) stated prior
  (base-carried organization, adapter-borne amplitude) is consistent
  with the grid; no adapter-dependent σ-structure (holes/blocks)
  appeared.
- **Identity-consistency column**: ρ_implied lands ≤ −0.97 (often
  < −1, the truncation-domain regime) at 9 of 10 adapter-bins —
  there the deep measured ρ is reported as **arithmetic** per the
  frozen rule. The single exception is **dirty σ = 0.7** (implied
  −0.868 vs measured −0.964): the one cell where measured enforcement
  exceeds what the magnitudes force; recorded, not generalized.
- **Reference context (no verdict weight)**: the twin's sincos rows
  sit lower in h(B) than the committed e193 σ = 0.7 row (0.0295 vs
  0.057 — cross-environment *scalar* drift at the large end of the
  tolerated band), and on the twin the smoke's h-ordering criterion
  would fail at 4/5 bins for sincos itself (h(B+C) ≳ h(B) in-window,
  marginal at σ = 0.7: 0.0338 vs 0.0295). The adapter verdicts are
  absolute per-bin and do not use the reference; recorded because any
  future sincos-vs-adapter *depth* comparison must use the twin rows,
  not e193's, per the environment amendment.

**Run record (2026-08-10)**: flat DONE
(`bench/results/20260810-1114-e26-grid-flat`, 3.9 h, args verified
against the frozen amendment). dirty's first submission
(`20260810-111408-a795d9`) crashed 6.5 min in with SIGBUS — the root
disk hit 100 % as flat's store landed; the partial store
(`20260810-1506-e26-grid-dirty`, 9.2 GB) was deleted, ~27 GB freed
(unrelated files + caches; no committed store touched), and dirty was
**resubmitted with bit-identical argv in the same boot** (job
`20260810-151849-b08823` → `20260810-1519-e26-grid-dirty`;
`/tmp/torchinductor_*` untouched, no reboot). Same-environment status
vs the twin reference is preserved; no protocol deviation beyond the
retry itself.

**Store reclamation (2026-08-10, post-read)**: the grid-family raw
arm_sums (flat, dirty, e28 native twin) were reclaimed for disk after
committing their across-σ B̂/Ĉ/R̂ tables to
`e26_grid_across_sigma.json` (`e26_grid_across_sigma.py`; estimand =
E24 `build_cond`/`cross_cos` verbatim, R leg per `e28_read768.r_cos`;
twin B-leg checked against the committed e28 table, PASS at 1e-4; no
verdict applied — the tables are roadmap §1(c) input only). Store
`manifest.json`s retained. Roadmap §3 records the full reclamation
(family A + B) and the go-forward policy.

## E26.0 smoke results (2026-08-09)

Runs: `bench/results/20260809-1323-e260-smoke-flat/` (job
`20260809-132306-f42e53`, ~46 min GPU) ·
`…-1409-e260-smoke-dirty/` (job `20260809-132312-55507b`).
Full record: `e260_smoke.json`.

### 26.0-1: **both adapters PASS** → full-grid amendment licensed

| adapter | G | h(B) | h(C) | h(B+C) | ρ | I | rel_B/rel_C |
|---|---|---|---|---|---|---|---|
| sincos (ref) | 0.108 | 0.057 | 0.177 | 0.044 | −0.890 | −0.163 | 0.88/0.85 |
| flat | 0.073 | 0.119 | 0.261 | **0.049** | **−0.939** | −0.368 | 0.97/0.92 |
| dirty | 0.023 | 0.497 | 0.725 | **0.347** | **−0.959** | −3.128 | 0.90/0.84 |

Every frozen criterion held on both adapters. Structure carried to
the full grid (and confirmed there):

1. **Cancellation depth orders with leg size**: ρ deepens
   sincos → flat → dirty exactly as the legs grow — the enforcement
   scales with the perturbation, it is not a fixed-depth artifact of
   the operating point. (At grid level this deepening is arithmetic
   at 9/10 bins per the identity column — see above.)
2. **The residual level mirrors E7's floor-level fact**: flat's
   h(B+C) ≈ sincos's (0.049 vs 0.044, legs 2× larger) while dirty's is
   ~8× (0.347) — the same flat-good / dirty-bad ordering as E7's
   checkpoint-dependent cos_floor (0.73 vs 0.50).
3. **Dirty caveat** (smoke-level only; did not express at grid
   level): G = 0.0226 (~5× below sincos) inflates its S/F/I via the
   2G² normalization (E13 caveat) and adds ref-direction noise to its
   h row; ρ and the h *ordering* were the robust reads.

### 26.0-2 (descriptive): rel_cos_R passes everywhere; cross-adapter cosines are frame-confounded

- **rel_cos_R**: sincos 0.72 · flat 0.68 · dirty 0.51 — all clear
  E25.0's ≥ 0.5 bar; the pooled residual direction exists on every
  adapter at this condition (dirty marginal).
- **Cross-adapter raw cosines ≈ the frame baseline**: B̂/Ĉ/R̂ pairwise
  0.27–0.35, but the control — cos of the pooled *native* gradient
  directions ĝ across adapters — is 0.37–0.39. Gradients w.r.t.
  different adapters' parameters live in mostly non-overlapping
  frames, so a raw param-space cosine cannot separate "the residual
  axis is adapter-specific" from "the frames differ." The stated
  E19.6 prior (B̂ highly shared) does not transfer: E19.6 moved the
  backbone under a fixed probe-adapter frame; here the frame itself
  changes. Verdict-shaped conclusion: *no evidence of a privileged
  shared axis above frame overlap, and no evidence against structure
  below it.* Consequently the frame-free cross-adapter axis estimand
  was **dropped** from the grid amendment (new instrument work the
  paper does not need); the "property of the model vs of the adapter"
  question is stated open in the limitation paragraph.
