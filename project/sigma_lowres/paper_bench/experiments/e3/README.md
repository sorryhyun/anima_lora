# E3 — aggregation-conditioned safety map

| | |
|---|---|
| **Status** | **DONE 2026-08-04** — run + a+b/B fit landed; §4 aggregation paragraph rewritten with the measured readout |
| **Verdict** | The 1/B collapse is monotone at every σ, and the fit's intercept — the coherent drift no batch size removes — is ≤ 0.02 (896) / ≤ 0.03 (768) at σ ≥ 0.44 but persists over the lower bins (+0.04–0.09 / +0.05–0.28 / +0.15–0.60 for 896/768/512): aggregation buys safety only where the per-example map was already near-safe. The per-example 896 endpoint gap (+0.042, E1b) averages out in the aggregate (+0.001). Shipped operating point is batch 1 × accum 1, so the published per-example map **is** the shipped-trainer object; the pooled maps bound accumulation users. |
| **Runs** | `runs/20260804-0002-e3-pooled-verdict/` (result.json `metrics.pool` = 10 strata + aggregate, pooled self-floors; `e3_fit.json` = the fit). First launch (2026-08-03, job `20260803-111942`) was kernel-OOM-killed — see Ops note. |
| **Scripts** | probe `--pool 4 --self_floor --pool_spill --pool_no_norm` (E1b grid: N=40, 8+endpoint bins, D=8, routes 896/768/512); fit `e3_fit.py` |
| **Why it exists** | Review finding R3: "never safe" is aggregation-dependent. `record/report.md` (pool4 addendum) has pooled gap_768 ≈ 0 at σ ≥ 0.875 and pooled gap_896 ≈ 0 at σ ≥ 0.625 — the per-image and batch-SGD objects genuinely disagree at high σ. |
| **Depends on** | [E1](../e1/) (paired debiased object + self-floors), [E8.1](../e8/) (ε\* is now the definition of "safe") |
| **In the paper** | §3.1 (aggregation operator is part of the estimand, not an application detail); the §4 map paragraph's **[pending]** marker replaced with the measured a + b/B readout |

**Fit conventions** (`e3_fit.py`): paired debiased gap (arm − reenc) at
every level, B ∈ {1 (per-image), 4 (strata), 40 (aggregate)}; cells with
gap outside [−0.5, 1.5] masked (non-physical ĉ > 1.5 — the 512 endpoint
cell hit −1.15); the aggregate point borrows the stratum SEM × √(P/N).
The σ = 0.3125 bin is anomalous across routes (aggregate floor dip to
0.87, imgsplit 0.76) — read it with the ε\* instrument resolution, not
at face value. Stratum-level redundancy Spearman is negative but
n = 10 non-significant (−0.26/−0.45/−0.49) — the pool4-addendum null
stands.

**Ops note (why the first launch OOM'd).** `--pool` used to disable the
streaming-arm-retirement path, so 10 self-floor arm lists × 9 bins ×
311 MB (~28 GB) sat resident per image on top of ~15 GB process
overhead, before the in-RAM per-stratum accumulator (~34–62 GB more)
even filled — kernel OOM at 45.7 GB anon RSS ~8.5 min in (and it took
the daemon down). Fixed 2026-08-04 in `sigma_probe/`: pooled arms
stream into the accumulator per-arm (bit-identical per-key order,
`tests/test_sigma_probe_pool.py`), and `--pool_spill` disk-backs the
per-stratum accumulator (`--pool_no_norm` drops the norm side-channel
so both spills fit the disk; ~67 GB transient, deleted at run end).
Peak RSS of the successful run: **< 1 GB anon** + evictable memmap
pages; wall 4.97 h.

Original spec (both parts now landed): one verdict-grid run with
`--pool` + `--self_floor` so pooled arms get self-floors too, and the
two-maps framing — per-example (worst case, and, at the shipped
batch 1 × accum 1, literally the shipped-trainer object) vs
batch-aggregate (what accumulation users consume), "safe" defined by
the ε\* non-inferiority test ([E8.1](../e8/)). The `--pool <actual
batch × accum>` reading was degenerate (shipped product = 1), so the
run used pool 4 to give the a + b/B fit three batch sizes
(B ∈ {1, 4, 40}) — the estimand fix registered in the v1
`action.md`.
