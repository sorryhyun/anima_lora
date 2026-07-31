# E10 — exact target-content vectors

| | |
|---|---|
| **Status** | **DONE 2026-07-30** |
| **Verdict** | **Parallel landing.** Demotion shortens the target-content vector *along* ĝ_src; rotation only at 512. This explains [E2](../e2/)'s flat α-slope (an angular gap is blind to parallel rescaling; κ⊥ is second-order). |
| **Runs** | `bench/results/20260730-2116-e10-kappa/` |
| **Instrument** | `bench/run_sigma_probe.py --target_alpha 0,1 --target_kappa --keep_arm_sums`, endpoint-only, N=40 |
| **Analysis** | `../../vector_ledger.py` (shared with [E9](../e9/)) |
| **Origin** | `paper/review/response.md`; manuscript consequence registered in `paper/action.md` §4.3 |
| **In the paper** | §4.3 — the "unresolvable share" paragraph |

**Design.** The forward pass is α-independent, so t = ḡ(1) − ḡ(0) is
exact at shared seeds; per-image + aggregate κ∥/κ⊥ of δt = t_dem − t_src
decide parallel-landing vs J^T-attenuation.

## Results

|t_src|/G ≈ 2.23 aggregate; δt κ∥ −0.75 / −1.18 / −1.86 vs κ⊥
0.09 / 0.14 / 0.20 on 896 / 768 / 512, rel ≥ 0.995, reenc control at the
noise floor. Demotion **shortens t along ĝ_src**, with rotation only at
512 (cos 0.74).
