# E11 — Δr̄ direction structure

| | |
|---|---|
| **Status** | **DONE 2026-07-30** — one residual open: the `--uncond` rerun (first launch stopped mid-run 2026-07-30, on hold) |
| **Verdict** | **Norm-only.** No rank-one common mismatch direction — directions are image-specific, so "universal m" downgrades to a **universal amplitude law**. The grid-conditional composition prior is refuted under full captions. |
| **Runs** | `bench/results/20260730-2054-resvec/` (derived `resid_structure.json` is the copied-out artifact) |
| **Instrument** | `bench/run_prior_distance.py --save_residuals` on the 1280 probe cache |
| **Scripts** | `resid_structure.py` — split-half-corrected pairwise cosines + normalized stacked-SVD top-mode share across route pairs per σ |
| **Origin** | `paper/review/response.md`; manuscript consequence in `paper/action.md` §4.4 |
| **In the paper** | §4.4 |

## Results — norm-only

- Non-adjacent corrected cos ≈ 0 at low σ, +0.2–0.33 at high σ.
- SVD top-mode share 0.33–0.36 vs a 0.25 uniform baseline.
- Cross-image direction consistency ≈ 0 everywhere — image-specific
  directions; the grid-conditional composition prior is refuted under
  full captions.
- Δr̄ goes low-frequency as σ rises.

## Open residual

The `--uncond` rerun is implemented; the first launch was stopped
mid-run 2026-07-30 (hold). Relaunch to close the caption-conditioning
caveat.

**Note on vector stores:** fp32 flat LoRA grads ≈ 311 MB × arms × bins,
so runs stay under the gitignored `bench/results/`; only the derived
`resid_structure.json` is copied into `paper_bench/`.
