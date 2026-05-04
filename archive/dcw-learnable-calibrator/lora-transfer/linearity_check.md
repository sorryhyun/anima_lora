# LoRA multiplier-linearity check
Inputs:
  base       = archive/dcw-learnable-calibrator/lora-transfer/base/per_step_bands.csv
  mid (m=0.5) = archive/dcw-learnable-calibrator/lora-transfer/hydra-mult05/per_step_bands.csv
  full (m=1.0) = archive/dcw-learnable-calibrator/lora-transfer/hydra-mult10/per_step_bands.csv

## Integrated signed gap per band (linearity at mult=0.5)

| band | base | obs @0.5 | obs @1.0 | linear-pred @0.5 | linerr (% of base) | Δ@1.0 vs base |
|---|---|---|---|---|---|---|
| **LL** |  -355.49 |  -361.75 |  -373.68 |  -364.58 | **+0.80%** | -5.12% |
| **LH** |  -160.78 |  -160.74 |  -159.92 |  -160.35 | **-0.24%** | +0.54% |
| **HL** |  -164.26 |  -164.36 |  -163.98 |  -164.12 | **-0.15%** | +0.17% |
| **HH** |  -121.33 |  -121.04 |  -119.55 |  -120.44 | **-0.49%** | +1.47% |

## Per-step LL linearity residual
- max |residual| anywhere: 0.338
- max relative residual at step 0 is meaningless (base ≈ 0 there)
- late-half (i ≥ 12) max relative residual: **1.07%**
- late-half (i ≥ 12) mean relative residual: 0.72%
