# Cross-LoRA bias-signature check

Base run: `archive/dcw-learnable-calibrator/lora-transfer/base`
n_steps=24

## Integrated signed gap per band (%Δ vs base)

| config | LL signed | Δ_LL% | LH | Δ% | HL | Δ% | HH | Δ% | branch |
|---|---|---|---|---|---|---|---|---|---|
| **base** | -355.49 | 0.00% | -160.78 | 0.00% | -164.26 | 0.00% | -121.33 | 0.00% | — |
| **artist@1.0** | -356.33 | **-0.24%** | -160.04 | +0.46% | -163.82 | +0.27% | -120.92 | +0.34% | base-DiT scope |
| **hydra@0.5** | -361.75 | **-1.76%** | -160.74 | +0.03% | -164.36 | -0.06% | -121.04 | +0.24% | base-DiT scope |
| **hydra@1.0** | -373.68 | **-5.12%** | -159.92 | +0.54% | -163.98 | +0.17% | -119.55 | +1.47% | base-DiT scope |

## Per-step LL gap shape correlation vs base

| config | Pearson r (24 steps) | late-half max \|residual\|/\|base\| |
|---|---|---|
| artist@1.0 | **r = 0.99999** | 0.29% |
| hydra@0.5 | **r = 0.99999** | 1.84% |
| hydra@1.0 | **r = 0.99992** | 5.21% |

## Cross-sample SNR profile (LL/LH/HL/HH)

| config | LL | LH | HL | HH |
|---|---|---|---|---|
| base | 0.993 | 2.707 | 2.943 | 2.184 |
| artist@1.0 | 1.001 | 2.670 | 2.915 | 2.168 |
| hydra@0.5 | 1.012 | 2.696 | 2.950 | 2.185 |
| hydra@1.0 | 1.040 | 2.629 | 2.879 | 2.125 |

## Branch assignment

All LoRA × multiplier configurations land in the **base-DiT scope** branch (max |Δ| ≤ 15% on every band). No per-LoRA artifact needed; one reference profile per base DiT release suffices.
