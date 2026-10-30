# Step status (2026-10-30)

Where the time of one compiled Anima LoRA step goes, bf16 base against the int8 `full`
base, per CUDA kernel. `make lora` defaults (method `lora`, preset `default`), B=1,
batches of 2940–4200 image tokens; 20 timed steps, 8 profiled.

    make daemon-run ARGS="--stall-timeout 1200 project/int8_train/src/profile_step.py \
        --int8 full --out project/int8_train/assets/step_profile_kernels_full.json \
        -- --method lora --preset default"

Jobs `20261030-140951-25f85d` (int8) and `20261030-140952-24ae32` (bf16). Raw numbers:
[`step_profile_kernels_full.json`](assets/step_profile_kernels_full.json),
[`step_profile_kernels_off.json`](assets/step_profile_kernels_off.json).

ms per step:

| category | bf16 | int8 | note |
|---|---|---|---|
| attention (FA2) | 169.0 | 180.2 | same kernels; the 11 ms gap is run to run |
| int8 GEMM | – | 79.3 | |
| rotate + quantize (`quantize_int8_rowwise_convrot64`) | – | 25.0 | new with int8 |
| bf16 GEMM (cross-attn kv, LoRA rank, adaLN) | 306.5 | 38.1 | |
| LayerNorm + adaLN modulate + residual (Triton fused) | 25.2 | 29.3 | |
| GELU forward / backward (fused) | 24.9 | 26.2 | |
| other add / mul / reductions (fused) | 24.4 | 27.3 | |
| RoPE / qk-norm | 9.8 | 9.9 | |
| fp32 → bf16 copy | 0.5 | 4.1 | the `x.to(bfloat16)` inside the int8 op, 102 calls |
| optimizer (fused AdamW, LoRA parameters) | 2.7 | 2.6 | |
| step wall time | 593.6 | 430.7 | |

## Reading

- Attention is 42% of the int8 step. Elementwise, norms and casts are 23%, the optimizer
  under 1%.
- The elementwise kernels are already fused by inductor and memory-bound. Halving all
  of them would bound the step at about 1.13x.
- Around the opaque int8 op, norm, add/mul and copy grow by about 10 ms over bf16: some
  fusions across the op are lost.
- Two levers, not yet tried: move the bf16 cast out of the custom op so inductor fuses it
  into its producer (~3.5 ms); fuse rotate + quantize into the producing norm/modulate or
  GELU kernel (up to ~15–25 ms), which needs a comfy_kitchen GEMM entry that takes a
  pre-quantized input (unchecked).
