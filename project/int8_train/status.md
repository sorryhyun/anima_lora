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

## Cast moved out of the op

`Int8TrainLinear.forward` now casts `x` to bf16 before the custom op. Re-profiled
([`step_profile_kernels_full_castout.json`](assets/step_profile_kernels_full_castout.json)):
the 102 fp32 → bf16 copies are gone (copy_cast 4.1 → 0.9 ms) and the LayerNorm kernels
drop 29.3 → 26.1 ms, as the cast folds into them. Step 430.7 → 424.5 ms. Attention moved
10 ms between the two runs on identical kernels, so the step delta is within run noise;
the kernel-level saving is 3–6 ms.

## Fusing rotate + quantize into the producer: not worth it

- comfy_kitchen 0.2.40's public API has no GEMM that takes a pre-quantized activation.
  The private `_C.cutlass_int8_dequant(xq, wq, x_scale, w_scale, …)` is that GEMM,
  ABI not stable.
- The public `int8_linear` fusions do not fit Anima's block: `input_act` folds
  `gelu_tanh`, `swiglu` or `rms_norm` into the quantizer, but Anima uses exact
  `nn.GELU()` and LayerNorm with shift + scale modulation; the `residual` epilogue
  would need the LoRA output added before the gate.
- LoRA sits on every converted linear and reads the same bf16 input, so the producer has
  to write it regardless. Fusing saves only the quantizer's bf16 read. The quantize
  kernels already run at 525–615 GB/s (60–70% of 896 GB/s); removing that read is at
  most ~16 ms, realistically ~10 ms (~2.5%), for custom Triton norm / modulate / GELU +
  rotate + quantize kernels in the forward and the backward on a private ABI.
