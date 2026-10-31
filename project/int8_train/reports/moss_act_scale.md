# Activation clipping and the kernel's input dtype, MOSS-Image (2026-10-31)

**Verdict: clipping the activation scale makes the prediction worse at every ratio. Giving
the kernel an fp32 input cuts the error 5–10% for 0.05 s/step, so the trainer and generator
now pass one.** On a bf16 input the kernel rounds around its activation division in bf16.

`src/clip_probe_moss.py`, the six held-out t2i images of
[moss_gptq.md](moss_gptq.md), GPTQ file, fp32 stream, forward only. `‖v − v_fp32‖ / ‖v_fp32‖`,
%. Raw numbers: [`moss_clip_probe.json`](../assets/moss_clip_probe.json) (job
`20261031-142430-9c1da9`), [`moss_kernel_dtype.json`](../assets/moss_kernel_dtype.json) (job
`20261031-143114-b6b9eb`).

## Clipping

The arms replay the kernel in fp32 with the scale `ratio · amax / 127`. Weights are not
refitted.

| σ | ratio 1 | 0.9 | 0.8 | 0.7 | 0.6 |
|---|---|---|---|---|---|
| 0.1 | 3.65 | 4.19 | 7.12 | 12.98 | 22.60 |
| 0.5 | 1.62 | 1.88 | 3.20 | 5.79 | 10.31 |
| 0.9 | 1.22 | 1.46 | 2.71 | 4.77 | 8.72 |

Every clipped ratio loses on all 18 (image, σ) pairs. Ratio 0.9 costs 15–20%, and 0.8 about
double. A token's largest rotated values carry too much signal: losing them outweighs the
smaller step. QuaRot's 0.9 is a 4-bit setting and does not carry over to 8 bits. Crest 4.6
([moss_fold_vo.md](moss_fold_vo.md)) is not slack.

## Kernel input dtype

Ratio 1 came in 5–9% below `grad_error_moss.py`'s `gptq_f32s`, so the replay was not the
kernel. The same run on the kernel itself:

| σ | kernel, bf16 input | kernel, fp32 input | change | replay, ratio 1 |
|---|---|---|---|---|
| 0.1 | 3.849 | 3.637 | −5.5% | 3.647 |
| 0.5 | 1.711 | 1.618 | −5.4% | 1.624 |
| 0.9 | 1.336 | 1.201 | −10.1% | 1.219 |

- The bf16 column reproduces `gptq_f32s` to all digits, and fp32 wins on all 18 pairs.
- The CUDA kernel rotates in fp32, then rounds to the input dtype three times: the rotated
  value, the scale for the division (the dequant multiplies by the fp32 scale), and the quotient
  before `nearbyintf`. Details: [int8.md](../../../docs/moss/optimization/int8.md).
- Against the 0.81% of bf16 GEMMs on an fp32 stream, the int8 excess at σ 0.5 goes 1.51 → 1.40.
- The block forward still casts `y` to bf16 before the linears, so this arm fixes only what
  happens inside the kernel. Keeping `y` in fp32 on an fp32 stream is untested.

## Cost

`comfy_kitchen.int8_linear`, 4600 tokens, input cast included, ms per call on the 5070 Ti:

| linear | bf16 input | fp32 input |
|---|---|---|
| `q`, `k`, `v`, `o` | 0.34 | 0.44 |
| `ffn_up_gate` | 1.40 | 1.50 |
| `ffn_down` | 0.85 | 1.12 |

A block goes 3.6 → 4.4 ms, about 34 ms per forward over 42 blocks. A real step on the trainer
goes 1.05 → 1.10 s (`ama_mitsuki_layout`, budget 0.3, median of steps 11–40, jobs
`20261031-135612-034f0e` and `20261031-143638-3fee40`), peak 12.98 GB in both.

On the trainer path, `grad_error_moss.py` reads 3.637 / 1.618 / 1.201% (job
`20261031-143638-4fff9c`). At σ = 0.5 over 30 images, 1.654 → 1.568%, with gradient medians
unchanged and p90s lower in five of six block groups (job `20261031-144550-e6b0d2`,
[`moss_fp32in_grad_error_t2i_n30.json`](../assets/moss_fp32in_grad_error_t2i_n30.json)). On a
bf16 stream the p90s of blocks 7–27 rise (2.08–2.36 → 2.58–3.25); the int8 plan runs fp32.
The backward's `dY`, ComfyUI's nodes and i2i were not scored.

## Kernel fix

[comfy-kitchen#289](https://github.com/Comfy-Org/comfy-kitchen/issues/289), fork branch
`int8-activation-fp32-division`, built for `120f` only. Job `20261031-150223-2acb67`,
[`moss_kernel_patched.json`](../assets/moss_kernel_patched.json):

- A bf16 input gives the output of the same input cast to fp32, bit for bit, on all three shapes.
- Prediction error 3.637 / 1.618 / 1.201%, the fp32 input's.
- 0.334 / 1.38 / 0.84 ms per call, unpatched bf16's speed.
- Upstream's `test_int8.py`, `test_int8_input_act.py`, `test_int8_residual.py` and
  `test_convrot_unpacked_quantization.py`: 301 passed, 38 skipped. The CUDA change alone fails two
  CUDA-against-eager parity tests, so eager and Triton change with it.

The trainer keeps the fp32 cast until a release carries the fix.
