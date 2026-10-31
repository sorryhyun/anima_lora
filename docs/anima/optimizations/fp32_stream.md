# fp32 residual stream

`--stream fp32` runs everything around the block GEMMs in fp32: the residual stream `x`, the
modulation, the q / k norms and RoPE, the embedder and the head. The block linears and FA2 stay
bf16 (or int8), and no parameter changes dtype. It is the default for generation.

```bash
make test                # fp32
make test STREAM=bf16    # the forward and sampler of before, bit for bit
```

`GenerationRequest(stream=...)` is the same switch. Training keeps its bf16 stream. EasyControl
stays on bf16, with a warning. In ComfyUI, `Anima Adapter Loader` has a `stream` input and
`Anima fp32 Stream` patches a MODEL with no adapter. `Anima INT8 Loader` loads an INT8 file.

**The same seed gives a different image from `bf16`.** The noise is the same, since an fp32 draw
rounds to the bf16 one. Final latents moved 12–24% in five 6–12 step runs at 768².

## How it works

`library/anima/stream.py::stream_fp32_` runs where int8 converts, before the adapters attach and
`compile_blocks`. It swaps the embedder, head and modulation linears to `Fp32Linear`, which casts
the weight for the call and leaves autocast, swaps the q / k norms to `Fp32RMSNorm`, and enters
bf16 autocast in `forward_mini_train_dit`, so a block linear's input rounds to bf16 and an int8
linear keeps its fp32 kernel input. The sampler state and σ follow the latent's dtype, which also
removes the per-step rounding of the sampler state.

ComfyUI (`~/ComfyUI-Anima_lora-Adapter/stream.py`) does the same around its fused norm + RoPE
kernel, so it leaves those two in the compute dtype.

## Accuracy

Prediction error, `‖v − v_fp32‖ / ‖v_fp32‖` in %, mean over six images, no LoRA
(`bench/anima_stream/comfy_check.py`):

| σ | `library` bf16 | `library` fp32, no RoPE | ComfyUI bf16 | ComfyUI fp32 | ComfyUI to `library` |
|---|---|---|---|---|---|
| 0.1 | 1.76 | 0.55 | 1.74 | **0.54** | 0.63 |
| 0.5 | 0.85 | 0.24 | 0.85 | **0.23** | 0.26 |
| 0.9 | 1.15 | 0.25 | 0.94 | **0.23** | 0.23 |

One 8-step trajectory at 768² and two images (`bench/anima_stream/run_bench.py`, a smoke, one draw
per cell; the full run was stopped, so these are not a result):

| | σ 0.5 / 0.9 | final latent, % from fp32 | ms/step | peak GiB |
|---|---|---|---|---|
| bf16 | 0.89 / 1.15 | 13.5 | 261 | 4.11 |
| fp32 | 0.25 / 0.27 | 8.5 | 280 | 4.17 |
| fp32, no RoPE | 0.27 / 0.29 | 11.1 | 268 | 4.15 |
| int8 | 1.01 / 1.18 | 15.4 | 164 | 2.64 |
| int8 fp32 | 0.51 / 0.56 | 12.8 | 179 | 2.68 |

ComfyUI on the INT8 file, same images. Its kernel reads bf16 unpatched, which rounds around the
activation division ([comfy-kitchen#289](https://github.com/Comfy-Org/comfy-kitchen/issues/289)):

| σ | stock | fp32 kernel input | fp32 stream | `library` int8 fp32, no RoPE |
|---|---|---|---|---|
| 0.1 | 2.29 | 2.13 | **1.33** | 1.32 |
| 0.5 | 0.99 | 0.95 | **0.48** | 0.48 |
| 0.9 | 1.07 | 1.05 | **0.50** | 0.51 |

Forward time was 186, 208 and 229 ms in the same order, one run each with the model load
included, so read it as a rough cost.

The attribution and the gradient: [`anima_stream_attrib.md`](../../../project/int8_train/reports/anima_stream_attrib.md).
