# webgpu_int8: INT8 ONNX weights for the in-browser Anima turbo

Opened 2026-11-12. The consumer is yaar's `user-apps/anima` (`../../yaar`), which runs the ONNX
files from `huggingface.co/sorryhyun/anima-turbo-4step/onnx/` on onnxruntime-web 1.30.0. That build
uses the native WebGPU EP (`ort.webgpu.bundle.min.mjs`). Step 1 is the text encoder (Qwen3-0.6B).
The DiT comes later.

## Text encoder: `onnx/int8/text_encoder_int8.onnx`

Same interface as `text_encoder_fp16.onnx`:
- Inputs: `input_ids` and `attention_mask`, both int64 `(1, 512)`.
- Output: fp16 `qwen_prompt_embeds` `(1, 512, 1024)`.
- Sidecar: **630,688,768 B**, against 1,192,421,376 B for fp16.

| Part | Format |
|---|---|
| 195 of 196 decoder linears | ConvRot: input rotated in-graph by blockdiag(H256) (Sylvester Hadamard / 16), then `MatMulNBits` with bits 8, block 32, fp16 scales, default zero point 128 and `accuracy_level` 4 |
| layer 2 `mlp.down_proj` | fp16 `MatMul`. Its input on token 0 is ~700× the other tokens' absmax (Qwen's massive activation) |
| embedding | `GatherBlockQuantized`: uint8, bits 8, one fp16 scale per 128 elements, zero point 128 |
| everything else | unchanged from the HF fp16 graph |

The op support was read from the ORT v1.30.0 source:
- `MatMulNBits` on WebGPU takes `nbits` 4 or 8.
- With `accuracy_level == 4`, block % 32 == 0, K % 128 == 0, N % 16 == 0 and the Subgroups
  feature on a non-Apple adapter, it takes the **DP4A** path: int8×int8 dot products, with the
  activation rounded per token per 128-wide K block (absmax / 127).
- Without DP4A (Apple, no subgroups) it dequantises the weight and computes in fp16, which is the
  "w8" column below.
- `GatherBlockQuantized` with uint8 and bits 8 is registered on WebGPU.
- Both ops need `shader-f16`. The fp16 graph already needs it.

### Calibration

`src/te_convrot_gptq.py` (daemon, about 3 min per variant):
- **Data**: training captions from `post_image_dataset/resized/**.txt`, shuffled with seed 0.
  The first 64 are the eval set; the next 256 (35.5k tokens) are calibration.
- **Method**: the layers are quantized in order, each layer's inputs coming from the layers
  already quantized (W8A8). GPTQ uses static block scales and act order. The target is
  act-aware least squares, as in `project/finished/moss_lora/src/te_gptq.py`
  (H̃ from the DP4A-rounded rotated input).

### Text representation error

Measured on the 64 held-out captions. The metric is the per-token relative L2 of
`last_hidden_state` against fp32. "tok1+" averages tokens 1 onward; tok0 is the attention-sink
token.

| Variant | W8A8 (DP4A) tok1+ / tok0 | W8 (weight-only) tok1+ / tok0 |
|---|---|---|
| fp16 compute (≈ the shipped fp16 ONNX) | 0.0013 / 0.020 | |
| bf16 compute (the TE caches training saw) | 0.0109 / 0.140 | |
| RTN, block 128 | 0.0198 / 1.104 | 0.0158 / 0.939 |
| GPTQ, block 128 | 0.0126 / 0.361 | 0.0045 / 0.059 |
| GPTQ, block 128, keep L2 down_proj | 0.0116 / 0.410 | 0.0044 / 0.067 |
| act-aware, block 128, keep | 0.0119 / 0.223 | 0.0071 / 0.141 |
| act-aware, block 128, no keep | 0.0786 / 0.162 | 0.0780 / 0.129 |
| **act-aware, block 32, keep (shipped)** | **0.0117 / 0.167** | **0.0068 / 0.144** |

On the DP4A path the shipped file sits at about 1.1× (tok1+) and 1.2× (tok0) of the bf16 noise
training was cached with. On the weight-only path it is at or below that noise. int4 (RTN,
block 32) measured 0.24 and was dropped.

The graph check, on ORT CPU against `src/te_ref_dump.py`'s 32 padded prompts:
- Built with `--acc 0`, it gives 0.0070 / 0.136, which matches the torch w8 simulation. This
  covers the rotation placement, the zero point, the int8 embedding and fp16 compute.
- Pad rows come out zero.
- CPU's `accuracy_level` 4 rounds activations differently from DP4A (0.0111 / 0.312), so it is
  not the browser number.

### Not checked yet

The browser run on WebGPU (the user will run it). The Chrome on this box, on Linux with NVIDIA,
exposed no `shader-f16`, so the session loaded but `GatherBlockQuantized` refused to run. That
setup cannot run the fp16 graph either.

## Files

- `src/te_convrot_gptq.py`: calibration and the variant sweep. Writes
  `output/webgpu_int8/te_int8.<variant>.safetensors` and `.report.json`.
- `src/te_onnx_int8.py`: rewrites the HF `text_encoder_fp16.onnx` with that file. Linears are
  matched by node order (q k v o gate up down per layer) and checked against the safetensors.
- `src/te_ref_dump.py` and `src/te_onnx_eval.py`: the ORT CPU check.
- `yaar_request.md`: the change needed in yaar to load the file.

Local copies of the outputs are in `output/webgpu_int8/`.
