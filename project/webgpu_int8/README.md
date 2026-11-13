# webgpu_int8: INT8 ONNX weights for the in-browser Anima turbo

Opened 2026-11-12. The consumer is yaar's `user-apps/anima` (`../../yaar`), which runs the ONNX
files from `huggingface.co/sorryhyun/anima-turbo-4step/onnx/` on onnxruntime-web 1.30.0. That build
uses the native WebGPU EP (`ort.webgpu.bundle.min.mjs`). Step 1 is the text encoder (Qwen3-0.6B),
step 2 the DiT and VAE decoder: one resolution-free graph set for every ratio, DiT in int8.

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

The browser run on WebGPU. On Linux with NVIDIA, Dawn withholds `shader-f16` on Vulkan
(crbug.com/42251215) until Chrome is launched with
`--enable-features=Vulkan --enable-dawn-features=vulkan_enable_f16_on_nvidia`, the set yaar's own
launchers pass (`packages/server/src/lib/browser/webgpu-flags.ts`). Without the Dawn toggle the
session loads but `GatherBlockQuantized` refuses to run, and the fp16 graph cannot run either.
`--enable-unsafe-webgpu` and `allow_unsafe_apis` do not lift the hold.

## DiT and VAE: one graph set for every resolution

The published DiT is five per-ratio exports (`dit_<bucket>_fp16_r16`, seven segments each, one
shared 3.9 GB sidecar), and the VAE decoder is five per-ratio graphs too. A ratio was a graph
only because the export constant-folded the resolution, not because the weights differ.

### What depends on resolution

The four non-square DiT exports are node-for-node identical. Diffing them shows:

- **seg1–6 (the blocks)**: six int64 Reshape targets per segment, holding the token count
  N = ph·pw (e.g. `[1, 1014, 16, 128]`).
- **seg0**: also the patchify shapes (ph, pw), the zero padding-mask channel `(1, 1, 1, H, W)`,
  and the Cosmos RoPE angle tables `(1, ph, pw, 21)` for rows and columns (T-axis angles all
  zero). The tables equal `fp16(outer(arange, f))` with f = 1 / θ^(arange(0, 42, 2) / 42),
  θ = 1e4·4^(42/40), bit for bit, in every bucket.
- **seg6**: the unpatchify shapes.

The VAE: Expand and Reshape targets at s·H, s·W (s = 1, 2, 4, 8) and H·W (mid-block attention),
and the nearest-2x upsample written as GatherND over index tables ⌊i/2⌋, ⌊j/2⌋.

### Rewrite

- `src/dit_onnx_dynamic.py` fits each differing shape element to a symbol (N, ph, pw, H, W)
  across the four buckets and rebuilds it in-graph from `Shape(latent)`. It rebuilds the RoPE
  tables with Range / Mul / Cast, which it checks bit-exact against every bucket, and the
  zero tensors with ConstantOfShape. Every segment takes `latent`, for its shape only. Seven
  graphs (~9 MB) over the unchanged HF sidecar: `output/webgpu_int8/onnx_dyn/dit_dyn_fp16_r16.*`.
- `src/vae_onnx_dynamic.py` does the same for the VAE, fitted on 624x416 / 416x624 (H and W
  swap, so the fit disambiguates; H·W is caught although equal in both). The index tables are
  rebuilt from the feature map's own shape. `onnx_dyn/vae_decoder_dyn_fp16.onnx`, 0.54 MB.
- `src/dit_onnx_int8.py --buckets dyn` rewrites the dynamic set onto the v4_preview turbo DiT
  (merged14 + turbo EMA 1112_1512 baked, `project/anima_turbo/deploy/deploy.sh bake`) with
  `dit_int8.anima_turbo_4step_v4_preview.safetensors` from `src/dit_convrot_gptq.py`. 224
  block linears are converted to MatMulNBits, each run in fp32 between two Casts
  (`--fp32_matmul`, see "DP4A and fp16" below). Sidecar **2,475,083,776 B** (fp16: 3,913,271,296).
  The weights it replaces are matched by value against v4_preview (min cos 0.987), so the HF fp16
  export was made from an earlier student. Output: `output/webgpu_int8/onnx_dyn_int8_f32/`
  (`dit_int8_mm32*`). The first upload, `dit_int8_dyn` with fp16 MatMulNBits
  (`onnx_dyn_int8/`, sidecar 2,372,323,328 B), is still on HF.

### Checks

Dynamic against per-bucket graphs, same weights, ORT CPU (`src/dit_onnx_dyn_eval.py`):

| | 624x416 | 384x688 | 512 (separate export) |
|---|---|---|---|
| DiT `noise_pred` rel L2 | 1.2e-5 | 1.5e-5 | 1.6e-5 |

The VAE matches all five buckets at max |Δ| 0.00098, one fp16 ulp near 1 (rel 2.1e-4). Neither
is bit-exact: ORT fuses differently once shapes are symbolic. At grids no bucket has (64x96 and 96x64)
both run and stay finite at the buckets' output scale.

Against the torch DiT in fp32 (`src/dit_ref_dump.py`: 8 cached images per grid at σ 1 / 0.9 /
0.75 / 0.5, own captions), relative L2 of `noise_pred`, mean over σ; ORT CUDA EP
(`src/dit_onnx_eval.py --ep cuda`):

| | 52x78 (bucket) | 64x96 (no bucket) |
|---|---|---|
| torch, bf16 autocast | 0.0249 | 0.0248 |
| dynamic, fp16, v4 weights | 0.0089 | 0.0087 |
| **dynamic, int8 (shipped)** | **0.0189** | **0.0160** |

Per σ the int8 set sits below bf16 at every step (52x78: 0.0090 / 0.0168 / 0.0248 / 0.0250 against
0.0114 / 0.0221 / 0.0317 / 0.0344). CUDA's MatMulNBits dequantises the weight to fp16, which is the
browser's non-DP4A path. The numbers are from the fp16-MatMulNBits set; on float activations
the fp32 set gives the same `noise_pred` (0.0068 against 0.0069 on one sample, ORT CPU).

### DP4A and fp16

The DP4A path is not measured in a browser end to end. A user reported the fp16-MatMulNBits
set as broken on it and fine on the dequant path. ORT 1.30.0's shaders, for an op of float
type T:

- quantize: `scale_a = T(absmax / 127)` per token per 128-wide K block.
- matmul, per 32-wide K chunk: `scale = T(scale_a * scale_b)`, then
  `y += T(f32(int_dot) * f32(scale))`.

`scale_a * scale_b` is 1e-5 to 1e-10 on this DiT. fp16's smallest normal is 6.1e-5 and its
subnormal step 6e-8, so in fp16 every product is subnormal, and on the linears carrying the
r16 fold (scales ×1/16) most round to zero. The dequant path computes `a * (q * scale_b)` and
never forms the product.

`src/dp4a_emu.py` replays that arithmetic on CPU per linear, on inputs tapped from the ONNX
set running on float activations (one 52x78 sample at σ 1, 128 tokens per linear). Relative
L2 against the same int8 weights on float activations, median over the 28 blocks (max in
brackets):

| Linear | r16 fold | W8A8, exact arithmetic | DP4A in fp16 | DP4A in fp32 (shipped) |
|---|---|---|---|---|
| self_attn q / k / v | 1 | 0.004–0.005 | 0.004–0.006 (0.007) | 0.004–0.005 (0.006) |
| cross_attn.q_proj | 1 | 0.005 | 0.005 (0.006) | 0.005 (0.006) |
| mlp.layer1 | 1 | 0.003 | 0.004 (0.010) | 0.003 (0.004) |
| self_attn.output_proj | 1/16 | 0.006 | 0.07 (0.81) | 0.006 (0.008) |
| mlp.layer2 | 1/16 | 0.005 | 0.17 (0.99) | 0.005 (0.007) |
| cross_attn.output_proj | 1/16 | 0.003 | 0.99 (1.31) | 0.003 (0.006) |

In fp16, `cross_attn.output_proj` has every scale product at zero in all 28 blocks. The replay
matches a WebGPU run: on the single-op probe (`src/dp4a_probe.py`, NVIDIA, GPTQ scales without
the fold) it gives 8.11e-3 where the browser gave 8.12e-3. That probe passed because it left
the fold out.

The text encoder runs the same shaders in fp16. It has no fold and its weight scales are
3.5e-4 to 7.9e-4; its activations were not replayed.

## Files

- `src/te_convrot_gptq.py`: calibration and the variant sweep. Writes
  `output/webgpu_int8/te_int8.<variant>.safetensors` and `.report.json`.
- `src/te_onnx_int8.py`: rewrites the HF `text_encoder_fp16.onnx` with that file. Linears are
  matched by node order (q k v o gate up down per layer) and checked against the safetensors.
- `src/te_ref_dump.py` and `src/te_onnx_eval.py`: the ORT CPU check.
- `src/dit_convrot_gptq.py`: DiT calibration (daemon). `src/dit_onnx_dynamic.py`,
  `src/vae_onnx_dynamic.py`, `src/dit_onnx_int8.py`: the graph rewrites above.
- `src/dit_ref_dump.py` (daemon, `--hw`, `--compute bf16`), `src/dit_onnx_dyn_eval.py`,
  `src/dit_onnx_eval.py`: the checks.
- `src/dp4a_probe.py`, `src/dp4a_serve.py`, `src/dp4a_probe_web/`: single-op MatMulNBits probe
  on the WebGPU EP. `src/dp4a_emu.py`: the CPU replay of the DP4A arithmetic.
- `yaar_request.md`: the changes needed in yaar to load the files (text encoder; DiT and VAE;
  the fp32-MatMulNBits DiT).

Local copies of the outputs are in `output/webgpu_int8/`.
