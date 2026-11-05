# MOSS-Image-heal: fp16 headroom, quantization, a LoRA A/B

[KBlueLeaf/MOSS-Image-heal](https://huggingface.co/KBlueLeaf/MOSS-Image-heal) (2026-11-04)
is a full finetune of MOSS-Image-Base's DiT. It ran 50k steps × 64 samples on about 1M
danbooru posts with GBC and tag prompts. Stream losses (write concentration, RMS) and an
outlier decay on the weights brought the residual stream's largest value from 58k to 3k, with
every weight within 64× its tensor's RMS. TE, VAE and pipeline are Base's, and its
`transformer/config.json` is identical to Base's. Measured 2026-11-05 on the 5070 Ti.

**Verdict:** the heal does not help the repo's paths. It is fp16-safe, but our generation and
training run int8 GEMMs with an fp32 stream, and they do not use fp16. GPTQ int8 error is the
same as Base's, and a caststation LoRA trained on it ties Base's. Its prior is different
(warmer, more saturated, knows danbooru artist tags), and the LoRA inherits some of that. The
wiring (`--model heal`) was reverted. Only the GPTQ file is kept.

## Precision probe

Six `_calib_mixed` items near 1 MP at σ 0.3 / 0.5 / 0.7 / 0.9 / 0.5 / 0.7, fixed noise. Each
row's prediction is scored as velocity rel. L2 against the same model in fp32. The probe runs
layer-major, so the fp32 reference holds one block on the card at a time. "fp32 stream" is
`stream_fp32_`: every parameter except the block linears in fp32.

| | bf16 | fp16 | fp32 stream + bf16 GEMM | fp32 stream + fp16 GEMM |
|---|---|---|---|---|
| Base | 2.0e-2 | 2.9e-3 | 7.6e-3 | 1.3e-3 |
| heal | 1.9e-2 | 2.3e-3 | 6.3e-3 | 0.9e-3 |
| stream max \|x\| | Base 61.4k, heal 3.3k | | | |

- **Base also runs in fp16 at 1 MP**, with no non-finite values. Its stream max of 61.4k
  leaves 6% to the fp16 limit, though, and a LoRA, a larger image or an i2i reference can
  cross it. heal has 20× headroom.
- **The bf16 error does not depend on the stream's magnitude.** It is the same on both models
  because bf16 rounding is relative. The fp32 stream still removes about 2/3 of it on both
  ([`bf16_residual_stream.md`](bf16_residual_stream.md)). What remains is GEMM rounding, which
  fp16 GEMMs cut about 6× (3 more mantissa bits).
- In fp32, heal vs Base on our layout_json prompts gives velocity rel. 5.6% and cos 0.997.
  Flow loss is 0.2665 vs 0.2658.

## The shipped quantized files

**`comfyui/moss_image_heal_int8_convrot.safetensors`** holds RTN ConvRot int8 weights (group
256, per-row scale). It carries a `comfy_attention` hint, so ComfyUI also runs
`comfy_kitchen.int8_attention` on it: INT8 Q / K / V and uint8 P, block-Hadamard on Q / K, a
generic kernel that is not model-specific. Velocity rel. error against heal fp32:

| | bf16 | weight-only, bf16 | W8A8, bf16 | W8A8 + int8 attn, bf16 | W8A8, fp16 | W8A8 + int8 attn, fp16 |
|---|---|---|---|---|---|---|
| rel. error | 1.86e-2 | 2.23e-2 | 2.44e-2 | 2.56e-2 | 1.55e-2 | 1.76e-2 |

Weight quantization is most of it, at 0.85–0.95% per layer as the card reports. Int8 attention
adds about 0.1 pt. On fp16 compute (`--fp16-unet`), the full int8 path is more accurate than
plain bf16.

**`comfyui/moss_image_heal_fp16.safetensors`** is the bf16 weights cast to fp16. Its median
per-tensor difference to the bf16 shards is 4e-9, and 99.988% of elements round-trip to the
same bf16 value (the rest are fp16 subnormals). The fp32 forward differs by 1.7e-6. It adds no
precision over casting `transformer/` at load. Both files were deleted after the check.

## GPTQ

`project/finished/int8_train/src/dit_gptq_moss.py --model heal` ran in 63 min on the same
calibration as Base's file. It wrote `models/moss_image/moss_image_heal_int8_gptq.safetensors`
(int8_id `39eb18514775727a`, report beside it). Mean per-linear output error:

| | Base | heal |
|---|---|---|
| W8A8 RTN | 1.28e-4 | 1.27e-4 |
| W8A8 GPTQ | 7.60e-5 | 7.87e-5 |
| W8A16 GPTQ | 7.5e-6 | 1.0e-5 |

Taking out the weight outliers did not make heal easier to quantize.

## caststation LoRA A/B

- **Control:** `caststation_fwdpv`, from 2026-11-03.
- **Treatment:** `caststation_heal`, the same recipe on heal (`configs/moss/lora.toml`, the
  `channel_(caststation)_layout_sure` cache, lr 1e-4, EMA window 0.139, 240 steps, int8
  stream + fp32 stream) with the heal GPTQ base.
- **Renders:** `scripts/moss/generate.py` (int8, 50 steps, CFG 3, prompt tail from the
  LoRA). The three `lora_ab.py` caststation prompts, seeds 1–6.
- **Output:** `output/tests/caststation_heal_ab/` (`sheet_p{0,1,2}_*.png`, columns Base / heal
  / Base+LoRA / heal+LoRA).

| | Base | heal |
|---|---|---|
| last-epoch loss | 0.2852 | 0.2825 |
| s / step | 1.10 | 1.05 |
| peak | 13.19 GB | 13.19 GB |

- **Training:** loss per epoch is paired (same seed and σ draws) and agrees within ±0.003.
  Neither run was unstable, so the heal had nothing to fix here.
- **Without a LoRA:** heal renders warmer and more saturated (orange hair on neru s5), with
  other compositions. On the natural-language prompt it drew a signature, so it likely knows
  the caststation tag from danbooru.
- **With the LoRA:** both pick up the style about equally. heal+LoRA keeps some of the warm
  cast (neru s5) and the signature (sailor s3). Base+LoRA had one bad composition (neru s2,
  phone over a mask) that heal+LoRA does not.

## Not kept

- The `--model heal` wiring: catalog pack `moss-heal`, `MOSS_VARIANT_DIRS`, and
  `forward_file` picking the local GPTQ file. The trainer reads non-block weights from
  `transformer/`, so using the GPTQ file here again means downloading `transformer/` (10 GB)
  and adding the wiring back.
- The probe scripts were one-off.
