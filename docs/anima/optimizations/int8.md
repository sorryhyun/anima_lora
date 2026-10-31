# int8 base for Anima

The frozen DiT's large linears run as ConvRot int8 GEMMs instead of bf16: six per block
(`self_attn.{qkv_proj,output_proj}`, `cross_attn.{q_proj,output_proj}`,
`mlp.{layer1,layer2}`), 168 on the 28-block base. Cross-attention `k` / `v`, adaLN,
attention, norms, the residual stream and the LoRA's own GEMMs stay in float.

Opt-in (`int8 = "off"` in `configs/base.toml`). The linears are `library/int8/`, shared with
MOSS-Image; Anima's targets and conversions are `library/anima/int8.py`.

## Running it

Train through the usual targets, with `--int8` on the CLI or in a method TOML:

    make lora ARGS="--int8 full"

| `--int8` | forward | input gradient `dX` | base weight |
|---|---|---|---|
| `off` | bf16 | bf16 | kept |
| `fwd` | int8 | bf16 | kept |
| `full` | int8 | int8, second copy of `Wᵀ` | dropped |

Generate with `inference.py --int8`, or `INT8=1` on any `make test-*`. LoRAs then attach as
P-GRAFT hooks whatever `--pgraft` says, because a static merge would fold them into the bf16
weight the conversion drops. The base runs `int8(x, W_q) + B·A·x`, the function the trainer
trained. Each LoRA's `ss_int8_base` / `ss_int8_base_id` are logged against the base in use.

## Forward copies

In order:

1. `--int8_file <path>`.
2. `<dit>_int8_gptq.safetensors` beside the DiT. On the catalog's `anima-base-v1.0` that is
   `models/diffusion_models/anima-base-v1.0_int8_gptq.safetensors`, row `anima_dit_int8`,
   downloaded on first use (2.37 GiB) or via `make download-model ARGS=anima-dit-int8`.
3. Round to nearest, also forced by `--int8_file rtn`.

The GPTQ file is made by `project/int8_train/src/dit_gptq.py` and published as
[sorryhyun/anima-base-v1.0-int8](https://huggingface.co/sorryhyun/anima-base-v1.0-int8). A
fingerprint refuses a file made from another checkpoint, so the 40-block Anima-2.9B needs its own
file or round to nearest.

## Limits

- `--blocks_to_swap > 0` is refused in training and generation. It has not been validated on
  Anima's offloader, so the `low_vram` preset does not run on int8.
- Training takes `LoRANetwork` only, and only method `lora` has been measured. Anything else
  (`EasyControl`, the bespoke turbo and soup loops) trains on bf16 or is refused.
- `--base_weights` with forward copies is refused: they were calibrated on the unmerged weights.
  `--int8_file rtn` quantizes the merged weights instead.
- `--lora_cutoff_step` is refused in generation.
- MOSS's `stream` mode, which moves `wq_t` to the host, is not wired for Anima.

## Measurements

All from `project/int8_train/reports/`, 2026-10-30, on a 5070 Ti.

| | bf16 | int8 `full` |
|---|---|---|
| `ama mitsuki`, 848 steps, method `lora` | 593 ms/step | 433 ms/step (1.37x) |
| final `avr_loss` | 0.0737 | 0.0737 |

Samples from the two LoRAs, both generated on the bf16 base, were of the same quality. The GPTQ
base cuts per-prediction error by about 10% and CFG velocity error by 2–7% against round to
nearest. Swapping the bf16 base for the int8 base moves generated pixels by about 4/255.

**The accuracy figures predate the fp32 kernel input.** `comfy_kitchen` 0.2.40 rounds to bf16
around its activation division on a bf16 input ([comfy-kitchen#289](https://github.com/Comfy-Org/comfy-kitchen/issues/289)),
so `Int8TrainLinear.forward` passes fp32. On MOSS's fp32 stream that cut prediction error 5–10%.
Anima's stream is bf16, and on MOSS's bf16 stream the gradient p90 of blocks 7–27 rose
([moss_act_scale.md](../../../project/int8_train/reports/moss_act_scale.md)). The accuracy effect
on Anima has not been measured. The same 848-step run with the fp32 input, 2026-10-31, took
424 ms/step including the first compile and ended at `avr_loss` 0.0739.
