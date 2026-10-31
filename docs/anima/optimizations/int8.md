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
| `stream` | int8 | as `full`, the copy on the host and streamed per block | dropped |

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
- Training takes `LoRANetwork` only, and only method `lora` has been measured. `EasyControl` and
  the soup loop train on bf16 or are refused.
- `make turbo ARGS="--int8 stream"` is wired and has not run a step. It refuses `dmad` and
  `dmad_probe`, and its teacher runs on the int8 base.
- `--base_weights` with forward copies is refused: they were calibrated on the unmerged weights.
  `--int8_file rtn` quantizes the merged weights instead.
- `--lora_cutoff_step` is refused in generation.

## `stream` and the budget

`ama mitsuki`, one run per arm, 2026-10-31, 5070 Ti
([anima_wqt_stream.md](../../../project/int8_train/reports/anima_wqt_stream.md)):

| arm | samples | median | peak |
|---|---|---|---|
| `full`, budget 0.99 | all 106 | 482 ms | 12.38 GB |
| `stream`, budget 0.99 | all 106 | 499 ms | 10.96 GB |
| `stream`, budget 1.0 | all 106 | OOM at step 0 | – |
| `stream`, budget 0.99 | 13 at ~3000 tokens | 333 ms | 8.84 GB |
| `stream`, budget 1.0 | 13 at ~3000 tokens | 333 ms | 12.90 GB |

Keep budget 0.99 on 16 GB. 1.0 saves 1.75x the activations for no speed, and needs about 16.5 GB at
4200 tokens.

Below 16 GB, lower the budget: with block swap refused it is the VRAM lever on this plan. FA2 and
the int8 linears are in the partitioner's knapsack (`set_activation_memory_budget`), so these are
the calls each budget recomputes on purpose. `stream`, one run per arm, median of 23–25 steps over
the set's first 30, peak over those steps, same day; two runs of one partition differed by 20 ms:

| budget | ms/step | peak | FA2 forwards | int8 forwards |
|---|---|---|---|---|
| 0.99 | 427 | 10.87 GB | 56 | 168 |
| 0.4 | 434–439 | 10.00 GB | 56 | 168 |
| 0.3 | 472 | 6.87 GB | 84 | 280 |
| 0.2 | 476–495 | 6.16 GB | 56 | 280 |
| 0.1 | 511 | 5.32 GB | 56 | 336 |

Budgets from about 0.48 to 0.99 are one partition. The 0.4 and 0.1 rows ran before the FA2 formula
read the key length; 0.4 does not reach the knapsack, 0.1 does. No full LoRA has run below 0.99.

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
so `Int8TrainLinear.forward` passes fp32. On Anima that cuts prediction error 6–8%. Anima's stream
is bf16, and an fp32 stream, which it does not have, would cut the int8 base's error a further
21–36%; the gradient effect is mixed
([anima_fp32_stream.md](../../../project/int8_train/reports/anima_fp32_stream.md)). The same
848-step run with the fp32 input, 2026-10-31, took 424 ms/step including the first compile and
ended at `avr_loss` 0.0739.
