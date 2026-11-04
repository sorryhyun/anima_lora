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
2. The DiT itself, when `dit_path` names an int8 checkpoint.
3. `<dit>_int8_gptq.safetensors` beside the DiT. On the catalog's `anima-base-v1.0` that is
   `models/diffusion_models/anima-base-v1.0_int8_gptq.safetensors`, row `anima_dit_int8`,
   downloaded on first use (2.37 GiB) or via `make download-model ARGS=anima-dit-int8`.
4. Round to nearest, also forced by `--int8_file rtn`.

The GPTQ file is a full DiT: block linears int8, everything else bf16. As `dit_path` it needs
`dit_nonquant_path`, the bf16 DiT, for the float weights `wq_t` and `weight_svd` / `grad_svd` init
read; `make turbo` ships both. Without it the loader rebuilds them from the int8 copies, 1.1% from
the true weight, which moves `wq_t` 1.6%. Generation needs neither. A static LoRA merge on a rebuilt
base is refused.

The GPTQ file is made by `project/finished/int8_train/src/dit_gptq.py` and published as
[sorryhyun/anima-base-v1.0-int8](https://huggingface.co/sorryhyun/anima-base-v1.0-int8). A
fingerprint refuses a file made from another checkpoint, so the 40-block Anima-2.9B needs its own
file or round to nearest.

## Limits

- `inference.py` has no block swap on any base, so generation stays resident.
- Training takes `LoRANetwork` only, and only method `lora` has been measured. `EasyControl` and
  the soup loop train on bf16 or are refused.
- `make turbo` ships `int8 = "stream"` and has run 30-step smokes only, no full student. It refuses
  `dmad` and `dmad_probe`, and its teacher runs on the int8 base. 2026-11-01, warm start, budget
  0.99: 6.28 s/step and 11.87 GB peak on the bf16 stream, 6.7–7.0 s/step and 12.39 GB on fp32. The
  bf16 base read 9.65 s/step on the same plan, 2026-08-29.
- `--base_weights` with forward copies is refused: they were calibrated on the unmerged weights.
  `--int8_file rtn` quantizes the merged weights instead.
- `--lora_cutoff_step` is refused in generation.

## `stream` and the budget

`ama mitsuki`, one run per arm, 2026-10-31, 5070 Ti
([anima_wqt_stream.md](../../../project/finished/int8_train/reports/anima_wqt_stream.md)):

| arm | samples | median | peak |
|---|---|---|---|
| `full`, budget 0.99 | all 106 | 482 ms | 12.38 GB |
| `stream`, budget 0.99 | all 106 | 499 ms | 10.96 GB |
| `stream`, budget 1.0 | all 106 | OOM at step 0 | – |
| `stream`, budget 0.99 | 13 at ~3000 tokens | 333 ms | 8.84 GB |
| `stream`, budget 1.0 | 13 at ~3000 tokens | 333 ms | 12.90 GB |

Keep budget 0.99 on 16 GB. 1.0 saves 1.75x the activations for no speed, and needs about 16.5 GB at
4200 tokens.

Below 16 GB, lower the budget first: it takes more off the peak than block swap does. FA2 and
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

## Block swap

`--blocks_to_swap` works in every mode. `wq` and `wq_t` sit in `HeldCopy` children, whose `.weight`
the offloader moves; `stream` keeps `wq_t` on the host either way. Output and `dX` on a swapped
stack equal the resident stack bit for bit, eager and compiled, on both attach orders
(`test_swapped_step_is_the_resident_step`). Turbo's sequence, a student forward whose graph outlives
teacher and fake forwards, matches too, CDM's second student forward included
(`test_turbo_step_on_swap_is_the_resident_step`). Those tests skip without `ANIMA_TEST_GPU=1`.

`ama mitsuki`, 106 steps, budget 0.99, GPTQ forward copies, fp32 stream, one run per arm, 2026-11-01,
5070 Ti:

| arm | median | peak |
|---|---|---|
| `stream`, swap 0 | 451 ms | 11.47 GB |
| `stream`, swap 14 | 475 ms | 10.44 GB |
| `stream`, swap 24 | 524 ms | 9.70 GB |
| `full`, swap 14 | 526 ms | 11.10 GB |
| `fwd`, swap 14 | 667 ms | 11.92 GB |

Turbo, 30 steps, CDM on, GAN off, fp32 stream: 5.8 s/step and 12.36 GB at swap 0, 8.4 s/step and
11.32 GB at swap 14. Every forward there rewinds the swap, so a step moves each swapped block
several times. With the GAN on, 20 steps at swap 14 read 9.6 s/step and 11.35 GB.

A seed draws different samples with swap on: losses across swap and resident runs are not
comparable step by step, and `loss/epoch_average` read 0.0749 against 0.0733 for that reason. A
swapped plan on `full` or `fwd` converts on the host, where round to nearest flips 0–3 entries per
linear by 1 LSB against the card.

## Measurements

All from `project/finished/int8_train/reports/`, 2026-10-30, on a 5070 Ti.

| | bf16 | int8 `full` |
|---|---|---|
| `ama mitsuki`, 848 steps, method `lora` | 593 ms/step | 433 ms/step (1.37x) |
| final `avr_loss` | 0.0737 | 0.0737 |

Samples from the two LoRAs, both generated on the bf16 base, were of the same quality. The GPTQ
base cuts per-prediction error by about 10% and CFG velocity error by 2–7% against round to
nearest. Swapping the bf16 base for the int8 base moves generated pixels by about 4/255.

**The accuracy figures predate the fp32 kernel input.** `comfy_kitchen` 0.2.40 rounds to bf16
around its activation division on a bf16 input ([comfy-kitchen#289](https://github.com/Comfy-Org/comfy-kitchen/issues/289)),
so `Int8TrainLinear.forward` passes fp32. On Anima that cuts prediction error 6–8%. An fp32 stream
([`--stream fp32`](fp32_stream.md)) cuts the int8 base's error a further 21–36% in that report; the
gradient effect is mixed
([anima_fp32_stream.md](../../../project/finished/int8_train/reports/anima_fp32_stream.md)). The same
848-step run with the fp32 input, 2026-10-31, took 424 ms/step including the first compile and
ended at `avr_loss` 0.0739.
