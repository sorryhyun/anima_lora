# int8 base for MOSS-Image

The 252 block linears (`layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}`) of the frozen DiT run
as ConvRot int8 GEMMs instead of bf16. Attention, norms, modulation, the residual stream
and the LoRA's own GEMMs stay in float. Each block's `q`, `k` and `v` run as one GEMM with their
rows concatenated, `block.qkv`, which saves about 24 ms of a training step
([report](../../../project/int8_train/reports/qkv_fuse.md)).

**Status (2026-10-31): the default training plan. A t2i LoRA has completed on the GPTQ base
(ama mitsuki, 848 steps, 14.6 min, 1.08 s/step, 10.95 GB), and a swap_text i2i LoRA
(3,000 steps, [i2i.md](../methods/i2i.md)).**
Generation on the file takes 6.3 GiB against 10.7 GiB at 1024². In training, `--int8 full`
gains speed and saves no VRAM; `--int8 stream` also saves about 4 GB ([VRAM](#vram)).

Training is `TrainRequest.int8`, generation `GenerateRequest.int8`, code in
`library/moss/dit_int8/` and, shared with Anima, `library/int8/`. The GPTQ build stays in
`project/int8_train/src/`. The GPTQ file is the catalog row `moss_dit_int8`
([`sorryhyun/MOSS-Image-Base-int8`](https://huggingface.co/sorryhyun/MOSS-Image-Base-int8)).
Base only; SFT has no file and trains on round-to-nearest weights.

## Running it

Train: the defaults are `--int8 stream --ema_card --stream fp32 --blocks_to_swap 0
--activation_memory_budget 0.3`. How the step gets there:
[training_step.md](training_step.md). The fp32 stream:
[methods/fp32_stream.md](../methods/fp32_stream.md).

    make daemon-run ARGS="--stall-timeout 900 scripts/moss/train.py --cache post_image_dataset/moss/<set>"

The file downloads to `models/moss_image/moss_image_base_int8_gptq.safetensors` on first
use (5.04 GiB: 4.43 GiB int8, 0.60 GiB bf16), or via `make download-model
ARGS=moss-dit-int8`. A fingerprint ties it to the DiT it was made from, and both entry
points refuse any other. Rebuild it (64.6 min, peak 14.1 GB; 233 images across t2i and i2i
sets) with:

    make daemon-run ARGS="--stall-timeout 1800 project/int8_train/src/dit_gptq_moss.py"

Generate (the GUI's `(int8)` items in the `model` dropdown set the same flags):

    make daemon-run ARGS="scripts/moss/generate.py --int8 on --lora <path> --stream fp32 \
        --out_dir output/tests/int8"

- `--int8_file rtn`: round-to-nearest (RTN) weights, quantized at load. Empty is the
  catalog's file on Base, RTN on SFT.
- Both refuse `algo loha` (it builds `W + ΔW`). `lora` and `lokr` run.
  `configs/moss/lora_bf16.toml` is the plan for it.
- Training takes `--blocks_to_swap`: `wq` (and `wq_t` in `full`) sit in `HeldCopy` children,
  whose `.weight` the offloader moves. A swapped block is about 0.11 GB against bf16's 0.21
  GB. Generation has no swap run behind it and refuses `--blocks_to_swap`.
- The i2i configs stay on bf16 (`int8 = "off"`, swap 38); one swap_text run has finished on
  the int8 base ([i2i.md](../methods/i2i.md)). A six-step fit on sanitize's largest sample
  (4150 image tokens, full-size reference) OOMs at swap 0 / budget 0.3 and peaks at 12.33 GB
  at swap 20 / budget 0.1, the GUI's i2i `(int8)` items.
- A LoRA trained this way carries `int8_base` and `int8_base_id`. A bf16-trained LoRA
  loads on the int8 base: the sanitize LoRA removed the page's text there.

## Speed and VRAM

The plan today, `hews_layout`, `stream`, `--ema_card`, one run per row:

| plan | s/step | peak |
|---|---|---|
| swap 0, budget 0.3 (default) | 1.05 | 12.88 GB |
| swap 0, budget 0.2 | 1.06 | 11.83 GB |
| swap 0, budget 0.1 | 1.07 | 10.78 GB |
| swap 0, budget 0.05 | 1.16 | 9.98 GB |
| swap 20, budget 0.1 | 1.08 | 8.67 GB |
| swap 0, checkpointing | 1.24 | 9.61 GB |
| bf16 plan (`lora_bf16.toml`) | 1.59 | 12.92 GB |

`ama_mitsuki_layout`, same plan, median of steps 11–40, one run per arm:

| budget | s/step | peak |
|---|---|---|
| 0.1 | 1.09 | 10.95 GB |
| 0.2 | 1.07 | 11.92 GB |
| 0.3 | 1.05 | 12.98 GB |
| 0.4 | OOM at step 3 | 14.39 GB allocated |

Two changes took the step from 1.30 s: `BatchedMuon` and saving FA2's output instead of
recomputing it ([training_step.md](training_step.md)).

The training figures below predate them, and the choices under [VRAM](#vram) rest on them. Full tables:
[`moss_scoping.md`](../../../project/int8_train/reports/moss_scoping.md) and
[`moss_wqt_stream.md`](../../../project/int8_train/reports/moss_wqt_stream.md).

- **int8 against bf16** (`channel_(caststation)_layout`, swap 0, checkpointing): bf16 2.26–2.34
  s/step, int8 RTN 1.44–1.49, 11.96 GB on both. RTN is 1.57x bf16 on the same plan and 1.28x
  the bf16 default of the day (swap 36, budget 0.5: 1.83–1.91 s/step, 13.46 GB). That run was
  stopped after 70 of 240 steps.
- **`full` against `stream`** (GPTQ, fp32 stream, swap 0, `hews_layout`, 30 steps, median of
  steps 11–30, one run per arm). `full`: 1.42 s/step at 13.58 GB with checkpointing, 1.33 at
  13.95 GB at budget 0.05, OOM at 0.1. `stream`: 1.45 at 9.36 GB with checkpointing, 1.33–1.37
  at 10.73–13.83 GB over budgets 0.1–0.4, OOM at 0.5.
- **`--ema_card`** (`stream`, budget 0.2, 60 steps, two paired runs): 1.30 s/step at 12.04 GB
  against the CPU shadow's 1.37 at 11.77 GB.
- **Block swap**: swap 0 at budget 0.2 ran 1.30 s/step at 11.99 GB, swap 20 at 0.2 ran 1.32 at
  9.88 GB, swap 20 at 0.5 ran 1.31 at 14.41 GB. Step losses at swap 20 stay within 3e-4 of swap
  0 at either budget, the size of the gap between two swap-0 runs. In eager, loss and dX match
  the resident stack (`test_swapped_dx_is_resident_dx`).

Generation, one render per arm (ER-SDE 50 steps + Spectrum, CFG 3, seed 7, compiled, all
blocks resident):

| job | arm | denoise | peak |
|---|---|---|---|
| t2i 1024² | bf16 | 21.6 s | 10.7 GiB |
| t2i 1024² | bf16, fp32 stream | 21.9 s | 12.3 GiB |
| t2i 1024² | int8 RTN | 15.0 s | 6.3 GiB |
| t2i 1024² | int8 GPTQ | 11.3 s | 6.3 GiB |
| t2i 1024² | int8 GPTQ, fp32 stream | 15.1 s | 7.8 GiB |
| i2i 1216×864, full-size reference | bf16 | 29.1 s | 13.0 GiB |
| i2i 〃 | int8 GPTQ | 27.7 s | 8.6 GiB |
| i2i 〃 | int8 GPTQ, fp32 stream | 27.7 s | 11.2 GiB |

RTN and GPTQ run the same kernels, so their 3.7 s gap is run-to-run noise: read int8
generation as 11–15 s. The i2i speed gain is small and was not profiled. A full-size
reference on an fp32 stream fits resident on int8; on bf16 it OOMs and needs
`blocks_to_swap 10`.

### VRAM

Generation drops about 4.4 GiB because mode `infer` keeps only the forward copy `wq`
(4.43 GiB against ~8.9 GiB of bf16 weights). Mode `full` also holds `wq_t` for the
backward, and two int8 copies are the bytes of one bf16 weight, so its peak matches bf16's
on the same plan (11.96 GB on both at swap 0). The fp32 stream adds about 1.5 GB on top.
Mode `fwd` is worse: it keeps `wq` and the bf16 weight.

Mode `stream` is `full` with `wq_t` in pinned host memory (4.4 GiB): at equal plans the peak
drops 4.2 GB, for at most 0.03 s/step more
([mechanism](training_step.md#backward-copy-on-the-host)).

**The budget needs 0.1 to hold FA2's output; past that each 0.1 costs about 1 GB for
0.01–0.02 s/step.** The default is 0.3, the fastest budget that ran on 16 GB on both sets
tried, 2–4% faster than 0.1. Below 16 GB run 0.1. Budget 0.4 is not safe on 16 GB: it ran
on `hews_layout` before FA2 joined the saved set and OOMs on `ama_mitsuki_layout` now. 0.5 does
not fit at swap 0 even with `stream`.

`--ema_card` and swap 20 are the other levers:
[EMA](training_step.md#ema-shadow-on-the-card),
[swap](training_step.md#block-swap-over-the-int8-copies). The bf16 plan turns `--ema_card`
off, since it leaves only 0.85 GB free on 16 GB.

## Accuracy

Prediction error, `‖v − v_fp32‖ / ‖v_fp32‖` in %, on images outside calibration. t2i: six
images, 30 for the marked row. i2i: 12 `i2i_sanitize_idt` pairs, centre-cropped to 48×48
because the full size OOMs in the fp32 reference.

| | σ | bf16 | int8 RTN | int8 GPTQ | RTN, fp32 stream | GPTQ, fp32 stream | bf16, fp32 stream |
|---|---|---|---|---|---|---|---|
| t2i | 0.1 | 4.81 | 6.99 | 5.93 | 5.31 | **3.85** | 1.64 |
| t2i | 0.5 | 1.93 | 2.82 | 2.43 | 2.24 | **1.71** | 0.81 |
| t2i | 0.9 | 2.08 | 2.58 | 2.30 | 1.69 | **1.34** | 0.59 |
| t2i, 30 images | 0.5 | 1.92 | 2.75 | 2.38 | 2.16 | **1.65** | – |
| i2i | 0.1 | 3.61 | 5.36 | 4.50 | 4.52 | **3.52** | 2.13 |
| i2i | 0.5 | 2.30 | 3.32 | 2.84 | 2.78 | **2.17** | 1.34 |
| i2i | 0.9 | 2.05 | 2.44 | 2.24 | 1.53 | **1.19** | 0.78 |

- GPTQ cuts int8's excess over bf16, in quadrature, by 28–35% (t2i) and 31–32% (i2i).
- Int8 GPTQ on an fp32 stream sits below today's bf16 at every σ tested. bf16 on an fp32
  stream stays well below it.

LoRA gradient, each image's distance from the fp32 gradient over bf16's, per group of
seven blocks, median over images [p90], σ = 0.5:

| arm | 0–6 | 7–13 | 14–20 | 21–27 | 28–34 | 35–41 |
|---|---|---|---|---|---|---|
| t2i (30), RTN | 1.07 [1.89] | 1.04 [2.07] | 1.04 [2.06] | 1.03 [1.92] | 1.56 [1.64] | 2.59 [2.78] |
| t2i (30), GPTQ | 1.04 [2.13] | 1.10 [2.08] | 1.08 [2.26] | 1.05 [2.36] | 1.23 [1.42] | 1.20 [1.29] |
| t2i (30), GPTQ, fp32 stream | 1.01 [1.48] | 1.05 [1.57] | 1.07 [1.79] | 0.98 [1.69] | 1.12 [1.38] | 0.92 [1.04] |
| i2i (12), RTN | 1.02 [4.28] | 1.03 [3.09] | 1.41 [3.36] | 1.41 [3.16] | 1.47 [2.43] | 1.65 [2.73] |
| i2i (12), GPTQ | 1.00 [1.56] | 0.98 [1.72] | 0.98 [1.72] | 1.07 [1.62] | 1.06 [1.65] | 1.00 [1.88] |

GPTQ fixes the late blocks, where bf16 is accurate and int8's excess is the weight term.
Blocks 0–27 stay at bf16's level with either: bf16's own gradient is already far from fp32
there ([bf16_residual_stream.md](../findings/bf16_residual_stream.md)).

Generation: all arms render cleanly. On the i2i page, mean |Δ| from bf16 is 0.73/255
(GPTQ) and 0.68/255 (with an fp32 stream). One t2i seed is chaotic (12.4–13.9/255 against
11.7 between bf16 and its own fp32 stream), so it orders nothing.

The tables above and the speed tables were measured on a bf16 kernel input.

## `comfy_kitchen` rounds activations to the input dtype

On a bf16 input, `int8_linear`'s CUDA quantizer (0.2.40, `quant_div_to_float` and
`quant_div_float_to_float` in `ops/int8_linear.cu`) adds three bf16 roundings. The rotation
itself runs in fp32.

- The rotated value is rounded to bf16 before the division.
- The scale is rounded to bf16 for the division, while the dequant multiplies by the fp32
  scale. Each token keeps a gain error of up to about 0.4%.
- The quotient is rounded to bf16 before `nearbyintf`, so values of 64 and above are on a 0.5
  step when they reach the int8 rounding.

An fp32 input skips all three. `Int8TrainLinear.forward` passes one in training and generation;
the backward's `dY` stays bf16, and `wq` is unchanged, so `int8_id` and LoRAs are not affected.

Prediction error on the fp32 stream, %, GPTQ, the same held-out images and seeds, 18 of 18
(image, σ) pairs better on the six-image rows:

| σ | bf16 input | fp32 input | change |
|---|---|---|---|
| 0.1 | 3.849 | 3.637 | −5.5% |
| 0.5 | 1.711 | 1.618 | −5.4% |
| 0.5, 30 images | 1.654 | 1.568 | −5.2% |
| 0.9 | 1.336 | 1.201 | −10.1% |

LoRA gradient on the fp32 stream, as in the table above, σ = 0.5, 30 images:

| input | 0–6 | 7–13 | 14–20 | 21–27 | 28–34 | 35–41 |
|---|---|---|---|---|---|---|
| bf16 | 1.01 [1.48] | 1.05 [1.57] | 1.07 [1.79] | 0.98 [1.69] | 1.12 [1.38] | 0.92 [1.04] |
| fp32 | 1.03 [1.37] | 1.04 [1.40] | 1.02 [1.28] | 0.97 [1.23] | 1.07 [1.38] | 0.89 [0.97] |

The cost is 1.05 → 1.10 s/step (`ama_mitsuki_layout`, budget 0.3, median of steps 11–40, one
run per arm), peak unchanged at 12.98 GB. The same weights through ComfyUI, the MOSS Toolkit and
the int8 TE still reach the kernel in bf16; those paths were not scored. i2i was not scored.
Numbers: [`moss_act_scale.md`](../../../project/int8_train/reports/moss_act_scale.md).

Upstream: [comfy-kitchen#289](https://github.com/Comfy-Org/comfy-kitchen/issues/289), with a fix
on [`sorryhyun/comfy-kitchen`](https://github.com/sorryhyun/comfy-kitchen/tree/int8-activation-fp32-division).
It divides in fp32, so a bf16 input gives the fp32 input's output, the same prediction error,
and bf16's speed. Once a release carries it, `Int8TrainLinear.forward` can go
back to bf16 and drop the 0.05 s/step.

## Single-token gradient outliers

A few (image, arm) pairs carry one token whose gradient dominates. `src/grad_spike.py`
traced the largest, `channel4` at σ = 0.5:

- That token's `dY` grows about 1.3x per block going back from block ~25 to 0, and holds
  up to 98% of the energy, entering through `q`.
- Every arm has a hot token, fp32 included, and which token it is changes from arm to arm.
  Its share of `‖dY_o‖²` is 0.11–0.24 in the fp32 and fp32-stream arms, up to 0.35 in bf16
  and up to 0.93 in `gptq_w8a16`.
- It is not `_RMSNorm`'s ε: the smallest per-token input RMS is 0.15, a gain of about 7.

Over 30 images the medians tie and the p90s are close, so this is a tail and not a GPTQ
cost. An fp32 stream lowers the GPTQ p90s in t2i but not in i2i at σ = 0.5, on 12 images.

The gradient norm also moves between runs of one seed, in bf16 as well as int8: 0.077
against 0.378 at step 2 for bf16 on an fp32 stream, eager, with the loss within 1e-5. So
the outliers are not an int8 effect and are not reproducible run to run.

## Records

- [`moss_step_split_int8.md`](../../../project/int8_train/reports/moss_step_split_int8.md):
  phase and kernel split of the step, `BatchedMuon`, FA2 in the budget's knapsack.
- [`moss_act_scale.md`](../../../project/int8_train/reports/moss_act_scale.md): activation
  clipping, the kernel's input dtype. [`moss_fold_vo.md`](../../../project/int8_train/reports/moss_fold_vo.md):
  SmoothQuant fold from `o` into `v`, no gain.
- [`moss_gptq.md`](../../../project/int8_train/reports/moss_gptq.md): GPTQ base, held-out
  error, outliers, trainer smoke test, generation, with job ids.
- [`moss_wqt_stream.md`](../../../project/int8_train/reports/moss_wqt_stream.md): `wq_t`
  streaming, budget sweep, same-seed gradient norms.
- [`moss_grad_error.md`](../../../project/int8_train/reports/moss_grad_error.md): RTN
  gradient gate, bf16 precision arms.
- [`moss_scoping.md`](../../../project/int8_train/reports/moss_scoping.md): GEMM bench, step
  split, swap and budget plans, RTN step time.
- [`project/int8_train/README.md`](../../../project/int8_train/README.md): design of the
  int8 linears, shared with Anima.
