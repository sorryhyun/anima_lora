# int8 base for MOSS-Image

The 252 block linears (`layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}`) of the frozen DiT run
as ConvRot int8 GEMMs instead of bf16. Attention, norms, modulation, the residual stream
and the LoRA's own GEMMs stay in float.

**Status (2026-10-31): the default training plan. One LoRA has completed on the GPTQ base
(ama mitsuki, 848 steps, 14.6 min, 1.08 s/step, 10.95 GB). It has no A/B against a
bf16-trained LoRA, and no multi-seed render A/B exists.**
Generation on the file works and is the better-supported half: 6.3 GiB against 10.7 GiB at
1024². In training, `--int8 full` gains speed and saves no VRAM; `--int8 stream` also saves
about 4 GB ([VRAM](#vram)).

Training is `TrainRequest.int8`, code in `library/moss/dit_int8/`. Generation and the GPTQ
build stay in `project/int8_train/src/`. The GPTQ file is the catalog row `moss_dit_int8`
([`sorryhyun/MOSS-Image-Base-int8`](https://huggingface.co/sorryhyun/MOSS-Image-Base-int8)).
Base only; SFT has no file and trains on round-to-nearest weights.

## Running it

Train: the defaults are `--int8 stream --ema_card --stream fp32 --blocks_to_swap 0
--activation_memory_budget 0.1`.

    make daemon-run ARGS="--stall-timeout 900 scripts/moss/train.py --cache post_image_dataset/moss/<set>"

The file downloads to `models/moss_image/moss_image_base_int8_gptq.safetensors` on first
use (5.04 GiB: 4.43 GiB int8, 0.60 GiB bf16), or via `make download-model
ARGS=moss-dit-int8`. A fingerprint ties it to the DiT it was made from, and both entry
points refuse any other. Rebuild it (64.6 min, peak 14.1 GB; 233 images across t2i and i2i
sets) with:

    make daemon-run ARGS="--stall-timeout 1800 project/int8_train/src/dit_gptq_moss.py"

Generate:

    make daemon-run ARGS="project/int8_train/src/infer_int8_moss.py --int8 on \
        --int8_file models/moss_image/moss_image_base_int8_gptq.safetensors \
        -- --lora <path> --stream fp32 --out_dir output/tests/int8"

- `--int8_file rtn` in training, or `--int8_file` omitted in generation: round-to-nearest
  (RTN) weights, quantized at load.
- Both refuse `algo loha` (it builds `W + ΔW`). `lora` and `lokr` run.
  `configs/moss/lora_bf16.toml` is the plan for it.
- Training takes `--blocks_to_swap`: `wq` (and `wq_t` in `full`) sit in `HeldCopy` children,
  whose `.weight` the offloader moves. A swapped block is about 0.11 GB against bf16's 0.21
  GB. Generation (`infer_int8_moss.py`) still has no swap run behind it.
- The i2i configs stay on bf16 (`int8 = "off"`, swap 38): i2i on the int8 base has run no
  step.
- A LoRA trained this way carries `int8_base` and `int8_base_id`. A bf16-trained LoRA
  loads on the int8 base: the sanitize LoRA removed the page's text there.

## Speed and VRAM

The plan today, `hews_layout`, `stream`, `--ema_card`, one run per row:

| plan | s/step | peak |
|---|---|---|
| swap 0, budget 0.1 (default) | 1.07 | 10.78 GB |
| swap 0, budget 0.2 | 1.06 | 11.83 GB |
| swap 0, budget 0.3 | 1.05 | 12.88 GB |
| swap 0, budget 0.05 | 1.16 | 9.98 GB |
| swap 20, budget 0.1 | 1.08 | 8.67 GB |
| swap 0, checkpointing | 1.24 | 9.61 GB |
| bf16 plan (`lora_bf16.toml`) | 1.59 | 12.92 GB |

The step is GPU-bound: kernels take 1054 of its 1059 ms. Attention is 387 ms, the int8 GEMMs
316 ms ([`moss_step_split_int8.md`](../../../project/int8_train/reports/moss_step_split_int8.md)).
Two changes took it from 1.30 s/step: `BatchedMuon` (154 → 15 ms) and saving FA2's output
instead of recomputing it (84 → 42 forward calls). The partitioner needs
`is_non_builtin_to_include` and flop formulas to see custom ops at all, so every figure
below this section, budget sweeps included, predates them.

Earlier measurements, `channel_(caststation)_layout` unless noted:

| plan | s/step | peak |
|---|---|---|
| bf16, swap 36, budget 0.5 (`lora_bf16.toml`) | 1.83–1.91 | 13.46 GB |
| bf16, swap 0, checkpointing | 2.26–2.34 | 11.96 GB |
| int8 RTN, swap 0, checkpointing | 1.44–1.49 | 11.96 GB |
| int8 GPTQ, fp32 stream, swap 0, checkpointing (`hews_layout`, 12 steps) | 1.47 | 13.52 GB |

RTN is 1.57x bf16 on the same plan and 1.28x the bf16 default. That run was stopped after
70 of 240 steps.

GPTQ, fp32 stream, swap 0, `hews_layout`, 30 steps, median of steps 11–30, one run per arm:

| arm | s/step | peak |
|---|---|---|
| `full`, checkpointing | 1.42 | 13.58 GB |
| `full`, budget 0.05 | 1.33 | 13.95 GB |
| `full`, budget 0.1 | OOM | – |
| `stream`, checkpointing | 1.45 | 9.36 GB |
| `stream`, budget 0.1 | 1.37 | 10.73 GB |
| `stream`, budget 0.2 | 1.35 | 11.77 GB |
| `stream`, budget 0.3 | 1.33 | 12.80 GB |
| `stream`, budget 0.4 | 1.35 | 13.83 GB |
| `stream`, budget 0.5 | OOM | – |
| `stream`, budget 0.2, `--ema_card` (60 steps, two runs) | 1.30 | 12.04 GB |

The same paired runs with the CPU shadow read 1.37 s/step at 11.77 GB.

Block swap, before the changes above: swap 0 at budget 0.2 ran 1.30 s/step at 11.99 GB,
swap 20 at 0.2 ran 1.32 at 9.88 GB, swap 20 at 0.5 ran 1.31 at 14.41 GB.

Step losses at swap 20 stay within 3e-4 of swap 0 at either budget, the size of the gap
between two swap-0 runs. In eager, loss and dX match the resident stack bit for bit
(`test_swapped_dx_is_resident_dx`).

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
on the same plan (11.96 GB in both swap-0 rows). The fp32 stream adds about 1.5 GB on top.
Mode `fwd` is worse: it keeps `wq` and the bf16 weight.

Mode `stream` is `full` with `wq_t` in pinned host memory (4.4 GiB) and two block-sized
slots on the card: the block the backward is in and the one below it, loading on a worker
thread. dX is bit-identical to `full` in the unit test, and the step costs at most 0.03 s
more. At equal plans the peak drops 4.2 GB (13.58 → 9.36 GB with checkpointing).

**Past budget 0.1 the activation budget buys little.** It needs 0.1 to hold FA2's output;
above that the peak climbs about 1 GB per 0.1 for 0.01 s/step. `--ema_card` is the better
use of the room: keeping the EMA shadow on
the card saves 0.065 s/step (4.8%) for 0.27 GB. The bf16 plan turns it off, since it leaves
only 0.85 GB free on 16 GB. Budget 0.5 does not fit on a 16 GB card at swap 0 even with `stream`.

Swap is a VRAM lever here: swap 20 takes 2.1 GB off the peak for 0.01–0.02 s/step.

Running dX on `wq` itself, transposed, would drop `wq_t` with no host copy, but dY could no
longer be rotated along N, which is what handles its outliers. Not tried.

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

The mechanism is unresolved. In training it would arrive as one outlier gradient.

## Not measured

- **A GPTQ-trained LoRA against a bf16-trained one.** The one complete run renders cleanly
  and moves the palette on one prompt and seed. Whether it matches a bf16-trained LoRA, in
  loss or samples, is unknown.
- **Any complete LoRA on the RTN base.** Its only MOSS run stopped at 70 of 240 steps.
  The Anima A/B (848 steps, 1.37x, loss and samples matching) has no MOSS counterpart.
- **Multi-seed render A/B** of the int8 base against bf16.
- **The effect of the outliers on training**: loss, convergence or samples, and how often
  they occur during a run.
- **i2i training on the int8 base.** It has run no recorded step, so whether a full-size
  reference fits under `stream`, and at what swap, is the open question.
- **Block swap beyond swap 20**, `full` mode with swap in a trainer run, and `--blocks_to_swap
  -1` sizing, whose reserve is fit to bf16.
- **i2i error beyond `sanitize` crops**: no `swap_text`, no full-size pairs.
- **Step time as a proper bench.** The GPTQ figures come from 30 steps on one set, one run
  per arm.
- **Budgets above 0.5**, which need more VRAM than a 16 GB card has at swap 0.
- **SFT.**

## Records

- [`moss_step_split_int8.md`](../../../project/int8_train/reports/moss_step_split_int8.md):
  phase and kernel split of the step, `BatchedMuon`, FA2 in the budget's knapsack.
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
