# What makes the MOSS training step fast

The default plan runs 1.05 s/step at a 12.88 GB peak on a 16 GB card, against 1.59 s/step at
12.92 GB for the bf16 plan (`hews_layout`, rank 32, every run on the code as it stands).
Numerics stay close to bf16's or better: [int8.md](int8.md#accuracy).

Every trick below rests on one fact: a LoRA freezes the base. Nothing computes the base
weights' gradient, so the base only has to answer two products per linear, and the backward
visits the blocks in a known order.

| trick | effect | code |
|---|---|---|
| [int8 ConvRot GEMMs, both directions](#int8-convrot-in-both-directions) | RTN alone: 1.57x bf16 on the same plan | `library/int8/linear.py` |
| [`wq_t` streamed from the host](#backward-copy-on-the-host) | 4.2 GB off the peak, at most 0.03 s/step | `library/int8/stream.py` |
| [Custom ops priced for the partitioner](#custom-ops-in-the-partitioners-knapsack) | FA2 forward 84 → 42 calls, 207 → 104 ms | `library/runtime/dynamo.py` |
| [`Muon` batched by shape](#muon-batched-by-shape) | 154 → 15 ms per step | `training/muon.py` |
| [EMA shadow on the card](#ema-shadow-on-the-card) | 0.065 s/step for 0.27 GB | `training/ema.py` |
| [Block swap over the int8 copies](#block-swap-over-the-int8-copies) | 2.1 GB off the peak, about 0.01 s/step | `library/int8/linear.py` |

Each row is its own before/after on the plan of its day, so the effects do not add up.
Paths are under `library/moss/`.

## int8 ConvRot in both directions

A frozen linear needs the forward `Y = X Wᵀ` and the input gradient `dX = dY W`. Both run
through `comfy_kitchen.int8_linear`: a Hadamard rotation per 256 columns, per-token int8
activations, per-row int8 weights, int32 accumulation. The kernel runs 2.6x bf16 in either
direction, and int32 accumulation is not throttled the way fp32 accumulation is on this GPU
(fp8 reaches only 1.9x).

- **Two int8 copies.** `wq` holds `W` rotated along K, for the forward. `wq_t` holds `Wᵀ`
  rotated along N, for the backward, so the online rotation acts on `dY`, where the gradient
  outliers sit. Together they weigh one bf16 weight, which is why `full` saves time and no
  VRAM.
- **One opaque node per linear.** `torch.library.custom_op` with `register_fake` and
  `register_autograd`: `torch.compile` traces through the op, and the backward is a call to
  the same kernel on `wq_t`. Inputs are not saved for the base; only LoRA saves its own.
- **`__class__` swap in place** (`convert_`), so LoRA modules holding the original object
  keep working. A converted linear keeps an empty bf16 `weight` buffer, because the block
  forward reads its dtype.
- **Cast outside the op.** `x.to(bf16)` sits in `forward`, where Inductor fuses it into the
  producer. Inside the op it ran as its own copy kernel, about 3.5 ms per step.

Only the forward copy comes from the GPTQ file. `wq_t` stays round to nearest: `fwd` and
`full` match to ±0.01 in prediction error, and GPTQ fixes the late-block gradient through the
forward copy alone.

## Backward copy on the host

The backward walks blocks 41 → 0 and reads each `wq_t` once, so `wq_t` does not have to live
on the card. `WqtStream` keeps every block's copy in a pinned host buffer and two block-sized
slots on the card: the block the backward is in, and the one below it, loading on a worker
thread through its own CUDA stream.

- **The op fetches, so there are no hooks.** `dx_stream` asks `get(sid)` for its copy. The
  weights are not inputs of the compiled graph.
- **`sid` is a tensor.** As a Python int it would be a graph constant, and the 42 blocks would
  need 42 graphs in place of one. It stays on the CPU and is read by the op.
- **Free the pages it leaves behind.** Moving `wq_t` off is not enough: what stays on the card
  (`wq`, `s`, `s_t`) sat between the dropped copies and kept their pages pinned. Free VRAM
  before training read 6.39 GB instead of 8.64 until `stream_targets` moved those off and back.
- **Out of order still works.** A block that was not prefetched is copied in on the spot, at
  the cost of the wait. Block 41 is the one cold block per step.

The peak drops 4.2 GB (13.58 → 9.36 GB, checkpointing), for 4.4 GiB of pinned host memory.
dX is bit-identical to `full` in `test_stream_dx_is_full_dx_and_prefetches`. PCIe is not the
limit: during a 790 ms backward the GPU idled about 4 ms while 170 ms of copies ran beside it.

## Custom ops in the partitioner's knapsack

Below budget 1, Inductor's min-cut partitioner decides what the backward saves and what it
recomputes. It only prices nodes the aggressive cut bans from recompute, by their flops, and an
op with no flop formula counts as 1. So FA2 and the int8 linears were recomputed at every
budget, and raising the budget bought nothing. Two changes make them candidates at their real
cost:

- `functorch_config.is_non_builtin_to_include = True`, in `set_activation_memory_budget`.
- `register_flop_formula` on `flash_attn::_flash_attn_forward` and the int8 ops, calibrated on
  the kernels: `2·B·H·D·L_q·L_k` for FA2 (2.46 ms), and `0.5·M·N·K` for int8 (0.42 ms against 0.87 ms
  for a bf16 3072² GEMM).

The saved set stays the same size (209 → 200 MB per block at budget 0.2) and now holds FA2's
`o` and `lse`, so FA2's forward runs once per block.

Budget 0.1 is the point where the saved set holds that output. At 0.05 it no longer fits and
the recompute returns (1.16 s/step). Past 0.1 each 0.1 buys 0.01–0.02 s/step for about 1 GB,
and 0.3 is the default: the fastest budget that has run on 16 GB on both sets tried
([table](int8.md#speed-and-vram)).

The compile cache signature carries `flops=fa2,int8`, so graphs partitioned before the formulas
are not reused.

## Muon batched by shape

`torch.optim.Muon` loops over the 504 matrices in Python, about 34 kernels each: 17k of the
step's 23k kernels, 154 ms for 40 ms of GPU time. The launches were the cost. A LoRA has four
matrix shapes at rank 32, so `BatchedMuon` stacks them and runs one Newton-Schulz per shape:
14.7 ms. Updates match torch within bf16 rounding, and the momentum buffers match exactly
(`test_batched_muon_matches_torch`).

## EMA shadow on the card

`ema_step_card` lerps the shadow on the card instead of copying the adapter to the host each
update. 0.065 s/step (4.8%) for 0.27 GB, in paired runs. The lerp matches the CPU shadow's to
1e-6 in the unit test. The bf16 plan turns it off for lack of room.

## Block swap over the int8 copies

The block swapper moves every `.weight` under a block. `wq`, and `wq_t` in `full`, sit in
`HeldCopy` children under that name, so the existing swapper moves them unchanged. Under
`stream` a block is about 0.11 GB against bf16's 0.21 GB. Swap 20 takes 2.1 GB off the peak for 0.01–0.02
s/step. It is the VRAM lever on this plan: the step stays GPU-bound, so copies hide behind it.

## What is left

Kernels take 1054 of the 1059 ms step, so Python and launches no longer matter. Attention is
387 ms and the int8 GEMMs 316 ms ([split](../../../project/int8_train/reports/moss_step_split_int8.md)).
A further gain has to come from a kernel. No Triton int8 attention arm has passed the accuracy gate
([reports](../../../project/int8_train/reports.md)).

## Carrying these elsewhere

- **Any frozen-base adapter training:** the int8 op, the backward copy on the host and the
  in-place class swap. A swapper that moves only `module.weight` needs the `HeldCopy` layout,
  and Anima's offloader is why `project/int8_train`'s Anima runs refuse swap. The kernel loses
  on short sequences: at M = 512 int8 is slower than bf16, so Anima keeps cross-attn `kv_proj`
  in bf16.
- **Anything compiled with custom ops under an activation budget:** check that the ops are in
  the knapsack. Without a flop formula and `is_non_builtin_to_include` the budget cannot save
  them, and it fails silently, as a budget that buys nothing.
- **Any trainer that keeps per-parameter loops in Python:** count kernels before optimizing the
  GPU work. `Muon.step` ran 17k of the step's 23k kernels for 40 ms of GPU time.
