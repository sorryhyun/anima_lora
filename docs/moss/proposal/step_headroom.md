# Step headroom around the attention kernels

Status: all four candidates are closed, 2026-11-02. 2 and 3 are built, 1 and 4 failed their gates. The FA2 tile choice and
the i2i config facts were read from the code. Applies to MOSS-Image and Anima unless a section says otherwise.

The block GEMMs, the numerics and t2i VRAM look exhausted (see
[Not worth pursuing](#not-worth-pursuing)). What is left sits in the attention kernels and
in the i2i reference branch. The Anima tensor flow these sections refer to is
[fp32_stream.png](../../anima/structure_images/fp32_stream.png).

## Where the step goes

| | Anima int8 ([status](../../../project/int8_train/status.md)) | MOSS int8 ([split](../../../project/int8_train/reports/moss_step_split_int8.md), [training_step](../optimization/training_step.md)) |
|---|---|---|
| step | 430 ms | 1059 ms |
| attention | 180 ms | 387 ms |
| int8 GEMM | 79 ms | 316 ms |

FA2 on `q [1, 4131, 16, 128]`: forward 1.63 ms, backward 3.93 ms
([attn_bwd.md](../../../project/int8_train/reports/attn_bwd.md)). On MOSS, the partitioner saves
FA2's forward, so it runs once per block.

## Candidates, in run order

| | candidate | line | estimated gain | numerics |
|---|---|---|---|---|
| 1 | FA2 forward tile for sm_120 | both | training 0.7%, MOSS generation about 1.6% | none, failed the 1.15x gate |
| 2 | fp16-accumulate attention backward | both | 4.6–5.5% on a real MOSS step, 5.1% on Anima | error under FA2's, gradient unchanged |
| 3 | one int8 linear for MOSS `proj.q/k/v` | MOSS | about 2%, measured, built | forward bit-identical, real-gradient dX not measured |
| 4 | reference branch under `no_grad` | MOSS i2i | 20% of an eager pass, measured | failed the gradient check |

## 1. FA2 forward tile for sm_120

`run_mha_fwd_hdim128` (`flash_fwd_launch_template.h`) sets
`is_sm8x = cc_major == 8 && cc_minor > 0`. The RTX 5070 Ti reports 12.0, so it takes the
A100 / H100 branch, `Flash_fwd_kernel_traits<128, 128, 64, 4, …>`. The profiled kernel name
confirms this (`step_profile_kernels_full_castout.json`). The sm86 / 89 branch picks
128×32 for non-causal calls, commented as the fastest because 48 KB of shared memory gives
2 CTAs per SM. 128×64 needs 64 KB.

Assumed, not checked: this card has about 100 KB of shared memory per SM, as sm86 / 89 do,
so 128×64 runs one CTA per SM. Forward throughput points the same way (about 86 TFLOPS on
Anima and 98 on MOSS, against an assumed 176 TFLOPS peak for bf16 with fp32 accumulation).
Whether 128×32 is faster on Blackwell is the open question.

Closed: 128×32 runs 1.060x on self-attention and 1.117x on cross-attention at FA2's error, under
the 1.15x gate, which is 3.1 ms of Anima's 430 ms step. CUDNN attention reaches 1.072x on the
forward with a slower backward. The wheel already holds both tiles, so this took no build.
[Finding](../findings/fa2_tile_sm120.md).

## 2. fp16-accumulate attention backward

Measured 2026-11-02: passes the gate with QKᵀ left in bf16, 1.340x the Triton backward and 1.304x
forward + backward at 1.02–1.10x its error. dQ and dK cost at most 1%, fp16acc QKᵀ alone is 2.1x,
and dV needs a fixed P × 256 for block 0. [Report](../../../project/int8_train/reports/attn_bwd_fp16acc.md).

Ported to FA2's backward the same day: a real MOSS step goes from 1071–1082 to 1022 ms at the same
peak and a real Anima step from 426.5 to 404.7 ms, with error at 0.74–0.97x FA2's and MOSS's LoRA
gradient unchanged. Repo jobs on sm_120 run it after `make fa2-sm120`. [Report](../../../project/int8_train/reports/fa2_fp16acc.md).

What follows is the proposal as written.

fp16-accumulate QKᵀ and PV already pass the gradient gate at 1.03–1.40x FA2's error and 1.31x
its forward time ([attn_fwd_grad.md](../../../project/int8_train/reports/attn_fwd_grad.md)).
That forward alone saves about 2.5% of the step. The backward is 2.4x its size.

One arm exists: on an int8 forward, moving dP and dV to fp16 accumulation took the backward
from 4.236 to 3.630 ms, and the error barely moved. S, dQ and dK in fp16 accumulation have
not been measured; `dqk` has no `fp16acc` mode in `int8_attn.py`.

Upper bound: the backward's MMA issue time is about 2.0 ms of 3.93 ms at the assumed 176 TFLOPS
peak, so fp16 accumulation saves at most 1.0 ms per call. With the forward, that is 8–9% of the
step on either line. 4–6% is more likely.

Risks:

- dQ = Σ dS·K and dK = Σ dSᵀ·Q are signed, near-cancelling sums. fp16 accumulator error
  scales with Σ|term|, not |Σ term|. dV is a sum with P ≥ 0, so it does not have this
  problem.
- |dO| is about 1e-7, under fp16's normal range. The kernel needs the bench's per
  (batch, head) scaling, which adds a reduction.
- Forward QKᵀ gained about a third of what PV gained from fp16 accumulation, unexplained.
  One guess was that the tile limits these kernels; 128×32 moved the FA2 forward by only 6%,
  so it explains little of this.

Check: add `out_dtype=tl.float16` to the dQ and dK dots and the backward's S, use the bf16
forward, and run `attn_bwd_bench.py` on the 24 captured calls, about two minutes. Judge it
against the Triton baseline at the [reports.md](../../../project/int8_train/reports.md) gate.
The Triton backward is 0.70x FA2, so a pass gains nothing until a CUDA port exists.

## 3. One int8 linear for MOSS `proj.q/k/v`

Built 2026-11-02: the kernel runs 1.30x on the forward and 1.31x on dX, about 24 ms over 42 blocks, and a real step
gains about 20 ms at the same peak. The forward is bit-identical. dX error on real gradients was not measured.
[Report](../../../project/int8_train/reports/qkv_fuse.md).

What follows is the proposal as written.

MOSS keeps three linears that read the same `y` (`_Block` in `library/moss/model.py`). Anima
already fuses them into `qkv_proj`. One linear means one quantization of `y`, and one read of
it by the LoRA.

Scaled from [gemm_microbench.md](../../../project/int8_train/reports/gemm_microbench.md): the
fused forward is 22% faster and dX 4%, which gives about 12 + 5 + 6 ms, or 2% of the step. The
forward stays bit-identical with row scales, since GPTQ rows concatenate. `wq_t`'s channel
scales and dY's token scales change to the 9216-column shape. Anima measured that case as
free. LoRA file keys stay `q` / `k` / `v`.

Check: `comfy_kitchen.int8_linear` once at 3072→9216 against three times at 3072→3072, 30
seconds. Do this only when `_Block` is being changed anyway.

## 4. Reference branch under `no_grad`

Measured 2026-11-02: stops here. The mean LoRA gradient turns by cos 0.66 at init and 0.65 at
epoch 1, and keeping `ref_mod` and K / V in the graph does not save the mod adapter
([moss_ref_detach_grad.md](../../../project/int8_train/reports/moss_ref_detach_grad.md)).
[reference_adapters.md](reference_adapters.md) follows up: the reference rows carry the copy, so the cut needs adapters of its own on that side.

swap_text v3 trains with `ref_isolate = true` and `ref_timestep = 0.0`, the premise of
`RefCache`. Generation runs the reference stream once. Training sets the same flags and leaves
`ref_cache` at None, so the reference tokens go through all 42 blocks with gradients. Running
each block's reference part under `no_grad` and passing `ref_kv` would remove those tokens from
the backward. A `RefCache` across steps does not work, because the LoRA changes every step, so
the reference forward stays.

For a full-size reference, this takes an estimated 30–35% off the 2.71 s/step of the int8 i2i
run and about 45% off the saved set, which would allow a lower `blocks_to_swap`. FA2 still
computes dK and dV for the reference keys.

**This drops part of the LoRA gradient, and on the current recipe it removes one part
entirely.** swap_text v3 uses `mod_rank = 16`, and only the reference rows use the mod adapter's
t = 0 end, which [marks the text in the reference stream](moss_i2i_ambitious.md#what-is-shared-already)
and moves the copied region's loss most. Under `no_grad` that end gets no gradient. The
candidate therefore applies to runs without `mod_rank`, or needs the reference rows' modulation
kept in the graph, which gives back part of the saving.

Check, as a later job: compare LoRA gradients with the reference branch attached and detached
on six pairs, with `grad_error_moss.py`'s sketch, about 10 minutes. If the distance is above
bf16's noise floor, this stops here. Otherwise a 25-minute micro arm and a render comparison
follow.

## Not worth pursuing

- **int8 GEMM.** MOSS runs at about 268 TOPS against the raw 322, and the gap is rotate +
  quantize, already judged at about 10 ms ([status](../../../project/int8_train/status.md)).
- **Stream numerics.** `y32` buys 2–8% on `wide`. The `sdpa` and `rope` arms do not move
  prediction ([anima_stream_attrib.md](../../../project/int8_train/reports/anima_stream_attrib.md)).
  Point D, cross `kv_proj`, has no arm of its own.
- **GELU in fp32, int8 or int32.** It feeds straight into int8 quantization, and an int32
  accumulator needs per-token and per-row scales before it holds a value.
- **Fusing dequant, GELU and quantize.** About 10 ms, the LoRA reads the bf16 input anyway, and
  the public API has only `gelu_tanh`.
- **Elementwise kernels.** Halving all of them bounds the Anima step at about 1.13x.
- **t2i VRAM.** Budget 0.1 → 0.3 buys 2–4% speed, so t2i is not limited by VRAM.
- **Anima cross KV trim.** About 9 ms; [removed with FA4](../../anima/optimizations/fa4.md).
- **MOSS `input_act` fusion, late blocks on bf16, GPTQ recalibration on fp32 inputs.** The first
  does not fit MOSS's activation; the others cost more than the numerics they could move.
