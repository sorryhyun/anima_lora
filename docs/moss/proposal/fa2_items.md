# FA2 sm_120 build: what is left to try

Status (2026-11-10): the casts item is built (`patches/f16in.patch` in the fork, not released) and
saves 19 ms of a MOSS step. The drain item was built and measured slower, and the tile item was
closed on paper. They come from a read of the shipped build
([fa2_sm120.md](../optimization/fa2_sm120.md), patches in `~/anima/flash-attention-fp16`).

| item | bound | state |
|---|---|---|
| [Casts fused upstream](#casts-fused-upstream) | about 17 ms of a 1022 ms MOSS step (1.6%) | built: −19 ms a step, same peak ([build](#build-2026-11-10)) |
| [Drain: assign, do not add](#drain-assign-do-not-add) | about 7 ms of a step (0.7%) | closed: the kernel ran 5.8% slower ([result](#result-2026-11-10-1)) |
| [Backward tile geometry](#backward-tile-geometry) | – | closed: the register file caps the SM at 8 warps |

## Where the time is

One MOSS self-attention call, `[1, 4600, 24, 128]`, RTX 5070 Ti
([ncu run](../../../project/finished/int8_train/reports/fa2_fp16acc.md#where-the-2x-went-2026-11-04)):

| piece | ms |
|---|---|
| Backward kernel, tensor pipe busy | 4.3 |
| Backward kernel, tensor pipe idle | 1.5 |
| Backward, casts and `abs` + `amax` outside the kernel | 0.34 |
| Forward | 2.44, of which three casts 0.24 |

The idle 1.5 ms is the drain: every tile, the four fp16 partials (dP, dV, dQ, dK) are converted
and added into fp32 fragments. With two warps per scheduler that work does not hide behind the
tensor pipe. Stock idles 0.7 ms.

## Casts fused upstream

### Today

The fp16-accumulate kernels need fp16 operands, so the op casts inside itself:

- `mha_fwd` runs `q.to(half)`, `k.to(half)`, `v.to(half)`: three kernels, 0.24 ms.
- `mha_bwd` runs the same three again on the saved bf16 tensors, plus `amax(abs(dO))`: 0.34 ms.

The model has just cast q and k itself. `library/moss/model.py:131` ends RoPE with
`.to(v.dtype)`, so q and k are written as bf16 and then read back and rewritten as fp16.

### Change

Write fp16 at the call site, and keep fp16 out of the gradient path:

```python
q = rotate(q.reshape(shape), rope)            # no cast: the gradient anchor
q16 = q.to(torch.float16).detach()            # fuses where .to(v.dtype) fuses today
out = FlashAttnFp16.apply(q, k, v, q16, k16, v16)
```

- The forward reads only the fp16 copies and returns bf16 O. Nothing reads the anchors' values,
  so the compiled graph does not materialise them.
- The backward saves the fp16 copies in place of bf16 q, k and v (same bytes), takes bf16 dO and
  returns dQ, dK, dV for the anchors. No gradient passes through an fp16 tensor: they sit near
  1e-7, under fp16's normal range.
- V comes from the `proj.v` adapter in bf16 and keeps one cast unless it fuses into that add.

Wheel side:

- `mha_bwd` taking fp16 Q, K, V with bf16 O and dO is the `fp16acc_precast` branch of
  `patches/dropped/precast.patch`, with its single `amax_abs_bh` kernel. It can be taken as is.
- `mha_fwd` needs an entry that takes fp16 Q, K, V and writes bf16 O. The PV kernel already
  writes bf16 from fp16 operands.
- The 1024-key gate on the PV forward exists only because of the casts, so it goes. Anima's
  cross-attention (512 keys, forward 0.89x and backward 0.98x today) would then gain, after the
  same call-site change in `networks/attention_dispatch.py`.

### Bound

Backward three casts to none, forward three to one: about 0.40 ms of the 8.0 ms per call, 17 ms
of a 1022 ms MOSS step. With V fused too, 0.48 ms and 20 ms. A generation forward goes from
2.44 to about 2.28 ms.

### Why this is not `precast` again

`precast` saved the same backward casts and won per call (7.95 → 7.73 ms), then lost on the
step: backward back at stock and the peak 0.16 GB higher
([`notes.md`](https://github.com/sorryhyun/flash-attention-fp16/blob/main/notes.md)). There the
fp16 copies were outputs of FA2's forward, so keeping them meant saving three more tensors or
re-running the forward. Here they come from a pointwise node, and the partitioner's choice for
them is the one it makes for bf16 q, k and v today. That argument is untested, and `precast`'s
loss was never traced to its cause, so only a real step decides this.

### First check, no build

An unpacked wheel whose dist-info names the `precast` build is still at
`output/int8_train/fa2_build_drain2/wheel/` (run the fork's `smoke.py` on it first). That
build's backward accepts fp16 Q, K, V from anywhere. A Python wrapper that makes the copies in the model
graph and hands them to that backward tests the partitioner question on a real step
(`phase_timer_moss.py`, as in `notes.md`'s table). The forward still casts inside, so only the
backward half shows: about 10 ms a step, against 3 ms between two stock runs there.

### Result (2026-11-10)

Run as above. A `torch.autograd.Function` replaced `library.moss.model.flash_attn_func`. Its forward
is the bf16 FA2 forward (casts inside), and it saves `q.to(fp16).detach()` (and the same for k and v)
for `flash_attn::_flash_attn_backward`. The precast wheel was put first on `sys.path`.
`configs/moss/lora.toml`, `ama_mitsuki_layout`, 40 steps, one run per arm, comfy-kitchen
`0.2.41+fp32div`. Out, dQ, dK and dV are bitwise equal to the precast wheel's own path.

| arm | forward, steps 20–40 | backward, steps 20–40 | backward median | peak |
|---|---|---|---|---|
| shipped fwdpv wheel | 6.07 s | 13.50 s | 666 ms | 12.96 GB |
| copies, forward through the wheel's op | 6.26 s | 14.65 s | 724 ms | 13.12 GB |
| copies, forward through an op returning out, lse, rng | 6.08 s | **13.29 s** | **656 ms** | 12.96 GB |

Peaks of the two copy arms have the script's 0.40 GB of check tensors, live for the whole run,
taken off.

The last arm is the change as proposed, backward half: −10 ms a step, the size the bound
predicted, same forward, same peak. Its partitioned block graph saves the same 11 tensors as
stock (`out`, `lse`, `rng_state`, the qkv int8 GEMM recomputed in the backward)
(`TORCH_LOGS=aot_graphs`).

The middle arm is what the partitioner does with a mispriced node. The knapsack prices a
recomputable node at the sum of its outputs (`_size_of` in `torch/_functorch/partitioners.py`). The
precast wheel's `_flash_attn_forward` also returns its three fp16 copies, real-sized in its fake, so
the knapsack saw FA2's forward at about four times its saved bytes. It then saved the qkv output
(`[L, 9216]`) and ran FA2's forward again in every block's backward (+58 ms). The bytes the backward
needs are `out` and `lse` in every arm.

Eager, one `[1, 4600, 24, 128]` backward on the precast wheel, all bitwise equal:

| backward input | ms | peak above live tensors |
|---|---|---|
| bf16 Q, K, V (casts inside `bwd`) | 5.55 | 243 MB |
| fp16 Q, K, V (precast branch) | 5.33 | 162 MB |
| fp16 Q, K, V, three casts outside | 5.56 | 243 MB |

So the item holds. The build needs an fp16-in forward op whose outputs are only out, lse and
`rng_state` (plus its flop formula), and the precast backward branch. The forward half (one cast
instead of three, the 1024-key gate gone) is still unmeasured.

### Build (2026-11-10)

`patches/f16in.patch`, applied after `stable_abi.patch`:

- `mha_fwd`: fp16 Q, K, V with a bf16 `out` tensor run the PV fp16acc forward without casts, at
  any key count. The 1024-key gate stays for bf16 input. `FLASH_ATTN_FP16ACC_FWD=all` also
  accumulates S in fp16.
- `mha_bwd`: fp16 Q, K, V with bf16 O and dO take the fp16acc backward without casts, and the
  gradients come back in bf16. `abs`, `amax` and `clamp_min` on dO are one kernel, which reads
  16 rows a block (44 µs against `precast`'s 73 µs and stock's three ops at 113 µs).
- Python: the op `flash_attn::_flash_attn_forward_f16in` returns only out, lse and `rng_state`.
  `flash_attn_f16in_func(q, k, v)` makes the fp16 copies inside its traced forward and saves them
  for the backward. `flash_attn_f16in_takes` gates it to head dim 128 with as many K/V heads as Q
  heads; `FLASH_ATTN_FP16ACC=0` or `FLASH_ATTN_FP16ACC_FWD=0` at import turns it off.
- anima_lora: `library/moss/model.py` calls `flash_attn_f16in_func` when the installed flash_attn
  has it. `library/runtime/dynamo.py` registers the flop formula for the new op.

Wheel `2.8.3+sm120.hdim64128.fp16acc.fwdpv.f16in.cu132.torch2.12plus`, `cp310-abi3`, sm_120 only.
Same training run as above, one run per arm:

| wheel | forward median | backward median | forward + backward, steps 20–40 | peak |
|---|---|---|---|---|
| shipped fwdpv | 297.7 ms | 666.3 ms | 19.57 s | 12.96 GB |
| f16in | 289.1 ms | 655.1 ms | 19.17 s | 12.96 GB |

That is −19 ms a step (2.0%): −9 ms forward from the fused casts, −11 ms backward. The block
graph saves the same 11 tensors as stock.

Per call, `[1, 4600, 24, 128]` (eager, so the forward casts still run as kernels): out, dQ, dK and
dV are bitwise equal to the bf16 path, and the error against fp32 is unchanged. Forward + backward
is 8.08 → 7.84 ms. Backward kernels, profiled:

| | main kernel | everything else | total |
|---|---|---|---|
| shipped, bf16 in | 4861 µs | 630 µs (three casts 241, `abs` 63, `amax` 50, `dot_do_o` 180) | 5.49 ms |
| f16in, fp16 in | 4882 µs | 305 µs | 5.19 ms |

Anima's cross-attention shape (`[1, 4096, 16, 128]` against 512 keys) now takes the PV forward:
forward + backward 0.823 → 0.770 ms, with out's error against fp32 1.69e-3 against 2.18e-3.
`networks/attention_dispatch.py` calls `flash_attn_f16in_func` for dense flash calls without dropout
or `deterministic`. An Anima step (`profile_step.py --int8 full`, `--method lora --preset default`,
median of 25) goes from 392.0 to 380.0 ms (3.1%).

### Watch

- A new op name needs its flop formula in `library/runtime/dynamo.py`. Without it the
  partitioner recomputes FA2's forward at any budget, 84 calls a step instead of 42.
- The op returns only what the backward reads. Every extra output counts toward the knapsack's
  price for the node; three fp16 copies were enough to make it recompute FA2's forward.
- Eager runs gain the backward half only: nothing fuses the forward cast.

## Drain: assign, do not add

### The fact

Of the four partials, only dK and dV accumulate across the query loop. dP and dQ are added into
an fp32 fragment that was cleared a few lines earlier:

- `clear(acc_dp)` at `flash_bwd_kernel.h:563`, then `add_fp16_accum(acc_dp, acc_dp16)`.
- `clear(acc_dq)` at `:599` (the `Seq_parallel` branch, which this kernel takes), then
  `add_fp16_accum(acc_dq, acc_dq16)`, then the atomicAdd.

Line numbers are stock 2.8.3. `+0.0 + x` is not `x` when `x` is `−0.0`, so nvcc keeps the add.
ncu agrees: `fadd` goes from 0.5 G to 4.1 G, a rise equal to the 3.6 G conversions.

Per 64×64 tile that is 4096 (dP) + 8192 (dQ) of the 28672 drained elements, 43% of the drain's
adds.

### Change

A second helper that writes `acc(i) = float(a16(i))`, used for dP and dQ in `fp16acc.patch`.
The fp32 `clear` of those two fragments goes with it. Results match bit for bit except the sign
of a zero. The `all` mode has the same pattern on `acc_s`, forward and backward.

### Bound

12288 of about 57 k drain instructions per tile, 21%. If time followed instructions that is
0.17 ms of the 5.81 ms kernel and 7 ms of a step.

It may be nothing. Draining every second tile cut FMA-pipe instructions 20% and ran 5.5% slower,
because it chained HMMAs across tiles. This change adds no chain, but instruction count has
already failed once as a predictor on this kernel.

### Cost

One translation unit. `output/int8_train/fa2_build_drain2/nvcc_fp16acc.cmd` is the saved nvcc
line from the drain experiment, and `project/finished/int8_train/src/fa2_ncu.py` reads the
result. It can share a wheel build with the casts item.

### Result (2026-11-10)

Built into the first f16in wheel (`assign_fp16_accum` for dP and dQ, their `clear` dropped). The
gradients stayed bitwise equal, but the main backward kernel ran 5106–5136 µs against 4861–4882 µs
without it (two alternating profiles of each build, the same shape). That is +5.8%, about +11 ms a
step. The SASS was not read. This is the second time on this kernel that fewer drain instructions ran
slower (see draining every second tile). The change was reverted before the wheel above.

### Only if it pays

dP's whole path is fp16 → fp32 → `dS = P·(dP − Δ)` → fp16, about six scalar instructions an
element by a read of the source.
In packed `half2` arithmetic it is about three instructions per two elements. Δ rounded to fp16
puts the same error on every entry of a row, which breaks `Σ dS = 0` per row and lands in dQ, so
this one needs the gradient-error bench before any timing.

## Backward tile geometry

**Closed on paper.** The idea was a smaller shared-memory footprint so that two blocks run per
SM and the drain hides behind another warp's HMMA. The card does not allow it.

`cudaDeviceGetAttribute` on the 5070 Ti:

| attribute | value |
|---|---|
| SMs | 70 |
| shared memory per SM | 102400 B |
| shared memory per block, opt-in | 101376 B |
| registers per SM | 65536 |

- The kernel uses 245 registers a thread and 256 threads: 62720 registers a block. Two blocks
  need 125440. The register file holds the SM at 8 warps whatever the tile.
- A 64×32 tile without the double buffer is 48 KB and would fit twice by shared memory, but not
  by registers. It only doubles the tile count, and with it the dQ atomicAdds (8192 a tile) and
  the Q and dO loads.
- A larger tile does not fit either. 128×64 without the double buffer is 112 KB against the
  99 KB limit, and its fp32 fragments add 64 registers to 245.
- V in shared memory instead of registers fits (96 KB) and frees about 16 registers. The
  double-buffered partials that the every-second-tile drain wanted need 32.

So 64×64 at 8 warps is the largest tile that fits both limits, and no smaller one raises
occupancy. Two blocks per SM would need the kernel under 128 registers a thread, which means
dK and dV accumulating outside registers: a different kernel, not a tile choice. What is left on
the idle 1.5 ms is the drain item above.

Not tried: the atom layouts (`AtomLayoutMSdP`, `NdKV`, `MdQ` = 4, 2, 2) at the same tile. There
is no argument that another split is faster with fp16 accumulation.
