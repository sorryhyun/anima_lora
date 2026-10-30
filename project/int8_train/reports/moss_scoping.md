# MOSS-Image scoping — GEMM bench and step split (2026-10-30)

Whether the int8 base carries over to the MOSS-Image line. On MOSS's block shapes the
int8 GEMMs run **2.85x forward, 2.72x dX** (Anima: 2.6x), and the six converted linears
take **47% of the default compiled step** (895 of 1902 ms). If the block swap keeps
hiding behind the faster compute, the step goes to ~1330 ms, a predicted **1.43x**.

## GEMM bench

`make daemon-run ARGS="project/int8_train/src/gemm_bench.py 4608 moss"` (job
`20261030-142652-bda5d9`): block 10's weights, M = 4608 (t2i's ~4.2k image + 0.4k
text tokens), ConvRot W8A8 through `comfy_kitchen.int8_linear`, rotation and
quantization included.

| linear | K → N | bf16 fwd | int8 fwd | bf16 dX | int8 dX |
|---|---|---|---|---|---|
| `q`, `k`, `v`, `o` (each) | 3072 → 3072 | 0.87 | 0.35 | 0.87 | 0.35 |
| `ffn_up_gate` | 3072 → 16384 | 4.59 | 1.39 | 4.95 | 1.79 |
| `ffn_down` | 8192 → 3072 | 2.30 | 0.85 | 2.40 | 0.78 |
| block total | | 10.37 | 3.64 | 10.81 | 3.98 |

ms per call. Raw throughput: bf16 99 TFLOPS, int8 314, fp8 196. Relative error
against fp32 on the bench's synthetic inputs is 0.012–0.015 for int8 (bf16: 0.0017),
as on Anima's shapes.

## Step split

`project/int8_train/src/profile_step_moss.py` wraps `library.moss.train.training_loss`
(one call per step) and profiles the shipped plan, `configs/moss/lora.toml`: rank 32
LoRA, Muon, compile, no checkpointing, `activation_memory_budget 0.5`,
`blocks_to_swap 36`, `swap_h2d_only`. Cache `channel_(caststation)_layout`, 4188–4596
joint tokens. Steps 11–30 wall-timed, 31–36 profiled. Job `20261030-142755-89531d`, raw
numbers in [`moss_step_profile.json`](../assets/moss_step_profile.json).

| part | ms per step |
|---|---|
| step wall time (median of 20) | 1902 |
| compute kernels (all but memcpy) | 1706 |
| converted GEMMs | 895 |
| of which 3072² (`q k v o`) | 325 |
| of which `ffn_up_gate` | 383 |
| of which `ffn_down` | 188 |
| other GEMMs (LoRA rank, modulation, text projection, Muon) | 119 |
| attention (FA2), forward 225 / backward 275 | 519 |
| Triton fused (norms, RoPE, SiTU, residual) | 161 |
| H2D block swap, on its own stream | 622 |

Call counts: the FFN GEMMs run exactly 42 blocks × (forward + dX); the 3072² ones once
more per block, the min-cut partitioner's recompute at budget 0.5. FA2's forward runs
86 times per step, so the partitioner recomputes attention in almost every block's
backward.

## Reading

- With the bench's 2.78x on 895 ms, compute drops 573 ms, to ~1130 ms of kernels.
- **The swap is the new constraint.** Each H2D block move is 8.5 ms (0.21 GB at
  ~25 GB/s), 72 per step. After int8 a block's forward is about 3.7 ms of GEMM, 2.6 ms
  of attention and ~1.3 ms of fused kernels, ~7.6 ms: just under one move. The forward
  of the 36 swapped blocks turns roughly PCIe-bound (an estimated 30–50 ms of stall); the
  backward (dX, attention backward and recompute, ~15 ms per block) still hides it.
  `tune_swap.py` should find the swap/budget knee lower than bf16's ~40.
- `full` mode keeps two int8 copies, the bytes of one bf16 weight. VRAM, the swap
  traffic and therefore the swap/budget pair stay as in bf16. The forward needs only
  `wq` and the backward only `wq_t` plus `wq` for the recompute, so a phase-aware swap
  could move less, but not without changing the partitioner's recompute.
- 196 ms of the step is outside kernel time (1902 against 1706) and int8 does not
  touch it.

## What porting takes

- **Block swap.** `library/runtime/offloading.py` moves `module.weight` only, and
  `HostMasters` builds its page-locked copy from `.weight`. Packing `wq` and `wq_t` into
  one flat int8 `.weight` (views for the two copies) and keeping the fp32 row scales
  resident on the card lets the swapper run unchanged.
- **Adapters.** The MOSS LoRA patches an instance-level `forward` on each target
  linear, as Anima's does, so the in-place `__class__` swap of `convert_` keeps working.
  `lora` and `lokr` (bypass) run on an int8 base; `loha` builds `W + ΔW` and does not.
  `down_init weight_svd*` and `svd_drop_loud` read bf16 weights, so conversion goes
  after `create_network`.
- **No producer fusion.** MOSS's norms are followed by `_affine` shift + scale and its
  FFN activation is SiTU, not SwiGLU, so comfy_kitchen's `input_act` fusions fit no
  better than on Anima.
- **Outliers.** FFN-input channel 42 is 11–26x the median with a peak-to-mean of 2.7
  (`docs/moss/channel_scaling.md`); spread by the 256-column Hadamard it is ~1.6x within
  its group. `ffn_down`'s input is bounded by SiTU's soft cap (±100).

## Lower swap, lower budget (bf16, measured; int8, predicted)

The same profile at three plans that trade swap for recompute (jobs
`20261030-143303-{4f1eb7,da987c,f7afaa}`, raw numbers in
`assets/moss_step_profile_{ckpt0,b005s0,b01s12}.json`). "int8 step" is the compute
kernels with the converted GEMMs divided by the bench's 2.78x, plus the measured time
outside kernels, assuming any swap stays hidden.

| plan | bf16 step | converted GEMMs | attention | H2D | peak | int8 step |
|---|---|---|---|---|---|---|
| swap 36, budget 0.5 (default) | 1902 | 895 | 519 | 622 | 13.46 GB | ~1330 |
| swap 12, budget 0.1 | 2152 | 1160 | 522 | 216 | 10.86 GB | ~1410 |
| swap 0, budget 0.05 | 2186 | 1194 | 522 | 12 | 12.34 GB | ~1420 |
| swap 0, `grad_checkpointing` | 2332 | 1286 | 513 | 12 | 11.96 GB | ~1510 |

- Attention costs the same in every plan: budget 0.5 already recomputes its forward
  in almost every block. What the budget buys is GEMM recompute, 895 → 1194 ms from
  budget 0.5 to 0.05, and int8 shrinks that 299 ms to ~108.
- The plans' spread shrinks from 1.23x in bf16 (1902–2332) to ~1.14x in int8, and the
  default stays fastest only if its 622 ms of H2D keeps hiding behind a ~7.6 ms
  block forward.
- Swap 0 at budget 0.05 peaks at 12.34 GB, so swap 0 has room for a higher budget
  (~0.15 by a linear fit to the default's peak), not measured.
- **Swap 0 needs no offloader change**: the int8 base can be measured there first, as
  Anima's `train_int8.py` runs, and the swap packing judged on what is left (~6%).

## int8 base, swap 0, `grad_checkpointing` (measured)

`src/train_int8_moss.py --int8 full` on the same config and cache, `--blocks_to_swap 0
--grad_checkpointing`, round-to-nearest weights; job `20261030-144409-8f98f2`, stopped
after 70 of 240 steps.

| plan | s/step (steps 10–40) | peak |
|---|---|---|
| bf16, swap 36, budget 0.5 (default) | 1.83–1.91 | 13.46 GB |
| bf16, swap 0, checkpointing | 2.26–2.34 | 11.96 GB |
| **int8, swap 0, checkpointing** | **1.44–1.49** | 11.96 GB |

1.57x over bf16 on the same plan and 1.28x over the bf16 default, against a predicted
~1.51 s. The step-10 and step-20 losses, the same sample and σ, differ from both bf16
runs (0.1806 / 0.1010 against 0.2083 / 0.1151) while step 30 matches (0.1483 / 0.1470);
not looked into.
