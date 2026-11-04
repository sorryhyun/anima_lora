# MOSS-Image step split on the int8 plan (2026-10-31)

**Verdict: 1.30 s/step at 11.99 GB becomes 1.07 s/step at 10.78 GB, numerics unchanged. Two
fixes: `Muon` batched by shape, and the attention forward saved instead of recomputed.
The new default is budget 0.1.**

`hews_layout`, GPTQ, `stream`, fp32 stream, `--ema_card`, swap 0. Phase times from
`src/phase_timer_moss.py` (sync at each boundary, no profiler), kernels from
`src/profile_step_moss.py`.

## Where the time went

Before, budget 0.2, 1297 ms:

| phase | ms |
|---|---|
| forward | 317 |
| backward | 828 |
| `Muon.step` | 154 |
| clip + EMA | 4 |

| kernels | ms |
|---|---|
| attention (backward 249, forward 207 in 84 calls) | 482 |
| int8 GEMM | 316 |
| Triton fused | 173 |
| other GEMM | 110 |
| ConvRot activation quantization | 72 |

- PCIe was not the bottleneck: the GPU idled about 4 ms of the 790 ms backward, while the
  170 ms of `wq_t` copies ran alongside. Swap hid the same way.
- `Muon.step` ran 17k of the step's 23k kernels for 40 ms of GPU time.

## Muon

`torch.optim.Muon` loops over the 504 matrices in Python, about 34 kernels each.
`library/moss/training/muon.py::BatchedMuon` stacks them by shape (four at rank 32) and
runs one batched Newton-Schulz per shape. 154 → 14.7 ms. Updates match torch within the
bf16 rounding, and the momentum buffers match exactly (`test_batched_muon_matches_torch`).

## Attention recompute

Why the budget bought no speed: below budget 1 the partitioner's knapsack only prices
nodes the aggressive min-cut bans from recompute. Custom ops are not banned, so FA2 and the
int8 linears were recomputed at any budget, and an op with no flop formula would count as
1 anyway. Two changes make them candidates at their real cost:

- `functorch_config.is_non_builtin_to_include = True`, in `set_activation_memory_budget`.
- `register_flop_formula` on `flash_attn::_flash_attn_forward` (`loader.py`) and the int8
  ops (`linear.py`), calibrated on the kernels: FA2 2.46 ms, int8 0.42 ms against 0.87 ms
  for a bf16 3072² GEMM.

The block's saved set at budget 0.2 goes from 209 to 200 MB, now with FA2's `o` and `lse`.
FA2's forward falls from 84 to 42 calls per step, 207 → 104 ms.

| plan | s/step | peak |
|---|---|---|
| budget 0.05 | 1.16 | 9.98 GB |
| **budget 0.1** | **1.07** | **10.78 GB** |
| budget 0.2 | 1.06 | 11.83 GB |
| budget 0.3 | 1.05 | 12.88 GB |
| swap 20, budget 0.1 | 1.08 | 8.67 GB |
| checkpointing | 1.24 | 9.61 GB |
| bf16 plan, `hews_layout` | 1.59 | 12.92 GB |
| bf16 plan, `channel_(caststation)_layout` | 1.66 | 13.48 GB |
| i2i sanitize config (bf16) | 3.71 | 11.69 GB |

At 0.05 the saved set no longer holds FA2's output and the recompute returns. Each row is
one run of 15–40 steps. Step losses match the runs before the change within 5e-4.

After both, kernels take 1054 of the 1059 ms step: the step is GPU-bound, about 6.9k
kernels. Attention is 387 ms and the int8 GEMMs 316 ms.

## Default moved to budget 0.3 (same day)

`ama_mitsuki_layout`, same plan, seed 0, median of steps 11–40, one run per arm. The 0.1 arm is
the first 40 steps of the 848-step run; the others ran `--max_steps 40`.

| budget | job | s/step | mean | peak |
|---|---|---|---|---|
| 0.1 | `20261031-132857-1a36c3` | 1.090 | 1.051 | 10.95 GB |
| 0.2 | `20261031-135426-211764` | 1.069 | 1.045 | 11.92 GB |
| 0.3 | `20261031-135612-034f0e` | 1.052 | 1.019 | 12.98 GB |
| 0.4 | `20261031-135746-540997` | OOM at step 3 | – | 14.39 GB allocated |

Step losses at steps 1, 2, 20 and 40 stay within 4e-4 of the 0.1 arm. The 0.4 OOM is a forward
allocation of 70 MiB with 86.5 MiB free, at a block's `model.py:215`. Its allocation already
sits above the 14 GB the 1 GB per 0.1 trend predicts; why was not traced.

## Not measured

- A complete LoRA with `BatchedMuon` against torch's `Muon`. The updates match per step
  within bf16 rounding, so none is expected to differ.
- Sets far from 4.6k joint tokens: FA2's flops scale with L² and its saved bytes with L,
  so where the budget stops holding it may move.
