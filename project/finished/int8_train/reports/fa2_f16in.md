# FA2 fp16-input path (`f16in`) (2026-11-10)

Moved out of `docs/moss/optimization/fa2_sm120.md`, which keeps the method. Code: commits
`9bc52e2d` (Linux) and `91c18467` (Windows release), `library/runtime/fa2.py`,
`library/moss/model.py`, `networks/attention_dispatch.py`.

## What it replaced

The fp16-accumulate kernels ([fa2_fp16acc.md](fa2_fp16acc.md), [fa2_fwd_fp16acc.md](fa2_fwd_fp16acc.md))
need fp16 operands, so a bf16 call cast Q, K and V inside the op: three kernels in the forward,
three again in the backward, plus `abs` and `amax` on dO. `flash_attn_f16in_func` casts in its own
traced forward, so under `torch.compile` the casts fuse into the kernel that wrote q, k and v.

## The partitioner trap

The first version of the op also returned its fp16 copies of Q, K and V for the backward. The
activation-budget knapsack prices a node at the sum of its outputs (`_size_of`), so the op looked
four times its saved size, and the partitioner ran FA2's forward again in every block's backward:
+58 ms a MOSS step. The shipped op `flash_attn::_flash_attn_forward_f16in` returns O, the logsumexp
and the RNG state only, and carries its own flop formula.

## Measured

RTX 5070 Ti, one run per arm. "shipped wheel" is the fwdpv release before f16in.

| | shipped wheel | f16in | |
|---|---|---|---|
| MOSS backward per call, `[1, 4600, 24, 128]` | 5.49 ms | 5.19 ms | main kernel 4861 / 4882 µs |
| MOSS step, `configs/moss/lora.toml`, `ama_mitsuki_layout` | 297.7 + 666.3 ms | 289.1 + 655.1 ms | forward + backward medians, 12.96 GB both |
| Anima step, `profile_step.py --int8 full`, `--method lora --preset default` | 392.0 ms | 380.0 ms | median of 25 |
| Anima cross call, Lk 512, forward + backward | 0.823 ms | 0.770 ms | O error 2.18e-3 → 1.69e-3 |

MOSS's O, dQ, dK and dV are bitwise equal to the bf16-input path. The dO max became one kernel
(44 µs on MOSS's shape, against 113 µs for the three ops). With no casts in the op, the PV forward
runs at any key count, so Anima's 512-key cross-attention takes it too.
