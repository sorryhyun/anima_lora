# sm_120 has no bf16-input, fp16-accumulate HMMA

Measured 2026-11-04 on an RTX 5070 Ti (driver 610.43.02, ptxas 13.2). Probe:
`_archive/bench/sm120_hmma/probe.py` (run dirs in `_archive/bench/sm120_hmma/results/`, gitignored; the numbers
below are the run of 20:46).

## Setting

The patched FA2 build ([fa2_sm120.md](../optimization/fa2_sm120.md)) runs its fp16-accumulate
kernels on fp16 operands: HMMA with fp16 accumulation is 2x the fp32-accumulate rate on this card,
and bf16 HMMA only accumulates in fp32. So Q, K and V are cast from bf16 to fp16 for it, and the
results are cast back. An HMMA taking bf16 inputs with an fp16 accumulator would drop those casts.
They are small: the dropped `precast` patch (flash-attention-fp16 `precast.md`), which skipped the
backward's three casts, took a MOSS self-attention call from 7.95 to 7.73 ms.

PTX has no such form. `mma.sync.aligned.m16n8k16.row.col.f16.bf16.bf16.f16` fails in ptxas with
"Unexpected instruction types specified for 'mma'", as does the bf16-accumulate form. The probe
therefore assembles the fp16 kernel (`HMMA.16816.F16`) and sets bit 82 of the instruction, which
nvdisasm prints as `HMMA.16816.F16.BF16`. Each variant runs in its own process, one warp, and the
kernel log's Xid 13 lines are matched to it by the Xid 43 pid.

## Measured

| variant | HMMA at | result | exception (Xid 13) | ESR `0x505730` | `0x505728` / `0x50572c` |
|---|---|---|---|---|---|
| `f16_ref`, unpatched | 0x70 | runs, d0 = 16 (correct) | – | – | – |
| `f16_shift40`, body moved +0x40 | 0xb0 | runs, d0 = 16 | – | – | – |
| `bf16`, bit 82 set | 0x70 | 715 `ILLEGAL_INSTRUCTION` | Illegal Instruction Encoding | 0x9 | 0x1c81fb60 / 0x1174 |
| `bf16_at30`, copied over the first LDG | 0x30 | 715 | Illegal Instruction Encoding | 0x9 | 0x1c81fb60 / 0x1174 |
| `bf16_shift40`, body moved +0x40 | 0xb0 | 715 | Illegal Instruction Encoding | 0x9 | 0x1c81fb60 / 0x1174 |
| `reg_oor`, C register R10 → R138 | 0x70 | 715 | Out Of Range Register | 0xd | 0x1c81fb60 / 0x1174 |

`0x505734` is 0x20 in every faulting run.

- **The decoder rejects the encoding itself.** Bit 82 faults as an encoding error wherever the
  instruction sits. A register out of range in the same instruction gets its own exception
  class, so the hardware does report operand-level faults separately, and the bf16 bit is not one.
  The CUDA error alone (715 for both) does not show the difference.
- **The two moves are sound.** `f16_shift40` runs and returns the reference result, so the moved
  body, NOP padding and rewritten EXIT offset (`EIATTR_EXIT_INSTR_OFFSETS`) are valid; the faults in
  `bf16_at30` and `bf16_shift40` come from the bit.
- **`0x505728` / `0x50572c` is not the faulting PC.** It stays at `0x1174_1c81fb60` while the
  instruction moves −0x40 and +0x40, and across processes (two earlier runs the same day, one
  with torch's modules loaded first, printed the same value). It is also not 128-byte aligned
  against the kernel's 128-aligned `.text`. The kernel log does not give the fault position.
- `0x505730` is the warp error code: 0x9 and 0xd go with the two exception names. One earlier
  run printed 0x10009; what the upper bits mean is not known.

## Untested

Other bit positions that might select bf16 inputs on the fp16-accumulate HMMA (one bit was tried:
the one nvdisasm decodes as `.BF16`). `m16n8k8` and the other HMMA shapes. FP8 with fp16
accumulation (`e4m3 … f16`), which ptxas does assemble for sm_120a, was not run. Another sm_120
card or driver.
