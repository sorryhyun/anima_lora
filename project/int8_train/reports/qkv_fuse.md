# One int8 linear for MOSS `proj.q/k/v` (2026-11-02)

**Verdict: built, and it is how the int8 base runs now.** The kernel goes 1.30x on the forward and
1.31x on dX, about 24 ms over 42 blocks, or 2% of a step. The forward is bit-identical to three
calls, and the LoRA gradient keeps its distance from fp32 (0.986–1.006x).

Candidate 3 of [step_headroom.md](../../../docs/moss/proposal/step_headroom.md). Script:
[`src/qkv_fuse_bench.py`](../src/qkv_fuse_bench.py), job `20261102-214504-9d1678`.

## Kernel

Block 10, GPTQ forward copies, 4600 tokens, RTX 5070 Ti, ms per block. dY is synthetic.

| | three calls | one call | ratio | saved over 42 blocks |
|---|---|---|---|---|
| forward | 1.058 | 0.813 | 1.302x | 10.3 ms |
| dX | 1.363 | 1.041 | 1.309x | 13.5 ms |

dX error against fp32 on that synthetic dY: 1.33% three calls, 1.45% one call.

## Real step

`phase_timer_moss.py`, `configs/moss/lora.toml`, `hews_layout`, 40 steps, median of steps 11+. Control:
the same build with q / k / v converted as three linears.

| order | arm | forward | backward | sum | peak |
|---|---|---|---|---|---|
| 1 | three | 363.2 | 692.1 | 1075.4 ms | 12.88 GB |
| 2 | one | 333.3 | 682.8 | 1035.1 ms | |
| 3 | three | 344.0 | 670.8 | 1033.8 ms | 12.88 GB |
| 4 | one | 328.8 | 665.6 | 1013.3 ms | 12.85 GB |

Repeats of one arm differ by 22–42 ms, more than the gain, and each arm got faster on its second run.
The 40 ms of pair 1 is mostly that spread. Pair 2's 20.5 ms agrees with the kernel's 24 ms, so the
kernel table is the number to use.

Losses at steps 10–40 match to 3e-4. `|g|` differs by up to 0.019 at the same steps, inside FA2's
existing spread ([fa2_fp16acc.md](fa2_fp16acc.md)).

## Implementation

`dit_int8/base.py::fuse_qkv_` builds `block.qkv`, `Int8TrainLinear` 3072→9216, with the GPTQ rows
concatenated and `wq_t` quantized from the concatenated weight. `proj.q/k/v` stay as `QKVPart`: the
adapters still patch them, so LoRA keys, `down_init` and the bf16 path do not change. `_Block._qkv` hands
each part its row, and `ref_rows` reads the rows past the adapters. `WqtStream` holds 4 copies per block
instead of 6, at the same bytes. Scripts that call `convert_moss_` per linear keep three GEMMs.

Short runs on the new path, all finished with finite loss: i2i `train_v3_en50_ko150_refkv` and `_refall`
(`ref_rows`, swap 38 and 40, 4 steps), `generate.py --int8 on` with a LoRA (768², 4 steps). Tests:
`test_moss_int8.py`.

## Real gradient

`grad_error_moss.py --modes bf16,gptq_f32s,gptq_fused_f32s`, job `20261102-220025-71c746`, 6 images × 3 σ,
the LoRA gradient's distance from fp32 over the between-image noise, median over the 252 linears. Raw
numbers: [`moss_grad_error_qkv.json`](../assets/moss_grad_error_qkv.json).

| σ | bf16 | three | one | one / three |
|---|---|---|---|---|
| 0.1 | 0.694 | 0.651 | 0.655 | 1.006x |
| 0.5 | 0.909 | 2.046 | 2.018 | 0.986x |
| 0.9 | 0.717 | 0.913 | 0.915 | 1.002x |

Sharing the scales moves nothing a reader could see. Per linear, the ratio's median is 0.99–1.007x and its
worst 1.036x; per image 0.966–1.018x. Prediction and loss match to every digit printed, as the forward is identical. The arm has no
same-arm repeat, so the ±1% here is not separated from FA2's own run-to-run spread.

The 2.0 at σ 0.5 sits in both int8 arms. It is the int8 base's, not this change's.

## Not measured

- **i2i step time**, and a complete LoRA on this path.
