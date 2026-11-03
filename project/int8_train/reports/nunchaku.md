# Nunchaku's FP16 attention against ours (2026-11-03)

Read from source, nothing run: `src/kernels/zgemm/attention.{cu,cuh}` on nunchaku-tech/nunchaku main,
plus the usage page (https://nunchaku.tech/docs/nunchaku/usage/attention.html). The docs say only "FP16
attention accumulation, up to 1.2× faster than FlashAttention-2 on 30-, 40-, 50-series, without
precision loss", with no bench setup.

**Verdict:** `set_attention_impl("nunchaku-fp16")` is the forward of our parked `tri_all` arm
([moss_fwd16_triton.md](moss_fwd16_triton.md)), forward only, and not a drop-in for us. Our own patch,
`patches/fwd_fp16acc.patch` in `flash-attention-fp16`, is now built and measured:
[fa2_fwd_fp16acc.md](fa2_fwd_fp16acc.md).

| | Nunchaku `nunchaku-fp16` | ours |
|---|---|---|
| scope | forward only | fp16acc backward shipped (dP / dV / dQ / dK); forward fp16acc parked in Triton |
| QKᵀ | `mma.m16n8k16` f16 accumulator over the whole head dim 128, then fp32 | fp32 accumulation (`tri_all`: fp16acc) |
| softmax | fp32, exp2, online rescale | fp32 |
| PV | f16 accumulator inside one 32-key tile, added into an fp32 O | same shape in `tri_pv` / `tri_all` |
| tile | BLOCK_M 128 × 32 keys, 8 warps | 128 × 32 on sm_120 |
| inputs | Q / K / V fp16 (output may be bf16) | FA2 needs fp16 too: three `to(half)` |
| head dim | 128 | 64, 128 |
| shapes | `Nq % 128 == 0`, `Nkv % 32 == 0`, no mask (`TODO: mask tokens in last block`), Q / K / V in a packed layout written by its W4A4 QKV GEMM epilogue | the FA2 API, any length |
| overflow | `fix_nan`: a NaN score becomes −inf, so the key drops out silently | none; it would show as a NaN gradient |

Speed: our Triton forward at `[1, 4600, 24, 128]` goes 2.882 → 2.281 ms (1.26×) with `tri_all`, in
line with their 1.2×. In training that is about 2.5–3% of a step after the fp16 casts (estimate).

Why not use theirs: no backward; no key mask, so MOSS `forward_batched` (masked padded keys) and text
lengths off a multiple of 32 cannot run on it without wrong attention; the packed layout comes from
their fused GEMM.

What it adds: a shipped CUDA kernel runs the `all` arm on every Nunchaku user, so the fp16 range is not
a problem in practice there (our S peak was 5373, 12× under 65504). For us the forward arm counts most
in generation, which is forward only and gets nothing from the shipped backward; that gain is
unmeasured.
