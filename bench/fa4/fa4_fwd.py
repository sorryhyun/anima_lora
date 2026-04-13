"""Quick FA4 kernel benchmark: TMA-optimized vs original SM120."""

import os
import torch
import time

# Match the Anima model's actual shapes
BATCH = 2
SEQLEN = 4096
N_HEADS = 32
HEAD_DIM = 64
DTYPE = torch.bfloat16
WARMUP = 10
ITERS = 50


def bench(label: str):
    from flash_attn.cute import flash_attn_func

    q = torch.randn(BATCH, SEQLEN, N_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    k = torch.randn(BATCH, SEQLEN, N_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    v = torch.randn(BATCH, SEQLEN, N_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")

    # Warmup (includes JIT compilation)
    for _ in range(WARMUP):
        out, lse = flash_attn_func(q, k, v)
    torch.cuda.synchronize()

    # Timed
    start = time.perf_counter()
    for _ in range(ITERS):
        out, lse = flash_attn_func(q, k, v)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / ITERS * 1000

    # Also bench cross-attn shape (shorter KV)
    k_cross = torch.randn(BATCH, 512, N_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    v_cross = torch.randn(BATCH, 512, N_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    for _ in range(WARMUP):
        out2, lse2 = flash_attn_func(q, k_cross, v_cross)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(ITERS):
        out2, lse2 = flash_attn_func(q, k_cross, v_cross)
    torch.cuda.synchronize()
    elapsed_cross = (time.perf_counter() - start) / ITERS * 1000

    print(f"[{label}]")
    print(f"  Self-attn  ({SEQLEN}x{SEQLEN}): {elapsed:.2f} ms")
    print(f"  Cross-attn ({SEQLEN}x512):  {elapsed_cross:.2f} ms")


if __name__ == "__main__":
    bench(os.environ.get("BENCH_LABEL", "current"))
