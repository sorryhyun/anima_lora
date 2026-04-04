"""Microbenchmark: MMA throughput on SM120 (RTX 5060 Ti).

Measures raw matmul throughput at attention-relevant shapes to understand
the performance ceiling for different numeric formats.

SM120 has two MMA instruction families:
  - SM80 MMA (mma.sync.aligned, FP16/BF16): shape (16,8,16), K=16 per inst
  - SM120 native MXF4 (mma.kind::mxf4): shape (16,8,64), K=64 per inst (4x)

cuBLAS uses SM80 MMA for BF16. The question is how much headroom
SM120-native MXF4 instructions could provide.
"""

import torch
import time

WARMUP = 20
ITERS = 100


def bench_mm(M, N, K, dtype, label):
    A = torch.randn(M, K, dtype=dtype, device="cuda")
    B = torch.randn(K, N, dtype=dtype, device="cuda")

    for _ in range(WARMUP):
        torch.mm(A, B)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(ITERS):
        torch.mm(A, B)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / ITERS * 1000

    flops = 2 * M * N * K
    tflops = flops / (ms / 1000) / 1e12
    print(f"  [{label}] {ms:.3f} ms  ({tflops:.1f} TFLOPS)")
    return ms, tflops


def bench_scaled_mm(M, N, K, in_dtype, label):
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda").to(in_dtype).contiguous()
    # _scaled_mm expects B as (K, N) column-major, i.e. B.T is contiguous
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda").to(in_dtype).T.contiguous().T

    scale_a = torch.ones(M, 1, dtype=torch.float32, device="cuda")
    scale_b = torch.ones(1, N, dtype=torch.float32, device="cuda")

    try:
        for _ in range(WARMUP):
            torch._scaled_mm(A, B, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(ITERS):
            torch._scaled_mm(A, B, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) / ITERS * 1000

        flops = 2 * M * N * K
        tflops = flops / (ms / 1000) / 1e12
        print(f"  [{label}] {ms:.3f} ms  ({tflops:.1f} TFLOPS)")
        return ms, tflops
    except Exception as e:
        print(f"  [{label}] failed: {e}")
        return None, None


def bench_bmm(B, M, N, K, dtype, label):
    """Batched matmul — closer to actual attention pattern."""
    A = torch.randn(B, M, K, dtype=dtype, device="cuda")
    Bt = torch.randn(B, K, N, dtype=dtype, device="cuda")

    for _ in range(WARMUP):
        torch.bmm(A, Bt)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(ITERS):
        torch.bmm(A, Bt)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / ITERS * 1000

    flops = 2 * B * M * N * K
    tflops = flops / (ms / 1000) / 1e12
    print(f"  [{label}] {ms:.3f} ms  ({tflops:.1f} TFLOPS)")
    return ms, tflops


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    print("=" * 65)
    print("SM120 MMA throughput benchmark")
    print("=" * 65)

    # Self-attention Q·K^T shape: per-head (4096, 64) x (64, 4096)
    # But actual attention does this for all heads at once via bmm
    configs = [
        # (label, B, M, N, K)
        ("Self-attn QK^T (1 head)", 1, 4096, 4096, 64),
        ("Self-attn QK^T (32 heads)", 32, 4096, 4096, 64),
        ("Self-attn SV   (32 heads)", 32, 4096, 64, 4096),
        ("Cross-attn QK^T (32 heads)", 32, 4096, 512, 64),
        ("Large GEMM (cuBLAS ceiling)", 1, 4096, 4096, 4096),
    ]

    for label, B, M_cfg, N_cfg, K_cfg in configs:
        flops = 2 * B * M_cfg * N_cfg * K_cfg
        print(f"\n{label}: B={B} M={M_cfg} N={N_cfg} K={K_cfg}  ({flops/1e9:.1f} GFLOP)")

        if B == 1:
            bench_mm(M_cfg, N_cfg, K_cfg, torch.bfloat16, "BF16 mm")
            bench_mm(M_cfg, N_cfg, K_cfg, torch.float16, "FP16 mm")
        else:
            bench_bmm(B, M_cfg, N_cfg, K_cfg, torch.bfloat16, "BF16 bmm")
            bench_bmm(B, M_cfg, N_cfg, K_cfg, torch.float16, "FP16 bmm")

    # Scaled MM (FP8) — uses different HW path?
    print(f"\n--- Scaled matmul (FP8) ---")
    for label, M_cfg, N_cfg, K_cfg in [
        ("4096x4096x64", 4096, 4096, 64),
        ("4096x4096x4096", 4096, 4096, 4096),
    ]:
        print(f"\n{label}:")
        bench_scaled_mm(M_cfg, N_cfg, K_cfg, torch.float8_e4m3fn, "FP8 E4M3 scaled_mm")
        bench_scaled_mm(M_cfg, N_cfg, K_cfg, torch.float8_e5m2, "FP8 E5M2 scaled_mm")

    # Peak theoretical
    print("\n--- Reference ---")
    print("RTX 5060 Ti SM120 peak (SM80 MMA BF16): ~209 TFLOPS")
    print("RTX 5060 Ti SM120 peak (MXF4 native):   ~838 TFLOPS (4x, theoretical)")
