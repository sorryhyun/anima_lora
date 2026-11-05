"""bf16 / fp16 / fp16-accumulate / int8 GEMM time on Anima's block shapes (``x @ Wᵀ``).

    make daemon-run ARGS="--stall-timeout 120 bench/anima_fp16/gemm.py"
"""

import torch
import torch.nn.functional as F

SHAPES = {
    "self qkv": (4096, 2048, 6144),
    "mlp 1": (4096, 2048, 8192),
    "mlp 2": (4096, 8192, 2048),
    "self out": (4096, 2048, 2048),
}
ARMS = ("bf16", "fp16", "fp16acc", "int8")


def bench(f, n=50):
    for _ in range(5):
        f()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(n):
        f()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def operands(arm, M, K, N):
    if arm == "int8":
        a = torch.randint(-127, 127, (M, K), dtype=torch.int8, device="cuda")
        b = torch.randint(-127, 127, (N, K), dtype=torch.int8, device="cuda").t()
        return lambda: torch._int_mm(a, b)
    dt = torch.bfloat16 if arm == "bf16" else torch.float16
    a = torch.randn(M, K, dtype=dt, device="cuda")
    b = torch.randn(N, K, dtype=dt, device="cuda")
    return lambda: F.linear(a, b)


def main():
    print(torch.cuda.get_device_name())
    for name, (M, K, N) in SHAPES.items():
        row = {}
        for arm in ARMS:
            torch.backends.cuda.matmul.allow_fp16_accumulation = arm == "fp16acc"
            ms = bench(operands(arm, M, K, N))
            row[arm] = f"{ms:.3f} ms {2 * M * K * N / ms / 1e9:.0f} TF"
        torch.backends.cuda.matmul.allow_fp16_accumulation = False
        print(name, row, flush=True)


if __name__ == "__main__":
    main()
