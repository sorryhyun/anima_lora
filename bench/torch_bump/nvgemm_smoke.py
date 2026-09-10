#!/usr/bin/env python3
"""Does Inductor offer / select the NVGEMM backend on this GPU for LoRA-shaped GEMMs?

torch 2.14's "NVIDIA Universal GEMM" (``cutlass.operators``) is NOT in the
default ``max_autotune_gemm_backends`` — it needs
``TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN,TRITON,NVGEMM`` plus the
``nvidia-cutlass-operators`` / ``nvidia-matmul-heuristics`` packages. This
probe compiles one ``gelu(x @ w) + b`` per shape under ``mode="max-autotune"``
and times compiled vs eager. Run with ``TORCH_LOGS=autotuning`` and grep the
log for ``nv_universal_gemm`` / ``NoKernelImageForDevice`` to see whether the
NVGEMM choices were benchmarked or dropped.

Shapes (M, K, N): a DiT qkv-ish projection, a LoRA down (K→r) and a LoRA up
(r→N) at a 4096-token sequence.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", "ATEN,TRITON,NVGEMM")

import torch  # noqa: E402
from torch._inductor import config as icfg  # noqa: E402
from torch._inductor import utils as iu  # noqa: E402

SHAPES = [(4096, 1024, 3072), (4096, 1024, 32), (4096, 32, 1024)]


def f(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x @ w) + b


def _time(fn, *args, iters: int = 50) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - t0) / iters


def main() -> None:
    print(
        f"cute={iu.ensure_cute_available()} nvgemm={iu.ensure_nv_universal_gemm_available()} "
        f"heuristics={iu.ensure_nvmatmul_heuristics_available()} "
        f"backends={icfg.max_autotune_gemm_backends} cap={torch.cuda.get_device_capability()}",
        flush=True,
    )
    for m, k, n in SHAPES:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        torch._dynamo.reset()
        g = torch.compile(f, mode="max-autotune", dynamic=False)
        t0 = time.perf_counter()
        g(x, w, b)
        torch.cuda.synchronize()
        print(f"[{m}x{k}x{n}] compile {time.perf_counter() - t0:.1f}s", flush=True)
        for _ in range(3):
            g(x, w, b)
        print(f"[{m}x{k}x{n}] compiled {_time(g, x, w, b):.3f} ms/it", flush=True)
        print(f"[{m}x{k}x{n}] eager    {_time(f, x, w, b):.3f} ms/it", flush=True)


if __name__ == "__main__":
    main()
