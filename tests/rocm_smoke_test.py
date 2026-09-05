#!/usr/bin/env python3
"""Small post-install check for the supported Windows ROCm path.

Two entry points on purpose: ``install.ps1`` / ``docs/guidelines/rocm.md`` run
this as a script (``python tests/rocm_smoke_test.py``) right after installing
the ROCm wheels, and pytest picks up ``test_rocm_smoke`` so ``make test-unit``
exercises the same body on real AMD hardware. Everywhere else (CUDA, CPU) the
test skips — it needs a live ROCm GPU, not a mock.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torchvision


def _validate_runtime() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm PyTorch cannot access an AMD GPU")
    if torch.version.hip is None:
        raise RuntimeError("installed PyTorch is not a ROCm build")
    if not str(torch.__version__).startswith("2.13.0+rocm10.0.0"):
        raise RuntimeError(f"expected torch 2.13.0+rocm10.0.0, got {torch.__version__}")
    if not str(torchvision.__version__).startswith("0.28.0+rocm10.0.0"):
        raise RuntimeError(
            f"expected torchvision 0.28.0+rocm10.0.0, got {torchvision.__version__}"
        )


def _run_visible_device() -> None:
    """Exercise the first GPU visible to this isolated child process."""
    device = torch.device("cuda", 0)
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", "unknown")
    print(f"testing: {name} ({arch})", flush=True)

    # Exercise device allocation, compile, SDPA, and backward rather than
    # accepting an import-only success, which misses runtime/device/Triton
    # failures and the attention path used by Anima on ROCm.
    @torch.compile
    def compiled_loss(value: torch.Tensor) -> torch.Tensor:
        return value.square().mean()

    x = torch.randn(64, 64, device=device, requires_grad=True)
    compiled_loss(x).backward()

    q = torch.randn(
        2,
        4,
        64,
        64,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    attention = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    attention.float().square().mean().backward()
    torch.cuda.synchronize()
    if not torch.isfinite(attention).all() or not torch.isfinite(q.grad).all():
        raise RuntimeError(f"ROCm PyTorch SDPA produced non-finite values on {name}")
    print(f"passed: {name} ({arch})", flush=True)


def run_smoke() -> None:
    """Raise unless every visible ROCm GPU passes the supported training path."""
    _validate_runtime()
    if os.environ.get("ANIMA_ROCM_SMOKE_CHILD") == "1":
        _run_visible_device()
        return

    # HIP can crash when one Windows process switches between unlike GPUs after
    # torch.compile initializes a device context. Test each adapter in a fresh
    # process instead; this also matches normal single-GPU training jobs.
    script = Path(__file__).resolve()
    for device_index in range(torch.cuda.device_count()):
        env = os.environ.copy()
        env["HIP_VISIBLE_DEVICES"] = str(device_index)
        env["ROCR_VISIBLE_DEVICES"] = str(device_index)
        env["ANIMA_ROCM_SMOKE_CHILD"] = "1"
        subprocess.run([sys.executable, str(script)], check=True, env=env)


@pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="needs a live ROCm GPU",
)
def test_rocm_smoke() -> None:
    run_smoke()


def main() -> int:
    print(f"torch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"HIP: {torch.version.hip}")
    if torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(device_index)
            arch = getattr(props, "gcnArchName", "unknown")
            print(
                f"GPU {device_index}: {torch.cuda.get_device_name(device_index)} "
                f"({arch})"
            )
    else:
        print("GPU: unavailable")

    run_smoke()

    print("ROCm tensor/compile/SDPA/backward smoke test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
