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

# Running this file directly does not put the repository on sys.path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _validate_runtime() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm PyTorch cannot access an AMD GPU")
    if torch.version.hip is None:
        raise RuntimeError("installed PyTorch is not a ROCm build")
    if str(torch.__version__) != "2.14.0+rocm10.1.0":
        raise RuntimeError(f"expected torch 2.14.0+rocm10.1.0, got {torch.__version__}")
    if not str(torchvision.__version__).startswith("0.29.0a0+rocm10.1.0"):
        raise RuntimeError(
            f"expected torchvision 0.29.0a0+rocm10.1.0, got {torchvision.__version__}"
        )


def _check_native_flash(device: torch.device) -> None:
    from flash_attn import flash_attn_func
    from library.runtime.backend import resolve_attention_mode
    from networks.attention_dispatch import AttentionParams, dispatch_attention

    if resolve_attention_mode("flash", torch) != "flash":
        raise RuntimeError("the rebuilt ROCm FlashAttention path was not selected")

    # Unequal, non-tile-aligned lengths cover the tail repair. Use independent
    # CPU/FP64 SDPA outputs and gradients rather than an import/finite-only check.
    for dtype in (torch.bfloat16, torch.float16):
        rtol, atol = (0.06, 0.015) if dtype == torch.bfloat16 else (0.015, 0.003)
        for dim in (32, 64, 128, 256):
            for causal in (False, True):
                for kv_heads in (2, 4):
                    q = (
                        torch.randn(1, 33, 4, dim, device=device, dtype=dtype) * 0.2
                    ).requires_grad_()
                    k = (
                        torch.randn(1, 65, kv_heads, dim, device=device, dtype=dtype)
                        * 0.2
                    ).requires_grad_()
                    v = (torch.randn_like(k) * 0.2).requires_grad_()
                    output = flash_attn_func(q, k, v, causal=causal)
                    dout = torch.randn_like(output) * 0.1
                    grads = torch.autograd.grad(output, (q, k, v), dout)
                    ref_q, ref_k, ref_v = [
                        t.detach().cpu().double().requires_grad_() for t in (q, k, v)
                    ]
                    # FlashAttention's causal mask aligns at bottom-right for
                    # Q != K; construct it explicitly for the SDPA reference.
                    mask = None
                    if causal:
                        mask = (
                            torch.arange(65)[None, :] <= torch.arange(33)[:, None] + 32
                        )
                    reference = torch.nn.functional.scaled_dot_product_attention(
                        ref_q.transpose(1, 2),
                        ref_k.transpose(1, 2),
                        ref_v.transpose(1, 2),
                        attn_mask=mask,
                        enable_gqa=kv_heads != 4,
                    ).transpose(1, 2)
                    ref_grads = torch.autograd.grad(
                        reference, (ref_q, ref_k, ref_v), dout.cpu().double()
                    )
                    for actual, expected in zip(
                        (output, *grads), (reference, *ref_grads)
                    ):
                        torch.testing.assert_close(
                            actual.detach().cpu().double(),
                            expected,
                            rtol=rtol,
                            atol=atol,
                        )

    # Exercise Anima's real masked/varlen dispatch and a compiled training loss.
    mask = torch.ones(1, 65, device=device, dtype=torch.int32)
    mask[:, 33:] = 0
    params = AttentionParams.create_attention_params_from_mask("flash", 0, mask)
    values = [
        torch.randn(
            1, 65, 4, 128, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        for _ in range(3)
    ]
    varlen_out = dispatch_attention(values[0], values[1], values[2], attn_params=params)
    varlen_out.float().square().mean().backward()
    for value in (varlen_out, *(t.grad for t in values)):
        if not torch.isfinite(value).all():
            raise RuntimeError(
                "Anima FlashAttention varlen path produced non-finite values"
            )

    @torch.compile
    def flash_loss(q, k, v):
        out = dispatch_attention(q, k, v, attn_params=AttentionParams("flash"))
        return out.float().square().mean()

    for _ in range(3):
        values = [
            torch.randn(
                1, 65, 4, 128, device=device, dtype=torch.bfloat16, requires_grad=True
            )
            for _ in range(3)
        ]
        loss = flash_loss(*values)
        loss.backward()
        if not torch.isfinite(loss) or not all(
            torch.isfinite(t.grad).all() for t in values
        ):
            raise RuntimeError(
                "compiled Anima FlashAttention backward produced non-finite values"
            )
    print(
        "native FlashAttention: 32 FP64 reference cases, varlen and compiled backward passed",
        flush=True,
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
    _check_native_flash(device)
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

    print("ROCm tensor/compile/SDPA/FlashAttention/backward smoke test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
