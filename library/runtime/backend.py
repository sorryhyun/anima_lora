"""Runtime accelerator compatibility helpers."""

from __future__ import annotations

import importlib
import shutil
import sys
from typing import Any


def is_rocm(torch_module: Any) -> bool:
    """Return whether *torch_module* is a ROCm/HIP PyTorch build."""
    return getattr(getattr(torch_module, "version", None), "hip", None) is not None


def supports_rocm_flash_attention(torch_module: Any) -> bool:
    """Only enable the rebuilt Windows wheel on its declared torch/GPU targets."""
    if sys.platform != "win32" or not is_rocm(torch_module):
        return False
    # Do not even load a native extension built for a different torch ABI.
    if str(getattr(torch_module, "__version__", "")) != "2.14.0+rocm10.1.0":
        return False
    try:
        flash = importlib.import_module("flash_attn")
    except (ImportError, OSError):
        return False
    build = getattr(flash, "__rocm_build__", None)
    if not isinstance(build, dict) or not build.get("tail_fix"):
        return False
    if build.get("torch") != str(getattr(torch_module, "__version__", "")):
        return False
    if not callable(getattr(flash, "flash_attn_func", None)) or not callable(
        getattr(flash, "flash_attn_varlen_func", None)
    ):
        return False
    if not torch_module.cuda.is_available():
        return False
    props = torch_module.cuda.get_device_properties(torch_module.cuda.current_device())
    arch = getattr(props, "gcnArchName", "").split(":", 1)[0]
    return arch in build.get("archs", ())


def needs_rocm_attention_fallback(requested: str | None, torch_module: Any) -> bool:
    """Keep SDPA for ROCm environments without a matching native Flash build."""
    return (
        requested == "flash"
        and is_rocm(torch_module)
        and not supports_rocm_flash_attention(torch_module)
    )


def resolve_attention_mode(requested: str | None, torch_module: Any) -> str:
    """Honor Flash on compatible ROCm builds; retain SDPA as the fallback."""
    mode = requested or "torch"
    if needs_rocm_attention_fallback(requested, torch_module):
        return "torch"
    return mode


def diagnose_cuda_unavailable(torch_module: Any) -> str | None:
    """Explain why CUDA is unavailable on a machine that clearly has an NVIDIA GPU.

    Returns an actionable message, or None when there is nothing to say
    (CUDA works, or no NVIDIA GPU is detectable). The wrong-torch-build cases
    cover a Windows env that resolved the ROCm torch (GH #92); a plain
    ``uv sync`` restores the CUDA stack.
    """
    if torch_module.cuda.is_available():
        return None
    if shutil.which("nvidia-smi") is None:
        return None

    resync = "uv sync"
    if is_rocm(torch_module):
        return (
            "PyTorch is a ROCm build but this machine has an NVIDIA GPU — "
            f"everything would silently run on CPU. Fix: run `{resync}` in the "
            "install directory, then relaunch."
        )
    if getattr(getattr(torch_module, "version", None), "cuda", None) is None:
        return (
            "PyTorch is a CPU-only build but this machine has an NVIDIA GPU — "
            f"everything would silently run on CPU. Fix: run `{resync}` in the "
            "install directory, then relaunch."
        )
    return (
        "PyTorch has CUDA support but torch.cuda.is_available() is False — "
        "likely a driver problem (check `nvidia-smi` output and reboot/update "
        "the NVIDIA driver)."
    )


_warned_cuda_unavailable = False


def warn_if_cuda_unavailable(torch_module: Any) -> None:
    """Print the CUDA-unavailable diagnosis once per process."""
    global _warned_cuda_unavailable
    if _warned_cuda_unavailable:
        return
    message = diagnose_cuda_unavailable(torch_module)
    if message is not None:
        _warned_cuda_unavailable = True
        print(f"WARNING: {message}", file=sys.stderr, flush=True)
