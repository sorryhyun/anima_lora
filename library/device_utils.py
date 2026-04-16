import functools
import gc
from typing import Optional, Union

import torch
import torch.nn as nn


def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()


def clean_memory_on_device(device: Optional[Union[str, torch.device]]):
    r"""
    Clean memory on the specified device, will be called from training scripts.
    """
    gc.collect()
    if device is None:
        return
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()


def synchronize_device(device: Optional[Union[str, torch.device]]):
    if device is None:
        return
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize()


def weighs_to_device(layer: nn.Module, device: torch.device):
    for module in layer.modules():
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data = module.weight.data.to(device, non_blocking=True)


def str_to_dtype(
    s: Optional[str], default_dtype: Optional[torch.dtype] = None
) -> torch.dtype:
    """Convert a string to a torch.dtype."""
    if s is None:
        return default_dtype
    if s in ["bf16", "bfloat16"]:
        return torch.bfloat16
    elif s in ["fp16", "float16"]:
        return torch.float16
    elif s in ["fp32", "float32", "float"]:
        return torch.float32
    elif s in ["fp8_e4m3fn", "e4m3fn", "float8_e4m3fn"]:
        return torch.float8_e4m3fn
    elif s in ["fp8_e4m3fnuz", "e4m3fnuz", "float8_e4m3fnuz"]:
        return torch.float8_e4m3fnuz
    elif s in ["fp8_e5m2", "e5m2", "float8_e5m2"]:
        return torch.float8_e5m2
    elif s in ["fp8_e5m2fnuz", "e5m2fnuz", "float8_e5m2fnuz"]:
        return torch.float8_e5m2fnuz
    elif s in ["fp8", "float8"]:
        return torch.float8_e4m3fn  # default fp8
    else:
        raise ValueError(f"Unsupported dtype: {s}")


@functools.lru_cache(maxsize=None)
def get_preferred_device() -> torch.device:
    r"""
    Do not call this function from training scripts. Use accelerator.device instead.
    """
    device = torch.device("cuda")
    print(f"get_preferred_device() -> {device}")
    return device
