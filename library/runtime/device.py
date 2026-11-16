import functools
import gc
from typing import Optional, Union

import torch
import torch.nn as nn


def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()


def free_vram_gb(device: int = 0) -> float:
    """``mem_get_info`` free bytes in GiB (the allocator's reserve and other
    processes included, unlike ``total - max_allocated``)."""
    free, _total = torch.cuda.mem_get_info(device)
    return free / 1024**3


def clean_memory_on_device(device: Optional[Union[str, torch.device]]):
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
