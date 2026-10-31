"""Anima's side of the int8 base linears (``library/moss/dit_int8/linear.py``): its
convert targets, its checkpoint keys, whole-DiT conversion. The linears themselves are
shared with MOSS and re-exported here for this folder's scripts."""

from __future__ import annotations

import re

import torch.nn as nn

from library.anima.weights import _SELF_ATTN_QKV_ORDER
from library.moss.dit_int8 import linear as _linear
from library.moss.dit_int8.linear import (  # noqa: F401  (this folder's import path)
    GROUP,
    MODES,
    Int8TrainLinear,
    convert_,
    int8_mm,
    linear_full,
    linear_fwd,
    quantize,
    stream_wqt_,
)

# per block: self qkv/out, cross q/out, MLP 1/2. Cross kv (M = context length) and
# adaLN (M = batch) stay bf16; phase 0 measured int8 slower there.
CONVERT_SUFFIXES = (
    "self_attn.qkv_proj",
    "self_attn.output_proj",
    "cross_attn.q_proj",
    "cross_attn.output_proj",
    "mlp.layer1",
    "mlp.layer2",
)
_DIT_BLOCK = re.compile(r"^blocks\.\d+\.")


def ckpt_keys(name: str) -> list[str]:
    """A converted module's weight keys in the checkpoint (``net.`` prefix, the fused
    ``qkv_proj`` split back into q / k / v)."""
    if name.endswith("self_attn.qkv_proj"):
        base = name.removesuffix("qkv_proj")
        return [f"net.{base}{p}.weight" for p in _SELF_ATTN_QKV_ORDER]
    return [f"net.{name}.weight"]


def load_int8_file(path: str, name_keys=ckpt_keys) -> tuple[dict, dict]:
    """``({module name: (wq, s)}, metadata)`` from a ``dit_gptq.py`` file (q / k / v
    rows concatenated into ``qkv_proj``)."""
    return _linear.load_int8_file(path, name_keys)


def convert_targets(dit: nn.Module) -> list[tuple[str, nn.Linear]]:
    """Every DiT block's large linears (not the llm_adapter's)."""
    return [
        (name, mod)
        for name, mod in dit.named_modules()
        if isinstance(mod, nn.Linear)
        and _DIT_BLOCK.match(name)
        and name.endswith(CONVERT_SUFFIXES)
    ]


def base_fingerprint(dit: nn.Module, targets=None) -> str:
    """``linear.base_fingerprint`` over ``targets``, by default the DiT's."""
    return _linear.base_fingerprint(targets or convert_targets(dit))


def convert_dit_(dit: nn.Module, mode: str, int8_file: str | None = None) -> list[str]:
    """Convert the DiT's large linears; returns the converted names. ``int8_file``
    supplies the forward copies, and must cover every target and match the loaded
    checkpoint's ``base_fingerprint``."""
    copies = {}
    if int8_file:
        copies, meta = load_int8_file(int8_file)
        if meta.get("base_fingerprint") != base_fingerprint(dit):
            raise ValueError(
                f"{int8_file} was made from {meta.get('source')}, not the loaded DiT"
            )
    names = []
    for name, mod in convert_targets(dit):
        if int8_file and name not in copies:
            raise ValueError(f"{int8_file} has no forward copy for {name}")
        convert_(mod, mode, copies.get(name))
        names.append(name)
    return names
