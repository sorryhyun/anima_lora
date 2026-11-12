"""Compiled decode for PE-mode ``generate`` on the INT8 TE.

Eager HF decode is launch-bound here (~69 ms/token at any batch, the card at ~110 W): every
token runs 32 layers of int8 ConvRot GEMMs, PE LoRA branches and GDN updates as separate
kernels. ``cache_implementation="static"`` makes ``generate`` compile the decode forward;
comfy_kitchen's W8A8 path reads data pointers (dlpack), which dynamo cannot trace, so
``enable()`` routes ``Int8ConvRotLinear`` through an opaque custom op first.
``bench_speed.py`` measured 18 ms/token against 66 at batch 1, greedy.
"""

from __future__ import annotations

import torch

from library.moss import int8


@torch.library.custom_op("pe_lora::int8_convrot_linear", mutates_args=())
def int8_convrot_linear(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, group: int
) -> torch.Tensor:
    import comfy_kitchen

    return comfy_kitchen.int8_linear(
        x,
        weight,
        scale,
        out_dtype=x.dtype,
        convrot=group > 0,
        convrot_groupsize=group or int8.GROUP,
    )


@int8_convrot_linear.register_fake
def _(x, weight, scale, group):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def _opaque_forward(self, x):
    if not self.act:
        return _eager_forward(self, x)
    return int8_convrot_linear(x, self.weight, self.weight_scale, self.group)


_eager_forward = int8.Int8ConvRotLinear.forward


def enable() -> dict:
    """Patch the int8 linears; returns the ``generate`` kwargs for the compiled path."""
    int8.Int8ConvRotLinear.forward = _opaque_forward
    return {"cache_implementation": "static"}
