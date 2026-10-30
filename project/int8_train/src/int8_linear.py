"""int8 ConvRot base linears for LoRA training on a frozen DiT.

A converted ``nn.Linear`` runs its forward ``Y = X Wᵀ`` through
``comfy_kitchen.int8_linear`` (per-256-column Hadamard rotation, per-token int8
activations, per-row int8 weight). Its input gradient ``dX = dY W`` runs either in
bf16 against the kept weight (mode ``fwd``) or through the same kernel against a
second int8 copy of ``Wᵀ`` rotated along N (mode ``full``, the weight is dropped).
The base weight gradient is never computed.

``convert_(module, mode)`` swaps the class in place, so LoRA modules that hold a
reference to the original module object keep working.
"""

from __future__ import annotations

import re

import comfy_kitchen
import torch
import torch.nn as nn

from library.moss.int8 import INT8_MAX, rotate, row_scale

GROUP = 256
MODES = ("fwd", "full")

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


def quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(int8 (N, K) rotated along K, fp32 (N, 1) row scale)`` on ``w``'s device."""
    w_rot = rotate(w.float())
    scale = row_scale(w_rot)
    q = (w_rot / scale[:, None]).round().clamp(-INT8_MAX - 1, INT8_MAX)
    return q.to(torch.int8), scale[:, None].contiguous()


@torch.library.custom_op("int8_train::mm", mutates_args=())
def int8_mm(x: torch.Tensor, wq: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """``x (…, K) @ dequant(wq)ᵀ`` → bf16 ``(…, N)``."""
    y = comfy_kitchen.int8_linear(
        x.reshape(-1, x.shape[-1]).to(torch.bfloat16),
        wq,
        s,
        out_dtype=torch.bfloat16,
        convrot=True,
        convrot_groupsize=GROUP,
    )
    return y.reshape(*x.shape[:-1], wq.shape[0])


@int8_mm.register_fake
def _(x, wq, s):
    return x.new_empty((*x.shape[:-1], wq.shape[0]), dtype=torch.bfloat16)


@torch.library.custom_op("int8_train::linear_full", mutates_args=())
def linear_full(
    x: torch.Tensor,
    wq: torch.Tensor,
    s: torch.Tensor,
    wq_t: torch.Tensor,
    s_t: torch.Tensor,
) -> torch.Tensor:
    return int8_mm(x, wq, s)


@linear_full.register_fake
def _(x, wq, s, wq_t, s_t):
    return x.new_empty((*x.shape[:-1], wq.shape[0]), dtype=torch.bfloat16)


def _full_setup(ctx, inputs, output):
    x, _wq, _s, wq_t, s_t = inputs
    ctx.save_for_backward(wq_t, s_t)
    ctx.x_dtype = x.dtype


def _full_backward(ctx, dy):
    wq_t, s_t = ctx.saved_tensors
    dx = None
    if ctx.needs_input_grad[0]:
        dx = int8_mm(dy, wq_t, s_t).to(ctx.x_dtype)
    return dx, None, None, None, None


linear_full.register_autograd(_full_backward, setup_context=_full_setup)


@torch.library.custom_op("int8_train::linear_fwd", mutates_args=())
def linear_fwd(
    x: torch.Tensor, wq: torch.Tensor, s: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    return int8_mm(x, wq, s)


@linear_fwd.register_fake
def _(x, wq, s, w):
    return x.new_empty((*x.shape[:-1], wq.shape[0]), dtype=torch.bfloat16)


def _fwd_setup(ctx, inputs, output):
    x, _wq, _s, w = inputs
    ctx.save_for_backward(w)
    ctx.x_dtype = x.dtype


def _fwd_backward(ctx, dy):
    (w,) = ctx.saved_tensors
    dx = None
    if ctx.needs_input_grad[0]:
        dx = (dy.to(w.dtype) @ w).to(ctx.x_dtype)
    return dx, None, None, None


linear_fwd.register_autograd(_fwd_backward, setup_context=_fwd_setup)


class Int8TrainLinear(nn.Linear):
    """Only ever made by :func:`convert_`; never constructed directly."""

    int8_mode: str

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.int8_mode == "full":
            return linear_full(x, self.wq, self.s, self.wq_t, self.s_t)
        return linear_fwd(x, self.wq, self.s, self.weight)


def convert_(module: nn.Linear, mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"int8 mode {mode!r}, expected one of {MODES}")
    if module.bias is not None:
        raise ValueError("int8 base linears are bias-free")
    w = module.weight.detach()
    wq, s = quantize(w)
    module.register_buffer("wq", wq, persistent=False)
    module.register_buffer("s", s, persistent=False)
    if mode == "full":
        wq_t, s_t = quantize(w.t().contiguous())
        module.register_buffer("wq_t", wq_t, persistent=False)
        module.register_buffer("s_t", s_t, persistent=False)
        del module.weight
    module.int8_mode = mode
    module.__class__ = Int8TrainLinear


def convert_dit_(dit: nn.Module, mode: str) -> list[str]:
    """Convert every DiT block's large linears (not the llm_adapter's); returns the
    converted names."""
    names = []
    for name, mod in dit.named_modules():
        if (
            isinstance(mod, nn.Linear)
            and _DIT_BLOCK.match(name)
            and name.endswith(CONVERT_SUFFIXES)
        ):
            convert_(mod, mode)
            names.append(name)
    return names
