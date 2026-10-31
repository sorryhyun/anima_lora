"""int8 ConvRot base linears for LoRA training on a frozen DiT.

A converted ``nn.Linear`` runs its forward ``Y = X Wᵀ`` through
``comfy_kitchen.int8_linear`` (per-256-column Hadamard rotation, per-token int8
activations, per-row int8 weight). Its input gradient ``dX = dY W`` runs either in
bf16 against the kept weight (mode ``fwd``) or through the same kernel against a
second int8 copy of ``Wᵀ`` rotated along N (mode ``full``, the weight is dropped).
The base weight gradient is never computed. Mode ``infer`` keeps only the forward copy,
for generation on the int8 base. :func:`stream_wqt_` turns ``full`` linears into
``stream`` ones, whose ``wq_t`` lives on the host and reaches the card only for the
block the backward is in.

``convert_(module, mode)`` swaps the class in place, so LoRA modules that hold a
reference to the original module object keep working. The forward copy is round to
nearest unless a calibrated one is given; the backward copy ``wq_t`` is always round
to nearest. Both sit in :class:`HeldCopy` children, where the block swapper's walk
over ``.weight`` finds them.
"""

from __future__ import annotations

import hashlib
import json

import comfy_kitchen
import torch
import torch.nn as nn
from safetensors import safe_open
from torch.utils.flop_counter import register_flop_formula

from library.moss.dit_int8.stream import WqtStream
from library.moss.int8 import INT8_MAX, rotate, row_scale

GROUP = 256
MODES = ("fwd", "full", "infer")


def quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(int8 (N, K) rotated along K, fp32 (N, 1) row scale)`` on ``w``'s device."""
    w_rot = rotate(w.float())
    scale = row_scale(w_rot)
    q = (w_rot / scale[:, None]).round().clamp(-INT8_MAX - 1, INT8_MAX)
    return q.to(torch.int8), scale[:, None].contiguous()


@torch.library.custom_op("int8_train::mm", mutates_args=())
def int8_mm(x: torch.Tensor, wq: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """``x (…, K) @ dequant(wq)ᵀ`` → bf16 ``(…, N)``. The kernel rounds around its
    activation division to ``x``'s dtype: the forward passes fp32, the backward its bf16
    ``dY``."""
    y = comfy_kitchen.int8_linear(
        x.reshape(-1, x.shape[-1]),
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


_stream: WqtStream | None = None


def stream_wqt_(blocks: list[list[nn.Linear]]) -> WqtStream:
    """``full`` linears to ``stream``: ``blocks`` holds each DiT block's converted
    linears in one order, in execution order, with the same shapes block to block."""
    global _stream
    if _stream is not None:
        _stream.close()
    _stream = WqtStream(blocks)
    return _stream


@torch.library.custom_op("int8_train::dx_stream", mutates_args=())
def dx_stream(dy: torch.Tensor, s_t: torch.Tensor, sid: torch.Tensor) -> torch.Tensor:
    return int8_mm(dy, _stream.get(int(sid)), s_t)


@dx_stream.register_fake
def _(dy, s_t, sid):
    return dy.new_empty((*dy.shape[:-1], s_t.shape[0]), dtype=torch.bfloat16)


# sid is a tensor: as an int it would be a constant, one graph per block
@torch.library.custom_op("int8_train::linear_stream", mutates_args=())
def linear_stream(
    x: torch.Tensor,
    wq: torch.Tensor,
    s: torch.Tensor,
    s_t: torch.Tensor,
    sid: torch.Tensor,
) -> torch.Tensor:
    return int8_mm(x, wq, s)


@linear_stream.register_fake
def _(x, wq, s, s_t, sid):
    return x.new_empty((*x.shape[:-1], wq.shape[0]), dtype=torch.bfloat16)


def _stream_setup(ctx, inputs, output):
    x, _wq, _s, s_t, sid = inputs
    ctx.save_for_backward(s_t, sid)
    ctx.x_dtype = x.dtype


def _stream_backward(ctx, dy):
    s_t, sid = ctx.saved_tensors
    dx = None
    if ctx.needs_input_grad[0]:
        dx = dx_stream(dy, s_t, sid).to(ctx.x_dtype)
    return dx, None, None, None, None


linear_stream.register_autograd(_stream_backward, setup_context=_stream_setup)


# The partitioner's knapsack prices an op by its flops, and an op with no formula at 1:
# recomputed first at any budget. 0.5 = int8's 0.42 ms against bf16's 0.87 ms (gemm_bench).
@register_flop_formula(
    [torch.ops.int8_train.linear_full, torch.ops.int8_train.linear_stream]
    + [torch.ops.int8_train.linear_fwd, torch.ops.int8_train.mm]
)
def _int8_flops(x_shape, wq_shape, *args, **kwargs) -> int:
    return int(0.5 * x_shape[-2] * wq_shape[0] * wq_shape[1])


class HeldCopy(nn.Module):
    """One int8 copy as a child's ``weight``, the name the block swapper moves."""

    def __init__(self, copy: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", copy, persistent=False)


class Int8TrainLinear(nn.Linear):
    """Only ever made by :func:`convert_`; never constructed directly."""

    int8_mode: str

    @property
    def wq(self) -> torch.Tensor:
        return self.held_wq.weight

    @property
    def wq_t(self) -> torch.Tensor:
        return self.held_wq_t.weight

    def _apply(self, fn, recurse=True):
        # the trainer's unet.to(dtype=bf16) would round the fp32 row scales; keep
        # them fp32 and follow wq (int8, so dtype casts skip it) to its device
        scales = {k: v for k, v in self._buffers.items() if k in ("s", "s_t")}
        sid = self._buffers.get("sid")
        super()._apply(fn, recurse)
        for k, v in scales.items():
            self._buffers[k] = v.to(self.wq.device)
        if sid is not None:  # read in Python by the op, so it stays on the CPU
            self._buffers["sid"] = sid
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # cast outside the opaque op, where inductor fuses it into the producer (inside
        # it ran as its own copy kernel, ~3.5 ms per step: status.md)
        x = x.float()
        if self.int8_mode == "infer":
            return int8_mm(x, self.wq, self.s)
        if self.int8_mode == "full":
            return linear_full(x, self.wq, self.s, self.wq_t, self.s_t)
        if self.int8_mode == "stream":
            return linear_stream(x, self.wq, self.s, self.s_t, self.sid)
        return linear_fwd(x, self.wq, self.s, self.weight)


def load_int8_file(path: str, name_keys) -> tuple[dict, dict]:
    """``({module name: (wq, s)}, metadata)`` from a GPTQ file, for the converted
    module names. ``name_keys``: module name → its checkpoint weight keys, whose rows
    are concatenated."""
    out = {}
    with safe_open(path, framework="pt") as f:
        meta = dict(f.metadata() or {})
        keys = set(f.keys())
        names = json.loads(meta["int8_modules"])
        for name in names:
            ks = name_keys(name)
            missing = [k for k in ks if k not in keys]
            if missing:
                raise ValueError(f"{path}: no {missing[0]}")
            wq = torch.cat([f.get_tensor(k) for k in ks])
            s = torch.cat(
                [f.get_tensor(k.removesuffix("weight") + "weight_scale") for k in ks]
            )
            out[name] = (wq, s.float().reshape(-1, 1))
    return out, meta


def convert_(module: nn.Linear, mode: str, forward_copy=None) -> None:
    """``forward_copy``: a calibrated ``(wq, s)`` in place of round to nearest."""
    if mode not in MODES:
        raise ValueError(f"int8 mode {mode!r}, expected one of {MODES}")
    if module.bias is not None:
        raise ValueError("int8 base linears are bias-free")
    w = module.weight.detach()
    if forward_copy is None:
        wq, s = quantize(w)
    else:
        wq, s = (t.to(w.device) for t in forward_copy)
    module.held_wq = HeldCopy(wq)
    module.register_buffer("s", s, persistent=False)
    if mode == "full":
        wq_t, s_t = quantize(w.t().contiguous())
        module.held_wq_t = HeldCopy(wq_t)
        module.register_buffer("s_t", s_t, persistent=False)
    if mode != "fwd":
        del module.weight
    module.int8_mode = mode
    module.__class__ = Int8TrainLinear


def base_fingerprint(targets: list[tuple[str, nn.Linear]]) -> str:
    """sha256 over the bf16 bytes of the first and last target weights, before any
    conversion: an int8 file applies only to the checkpoint it was made from."""
    h = hashlib.sha256()
    for name, mod in (targets[0], targets[-1]):
        h.update(name.encode())
        h.update(
            mod.weight.detach()
            .to(torch.bfloat16)
            .cpu()
            .view(torch.int16)
            .numpy()
            .tobytes()
        )
    return h.hexdigest()
