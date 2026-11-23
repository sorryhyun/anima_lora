"""The pixel head's depthwise convs as opaque custom ops, for compiling over many sizes.

Inductor's ``convolution_backward_lowering`` turns the conv-backward strides into ints
(``guard_int_seq``), which pins (gh, gw): a head compiled with ``dynamic=True`` still
recompiles for every distinct image size and spills to eager at the recompile limit.
Behind a custom op the conv runs the eager cuDNN kernels, forward and backward, and the
rest of the graph keeps H and W symbolic: one compile covers every size.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.library.custom_op("anima_pixel::dwconv", mutates_args=())
def dwconv(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, pad: int) -> torch.Tensor:
    # contiguous: cuDNN may hand back channels-last, the fake impl says contiguous
    return F.conv2d(x, w, b, padding=pad, groups=x.shape[1]).contiguous()


@dwconv.register_fake
def _(x, w, b, pad):
    return x.new_empty(x.shape)


@torch.library.custom_op("anima_pixel::dwconv_bwd", mutates_args=())
def dwconv_bwd(
    g: torch.Tensor, x: torch.Tensor, w: torch.Tensor, pad: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gx, gw, gb = torch.ops.aten.convolution_backward(
        g, x, w, [w.shape[0]], [1, 1], [pad, pad], [1, 1], False, [0, 0],
        x.shape[1], [True, True, True],
    )  # fmt: skip
    return gx.contiguous(), gw.contiguous(), gb.contiguous()


@dwconv_bwd.register_fake
def _(g, x, w, pad):
    return x.new_empty(x.shape), w.new_empty(w.shape), w.new_empty(w.shape[0])


def _setup(ctx, inputs, output):
    x, w, _, pad = inputs
    ctx.save_for_backward(x, w)
    ctx.pad = pad


def _backward(ctx, g):
    x, w = ctx.saved_tensors
    gx, gw, gb = dwconv_bwd(g.contiguous(), x, w, ctx.pad)
    return gx, gw, gb, None


dwconv.register_autograd(_backward, setup_context=_setup)


class OpaqueDWConv2d(nn.Conv2d):
    """A stride-1 depthwise ``nn.Conv2d`` with bias, through :func:`dwconv`. Same
    parameters and state-dict keys; autocast as ``F.conv2d`` would apply it."""

    def forward(self, x):
        dev = x.device.type
        dt = (
            torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else x.dtype
        )
        with torch.autocast(dev, enabled=False):
            return dwconv(
                x.to(dt), self.weight.to(dt), self.bias.to(dt), self.padding[0]
            )


def opaque_dwconv_(head: nn.Module) -> nn.Module:
    """Swap every ConvNeXt block's depthwise conv in ``head`` for
    :class:`OpaqueDWConv2d`, in place; → ``head``."""
    from library.anima.pixel_head import ConvNeXtBlock

    for m in head.modules():
        if isinstance(m, ConvNeXtBlock):
            c = m.dw
            assert (
                c.groups == c.in_channels and c.stride == (1, 1) and c.bias is not None
            )
            c.__class__ = OpaqueDWConv2d
    return head
