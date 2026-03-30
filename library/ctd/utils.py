"""Utility functions vendored from manga-image-translator (GPL-3.0).

Contains: letterbox, fuse_conv_and_bn, weight init helpers.
Source: github.com/zyddnys/manga-image-translator
"""

import math

import cv2
import numpy as np
import torch
import torch.nn as nn


def letterbox(
    im,
    new_shape=(640, 640),
    color=(0, 0, 0),
    auto=False,
    scaleFill=False,
    scaleup=True,
    stride=128,
):
    """Resize and pad image while meeting stride-multiple constraints."""
    shape = im.shape[:2]
    if not isinstance(new_shape, tuple):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dh, dw = int(dh), int(dw)

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


def fuse_conv_and_bn(conv, bn):
    """Fuse Conv2d + BatchNorm2d into a single Conv2d."""
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    b_conv = (
        torch.zeros(conv.weight.size(0), device=conv.weight.device)
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps)
    )
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fusedconv


def check_anchor_order(m):
    a = m.anchors.prod(-1).view(-1)
    da = a[-1] - a[0]
    ds = m.stride[-1] - m.stride[0]
    if da.sign() != ds.sign():
        m.anchors[:] = m.anchors.flip(0)


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


def make_divisible(x, divisor):
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor


def check_version(
    current="0.0.0", minimum="0.0.0", name="version ", pinned=False, hard=False
):
    from packaging import version

    current, minimum = (version.parse(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)
    if hard:
        assert result, f"{name}{minimum} required, but {name}{current} is installed"
    return result


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0)
