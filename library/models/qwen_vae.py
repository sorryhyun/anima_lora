# Copied and modified from Diffusers (via Musubi-Tuner). Original copyright notice follows.

# Copyright 2025 The Qwen-Image Team, Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We gratefully acknowledge the Wan Team for their outstanding contributions.
# QwenImageVAE is further fine-tuned from the Wan Video VAE to achieve improved performance.
# For more information about the Wan VAE, please refer to:
# - GitHub: https://github.com/Wan-Video/Wan2.1
# - arXiv: https://arxiv.org/abs/2503.20314

import json
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint
import numpy as np

from library.env import resolve_under_home
from library.io.safetensors import load_safetensors

from library.log import setup_logging

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)


CACHE_T = 2

SCALE_FACTOR = 8  # VAE downsampling factor


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        # make sure sample is on the same device as the parameters and has same dtype
        if (
            generator is not None
            and generator.device.type != self.parameters.device.type
        ):
            rand_device = generator.device
        else:
            rand_device = self.parameters.device
        sample = torch.randn(
            self.mean.shape,
            generator=generator,
            device=rand_device,
            dtype=self.parameters.dtype,
        ).to(self.parameters.device)
        x = self.mean + self.std * sample
        return x

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(
        self, sample: torch.Tensor, dims: Tuple[int, ...] = [1, 2, 3]
    ) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        return self.mean


class ChunkedConv2d(nn.Conv2d):
    """Conv2d that processes input in height-wise chunks to cut peak memory."""

    def __init__(self, *args, **kwargs):
        if "spatial_chunk_size" in kwargs:
            self.spatial_chunk_size = kwargs.pop("spatial_chunk_size", None)
        else:
            self.spatial_chunk_size = None
        super().__init__(*args, **kwargs)
        assert self.padding_mode == "zeros", "Only 'zeros' padding mode is supported."
        assert self.dilation == (1, 1), "Only dilation=1 is supported."
        assert self.groups == 1, "Only groups=1 is supported."
        assert self.kernel_size[0] == self.kernel_size[1], (
            "Only square kernels are supported."
        )
        assert self.stride[0] == self.stride[1], "Only equal strides are supported."
        self.original_padding = self.padding
        self.padding = (0, 0)  # We handle padding manually in forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Chunk along height only; skip entirely if input is under threshold.
        if (
            self.spatial_chunk_size is None
            or x.shape[2]
            <= self.spatial_chunk_size
            + self.kernel_size[0]
            + self.spatial_chunk_size // 4
        ):
            self.padding = self.original_padding
            x = super().forward(x)
            self.padding = (0, 0)
            return x

        org_shape = x.shape

        # Overlapping chunks needed unless kernel size is 1
        overlap = self.kernel_size[0] // 2
        if self.original_padding[0] == 0:
            overlap = 0

        # stride > 1 is handled by QwenImageVAE's manual zero-padding upstream
        y_height = org_shape[2] // self.stride[0]
        y_width = org_shape[3] // self.stride[1]
        y = torch.zeros(
            (org_shape[0], self.out_channels, y_height, y_width),
            dtype=x.dtype,
            device=x.device,
        )
        yi = 0
        i = 0
        while i < org_shape[2]:
            si = i if i == 0 else i - overlap
            ei = i + self.spatial_chunk_size + overlap + self.stride[0] - 1

            # Fold a too-small remainder into the last chunk
            if ei > org_shape[2] or ei + self.spatial_chunk_size // 4 > org_shape[2]:
                ei = org_shape[2]

            chunk = x[:, :, si:ei, :]

            if i == 0 and overlap > 0:  # first chunk: pad all but bottom
                chunk = torch.nn.functional.pad(
                    chunk, (overlap, overlap, overlap, 0), mode="constant", value=0
                )
            elif ei == org_shape[2] and overlap > 0:  # last chunk: pad all but top
                chunk = torch.nn.functional.pad(
                    chunk, (overlap, overlap, 0, overlap), mode="constant", value=0
                )
            elif overlap > 0:  # middle chunk: pad left/right only
                chunk = torch.nn.functional.pad(
                    chunk, (overlap, overlap), mode="constant", value=0
                )

            chunk = super().forward(chunk)
            y[:, :, yi : yi + chunk.shape[2], :] = chunk
            yi += chunk.shape[2]
            del chunk

            if ei == org_shape[2]:
                break
            i += self.spatial_chunk_size

        assert yi == y_height, f"yi={yi}, y_height={y_height}"

        return y


class QwenImageCausalConv3d(nn.Conv3d):
    r"""Conv3d enforcing causality in the time dimension, with feature caching for streaming inference."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        spatial_chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)
        self.spatial_chunk_size = spatial_chunk_size
        self._supports_spatial_chunking = (
            self.groups == 1
            and self.dilation[1] == 1
            and self.dilation[2] == 1
            and self.stride[1] == 1
            and self.stride[2] == 1
        )

    def _forward_chunked_height(self, x: torch.Tensor) -> torch.Tensor:
        chunk_size = self.spatial_chunk_size
        if chunk_size is None or chunk_size <= 0:
            return super().forward(x)
        if not self._supports_spatial_chunking:
            return super().forward(x)

        kernel_h = self.kernel_size[1]
        if kernel_h <= 1 or x.shape[3] <= chunk_size:
            return super().forward(x)

        receptive_h = kernel_h
        out_h = x.shape[3] - receptive_h + 1
        if out_h <= 0:
            return super().forward(x)

        y0 = 0
        out = None
        while y0 < out_h:
            y1 = min(y0 + chunk_size, out_h)
            in0 = y0
            in1 = y1 + receptive_h - 1
            out_chunk = super().forward(x[:, :, :, in0:in1, :])
            if out is None:
                out_shape = list(out_chunk.shape)
                out_shape[3] = out_h
                out = out_chunk.new_empty(out_shape)
            out[:, :, :, y0:y1, :] = out_chunk
            y0 = y1

        return out

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return self._forward_chunked_height(x)


class Folded2DConv(nn.Module):
    r"""2D-equivalent of a single-image (``T=1``) :class:`QwenImageCausalConv3d`.

    For one frame, causal temporal padding zero-pads with the real frame in the
    last temporal slot, so only the **last** temporal tap of the 3D kernel sees
    data — hence the equivalent is a plain ``nn.Conv2d`` with weight
    ``weight[:, :, -1, :, :]`` run on the squeezed ``(B, C, H, W)`` frame.
    Bit-exact per-layer (fp64-verified); whole-VAE matches within bf16 noise
    at ~2x speed / ~0.65-0.7x memory (see ``_archive/bench/qwen_vae_2d/``).
    Accepts the replaced conv's ``cache_x`` arg (ignored — no-op here).
    ONLY valid for ``T == 1``; asserts on every forward.
    """

    def __init__(self, c3d: "QwenImageCausalConv3d") -> None:
        super().__init__()
        # _padding layout: (pad_w, pad_w, pad_h, pad_h, 2*pad_t, 0)
        pad_w, pad_h = c3d._padding[0], c3d._padding[2]
        self.conv = nn.Conv2d(
            c3d.in_channels,
            c3d.out_channels,
            kernel_size=(c3d.kernel_size[1], c3d.kernel_size[2]),
            stride=(c3d.stride[1], c3d.stride[2]),
            padding=(pad_h, pad_w),
            bias=c3d.bias is not None,
        ).to(device=c3d.weight.device, dtype=c3d.weight.dtype)
        with torch.no_grad():
            self.conv.weight.copy_(c3d.weight[:, :, -1, :, :])
            if c3d.bias is not None:
                self.conv.bias.copy_(c3d.bias)

    def forward(self, x: torch.Tensor, cache_x=None) -> torch.Tensor:
        assert x.shape[2] == 1, (
            f"Folded2DConv only supports single-frame (T=1) input, got T={x.shape[2]}"
        )
        x = self.conv(x[:, :, 0])  # (B,C,1,H,W) -> (B,C,H,W) -> (B,C,H',W')
        return x[:, :, None]  # -> (B,C,1,H',W')


class QwenImageRMS_norm(nn.Module):
    r"""RMS normalization layer."""

    def __init__(
        self,
        dim: int,
        channel_first: bool = True,
        images: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return (
            F.normalize(x, dim=(1 if self.channel_first else -1))
            * self.scale
            * self.gamma
            + self.bias
        )


class QwenImageUpsample(nn.Upsample):
    r"""Upsample while preserving the input dtype (upsamples in fp32 internally)."""

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class QwenImageResample(nn.Module):
    r"""Up/downsampling module. ``mode`` in {none, upsample2d, upsample3d, downsample2d, downsample3d};
    the 3d variants add a causal temporal conv."""

    def __init__(self, dim: int, mode: str) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == "upsample2d":
            self.resample = nn.Sequential(
                QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                ChunkedConv2d(dim, dim // 2, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                ChunkedConv2d(dim, dim // 2, 3, padding=1),
            )
            self.time_conv = QwenImageCausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0)
            )

        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), ChunkedConv2d(dim, dim, 3, stride=(2, 2))
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), ChunkedConv2d(dim, dim, 3, stride=(2, 2))
            )
            self.time_conv = QwenImageCausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if (
                        cache_x.shape[2] < 2
                        and feat_cache[idx] is not None
                        and feat_cache[idx] != "Rep"
                    ):
                        cache_x = torch.cat(
                            [
                                feat_cache[idx][:, :, -1, :, :]
                                .unsqueeze(2)
                                .to(cache_x.device),
                                cache_x,
                            ],
                            dim=2,
                        )
                    if (
                        cache_x.shape[2] < 2
                        and feat_cache[idx] is not None
                        and feat_cache[idx] == "Rep"
                    ):
                        cache_x = torch.cat(
                            [torch.zeros_like(cache_x).to(cache_x.device), cache_x],
                            dim=2,
                        )
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2)
                    )
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


class QwenImageResidualBlock(nn.Module):
    r"""Residual block: norm -> act -> causal conv, twice, plus a shortcut conv."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
    ) -> None:
        assert non_linearity in ["silu"], (
            "Only 'silu' non-linearity is supported currently."
        )
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinearity = nn.SiLU()  # get_activation(non_linearity)

        self.norm1 = QwenImageRMS_norm(in_dim, images=False)
        self.conv1 = QwenImageCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = QwenImageRMS_norm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = QwenImageCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = (
            QwenImageCausalConv3d(in_dim, out_dim, 1)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.conv_shortcut(x)

        x = self.norm1(x)
        x = self.nonlinearity(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )

            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )

            x = self.conv2(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv2(x)

        return x + h


class QwenImageAttentionBlock(nn.Module):
    r"""Causal self-attention with a single head."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.norm = QwenImageRMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        batch_size, channels, time, height, width = x.size()

        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
        x = self.norm(x)

        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)

        x = F.scaled_dot_product_attention(q, k, v)

        x = (
            x.squeeze(1)
            .permute(0, 2, 1)
            .reshape(batch_size * time, channels, height, width)
        )

        x = self.proj(x)

        # [(b*t), c, h, w] -> [b, c, t, h, w]
        x = x.view(batch_size, time, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)

        return x + identity


class QwenImageMidBlock(nn.Module):
    """Middle block for QwenImageVAE encoder and decoder."""

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        num_layers: int = 1,
    ):
        super().__init__()
        self.dim = dim

        resnets = [QwenImageResidualBlock(dim, dim, dropout, non_linearity)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(QwenImageAttentionBlock(dim))
            resnets.append(QwenImageResidualBlock(dim, dim, dropout, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x = self.resnets[0](x, feat_cache, feat_idx)

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)

            x = resnet(x, feat_cache, feat_idx)

        return x


class QwenImageEncoder3d(nn.Module):
    r"""3D encoder: conv_in -> downsample blocks -> mid block -> norm/act/conv_out."""

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
        input_channels: int = 3,
        non_linearity: str = "silu",
    ):
        super().__init__()
        assert non_linearity in ["silu"], (
            "Only 'silu' non-linearity is supported currently."
        )
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.nonlinearity = nn.SiLU()  # get_activation(non_linearity)

        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        self.conv_in = QwenImageCausalConv3d(input_channels, dims[0], 3, padding=1)

        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    QwenImageResidualBlock(in_dim, out_dim, dropout)
                )
                if scale in attn_scales:
                    self.down_blocks.append(QwenImageAttentionBlock(out_dim))
                in_dim = out_dim

            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                self.down_blocks.append(QwenImageResample(out_dim, mode=mode))
                scale /= 2.0

        self.mid_block = QwenImageMidBlock(
            out_dim, dropout, non_linearity, num_layers=1
        )

        self.norm_out = QwenImageRMS_norm(out_dim, images=False)
        self.conv_out = QwenImageCausalConv3d(out_dim, z_dim, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        ## downsamples
        for layer in self.down_blocks:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        x = self.mid_block(x, feat_cache, feat_idx)

        ## head
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


class QwenImageUpBlock(nn.Module):
    """Upsampling block for the QwenImageVAE decoder."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: Optional[str] = None,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(
                QwenImageResidualBlock(current_dim, out_dim, dropout, non_linearity)
            )
            current_dim = out_dim

        self.resnets = nn.ModuleList(resnets)

        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList(
                [QwenImageResample(out_dim, mode=upsample_mode)]
            )

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        for resnet in self.resnets:
            if feat_cache is not None:
                x = resnet(x, feat_cache, feat_idx)
            else:
                x = resnet(x)

        if self.upsamplers is not None:
            if feat_cache is not None:
                x = self.upsamplers[0](x, feat_cache, feat_idx)
            else:
                x = self.upsamplers[0](x)
        return x


class QwenImageDecoder3d(nn.Module):
    r"""3D decoder: conv_in -> mid block -> upsample blocks -> norm/act/conv_out."""

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
        output_channels: int = 3,
        non_linearity: str = "silu",
    ):
        super().__init__()
        assert non_linearity in ["silu"], (
            "Only 'silu' non-linearity is supported currently."
        )
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        self.nonlinearity = nn.SiLU()  # get_activation(non_linearity)

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        self.conv_in = QwenImageCausalConv3d(z_dim, dims[0], 3, padding=1)

        self.mid_block = QwenImageMidBlock(
            dims[0], dropout, non_linearity, num_layers=1
        )

        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i > 0:
                in_dim = in_dim // 2

            upsample_mode = None
            if i != len(dim_mult) - 1:
                upsample_mode = "upsample3d" if temperal_upsample[i] else "upsample2d"

            up_block = QwenImageUpBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                num_res_blocks=num_res_blocks,
                dropout=dropout,
                upsample_mode=upsample_mode,
                non_linearity=non_linearity,
            )
            self.up_blocks.append(up_block)

            if upsample_mode is not None:
                scale *= 2.0

        self.norm_out = QwenImageRMS_norm(out_dim, images=False)
        self.conv_out = QwenImageCausalConv3d(out_dim, output_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        x = self.mid_block(x, feat_cache, feat_idx)

        # Checkpointing up_blocks is only correct with feat_cache=None — recompute
        # would otherwise re-mutate the stateful causal-conv cache. See
        # enable_gradient_checkpointing().
        use_ckpt = (
            self.gradient_checkpointing
            and feat_cache is None
            and torch.is_grad_enabled()
            and x.requires_grad
        )
        for up_block in self.up_blocks:
            if use_ckpt:
                x = torch_checkpoint.checkpoint(up_block, x, use_reentrant=False)
            else:
                x = up_block(x, feat_cache, feat_idx)

        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


class AutoencoderKLQwenImage(
    nn.Module
):  # ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""KL-VAE for encoding videos/images into latents and decoding back."""

    _supports_gradient_checkpointing = False

    # @register_to_config
    def __init__(
        self,
        base_dim: int = 96,
        z_dim: int = 16,
        dim_mult: Tuple[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: List[float] = [],
        temperal_downsample: List[bool] = [False, True, True],
        dropout: float = 0.0,
        latents_mean: List[float] = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ],
        latents_std: List[float] = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ],
        input_channels: int = 3,
        spatial_chunk_size: Optional[int] = None,
        disable_cache: bool = False,
    ) -> None:
        super().__init__()

        self.z_dim = z_dim
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]
        self.latents_mean = latents_mean
        self.latents_std = latents_std

        self.encoder = QwenImageEncoder3d(
            base_dim,
            z_dim * 2,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_downsample,
            dropout,
            input_channels,
        )
        self.quant_conv = QwenImageCausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = QwenImageCausalConv3d(z_dim, z_dim, 1)

        self.decoder = QwenImageDecoder3d(
            base_dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_upsample,
            dropout,
            input_channels,
        )

        self.spatial_compression_ratio = 2 ** len(self.temperal_downsample)

        # Decode one batch item at a time to save memory
        self.use_slicing = False

        # Decode in overlapping spatial tiles, blended together, to save memory
        self.use_tiling = False

        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256

        # Overlap between adjacent tiles
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192

        # Cached for clear_cache() speedup
        self._cached_conv_counts = {
            "decoder": sum(
                isinstance(m, QwenImageCausalConv3d) for m in self.decoder.modules()
            )
            if self.decoder is not None
            else 0,
            "encoder": sum(
                isinstance(m, QwenImageCausalConv3d) for m in self.encoder.modules()
            )
            if self.encoder is not None
            else 0,
        }

        self.spatial_chunk_size = None
        if spatial_chunk_size is not None and spatial_chunk_size > 0:
            self.enable_spatial_chunking(spatial_chunk_size)

        self.cache_disabled = False
        if disable_cache:
            self.disable_cache()

        self.is_2d = False

    @property
    def dtype(self):
        return self.encoder.parameters().__next__().dtype

    @property
    def device(self):
        return self.encoder.parameters().__next__().device

    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_sample_stride_height: Optional[float] = None,
        tile_sample_stride_width: Optional[float] = None,
    ) -> None:
        r"""Enable tiled encode/decode: splits the input into overlapping tiles to
        cut memory use for large images. Stride < min size sets the overlap
        (blended on merge to avoid seams)."""
        self.use_tiling = True
        self.tile_sample_min_height = (
            tile_sample_min_height or self.tile_sample_min_height
        )
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_stride_height = (
            tile_sample_stride_height or self.tile_sample_stride_height
        )
        self.tile_sample_stride_width = (
            tile_sample_stride_width or self.tile_sample_stride_width
        )

    def disable_tiling(self) -> None:
        r"""Undo :meth:`enable_tiling`."""
        self.use_tiling = False

    def enable_slicing(self) -> None:
        r"""Decode one batch item at a time to save memory."""
        self.use_slicing = True

    def disable_slicing(self) -> None:
        r"""Undo :meth:`enable_slicing`."""
        self.use_slicing = False

    def enable_spatial_chunking(self, spatial_chunk_size: int) -> None:
        r"""Chunk all causal Conv3d/Conv2d layers along height to cut memory use."""
        if spatial_chunk_size is None or spatial_chunk_size <= 0:
            raise ValueError(
                f"`spatial_chunk_size` must be a positive integer, got {spatial_chunk_size}."
            )
        self.spatial_chunk_size = int(spatial_chunk_size)
        for module in self.modules():
            if isinstance(module, QwenImageCausalConv3d):
                module.spatial_chunk_size = self.spatial_chunk_size
            elif isinstance(module, ChunkedConv2d):
                module.spatial_chunk_size = self.spatial_chunk_size

    def disable_spatial_chunking(self) -> None:
        r"""Undo :meth:`enable_spatial_chunking`."""
        self.spatial_chunk_size = None
        for module in self.modules():
            if isinstance(module, QwenImageCausalConv3d):
                module.spatial_chunk_size = None
            elif isinstance(module, ChunkedConv2d):
                module.spatial_chunk_size = None

    def convert_to_2d(self) -> int:
        r"""Fold every causal Conv3d into an equivalent 2D conv (image-only).

        After this the VAE handles **only single images** (``T=1``) — multi-frame
        encode/decode will assert. ~2x faster / ~0.65-0.7x memory vs the 3D path,
        numerically equivalent within bf16 noise. Returns the number of convs
        folded.
        """
        n = 0
        for parent in self.modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, QwenImageCausalConv3d):
                    setattr(parent, name, Folded2DConv(child))
                    n += 1
        self.disable_cache()
        self.is_2d = True
        return n

    def enable_gradient_checkpointing(self) -> None:
        r"""Checkpoint the decoder up-blocks to cut backward activation memory.

        Forces cache-free decode (calls ``disable_cache()``) — checkpoint
        recompute would otherwise corrupt the stateful causal-conv ``feat_cache``.
        """
        self.decoder.gradient_checkpointing = True
        if not self.cache_disabled:
            self.disable_cache()

    def disable_gradient_checkpointing(self) -> None:
        r"""Undo :meth:`enable_gradient_checkpointing` (cache stays as-is)."""
        self.decoder.gradient_checkpointing = False

    def disable_cache(self) -> None:
        r"""Disable the feature cache in encoder and decoder."""
        self.cache_disabled = True
        self.clear_cache = lambda: None
        self._feat_map = None
        self._enc_feat_map = None

    def clear_cache(self):
        def _count_conv3d(model):
            count = 0
            for m in model.modules():
                if isinstance(m, QwenImageCausalConv3d):
                    count += 1
            return count

        self._conv_num = _count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = _count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    def _encode(self, x: torch.Tensor):
        _, _, num_frame, height, width = x.shape
        assert num_frame == 1 or not self.cache_disabled, (
            "Caching must be enabled for encoding multiple frames."
        )

        if self.use_tiling and (
            width > self.tile_sample_min_width or height > self.tile_sample_min_height
        ):
            return self.tiled_encode(x)

        self.clear_cache()
        iter_ = 1 + (num_frame - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = torch.cat([out, out_], 2)

        enc = self.quant_conv(out)
        self.clear_cache()
        return enc

    # @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[Dict[str, torch.Tensor], Tuple[DiagonalGaussianDistribution]]:
        r"""Encode a batch of images into a `DiagonalGaussianDistribution` (dict if `return_dict`)."""
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)
        return {"latent_dist": posterior}

    def _decode(self, z: torch.Tensor, return_dict: bool = True):
        _, _, num_frame, height, width = z.shape
        assert num_frame == 1 or not self.cache_disabled, (
            "Caching must be enabled for encoding multiple frames."
        )
        tile_latent_min_height = (
            self.tile_sample_min_height // self.spatial_compression_ratio
        )
        tile_latent_min_width = (
            self.tile_sample_min_width // self.spatial_compression_ratio
        )

        if self.use_tiling and (
            width > tile_latent_min_width or height > tile_latent_min_height
        ):
            return self.tiled_decode(z, return_dict=return_dict)

        self.clear_cache()
        x = self.post_quant_conv(z)
        for i in range(num_frame):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
            else:
                out_ = self.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
                out = torch.cat([out, out_], 2)

        out = torch.clamp(out, min=-1.0, max=1.0)
        self.clear_cache()
        if not return_dict:
            return (out,)

        return {"sample": out}

    # @apply_forward_hook
    def decode(
        self, z: torch.Tensor, return_dict: bool = True
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        r"""Decode a batch of latents into images (dict with key "sample" if `return_dict`)."""
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice)["sample"] for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z)["sample"]

        if not return_dict:
            return (decoded,)
        return {"sample": decoded}

    def decode_to_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        is_4d = latents.dim() == 4
        if is_4d:
            latents = latents.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]

        latents = latents.to(self.dtype)
        latents_mean = (
            torch.tensor(self.latents_mean)
            .view(1, self.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.latents_std).view(
            1, self.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean

        image = self.decode(latents, return_dict=False)[0]  # -1 to 1
        if is_4d:
            image = image.squeeze(2)  # [B, C, 1, H, W] -> [B, C, H, W]

        return image.clamp(-1.0, 1.0)

    def encode_pixels_to_latents(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode pixels (4D or 5D, [0,1] range) to mean/std-normalized latents; output ndim matches input."""
        is_4d = pixels.dim() == 4
        if is_4d:
            pixels = pixels.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]

        pixels = pixels.to(self.dtype)

        posterior = self.encode(pixels, return_dict=False)[0]
        latents = posterior.mode()  # mode, not sample, for deterministic results

        latents_mean = (
            torch.tensor(self.latents_mean)
            .view(1, self.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.latents_std).view(
            1, self.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * latents_std

        if is_4d:
            latents = latents.squeeze(2)  # [B, C, 1, H, W] -> [B, C, H, W]

        return latents

    def blend_v(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (
                1 - y / blend_extent
            ) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    def blend_h(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (
                1 - x / blend_extent
            ) + b[:, :, :, :, x] * (x / blend_extent)
        return b

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        r"""Encode a batch of images using a tiled encoder."""
        _, _, num_frames, height, width = x.shape
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio

        tile_latent_min_height = (
            self.tile_sample_min_height // self.spatial_compression_ratio
        )
        tile_latent_min_width = (
            self.tile_sample_min_width // self.spatial_compression_ratio
        )
        tile_latent_stride_height = (
            self.tile_sample_stride_height // self.spatial_compression_ratio
        )
        tile_latent_stride_width = (
            self.tile_sample_stride_width // self.spatial_compression_ratio
        )

        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        rows = []
        for i in range(0, height, self.tile_sample_stride_height):
            row = []
            for j in range(0, width, self.tile_sample_stride_width):
                self.clear_cache()
                time = []
                frame_range = 1 + (num_frames - 1) // 4
                for k in range(frame_range):
                    self._enc_conv_idx = [0]
                    if k == 0:
                        tile = x[
                            :,
                            :,
                            :1,
                            i : i + self.tile_sample_min_height,
                            j : j + self.tile_sample_min_width,
                        ]
                    else:
                        tile = x[
                            :,
                            :,
                            1 + 4 * (k - 1) : 1 + 4 * k,
                            i : i + self.tile_sample_min_height,
                            j : j + self.tile_sample_min_width,
                        ]
                    tile = self.encoder(
                        tile, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx
                    )
                    tile = self.quant_conv(tile)
                    time.append(tile)
                row.append(torch.cat(time, dim=2))
            rows.append(row)
        self.clear_cache()

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(
                    tile[:, :, :, :tile_latent_stride_height, :tile_latent_stride_width]
                )
            result_rows.append(torch.cat(result_row, dim=-1))

        enc = torch.cat(result_rows, dim=3)[:, :, :, :latent_height, :latent_width]
        return enc

    def tiled_decode(
        self, z: torch.Tensor, return_dict: bool = True
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        r"""Decode a batch of latents using a tiled decoder (dict with key "sample" if `return_dict`)."""
        _, _, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = (
            self.tile_sample_min_height // self.spatial_compression_ratio
        )
        tile_latent_min_width = (
            self.tile_sample_min_width // self.spatial_compression_ratio
        )
        tile_latent_stride_height = (
            self.tile_sample_stride_height // self.spatial_compression_ratio
        )
        tile_latent_stride_width = (
            self.tile_sample_stride_width // self.spatial_compression_ratio
        )

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

        rows = []
        for i in range(0, height, tile_latent_stride_height):
            row = []
            for j in range(0, width, tile_latent_stride_width):
                self.clear_cache()
                time = []
                for k in range(num_frames):
                    self._conv_idx = [0]
                    tile = z[
                        :,
                        :,
                        k : k + 1,
                        i : i + tile_latent_min_height,
                        j : j + tile_latent_min_width,
                    ]
                    tile = self.post_quant_conv(tile)
                    decoded = self.decoder(
                        tile, feat_cache=self._feat_map, feat_idx=self._conv_idx
                    )
                    time.append(decoded)
                row.append(torch.cat(time, dim=2))
            rows.append(row)
        self.clear_cache()

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(
                    tile[
                        :,
                        :,
                        :,
                        : self.tile_sample_stride_height,
                        : self.tile_sample_stride_width,
                    ]
                )
            result_rows.append(torch.cat(result_row, dim=-1))

        dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]

        if not return_dict:
            return (dec,)
        return {"sample": dec}

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Full VAE roundtrip: encode -> sample/mode -> decode."""
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z, return_dict=return_dict)
        return dec


# This section is not included in the original implementation. Added for musubi-tuner/sd-scripts.


# Convert ComfyUI keys to standard keys if necessary
def convert_comfyui_state_dict(sd):
    if "conv1.bias" not in sd:
        return sd

    # Key mapping from ComfyUI VAE to official VAE, auto-generated by a script
    key_map = {
        "conv1": "quant_conv",
        "conv2": "post_quant_conv",
        "decoder.conv1": "decoder.conv_in",
        "decoder.head.0": "decoder.norm_out",
        "decoder.head.2": "decoder.conv_out",
        "decoder.middle.0.residual.0": "decoder.mid_block.resnets.0.norm1",
        "decoder.middle.0.residual.2": "decoder.mid_block.resnets.0.conv1",
        "decoder.middle.0.residual.3": "decoder.mid_block.resnets.0.norm2",
        "decoder.middle.0.residual.6": "decoder.mid_block.resnets.0.conv2",
        "decoder.middle.1.norm": "decoder.mid_block.attentions.0.norm",
        "decoder.middle.1.proj": "decoder.mid_block.attentions.0.proj",
        "decoder.middle.1.to_qkv": "decoder.mid_block.attentions.0.to_qkv",
        "decoder.middle.2.residual.0": "decoder.mid_block.resnets.1.norm1",
        "decoder.middle.2.residual.2": "decoder.mid_block.resnets.1.conv1",
        "decoder.middle.2.residual.3": "decoder.mid_block.resnets.1.norm2",
        "decoder.middle.2.residual.6": "decoder.mid_block.resnets.1.conv2",
        "decoder.upsamples.0.residual.0": "decoder.up_blocks.0.resnets.0.norm1",
        "decoder.upsamples.0.residual.2": "decoder.up_blocks.0.resnets.0.conv1",
        "decoder.upsamples.0.residual.3": "decoder.up_blocks.0.resnets.0.norm2",
        "decoder.upsamples.0.residual.6": "decoder.up_blocks.0.resnets.0.conv2",
        "decoder.upsamples.1.residual.0": "decoder.up_blocks.0.resnets.1.norm1",
        "decoder.upsamples.1.residual.2": "decoder.up_blocks.0.resnets.1.conv1",
        "decoder.upsamples.1.residual.3": "decoder.up_blocks.0.resnets.1.norm2",
        "decoder.upsamples.1.residual.6": "decoder.up_blocks.0.resnets.1.conv2",
        "decoder.upsamples.10.residual.0": "decoder.up_blocks.2.resnets.2.norm1",
        "decoder.upsamples.10.residual.2": "decoder.up_blocks.2.resnets.2.conv1",
        "decoder.upsamples.10.residual.3": "decoder.up_blocks.2.resnets.2.norm2",
        "decoder.upsamples.10.residual.6": "decoder.up_blocks.2.resnets.2.conv2",
        "decoder.upsamples.11.resample.1": "decoder.up_blocks.2.upsamplers.0.resample.1",
        "decoder.upsamples.12.residual.0": "decoder.up_blocks.3.resnets.0.norm1",
        "decoder.upsamples.12.residual.2": "decoder.up_blocks.3.resnets.0.conv1",
        "decoder.upsamples.12.residual.3": "decoder.up_blocks.3.resnets.0.norm2",
        "decoder.upsamples.12.residual.6": "decoder.up_blocks.3.resnets.0.conv2",
        "decoder.upsamples.13.residual.0": "decoder.up_blocks.3.resnets.1.norm1",
        "decoder.upsamples.13.residual.2": "decoder.up_blocks.3.resnets.1.conv1",
        "decoder.upsamples.13.residual.3": "decoder.up_blocks.3.resnets.1.norm2",
        "decoder.upsamples.13.residual.6": "decoder.up_blocks.3.resnets.1.conv2",
        "decoder.upsamples.14.residual.0": "decoder.up_blocks.3.resnets.2.norm1",
        "decoder.upsamples.14.residual.2": "decoder.up_blocks.3.resnets.2.conv1",
        "decoder.upsamples.14.residual.3": "decoder.up_blocks.3.resnets.2.norm2",
        "decoder.upsamples.14.residual.6": "decoder.up_blocks.3.resnets.2.conv2",
        "decoder.upsamples.2.residual.0": "decoder.up_blocks.0.resnets.2.norm1",
        "decoder.upsamples.2.residual.2": "decoder.up_blocks.0.resnets.2.conv1",
        "decoder.upsamples.2.residual.3": "decoder.up_blocks.0.resnets.2.norm2",
        "decoder.upsamples.2.residual.6": "decoder.up_blocks.0.resnets.2.conv2",
        "decoder.upsamples.3.resample.1": "decoder.up_blocks.0.upsamplers.0.resample.1",
        "decoder.upsamples.3.time_conv": "decoder.up_blocks.0.upsamplers.0.time_conv",
        "decoder.upsamples.4.residual.0": "decoder.up_blocks.1.resnets.0.norm1",
        "decoder.upsamples.4.residual.2": "decoder.up_blocks.1.resnets.0.conv1",
        "decoder.upsamples.4.residual.3": "decoder.up_blocks.1.resnets.0.norm2",
        "decoder.upsamples.4.residual.6": "decoder.up_blocks.1.resnets.0.conv2",
        "decoder.upsamples.4.shortcut": "decoder.up_blocks.1.resnets.0.conv_shortcut",
        "decoder.upsamples.5.residual.0": "decoder.up_blocks.1.resnets.1.norm1",
        "decoder.upsamples.5.residual.2": "decoder.up_blocks.1.resnets.1.conv1",
        "decoder.upsamples.5.residual.3": "decoder.up_blocks.1.resnets.1.norm2",
        "decoder.upsamples.5.residual.6": "decoder.up_blocks.1.resnets.1.conv2",
        "decoder.upsamples.6.residual.0": "decoder.up_blocks.1.resnets.2.norm1",
        "decoder.upsamples.6.residual.2": "decoder.up_blocks.1.resnets.2.conv1",
        "decoder.upsamples.6.residual.3": "decoder.up_blocks.1.resnets.2.norm2",
        "decoder.upsamples.6.residual.6": "decoder.up_blocks.1.resnets.2.conv2",
        "decoder.upsamples.7.resample.1": "decoder.up_blocks.1.upsamplers.0.resample.1",
        "decoder.upsamples.7.time_conv": "decoder.up_blocks.1.upsamplers.0.time_conv",
        "decoder.upsamples.8.residual.0": "decoder.up_blocks.2.resnets.0.norm1",
        "decoder.upsamples.8.residual.2": "decoder.up_blocks.2.resnets.0.conv1",
        "decoder.upsamples.8.residual.3": "decoder.up_blocks.2.resnets.0.norm2",
        "decoder.upsamples.8.residual.6": "decoder.up_blocks.2.resnets.0.conv2",
        "decoder.upsamples.9.residual.0": "decoder.up_blocks.2.resnets.1.norm1",
        "decoder.upsamples.9.residual.2": "decoder.up_blocks.2.resnets.1.conv1",
        "decoder.upsamples.9.residual.3": "decoder.up_blocks.2.resnets.1.norm2",
        "decoder.upsamples.9.residual.6": "decoder.up_blocks.2.resnets.1.conv2",
        "encoder.conv1": "encoder.conv_in",
        "encoder.downsamples.0.residual.0": "encoder.down_blocks.0.norm1",
        "encoder.downsamples.0.residual.2": "encoder.down_blocks.0.conv1",
        "encoder.downsamples.0.residual.3": "encoder.down_blocks.0.norm2",
        "encoder.downsamples.0.residual.6": "encoder.down_blocks.0.conv2",
        "encoder.downsamples.1.residual.0": "encoder.down_blocks.1.norm1",
        "encoder.downsamples.1.residual.2": "encoder.down_blocks.1.conv1",
        "encoder.downsamples.1.residual.3": "encoder.down_blocks.1.norm2",
        "encoder.downsamples.1.residual.6": "encoder.down_blocks.1.conv2",
        "encoder.downsamples.10.residual.0": "encoder.down_blocks.10.norm1",
        "encoder.downsamples.10.residual.2": "encoder.down_blocks.10.conv1",
        "encoder.downsamples.10.residual.3": "encoder.down_blocks.10.norm2",
        "encoder.downsamples.10.residual.6": "encoder.down_blocks.10.conv2",
        "encoder.downsamples.2.resample.1": "encoder.down_blocks.2.resample.1",
        "encoder.downsamples.3.residual.0": "encoder.down_blocks.3.norm1",
        "encoder.downsamples.3.residual.2": "encoder.down_blocks.3.conv1",
        "encoder.downsamples.3.residual.3": "encoder.down_blocks.3.norm2",
        "encoder.downsamples.3.residual.6": "encoder.down_blocks.3.conv2",
        "encoder.downsamples.3.shortcut": "encoder.down_blocks.3.conv_shortcut",
        "encoder.downsamples.4.residual.0": "encoder.down_blocks.4.norm1",
        "encoder.downsamples.4.residual.2": "encoder.down_blocks.4.conv1",
        "encoder.downsamples.4.residual.3": "encoder.down_blocks.4.norm2",
        "encoder.downsamples.4.residual.6": "encoder.down_blocks.4.conv2",
        "encoder.downsamples.5.resample.1": "encoder.down_blocks.5.resample.1",
        "encoder.downsamples.5.time_conv": "encoder.down_blocks.5.time_conv",
        "encoder.downsamples.6.residual.0": "encoder.down_blocks.6.norm1",
        "encoder.downsamples.6.residual.2": "encoder.down_blocks.6.conv1",
        "encoder.downsamples.6.residual.3": "encoder.down_blocks.6.norm2",
        "encoder.downsamples.6.residual.6": "encoder.down_blocks.6.conv2",
        "encoder.downsamples.6.shortcut": "encoder.down_blocks.6.conv_shortcut",
        "encoder.downsamples.7.residual.0": "encoder.down_blocks.7.norm1",
        "encoder.downsamples.7.residual.2": "encoder.down_blocks.7.conv1",
        "encoder.downsamples.7.residual.3": "encoder.down_blocks.7.norm2",
        "encoder.downsamples.7.residual.6": "encoder.down_blocks.7.conv2",
        "encoder.downsamples.8.resample.1": "encoder.down_blocks.8.resample.1",
        "encoder.downsamples.8.time_conv": "encoder.down_blocks.8.time_conv",
        "encoder.downsamples.9.residual.0": "encoder.down_blocks.9.norm1",
        "encoder.downsamples.9.residual.2": "encoder.down_blocks.9.conv1",
        "encoder.downsamples.9.residual.3": "encoder.down_blocks.9.norm2",
        "encoder.downsamples.9.residual.6": "encoder.down_blocks.9.conv2",
        "encoder.head.0": "encoder.norm_out",
        "encoder.head.2": "encoder.conv_out",
        "encoder.middle.0.residual.0": "encoder.mid_block.resnets.0.norm1",
        "encoder.middle.0.residual.2": "encoder.mid_block.resnets.0.conv1",
        "encoder.middle.0.residual.3": "encoder.mid_block.resnets.0.norm2",
        "encoder.middle.0.residual.6": "encoder.mid_block.resnets.0.conv2",
        "encoder.middle.1.norm": "encoder.mid_block.attentions.0.norm",
        "encoder.middle.1.proj": "encoder.mid_block.attentions.0.proj",
        "encoder.middle.1.to_qkv": "encoder.mid_block.attentions.0.to_qkv",
        "encoder.middle.2.residual.0": "encoder.mid_block.resnets.1.norm1",
        "encoder.middle.2.residual.2": "encoder.mid_block.resnets.1.conv1",
        "encoder.middle.2.residual.3": "encoder.mid_block.resnets.1.norm2",
        "encoder.middle.2.residual.6": "encoder.mid_block.resnets.1.conv2",
    }

    new_state_dict = {}
    for key in sd.keys():
        new_key = key
        key_without_suffix = key.rsplit(".", 1)[0]
        if key_without_suffix in key_map:
            new_key = key.replace(key_without_suffix, key_map[key_without_suffix])
        new_state_dict[new_key] = sd[key]

    logger.info("Converted ComfyUI AutoencoderKL state dict keys to official format")
    return new_state_dict


def load_vae(
    vae_path: str,
    input_channels: int = 3,
    device: Union[str, torch.device] = "cpu",
    disable_mmap: bool = False,
    spatial_chunk_size: Optional[int] = None,
    disable_cache: bool = False,
    dtype: Optional[torch.dtype] = None,
    eval: bool = False,
    vae_2d: bool = False,
) -> AutoencoderKLQwenImage:
    """Load VAE from a given path.

    ``dtype``/``eval`` are shortcuts for the common ``vae.to(dtype); vae.eval()``
    (both default off). ``vae_2d`` folds the
    causal Conv3d stack into 2D convs (see :meth:`AutoencoderKLQwenImage.convert_to_2d`)
    — faster/leaner but **image-only** (single-frame); good for latent caching/decode.
    """
    vae_path = str(resolve_under_home(vae_path))
    VAE_CONFIG_JSON = """
{
  "_class_name": "AutoencoderKLQwenImage",
  "_diffusers_version": "0.34.0.dev0",
  "attn_scales": [],
  "base_dim": 96,
  "dim_mult": [
    1,
    2,
    4,
    4
  ],
  "dropout": 0.0,
  "latents_mean": [
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921
  ],
  "latents_std": [
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.916
  ],
  "num_res_blocks": 2,
  "temperal_downsample": [
    false,
    true,
    true
  ],
  "z_dim": 16
}
"""
    logger.info("Initializing VAE")

    if spatial_chunk_size is not None and spatial_chunk_size % 2 != 0:
        spatial_chunk_size += 1
        logger.warning(
            f"Adjusted spatial_chunk_size to the next even number: {spatial_chunk_size}"
        )

    config = json.loads(VAE_CONFIG_JSON)
    vae = AutoencoderKLQwenImage(
        base_dim=config["base_dim"],
        z_dim=config["z_dim"],
        dim_mult=config["dim_mult"],
        num_res_blocks=config["num_res_blocks"],
        attn_scales=config["attn_scales"],
        temperal_downsample=config["temperal_downsample"],
        dropout=config["dropout"],
        latents_mean=config["latents_mean"],
        latents_std=config["latents_std"],
        input_channels=input_channels,
        spatial_chunk_size=spatial_chunk_size,
        disable_cache=disable_cache,
    )

    logger.info(f"Loading VAE from {vae_path}")
    state_dict = load_safetensors(vae_path, device=device, disable_mmap=disable_mmap)

    state_dict = convert_comfyui_state_dict(state_dict)

    info = vae.load_state_dict(state_dict, strict=True, assign=True)
    logger.info(f"Loaded VAE: {info}")

    vae.to(device)
    if dtype is not None:
        vae.to(dtype)
    if vae_2d:
        n = vae.convert_to_2d()
        logger.info(f"Folded VAE to 2D (image-only): {n} Conv3d -> Conv2d")
    if eval:
        vae.eval()
    return vae


if __name__ == "__main__":
    import argparse
    import glob
    import os
    import time

    from PIL import Image

    from library.runtime.device import get_preferred_device, synchronize_device

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vae", type=str, required=True, help="Path to the VAE model file."
    )
    parser.add_argument(
        "--input_image_dir",
        type=str,
        required=True,
        help="Path to the input image directory.",
    )
    parser.add_argument(
        "--output_image_dir",
        type=str,
        required=True,
        help="Path to the output image directory.",
    )
    args = parser.parse_args()

    vae = load_vae(args.vae, device=get_preferred_device())

    def encode_decode_image(image_path, output_path):
        image = Image.open(image_path).convert("RGB")

        # Crop to multiple of 8 (VAE downsampling factor)
        width, height = image.size
        new_width = (width // 8) * 8
        new_height = (height // 8) * 8
        if new_width != width or new_height != height:
            image = image.crop((0, 0, new_width, new_height))

        image_tensor = (
            torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float()
            / 255.0
            * 2
            - 1
        )
        image_tensor = image_tensor.to(vae.dtype).to(vae.device)

        with torch.no_grad():
            latents = vae.encode_pixels_to_latents(image_tensor)
            reconstructed = vae.decode_to_pixels(latents)

        diff = (image_tensor - reconstructed).abs().mean().item()
        print(
            f"Processed {image_path} (size: {image.size}), reconstruction diff: {diff}"
        )

        reconstructed_image = (
            (reconstructed.squeeze(0).permute(1, 2, 0).float().cpu().numpy() + 1)
            / 2
            * 255
        ).astype(np.uint8)
        Image.fromarray(reconstructed_image).save(output_path)

    def process_directory(input_dir, output_dir):
        if get_preferred_device().type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()

        synchronize_device(get_preferred_device())
        start_time = time.perf_counter()

        os.makedirs(output_dir, exist_ok=True)
        image_paths = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(
            os.path.join(input_dir, "*.png")
        )
        for image_path in image_paths:
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, filename)
            encode_decode_image(image_path, output_path)

        if get_preferred_device().type == "cuda":
            max_mem = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"Max GPU memory allocated: {max_mem:.2f} GB")

        synchronize_device(get_preferred_device())
        end_time = time.perf_counter()
        print(f"Processing time: {end_time - start_time:.2f} seconds")

    print("Starting image processing with default settings...")
    process_directory(args.input_image_dir, args.output_image_dir)

    print(
        "Starting image processing with spatial chunking enabled with chunk size 64..."
    )
    vae.enable_spatial_chunking(64)
    process_directory(args.input_image_dir, args.output_image_dir + "_chunked_64")

    print(
        "Starting image processing with spatial chunking enabled with chunk size 16..."
    )
    vae.enable_spatial_chunking(16)
    process_directory(args.input_image_dir, args.output_image_dir + "_chunked_16")

    print(
        "Starting image processing without caching and chunking enabled with chunk size 64..."
    )
    vae.enable_spatial_chunking(64)
    vae.disable_cache()
    process_directory(
        args.input_image_dir, args.output_image_dir + "_no_cache_chunked_64"
    )

    print(
        "Starting image processing without caching and chunking enabled with chunk size 16..."
    )
    vae.disable_cache()
    process_directory(
        args.input_image_dir, args.output_image_dir + "_no_cache_chunked_16"
    )

    print("Starting image processing without caching and chunking disabled...")
    vae.disable_spatial_chunking()
    process_directory(args.input_image_dir, args.output_image_dir + "_no_cache")

    print("Processing completed.")
