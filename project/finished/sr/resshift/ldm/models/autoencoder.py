import torch
import torch.nn as nn

from functools import partial

import loralib as lora

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer


class VQModelTorch(nn.Module):
    def __init__(
        self,
        ddconfig,
        n_embed,
        embed_dim,
        remap=None,
        rank=8,  # rank for lora
        lora_alpha=1.0,
        lora_tune_decoder=False,
        sane_index_shape=False,  # tell vector quantizer to return indices as bhw
    ):
        super().__init__()
        if lora_tune_decoder:
            conv_layer = partial(lora.Conv2d, r=rank, lora_alpha=lora_alpha)
        else:
            conv_layer = nn.Conv2d

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(
            rank=rank, lora_alpha=lora_alpha, lora_tune=lora_tune_decoder, **ddconfig
        )
        self.quantize = VectorQuantizer(
            n_embed,
            embed_dim,
            beta=0.25,
            remap=remap,
            sane_index_shape=sane_index_shape,
        )
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = conv_layer(embed_dim, ddconfig["z_channels"], 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b, force_not_quantize=True)
        return dec

    def forward(self, input, force_not_quantize=False):
        h = self.encode(input)
        dec = self.decode(h, force_not_quantize)
        return dec


class AutoencoderKLTorch(torch.nn.Module):
    def __init__(
        self,
        ddconfig,
        embed_dim,
    ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

    def encode(self, x, sample_posterior=True, return_moments=False):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        if return_moments:
            return z, moments
        else:
            return z

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        z = self.encode(input, sample_posterior, return_moments=False)
        dec = self.decode(z)
        return dec


class EncoderKLTorch(torch.nn.Module):
    def __init__(
        self,
        ddconfig,
        embed_dim,
    ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.embed_dim = embed_dim

    def encode(self, x, sample_posterior=True, return_moments=False):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        if return_moments:
            return z, moments
        else:
            return z

    def forward(self, x, sample_posterior=True, return_moments=False):
        return self.encode(x, sample_posterior, return_moments)


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x
