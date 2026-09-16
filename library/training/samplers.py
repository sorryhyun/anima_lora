"""Noise / timestep samplers.

The noise-and-timestep draw for `get_noise_pred_and_target`; new schedules
register a `SamplerFn`. The default sampler delegates to
`noise_utils.get_noisy_model_input_and_timesteps`, which already returns the
DiT-scale time argument (σ∈[0,1]); no rescale is applied here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import torch

from library.runtime import noise as noise_utils


@dataclass
class SamplerContext:
    args: argparse.Namespace
    noise_scheduler: object
    latents: torch.Tensor
    noise: torch.Tensor
    device: torch.device
    weight_dtype: torch.dtype
    # Pre-drawn flat (bsz,) σ (already range-restricted), or None to draw
    # inside. sigma_lowres draws σ first (the latent grid depends on it) via
    # noise_utils.draw_flat_sigmas and passes it through here.
    sigmas: "torch.Tensor | None" = None


@dataclass
class SamplerOut:
    noisy_input: torch.Tensor
    timesteps: torch.Tensor
    sigmas: torch.Tensor


SamplerFn = Callable[[SamplerContext], SamplerOut]


def _default_sampler(ctx: SamplerContext) -> SamplerOut:
    noisy_input, timesteps, sigmas = noise_utils.get_noisy_model_input_and_timesteps(
        ctx.args,
        ctx.noise_scheduler,
        ctx.latents,
        ctx.noise,
        ctx.device,
        ctx.weight_dtype,
        sigmas=ctx.sigmas,
    )
    return SamplerOut(noisy_input=noisy_input, timesteps=timesteps, sigmas=sigmas)


SAMPLER_REGISTRY: dict[str, SamplerFn] = {"default": _default_sampler}
