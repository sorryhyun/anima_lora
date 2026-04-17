"""Noise / timestep samplers.

M1 extraction (plan.md): isolates the noise-and-timestep draw out of
`get_noise_pred_and_target` so new schedules can be added by registering a
`SamplerFn` instead of branching in-trainer. The default sampler preserves
the pre-refactor behavior byte-for-byte — it delegates to
`noise_utils.get_noisy_model_input_and_timesteps` and applies the /1000
rescale.
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
    )
    timesteps = timesteps / 1000.0
    return SamplerOut(noisy_input=noisy_input, timesteps=timesteps, sigmas=sigmas)


SAMPLER_REGISTRY: dict[str, SamplerFn] = {"default": _default_sampler}
