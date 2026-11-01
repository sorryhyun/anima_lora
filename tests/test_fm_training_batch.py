"""Bit-match invariant for the ``fm_training_batch`` bench helper (issues.md DX2).

``library.runtime.noise.fm_training_batch`` is the plain-kwargs entry point a
bench imports to get ``(noisy, timesteps, target)`` without reassembling an
``args`` Namespace + a scheduler and re-deriving the target by hand. Its whole
value is that it produces *exactly* what the trainer does — so this test pins it
against the real in-trainer path: the registered ``_default_sampler`` (which the
trainer calls via ``SAMPLER_REGISTRY``) plus ``target = noise - latents`` (the
literal ``train.py`` line). CPU-only, no model.

If someone changes the σ recipe on one side without the other, this trips.
"""

from __future__ import annotations

import argparse

import torch

from library.runtime.noise import FlowMatchEulerDiscreteScheduler, fm_training_batch
from library.training.samplers import SamplerContext, _default_sampler

# The trainer's argument defaults for the fields the sampler reads
# (configs/base.toml + argparse defaults in library/{anima/training,config/cli_args}.py).
_DEFAULTS = dict(
    timestep_sampling="sigmoid",
    discrete_flow_shift=1.0,
    sigmoid_scale=1.0,
    sigmoid_bias=0.0,
    weighting_scheme="uniform",
    logit_mean=0.0,
    logit_std=1.0,
    mode_scale=1.29,
    ip_noise_gamma=None,
    ip_noise_gamma_random_strength=False,
    t_min=None,
    t_max=None,
)


def _args(**overrides):
    return argparse.Namespace(**{**_DEFAULTS, **overrides})


def _reference(args, latents, noise, device, dtype, seed):
    """The real in-trainer path: registered sampler + rectified-flow target."""
    sched = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=args.discrete_flow_shift
    )
    torch.manual_seed(seed)
    out = _default_sampler(
        SamplerContext(
            args=args,
            noise_scheduler=sched,
            latents=latents,
            noise=noise,
            device=device,
            weight_dtype=dtype,
        )
    )
    return out.noisy_input, out.timesteps, noise - latents


def _candidate(overrides, latents, noise, device, dtype, seed):
    torch.manual_seed(seed)
    return fm_training_batch(
        latents, noise, device=device, dtype=dtype, **{**_DEFAULTS, **overrides}
    )


def _check(overrides, *, seed=1234):
    device = torch.device("cpu")
    dtype = torch.float32
    torch.manual_seed(0)
    latents = torch.randn(2, 16, 8, 8)
    noise = torch.randn(2, 16, 8, 8)

    ref = _reference(_args(**overrides), latents, noise, device, dtype, seed)
    cand = _candidate(overrides, latents, noise, device, dtype, seed)

    for name, a, b in zip(("noisy", "timesteps", "target"), ref, cand):
        assert torch.equal(a, b), f"{name} mismatch for overrides={overrides}"


def test_all_sampling_modes_bit_match():
    for mode in ("sigma", "uniform", "sigmoid", "shift", "flux_shift"):
        _check({"timestep_sampling": mode})
