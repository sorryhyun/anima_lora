"""SNR-t bias measurement for validation logging.

Reusable distillation of ``archive/dcw/measure_bias.py`` — given a velocity
forward closure and cached x_0 latents, compute per-step:

    v_fwd_norm(i) = || v_θ((1−σ_i)·x_0 + σ_i·ε, t_i) ||      (forward arm)
    v_rev_norm(i) = || v_θ(x_hat_i, t_i) ||                   (reverse arm)
    gap(i)        = v_rev_norm(i) − v_fwd_norm(i)

The reverse trajectory is a plain Euler rollout from σ≈1 noise. The forward
arm reuses cached x_0 with fresh per-step noise, deterministic from the
caller's seed.

The forward closure encapsulates per-method DiT call setup (text conds,
padding mask, crossattn_seqlens, hydra σ propagation) so this module stays
free of the trainer's call-site details. See ``docs/methods/dcw.md`` and
``archive/dcw/findings.md`` for the bias signal it tracks.
"""
from __future__ import annotations

from typing import Callable

import torch

from library.inference.sampling import get_timesteps_sigmas


@torch.no_grad()
def measure_bias_trajectory(
    forward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_0: torch.Tensor,
    *,
    num_steps: int,
    flow_shift: float,
    noise_seed: int,
) -> dict:
    """Per-step paired forward/reverse velocity norms.

    Args:
        forward_fn: ``(x_t, sigma_b) -> v_θ`` — sigma_b is shape (B,) on
            x_0.device, dtype float32.
        x_0: cached latents, shape (B, C, 1, H, W) on the model device.
        num_steps: schedule length; same math as inference.
        flow_shift: σ-shift parameter (matches inference.py default 1.0).
        noise_seed: deterministic seed for both the forward arm's per-step
            noise and the reverse arm's σ≈1 init noise.

    Returns:
        dict with float64 CPU tensors of length ``num_steps``:
            ``v_fwd``, ``v_rev``, ``gap``, ``sigmas``.
    """
    device, dtype = x_0.device, x_0.dtype
    B = x_0.shape[0]
    _, sigmas = get_timesteps_sigmas(num_steps, flow_shift, device)
    sigmas_cpu = sigmas.cpu()

    v_fwd = torch.zeros(num_steps, dtype=torch.float64)
    v_rev = torch.zeros(num_steps, dtype=torch.float64)

    g_fwd = torch.Generator(device="cpu").manual_seed(noise_seed + 10_000)
    for i in range(num_steps):
        sigma_i = float(sigmas_cpu[i])
        sigma_b = torch.full((B,), sigma_i, device=device, dtype=torch.float32)
        eps = torch.randn(x_0.shape, generator=g_fwd).to(device, dtype)
        x_t = (1.0 - sigma_i) * x_0 + sigma_i * eps
        v = forward_fn(x_t, sigma_b)
        v_fwd[i] = v.float().flatten().norm().item()

    g_rev = torch.Generator(device="cpu").manual_seed(noise_seed)
    eps_init = torch.randn(x_0.shape, generator=g_rev).to(device, dtype)
    x_hat = eps_init
    for i in range(num_steps):
        sigma_i = float(sigmas_cpu[i])
        sigma_next = float(sigmas_cpu[i + 1])
        sigma_b = torch.full((B,), sigma_i, device=device, dtype=torch.float32)
        v = forward_fn(x_hat, sigma_b)
        v_rev[i] = v.float().flatten().norm().item()
        x_hat = (x_hat.float() + (sigma_next - sigma_i) * v.float()).to(dtype)

    return {
        "v_fwd": v_fwd,
        "v_rev": v_rev,
        "gap": v_rev - v_fwd,
        "sigmas": sigmas_cpu[:num_steps].to(torch.float64),
    }
