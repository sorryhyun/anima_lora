"""SNR-t bias measurement for validation logging.

Reusable distillation of ``archive/dcw/measure_bias.py`` — given a velocity
forward closure and cached x_0 latents, compute per-step:

    v_fwd_norm(i) = || v_θ((1−σ_i)·x_0 + σ_i·ε, t_i) ||      (forward arm)
    v_rev_norm(i) = || v_θ(x_hat_i, t_i) ||                   (reverse arm)
    gap(i)        = v_rev_norm(i) − v_fwd_norm(i)

The reverse trajectory is a plain Euler rollout from σ≈1 noise. The forward
arm reuses cached x_0 with fresh per-step noise, deterministic from the
caller's seed.

In addition to the global norms, the trajectory dict carries the LL Haar
subband norms (``v_fwd_LL``, ``v_rev_LL``, ``gap_LL``). LL is the only band
needed by the per-LoRA DCW calibration solver — see
``archive/dcw-learnable-calibrator/proposals/lora-dcw-proposal.md`` and the 2026-05-03 LL-only finding in
``docs/methods/dcw.md``.

The forward closure encapsulates per-method DiT call setup (text conds,
padding mask, crossattn_seqlens, hydra σ propagation) so this module stays
free of the trainer's call-site details. See ``docs/methods/dcw.md`` and
``archive/dcw/findings.md`` for the bias signal it tracks.
"""
from __future__ import annotations

from typing import Callable

import torch

from library.inference.sampling import get_timesteps_sigmas


def _haar_LL_norm(v: torch.Tensor) -> float:
    """L2 norm of the single-level 2D orthonormal Haar LL subband on the
    last two dims of ``v``. ~4× cheaper than running the full DWT since
    only the lowpass coefficient is materialized.
    """
    v = v.float()
    a = v[..., 0::2, 0::2]
    b = v[..., 0::2, 1::2]
    c = v[..., 1::2, 0::2]
    d = v[..., 1::2, 1::2]
    LL = (a + b + c + d) * 0.5
    return LL.flatten().norm().item()


@torch.no_grad()
def measure_bias_trajectory(
    forward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_0: torch.Tensor,
    *,
    num_steps: int,
    flow_shift: float,
    noise_seed: int,
    dcw_probe_lambda: float = 0.0,
    dcw_probe_schedule: str = "one_minus_sigma",
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
        dcw_probe_lambda: when nonzero, inject LL-band DCW correction
            with this λ on every reverse-rollout Euler step (skipped at
            the final step where σ_next == 0, matching inference). The
            forward arm is unaffected — this only perturbs the reverse
            trajectory, which is what the calibration solver needs.
        dcw_probe_schedule: σ-schedule for the probe correction. Should
            match the schedule the recipe will be applied with at
            inference (default ``one_minus_sigma``).

    Returns:
        dict with float64 CPU tensors of length ``num_steps``:
            ``v_fwd``, ``v_rev``, ``gap``, ``v_fwd_LL``, ``v_rev_LL``,
            ``gap_LL``, ``sigmas``.
    """
    device, dtype = x_0.device, x_0.dtype
    B = x_0.shape[0]
    _, sigmas = get_timesteps_sigmas(num_steps, flow_shift, device)
    sigmas_cpu = sigmas.cpu()

    v_fwd = torch.zeros(num_steps, dtype=torch.float64)
    v_rev = torch.zeros(num_steps, dtype=torch.float64)
    v_fwd_LL = torch.zeros(num_steps, dtype=torch.float64)
    v_rev_LL = torch.zeros(num_steps, dtype=torch.float64)

    g_fwd = torch.Generator(device="cpu").manual_seed(noise_seed + 10_000)
    for i in range(num_steps):
        sigma_i = float(sigmas_cpu[i])
        sigma_b = torch.full((B,), sigma_i, device=device, dtype=torch.float32)
        eps = torch.randn(x_0.shape, generator=g_fwd).to(device, dtype)
        x_t = (1.0 - sigma_i) * x_0 + sigma_i * eps
        v = forward_fn(x_t, sigma_b)
        v_fwd[i] = v.float().flatten().norm().item()
        v_fwd_LL[i] = _haar_LL_norm(v)

    g_rev = torch.Generator(device="cpu").manual_seed(noise_seed)
    eps_init = torch.randn(x_0.shape, generator=g_rev).to(device, dtype)
    x_hat = eps_init
    if dcw_probe_lambda != 0.0:
        from networks.dcw import apply_dcw
    for i in range(num_steps):
        sigma_i = float(sigmas_cpu[i])
        sigma_next = float(sigmas_cpu[i + 1])
        sigma_b = torch.full((B,), sigma_i, device=device, dtype=torch.float32)
        v = forward_fn(x_hat, sigma_b)
        v_rev[i] = v.float().flatten().norm().item()
        v_rev_LL[i] = _haar_LL_norm(v)
        x_hat_f = x_hat.float()
        v_f = v.float()
        x_next = x_hat_f + (sigma_next - sigma_i) * v_f
        if dcw_probe_lambda != 0.0 and sigma_next > 0.0:
            x0_pred = x_hat_f - sigma_i * v_f
            x_next = apply_dcw(
                x_next,
                x0_pred,
                sigma_i,
                lam=dcw_probe_lambda,
                schedule=dcw_probe_schedule,  # type: ignore[arg-type]
                bands=frozenset({"LL"}),
            )
        x_hat = x_next.to(dtype)

    return {
        "v_fwd": v_fwd,
        "v_rev": v_rev,
        "gap": v_rev - v_fwd,
        "v_fwd_LL": v_fwd_LL,
        "v_rev_LL": v_rev_LL,
        "gap_LL": v_rev_LL - v_fwd_LL,
        "sigmas": sigmas_cpu[:num_steps].to(torch.float64),
    }
