"""Flow-matching sampling utilities for Anima inference."""

from typing import Optional, Tuple

import torch


def get_timesteps_sigmas(sampling_steps: int, shift: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate timesteps and sigmas for diffusion sampling.

    Args:
        sampling_steps: Number of sampling steps.
        shift: Sigma shift parameter for schedule modification.
        device: Target device for tensors.

    Returns:
        Tuple of (timesteps, sigmas) tensors.
    """
    sigmas = torch.linspace(1, 0, sampling_steps + 1)
    sigmas = (shift * sigmas) / (1 + (shift - 1) * sigmas)
    sigmas = sigmas.to(torch.float32)
    timesteps = (sigmas[:-1] * 1000).to(dtype=torch.float32, device=device)
    return timesteps, sigmas


def step(latents, noise_pred, sigmas, step_i):
    """Euler ODE step."""
    return latents.float() - (sigmas[step_i] - sigmas[step_i + 1]) * noise_pred.float()


class ERSDESampler:
    """Extended Reverse-Time SDE solver (ER-SDE-Solver-3).

    Paper: https://arxiv.org/abs/2309.06169

    For flow-matching (CONST schedule): sigma = t, alpha = 1-t,
    half_log_snr = log((1-sigma)/sigma) = -logit(sigma),
    er_lambda = sigma / (1 - sigma).
    """

    def __init__(
        self,
        sigmas: torch.Tensor,
        seed: Optional[int] = None,
        s_noise: float = 1.0,
        max_stage: int = 3,
        device: torch.device = torch.device("cpu"),
    ):
        self.s_noise = s_noise
        self.max_stage = max_stage
        self.num_integration_points = 200.0

        # Offset first sigma away from 1.0 to avoid logit(1)=inf
        sigmas = sigmas.clone().float()
        if sigmas[0] >= 1.0:
            sigmas[0] = 1.0 - 1e-4
        self.sigmas = sigmas

        # half_log_snr = log((1-σ)/σ) for flow-matching
        half_log_snrs = sigmas.logit().neg()
        self.er_lambdas = half_log_snrs.neg().exp()  # σ / (1 - σ)

        self.point_indices = torch.arange(0, self.num_integration_points, dtype=torch.float32, device=device)

        # Noise generator
        self._generator = None
        if seed is not None:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(seed)
        self._noise_device = device

        self.old_denoised = None
        self.old_denoised_d = None

    @staticmethod
    def _noise_scaler(x: torch.Tensor) -> torch.Tensor:
        return x * ((x ** 0.3).exp() + 10.0)

    def _sample_noise(self, shape, dtype):
        return torch.randn(shape, dtype=dtype, device=self._noise_device, generator=self._generator)

    def step(self, latents: torch.Tensor, denoised: torch.Tensor, step_i: int) -> torch.Tensor:
        """Perform one ER-SDE step.

        Args:
            latents: Current noisy latents (x_s).
            denoised: Model's denoised prediction (x_0 hat) for this step.
            step_i: Current step index.

        Returns:
            Updated latents (x_t).
        """
        x = latents.float()
        denoised = denoised.float()
        sigmas = self.sigmas
        er_lambdas = self.er_lambdas

        # Final step: just return denoised
        if sigmas[step_i + 1] == 0:
            self.old_denoised = denoised
            return denoised

        stage_used = min(self.max_stage, step_i + 1)

        er_lambda_s = er_lambdas[step_i]
        er_lambda_t = er_lambdas[step_i + 1]
        alpha_s = 1.0 - sigmas[step_i]   # alpha = 1 - sigma for flow-matching
        alpha_t = 1.0 - sigmas[step_i + 1]
        r_alpha = alpha_t / alpha_s
        r = self._noise_scaler(er_lambda_t) / self._noise_scaler(er_lambda_s)

        # Stage 1: Euler
        x = r_alpha * r * x + alpha_t * (1 - r) * denoised

        if stage_used >= 2:
            dt = er_lambda_t - er_lambda_s
            lambda_step_size = -dt / self.num_integration_points
            lambda_pos = er_lambda_t + self.point_indices * lambda_step_size
            scaled_pos = self._noise_scaler(lambda_pos)

            # Stage 2
            s = torch.sum(1 / scaled_pos) * lambda_step_size
            denoised_d = (denoised - self.old_denoised) / (er_lambda_s - er_lambdas[step_i - 1])
            x = x + alpha_t * (dt + s * self._noise_scaler(er_lambda_t)) * denoised_d

            if stage_used >= 3:
                # Stage 3
                s_u = torch.sum((lambda_pos - er_lambda_s) / scaled_pos) * lambda_step_size
                denoised_u = (denoised_d - self.old_denoised_d) / ((er_lambda_s - er_lambdas[step_i - 2]) / 2)
                x = x + alpha_t * ((dt ** 2) / 2 + s_u * self._noise_scaler(er_lambda_t)) * denoised_u
            self.old_denoised_d = denoised_d

        # Stochastic noise injection
        if self.s_noise > 0:
            noise_coeff = (er_lambda_t ** 2 - er_lambda_s ** 2 * r ** 2).sqrt().nan_to_num(nan=0.0)
            x = x + alpha_t * self._sample_noise(x.shape, x.dtype) * self.s_noise * noise_coeff

        self.old_denoised = denoised
        return x
