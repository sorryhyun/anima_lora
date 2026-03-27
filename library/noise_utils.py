# Noise utilities for flow-matching training
# Extracted from sd3_train_utils.py and flux_train_utils.py

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import BaseOutput


# region Timestep sampling utilities (from flux_train_utils.py)


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    timesteps = torch.linspace(1, 0, num_steps + 1)
    if shift:
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)
    return timesteps.tolist()


def get_sigmas(noise_scheduler, timesteps, device, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    return sigma


def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    """Compute the density for sampling the timesteps when doing flow-matching training.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u


def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    """Computes loss weighting scheme for flow-matching training."""
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting


def get_noisy_model_input_and_timesteps(
    args, noise_scheduler, latents: torch.Tensor, noise: torch.Tensor, device, dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, h, w = latents.shape[0], latents.shape[-2], latents.shape[-1]
    assert bsz > 0, "Batch size not large enough"
    num_timesteps = noise_scheduler.config.num_train_timesteps
    if args.timestep_sampling == "uniform" or args.timestep_sampling == "sigmoid":
        if args.timestep_sampling == "sigmoid":
            sigmas = torch.sigmoid(args.sigmoid_scale * torch.randn((bsz,), device=device))
        else:
            sigmas = torch.rand((bsz,), device=device)
        timesteps = sigmas * num_timesteps
    elif args.timestep_sampling == "shift":
        shift = args.discrete_flow_shift
        sigmas = torch.randn(bsz, device=device)
        sigmas = sigmas * args.sigmoid_scale
        sigmas = sigmas.sigmoid()
        sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_timesteps
    elif args.timestep_sampling == "flux_shift":
        sigmas = torch.randn(bsz, device=device)
        sigmas = sigmas * args.sigmoid_scale
        sigmas = sigmas.sigmoid()
        mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
        sigmas = time_shift(mu, 1.0, sigmas)
        timesteps = sigmas * num_timesteps
    else:
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=bsz,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
        )
        indices = (u * num_timesteps).long()
        timesteps = noise_scheduler.timesteps[indices].to(device=device)
        sigmas = get_sigmas(noise_scheduler, timesteps, device, n_dim=latents.ndim, dtype=dtype)

    # Restrict sigma range (P-GRAFT-inspired timestep restriction)
    t_min = getattr(args, "t_min", None)
    t_max = getattr(args, "t_max", None)
    if t_min is not None or t_max is not None:
        lo = t_min if t_min is not None else 0.0
        hi = t_max if t_max is not None else 1.0
        sigmas = lo + sigmas * (hi - lo)
        timesteps = sigmas * num_timesteps

    # Broadcast sigmas to latent shape
    sigmas = sigmas.view(-1, 1, 1, 1) if latents.ndim == 4 else sigmas.view(-1, 1, 1, 1, 1)

    # Add noise to the latents according to the noise magnitude at each timestep
    if args.ip_noise_gamma:
        xi = torch.randn_like(latents, device=latents.device, dtype=dtype)
        if args.ip_noise_gamma_random_strength:
            ip_noise_gamma = torch.rand(1, device=latents.device, dtype=dtype) * args.ip_noise_gamma
        else:
            ip_noise_gamma = args.ip_noise_gamma
        noisy_model_input = (1.0 - sigmas) * latents + sigmas * (noise + ip_noise_gamma * xi)
    else:
        noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise

    return noisy_model_input.to(dtype), timesteps.to(dtype), sigmas


def apply_model_prediction_type(args, model_pred, noisy_model_input, sigmas):
    weighting = None
    if args.model_prediction_type == "raw":
        pass
    elif args.model_prediction_type == "additive":
        model_pred = model_pred + noisy_model_input
    elif args.model_prediction_type == "sigma_scaled":
        model_pred = model_pred * (-sigmas) + noisy_model_input
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
    return model_pred, weighting


# endregion


# region FlowMatchEulerDiscreteScheduler (from sd3_train_utils.py)


@dataclass
class FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    """Output class for the scheduler's `step` function output."""

    prev_sample: torch.FloatTensor


class FlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler scheduler for flow-matching models.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
    ):
        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps

        self._step_index = None
        self._begin_index = None

        self.sigmas = sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]
        sample = sigma * noise + (1.0 - sigma) * sample

        return sample

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        self.num_inference_steps = num_inference_steps

        timesteps = np.linspace(self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps)

        sigmas = timesteps / self.config.num_train_timesteps
        sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)

        timesteps = sigmas * self.config.num_train_timesteps
        self.timesteps = timesteps.to(device=device)
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        if isinstance(timestep, int) or isinstance(timestep, torch.IntTensor) or isinstance(timestep, torch.LongTensor):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        sample = sample.to(torch.float32)

        sigma = self.sigmas[self.step_index]

        gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0

        noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator)

        eps = noise * s_noise
        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        denoised = sample - model_output * sigma
        derivative = (sample - denoised) / sigma_hat

        dt = self.sigmas[self.step_index + 1] - sigma_hat

        prev_sample = sample + derivative * dt
        prev_sample = prev_sample.to(model_output.dtype)

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps


# endregion
