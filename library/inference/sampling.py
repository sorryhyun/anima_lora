"""Flow-matching sampling utilities for Anima inference."""

from typing import Optional, Tuple, Union

import torch
from torchvision import transforms

# Single source of truth for the CFG++ reweight — shared with the FSG operator
# and (vendored) the ComfyUI Spectrum node. Re-exported here for callers that
# read it off the sampling module (networks/spectrum.py, generation.py, tests).
from library.inference.corrections.fsg_core import cfgpp_guidance_weight  # noqa: F401
from diffusers import EulerAncestralDiscreteScheduler
import diffusers.schedulers.scheduling_euler_ancestral_discrete
from diffusers.schedulers.scheduling_euler_ancestral_discrete import (
    EulerAncestralDiscreteSchedulerOutput,
)


def get_timesteps_sigmas(
    sampling_steps: int, shift: float, device: torch.device, tail_power: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate timesteps and sigmas for diffusion sampling.

    Args:
        sampling_steps: Number of sampling steps.
        shift: Sigma shift parameter for schedule modification.
        device: Target device for tensors.
        tail_power: σ-reshape exponent applied to the uniform grid *before* the
            flow-shift map. ``1.0`` (default) is the canonical schedule, bit-for-bit.
            ``>1`` warps the grid toward σ=0, packing more knots into the low-σ
            resolve tail (σ<0.45) where local discretization error concentrates
            (``_archive/bench/dynamic_spectrum`` Q1: ~60% of fattening-error mass, only 19%
            of steps); ``<1`` does the opposite. The endpoints σ∈{0,1} are fixed,
            so this redistributes steps without changing the integration range.

    Returns:
        Tuple of (timesteps, sigmas) tensors. ``timesteps`` is the DiT's time
        argument on the σ∈[0,1] scale (== ``sigmas[:-1]``); the model rescales
        nothing further, so callers feed it directly (no /1000).
    """
    sigmas = torch.linspace(1, 0, sampling_steps + 1)
    if tail_power != 1.0:
        # u∈[0,1] uniform; u**p with p>1 pulls interior knots toward 0 (the σ tail).
        # Endpoints (1→1, 0→0) are preserved, so only the spacing is reshaped.
        sigmas = sigmas.clamp_min(0.0) ** tail_power
    sigmas = (shift * sigmas) / (1 + (shift - 1) * sigmas)
    sigmas = sigmas.to(torch.float32)
    timesteps = sigmas[:-1].to(dtype=torch.float32, device=device)
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
        cns=None,
    ):
        self.s_noise = s_noise
        self.max_stage = max_stage
        self.num_integration_points = 200.0
        # Optional CNS recolorer (library.inference.corrections.cns.CNSRecolorer).
        # When set, the per-step injected noise is frequency-recolored by
        # sqrt(1-γ) so the fixed stochastic budget lands in unresolved bands.
        self.cns = cns

        # Offset first sigma away from 1.0 to avoid logit(1)=inf
        sigmas = sigmas.clone().float()
        if sigmas[0] >= 1.0:
            sigmas[0] = 1.0 - 1e-4
        self.sigmas = sigmas

        # half_log_snr = log((1-σ)/σ) for flow-matching
        half_log_snrs = sigmas.logit().neg()
        self.er_lambdas = half_log_snrs.neg().exp()  # σ / (1 - σ)

        self.point_indices = torch.arange(
            0, self.num_integration_points, dtype=torch.float32, device=device
        )

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
        return x * ((x**0.3).exp() + 10.0)

    def _sample_noise(self, shape, dtype, sigma_s=None):
        white = torch.randn(
            shape, dtype=dtype, device=self._noise_device, generator=self._generator
        )
        if self.cns is not None and sigma_s is not None:
            return self.cns.recolor(white, sigma_s)
        return white

    def step(
        self, latents: torch.Tensor, denoised: torch.Tensor, step_i: int
    ) -> torch.Tensor:
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
        alpha_s = 1.0 - sigmas[step_i]  # alpha = 1 - sigma for flow-matching
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
            denoised_d = (denoised - self.old_denoised) / (
                er_lambda_s - er_lambdas[step_i - 1]
            )
            x = x + alpha_t * (dt + s * self._noise_scaler(er_lambda_t)) * denoised_d

            if stage_used >= 3:
                # Stage 3
                s_u = (
                    torch.sum((lambda_pos - er_lambda_s) / scaled_pos)
                    * lambda_step_size
                )
                denoised_u = (denoised_d - self.old_denoised_d) / (
                    (er_lambda_s - er_lambdas[step_i - 2]) / 2
                )
                x = (
                    x
                    + alpha_t
                    * ((dt**2) / 2 + s_u * self._noise_scaler(er_lambda_t))
                    * denoised_u
                )
            self.old_denoised_d = denoised_d

        # Stochastic noise injection
        if self.s_noise > 0:
            noise_coeff = (
                (er_lambda_t**2 - er_lambda_s**2 * r**2).sqrt().nan_to_num(nan=0.0)
            )
            x = (
                x
                + alpha_t
                * self._sample_noise(x.shape, x.dtype, sigma_s=float(sigmas[step_i]))
                * self.s_noise
                * noise_coeff
            )

        self.old_denoised = denoised
        return x


class LCMSampler:
    """LCM-style stochastic sampler for distilled few-step flow-matching models.

    On each step, treats the model's x0 estimate as a refinement target and
    re-noises to the next timestep via the flow-matching CONST schedule:

        x_next = (1 - σ_next) * x0_hat + σ_next * ε

    Final step (σ_next == 0) returns x0_hat directly. The caller computes
    `denoised = latents - σ * v_pred` (x0 estimate) before invoking .step,
    matching the ERSDESampler interface.

    Targeted at under-converged few-step distillations where deterministic
    Euler can look dead — the per-step noise re-injection gives the next
    forward something to refine. Trades seed determinism for texture /
    detail recovery.

    Reference: Luo et al. 2023 (Latent Consistency Models, arXiv:2310.04378);
    ComfyUI's `comfy/k_diffusion/sampling.py::sample_lcm`.
    """

    def __init__(
        self,
        sigmas: torch.Tensor,
        seed: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.sigmas = sigmas.clone().float()
        self._generator = None
        if seed is not None:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(seed)
        self._noise_device = device

    def step(
        self, latents: torch.Tensor, denoised: torch.Tensor, step_i: int
    ) -> torch.Tensor:
        denoised = denoised.float()
        sigma_next = float(self.sigmas[step_i + 1])

        if sigma_next == 0.0:
            return denoised

        noise = torch.randn(
            denoised.shape,
            dtype=denoised.dtype,
            device=self._noise_device,
            generator=self._generator,
        )
        return (1.0 - sigma_next) * denoised + sigma_next * noise


class GradualLatent:
    def __init__(
        self,
        ratio,
        start_timesteps,
        every_n_steps,
        ratio_step,
        s_noise=1.0,
        gaussian_blur_ksize=None,
        gaussian_blur_sigma=0.5,
        gaussian_blur_strength=0.5,
        unsharp_target_x=True,
    ):
        self.ratio = ratio
        self.start_timesteps = start_timesteps
        self.every_n_steps = every_n_steps
        self.ratio_step = ratio_step
        self.s_noise = s_noise
        self.gaussian_blur_ksize = gaussian_blur_ksize
        self.gaussian_blur_sigma = gaussian_blur_sigma
        self.gaussian_blur_strength = gaussian_blur_strength
        self.unsharp_target_x = unsharp_target_x

    def __str__(self) -> str:
        return (
            f"GradualLatent(ratio={self.ratio}, start_timesteps={self.start_timesteps}, "
            + f"every_n_steps={self.every_n_steps}, ratio_step={self.ratio_step}, s_noise={self.s_noise}, "
            + f"gaussian_blur_ksize={self.gaussian_blur_ksize}, gaussian_blur_sigma={self.gaussian_blur_sigma}, gaussian_blur_strength={self.gaussian_blur_strength}, "
            + f"unsharp_target_x={self.unsharp_target_x})"
        )

    def apply_unshark_mask(self, x: torch.Tensor):
        if self.gaussian_blur_ksize is None:
            return x
        blurred = transforms.functional.gaussian_blur(
            x, self.gaussian_blur_ksize, self.gaussian_blur_sigma
        )
        mask = (x - blurred) * self.gaussian_blur_strength
        sharpened = x + mask
        return sharpened

    def interpolate(self, x: torch.Tensor, resized_size, unsharp=True):
        org_dtype = x.dtype
        if org_dtype == torch.bfloat16:
            x = x.float()

        x = torch.nn.functional.interpolate(
            x, size=resized_size, mode="bicubic", align_corners=False
        ).to(dtype=org_dtype)

        if unsharp and self.gaussian_blur_ksize:
            x = self.apply_unshark_mask(x)

        return x


class EulerAncestralDiscreteSchedulerGL(EulerAncestralDiscreteScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resized_size = None
        self.gradual_latent = None

    def set_gradual_latent_params(self, size, gradual_latent: GradualLatent):
        self.resized_size = size
        self.gradual_latent = gradual_latent

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[EulerAncestralDiscreteSchedulerOutput, Tuple]:
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                " one of the `scheduler.timesteps` as a timestep.",
            )

        if not self.is_scale_input_called:
            print(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        if self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma * model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (
                sample / (sigma**2 + 1)
            )
        elif self.config.prediction_type == "sample":
            raise NotImplementedError("prediction_type not implemented yet: sample")
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        sigma_from = self.sigmas[self.step_index]
        sigma_to = self.sigmas[self.step_index + 1]
        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
        sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma

        dt = sigma_down - sigma

        device = model_output.device
        if self.resized_size is None:
            prev_sample = sample + derivative * dt

            noise = (
                diffusers.schedulers.scheduling_euler_ancestral_discrete.randn_tensor(
                    model_output.shape,
                    dtype=model_output.dtype,
                    device=device,
                    generator=generator,
                )
            )
            s_noise = 1.0
        else:
            print(
                "resized_size",
                self.resized_size,
                "model_output.shape",
                model_output.shape,
                "sample.shape",
                sample.shape,
            )
            s_noise = self.gradual_latent.s_noise

            if self.gradual_latent.unsharp_target_x:
                prev_sample = sample + derivative * dt
                prev_sample = self.gradual_latent.interpolate(
                    prev_sample, self.resized_size
                )
            else:
                sample = self.gradual_latent.interpolate(sample, self.resized_size)
                derivative = self.gradual_latent.interpolate(
                    derivative, self.resized_size, unsharp=False
                )
                prev_sample = sample + derivative * dt

            noise = (
                diffusers.schedulers.scheduling_euler_ancestral_discrete.randn_tensor(
                    (
                        model_output.shape[0],
                        model_output.shape[1],
                        self.resized_size[0],
                        self.resized_size[1],
                    ),
                    dtype=model_output.dtype,
                    device=device,
                    generator=generator,
                )
            )

        prev_sample = prev_sample + noise * sigma_up * s_noise

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return EulerAncestralDiscreteSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample
        )
