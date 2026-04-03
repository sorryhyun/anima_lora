"""Spectrum: Adaptive Spectral Feature Forecasting for ComfyUI.

Drop-in KSampler replacement that accelerates diffusion sampling via
Chebyshev polynomial feature forecasting (Han et al., CVPR 2026).

On "actual" steps the full model runs and block outputs are captured.
On "cached" steps all transformer blocks are skipped — only t_embedder +
final_layer + unpatchify execute, using predicted features from a
Chebyshev ridge-regression fit. Works with any ComfyUI sampler (Euler,
DPM, er_sde, etc.) because caching is handled transparently inside the
model_function_wrapper.
"""

import math
import logging
from typing import Optional, Tuple, Dict

import torch

import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_management
import latent_preview

logger = logging.getLogger(__name__)

DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# Chebyshev polynomial forecaster (adapted from anima_lora/library/spectrum.py)
# ---------------------------------------------------------------------------


def _flatten(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
    shape = x.shape
    return x.reshape(1, -1), shape


def _unflatten(x_flat: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return x_flat.reshape(shape)


class ChebyshevForecaster:
    """Chebyshev T-polynomial ridge regression forecaster."""

    def __init__(
        self,
        M: int = 4,
        K: int = 10,
        lam: float = 1e-3,
        device: Optional[torch.device] = None,
        total_steps: int = 30,
    ):
        assert K >= M + 2, "K should exceed basis size for stability"
        self.M = M
        self.K = K
        self.lam = lam
        self.device = device
        self.total_steps = total_steps

        self.t_buf = torch.empty(0)
        self._H_buf: Optional[torch.Tensor] = None
        self._shape: Optional[torch.Size] = None
        self._coef: Optional[torch.Tensor] = None

    @property
    def P(self) -> int:
        return self.M + 1

    def _taus(self, t: torch.Tensor) -> torch.Tensor:
        return 2.0 * (t / self.total_steps) - 1.0

    def _build_design(self, taus: torch.Tensor) -> torch.Tensor:
        taus = taus.reshape(-1, 1)
        K = taus.shape[0]
        T0 = torch.ones((K, 1), device=taus.device, dtype=taus.dtype)
        if self.M == 0:
            return T0
        T1 = taus
        cols = [T0, T1]
        for _ in range(2, self.M + 1):
            Tm = 2 * taus * cols[-1] - cols[-2]
            cols.append(Tm)
        return torch.cat(cols[: self.M + 1], dim=1)

    def update(self, t: float, h: torch.Tensor) -> None:
        device = self.device or h.device
        t_tensor = torch.as_tensor(t, dtype=DTYPE, device=device)
        h_flat, shape = _flatten(h)
        h_flat = h_flat.to(device)

        if self._shape is None:
            self._shape = shape
        else:
            assert shape == self._shape, "Feature shape must remain constant"

        if self.t_buf.numel() == 0:
            self.t_buf = t_tensor[None]
            self._H_buf = h_flat
        else:
            self.t_buf = torch.cat([self.t_buf, t_tensor[None]], dim=0)
            self._H_buf = torch.cat([self._H_buf, h_flat], dim=0)
            if self.t_buf.numel() > self.K:
                self.t_buf = self.t_buf[-self.K :]
                self._H_buf = self._H_buf[-self.K :]

        self._coef = None

    def _fit_if_needed(self) -> None:
        if self._coef is not None:
            return
        taus = self._taus(self.t_buf)
        X = self._build_design(taus).to(torch.float32)
        H = self._H_buf.to(torch.float32)
        P = X.shape[1]

        lamI = self.lam * torch.eye(P, device=X.device, dtype=X.dtype)
        Xt = X.T
        XtX = Xt @ X + lamI
        try:
            L = torch.linalg.cholesky(XtX)
        except torch.linalg.LinAlgError:
            jitter = 1e-6 * XtX.diag().mean()
            L = torch.linalg.cholesky(
                XtX + jitter * torch.eye(P, device=X.device, dtype=X.dtype)
            )
        XtH = Xt @ H
        self._coef = torch.cholesky_solve(XtH, L).to(DTYPE)

    @torch.no_grad()
    def predict(self, t_star: torch.Tensor) -> torch.Tensor:
        assert self._shape is not None
        self._fit_if_needed()
        tau_star = self._taus(t_star)
        x_star = self._build_design(tau_star[None])
        h_flat = x_star @ self._coef
        return _unflatten(h_flat, self._shape)


class SpectrumPredictor:
    """Chebyshev polynomial forecaster with optional first-order Taylor blending."""

    def __init__(self, m: int, lam: float, w: float, device: torch.device,
                 feature_shape, total_steps: int = 30):
        self.cheb = ChebyshevForecaster(M=m, K=100, lam=lam, device=device,
                                         total_steps=total_steps)
        self.w = w

    def update(self, t: float, h: torch.Tensor):
        self.cheb.update(t, h)

    @torch.no_grad()
    def predict(self, t_star: float) -> torch.Tensor:
        device = self.cheb.t_buf.device
        t_star_t = torch.as_tensor(t_star, dtype=DTYPE, device=device)
        h_cheb = self.cheb.predict(t_star_t)

        if self.w >= 1.0 or self.cheb.t_buf.numel() < 2:
            return h_cheb

        H = self.cheb._H_buf
        t = self.cheb.t_buf
        dt = (t[-1] - t[-2]).clamp_min(1e-8)
        k = ((t_star_t - t[-1]) / dt).to(H.dtype)
        h_taylor = (H[-1] + k * (H[-1] - H[-2])).reshape(h_cheb.shape)
        return (1 - self.w) * h_taylor + self.w * h_cheb


# ---------------------------------------------------------------------------
# Fast-forward: t_embedder -> final_layer -> unpatchify (skip all blocks)
# ---------------------------------------------------------------------------

def _spectrum_fast_forward(
    dit, timestep: torch.Tensor, predicted_feature: torch.Tensor
) -> torch.Tensor:
    """Runs only t_embedder + final_layer + unpatchify on predicted features.

    Returns the same shape as diffusion_model.forward() — 5D for video DiTs.
    """
    if timestep.ndim == 1:
        timestep = timestep.unsqueeze(1)
    # Replicate the model's two-step t_embedder call: Timesteps (sinusoidal,
    # always float32) -> cast to model dtype -> TimestepEmbedding (linear layers).
    # Calling t_embedder as a single Sequential skips the intermediate cast.
    t_sinusoidal = dit.t_embedder[0](timestep)
    t_emb, adaln = dit.t_embedder[1](t_sinusoidal.to(predicted_feature.dtype))
    t_emb = dit.t_embedding_norm(t_emb)
    x = dit.final_layer(predicted_feature, t_emb, adaln_lora_B_T_3D=adaln)
    return dit.unpatchify(x)


# ---------------------------------------------------------------------------
# Spectrum state (shared between wrapper and node)
# ---------------------------------------------------------------------------

class SpectrumState:
    def __init__(self, window_size: float, flex_window: float, warmup_steps: int,
                 w: float, m: int, lam: float, num_steps: int):
        self.window_size = window_size
        self.flex_window = flex_window
        self.warmup_steps = warmup_steps
        self.w = w
        self.m_param = m
        self.lam = lam
        self.num_steps = num_steps

        # Runtime
        self.step_idx = -1
        self.last_sigma: Optional[float] = None
        self.mode = "actual"
        self.curr_ws = window_size
        self.consec_cached = 0
        self.fwd_count = 0

        # Forecasters keyed by cond_or_uncond value (0=cond, 1=uncond)
        self.forecasters: Dict[int, SpectrumPredictor] = {}
        self.captured_feat: Optional[torch.Tensor] = None
        self._hook_handle = None
        self._hook_installed = False

    def should_cache(self) -> bool:
        if self.step_idx < self.warmup_steps:
            return False
        stop_at = self.num_steps - 3
        if self.step_idx >= stop_at:
            return False
        return (self.consec_cached + 1) % max(1, math.floor(self.curr_ws)) != 0

    def has_forecasters(self, cond_or_uncond: list) -> bool:
        return all(cou in self.forecasters for cou in cond_or_uncond)

    def install_hook(self, dit):
        if self._hook_installed:
            return

        def capture_pre_hook(module, args):
            self.captured_feat = args[0].detach().clone()

        self._hook_handle = dit.final_layer.register_forward_pre_hook(capture_pre_hook)
        self._hook_installed = True

    def remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
            self._hook_installed = False


# ---------------------------------------------------------------------------
# ComfyUI node
# ---------------------------------------------------------------------------

class SpectrumKSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "window_size": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.25,
                                          "tooltip": "Initial caching window N — actual forward every floor(N) steps."}),
                "flex_window": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.05,
                                          "tooltip": "Window growth rate — N increases by this after each actual forward."}),
                "warmup_steps": ("INT", {"default": 6, "min": 0, "max": 50,
                                         "tooltip": "Number of initial steps that always run actual forwards."}),
                "blend_w": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "Chebyshev/Taylor blend weight (1.0 = pure Chebyshev)."}),
                "cheby_degree": ("INT", {"default": 3, "min": 1, "max": 10,
                                         "tooltip": "Number of Chebyshev basis functions."}),
                "ridge_lambda": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 10.0, "step": 0.01,
                                           "tooltip": "Ridge regression regularization strength."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    DESCRIPTION = ("Spectrum-accelerated sampler. Drop-in KSampler replacement that "
                   "skips transformer blocks on predicted steps via Chebyshev polynomial "
                   "feature forecasting for ~2-3x speedup.")

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive,
               negative, latent_image, denoise=1.0, window_size=2.0, flex_window=0.25,
               warmup_steps=6, blend_w=0.3, cheby_degree=3, ridge_lambda=0.1):

        # --- Clone model and install Spectrum ---
        m = model.clone()

        state = SpectrumState(
            window_size=window_size,
            flex_window=flex_window,
            warmup_steps=warmup_steps,
            w=blend_w,
            m=cheby_degree,
            lam=ridge_lambda,
            num_steps=steps,
        )

        dit = m.model.diffusion_model
        model_sampling = m.model.model_sampling

        # Chain with any existing wrapper (e.g., FlashAttention4)
        old_wrapper = m.model_options.get("model_function_wrapper")

        def spectrum_wrapper(apply_model, args):
            input_x = args["input"]
            timestep = args["timestep"]
            c = args["c"]
            cond_or_uncond = args["cond_or_uncond"]

            # Install hook lazily (model is on GPU by now)
            if not state._hook_installed:
                state.install_hook(dit)

            sigma_val = timestep[0].item()

            # --- Detect new sampling step via sigma change ---
            if state.last_sigma is None or abs(sigma_val - state.last_sigma) > 1e-8:
                # Bookkeeping for the previous step
                if state.step_idx >= 0:
                    if state.mode == "actual":
                        state.fwd_count += 1
                        if state.step_idx >= state.warmup_steps:
                            state.curr_ws = round(state.curr_ws + state.flex_window, 3)
                        state.consec_cached = 0
                    else:
                        state.consec_cached += 1

                # Advance
                state.step_idx += 1
                state.last_sigma = sigma_val
                state.mode = "cached" if state.should_cache() else "actual"

            # --- Cached step: predict features, skip all blocks ---
            if state.mode == "cached" and state.has_forecasters(cond_or_uncond):
                predictions = []
                for cou in cond_or_uncond:
                    pred_feat = state.forecasters[cou].predict(float(state.step_idx))
                    predictions.append(pred_feat)

                batched_feat = torch.cat(predictions, dim=0)
                t_internal = model_sampling.timestep(timestep).to(batched_feat.dtype)
                noise_pred = _spectrum_fast_forward(dit, t_internal, batched_feat)
                return model_sampling.calculate_denoised(
                    timestep, noise_pred.float(), input_x
                )

            # --- Actual step: full forward ---
            state.mode = "actual"  # In case we fell through from cached

            if old_wrapper is not None:
                result = old_wrapper(apply_model, args)
            else:
                result = apply_model(input_x, timestep, **c)

            # Capture features from the hook and update forecasters
            feat = state.captured_feat
            if feat is not None:
                batch_chunks = len(cond_or_uncond)
                feat_chunks = feat.chunk(batch_chunks, dim=0)
                for idx, cou in enumerate(cond_or_uncond):
                    if cou not in state.forecasters:
                        state.forecasters[cou] = SpectrumPredictor(
                            state.m_param, state.lam, state.w,
                            feat.device, feat_chunks[idx].shape,
                            state.num_steps,
                        )
                    state.forecasters[cou].update(
                        float(state.step_idx), feat_chunks[idx]
                    )

            return result

        m.set_model_unet_function_wrapper(spectrum_wrapper)

        # --- Run standard ComfyUI sampling pipeline ---
        latent_img = latent_image["samples"].clone()
        latent_img = comfy.sample.fix_empty_latent_channels(
            m, latent_img, latent_image.get("downscale_ratio_spacial")
        )

        batch_inds = latent_image.get("batch_index")
        noise = comfy.sample.prepare_noise(latent_img, seed, batch_inds)

        noise_mask = latent_image.get("noise_mask")
        callback = latent_preview.prepare_callback(m, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        samples = comfy.sample.sample(
            m, noise, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_img,
            denoise=denoise, noise_mask=noise_mask,
            callback=callback, disable_pbar=disable_pbar, seed=seed,
        )

        # Final-step bookkeeping
        if state.step_idx >= 0:
            if state.mode == "actual":
                state.fwd_count += 1
            else:
                state.consec_cached += 1

        # Cleanup
        state.remove_hook()

        # Log
        actual = state.fwd_count
        total = state.step_idx + 1
        speedup = total / max(1, actual)
        do_cfg = not math.isclose(cfg, 1.0)
        cfg_note = " (x2 for CFG)" if do_cfg else ""
        logger.info(
            f"Spectrum: {actual}/{total} actual forwards "
            f"({speedup:.2f}x theoretical speedup{cfg_note})"
        )

        out = latent_image.copy()
        out.pop("downscale_ratio_spacial", None)
        out["samples"] = samples
        return (out,)


NODE_CLASS_MAPPINGS = {
    "SpectrumKSampler": SpectrumKSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpectrumKSampler": "KSampler (Spectrum)",
}
