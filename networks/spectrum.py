"""Spectrum: Adaptive Spectral Feature Forecasting for Anima inference acceleration.

Implements the Spectrum method (Han et al., CVPR 2026) for training-free diffusion
sampling acceleration via Chebyshev polynomial feature forecasting.

Instead of running all transformer blocks at every denoising step, Spectrum:
1. Observes block outputs at a subset of steps (actual forwards)
2. Fits Chebyshev polynomial coefficients via ridge regression
3. Forecasts block outputs at skipped steps (cached)
4. Runs only t_embedder + final_layer + unpatchify on cached steps

Core forecasting algorithm adapted from:
  Spectrum (Han et al., CVPR 2026) — https://github.com/yangheng95/Spectrum
  Original source: src/utils/basis_utils.py
"""

import math
import logging
from typing import Optional, Tuple

import torch
from tqdm import tqdm

from library.inference.adapters import clear_hydra_sigma, set_hydra_sigma
from library.inference import sampling as inference_utils

logger = logging.getLogger(__name__)

DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# Chebyshev polynomial forecaster (adapted from Spectrum repo)
# ---------------------------------------------------------------------------


def _flatten(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
    shape = x.shape
    return x.reshape(1, -1), shape


def _unflatten(x_flat: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return x_flat.reshape(shape)


class ChebyshevForecaster:
    """Chebyshev T-polynomial ridge regression forecaster.

    Maintains a sliding window of (t, feature) observations, fits Chebyshev
    polynomial coefficients via ridge regression (Cholesky solve), and predicts
    features at arbitrary timesteps.

    Args:
        M: Number of Chebyshev basis functions (degree).
        K: Maximum window size (number of observations to keep).
        lam: Ridge regression regularization strength.
        device: Torch device for buffers.
    """

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

        self.t_buf = torch.empty(0)  # (<=K,)
        self._H_buf: Optional[torch.Tensor] = None  # (<=K, F)
        self._shape: Optional[torch.Size] = None
        self._coef: Optional[torch.Tensor] = None  # (P, F)

    @property
    def P(self) -> int:
        return self.M + 1

    def _taus(self, t: torch.Tensor) -> torch.Tensor:
        """Map step index t ∈ [0, total_steps) to τ ∈ [-1, 1]."""
        return 2.0 * (t / self.total_steps) - 1.0

    def _build_design(self, taus: torch.Tensor) -> torch.Tensor:
        """Build Chebyshev design matrix [T0, T1, ..., TM] via recurrence."""
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
        """Append observation (t, h) to the sliding window."""
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

        self._coef = None  # invalidate cached coefficients

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
        """Predict feature at timestep t_star via Chebyshev regression."""
        assert self._shape is not None
        self._fit_if_needed()
        tau_star = self._taus(t_star)
        x_star = self._build_design(tau_star[None])  # (1, P)
        h_flat = x_star @ self._coef  # (1, F)
        return _unflatten(h_flat, self._shape)


# ---------------------------------------------------------------------------
# Spectrum predictor (Chebyshev + optional Taylor blending)
# ---------------------------------------------------------------------------


class SpectrumPredictor:
    """Chebyshev polynomial forecaster with optional first-order Taylor blending.

    Wraps ChebyshevForecaster and blends with a discrete Newton forward-difference
    extrapolation for improved stability on the most recent observations.
    """

    def __init__(
        self,
        m: int,
        lam: float,
        w: float,
        device: torch.device,
        feature_shape,
        total_steps: int = 30,
    ):
        self.cheb = ChebyshevForecaster(
            M=m, K=100, lam=lam, device=device, total_steps=total_steps
        )
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

        # First-order discrete Taylor (Newton forward difference)
        H = self.cheb._H_buf  # (K, F) flattened
        t = self.cheb.t_buf
        dt = (t[-1] - t[-2]).clamp_min(1e-8)
        k = ((t_star_t - t[-1]) / dt).to(H.dtype)
        h_taylor = (H[-1] + k * (H[-1] - H[-2])).reshape(h_cheb.shape)

        return (1 - self.w) * h_taylor + self.w * h_cheb


def _spectrum_fast_forward(
    model, timesteps_B_T: torch.Tensor, predicted_feature: torch.Tensor
) -> torch.Tensor:
    """Fast path: t_embedder -> final_layer -> unpatchify (skips all blocks)."""
    if timesteps_B_T.ndim == 1:
        timesteps_B_T = timesteps_B_T.unsqueeze(1)
    t_emb, adaln = model.t_embedder(timesteps_B_T)
    t_emb = model.t_embedding_norm(t_emb)
    # Unconditional: buffer is zeros when mod guidance is disabled (see
    # Anima.__init__), so this collapses to identity.
    t_emb = t_emb + model._mod_guidance_delta.unsqueeze(1)
    x = model.final_layer(predicted_feature, t_emb, adaln_lora_B_T_3D=adaln)
    return model.unpatchify(x)


def spectrum_denoise(
    anima,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    sigmas: torch.Tensor,
    embed: torch.Tensor,
    negative_embed: torch.Tensor,
    padding_mask: torch.Tensor,
    guidance_scale: float,
    sampler,  # ERSDESampler or None
    device: torch.device,
    *,
    window_size: float = 2.0,
    flex_window: float = 0.25,
    warmup_steps: int = 6,
    w: float = 0.3,
    m: int = 3,
    lam: float = 0.1,
    stop_caching_step: int = -1,
    calibration_strength: float = 0.0,
    autocast_enabled: bool = False,
    pgraft_network=None,
    lora_cutoff_step: Optional[int] = None,
    pooled_text_pos: Optional[torch.Tensor] = None,
    pooled_text_neg: Optional[torch.Tensor] = None,
    postfix_net=None,
    postfix_base_embed: Optional[torch.Tensor] = None,
    postfix_base_neg: Optional[torch.Tensor] = None,
    postfix_embed_seqlens: Optional[torch.Tensor] = None,
    postfix_neg_seqlens: Optional[torch.Tensor] = None,
    dcw: bool = False,
    dcw_lambda: float = -0.015,
    dcw_schedule: str = "one_minus_sigma",
    dcw_band_mask: str = "LL",
    dcw_calibrator=None,
) -> torch.Tensor:
    """Spectrum-accelerated denoising loop.

    Replaces the standard step-by-step denoising with adaptive scheduling:
    early steps (high noise) get more actual forwards; later steps (refinement)
    are increasingly predicted via Chebyshev polynomial fitting.

    Args:
        window_size: Initial window N — actual forward every floor(N) steps.
        flex_window: Growth rate alpha — N += alpha after each actual forward.
        warmup_steps: Number of initial steps that always run actual forwards.
        w: Chebyshev/Taylor blend weight (1.0 = pure Chebyshev).
        m: Number of Chebyshev basis functions.
        lam: Ridge regression regularization strength.
        stop_caching_step: Force actual forwards from this step onward (-1 = auto: total_steps - 3).
        calibration_strength: Residual calibration strength (0.0 = disabled). On actual forwards,
            computes residual = actual - predicted; on cached steps, adds residual * strength.
    """
    do_cfg = guidance_scale != 1.0
    num_steps = len(timesteps)

    # Adaptive window schedule state
    curr_ws = window_size
    consec_cached = 0
    fwd_count = 0
    stop_at = num_steps - 3 if stop_caching_step < 0 else stop_caching_step

    # Forecasters (created lazily on first actual forward)
    cond_fc: Optional[SpectrumPredictor] = None
    uncond_fc: Optional[SpectrumPredictor] = None

    # Residual calibration: bias correction from last actual forward
    cond_residual: Optional[torch.Tensor] = None
    uncond_residual: Optional[torch.Tensor] = None

    # Register hook on final_layer to capture block output (its input)
    captured = {}

    def _capture_pre_hook(module, args):
        # args[0] = x_B_T_H_W_D (block output, after static unpadding)
        captured["feat"] = args[0].detach().clone()

    hook = anima.final_layer.register_forward_pre_hook(_capture_pre_hook)

    try:
        with tqdm(total=num_steps, desc="Spectrum") as pbar:
            for i, t in enumerate(timesteps):
                # P-GRAFT cutoff
                if (
                    pgraft_network is not None
                    and lora_cutoff_step is not None
                    and i == lora_cutoff_step
                ):
                    pgraft_network.set_enabled(False)
                    logger.info(f"P-GRAFT: Disabled LoRA at step {i}/{num_steps}")

                # Decide: actual forward or cached prediction?
                if i < warmup_steps or i >= stop_at:
                    actual = True
                else:
                    actual = (consec_cached + 1) % max(1, math.floor(curr_ws)) == 0

                t_exp = t.expand(latents.shape[0])
                set_hydra_sigma(anima, t_exp)

                # σ-conditional postfix: recompute per step on actual forwards.
                # Cached steps skip all blocks so cross-attn (and thus postfix) is
                # never consumed there — nothing to plumb on that branch.
                if postfix_net is not None:
                    step_embed = postfix_net.append_postfix(
                        postfix_base_embed, postfix_embed_seqlens, timesteps=t_exp
                    )
                    step_negative = postfix_net.append_postfix(
                        postfix_base_neg, postfix_neg_seqlens, timesteps=t_exp
                    )
                else:
                    step_embed = embed
                    step_negative = negative_embed

                if actual:
                    # --- Full forward pass ---
                    with (
                        torch.no_grad(),
                        torch.autocast(
                            device_type=device.type,
                            dtype=torch.bfloat16,
                            enabled=autocast_enabled,
                        ),
                    ):
                        _pos_kw = (
                            {"pooled_text_override": pooled_text_pos}
                            if pooled_text_pos is not None
                            else {}
                        )
                        noise_pred = anima(
                            latents, t_exp, step_embed, padding_mask=padding_mask, **_pos_kw
                        )
                    feat = captured["feat"]
                    if cond_fc is None:
                        cond_fc = SpectrumPredictor(
                            m, lam, w, device, feat.shape[1:], num_steps
                        )
                    # Residual calibration: measure prediction error before updating
                    if calibration_strength > 0 and cond_fc.cheb.t_buf.numel() >= 2:
                        cond_residual = feat - cond_fc.predict(float(i))
                    cond_fc.update(float(i), feat)

                    if do_cfg:
                        with (
                            torch.no_grad(),
                            torch.autocast(
                                device_type=device.type,
                                dtype=torch.bfloat16,
                                enabled=autocast_enabled,
                            ),
                        ):
                            _neg_kw = (
                                {"pooled_text_override": pooled_text_neg}
                                if pooled_text_neg is not None
                                else {}
                            )
                            uncond_noise_pred = anima(
                                latents,
                                t_exp,
                                step_negative,
                                padding_mask=padding_mask,
                                **_neg_kw,
                            )
                        ufeat = captured["feat"]
                        if uncond_fc is None:
                            uncond_fc = SpectrumPredictor(
                                m, lam, w, device, ufeat.shape[1:], num_steps
                            )
                        if (
                            calibration_strength > 0
                            and uncond_fc.cheb.t_buf.numel() >= 2
                        ):
                            uncond_residual = ufeat - uncond_fc.predict(float(i))
                        uncond_fc.update(float(i), ufeat)
                        noise_pred = uncond_noise_pred + guidance_scale * (
                            noise_pred - uncond_noise_pred
                        )

                    # Advance schedule (only post-warmup to avoid inflating window)
                    if i >= warmup_steps:
                        curr_ws = round(curr_ws + flex_window, 3)
                    consec_cached = 0
                    fwd_count += 1
                    pbar.set_postfix(mode="fwd", ws=f"{curr_ws:.1f}", n=fwd_count)

                else:
                    # --- Cached step: predict features, skip all blocks ---
                    with torch.no_grad():
                        pred_feat = cond_fc.predict(float(i))
                        if cond_residual is not None:
                            pred_feat = pred_feat + calibration_strength * cond_residual
                        noise_pred = _spectrum_fast_forward(anima, t_exp, pred_feat)

                        if do_cfg:
                            upred_feat = uncond_fc.predict(float(i))
                            if uncond_residual is not None:
                                upred_feat = (
                                    upred_feat + calibration_strength * uncond_residual
                                )
                            uncond_noise_pred = _spectrum_fast_forward(
                                anima, t_exp, upred_feat
                            )
                            noise_pred = uncond_noise_pred + guidance_scale * (
                                noise_pred - uncond_noise_pred
                            )

                    consec_cached += 1
                    pbar.set_postfix(mode="cached", n=fwd_count)

                # Sampler step
                denoised = latents.float() - sigmas[i] * noise_pred.float()
                if sampler is not None:
                    new_latents = sampler.step(latents, denoised, i)
                else:
                    new_latents = inference_utils.step(latents, noise_pred, sigmas, i)

                # DCW v4: observe post-CFG noise_pred + maybe fire the head.
                # Warmup observations all land within Spectrum's warmup window
                # (Spectrum forces actual forwards while i < warmup_steps), so
                # v4 sees real-DiT velocities even when caching kicks in later.
                if dcw_calibrator is not None:
                    dcw_calibrator.record(i, noise_pred)
                    dcw_calibrator.fire_head_if_due(i)

                # DCW: bias-correct against denoised x0_pred (carries Spectrum's
                # cached-step prediction error, but correction is bias-agnostic).
                if float(sigmas[i + 1]) > 0.0 and (
                    dcw_calibrator is not None or dcw
                ):
                    from networks.dcw import apply_dcw, parse_band_mask

                    if dcw_calibrator is not None:
                        lam_i_calib = dcw_calibrator.lambda_for_step(i, float(sigmas[i]))
                        new_latents = apply_dcw(
                            new_latents.float(),
                            denoised,
                            float(sigmas[i]),
                            lam=lam_i_calib,
                            schedule="const",
                            bands=frozenset({"LL"}),
                        )
                    else:
                        new_latents = apply_dcw(
                            new_latents.float(),
                            denoised,
                            float(sigmas[i]),
                            lam=dcw_lambda,
                            schedule=dcw_schedule,
                            bands=parse_band_mask(dcw_band_mask),
                        )

                latents = new_latents.to(latents.dtype)

                pbar.update()

        speedup = num_steps / max(1, fwd_count)
        cfg_label = " (x2 for CFG)" if do_cfg else ""
        logger.info(
            f"Spectrum: {fwd_count}/{num_steps} actual forwards "
            f"({speedup:.2f}x theoretical speedup{cfg_label})"
        )

    finally:
        clear_hydra_sigma(anima)
        # P-GRAFT restore
        if pgraft_network is not None and lora_cutoff_step is not None:
            pgraft_network.set_enabled(True)
        hook.remove()

    return latents


# Register with library.inference.generation so generate() can dispatch to us
# without holding a hard import edge from generation.py back into this file.
from library.inference.generation import register_spectrum_runner  # noqa: E402

register_spectrum_runner(spectrum_denoise)
