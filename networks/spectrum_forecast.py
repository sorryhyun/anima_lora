"""Chebyshev feature forecasters for Spectrum — pure-compute single source.

The Chebyshev T-polynomial ridge-regression forecaster (``ChebyshevForecaster``)
and the Taylor-blended predictor (``SpectrumPredictor``) that ``networks/spectrum.py``
fits on captured ``final_layer`` features at actual steps and queries at cached
steps. Shared by the CLI Spectrum runner and the ComfyUI node
(``ComfyUI-Spectrum-KSampler``).

torch/stdlib only — no ``comfy``, no anima-model imports.

Core forecasting algorithm adapted from:
  Spectrum (Han et al., CVPR 2026) — https://github.com/yangheng95/Spectrum
  Original source: src/utils/basis_utils.py
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

DTYPE = torch.bfloat16


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
        """Predict feature at timestep t_star via Chebyshev regression."""
        assert self._shape is not None
        self._fit_if_needed()
        tau_star = self._taus(t_star)
        x_star = self._build_design(tau_star[None])  # (1, P)
        h_flat = x_star @ self._coef  # (1, F)
        return _unflatten(h_flat, self._shape)


class SpectrumPredictor:
    """Chebyshev polynomial forecaster with optional first-order Taylor blending.

    Wraps ChebyshevForecaster and blends with a discrete Newton forward-difference
    extrapolation for improved stability on the most recent observations.

    ``K`` is the sliding-window size handed to the underlying forecaster (the node
    threads its ``history_size`` knob through here; the CLI runner keeps the
    default 100).
    """

    def __init__(
        self,
        m: int,
        lam: float,
        w: float,
        device: torch.device,
        feature_shape,
        total_steps: int = 30,
        K: int = 100,
    ):
        self.cheb = ChebyshevForecaster(
            M=m, K=K, lam=lam, device=device, total_steps=total_steps
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
