"""CNS — Colored Noise Sampling recolorer numerics core (arXiv 2605.30332).

Pure compute, no anima/comfy imports and no path resolution: shared by the
library plugin (``corrections/cns.py``) and the ComfyUI Spectrum node (vendored
by sync_vendor.py); each adds a thin ``from_path`` that calls
:meth:`CNSRecolorer.from_npz`.

Replaces the white noise an ER-SDE sampler injects each step with
frequency-colored noise that moves the fixed variance budget into the radial
bands the network has not yet resolved (per the completion matrix γ(f, t)).
The RMS renormalization (paper §A) is load-bearing: it keeps the reallocation
zero-sum.

DO NOT EDIT the vendored copy — regenerate it via scripts/release/sync_vendor.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch


def radial_bins(h: int, w: int, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Radial-frequency bin index per FFT cell + bin centers in [0, 1].

    Kept identical to ``scripts/calibration/gamma_probe.py::_radial_bins``.
    ``r / r.max()`` makes bin centers independent of (h, w), so a γ calibrated
    on one grid maps onto another by bin index.
    """
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fy**2 + fx**2)
    r = r / r.max()
    edges = np.linspace(0.0, 1.0 + 1e-9, n_bins + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return idx, centers


class CNSRecolorer:
    """Recolors per-step SDE white noise from a precomputed γ(f, t) matrix.

    γ is stored per aspect. On first use the recolorer locks onto the closest
    calibrated aspect ratio, then per step σ-interpolates γ (robust to a
    step-count or flow_shift mismatch vs calibration) and applies::

        scale(f)  = sqrt(1 - γ(f, σ))          # Eq. 11 numerator
        W         = fft2(white) * scale[bin(f)]
        w_c       = ifft2(W).real
        w_c      /= std(w_c)                    # RMS-normalize → conserve variance

    ``strength`` ∈ [0, 1] blends white↔recolored, then renormalizes.
    """

    def __init__(
        self,
        gamma: np.ndarray,  # (A, T, F)
        aspects: np.ndarray,  # (A, 2) pixel (H, W)
        sigmas: np.ndarray,  # (T+1,) calibration σ schedule
        strength: float = 1.0,
    ) -> None:
        if gamma.ndim != 3:
            raise ValueError(f"CNS gamma must be (A, T, F); got shape {gamma.shape}")
        self.gamma_all = np.asarray(gamma, dtype=np.float64)
        self.aspects = np.asarray(aspects, dtype=np.float64).reshape(-1, 2)
        self.F = self.gamma_all.shape[-1]
        self.strength = float(strength)

        # np.interp needs increasing xp; sigmas[:-1] is decreasing.
        sig_mid = np.asarray(sigmas, dtype=np.float64)[:-1]
        self._order = np.argsort(sig_mid)
        self._sig_asc = sig_mid[self._order]

        self._bin_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._sel_gamma: Optional[np.ndarray] = (
            None  # (T, F) ascending-σ, chosen aspect
        )
        self._sel_idx: Optional[int] = None

    @classmethod
    def from_npz(cls, resolved_path: str, strength: float = 1.0) -> "CNSRecolorer":
        """Construct from an already-resolved ``.npz`` path."""
        if not Path(resolved_path).exists():
            raise FileNotFoundError(
                f"CNS calibration not found: {resolved_path}. Generate it with "
                "`python scripts/calibration/cns_calibrate.py` (cfg=4.0, top-3 aspects)."
            )
        d = np.load(resolved_path)
        return cls(d["gamma"], d["aspects"], d["sigmas"], strength=strength)

    def _select_aspect(self, h_lat: int, w_lat: int) -> None:
        """Lock onto the calibrated aspect closest in AR to this latent shape."""
        ar = w_lat / max(h_lat, 1)
        cal_ar = self.aspects[:, 1] / np.maximum(self.aspects[:, 0], 1.0)
        self._sel_idx = int(np.argmin(np.abs(cal_ar - ar)))
        self._sel_gamma = self.gamma_all[self._sel_idx][self._order]  # (T, F)

    def _gamma_row(self, sigma_s: float) -> np.ndarray:
        """γ(f) at this σ for the selected aspect, via per-bin interpolation."""
        g = self._sel_gamma
        return np.array(
            [np.interp(sigma_s, self._sig_asc, g[:, f]) for f in range(self.F)],
            dtype=np.float64,
        )

    def _bin_map(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        key = (h, w)
        cached = self._bin_cache.get(key)
        if cached is None:
            idx, _ = radial_bins(h, w, self.F)
            cached = torch.from_numpy(idx).to(device=device, dtype=torch.long)
            self._bin_cache[key] = cached
        return cached

    def recolor(self, white: torch.Tensor, sigma_s: float) -> torch.Tensor:
        """Return frequency-recolored noise of the same shape/dtype as ``white``.

        ``white`` is ``(B, C, 1, H, W)`` (Anima 5D) or ``(B, C, H, W)``; FFT and
        RMS renorm run over the trailing two dims, so each channel keeps unit
        variance like the white noise it replaces.
        """
        if self.strength <= 0.0:
            return white
        h, w = white.shape[-2], white.shape[-1]
        if self._sel_gamma is None:
            self._select_aspect(h, w)

        scale_vec = np.sqrt(np.clip(1.0 - self._gamma_row(float(sigma_s)), 0.0, 1.0))
        bin_idx = self._bin_map(h, w, white.device)
        scale = torch.from_numpy(scale_vec).to(device=white.device, dtype=torch.float32)
        scale_map = scale[bin_idx]  # (H, W)

        wf = torch.fft.fft2(white.float(), dim=(-2, -1)) * scale_map
        wc = torch.fft.ifft2(wf, dim=(-2, -1)).real

        wc = wc / wc.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        if self.strength < 1.0:
            wc = (1.0 - self.strength) * white.float() + self.strength * wc
            wc = wc / wc.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return wc.to(white.dtype)
