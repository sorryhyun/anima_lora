"""CNS — Colored Noise Sampling recolorer numerics core (arXiv 2605.30332).

Pure compute: radial-frequency binning + the γ(f, t)-driven noise recolorer.
**No anima/comfy imports and no path resolution** — the single source of truth
shared verbatim between the library plugin (``library/inference/corrections/cns.py``,
which resolves the γ artifact under the repo home) and the ComfyUI Spectrum node
(which resolves it under ComfyUI's models dir + auto-downloads). Each side adds
only a thin ``from_path`` that resolves its artifact location then calls
:meth:`CNSRecolorer.from_npz`. The node copy is vendored by sync_vendor.py.

Training-free SDE plug-in. Replaces the **white** noise an ER-SDE sampler injects
each step with **frequency-colored**, RMS-normalized noise that dumps the fixed
stochastic-energy budget into the radial bands the network has not yet resolved
at that step (per the precomputed completion matrix γ(f, t)). Zero-sum
reallocation of a fixed variance budget — RMS renormalization (paper §A) is
load-bearing.

DO NOT EDIT the vendored copy — regenerate it via scripts/release/sync_vendor.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch


def radial_bins(h: int, w: int, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Radial-frequency bin index per FFT cell + bin centers in [0, 1].

    Kept identical to ``scripts/calibration/gamma_probe.py::_radial_bins``. The
    normalization (r / r.max()) makes the bin *centers*
    independent of (h, w), so a γ matrix calibrated at one aspect's grid maps
    cleanly onto another shape's radial map by bin index.
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

    γ is stored per aspect (the spectral content — hence the staircase — shifts
    with aspect). On first use the recolorer
    locks onto the calibrated aspect closest in aspect-ratio to the inference
    latent shape, then for each step σ-interpolates γ (robust to a step-count or
    flow_shift mismatch vs the calibration schedule) and applies::

        scale(f)  = sqrt(1 - γ(f, σ))          # Eq. 11 numerator
        W         = fft2(white) * scale[bin(f)]
        w_c       = ifft2(W).real
        w_c      /= std(w_c)                    # RMS-normalize → conserve variance

    ``strength`` ∈ [0, 1] blends white↔recolored (then renormalizes) as a safety
    knob; 1.0 is full CNS, 0.0 is a pass-through (== white noise).
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

        # σ at each step (== sigmas[:-1]), decreasing. Sort ascending once so
        # np.interp (which needs increasing xp, and clamps at the endpoints) can
        # evaluate γ at an arbitrary inference σ.
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
        """Construct from an already-resolved ``.npz`` path (no auto/home logic).

        Path *resolution* is the per-framework seam — callers resolve their
        artifact location (repo home vs ComfyUI models dir) then hand the path
        here. ``cls`` keeps subclassing working: each side's thin ``from_path``
        delegates to this and gets an instance of its own subclass.
        """
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

        ``white`` is ``(B, C, 1, H, W)`` (Anima's fake-5D latent) or ``(B, C, H,
        W)`` — FFT runs over the trailing two dims either way; RMS renorm is
        per-(leading-dims) over the spatial plane so each channel keeps unit
        variance, exactly like the white noise it replaces.
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
