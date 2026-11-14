"""CNS (colored-noise SDE, ``docs/inference/cns.md``) on MOSS-Image's packed latent.

The recolor numerics are Anima's (``library.inference.corrections.cns_core``). MOSS
adds one thing: the FFT runs on the **un-patchified** grid ``(B, 32, 2h, 2w)``, not
on the packed ``(B, 128, h, w)`` the DiT sees. On the packed grid the top octave of
the FLUX.2 latent lives in the 2×2 phase channels, out of reach of a spatial
filter, and that octave is the one CNS feeds late. The reshape is
``model.unpatchify``, ``unpack_latents`` without its BatchNorm affine: white noise
stays white under it, and γ is per-channel scale-invariant.

γ comes from the deterministic Euler trajectory (``gamma_for_trajectory``, the
same estimator as ``scripts/calibration/gamma_probe.py``); the recolor applies on
``sampler="er_sde"`` only. ``MossCNS.auto`` picks the shipped γ calibrated for the
variant at the cfg nearest the sampler's (``GAMMA_PATHS``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from library.env import resolve_under_home
from library.inference.corrections.cns_core import CNSRecolorer, radial_bins
from library.moss.model import patchify, unpatchify

# (variant, cfg) → γ from scripts/calibration/moss_cns_calibrate.py (50 steps).
GAMMA_PATHS = {
    ("base", 3.0): "networks/calibration/moss_cns_gamma.npz",
    ("base", 4.0): "networks/calibration/moss_cns_gamma_cfg4.npz",
    ("base", 5.0): "networks/calibration/moss_cns_gamma_cfg5.npz",
    ("sft", 3.0): "networks/calibration/moss_cns_gamma_sft_cfg3.npz",
    ("sft", 4.0): "networks/calibration/moss_cns_gamma_sft_cfg4.npz",
    ("sft", 5.0): "networks/calibration/moss_cns_gamma_sft_cfg5.npz",
}


def auto_gamma_path(variant: str, cfg: float) -> Path:
    """The shipped γ for ``variant`` at the calibrated cfg nearest ``cfg``."""
    cfgs = [c for v, c in GAMMA_PATHS if v == variant]
    if not cfgs:
        raise ValueError(f"no MOSS CNS γ for variant {variant!r}")
    nearest = min(cfgs, key=lambda c: abs(c - cfg))
    return resolve_under_home(GAMMA_PATHS[(variant, nearest)])


class MossCNS:
    """``recolor(white, σ)`` over the packed latent, for ``ERSDESampler(cns=…)``."""

    def __init__(self, core: CNSRecolorer):
        self.core = core

    @classmethod
    def from_npz(cls, path: str | Path, strength: float = 1.0) -> "MossCNS":
        return cls(CNSRecolorer.from_npz(str(path), strength=strength))

    @classmethod
    def from_spec(
        cls, spec: str | Path, variant: str, cfg: float, strength: float = 1.0
    ) -> "MossCNS":
        """``"auto"`` → :func:`auto_gamma_path`; anything else is an npz path."""
        path = auto_gamma_path(variant, cfg) if str(spec) == "auto" else spec
        return cls.from_npz(resolve_under_home(path), strength=strength)

    def recolor(self, white: torch.Tensor, sigma_s: float) -> torch.Tensor:
        return patchify(self.core.recolor(unpatchify(white), sigma_s))


@torch.no_grad()
def gamma_for_trajectory(
    x0_preds: list[torch.Tensor], x0_final: torch.Tensor, n_bins: int
) -> np.ndarray:
    """γ ``(T, F)`` for one trajectory of packed ``(1,128,h,w)`` latents:
    ``1 − |X̂₀ − X₀|² / |X₀|²`` per FFT cell, channel-mean, clipped, radially binned."""
    xf = torch.fft.fft2(unpatchify(x0_final.float())[0])
    pf = xf.abs() ** 2
    eps = 1e-8 * pf.mean(dim=(-2, -1), keepdim=True) + 1e-12
    idx, _ = radial_bins(xf.shape[-2], xf.shape[-1], n_bins)
    idx = torch.from_numpy(idx).to(xf.device).reshape(-1)
    counts = torch.bincount(idx, minlength=n_bins).double().clamp_min(1.0)
    out = np.zeros((len(x0_preds), n_bins), dtype=np.float64)
    for t, xp in enumerate(x0_preds):
        xpf = torch.fft.fft2(unpatchify(xp.float())[0])
        g = (1.0 - (xpf - xf).abs() ** 2 / (pf + eps)).mean(0).clamp(0.0, 1.0)
        out[t] = (
            (
                torch.bincount(idx, weights=g.reshape(-1).double(), minlength=n_bins)
                / counts
            )
            .cpu()
            .numpy()
        )
    return out
