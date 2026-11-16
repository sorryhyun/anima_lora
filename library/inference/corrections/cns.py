"""CNS: Colored Noise Sampling recolorer (arXiv 2605.30332, Algorithm 1).

Numerics live in :mod:`library.inference.corrections.cns_core` (shared with the
ComfyUI Spectrum node); this module resolves the shipped γ artifact under the
repo home (``--cns auto`` → ``DEFAULT_GAMMA_PATH``). The γ table is built by
``scripts/calibration/cns_calibrate.py``.

Seam: ``ERSDESampler._sample_noise`` (``library/inference/sampling.py``). CNS is a
no-op on the euler/ODE default (no injected noise); only ``--sampler er_sde``
has a surface. See ``docs/inference/cns.md``.
"""

from __future__ import annotations

from library.env import resolve_under_home
from library.inference.corrections.cns_core import CNSRecolorer as _CNSRecolorerCore
from library.inference.corrections.cns_core import radial_bins

__all__ = ["CNSRecolorer", "radial_bins", "DEFAULT_GAMMA_PATH"]

# Aspect-averaged single γ, shape (1, T, F): cross-aspect variation is cosmetic
# (β MAD ~0.01), so the nearest-aspect select always picks row 0.
DEFAULT_GAMMA_PATH = "networks/calibration/cns_gamma.npz"


class CNSRecolorer(_CNSRecolorerCore):
    """:class:`~library.inference.corrections.cns_core.CNSRecolorer` with repo-home path resolution."""

    @classmethod
    def from_path(cls, path: str, strength: float = 1.0) -> "CNSRecolorer":
        """Load from an npz path, or the literal ``"auto"`` (shipped default)."""
        p = DEFAULT_GAMMA_PATH if path == "auto" else path
        return cls.from_npz(resolve_under_home(p), strength=strength)
