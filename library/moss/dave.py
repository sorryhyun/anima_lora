"""DAVE (DC attenuation for diversity, ``docs/inference/dave.md``) on MOSS-Image.

The edit is Anima's: on each pooled block, during the first τ of the steps,
``h ← h − strength·μ`` with ``μ`` the per-channel mean over the image tokens.
What differs from ``library.inference.corrections.dave``:

- MOSS is single-stream, so a block's output is ``[image (H·W) | text]``. ``μ`` is
  taken over the image tokens only and only they are edited; the text tokens
  pass through.
- The edit runs through ``MOSSImageTransformer.layer_hook``, not a forward hook:
  under per-sample ``skip_layers`` (the skip-22 uncond) a block runs on a subset
  of rows, and the hook sees exactly those rows.
- τ is a step count (``denoise`` knows the step index), not a σ boundary.
- The massive-activation channels are left out of the edit. On MOSS the image
  tokens' mean is almost entirely one channel (1356) plus a few more, identical
  across seeds and prompts; removing a fraction of it would shift the block output
  instead of unlocking it. The list is ``DEFAULT_CHANNELS_PATH``, from
  ``scripts/calibration/moss_massive_channels.py``.

The pool is a ``{0,1}`` vector over the 42 blocks: a ``dave_alpha.npz`` (``auto`` =
the shipped one, blocks 19–30), or a block range ``"lo-hi"``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch

from library.env import resolve_under_home

# ``dave="auto"``: blocks 19–30 (MOSS's own pool, not Anima's mask).
DEFAULT_MASK_PATH = "library/moss/calibration/dave_alpha.npz"
DEFAULT_CHANNELS_PATH = "library/moss/calibration/massive_channels.npz"
_RANGE = re.compile(r"(\d+)-(\d+)")


def pool_weight(spec: str, num_blocks: int) -> np.ndarray:
    """``(num_blocks,)`` pool membership from ``"auto"``, ``"lo-hi"`` or an npz path."""
    m = _RANGE.fullmatch(spec.strip())
    if m:
        lo, hi = int(m[1]), int(m[2])
        if not 0 <= lo <= hi < num_blocks:
            raise ValueError(f"DAVE block range {spec!r} outside 0-{num_blocks - 1}")
        w = np.zeros(num_blocks)
        w[lo : hi + 1] = 1.0
        return w
    path = resolve_under_home(DEFAULT_MASK_PATH if spec == "auto" else spec)
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"DAVE mask not found: {path}. Pass a block range like '19-30'."
        )
    w = np.load(path)["weight"].astype(np.float64)
    if w.shape != (num_blocks,):
        raise ValueError(
            f"DAVE mask {path} has {w.shape[0]} blocks, the model has {num_blocks}"
        )
    return w


def massive_channels(path: str = DEFAULT_CHANNELS_PATH) -> np.ndarray:
    """Channel indices DAVE leaves out of the mean it subtracts."""
    resolved = resolve_under_home(path)
    if not Path(resolved).is_file():
        raise FileNotFoundError(
            f"MOSS massive-channel list not found: {resolved}. Derive it with "
            "scripts/calibration/moss_massive_channels.py."
        )
    return np.load(resolved)["channels"].astype(np.int64)


class MossDAVE:
    """``layer_hook`` for the first ``active_steps`` steps; ``denoise`` arms it."""

    def __init__(self, weight: np.ndarray, strength: float, tau: float, exclude=()):
        self.atten = np.clip(strength * np.asarray(weight, dtype=np.float64), 0.0, 1.0)
        self.tau = tau
        self.exclude = np.asarray(exclude, dtype=np.int64)
        self.active_steps = 0
        self._keep = None

    @classmethod
    def from_spec(
        cls, spec: str, num_blocks: int, strength: float = 0.3, tau: float = 0.10
    ) -> "MossDAVE":
        return cls(pool_weight(spec, num_blocks), strength, tau, massive_channels())

    def begin(self, num_steps: int) -> int:
        """Steps ``0 .. k-1`` attenuate, ``k = round(τ·steps)`` (at least 1)."""
        self.active_steps = max(1, min(num_steps, round(self.tau * num_steps)))
        return self.active_steps

    def is_active(self, i: int) -> bool:
        return i < self.active_steps

    def __call__(self, index: int, x: torch.Tensor, n_img: int) -> torch.Tensor:
        a = float(self.atten[index])
        if a == 0.0:
            return x
        img = x[:, :n_img].float()
        mu = img.mean(1, keepdim=True)
        if self.exclude.size:
            if self._keep is None or self._keep.device != x.device:
                self._keep = torch.ones(x.shape[-1], device=x.device)
                self._keep[torch.from_numpy(self.exclude).to(x.device)] = 0.0
            mu = mu * self._keep
        img = img - a * mu
        return torch.cat((img.to(x.dtype), x[:, n_img:]), 1)

    def describe(self) -> str:
        pool = np.flatnonzero(self.atten > 0)
        blocks = f"{pool[0]}-{pool[-1]} ({pool.size})" if pool.size else "none"
        return (
            f"DAVE blocks {blocks}, strength {self.atten.max():.2f}, "
            f"first {self.active_steps} steps, {self.exclude.size} channels left out"
        )
