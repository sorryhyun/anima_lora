"""DAVE — DC Attenuation for diVersity Enhancement (training-free, ICML'26).

Recovers same-prompt sample diversity by attenuating the **DC component** of each
target Transformer block's output — the spatial average ``μ^ℓ`` that the probe
found to be near-perfectly cross-seed-shared (it carries the conditioning /
global layout) while the **AC residual** ``h − μ`` holds the seed-specific
structure. The per-block edit::

    ĥ^ℓ = α_ℓ · μ^ℓ + (h^ℓ − μ^ℓ) = h^ℓ − (1 − α_ℓ) · μ^ℓ      (α_ℓ ≤ 1)

lets the diverse AC breathe without rewriting it. ``(1 − α_ℓ)`` is the per-block
attenuation factor, ``strength · w(ℓ)`` where ``w(ℓ)`` is the offline-derived
"Power Ratio" gate (``bench/dave/derive_alpha_mask.py``) that self-zeros block 0
and the AC-shared early-mid blocks. ``strength`` (``--dave_strength``) is the live
knob; the edit is additionally σ-gated to ``[--dave_sigma_lo, --dave_sigma_hi]``.

Design (per ``bench/dave/README.md``): a **post-`forward` hook** on each block
(forward_hook-not-override invariant) + a per-step model buffer (``_dave_cur_sigma``,
restamped each forward from the timestep), the Spectrum/mod-guidance pattern — *not*
a sampler-boundary correction like CNS. Branchless (``α=1 → atten=0 → exact
no-op``), so it survives ``compile_blocks()`` (the hook fires eager around each
block's compiled ``_forward``; the DC mean over dims (1,2,3) is correct for both the
eager 5D ``(B,T,H,W,D)`` and the native-flattened fake-5D ``(B,1,seq,1,D)`` layout).

Standard denoise loop only (no Spectrum/SPD compose). The edit applies to both the cond and uncond CFG forwards (a uniform
representation edit; the hook does not distinguish the two passes).

Reference: "Breaking the Lock-in: Diversifying Text-to-Image Generation via
Representation Modulation", ICML 2026 — https://github.com/daheekwon/DAVE
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import torch

from library.env import resolve_under_home

logger = logging.getLogger(__name__)

# Shipped mask (relative to repo home). ``--dave auto`` resolves here;
# an explicit path overrides. Produced by bench/dave/derive_alpha_mask.py.
DEFAULT_MASK_PATH = "networks/calibration/dave_alpha.npz"


class DAVEHooks:
    """Owns the per-block forward-hook handles so generation can detach them."""

    def __init__(self, handles: List[torch.utils.hooks.RemovableHandle]) -> None:
        self._handles = handles

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def _load_weight(path: str, num_blocks: int) -> np.ndarray:
    """Load the per-block weight vector w(ℓ) ∈ [0, 1] from the mask npz."""
    p = DEFAULT_MASK_PATH if path == "auto" else path
    resolved = resolve_under_home(p)
    if not Path(resolved).exists():
        raise FileNotFoundError(
            f"DAVE mask not found: {resolved}. Derive it with "
            "`uv run python bench/dave/derive_alpha_mask.py`."
        )
    w = np.load(resolved)["weight"].astype(np.float64)
    if w.shape[0] != num_blocks:
        raise ValueError(
            f"DAVE mask has {w.shape[0]} blocks but the model has {num_blocks}. "
            "Re-derive the mask from a probe run on this model."
        )
    return w


def _make_hook(anima, bidx: int):
    """Post-`forward` hook for block ``bidx``: ``out ← out − gate·atten·μ``.

    Branchless. ``atten`` (the per-block ``1−α``) is 0 for self-zeroed blocks, so
    those hooks are exact no-ops; ``gate`` is 0 outside the σ window. The DC mean
    over dims (1,2,3) is the per-channel spatial average for both the eager 5D and
    native-flattened fake-5D block layouts.
    """

    def hook(_module, _inputs, output: torch.Tensor) -> torch.Tensor:
        atten = anima._dave_atten[bidx]  # 0-d tensor
        sig = anima._dave_cur_sigma
        gate = (sig >= anima._dave_sigma_lo) & (sig <= anima._dave_sigma_hi)
        factor = atten * gate.to(atten.dtype)
        mu = output.float().mean(dim=(1, 2, 3), keepdim=True)  # (B,1,1,1,D)
        return (output.float() - factor.float() * mu).to(output.dtype)

    return hook


def setup_dave(
    args: argparse.Namespace,
    anima,
    device: torch.device,
) -> DAVEHooks:
    """Arm DAVE on ``anima``: load the mask, set the runtime buffers, hook blocks.

    Returns a :class:`DAVEHooks` handle the caller must ``.remove()`` after the
    generation (the model is shared across seeds — hooks would otherwise stack).
    """
    num_blocks = len(anima.blocks)
    weight = _load_weight(args.dave, num_blocks).copy()
    strength = float(getattr(args, "dave_strength", 0.1))
    sigma_lo = float(getattr(args, "dave_sigma_lo", 0.0))
    sigma_hi = float(getattr(args, "dave_sigma_hi", 1.0))

    # Temporal cutoff τ (the DAVE paper's lever): attenuate only the first τ
    # fraction of denoising steps. We reconstruct the *live* σ schedule and pick
    # the σ boundary between the last active and first inactive step, so the
    # window tracks --infer_steps/--flow_shift instead of a hand-guessed σ. This
    # overrides the σ window (the more specific intent wins).
    tau = float(getattr(args, "dave_tau", 0.0))
    if tau > 0.0:
        from library.inference.sampling import get_timesteps_sigmas

        n_steps = int(getattr(args, "infer_steps"))
        _, sched = get_timesteps_sigmas(
            n_steps,
            float(getattr(args, "flow_shift")),
            torch.device("cpu"),
            tail_power=float(getattr(args, "sigma_tail_power", 1.0)),
        )
        # k = number of leading steps to keep active (steps 0..k-1); σ decreases
        # monotonically, so the active window is σ ≥ boundary.
        k = max(1, min(n_steps, round(tau * n_steps)))
        if k >= n_steps:
            sigma_lo = 0.0  # τ covers the whole schedule → all σ
        else:
            sigma_lo = 0.5 * (float(sched[k - 1]) + float(sched[k]))
        sigma_hi = 1.0
        logger.info(
            f"DAVE τ={tau}: first {k}/{n_steps} steps active → σ∈[{sigma_lo:.3f}, 1.0]"
        )

    # Block-range cap: spare blocks outside [lo, hi] (e.g. the final content
    # blocks whose DC IS the image). hi=-1 means the last block.
    blk_lo = int(getattr(args, "dave_block_lo", 0))
    blk_hi = int(getattr(args, "dave_block_hi", -1))
    if blk_hi < 0:
        blk_hi = num_blocks - 1
    mask = np.zeros(num_blocks, dtype=bool)
    mask[max(0, blk_lo) : min(num_blocks, blk_hi + 1)] = True
    weight[~mask] = 0.0

    # atten = (1 − α) = strength · w(ℓ), clamped so α stays in [0, 1].
    atten = np.clip(strength * weight, 0.0, 1.0)

    buf = anima._dave_atten
    buf.copy_(torch.tensor(atten, device=buf.device, dtype=buf.dtype))
    anima._dave_sigma_lo.fill_(sigma_lo)
    anima._dave_sigma_hi.fill_(sigma_hi)
    anima._dave_cur_sigma.zero_()
    anima.enable_dave = True

    handles = [
        blk.register_forward_hook(_make_hook(anima, i))
        for i, blk in enumerate(anima.blocks)
    ]

    active = [(i, float(a)) for i, a in enumerate(atten) if a > 1e-3]
    logger.info(
        f"DAVE armed: strength={strength}, σ∈[{sigma_lo}, {sigma_hi}], "
        f"{len(active)}/{num_blocks} blocks active"
    )
    if active:
        peak = max(active, key=lambda t: t[1])
        logger.info(
            f"  peak block {peak[0]} atten={peak[1]:.3f} (α={1 - peak[1]:.3f}); "
            f"active blocks {active[0][0]}..{active[-1][0]}"
        )
    return DAVEHooks(handles)
