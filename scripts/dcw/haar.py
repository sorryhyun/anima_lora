"""Single-level 2D orthonormal Haar DWT on the latent (H, W) plane.

Σ_b ||v_b||² = ||v||² (Parseval); iDWT(DWT(x)) == x to float roundoff.
"""

from __future__ import annotations

import torch

BANDS = ("LL", "LH", "HL", "HH")


@torch.no_grad()
def haar_dwt_2d(
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a = v[..., 0::2, 0::2]
    b = v[..., 0::2, 1::2]
    c = v[..., 1::2, 0::2]
    d = v[..., 1::2, 1::2]
    s = 0.5
    LL = (a + b + c + d) * s
    LH = (a + b - c - d) * s
    HL = (a - b + c - d) * s
    HH = (a - b - c + d) * s
    return LL, LH, HL, HH


@torch.no_grad()
def haar_idwt_2d(
    LL: torch.Tensor, LH: torch.Tensor, HL: torch.Tensor, HH: torch.Tensor
) -> torch.Tensor:
    s = 0.5
    a = (LL + LH + HL + HH) * s
    b = (LL + LH - HL - HH) * s
    c = (LL - LH + HL - HH) * s
    d = (LL - LH - HL + HH) * s
    out = torch.empty(
        *LL.shape[:-2],
        LL.shape[-2] * 2,
        LL.shape[-1] * 2,
        dtype=LL.dtype,
        device=LL.device,
    )
    out[..., 0::2, 0::2] = a
    out[..., 0::2, 1::2] = b
    out[..., 1::2, 0::2] = c
    out[..., 1::2, 1::2] = d
    return out


@torch.no_grad()
def haar_band_norms_batched(v: torch.Tensor) -> torch.Tensor:
    """Per-batch Haar-subband L2 norms, on-device.

    v: (B, ...) — DiT velocity. Returns (B, 4) float32 with columns
    ordered as ``BANDS`` = (LL, LH, HL, HH). Stays on GPU so the caller
    can stack ``n_steps`` rows into a single accumulator and sync once
    at trajectory end (avoids 4 + 1 ``.item()`` syncs per step).
    """
    LL, LH, HL, HH = haar_dwt_2d(v.float())
    return torch.stack(
        [
            LL.flatten(start_dim=1).norm(dim=1),
            LH.flatten(start_dim=1).norm(dim=1),
            HL.flatten(start_dim=1).norm(dim=1),
            HH.flatten(start_dim=1).norm(dim=1),
        ],
        dim=1,
    )


def apply_dcw_LL_only_batched(
    prev: torch.Tensor, x0_pred: torch.Tensor, scalars: torch.Tensor
) -> torch.Tensor:
    """LL-only pixel-mode DCW with per-row scalar.

    prev / x0_pred: (B, C, T, H, W) float. scalars: (B,) float — already
    multiplied by the schedule (e.g. ``λ · (1 − σ_i)``). Rows with
    scalar=0 produce a zero correction and are bit-identical to the
    unbatched ``λ=0`` early-out (no early-out kept here so the call is
    graph-stable for ``torch.compile``).
    """
    diff = prev - x0_pred
    LL, LH, HL, HH = haar_dwt_2d(diff)
    z = torch.zeros_like(LL)
    masked = haar_idwt_2d(LL, z, z, z)
    sc = scalars.view(-1, *([1] * (prev.dim() - 1)))
    return prev + sc * masked
