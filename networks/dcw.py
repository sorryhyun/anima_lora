"""DCW: post-step correction for SNR-t bias on flow-matching DiTs.

Anima form (pixel mode, opposite sign from paper, λ ≈ -0.015 for LL-only):
    prev += λ · sched(σ_i) · (prev - x0_pred)

where sched ∈ {one_minus_sigma, sigma_i, const, none}. Default
schedule one_minus_sigma matches Anima's bias envelope (concentrates
correction at low σ where |gap| is largest). See docs/methods/dcw.md.

The correction can be restricted to a subset of single-level Haar
subbands ({LL, LH, HL, HH}) via the ``bands`` argument. LL-only is the
shipped default — the per-band sweep at
``bench/dcw/results/20260503-2102-band-mask-eyeball/`` showed the
broadband correction degrades detail bands while LL-only improves all
four bands. ``bands == ALL_BANDS`` falls through to the cheap
no-DWT path (orthonormal Haar reconstructs exactly to float roundoff).

Paper: Yu et al., "Elucidating the SNR-t Bias of Diffusion Probabilistic
Models" (CVPR 2026, arXiv:2604.16044).
"""

from typing import Literal

import torch
import torch.nn as nn

Schedule = Literal["one_minus_sigma", "sigma_i", "const", "none"]

BANDS = ("LL", "LH", "HL", "HH")
ALL_BANDS = frozenset(BANDS)


def parse_band_mask(label: str) -> frozenset[str]:
    """CLI string → frozenset of band names. ``all`` → all four bands.

    Format: ``LL``, ``HH``, ``LH+HL+HH``, ``all``. Case-insensitive on
    the band names; ``all`` must be exactly that token.
    """
    if label == "all":
        return ALL_BANDS
    parts = [p.upper() for p in label.split("+") if p]
    bad = [p for p in parts if p not in BANDS]
    if bad or not parts:
        raise ValueError(
            f"unknown band(s) in mask {label!r}: {bad or '<empty>'}; "
            f"valid bands {BANDS!r} or 'all'"
        )
    return frozenset(parts)


def _haar_dwt_2d(
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-level 2D orthonormal Haar DWT on the last two dims.

    Returns (LL, LH, HL, HH), each (..., H/2, W/2).
    """
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


def _haar_idwt_2d(
    LL: torch.Tensor, LH: torch.Tensor, HL: torch.Tensor, HH: torch.Tensor
) -> torch.Tensor:
    s = 0.5
    a = (LL + LH + HL + HH) * s
    b = (LL + LH - HL - HH) * s
    c = (LL - LH + HL - HH) * s
    d = (LL - LH - HL + HH) * s
    out = torch.empty(
        *LL.shape[:-2], LL.shape[-2] * 2, LL.shape[-1] * 2,
        dtype=LL.dtype, device=LL.device,
    )
    out[..., 0::2, 0::2] = a
    out[..., 0::2, 1::2] = b
    out[..., 1::2, 0::2] = c
    out[..., 1::2, 1::2] = d
    return out


def _sched(sigma_i: float, schedule: Schedule) -> float:
    if schedule == "one_minus_sigma":
        return 1.0 - sigma_i
    if schedule == "sigma_i":
        return sigma_i
    if schedule == "const":
        return 1.0
    return 0.0  # "none" — for ablation


def apply_dcw(
    prev_sample: torch.Tensor,
    x0_pred: torch.Tensor,
    sigma_i: float,
    *,
    lam: float = -0.015,
    schedule: Schedule = "one_minus_sigma",
    bands: frozenset[str] = ALL_BANDS,
) -> torch.Tensor:
    """Apply pixel-mode DCW correction to prev_sample, optionally restricted
    to a subset of Haar subbands.

    Returns prev_sample unchanged if lam == 0 or schedule == "none".
    Callers should also skip the final step (sigma_{i+1} == 0) where
    prev_sample == x0_pred exactly and the (1-σ_i) weight is near 1.

    When ``bands == ALL_BANDS`` (default), the full pixel-mode update
    runs without any DWT round-trip — bit-identical to the pre-band-sweep
    behaviour. Otherwise the differential ``(prev - x0_pred)`` is
    decomposed via single-level Haar DWT, non-mask bands are zeroed,
    reconstructed, and added. Latent space is 16-ch Qwen-VAE; the
    "low/high freq" labels are *latent*, not pixel-space.
    """
    s = lam * _sched(sigma_i, schedule)
    if s == 0.0:
        return prev_sample
    if bands == ALL_BANDS:
        return prev_sample + s * (prev_sample - x0_pred)
    diff = (prev_sample - x0_pred).float()
    LL, LH, HL, HH = _haar_dwt_2d(diff)
    z = torch.zeros_like(LL)
    LL_m = LL if "LL" in bands else z
    LH_m = LH if "LH" in bands else z
    HL_m = HL if "HL" in bands else z
    HH_m = HH if "HH" in bands else z
    masked = _haar_idwt_2d(LL_m, LH_m, HL_m, HH_m).to(prev_sample.dtype)
    return prev_sample + s * masked


def haar_LL_norm(v: torch.Tensor) -> float:
    """Single-level Haar LL-band Frobenius norm of a velocity tensor.

    Mirrors ``haar_band_norms_batched`` in ``scripts/dcw_measure_bias.py`` —
    used at inference to compute v_rev_LL[i] for the v4 fusion head's g_obs
    channel. Caller passes velocity in any layout where the last two dims
    are spatial (e.g. (B, C, T, H, W) or (B, C, H, W)); the DWT is taken on
    those last two dims and we take the global Frobenius norm of LL.
    """
    LL, _, _, _ = _haar_dwt_2d(v.float())
    return float(LL.flatten().norm())


class FusionHead(nn.Module):
    """v4 fusion head: (c_pool, aspect, g_obs[0:k], aux) → (α̂, log σ̂²).

    Shared by the trainer (``scripts/dcw_train_fusion_head.py``) and the
    inference controller (``library/inference/dcw_v4.py``) so the MLP
    architecture is in one place. The trainer's saved state-dict keys are
    prefixed with ``head.`` (so e.g. ``head.aspect_emb.weight`` /
    ``head.mlp.X.weight`` in the artifact); inference module strips the
    prefix when loading.
    """

    def __init__(
        self,
        c_pool_dim: int = 1024,
        n_aspects: int = 3,
        aspect_emb_dim: int = 16,
        k: int = 7,
        aux_dim: int = 3,
        c_proj_dim: int = 0,
        hidden: tuple[int, ...] = (256, 128),
        sigma_hidden: int = 64,
        dropout: float = 0.2,
        log_sigma2_init: float = 0.0,
    ):
        super().__init__()
        self.k = k
        self.c_pool_dim = c_pool_dim
        # c_proj_dim == 0 → identity passthrough (raw c_pool into concat).
        # > 0 → LN + Linear(c_pool_dim → c_proj_dim) before concat. The
        # 2026-05-05 sweep showed projection balances slot capacity but
        # hurts CV metrics on Anima — supervision-side variance, not arch
        # capacity, is the bottleneck. Kept as an ablation knob.
        self.c_proj_dim = c_proj_dim
        if c_proj_dim > 0:
            self.c_proj = nn.Sequential(
                nn.LayerNorm(c_pool_dim),
                nn.Linear(c_pool_dim, c_proj_dim),
            )
            cat_dim = c_proj_dim
        else:
            self.c_proj = nn.Identity()
            cat_dim = c_pool_dim
        self.aspect_emb = nn.Embedding(n_aspects, aspect_emb_dim)
        nn.init.normal_(self.aspect_emb.weight, std=0.02)
        in_dim = cat_dim + aspect_emb_dim + k + aux_dim

        # α̂ and log σ̂² use independent trunks. Sharing a single trunk turned
        # the per-prompt seed-variance aux loss into a destructive interference
        # term (any λ>0 collapsed α̂'s correlation), because aux gradients
        # rewrote shared features the NLL needed for point prediction.
        alpha_layers: list[nn.Module] = [nn.LayerNorm(in_dim)]
        prev = in_dim
        for h in hidden:
            alpha_layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        alpha_out = nn.Linear(prev, 1)
        nn.init.zeros_(alpha_out.weight)
        nn.init.zeros_(alpha_out.bias)
        alpha_layers.append(alpha_out)
        self.alpha_mlp = nn.Sequential(*alpha_layers)

        # Smaller σ̂² trunk: aux supervision is per-prompt aggregate (only
        # ~175 effective points), so capacity must be modest.
        sigma_out = nn.Linear(sigma_hidden, 1)
        nn.init.zeros_(sigma_out.weight)
        nn.init.zeros_(sigma_out.bias)
        with torch.no_grad():
            sigma_out.bias.fill_(log_sigma2_init)
        self.sigma_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, sigma_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            sigma_out,
        )

    def forward(
        self,
        c_pool: torch.Tensor,
        aspect_id: torch.Tensor,
        g_obs: torch.Tensor,
        aux: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.aspect_emb(aspect_id)
        c = self.c_proj(c_pool)
        x = torch.cat([c, a, g_obs, aux], dim=-1)
        return self.alpha_mlp(x).squeeze(-1), self.sigma_mlp(x).squeeze(-1)
