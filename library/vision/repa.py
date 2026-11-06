"""REPA alignment math: grid match, token pooling, Gram loss, DoG band-pass.

Shared by the Anima trainer (``library/training/repa.py``) and the MOSS trainer
(``library/moss/repa.py``); method docs in ``_archive/repa/repa.md``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def resolve_pe_grid(spec, n_pe: int, h_lat: int, w_lat: int) -> tuple[int, int]:
    """Resolve the encoder ``(gh, gw)`` patch grid for ``n_pe`` patch tokens.
    Token count alone is ambiguous (the bucket table is aspect-symmetric:
    46×23 and 23×46 both have 1058 patches), so disambiguate by latent
    orientation. Returns the unique bucket whose patch product == ``n_pe``.
    """
    cands = [(h, w) for (h, w) in spec.buckets if h * w == n_pe]
    if not cands:
        raise RuntimeError(
            f"REPA: no {spec.encoder} bucket has {n_pe} patches "
            f"(buckets={spec.buckets}). Cache encoder/grid mismatch?"
        )
    if len(cands) > 1:
        portrait = h_lat >= w_lat
        cands = [(h, w) for (h, w) in cands if (h >= w) == portrait] or cands
    return cands[0]


def pool_dit_tokens_to_grid(
    captured: torch.Tensor,
    latent_hw: tuple[int, int],
    patch: int,
    gh: int,
    gw: int,
    trim_tail: int = 0,
) -> torch.Tensor:
    """Captured block output → ``(B, gh*gw, D)`` fp32 tokens on the encoder grid.

    Layout-agnostic over the two block-output shapes (eager ``(B,1,H,W,D)`` /
    native-flatten ``(B,1,seq,1,D)``) — both flatten to row-major
    ``(B, N_dit, D)``, verified against the latent patch grid and
    adaptive-avg-pooled to ``(gh, gw)``.

    ``trim_tail`` drops that many trailing register tokens before the grid
    check — they're non-decoded scratchpad appended at the END of the
    self-attn sequence and must not enter the alignment loss.
    """
    b = captured.shape[0]
    d_dit = captured.shape[-1]
    tokens = captured.reshape(b, -1, d_dit)
    if trim_tail > 0:
        tokens = tokens[:, :-trim_tail]

    h_lat, w_lat = latent_hw
    h_dit, w_dit = h_lat // patch, w_lat // patch
    if tokens.shape[1] != h_dit * w_dit:
        raise RuntimeError(
            f"REPA: captured {tokens.shape[1]} DiT tokens but latent grid is "
            f"{h_dit}x{w_dit}={h_dit * w_dit} (patch={patch}). "
            "Block/patch-grid mismatch."
        )

    dit_grid = tokens.reshape(b, h_dit, w_dit, d_dit).permute(0, 3, 1, 2)
    dit_pooled = F.adaptive_avg_pool2d(dit_grid.float(), (gh, gw))
    return dit_pooled.flatten(2).transpose(1, 2)  # (B, gh*gw, D) fp32


def relational_gram_loss(
    dit_tok: torch.Tensor,
    pe: torch.Tensor,
    *,
    spatial_norm: bool = False,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Arm-B relational loss: ``MSE(Gram(dit_tok), Gram(pe))`` in fp32.

    Per-token L2-norm within each space, then match the N×N affinity
    structure. ``spatial_norm`` applies iREPA target standardization:
    standardize ``pe`` across the token axis before per-token L2-norm,
    removing the shared global component that compresses pairwise cosines.

    ``sample_weights`` (``[B]`` or None): per-sample multipliers applied
    before the batch mean (timestep reweighting); None is bit-exact to the
    unweighted ``F.mse_loss`` reduction.
    """
    if spatial_norm:
        pe = (pe - pe.mean(dim=1, keepdim=True)) / (pe.std(dim=1, keepdim=True) + 1e-6)
    dit_hat = F.normalize(dit_tok, dim=-1)
    pe_hat = F.normalize(pe, dim=-1)
    g_dit = torch.bmm(dit_hat, dit_hat.transpose(1, 2))  # (B, N, N)
    g_pe = torch.bmm(pe_hat, pe_hat.transpose(1, 2))
    if sample_weights is None:
        return F.mse_loss(g_dit, g_pe)
    per_sample = (g_dit - g_pe).pow(2).mean(dim=(1, 2))  # [B]
    return (per_sample * sample_weights.to(per_sample.dtype)).mean()


def _safe_blur(grid: torch.Tensor, sigma: float, gh: int, gw: int) -> torch.Tensor:
    """``gaussian_blur_2d`` guarded against a kernel wider than the grid.

    ``gaussian_blur_2d`` reflect-pads by ``ceil(3σ)``, which needs
    ``pad ≤ min_dim − 1``. Clamp σ defensively so an aggressive small divisor
    degrades to the widest valid blur instead of crashing.
    """
    from library.runtime.fei import gaussian_blur_2d

    if sigma <= 0:
        return grid
    max_pad = min(gh, gw) - 1
    if math.ceil(3.0 * sigma) > max_pad:
        sigma = max(1e-3, (max_pad - 0.01) / 3.0)
    return gaussian_blur_2d(grid, sigma)


def dog_standardize(
    pe: torch.Tensor,
    gh: int,
    gw: int,
    sigma1_div: float,
    sigma2_div: float = 0.0,
    norm_std: float = 0.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Difference-of-Gaussians band-pass of the target tokens (REPA-DoG).

    Generalizes ``spatial_norm``'s DC removal to a broader low-band strip
    (arXiv:2603.14645v1 §3.5; best ``σ₁ = min/16`` in bench). ``pe``
    is ``(B, N, d)`` with ``N == gh*gw`` (CLS dropped); reshaped to
    ``(gh, gw)`` row-major, band-passed, standardized, flattened back — slots
    in **instead of** ``relational_gram_loss``'s ``spatial_norm`` block.

    Filter ``H(Z)`` on ``Z (B, d, gh, gw)``:

    * ``sigma2_div <= 0`` → ``Z − LP(Z, σ₁)``          high-pass
    * ``sigma2_div > 0``  → ``LP(Z, σ₂) − LP(Z, σ₁)``  band-pass

    with ``σ₁ = min(gh,gw)/sigma1_div``, ``σ₂ = min(gh,gw)/sigma2_div``. At
    ``σ₁→0`` this reduces to the shipped DC removal (``spatial_norm`` is its
    degenerate special case).

    ``norm_std``: ``0`` (default) divides by the empirical per-channel
    spatial std (identical to ``spatial_norm``'s, so an A/B isolates the
    band-pass); ``> 0`` divides by that fixed constant instead (paper's
    ``normalization_std`` regime).
    """
    b, n, d = pe.shape
    grid = pe.transpose(1, 2).reshape(b, d, gh, gw)  # row-major (B, d, gh, gw)
    s1 = float(min(gh, gw)) / float(sigma1_div)
    if sigma2_div > 0:
        s2 = float(min(gh, gw)) / float(sigma2_div)
        h = _safe_blur(grid, s2, gh, gw) - _safe_blur(grid, s1, gh, gw)
    else:
        h = grid - _safe_blur(grid, s1, gh, gw)
    denom = (
        (h.std(dim=(2, 3), keepdim=True) + eps) if norm_std <= 0 else float(norm_std)
    )
    h = h / denom
    return h.reshape(b, d, n).transpose(1, 2)  # back to (B, N, d)


def relational_align_loss(
    captured: torch.Tensor,
    pe: torch.Tensor,
    latent_hw: tuple[int, int],
    patch: int,
    spec,
    *,
    spatial_norm: bool = False,
    dog: bool = False,
    dog_sigma1_div: float = 16.0,
    dog_sigma2_div: float = 0.0,
    dog_norm_std: float = 0.0,
) -> torch.Tensor:
    """One-shot relational alignment: CLS-drop → grid-match → pool → Gram MSE.

    ``captured`` is a raw block output (either layout), ``pe`` the cached
    encoder features ``(B, T, d_enc)`` with CLS still at index 0 when the spec
    carries one. This is the probe entry point (``bench/turbo/``); the
    training adapter composes the same pieces itself (it also needs the
    pooled tokens for the absolute-arm head / grad-heatmap probe).

    ``dog`` applies the REPA-DoG band-pass instead of ``spatial_norm``
    (mutually exclusive; ``dog`` wins).
    """
    n_pe = pe.shape[1] - (1 if spec.use_cls else 0)
    gh, gw = resolve_pe_grid(spec, n_pe, latent_hw[0], latent_hw[1])
    if spec.use_cls:
        pe = pe[:, 1:, :]
    dit_tok = pool_dit_tokens_to_grid(captured, latent_hw, patch, gh, gw)
    if dog:
        pe = dog_standardize(pe, gh, gw, dog_sigma1_div, dog_sigma2_div, dog_norm_std)
        return relational_gram_loss(dit_tok, pe, spatial_norm=False)
    return relational_gram_loss(dit_tok, pe, spatial_norm=spatial_norm)
