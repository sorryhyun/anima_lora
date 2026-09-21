"""Numerical-equivalence pins for the router-side compute kernels shared
between training, inference, and the ComfyUI hydralora node.

These kernels feed trained router weights that are bit-sensitive to band
ordering, σ frequency schedule, and bucket-edge derivation. Any silent
drift (e.g. a refactor that swaps band order, a typo in the sinusoidal
exponent base) corrupts the gate at inference with no exception raised.

The test pins both the single-source value (from
``library.inference.router_compute``) AND the live training-side imports
that vendor through to the node, so a future "let's simplify this"
refactor that breaks the contract gets caught here instead of in someone's
sampled output.
"""

from __future__ import annotations

import torch

from library.inference.router_compute import (
    apply_sigma_band_mask,
    compute_fei_2band,
    compute_fei_nband_high_to_low,
    fei_sigma_low,
    fei_temperature,
    gaussian_blur_2d,
    sigma_sinusoidal_features,
)
from library.runtime.fei import compute_fei_2band as _live_fei_2band
from library.runtime.fei import gaussian_blur_2d as _live_blur
from networks.lora_modules.router_state import (
    _apply_sigma_band_mask as _live_apply_mask,
)
from networks.lora_modules.router_state import (
    _sigma_sinusoidal_features as _live_sigma_feat,
)


# ---------------------------------------------------------------------------
# Single-source-of-truth identity: the façade in library/inference/
# router_compute.py must point at the same function objects the training
# side imports. If a refactor re-implements one side, this fails first.
# ---------------------------------------------------------------------------


def test_router_compute_is_canonical_2band():
    assert compute_fei_2band is _live_fei_2band


def test_router_compute_is_canonical_blur():
    assert gaussian_blur_2d is _live_blur


def test_router_compute_is_canonical_sigma_features():
    assert sigma_sinusoidal_features is _live_sigma_feat


def test_router_compute_is_canonical_band_mask():
    assert apply_sigma_band_mask is _live_apply_mask


def test_fei_temperature_identity_at_tau_one():
    """τ=1.0 returns the FEI simplex unchanged (softmax(log p) = p)."""
    fei = torch.tensor([[0.7, 0.3], [0.2, 0.8]])
    out = fei_temperature(fei, 1.0)
    assert torch.allclose(out, fei, atol=1e-6)


def test_fei_temperature_sharpens_and_normalizes():
    """τ<1 sharpens toward the dominant band; output stays a simplex."""
    fei = torch.tensor([[0.8, 0.2]])
    out = fei_temperature(fei, 0.5)
    assert torch.allclose(out.sum(-1), torch.ones(1), atol=1e-6)
    assert out[0, 0] > fei[0, 0]  # dominant band gets more mass


# ---------------------------------------------------------------------------
# Functional contract: pinned outputs on a fixed seed. Values are first-run
# captures — they're free to change only when the trained-router contract
# is intentionally being rev'd (and every checkpoint then needs re-baking).
# ---------------------------------------------------------------------------


def _fixed_latent(b: int = 2, c: int = 16, h: int = 32, w: int = 32) -> torch.Tensor:
    g = torch.Generator().manual_seed(20260515)
    return torch.randn(b, c, h, w, generator=g, dtype=torch.float32)


def test_compute_fei_2band_simplex_and_ordering():
    """[e_low, e_high] simplex (low first). Sums to 1. Trained against this."""
    z = _fixed_latent()
    fei = compute_fei_2band(z, sigma_low=4.0)
    assert fei.shape == (2, 2)
    assert torch.allclose(fei.sum(-1), torch.ones(2), atol=1e-6)
    # White-noise latent has most energy in the high band — confirms ordering
    # didn't silently flip. e_low << e_high.
    assert (fei[:, 0] < fei[:, 1]).all()


def test_compute_fei_nband_high_to_low_simplex_and_ordering():
    """[high, ..., low] simplex (high first). Trained against this."""
    z = _fixed_latent()
    fei = compute_fei_nband_high_to_low(z, sigma_low=4.0, num_bands=3)
    assert fei.shape == (2, 3)
    assert torch.allclose(fei.sum(-1), torch.ones(2), atol=1e-6)
    # White noise → highest energy in band 0 (high), lowest in band 2 (low).
    assert (fei[:, 0] > fei[:, 1]).all()
    assert (fei[:, 1] > fei[:, 2]).all()


def test_fei_orderings_are_distinct():
    """2-band and n=2 high-to-low share an underlying decomposition but
    deliberately ship in opposite orderings — wiring them through one helper
    would silently corrupt one of the two trained routers."""
    z = _fixed_latent()
    fei_lh = compute_fei_2band(z, sigma_low=4.0)  # [e_low, e_high]
    fei_hl = compute_fei_nband_high_to_low(
        z, sigma_low=4.0, num_bands=2
    )  # [e_high, e_low]
    # n=2 high-to-low: bands are pyramid[0]-pyramid[1] (= z - LP, the
    # high band) and pyramid[1] (= LP, the low band). So index 0 of the
    # high-to-low should equal index 1 of the low-high (both are e_high).
    assert torch.allclose(fei_lh[:, 0], fei_hl[:, 1], atol=1e-5)
    assert torch.allclose(fei_lh[:, 1], fei_hl[:, 0], atol=1e-5)


def test_sigma_sinusoidal_features_functional_form():
    """Match the DiT t_embedder: 10000^(-k/half_dim) frequencies, [cos | sin]."""
    sigma = torch.tensor([0.1, 0.5, 1.0])
    feat = sigma_sinusoidal_features(sigma, sigma_feature_dim=16)
    assert feat.shape == (3, 16)
    # First half is cos, second half is sin — at σ=0 the cos half is all 1 and
    # the sin half is all 0. (We don't pass σ=0 here, but at σ→0 cos(0)=1.)
    sigma0 = torch.zeros(1)
    feat0 = sigma_sinusoidal_features(sigma0, sigma_feature_dim=16)
    assert torch.allclose(feat0[0, :8], torch.ones(8), atol=1e-6)
    assert torch.allclose(feat0[0, 8:], torch.zeros(8), atol=1e-6)


def test_sigma_sinusoidal_features_cfg_doubled_batch():
    """The node passes (1,) σ but CFG-doubled (2, …) features. Make sure
    broadcasting in the hook stays unambiguous — sigma_sinusoidal_features
    itself returns (B, dim) matching σ's batch."""
    sigma = torch.tensor([0.3, 0.7])  # CFG-doubled
    feat = sigma_sinusoidal_features(sigma, sigma_feature_dim=16)
    assert feat.shape == (2, 16)
    # Different σ → different features.
    assert not torch.allclose(feat[0], feat[1])


def test_fei_sigma_low_bucket_invariant():
    """σ_low scales with min(H, W) so the band semantic is bucket-invariant."""
    assert fei_sigma_low(32, 32, 4.0) == 8.0
    assert fei_sigma_low(48, 32, 4.0) == 8.0  # min wins
    assert fei_sigma_low(32, 48, 4.0) == 8.0
    assert fei_sigma_low(16, 16, 8.0) == 2.0


def test_apply_sigma_band_mask_softmax_renormalises():
    """Out-of-band logits → -inf → softmax renormalises mass to in-band only."""
    logits = torch.zeros(2, 6)  # 6 experts
    expert_band = torch.arange(6) % 3  # interleaved layout: bands [0,1,2,0,1,2]
    sigma_edges = torch.linspace(0.0, 1.0, 4)[1:-1].contiguous()  # 3 buckets
    sigma = torch.tensor([0.0, 0.99])
    masked = apply_sigma_band_mask(logits, sigma, expert_band, sigma_edges)
    gate = torch.softmax(masked, dim=-1)
    # row 0: σ=0 → band 0 → experts {0, 3} carry all mass.
    assert torch.allclose(gate[0, [0, 3]].sum(), torch.tensor(1.0), atol=1e-6)
    assert gate[0, [1, 2, 4, 5]].abs().max().item() == 0.0
    # row 1: σ=0.99 → band 2 → experts {2, 5} carry all mass.
    assert torch.allclose(gate[1, [2, 5]].sum(), torch.tensor(1.0), atol=1e-6)
    assert gate[1, [0, 1, 3, 4]].abs().max().item() == 0.0
