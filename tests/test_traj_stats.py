"""Invariant tests for the --traj_stats passive recorder (Phase 0).

Pins the proposal's Tier-1.5 contract (_archive/proposals/traj_latent_stats.md):
pure observation (inputs bit-identical after record), explicit dim-2 squeeze
discipline, and the Qwen VAE normalization constants staying in lockstep with
the model defaults.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from library.inference.traj_stats import (
    QWEN_LATENTS_MEAN,
    QWEN_LATENTS_STD,
    TrajStatsRecorder,
)


def _mk_recorder(tmp_path, **kw) -> TrajStatsRecorder:
    return TrajStatsRecorder(str(tmp_path), **kw)


def test_constants_match_qwen_vae_defaults():
    from library.models.qwen_vae import AutoencoderKLQwenImage

    sig = inspect.signature(AutoencoderKLQwenImage.__init__)
    assert sig.parameters["latents_mean"].default == QWEN_LATENTS_MEAN
    assert sig.parameters["latents_std"].default == QWEN_LATENTS_STD


def test_record_is_pure_observation(tmp_path):
    torch.manual_seed(0)
    rec = _mk_recorder(tmp_path)
    latents = torch.randn(2, 16, 1, 8, 8, dtype=torch.bfloat16)
    noise = torch.randn(2, 16, 1, 8, 8, dtype=torch.bfloat16)
    uncond = torch.randn(2, 16, 1, 8, 8, dtype=torch.bfloat16)
    refs = [t.clone() for t in (latents, noise, uncond)]
    for i, sigma in enumerate([1.0, 0.6, 0.2]):
        rec.record(i, sigma, latents, noise, uncond)
    for t, ref in zip((latents, noise, uncond), refs):
        assert torch.equal(t, ref)


def test_dim2_squeeze_discipline_b1(tmp_path):
    # B=1 is the trap a bare squeeze() would corrupt (batch dim dropped).
    rec = _mk_recorder(tmp_path)
    x = torch.randn(1, 16, 1, 8, 8)
    rec.record(0, 1.0, x, torch.zeros_like(x))
    step = rec._steps[0]
    assert step["activity"].shape == (1, 4, 4)
    assert step["codes"].shape == (1, 16, 4, 4)
    assert step["cbits"].shape == (16,)
    # a wrong T dim must be rejected, not silently squeezed
    with pytest.raises(AssertionError):
        rec.record(1, 0.5, torch.randn(1, 16, 2, 8, 8), torch.zeros(1, 16, 2, 8, 8))


def test_invert_recorder_is_pure_observation(tmp_path):
    # Phase 1 real-image arm: the recorder threaded through DirectEdit
    # inversion must leave z_inv / delta_z bit-identical.
    from library.inference.editing.directedit import invert

    def fake_anima(latents, t_expand, embed, padding_mask=None):
        # deterministic, latent- and σ-dependent pseudo-velocity
        return (latents * 0.25 + t_expand.view(-1, 1, 1, 1, 1).to(latents.dtype)).to(
            latents.dtype
        )

    torch.manual_seed(3)
    z_clean = torch.randn(1, 16, 1, 8, 8, dtype=torch.bfloat16)
    embed = torch.randn(1, 4, 8, dtype=torch.bfloat16)
    sigmas = torch.linspace(1.0, 0.0, 5)

    z_ref, d_ref = invert(fake_anima, z_clean.clone(), embed, None, sigmas)

    rec = _mk_recorder(tmp_path)
    z_rec, d_rec = invert(
        fake_anima, z_clean.clone(), embed, None, sigmas, traj_recorder=rec
    )
    assert all(torch.equal(a, b) for a, b in zip(z_ref, z_rec))
    assert all(torch.equal(a, b) for a, b in zip(d_ref, d_rec))

    # one record per step, natural inversion order: σ ascending (clean → noise)
    path = rec.flush()
    d = np.load(path)
    assert d["codes"].shape[0] == sigmas.shape[0] - 1
    rec_sigmas = d["sigmas"]
    assert all(a < b for a, b in zip(rec_sigmas, rec_sigmas[1:]))
    assert rec_sigmas[-1] == pytest.approx(1.0)
