"""Invariants for the gradient-SVD ``lora_down`` init and the min_snr weighting.

Tier 1.5 companion to ``bench/grad_init/`` (the measurement) and
``docs/proposal/grad_basis_init.md`` (E0/E1). What must hold:

* a gradient-seeded ``A`` is ``V_rᵀ / sqrt(3)`` — the SAME row-norm match
  ``weight_svd`` uses, so an init arm is never also a step-size arm;
* ΔW is still 0 at step 0 (it is ordinary LoRA, not a warm start);
* a basis that does not cover a module leaves Kaiming rather than zeros;
* a depth-baked basis refuses a DiT of another depth;
* ``min_snr`` has mean weight 1 under the run's σ density (so it is a reshape,
  not a learning-rate change) and tilts away from high σ.
"""

import math

import pytest
import torch

from library.anima.training import (
    compute_loss_weighting_for_anima,
    min_snr_normalizer,
    min_snr_weighting,
)
from networks.grad_basis import (
    basis_from_sketches,
    count_blocks,
    load_basis,
    save_basis,
    top_right_basis,
)
from networks.lora_modules.lora import LoRAModule

IN, OUT, RANK = 16, 8, 4


def _orthonormal(n_in: int, r: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(n_in, r, generator=gen)).Q


def _module(**kw) -> LoRAModule:
    org = torch.nn.Linear(IN, OUT, bias=False)
    return LoRAModule("lora_unet_blocks_0_self_attn_qkv_proj", org, 1.0, RANK, 1, **kw)


# --------------------------------------------------------------------------- #
# init
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["grad_svd", "basis_file"])
def test_seeded_down_is_basis_over_sqrt3(mode):
    V = _orthonormal(IN, RANK)
    mod = _module(down_init=mode, grad_basis=V)
    assert torch.allclose(mod.lora_down.weight.data, V.T / math.sqrt(3), atol=1e-6)
    # orthogonal rows, each at Kaiming's expected row-norm (E‖·‖² ≈ 1/3)
    gram = mod.lora_down.weight.data @ mod.lora_down.weight.data.T
    assert torch.allclose(gram, torch.eye(RANK) / 3.0, atol=1e-5)


def test_delta_w_is_zero_at_step_zero():
    mod = _module(down_init="basis_file", grad_basis=_orthonormal(IN, RANK))
    assert torch.count_nonzero(mod.lora_up.weight.data) == 0
    assert torch.count_nonzero(mod.get_weight()) == 0


def test_short_basis_seeds_leading_rows_and_keeps_kaiming_tail():
    """``r_store < network_dim``: seed what the basis has, leave the rest Kaiming.

    Zero-filling the tail would hand those directions no gradient at step 0 and
    quietly shrink the effective rank.
    """
    V = _orthonormal(IN, 2)
    ref = _module(down_init="kaiming").lora_down.weight.data.clone()
    torch.manual_seed(0)
    mod = _module(down_init="basis_file", grad_basis=V)
    A = mod.lora_down.weight.data
    assert torch.allclose(A[:2], V.T / math.sqrt(3), atol=1e-6)
    assert torch.count_nonzero(A[2:]) == A[2:].numel()
    assert not torch.allclose(A[2:], ref[2:])  # still random, not a copy of the seed


def test_missing_basis_entry_keeps_kaiming():
    mod = _module(down_init="basis_file", grad_basis=None)
    A = mod.lora_down.weight.data
    assert torch.count_nonzero(A) == A.numel()
    gram = A @ A.T
    assert not torch.allclose(gram, torch.eye(RANK) / 3.0, atol=1e-3)
    assert mod._grad_basis_seeded == 0


def test_basis_in_features_mismatch_raises():
    with pytest.raises(ValueError, match="in_features"):
        _module(down_init="basis_file", grad_basis=_orthonormal(IN + 1, RANK))


def test_unknown_down_init_lists_every_mode():
    with pytest.raises(ValueError, match="grad_svd"):
        _module(down_init="nope")


# --------------------------------------------------------------------------- #
# subspace math
# --------------------------------------------------------------------------- #
def test_top_right_basis_recovers_a_planted_row_space():
    gen = torch.Generator().manual_seed(1)
    V_true = _orthonormal(IN, RANK, seed=2)
    # S = (q × in) with all its energy inside span(V_true)
    S = torch.randn(RANK + 8, RANK, generator=gen) @ V_true.T
    V = top_right_basis(S, RANK)
    captured = (S @ V).pow(2).sum() / S.pow(2).sum()
    assert captured > 0.999


def test_basis_from_sketches_skips_empty_layers():
    sketches = {
        "a": torch.randn(8, IN, generator=torch.Generator().manual_seed(3)),
        "b": torch.zeros(8, IN),
    }
    basis = basis_from_sketches(sketches, RANK)
    assert set(basis) == {"a"}
    assert basis["a"].shape == (IN, RANK)


# --------------------------------------------------------------------------- #
# artifact
# --------------------------------------------------------------------------- #
def test_count_blocks_reads_depth_from_keys():
    assert count_blocks(["lora_unet_blocks_0_mlp_0", "lora_unet_blocks_27_mlp_0"]) == 28
    assert count_blocks(["lora_unet_final_layer_proj"]) == 0


def test_roundtrip_and_depth_mismatch_refused(tmp_path):
    basis = {"lora_unet_blocks_0_self_attn_qkv_proj": _orthonormal(IN, RANK)}
    path = save_basis(tmp_path / "b.safetensors", basis, num_blocks=28)

    loaded, meta = load_basis(path, num_blocks=28)
    assert meta["ss_num_blocks"] == "28" and meta["rank"] == str(RANK)
    assert torch.allclose(
        loaded["lora_unet_blocks_0_self_attn_qkv_proj"],
        basis["lora_unet_blocks_0_self_attn_qkv_proj"],
        atol=1e-3,  # stored fp16
    )

    # A LoRA's module names carry the block index, so a 28-block basis on a
    # 40-block DiT would silently leave the tail blocks unseeded.
    with pytest.raises(ValueError, match="depth-baked"):
        load_basis(path, num_blocks=40)


def test_load_missing_file_points_at_the_builders(tmp_path):
    with pytest.raises(FileNotFoundError, match="grad_svd"):
        load_basis(tmp_path / "nope.safetensors")


# --------------------------------------------------------------------------- #
# weight_svd slices
# --------------------------------------------------------------------------- #
def _right_basis(module: LoRAModule) -> torch.Tensor:
    return module.lora_down.weight.data.T * (3**0.5)  # (in, r), orthonormal cols


def _overlap(qa: torch.Tensor, qb: torch.Tensor) -> float:
    return (torch.linalg.matrix_norm(qa.T @ qb) ** 2 / qa.shape[1]).item()


def test_svd_slice_zero_is_the_top_r_window():
    torch.manual_seed(0)
    org = torch.nn.Linear(IN, OUT, bias=False)
    a = LoRAModule("lora_unet_blocks_0_x", org, 1.0, RANK, 1, down_init="weight_svd")
    b = LoRAModule(
        "lora_unet_blocks_0_x", org, 1.0, RANK, 1, down_init="weight_svd", svd_slice=0
    )
    # randomized SVD has per-vector sign freedom, so compare subspaces, not rows
    assert _overlap(_right_basis(a), _right_basis(b)) == pytest.approx(1.0, abs=1e-3)
    _, _, vh = torch.linalg.svd(org.weight.data.float(), full_matrices=False)
    assert _overlap(_right_basis(a), vh[:RANK].T) == pytest.approx(1.0, abs=1e-3)


def test_svd_slices_are_mutually_orthogonal_and_next_in_spectrum():
    torch.manual_seed(0)
    org = torch.nn.Linear(IN, OUT, bias=False)  # spectrum has OUT = 8 = 2·r vectors
    a = LoRAModule("m", org, 1.0, RANK, 1, down_init="weight_svd", svd_slice=0)
    b = LoRAModule("m", org, 1.0, RANK, 1, down_init="weight_svd", svd_slice=1)
    qa, qb = _right_basis(a), _right_basis(b)
    assert _overlap(qa, qb) == pytest.approx(0.0, abs=1e-3)
    _, _, vh = torch.linalg.svd(org.weight.data.float(), full_matrices=False)
    assert _overlap(qb, vh[RANK : 2 * RANK].T) == pytest.approx(1.0, abs=1e-3)
    c = LoRAModule("m", org, 1.0, RANK, 1, down_init="weight_svd", svd_slice=1)
    assert torch.equal(b.lora_down.weight, c.lora_down.weight)  # exact ⇒ identical
    assert torch.allclose(
        qb.norm(dim=0), torch.ones(RANK), atol=1e-4
    )  # same 1/sqrt(3) row-norm match as slice 0


def test_svd_slice_beyond_spectrum_refuses():
    org = torch.nn.Linear(IN, OUT, bias=False)
    with pytest.raises(ValueError, match="exceeds"):
        LoRAModule("m", org, 1.0, RANK, 1, down_init="weight_svd", svd_slice=2)


def test_cfg_svd_slice_needs_weight_svd():
    with pytest.raises(ValueError, match="only applies to down_init='weight_svd'"):
        _cfg(down_init="kaiming", svd_slice="1")
    assert _cfg(down_init="weight_svd", svd_slice="3").svd_slice == 3
    assert _cfg(down_init="kaiming").svd_slice == 0


# --------------------------------------------------------------------------- #
# cfg validation
# --------------------------------------------------------------------------- #
def _cfg(**kwargs):
    from networks.lora_anima.config import LoRANetworkCfg

    return LoRANetworkCfg.from_kwargs(
        kwargs,
        network_dim=RANK,
        network_alpha=1.0,
        neuron_dropout=None,
        module_class=LoRAModule,
        grad_basis_dict=kwargs.pop("_basis", None),
    )


def test_cfg_rejects_gradient_mode_without_a_basis():
    with pytest.raises(ValueError, match="needs a gradient basis"):
        _cfg(down_init="basis_file")


def test_cfg_accepts_gradient_mode_with_a_basis():
    cfg = _cfg(down_init="grad_svd", _basis={"x": _orthonormal(IN, RANK)})
    assert cfg.down_init == "grad_svd" and cfg.grad_basis_dict is not None


def test_cfg_rejects_gradient_mode_on_a_non_plain_variant():
    with pytest.raises(ValueError, match="only applies to plain LoRA"):
        _cfg(
            down_init="basis_file",
            use_ortho="true",
            _basis={"x": _orthonormal(IN, RANK)},
        )


# --------------------------------------------------------------------------- #
# min_snr weighting
# --------------------------------------------------------------------------- #
class _Args:
    min_snr_gamma = 5.0
    timestep_sampling = "sigmoid"
    sigmoid_scale = 1.0
    sigmoid_bias = 0.0


def test_min_snr_has_unit_mean_under_the_run_density():
    """Mean-1 is what keeps the reweighting arm from also being an LR arm."""
    gen = torch.Generator().manual_seed(7)
    sig = torch.sigmoid(torch.randn(1 << 16, generator=gen))
    w = compute_loss_weighting_for_anima("min_snr", sig, _Args())
    assert abs(float(w.mean()) - 1.0) < 0.02


def test_min_snr_peaks_at_snr_equals_gamma_and_falls_off_above():
    gamma = 5.0
    peak_sigma = 1.0 / (1.0 + math.sqrt(gamma))  # SNR((1-σ)/σ)² = γ
    sig = torch.tensor([peak_sigma, 0.5, 0.66, 0.8, 0.95])
    w = min_snr_weighting(sig, gamma)
    assert w[0] == w.max()
    assert torch.all(w[1:].diff() < 0)  # monotone decreasing above the peak
    # the measured motivation: σ>0.5 steps carry 4-10× the noise for equal signal
    assert float(w[2] / w[1]) < 0.5


def test_min_snr_gamma_moves_the_peak():
    """γ only bites where SNR>γ (low σ) — the tilt it produces is post-normalization.

    Un-normalized, ``min(SNR,γ)/(SNR+1)`` at σ=0.6 (SNR 0.44) is γ-independent;
    what a smaller γ does is flatten the low-σ half, which lowers E[w] and so
    raises the *relative* weight the high-σ half receives.
    """
    sig = torch.tensor([0.6])
    assert float(min_snr_weighting(sig, 0.5)) == float(min_snr_weighting(sig, 20.0))

    class _A(_Args):
        pass

    _A.min_snr_gamma = 0.5
    lo = float(compute_loss_weighting_for_anima("min_snr", sig, _A()))
    _A.min_snr_gamma = 20.0
    hi = float(compute_loss_weighting_for_anima("min_snr", sig, _A()))
    assert lo > hi
    assert min_snr_normalizer(5.0) == min_snr_normalizer(5.0)  # cached, stable


def test_other_schemes_are_untouched():
    sig = torch.linspace(0.05, 0.95, 16)
    assert torch.equal(
        compute_loss_weighting_for_anima("uniform", sig), torch.ones_like(sig)
    )
    assert torch.equal(
        compute_loss_weighting_for_anima("none", sig), torch.ones_like(sig)
    )
    assert torch.allclose(
        compute_loss_weighting_for_anima("sigma_sqrt", sig), sig**-2.0
    )
    # min_snr never reaches the weighting function without args (default γ=5)
    assert compute_loss_weighting_for_anima("min_snr", sig).mean() > 0
