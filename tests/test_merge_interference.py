"""Invariant tests for LoRA merge weight-space interference analysis.

Covers the three reference regimes of ``library.anima.merge_analysis.analyze``:
identical deltas (cos +1, ratio →N²/N), negated deltas (cos -1, cancellation),
and orthogonal deltas (cos 0, ratio 1). Also checks the Gram-trace inner product
matches a dense ``⟨ΔW_i, ΔW_j⟩_F`` and that ``--weights`` scaling is respected.
"""

from __future__ import annotations

import torch

from library.anima import merge_analysis as ma


def _lora(down: torch.Tensor, up: torch.Tensor, stem: str = "blk.0.attn"):
    """Build a one-module loaded-LoRA tuple with alpha == rank (scale 1)."""
    r = down.shape[0]
    return ({stem: down}, {stem: up}, {stem: float(r)})


def _dense_inner(up_i, down_i, up_j, down_j) -> float:
    return torch.tensordot(up_i @ down_i, up_j @ down_j, dims=2).item()


def test_gram_inner_matches_dense():
    torch.manual_seed(0)
    up_i, down_i = torch.randn(8, 4), torch.randn(4, 6)
    up_j, down_j = torch.randn(8, 4), torch.randn(4, 6)
    got = ma._gram_inner(up_i, down_i, up_j, down_j)
    assert abs(got - _dense_inner(up_i, down_i, up_j, down_j)) < 1e-4


def test_identical_loras_constructive():
    torch.manual_seed(1)
    down, up = torch.randn(4, 6), torch.randn(8, 4)
    rep = ma.analyze([_lora(down, up), _lora(down, up)], ["a", "b"])
    assert rep.pair_cosine[(0, 1)] == 1.0 or abs(rep.pair_cosine[(0, 1)] - 1.0) < 1e-5
    # Two identical deltas: ‖2ΔW‖² / 2‖ΔW‖² = 4/2 = 2.
    assert abs(rep.overall_energy_ratio - 2.0) < 1e-4
    assert rep.n_shared_modules == 1


def test_negated_loras_destructive():
    torch.manual_seed(2)
    down, up = torch.randn(4, 6), torch.randn(8, 4)
    rep = ma.analyze([_lora(down, up), _lora(down, -up)], ["a", "b"])
    assert abs(rep.pair_cosine[(0, 1)] - (-1.0)) < 1e-5
    # Perfect cancellation: ‖ΔW - ΔW‖² = 0.
    assert abs(rep.overall_energy_ratio) < 1e-4
    assert rep.worst_pair[0] == (0, 1)


def test_orthogonal_loras_independent():
    # Disjoint output rows → ⟨ΔW_i, ΔW_j⟩ = 0 regardless of down.
    down = torch.randn(4, 6)
    up_a = torch.zeros(8, 4)
    up_a[:4] = torch.randn(4, 4)
    up_b = torch.zeros(8, 4)
    up_b[4:] = torch.randn(4, 4)
    rep = ma.analyze([_lora(down, up_a), _lora(down, up_b)], ["a", "b"])
    assert abs(rep.pair_cosine[(0, 1)]) < 1e-5
    assert abs(rep.overall_energy_ratio - 1.0) < 1e-4


def test_weight_scaling_changes_energy_not_cosine():
    torch.manual_seed(3)
    down, up = torch.randn(4, 6), torch.randn(8, 4)
    loaded = [_lora(down, up), _lora(down, -up)]
    base = ma.analyze(loaded, ["a", "b"])
    scaled = ma.analyze(loaded, ["a", "b"], weights=[1.0, 0.5])
    # Cosine is scale-invariant; still anti-aligned.
    assert abs(scaled.pair_cosine[(0, 1)] - base.pair_cosine[(0, 1)]) < 1e-5
    # But partial cancellation now leaves residual energy (ratio > 0).
    assert scaled.overall_energy_ratio > 0.1


def test_identical_loras_full_subspace_overlap():
    torch.manual_seed(4)
    down, up = torch.randn(4, 64), torch.randn(48, 4)
    rep = ma.analyze([_lora(down, up), _lora(down, up)], ["a", "b"])
    ov = rep.pair_overlap[(0, 1)]
    assert abs(ov.out - 1.0) < 1e-5
    assert abs(ov.inp - 1.0) < 1e-5
    # Identical deltas fully overlap AND point the same way (cos +1): the
    # overlap is sign-explained, so it reinforces — it does NOT "collide".
    assert ov.band == "reinforcing"


def test_negated_loras_still_full_overlap():
    # Anti-aligned directions live in the SAME subspace — overlap is unsigned.
    torch.manual_seed(5)
    down, up = torch.randn(4, 64), torch.randn(48, 4)
    rep = ma.analyze([_lora(down, up), _lora(down, -up)], ["a", "b"])
    ov = rep.pair_overlap[(0, 1)]
    assert abs(ov.out - 1.0) < 1e-5
    # Full overlap, cos −1: the shared subspace cancels, not collides.
    assert ov.band == "cancelling"


def test_disjoint_output_rows_zero_out_overlap_full_in_overlap():
    # Same down (identical input subspace), disjoint output rows (orthogonal
    # column spaces): cosine 0, out-overlap 0, in-overlap 1.
    torch.manual_seed(6)
    down = torch.randn(4, 64)
    up_a = torch.zeros(48, 4)
    up_a[:24] = torch.randn(24, 4)
    up_b = torch.zeros(48, 4)
    up_b[24:] = torch.randn(24, 4)
    rep = ma.analyze([_lora(down, up_a), _lora(down, up_b)], ["a", "b"])
    ov = rep.pair_overlap[(0, 1)]
    assert ov.out < 1e-6
    assert abs(ov.inp - 1.0) < 1e-5


def test_shared_subspace_orthogonal_directions_is_the_blind_spot():
    # The case the Frobenius cosine cannot see: both LoRAs confined to the
    # same low-dim output subspace but with ⟨ΔW_a, ΔW_b⟩ ≈ 0. Overlap must
    # flag it as far above chance.
    torch.manual_seed(7)
    d_out, r = 256, 4
    basis, _ = torch.linalg.qr(torch.randn(d_out, 2 * r))  # shared 8-dim subspace
    down_a, down_b = torch.randn(r, 64), torch.randn(r, 64)
    up_a = basis @ torch.randn(2 * r, r)  # independent random mixes of the
    up_b = basis @ torch.randn(2 * r, r)  # same 8 directions
    rep = ma.analyze([_lora(down_a, up_a), _lora(down_b, up_b)], ["a", "b"])
    ov = rep.pair_overlap[(0, 1)]
    # Both column spaces sit inside the shared 8-dim subspace of a 256-dim
    # ambient: chance ≈ r/d_out = 0.016, observed ≈ r/(2r) = 0.5.
    assert ov.out > 0.3
    assert ov.out_xrandom > ma.OVERLAP_COLLIDING_X
    # And the signed cosine stays uninformative (independent random mixes) — so
    # the sign can't explain the overlap and it stays a genuine collision.
    assert abs(rep.pair_cosine[(0, 1)]) < 0.5
    assert ov.band == "colliding"


def test_random_loras_overlap_near_chance():
    torch.manual_seed(8)
    loaded = [
        _lora(torch.randn(8, 512), torch.randn(512, 8)),
        _lora(torch.randn(8, 512), torch.randn(512, 8)),
    ]
    rep = ma.analyze(loaded, ["a", "b"])
    ov = rep.pair_overlap[(0, 1)]
    # Independent random subspaces: ×random ≈ 1, comfortably below 'elevated'.
    assert 0.2 < ov.out_xrandom < ma.OVERLAP_ELEVATED_X
    assert 0.2 < ov.in_xrandom < ma.OVERLAP_ELEVATED_X
    assert ov.band == "chance"


def test_tlora_masked_ranks_do_not_dilute_overlap():
    # Half the rank columns zeroed (T-LoRA style): the effective subspace is
    # rank 2, and two adapters sharing it must still read overlap ≈ 1.
    torch.manual_seed(9)
    down = torch.randn(4, 64)
    core = torch.randn(48, 2)
    up_a = torch.cat([core @ torch.randn(2, 2), torch.zeros(48, 2)], dim=1)
    up_b = torch.cat([core @ torch.randn(2, 2), torch.zeros(48, 2)], dim=1)
    rep = ma.analyze([_lora(down, up_a), _lora(down, up_b)], ["a", "b"])
    assert abs(rep.pair_overlap[(0, 1)].out - 1.0) < 1e-4


def test_shared_init_basin_reinforces_not_collides():
    # The soup case: N adapters fine-tuned from ONE shared uncond init carry a
    # dominant common-mode ΔW + small per-artist residual → high cos AND high
    # overlap. That is reinforcement (benign), not a collision — the whole point
    # of the sign-aware band. The old sign-blind band called every pair here
    # "colliding", which is exactly backwards for a maximally-constructive merge.
    torch.manual_seed(11)
    down_common, up_common = torch.randn(4, 64), torch.randn(48, 4)
    loaded = []
    for k in range(4):
        torch.manual_seed(100 + k)
        # 0.9·shared + 0.1·noise → strongly aligned, same subspace.
        down = down_common + 0.1 * torch.randn(4, 64)
        up = up_common + 0.1 * torch.randn(48, 4)
        loaded.append(_lora(down, up))
    rep = ma.analyze(loaded, [f"soup{k}" for k in range(4)])
    # Net constructive and near-parallel.
    assert rep.overall_energy_ratio > 1.05
    for pair, ov in rep.pair_overlap.items():
        assert rep.pair_cosine[pair] > ma.COS_ALIGNED
        assert ov.out_xrandom > ma.OVERLAP_COLLIDING_X  # subspaces DO overlap...
        assert ov.band == "reinforcing"  # ...but that's benign, not a collision
    # The rendered report must not list any "colliding" module for this merge.
    text = ma.format_report(rep)
    assert "most colliding modules" not in text
    # worst_overlap surfaces the (reinforcing) pair without escalating it.
    assert rep.worst_overlap[1].band == "reinforcing"


def test_marker_payload_carries_overlap():
    torch.manual_seed(10)
    down, up = torch.randn(4, 64), torch.randn(48, 4)
    rep = ma.analyze([_lora(down, up), _lora(down, up)], ["a", "b"])
    payload = rep.marker_payload()
    a, b, out, x = payload["overlap"]
    assert {a, b} == {"a", "b"}
    assert abs(out - 1.0) < 1e-3
    assert x > ma.OVERLAP_COLLIDING_X


def _run_merge(tmp_path, *extra):
    import importlib.util
    import sys
    from pathlib import Path

    from safetensors.torch import load_file, save_file

    spec = importlib.util.spec_from_file_location(
        "merge_loras",
        Path(__file__).resolve().parents[1] / "scripts/toolkits/merge_loras.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    torch.manual_seed(3)
    paths, deltas = [], {"attn": [], "adaln": []}
    stems = {
        "attn": "lora_unet_blocks_0_self_attn_q_proj",
        "adaln": "lora_unet_blocks_0_adaln_modulation_mlp_2",
    }
    for i in range(2):
        sd = {}
        for kind, stem in stems.items():
            down, up = torch.randn(2, 6), torch.randn(8, 2)
            sd[f"{stem}.lora_down.weight"] = down
            sd[f"{stem}.lora_up.weight"] = up
            sd[f"{stem}.alpha"] = torch.tensor(2.0)
            deltas[kind].append(up @ down)
        p = tmp_path / f"in{i}.safetensors"
        save_file(sd, str(p))
        paths.append(str(p))
    out = tmp_path / "out.safetensors"
    argv = sys.argv
    sys.argv = ["merge_loras.py", *paths, "--out", str(out), "--dtype", "fp32", *extra]
    try:
        assert mod.main() == 0
    finally:
        sys.argv = argv
    merged = load_file(str(out))

    def dw(stem):
        return merged[f"{stem}.lora_up.weight"] @ merged[f"{stem}.lora_down.weight"]

    return {k: dw(s) for k, s in stems.items()}, deltas


def test_merge_averages_adaln_and_keeps_it_out_of_the_normalize(tmp_path):
    got, deltas = _run_merge(tmp_path, "--normalize", "global")
    assert torch.allclose(
        got["adaln"], (deltas["adaln"][0] + deltas["adaln"][1]) / 2, atol=1e-5
    )
    # global scale is fit on the attn module alone: ‖merged‖ = avg single norm
    target = sum(d.norm() for d in deltas["attn"]) / 2
    assert torch.isclose(got["attn"].norm(), target, rtol=1e-4)


def test_merge_adaln_sum_restores_the_old_path(tmp_path):
    got, deltas = _run_merge(tmp_path, "--normalize", "off", "--adaln", "sum")
    assert torch.allclose(
        got["adaln"], deltas["adaln"][0] + deltas["adaln"][1], atol=1e-5
    )
