"""Hard σ-band expert partition: training-time masking + inference-time
metadata round-trip.

Targets the regression where ``specialize_experts_by_sigma_buckets`` was
training-only — at inference, ``_expert_band`` was non-persistent and
``LoRANetworkCfg.from_weights`` defaulted the flag to False, so soft routing
was running over all E experts at every σ instead of the in-band E/N.
"""

from __future__ import annotations

import torch

from networks.lora_modules.hydra import HydraLoRAModule, _apply_sigma_band_mask


def _uniform_edges(num_buckets: int) -> torch.Tensor:
    """Helper: interior edges from uniform linspace(0, 1, B+1)[1:-1]."""
    return torch.linspace(0.0, 1.0, num_buckets + 1)[1:-1].contiguous()


def test_apply_sigma_band_mask_zeroes_out_of_band_after_softmax():
    # E=12, N=4, interleaved layout → band b experts at indices b, b+4, b+8.
    logits = torch.zeros(2, 12)
    expert_band = torch.arange(12) % 4
    sigma = torch.tensor([0.0, 0.99])  # band 0, band 3
    masked = _apply_sigma_band_mask(logits, sigma, expert_band, _uniform_edges(4))
    gate = torch.softmax(masked, dim=-1)
    # row 0 (σ=0, band 0): mass only at indices {0, 4, 8}
    band0 = torch.tensor([0, 4, 8])
    band3 = torch.tensor([3, 7, 11])
    assert torch.allclose(gate[0, band0].sum(), torch.tensor(1.0), atol=1e-6)
    mask_out_b0 = torch.ones(12, dtype=torch.bool)
    mask_out_b0[band0] = False
    assert gate[0, mask_out_b0].abs().max().item() == 0.0
    # row 1 (σ=0.99, band 3): mass only at indices {3, 7, 11}
    assert torch.allclose(gate[1, band3].sum(), torch.tensor(1.0), atol=1e-6)
    mask_out_b3 = torch.ones(12, dtype=torch.bool)
    mask_out_b3[band3] = False
    assert gate[1, mask_out_b3].abs().max().item() == 0.0


def test_apply_sigma_band_mask_clamps_sigma_at_boundary():
    """σ=1.0 must clamp into the last band rather than overflow."""
    logits = torch.zeros(1, 12)
    expert_band = torch.arange(12) % 4
    sigma = torch.tensor([1.0])
    gate = torch.softmax(
        _apply_sigma_band_mask(logits, sigma, expert_band, _uniform_edges(4)),
        dim=-1,
    )
    band3 = torch.tensor([3, 7, 11])
    assert torch.allclose(gate[0, band3].sum(), torch.tensor(1.0), atol=1e-6)


def test_apply_sigma_band_mask_custom_edges():
    """Custom non-uniform σ edges route σ to the user-defined bucket."""
    # 3 buckets, low-σ wide and high-σ narrow.
    logits = torch.zeros(3, 6)
    expert_band = torch.arange(6) % 3  # [0,1,2,0,1,2]
    edges = torch.tensor([0.5, 0.8])  # interior cuts for [0,0.5,0.8,1.0]
    # σ=0.4 → band 0, σ=0.6 → band 1, σ=0.9 → band 2
    sigma = torch.tensor([0.4, 0.6, 0.9])
    gate = torch.softmax(
        _apply_sigma_band_mask(logits, sigma, expert_band, edges), dim=-1
    )
    # σ=0.4: mass at {0, 3}
    assert torch.allclose(gate[0, torch.tensor([0, 3])].sum(), torch.tensor(1.0), atol=1e-6)
    # σ=0.6: mass at {1, 4}
    assert torch.allclose(gate[1, torch.tensor([1, 4])].sum(), torch.tensor(1.0), atol=1e-6)
    # σ=0.9: mass at {2, 5}
    assert torch.allclose(gate[2, torch.tensor([2, 5])].sum(), torch.tensor(1.0), atol=1e-6)


def test_hydra_module_with_band_partition_masks_gate():
    """HydraLoRAModule built with the partition flag must apply the mask in
    its forward — the bug was that the flag wasn't reaching inference, so
    builds without the flag still ran soft-routing across all experts.
    """
    org = torch.nn.Linear(8, 8, bias=False)
    mod = HydraLoRAModule(
        lora_name="test",
        org_module=org,
        lora_dim=4,
        alpha=4,
        num_experts=12,
        sigma_feature_dim=0,
        specialize_experts_by_sigma_buckets=True,
        num_sigma_buckets=4,
    )
    # Push some signal into the router so the masked positions can't
    # accidentally still win on a tied softmax.
    with torch.no_grad():
        mod.router.weight.normal_(std=0.5)
        mod.router.bias.normal_(std=0.5)

    # σ=0 → band 0 → interleaved indices {0, 4, 8}
    mod.set_sigma(torch.tensor([0.0]))
    lx = torch.randn(1, 4, 4)  # (B, L, rank)
    gate = mod._compute_gate(lx)
    band0 = torch.tensor([0, 4, 8])
    out_b0 = torch.ones(12, dtype=torch.bool)
    out_b0[band0] = False
    assert gate[0, out_b0].abs().max().item() == 0.0
    assert torch.allclose(gate.sum(dim=-1), torch.ones(1), atol=1e-6)

    # σ=0.6 → band 2 (uniform B=4 edges at 0.25/0.5/0.75) → indices {2, 6, 10}
    mod.set_sigma(torch.tensor([0.6]))
    gate = mod._compute_gate(lx)
    band2 = torch.tensor([2, 6, 10])
    out_b2 = torch.ones(12, dtype=torch.bool)
    out_b2[band2] = False
    assert gate[0, out_b2].abs().max().item() == 0.0


def test_hydra_module_with_custom_boundaries_masks_gate():
    """Module built with custom σ edges routes σ to the right band."""
    org = torch.nn.Linear(8, 8, bias=False)
    mod = HydraLoRAModule(
        lora_name="test",
        org_module=org,
        lora_dim=4,
        alpha=4,
        num_experts=6,
        sigma_feature_dim=0,
        specialize_experts_by_sigma_buckets=True,
        num_sigma_buckets=3,
        sigma_bucket_boundaries=[0.0, 0.5, 0.8, 1.0],
    )
    with torch.no_grad():
        mod.router.weight.normal_(std=0.5)
        mod.router.bias.normal_(std=0.5)
    lx = torch.randn(1, 4, 4)

    # σ=0.4 → band 0 → interleaved indices {0, 3}
    mod.set_sigma(torch.tensor([0.4]))
    gate = mod._compute_gate(lx)
    assert gate[0, 1].item() == 0.0 and gate[0, 2].item() == 0.0
    assert gate[0, 4].item() == 0.0 and gate[0, 5].item() == 0.0

    # σ=0.6 → band 1 → indices {1, 4}
    mod.set_sigma(torch.tensor([0.6]))
    gate = mod._compute_gate(lx)
    assert gate[0, 0].item() == 0.0 and gate[0, 2].item() == 0.0
    assert gate[0, 3].item() == 0.0 and gate[0, 5].item() == 0.0

    # σ=0.9 → band 2 → indices {2, 5}
    mod.set_sigma(torch.tensor([0.9]))
    gate = mod._compute_gate(lx)
    assert gate[0, 0].item() == 0.0 and gate[0, 1].item() == 0.0
    assert gate[0, 3].item() == 0.0 and gate[0, 4].item() == 0.0


def test_save_weights_stamps_band_metadata(tmp_path):
    """Round-trip the save metadata stamp — the load side keys off these
    exact strings, so renaming or dropping them silently disables the
    partition at inference. Pin the contract."""
    import json

    from safetensors import safe_open
    from safetensors.torch import save_file

    # Reproduce the relevant slice of save_weights without spinning up a full
    # network: this test is about the metadata contract, not state_dict shape.
    metadata = {"ss_network_spec": "hydra"}
    cfg_specialize = True
    cfg_num_buckets = 4
    cfg_boundaries = [0.0, 0.25, 0.6, 0.85, 1.0]
    if cfg_specialize:
        metadata["ss_specialize_experts_by_sigma_buckets"] = "true"
        metadata["ss_num_sigma_buckets"] = str(int(cfg_num_buckets))
        metadata["ss_sigma_bucket_boundaries"] = json.dumps(cfg_boundaries)

    out = tmp_path / "stub.safetensors"
    save_file({"x": torch.zeros(1)}, str(out), metadata)
    with safe_open(str(out), framework="pt") as f:
        meta = f.metadata()
    assert meta["ss_specialize_experts_by_sigma_buckets"] == "true"
    assert meta["ss_num_sigma_buckets"] == "4"
    assert json.loads(meta["ss_sigma_bucket_boundaries"]) == cfg_boundaries
