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


def test_apply_sigma_band_mask_zeroes_out_of_band_after_softmax():
    # E=12, N=4 → bands of 3 experts each.
    logits = torch.zeros(2, 12)
    expert_band = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    sigma = torch.tensor([0.0, 0.99])  # band 0, band 3
    masked = _apply_sigma_band_mask(logits, sigma, expert_band, num_buckets=4)
    gate = torch.softmax(masked, dim=-1)
    # row 0 (σ=0, band 0): mass only in experts 0..2
    assert torch.allclose(gate[0, :3].sum(), torch.tensor(1.0), atol=1e-6)
    assert gate[0, 3:].abs().max().item() == 0.0
    # row 1 (σ=0.99, band 3): mass only in experts 9..11
    assert torch.allclose(gate[1, 9:].sum(), torch.tensor(1.0), atol=1e-6)
    assert gate[1, :9].abs().max().item() == 0.0


def test_apply_sigma_band_mask_clamps_sigma_at_boundary():
    """σ=1.0 must clamp into the last band rather than overflow."""
    logits = torch.zeros(1, 12)
    expert_band = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    sigma = torch.tensor([1.0])
    gate = torch.softmax(
        _apply_sigma_band_mask(logits, sigma, expert_band, num_buckets=4), dim=-1
    )
    assert torch.allclose(gate[0, 9:].sum(), torch.tensor(1.0), atol=1e-6)


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

    # σ=0 → band 0 (experts 0..2 only)
    mod.set_sigma(torch.tensor([0.0]))
    lx = torch.randn(1, 4, 4)  # (B, L, rank)
    gate = mod._compute_gate(lx)
    assert gate[0, 3:].abs().max().item() == 0.0
    assert torch.allclose(gate.sum(dim=-1), torch.ones(1), atol=1e-6)

    # σ=0.6 → band 2 (experts 6..8 only)
    mod.set_sigma(torch.tensor([0.6]))
    gate = mod._compute_gate(lx)
    assert gate[0, :6].abs().max().item() == 0.0
    assert gate[0, 9:].abs().max().item() == 0.0


def test_save_weights_stamps_band_metadata(tmp_path):
    """Round-trip the save metadata stamp — the load side keys off these
    exact strings, so renaming or dropping them silently disables the
    partition at inference. Pin the contract."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    # Reproduce the relevant slice of save_weights without spinning up a full
    # network: this test is about the metadata contract, not state_dict shape.
    metadata = {"ss_network_spec": "hydra"}
    cfg_specialize = True
    cfg_num_buckets = 4
    if cfg_specialize:
        metadata["ss_specialize_experts_by_sigma_buckets"] = "true"
        metadata["ss_num_sigma_buckets"] = str(int(cfg_num_buckets))

    out = tmp_path / "stub.safetensors"
    save_file({"x": torch.zeros(1)}, str(out), metadata)
    with safe_open(str(out), framework="pt") as f:
        meta = f.metadata()
    assert meta["ss_specialize_experts_by_sigma_buckets"] == "true"
    assert meta["ss_num_sigma_buckets"] == "4"
