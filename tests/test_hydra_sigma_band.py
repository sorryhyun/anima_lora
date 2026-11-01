"""Hard σ-band expert partition: gate masking and σ reaching the modules.

Targets the regression where ``specialize_experts_by_sigma_buckets`` was
training-only — at inference, ``_expert_band`` was non-persistent and
``LoRANetworkCfg.from_weights`` defaulted the flag to False, so soft routing
was running over all E experts at every σ instead of the in-band E/N.
"""

from __future__ import annotations

import pytest
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


def _make_minimal_hydra_network(num_experts: int = 6, num_buckets: int = 3):
    """Build a tiny LoRANetwork with σ-band partitioning enabled.

    Bypasses the model-loading machinery: hand-crafts two HydraLoRAModules,
    runs ``_wire_shared_sigma_buffers`` directly, and exposes the same
    ``set_sigma`` / ``clear_sigma`` surface that production hits.
    """
    from networks.lora_anima.config import LoRANetworkCfg
    from networks.lora_anima.network import LoRANetwork

    cfg = LoRANetworkCfg(
        num_experts=num_experts,
        num_sigma_buckets=num_buckets,
        sigma_bucket_boundaries=[0.0, 0.5, 0.8, 1.0],
        specialize_experts_by_sigma_buckets=True,
        use_moe_style="shared_A",
        route_per_layer=True,
        router_source="sigma",
        sigma_feature_dim=8,
        lora_dim=4,
        alpha=4.0,
    )
    net = LoRANetwork.__new__(LoRANetwork)
    torch.nn.Module.__init__(net)
    net.cfg = cfg
    net.unet_loras = []
    net.text_encoder_loras = []
    net._last_sigma = None
    net._router_stats_cache = None
    net._sigma_router_hits = 0
    net._sigma_router_names = None
    net._sigma_router_re = None
    net._use_hydra = True
    for i in range(2):
        org = torch.nn.Linear(8, 8, bias=False)
        mod = HydraLoRAModule(
            lora_name=f"m{i}",
            org_module=org,
            lora_dim=cfg.lora_dim,
            alpha=cfg.alpha,
            num_experts=cfg.num_experts,
            sigma_feature_dim=cfg.sigma_feature_dim,
            specialize_experts_by_sigma_buckets=True,
            num_sigma_buckets=cfg.num_sigma_buckets,
            sigma_bucket_boundaries=cfg.sigma_bucket_boundaries,
        )
        # Break the zero-router degeneracy so argmax actually differs by σ.
        with torch.no_grad():
            mod.router.weight.normal_(std=0.5)
            mod.router.bias.normal_(std=0.5)
        net.add_module(f"lora_m{i}", mod)
        net.unet_loras.append(mod)
    net._wire_shared_sigma_buffers()
    return net


def test_set_sigma_recovers_aliasing_after_to_device():
    """Regression: ``Module._apply`` (``.to(device)``) reallocates each LoRA
    module's ``_sigma`` buffer independently, breaking the aliasing
    established by ``_wire_shared_sigma_buffers``. The fast in-place path
    of ``set_sigma`` must detect this and re-alias — otherwise the
    network-level ``_shared_sigma`` becomes orphaned and per-module
    ``_sigma`` stays at its zero-init value forever, collapsing σ-band
    partition to band 0 only.
    """
    net = _make_minimal_hydra_network(num_experts=6, num_buckets=3)
    # Simulate ``network.to("meta")``-style buffer re-allocation: PyTorch's
    # ``Module._apply`` rebinds each ``_buffers`` entry independently, which
    # is what kills the aliasing in production. Reproduce that here without
    # actually needing a different device.
    for lora in net._sigma_aware_loras:
        lora._buffers["_sigma"] = lora._buffers["_sigma"].clone()
        lora._buffers["_sigma_features"] = lora._buffers["_sigma_features"].clone()
    # _shared_sigma is a plain Python attribute, so it is *not* touched by
    # ``_apply``; this models the post-``.to()`` orphaned state.
    pre_canonical = net._sigma_aware_loras[0]._buffers["_sigma"]
    assert net._shared_sigma is not pre_canonical, (
        "test setup: aliasing must be broken before set_sigma runs"
    )
    # σ=0.6 lives in band 1 under boundaries [0.5, 0.8]; pre-fix this would
    # be silently dropped because copy_ targeted the orphaned shared tensor.
    net.set_sigma(torch.tensor([0.6]))
    # All modules' live ``_sigma`` must now hold the value the caller passed.
    for lora in net._sigma_aware_loras:
        assert abs(lora._sigma.item() - 0.6) < 1e-5, (
            f"set_sigma did not propagate to {lora}; live _sigma={lora._sigma}"
        )
    # And aliasing must be re-established so the *next* call hits the in-place
    # fast path (production cudagraph correctness depends on stable pointers).
    canonical = net._sigma_aware_loras[0]._buffers["_sigma"]
    assert net._shared_sigma is canonical
    for lora in net._sigma_aware_loras[1:]:
        assert lora._buffers["_sigma"] is canonical
    # Same contract for sinusoidal feature buffers.
    canonical_feat = net._sigma_aware_loras[0]._buffers["_sigma_features"]
    assert net._shared_sigma_features[8] is canonical_feat
    for lora in net._sigma_aware_loras[1:]:
        assert lora._buffers["_sigma_features"] is canonical_feat
    # Round-trip a second call to confirm the fast path still propagates.
    net.set_sigma(torch.tensor([0.9]))
    for lora in net._sigma_aware_loras:
        assert abs(lora._sigma.item() - 0.9) < 1e-5


@pytest.mark.core
def test_set_sigma_band_partition_routes_to_correct_band_after_aliasing_break():
    """End-to-end consequence of the aliasing recovery: σ in band 1 must
    actually steer ``_compute_gate`` to band 1 experts even after a
    simulated ``.to(device)`` rebind. Pre-fix this collapsed to band 0.
    """
    net = _make_minimal_hydra_network(num_experts=6, num_buckets=3)
    # Break aliasing as ``Module._apply`` would.
    for lora in net._sigma_aware_loras:
        lora._buffers["_sigma"] = lora._buffers["_sigma"].clone()
        lora._buffers["_sigma_features"] = lora._buffers["_sigma_features"].clone()
    # σ=0.6 → band 1 (boundaries [0.5, 0.8]) → in-band experts {1, 4}.
    net.set_sigma(torch.tensor([0.6]))
    lx = torch.randn(1, 4, 4)
    for lora in net._sigma_aware_loras:
        gate = lora._compute_gate(lx)
        # Bands 0 and 2 must be exactly zero post-softmax; band 1 carries all mass.
        assert gate[0, 0].item() == 0.0 and gate[0, 3].item() == 0.0  # band 0
        assert gate[0, 2].item() == 0.0 and gate[0, 5].item() == 0.0  # band 2
        in_band = gate[0, torch.tensor([1, 4])].sum().item()
        assert abs(in_band - 1.0) < 1e-5
