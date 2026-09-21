"""Smoke tests for the GlobalRouter wiring on plan2 task #4.

Covers:
  * GlobalRouter init is uniform-at-step-0 (zero output layer + softmax/τ).
  * ``set_routing_weights`` broadcasts to every routing-aware module.
  * ``set_fei`` end-to-end: fires the GlobalRouter and broadcasts.
  * Aliasing-recovery after a simulated ``Module._apply`` orphans the link
    (mirrors ``test_hydra_sigma_band.test_set_sigma_recovers_aliasing_after_to_device``).

The vehicle is the ``use_moe_style="shared_A"`` + ``route_per_layer=False``
cell: ``HydraLoRAModule(use_global_router=True)`` drops its per-layer router
and consumes the gates the network-level ``GlobalRouter`` broadcasts.
"""

from __future__ import annotations

import torch

from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.network import GlobalRouter, LoRANetwork
from networks.lora_modules import HydraLoRAModule


def _make_minimal_hydra_global_router_network(
    *,
    num_experts: int = 3,
    fei_dim: int = 2,
    in_dim: int = 16,
    out_dim: int = 16,
    lora_dim: int = 4,
) -> LoRANetwork:
    """Hand-roll a tiny LoRANetwork with two ``shared_A`` Hydra modules built
    with ``use_global_router=True`` (the ``route_per_layer=False`` cell).

    Bypasses the DiT model-loading machinery — same pattern as
    ``test_hydra_sigma_band._make_minimal_hydra_network``.
    """
    cfg = LoRANetworkCfg(
        num_experts=num_experts,
        use_moe_style="shared_A",
        route_per_layer=False,
        router_source="fei",
        fei_feature_dim=fei_dim,
        router_hidden_dim=32,
        router_tau=0.7,
        lora_dim=lora_dim,
        alpha=float(lora_dim),
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
    net._hydra_router_re = None
    net._hydra_router_names = None
    net._hydra_router_hits = 0
    net._hydra_router_misses = 0
    net._fei_router_hits = 0
    net._fei_router_re = None
    net._fei_router_names = None
    net.use_fei_router = True
    net.use_sigma_router = False
    net._channel_scale_misses = []
    net._channel_scale_hits = 0
    net._last_up_grad_stats = {}
    net._use_hydra = True
    net._balance_loss_weight = 0.0
    for i in range(2):
        org = torch.nn.Linear(in_dim, out_dim, bias=False)
        mod = HydraLoRAModule(
            lora_name=f"m{i}",
            org_module=org,
            lora_dim=cfg.lora_dim,
            alpha=cfg.alpha,
            num_experts=cfg.num_experts,
            use_global_router=True,
        )
        net.add_module(f"lora_m{i}", mod)
        net.unet_loras.append(mod)
    # Wire the aliased-buffer registry + the GlobalRouter (mirrors the tail
    # of ``LoRANetwork.__init__`` that we skipped by using ``__new__``).
    net._wire_shared_sigma_buffers()
    net._wire_shared_fei_buffers()
    net._wire_shared_routing_buffers()
    net.global_router = GlobalRouter(
        input_dim=fei_dim,
        num_experts=num_experts,
        hidden_dim=cfg.router_hidden_dim,
        tau=cfg.router_tau,
    )
    net.add_module("global_router", net.global_router)
    return net


def test_global_router_uniform_at_init():
    """Zero-init output layer + softmax/τ → exactly uniform over experts."""
    router = GlobalRouter(input_dim=4, num_experts=5, hidden_dim=8, tau=0.7)
    x = torch.randn(3, 4)
    gates = router(x)
    assert gates.shape == (3, 5)
    assert torch.allclose(gates, torch.full_like(gates, 1.0 / 5.0))
    # Side effects: _last_gates / _last_input populated and detached.
    assert router._last_gates is not None
    assert not router._last_gates.requires_grad
    assert router._last_input is not None


def test_global_router_pools_and_layernorms_text_input():
    """``crossattn_emb`` source: ``GlobalRouter(apply_layer_norm=True)`` takes
    a raw ``(B, L, D)`` text tensor, RMS-pools over the sequence axis, and
    stays uniform-at-init (zero output layer → softmax = 1/E)."""
    from networks.lora_anima.network import CROSSATTN_EMB_DIM

    router = GlobalRouter(
        input_dim=CROSSATTN_EMB_DIM,
        num_experts=4,
        hidden_dim=8,
        tau=0.7,
        apply_layer_norm=True,
    )
    assert router.ln_in is not None
    x = torch.randn(3, 7, CROSSATTN_EMB_DIM)  # (B, L, D)
    gates = router(x)
    assert gates.shape == (3, 4)
    assert torch.allclose(gates, torch.full_like(gates, 0.25), atol=1e-6)


def test_set_crossattn_routing_broadcasts_text_gates():
    """``set_crossattn_routing`` fires the crossattn GlobalRouter on a raw
    ``(B, L, D)`` text tensor and broadcasts the gates to every routing-aware
    module — the same ``_routing_weights`` slot the σ/FEI router writes."""
    from networks.lora_anima.network import CROSSATTN_EMB_DIM

    net = _make_minimal_hydra_global_router_network(num_experts=3, fei_dim=2)
    # Reconfigure for the crossattn-emb cell: advertise the flag and swap in a
    # text-fed GlobalRouter (pooling + LN).
    net.use_crossattn_router = True
    net.global_router = GlobalRouter(
        input_dim=CROSSATTN_EMB_DIM,
        num_experts=3,
        hidden_dim=32,
        tau=0.7,
        apply_layer_norm=True,
    )
    net.add_module("global_router", net.global_router)
    with torch.no_grad():
        net.global_router.net[-1].weight.normal_(std=1.0)
        net.global_router.net[-1].bias.normal_(std=1.0)
    emb = torch.randn(1, 5, CROSSATTN_EMB_DIM)  # (B, L, D)
    net.set_crossattn_routing(emb)
    canonical = net._routing_aware_loras[0]._buffers["_routing_weights"]
    with torch.no_grad():
        expected = net.global_router(emb)
    assert torch.allclose(canonical, expected, atol=1e-6)


def test_set_crossattn_routing_noop_without_flag():
    """No crossattn router wired (``use_crossattn_router`` falsy) → the call
    leaves the uniform placeholder untouched."""
    net = _make_minimal_hydra_global_router_network(num_experts=3, fei_dim=2)
    net.clear_routing_weights()
    before = net._routing_aware_loras[0]._buffers["_routing_weights"].clone()
    # Default helper sets up a FEI router, not a crossattn one.
    net.set_crossattn_routing(torch.randn(1, 4, 1024))
    after = net._routing_aware_loras[0]._buffers["_routing_weights"]
    assert torch.allclose(before, after)


def test_set_routing_weights_broadcasts():
    """``set_routing_weights`` writes one gate tensor to every routing-aware module."""
    net = _make_minimal_hydra_global_router_network(num_experts=3, fei_dim=2)
    gates = torch.tensor([[0.5, 0.3, 0.2]])
    net.set_routing_weights(gates)
    # Every routing-aware module's _routing_weights now holds those gates,
    # and they all alias the same shared tensor.
    canonical = net._routing_aware_loras[0]._buffers["_routing_weights"]
    for lora in net._routing_aware_loras:
        assert lora._buffers["_routing_weights"] is canonical
    assert torch.allclose(canonical, gates)


def test_clear_routing_weights_resets_to_uniform():
    net = _make_minimal_hydra_global_router_network(num_experts=4, fei_dim=2)
    net.set_routing_weights(torch.tensor([[0.7, 0.1, 0.1, 0.1]]))
    net.clear_routing_weights()
    canonical = net._routing_aware_loras[0]._buffers["_routing_weights"]
    assert torch.allclose(canonical, torch.full_like(canonical, 0.25))


def test_set_fei_fires_global_router_and_broadcasts():
    """End-to-end: caller pushes FEI via ``set_fei``, the GlobalRouter
    runs, and its gates land on every routing-aware module."""
    net = _make_minimal_hydra_global_router_network(num_experts=3, fei_dim=2)
    # Perturb router weights so its output is no longer the uniform init —
    # otherwise we can't distinguish "the router fired" from "the
    # placeholder was already uniform".
    with torch.no_grad():
        net.global_router.net[-1].weight.normal_(std=1.0)
        net.global_router.net[-1].bias.normal_(std=1.0)
    fei = torch.tensor([[0.7, 0.3]])
    net.set_fei(fei)
    canonical = net._routing_aware_loras[0]._buffers["_routing_weights"]
    # Should match what the router produces on this FEI.
    with torch.no_grad():
        expected = net.global_router(fei)
    assert torch.allclose(canonical, expected, atol=1e-6)


def test_set_routing_weights_recovers_aliasing_after_apply():
    """Regression: ``Module._apply`` (``.to(device)``) reallocates each
    routing-aware module's ``_routing_weights`` buffer independently,
    breaking the aliasing established by ``_wire_shared_routing_buffers``.

    Mirrors ``test_hydra_sigma_band.test_set_sigma_recovers_aliasing_after_to_device``
    for the new buffer. Without aliasing recovery, the in-place ``copy_``
    targets the orphaned shared tensor and per-module ``_routing_weights``
    stays at its uniform placeholder forever.
    """
    net = _make_minimal_hydra_global_router_network(num_experts=3, fei_dim=2)
    # Simulate Module._apply: rebind each module's buffer to an independent
    # clone, so the shared-attribute identity link breaks.
    for lora in net._routing_aware_loras:
        lora._buffers["_routing_weights"] = lora._buffers["_routing_weights"].clone()
    pre_canonical = net._routing_aware_loras[0]._buffers["_routing_weights"]
    assert net._shared_routing_weights is not pre_canonical, (
        "test setup: aliasing must be broken before set_routing_weights runs"
    )
    gates = torch.tensor([[0.8, 0.1, 0.1]])
    net.set_routing_weights(gates)
    # All modules' live ``_routing_weights`` must hold the value the caller
    # passed, and aliasing must be re-established.
    canonical = net._routing_aware_loras[0]._buffers["_routing_weights"]
    assert net._shared_routing_weights is canonical
    for lora in net._routing_aware_loras:
        assert lora._buffers["_routing_weights"] is canonical
        assert torch.allclose(lora._buffers["_routing_weights"], gates)


def test_hydra_module_consumes_routing_weights():
    """``HydraLoRAModule(use_global_router=True).forward`` weights the
    per-expert sum by ``_routing_weights``. Confirm the broadcast actually
    changes outputs.
    """
    net = _make_minimal_hydra_global_router_network(num_experts=3, fei_dim=2)
    mod = net.unet_loras[0]
    # Break zero-init expert ups so the adapter contribution is non-zero.
    with torch.no_grad():
        mod.lora_up_weight.normal_(std=0.1)
    mod.apply_to()
    x = torch.randn(4, 16)
    # Uniform routing.
    net.clear_routing_weights()
    y_uniform = mod(x)
    # One-hot on expert 0.
    one_hot = torch.zeros(4, 3)
    one_hot[:, 0] = 1.0
    net.set_routing_weights(one_hot)
    y_e0 = mod(x)
    # One-hot on expert 1.
    one_hot[:] = 0.0
    one_hot[:, 1] = 1.0
    net.set_routing_weights(one_hot)
    y_e1 = mod(x)
    # Different gates must produce different outputs (with overwhelmingly
    # high probability for random expert weights).
    assert not torch.allclose(y_uniform, y_e0, atol=1e-4)
    assert not torch.allclose(y_e0, y_e1, atol=1e-4)


def test_metrics_emits_fera_keys_when_global_router_fired():
    """After ``set_fei`` fires the GlobalRouter, ``metrics`` emits the
    ``fera/router_entropy`` / ``fera/router_margin`` / ``fera/expert_usage/*``
    keys (the ``fera/`` prefix is kept for the GlobalRouter). Hand-rolling the
    network skips the model-loading machinery; the
    ``_use_hydra`` attr is set in the helper so the gate before the
    ``fera/*`` block is satisfied.
    """
    from library.training.metrics import MetricContext

    net = _make_minimal_hydra_global_router_network(num_experts=3, fei_dim=2)
    with torch.no_grad():
        net.global_router.net[-1].weight.normal_(std=1.0)
    fei = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
    net.set_fei(fei)
    out = net.metrics(MetricContext(args=None, network=net))
    assert "fera/router_entropy" in out
    assert "fera/router_margin" in out
    assert "fera/expert_usage/0" in out
    assert "fera/expert_usage/1" in out
    assert "fera/expert_usage/2" in out
    # Usage sums to ~1 (per-expert mean gate weight, softmax rows → 1).
    total = sum(out[f"fera/expert_usage/{i}"] for i in range(3))
    assert abs(total - 1.0) < 1e-5


def test_hydra_global_router_receives_gradient_from_expert_forward():
    """Gates appear as live multipliers in the expert mixing, so plain
    L_denoise backprop must reach the GlobalRouter parameters: ``set_fei``
    fires the GlobalRouter, gates land on every ``HydraLoRAModule(use_global_router=True)``
    via a live (non-detached) buffer reference, and ``L_denoise`` backprop
    populates ``global_router.net[-1].weight.grad``.

    Also the regression for the pre-fix state where ``set_fei`` ran the router
    under ``torch.no_grad()`` and ``set_routing_weights`` detached the gates,
    leaving the router stuck at zero-init (uniform 1/E) forever, and for the
    latent failure mode where ``HydraLoRAModule.set_routing_weights`` detaches
    the broadcast tensor — the network-level set_routing_weights bypasses the
    module method today, but the module-level fallback path must not silently
    break the gradient path either.
    """
    torch.manual_seed(0)
    net = _make_minimal_hydra_global_router_network(num_experts=3, fei_dim=2)
    with torch.no_grad():
        net.global_router.net[-1].weight.normal_(std=0.1)
        net.global_router.net[-1].bias.normal_(std=0.1)
        for lora in net._routing_aware_loras:
            # Break shared-A zero-init so the per-expert sum is non-constant
            # and the gate gradient has signal to land on.
            lora.lora_up_weight.data.normal_(std=0.1)

    for lora in net._routing_aware_loras:
        lora.apply_to()
    fei = torch.tensor([[0.7, 0.3]])
    net.set_fei(fei)
    x = torch.randn(1, 16, requires_grad=False)
    y = net._routing_aware_loras[0](x)
    loss = y.pow(2).sum()
    loss.backward()

    final_layer = net.global_router.net[-1]
    assert final_layer.weight.grad is not None, (
        "Hydra GlobalRouter final-layer.weight.grad is None — autograd path "
        "from expert forward to router is broken."
    )
    assert final_layer.weight.grad.abs().sum().item() > 0.0, (
        "Hydra GlobalRouter final-layer.weight.grad is all zeros — gates are "
        "not carrying gradient through the broadcast."
    )
