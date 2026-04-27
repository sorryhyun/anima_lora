"""Forward + gradient equality tests for the custom LoRA down-projection autograd.

The custom Functions in ``networks.lora_modules.custom_autograd`` are intended
to be a memory optimization that does not change any math. Forward must match
``F.linear(rebalance(x).float(), weight.float())`` exactly, and gradients on
``x`` and ``weight`` must match what the existing path produces.
"""

from __future__ import annotations

import torch

from networks.lora_modules.custom_autograd import (
    LoRADownProjectFn,
    ScaledLoRADownProjectFn,
    lora_down_project,
)


def _reference_unscaled(x, weight):
    return torch.nn.functional.linear(x.float(), weight.float())


def _reference_scaled(x, weight, inv_scale):
    x_lora = x * inv_scale
    return torch.nn.functional.linear(x_lora.float(), weight.float())


def _make_inputs(in_dim=64, out_dim=16, tokens=32, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(1, tokens, in_dim, dtype=dtype, requires_grad=True)
    weight = torch.randn(out_dim, in_dim, dtype=torch.float32, requires_grad=True)
    return x, weight


def _clone_leaf(t: torch.Tensor) -> torch.Tensor:
    c = t.detach().clone()
    c.requires_grad_(t.requires_grad)
    return c


def test_unscaled_forward_matches_reference():
    x, weight = _make_inputs()
    out_custom = LoRADownProjectFn.apply(x, weight)
    out_ref = _reference_unscaled(x, weight)
    assert torch.equal(out_custom, out_ref), (
        "forward output must be bitwise equal — same fp32 matmul"
    )


def test_unscaled_grads_match_reference():
    x, weight = _make_inputs()
    xc, wc = _clone_leaf(x), _clone_leaf(weight)

    out_custom = LoRADownProjectFn.apply(x, weight)
    out_ref = _reference_unscaled(xc, wc)
    assert torch.equal(out_custom, out_ref)

    grad_out = torch.randn_like(out_custom)
    out_custom.backward(grad_out)
    out_ref.backward(grad_out)

    assert torch.equal(x.grad, xc.grad), "grad_x must match the reference"
    assert torch.equal(weight.grad, wc.grad), "grad_weight must match the reference"


def test_scaled_forward_matches_reference():
    x, weight = _make_inputs()
    torch.manual_seed(1)
    inv_scale = torch.rand(x.shape[-1], dtype=torch.float32) + 0.5  # [0.5, 1.5)

    out_custom = ScaledLoRADownProjectFn.apply(x, weight, inv_scale)
    out_ref = _reference_scaled(x, weight, inv_scale)
    assert torch.equal(out_custom, out_ref)


def test_scaled_grads_match_reference():
    x, weight = _make_inputs()
    xc, wc = _clone_leaf(x), _clone_leaf(weight)
    torch.manual_seed(1)
    inv_scale = torch.rand(x.shape[-1], dtype=torch.float32) + 0.5

    out_custom = ScaledLoRADownProjectFn.apply(x, weight, inv_scale)
    out_ref = _reference_scaled(xc, wc, inv_scale)
    assert torch.equal(out_custom, out_ref)

    grad_out = torch.randn_like(out_custom)
    out_custom.backward(grad_out)
    out_ref.backward(grad_out)

    assert torch.equal(x.grad, xc.grad), "grad_x must match scaled reference"
    assert torch.equal(weight.grad, wc.grad), "grad_weight must match scaled reference"


def test_dispatch_helper_routes_correctly():
    x, weight = _make_inputs()
    out_unscaled = lora_down_project(x, weight, None)
    assert torch.equal(out_unscaled, _reference_unscaled(x, weight))

    inv_scale = torch.rand(x.shape[-1], dtype=torch.float32) + 0.5
    out_scaled = lora_down_project(x, weight, inv_scale)
    assert torch.equal(out_scaled, _reference_scaled(x, weight, inv_scale))


def test_bf16_weight_storage_returns_correct_grad_dtype():
    """Under full_bf16, lora_down.weight is bf16. grad_weight must come back
    in that dtype (matching the existing path's implicit cast)."""
    torch.manual_seed(0)
    x = torch.randn(1, 8, 16, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(4, 16, dtype=torch.bfloat16, requires_grad=True)

    out = LoRADownProjectFn.apply(x, weight)
    out.sum().backward()
    assert weight.grad.dtype == torch.bfloat16
    assert x.grad.dtype == torch.bfloat16


def test_module_flag_off_is_bitwise_identical_to_legacy_path():
    """With ``use_custom_down_autograd=False`` (the default), a LoRAModule's
    forward must be identical to the pre-change path.
    """
    from networks.lora_modules.lora import LoRAModule

    torch.manual_seed(0)
    base = torch.nn.Linear(32, 24, bias=False).to(torch.bfloat16)
    module = LoRAModule("test", base, multiplier=1.0, lora_dim=4, alpha=4)
    module.apply_to()
    module.train()
    assert module.use_custom_down_autograd is False

    x = torch.randn(1, 8, 32, dtype=torch.bfloat16)
    out_default = base.forward(x)  # org_module is deleted after apply_to; uses LoRA forward

    # Flip the flag on and confirm equality of forward output
    module.use_custom_down_autograd = True
    out_flag = base.forward(x)

    assert torch.equal(out_default, out_flag), (
        "forward must match regardless of the use_custom_down_autograd flag"
    )


def test_module_flag_on_matches_legacy_gradients():
    """End-to-end through LoRAModule: flipping the flag should not change the
    gradients that land on lora_down.weight or lora_up.weight.
    """
    from networks.lora_modules.lora import LoRAModule

    def run(use_custom: bool):
        torch.manual_seed(0)
        base = torch.nn.Linear(32, 24, bias=False).to(torch.bfloat16)
        base.weight.requires_grad_(False)
        module = LoRAModule("t", base, multiplier=1.0, lora_dim=4, alpha=4)
        module.apply_to()
        module.train()
        module.use_custom_down_autograd = use_custom

        x = torch.randn(1, 8, 32, dtype=torch.bfloat16, requires_grad=True)
        out = base.forward(x)
        out.sum().backward()
        return (
            out.detach().clone(),
            module.lora_down.weight.grad.detach().clone(),
            module.lora_up.weight.grad.detach().clone(),
            x.grad.detach().clone(),
        )

    o_legacy, gd_legacy, gu_legacy, gx_legacy = run(False)
    o_custom, gd_custom, gu_custom, gx_custom = run(True)

    assert torch.equal(o_legacy, o_custom), "module forward differs with flag on"
    assert torch.equal(gd_legacy, gd_custom), "lora_down grad differs"
    assert torch.equal(gu_legacy, gu_custom), "lora_up grad differs"
    assert torch.equal(gx_legacy, gx_custom), "grad_x differs"


# ---------------------------------------------------------------------------
# Hydra / Ortho / OrthoHydra — end-to-end flag-on vs flag-off equality
# ---------------------------------------------------------------------------


def _named_trainable_grads(module: torch.nn.Module):
    """Snapshot cloned .grad for every trainable param in a module."""
    return {
        n: p.grad.detach().clone()
        for n, p in module.named_parameters()
        if p.grad is not None
    }


def _assert_grads_equal(a: dict, b: dict, label: str):
    assert a.keys() == b.keys(), f"{label}: param sets differ: {a.keys() ^ b.keys()}"
    for k in a:
        assert torch.equal(a[k], b[k]), f"{label}: grad on {k!r} differs"


def test_hydra_flag_on_matches_legacy_gradients():
    """HydraLoRAModule: flipping the flag must not change any trainable grad
    (lora_down, lora_up_weight, router.weight, router.bias), the forward
    output, or grad_x — router pooling consumes the rank-space ``lx``, which
    is unchanged bit-for-bit."""
    from networks.lora_modules.hydra import HydraLoRAModule

    def run(use_custom: bool):
        torch.manual_seed(0)
        base = torch.nn.Linear(32, 24, bias=False).to(torch.bfloat16)
        base.weight.requires_grad_(False)
        module = HydraLoRAModule(
            "h", base, multiplier=1.0, lora_dim=4, alpha=4,
            num_experts=3, sigma_feature_dim=0,
        )
        module.apply_to()
        module.train()
        module.use_custom_down_autograd = use_custom

        torch.manual_seed(1)
        x = torch.randn(2, 8, 32, dtype=torch.bfloat16, requires_grad=True)
        out = base.forward(x)
        out.sum().backward()
        return out.detach().clone(), _named_trainable_grads(module), x.grad.detach().clone()

    o_legacy, g_legacy, gx_legacy = run(False)
    o_custom, g_custom, gx_custom = run(True)

    assert torch.equal(o_legacy, o_custom), "Hydra forward differs with flag on"
    _assert_grads_equal(g_legacy, g_custom, "Hydra")
    assert torch.equal(gx_legacy, gx_custom), "Hydra grad_x differs"


def test_hydra_sigma_feature_cache_updates_and_clears():
    """Sigma-router features are precomputed once per step and cached on modules."""
    from networks.lora_modules.hydra import (
        HydraLoRAModule,
        _sigma_sinusoidal_features,
    )

    torch.manual_seed(0)
    base = torch.nn.Linear(32, 24, bias=False)
    module = HydraLoRAModule(
        "h", base, multiplier=1.0, lora_dim=4, alpha=4,
        num_experts=3, sigma_feature_dim=8,
    )

    sigmas = torch.tensor([0.25, 0.5], dtype=torch.float32)
    expected = _sigma_sinusoidal_features(sigmas, 8)
    module.set_sigma(sigmas, expected)

    assert torch.equal(module._sigma, sigmas)
    assert torch.equal(module._sigma_features, expected)

    module.clear_sigma()
    assert torch.equal(module._sigma, torch.zeros_like(sigmas))
    assert torch.equal(
        module._sigma_features,
        _sigma_sinusoidal_features(torch.zeros_like(sigmas), 8),
    )


def test_ortho_flag_on_matches_legacy_gradients():
    """OrthoLoRAExpModule: custom fn treats Q_eff as the 'weight' input.
    grad_Q_eff must propagate through R_q @ Q_basis into S_q, and through
    the P path into S_p + lambda_layer — all bitwise equal to the legacy
    path."""
    from networks.lora_modules.ortho import OrthoLoRAExpModule

    def run(use_custom: bool):
        torch.manual_seed(0)
        base = torch.nn.Linear(32, 24, bias=False).to(torch.bfloat16)
        base.weight.requires_grad_(False)
        module = OrthoLoRAExpModule("o", base, multiplier=1.0, lora_dim=4, alpha=4)
        module.apply_to()
        module.train()
        module.use_custom_down_autograd = use_custom

        # Randomize S_q / S_p / lambda so Cayley(0)=I isn't a degenerate case
        with torch.no_grad():
            module.S_q.copy_(torch.randn_like(module.S_q) * 0.1)
            module.S_p.copy_(torch.randn_like(module.S_p) * 0.1)
            module.lambda_layer.copy_(torch.randn_like(module.lambda_layer) * 0.1)

        torch.manual_seed(1)
        x = torch.randn(2, 8, 32, dtype=torch.bfloat16, requires_grad=True)
        out = base.forward(x)
        out.sum().backward()
        return out.detach().clone(), _named_trainable_grads(module), x.grad.detach().clone()

    o_legacy, g_legacy, gx_legacy = run(False)
    o_custom, g_custom, gx_custom = run(True)

    assert torch.equal(o_legacy, o_custom), "Ortho forward differs with flag on"
    _assert_grads_equal(g_legacy, g_custom, "Ortho")
    assert torch.equal(gx_legacy, gx_custom), "Ortho grad_x differs"
    # Sanity: S_q actually got gradient (otherwise the test is vacuous)
    assert "S_q" in g_legacy and g_legacy["S_q"].abs().sum() > 0, (
        "S_q grad is zero — test would pass vacuously; inputs need randomization"
    )


def test_ortho_hydra_flag_on_matches_legacy_gradients():
    """OrthoHydraLoRAExpModule: Cayley-on-Q + MoE per-expert P. Flag toggles
    only the shared Q_eff projection; router / P_eff / λ paths are unchanged.
    """
    from networks.lora_modules.ortho import OrthoHydraLoRAExpModule

    def run(use_custom: bool):
        torch.manual_seed(0)
        base = torch.nn.Linear(32, 24, bias=False).to(torch.bfloat16)
        base.weight.requires_grad_(False)
        module = OrthoHydraLoRAExpModule(
            "oh", base, multiplier=1.0, lora_dim=4, alpha=4,
            num_experts=3, sigma_feature_dim=0,
        )
        module.apply_to()
        module.train()
        module.use_custom_down_autograd = use_custom

        with torch.no_grad():
            module.S_q.copy_(torch.randn_like(module.S_q) * 0.1)
            module.S_p.copy_(torch.randn_like(module.S_p) * 0.1)
            module.lambda_layer.copy_(torch.randn_like(module.lambda_layer) * 0.1)

        torch.manual_seed(1)
        x = torch.randn(2, 8, 32, dtype=torch.bfloat16, requires_grad=True)
        out = base.forward(x)
        out.sum().backward()
        return out.detach().clone(), _named_trainable_grads(module), x.grad.detach().clone()

    o_legacy, g_legacy, gx_legacy = run(False)
    o_custom, g_custom, gx_custom = run(True)

    assert torch.equal(o_legacy, o_custom), "OrthoHydra forward differs with flag on"
    _assert_grads_equal(g_legacy, g_custom, "OrthoHydra")
    assert torch.equal(gx_legacy, gx_custom), "OrthoHydra grad_x differs"
    assert "S_q" in g_legacy and g_legacy["S_q"].abs().sum() > 0
