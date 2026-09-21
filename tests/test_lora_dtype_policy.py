"""Dtype-policy regression tests for the LoRA-family training forwards.

2026-06-10: the fp32-bottleneck matmul policy (``F.linear(x.float(),
w.float())`` + the custom down-projection autograd) was removed. Training
GEMMs now run in the adapter PARAMETER dtype (bf16). The first cut keyed the GEMM dtype off the *activation*
(``x.dtype`` / ``weight.to(x_lora.dtype)``); that silently upcast, because the
AdaLN ``nn.LayerNorm`` feeding the adapted Linears emits fp32 under
autocast(bf16), so ``x`` arrives fp32 — materializing a full fp32 weight copy
AND (with per-channel scaling on) a fresh fp32 activation out of ``_rebalance``
(``inv_scale``), OOMing for zero numeric gain. The modules now cast ``x`` DOWN
to the param dtype before the rank GEMMs, pinned by
``test_*_fp32_activation_no_upcast`` below. The justification, measured in
``bench/lora_fp32_bottleneck``:

  * ``train.py`` wraps the training forward in ``accelerator.autocast()``
    (default ``mixed_precision="bf16"``). Autocast re-cast the fp32 inputs of
    every ``F.linear``/``einsum``/``bmm`` back to bf16, so the fp32 matmuls
    never executed — live training was already pure-bf16 GEMMs plus dead cast
    traffic.
  * cuBLAS accumulates bf16 GEMMs in fp32 internally, so the only precision
    delta vs a true fp32 GEMM is the final rounding of the rank-R bottleneck
    (invisible to a 200-step Adam probe).

These tests pin the two contracts the rewrite must keep:

  1. **Legacy parity** — under autocast, the new forward (and its grads) is
     bitwise identical to an inline replica of the retired fp32-bottleneck
     code. This is "behavior unchanged for train.py".
  2. **Dtype honesty** — the training forward produces the same result with
     and without autocast, so non-autocast callers (bespoke loops, tests) see
     the same numerics as train.py.

Inference paths are NOT touched by the rewrite: HydraLoRAModule at eval and
EasyControl's KV prefill keep their historical fp32 compute (the inference
engine runs without autocast). Asserted below.

Also home to the flag-independent invariant that lived in the retired
``test_lora_custom_autograd.py``: the hydra σ-feature cache.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# bf16 has ~8 bits of mantissa; per-op relative noise is ~4e-3. Multi-step
# accumulation in matmul + grad makes the achievable rtol ~1e-2 for grads.
_CS_ATOL = 5e-2
_CS_RTOL = 1e-2


def _autocast():
    return torch.autocast("cpu", dtype=torch.bfloat16)


def _make_channel_scale(in_features: int, seed: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(in_features, generator=g, dtype=torch.float32) * 2.0 + 0.5


def _named_trainable_grads(module: torch.nn.Module):
    return {
        n: p.grad.detach().clone()
        for n, p in module.named_parameters()
        if p.grad is not None
    }


def _assert_grads_equal(a: dict, b: dict, label: str):
    assert a.keys() == b.keys(), f"{label}: param sets differ: {a.keys() ^ b.keys()}"
    for k in a:
        assert torch.equal(a[k], b[k]), f"{label}: grad on {k!r} differs"


# ---------------------------------------------------------------------------
# Legacy-path replicas: the retired fp32-bottleneck training branches,
# verbatim. Run under autocast they reproduce what train.py executed before
# the rewrite; the new forwards must match them bitwise.
# ---------------------------------------------------------------------------


def _legacy_lora_forward(module, x):
    org_forwarded = module.org_forward(x)
    x_lora = module._rebalance(x)
    lx = F.linear(x_lora.float(), module.lora_down.weight.float())
    lx = lx * module._timestep_mask
    lx, scale = module._apply_rank_dropout(lx)
    lx = F.linear(lx, module.lora_up.weight.float())
    return org_forwarded + (lx * module.multiplier * scale).to(org_forwarded.dtype)


def _legacy_hydra_forward(module, x):
    org_forwarded = module.org_forward(x)
    x_lora = module._rebalance(x)
    lx = F.linear(x_lora.float(), module.lora_down.weight.float())
    gate = module._compute_gate(lx)
    lx = lx * module._timestep_mask
    lx, scale = module._apply_rank_dropout(lx)
    combined = torch.einsum("be,eod->bod", gate.float(), module.lora_up_weight.float())
    orig_shape = lx.shape
    lx_3d = lx.reshape(orig_shape[0], -1, orig_shape[-1])
    out = torch.bmm(lx_3d, combined.transpose(1, 2)).reshape(*orig_shape[:-1], -1)
    return org_forwarded + (out * module.multiplier * scale).to(org_forwarded.dtype)


def _run_pair(make_module, new_fn, legacy_fn, *, in_dim=32, seed=1):
    """Build two identical modules, run new vs legacy forward+backward under
    autocast, return (out, grads, grad_x) for each."""

    def run(fn):
        base, module = make_module()
        module.train()
        torch.manual_seed(seed)
        x = torch.randn(2, 8, in_dim, dtype=torch.bfloat16, requires_grad=True)
        with _autocast():
            out = fn(module, x)
        out.float().sum().backward()
        return out.detach().clone(), _named_trainable_grads(module), x.grad.clone()

    return run(new_fn), run(legacy_fn)


def test_lora_training_matches_legacy_under_autocast():
    from networks.lora_modules.lora import LoRAModule

    def make():
        torch.manual_seed(0)
        base = torch.nn.Linear(32, 24, bias=False).to(torch.bfloat16)
        base.weight.requires_grad_(False)
        module = LoRAModule("m", base, multiplier=1.0, lora_dim=4, alpha=4)
        with torch.no_grad():
            module.lora_up.weight.copy_(torch.randn_like(module.lora_up.weight) * 0.1)
        module.apply_to()
        return base, module

    (o_new, g_new, gx_new), (o_old, g_old, gx_old) = _run_pair(
        make, lambda m, x: m.forward(x), _legacy_lora_forward
    )
    assert torch.equal(o_new, o_old), "LoRA forward != legacy fp32 path under autocast"
    _assert_grads_equal(g_new, g_old, "LoRA")
    assert torch.equal(gx_new, gx_old), "LoRA grad_x differs"


def test_lora_channel_scale_training_matches_legacy_under_autocast():
    from networks.lora_modules.lora import LoRAModule

    cs = _make_channel_scale(32)

    def make():
        torch.manual_seed(0)
        base = torch.nn.Linear(32, 24, bias=False).to(torch.bfloat16)
        base.weight.requires_grad_(False)
        module = LoRAModule(
            "m", base, multiplier=1.0, lora_dim=4, alpha=4, channel_scale=cs.clone()
        )
        with torch.no_grad():
            module.lora_up.weight.copy_(torch.randn_like(module.lora_up.weight) * 0.1)
        module.apply_to()
        return base, module

    # The legacy *default* path also routed through _rebalance, so even with
    # channel scaling this stays bitwise (only the retired custom Function's
    # fp32 fold differed by one rounding).
    (o_new, g_new, gx_new), (o_old, g_old, gx_old) = _run_pair(
        make, lambda m, x: m.forward(x), _legacy_lora_forward
    )
    assert torch.equal(o_new, o_old)
    _assert_grads_equal(g_new, g_old, "LoRA+channel_scale")
    assert torch.equal(gx_new, gx_old)


def test_hydra_training_matches_legacy_under_autocast():
    from networks.lora_modules.hydra import HydraLoRAModule

    def make():
        torch.manual_seed(0)
        base = torch.nn.Linear(32, 24, bias=False).to(torch.bfloat16)
        base.weight.requires_grad_(False)
        module = HydraLoRAModule(
            "h", base, multiplier=1.0, lora_dim=4, alpha=4, num_experts=3
        )
        with torch.no_grad():
            module.lora_up_weight.copy_(torch.randn_like(module.lora_up_weight) * 0.1)
        module.apply_to()
        return base, module

    (o_new, g_new, gx_new), (o_old, g_old, gx_old) = _run_pair(
        make, lambda m, x: m.forward(x), _legacy_hydra_forward
    )
    assert torch.equal(o_new, o_old), "Hydra forward != legacy under autocast"
    _assert_grads_equal(g_new, g_old, "Hydra")
    assert torch.equal(gx_new, gx_old)
    assert g_new["router.weight"].abs().sum() > 0, "router must receive gradient"


def test_training_forward_is_autocast_independent():
    """Dtype honesty: with bf16 inputs the new training forward computes the
    same thing with or without autocast — non-autocast callers (bespoke
    loops, tests) see train.py numerics."""
    from networks.lora_modules.lora import LoRAModule

    torch.manual_seed(0)
    base = torch.nn.Linear(32, 24, bias=False).to(torch.bfloat16)
    base.weight.requires_grad_(False)
    module = LoRAModule("m", base, multiplier=1.0, lora_dim=4, alpha=4)
    with torch.no_grad():
        module.lora_up.weight.copy_(torch.randn_like(module.lora_up.weight) * 0.1)
    module.apply_to()
    module.train()

    torch.manual_seed(1)
    x = torch.randn(2, 8, 32, dtype=torch.bfloat16)
    with _autocast():
        y_ac = module.forward(x)
    y_plain = module.forward(x)
    assert torch.equal(y_ac, y_plain)


def _fp32_activation_no_upcast(make, *, in_dim=32):
    """The AdaLN ``nn.LayerNorm`` feeding the adapted Linears emits fp32 under
    autocast(bf16), so ``x`` arrives fp32 in training. The forward must compute
    the rank path in the model's bf16 compute dtype (``org_forwarded.dtype``) —
    NOT key it off ``x.dtype`` and run fp32, which (with per-channel scaling on)
    allocated a full fp32 activation out of ``_rebalance`` and OOMed.

    Pin it: under autocast, feeding the fp32 activation must give the SAME result
    as feeding its bf16 rounding — i.e. the LoRA params (incl. the ``inv_scale``
    multiply, which autocast does NOT downcast) ran at bf16, not fp32. The
    retired ``x.dtype`` policy applied ``inv_scale`` in fp32 for the fp32 input
    and diverged."""
    base, module = make()
    module.train()
    torch.manual_seed(2)
    x_fp32 = torch.randn(2, 8, in_dim, dtype=torch.float32)
    x_bf16 = x_fp32.to(torch.bfloat16)  # same values, bf16 dtype
    with _autocast():
        y_from_fp32 = module.forward(x_fp32)
        y_from_bf16 = module.forward(x_bf16)
    assert torch.equal(y_from_fp32, y_from_bf16), (
        "fp32 activation diverges from its bf16 rounding — the rank path "
        "(inv_scale) ran in fp32 instead of the bf16 compute dtype (silent "
        "upcast regression)"
    )


def test_lora_fp32_activation_no_upcast():
    from networks.lora_modules.lora import LoRAModule

    cs = _make_channel_scale(32)  # inv_scale ON — the path that OOMed

    def make():
        torch.manual_seed(0)
        base = torch.nn.Linear(32, 24, bias=False).to(torch.bfloat16)
        base.weight.requires_grad_(False)
        module = LoRAModule(
            "m", base, multiplier=1.0, lora_dim=4, alpha=4, channel_scale=cs
        )
        with torch.no_grad():
            module.lora_up.weight.copy_(torch.randn_like(module.lora_up.weight) * 0.1)
        module.apply_to()
        return base, module

    _fp32_activation_no_upcast(make)


def test_hydra_fp32_activation_no_upcast():
    from networks.lora_modules.hydra import HydraLoRAModule

    cs = _make_channel_scale(32)  # inv_scale ON

    def make():
        torch.manual_seed(0)
        base = torch.nn.Linear(32, 24, bias=False).to(torch.bfloat16)
        module = HydraLoRAModule(
            "h",
            base,
            multiplier=1.0,
            lora_dim=4,
            alpha=4,
            num_experts=3,
            channel_scale=cs,
        )
        with torch.no_grad():
            module.lora_up_weight.copy_(torch.randn_like(module.lora_up_weight) * 0.1)
        module.apply_to()
        return base, module

    _fp32_activation_no_upcast(make)


def test_hydra_inference_keeps_fp32_compute():
    """The rewrite is training-only: at eval the hydra forward still computes
    in fp32 (router-live checkpoints run through the no-autocast inference
    engine and must produce unchanged outputs)."""
    from networks.lora_modules.hydra import HydraLoRAModule

    torch.manual_seed(0)
    base = torch.nn.Linear(32, 24, bias=False).to(torch.bfloat16)
    module = HydraLoRAModule(
        "h", base, multiplier=1.0, lora_dim=4, alpha=4, num_experts=3
    )
    with torch.no_grad():
        module.lora_up_weight.copy_(torch.randn_like(module.lora_up_weight) * 0.1)
    module.apply_to()
    module.eval()

    torch.manual_seed(1)
    x = torch.randn(2, 8, 32, dtype=torch.bfloat16)

    # fp32 reference: the historical eval compute, spelled out.
    org = module.org_forward(x)
    lx = F.linear(module._rebalance(x).float(), module.lora_down.weight.float())
    gate = module._compute_gate(lx)
    combined = torch.einsum("be,eod->bod", gate.float(), module.lora_up_weight.float())
    lx_3d = lx.reshape(2, -1, lx.shape[-1])
    out = torch.bmm(lx_3d, combined.transpose(1, 2)).reshape(2, 8, -1)
    ref = org + (out * module.multiplier * module.scale).to(org.dtype)

    assert torch.equal(module.forward(x), ref)


def test_lora_channel_scale_absorption_preserves_output():
    """SmoothQuant-style absorption: a channel-scaled module must produce the
    same delta as an unscaled twin (weights rebalanced, output unchanged)."""
    from networks.lora_modules.lora import LoRAModule

    def make(channel_scale):
        torch.manual_seed(0)
        base = torch.nn.Linear(32, 24, bias=False).to(torch.bfloat16)
        base.weight.requires_grad_(False)
        module = LoRAModule(
            "m",
            base,
            multiplier=1.0,
            lora_dim=4,
            alpha=4,
            channel_scale=channel_scale,
        )
        with torch.no_grad():
            module.lora_up.weight.copy_(torch.randn_like(module.lora_up.weight) * 0.1)
        module.apply_to()
        module.train()
        return base, module

    torch.manual_seed(1)
    x = torch.randn(2, 8, 32, dtype=torch.bfloat16)
    _, plain = make(None)
    _, scaled = make(_make_channel_scale(32))
    with _autocast():
        y_plain = plain.forward(x)
        y_scaled = scaled.forward(x)
    assert torch.allclose(
        y_plain.float(), y_scaled.float(), atol=_CS_ATOL, rtol=_CS_RTOL
    )


# ---------------------------------------------------------------------------
# Hydra σ-feature cache (flag-independent invariant, ported from the retired
# test_lora_custom_autograd.py).
# ---------------------------------------------------------------------------


def test_hydra_sigma_feature_cache_updates_and_clears():
    """Sigma-router features are precomputed once per step and cached on modules."""
    from networks.lora_modules.hydra import (
        HydraLoRAModule,
        _sigma_sinusoidal_features,
    )

    torch.manual_seed(0)
    base = torch.nn.Linear(32, 24, bias=False)
    module = HydraLoRAModule(
        "h",
        base,
        multiplier=1.0,
        lora_dim=4,
        alpha=4,
        num_experts=3,
        sigma_feature_dim=8,
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
