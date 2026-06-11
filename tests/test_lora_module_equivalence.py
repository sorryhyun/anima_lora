"""Golden-tensor equivalence harness for the ``lora_modules/`` forwards (Part B0).

This is the safety net for the ``lora_modules/`` dedup refactor (proposal
``docs/proposal/programmatic_stacking_and_lora_module_dedup.md`` Part B). It
captures the *pre-refactor* forward (train + eval mode) and backward (grad wrt
input) of every variant module as golden tensors, checked in under
``tests/golden/``. Any later refactor that claims to be numerically inert
(B1's forward scaffold, B2's router mixin) must reproduce these bit-exactly —
``torch.equal``, not ``allclose``.

Determinism: ``tests/conftest.py`` forces ``CUDA_VISIBLE_DEVICES=""`` so the
SVD-based inits (ortho / hydra / chimera read ``torch.cuda.is_available()``)
run on CPU; builders also patch ``torch.cuda.is_available`` directly so the
standalone ``--write`` path below is CPU-deterministic regardless of how it is
invoked. Every builder re-seeds before construction, so ``svd_lowrank``'s
internal random projection is reproducible.

Regenerate the goldens (only when the *reference* forward legitimately changes —
NOT to paper over a refactor regression)::

    python tests/test_lora_module_equivalence.py --write

The goldens are tiny by construction (r ≤ 8, dim ≤ 64) — a few KB each.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

# Standalone ``--write`` invocation does not import conftest; force CPU before
# torch picks a device (no-op when pytest already set it).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def _force_cpu():
    """Pin the SVD-init device to CPU (mirrors ``test_lora_dtype_policy._cpu_only``)."""
    return mock.patch("torch.cuda.is_available", return_value=False)


# ---------------------------------------------------------------------------
# Shared builder helpers
# ---------------------------------------------------------------------------


def _linear_base(in_f=32, out_f=24, seed=0):
    torch.manual_seed(seed)
    base = torch.nn.Linear(in_f, out_f, bias=False).to(torch.bfloat16)
    base.weight.requires_grad_(False)
    return base


def _conv_base(in_c=8, out_c=12, seed=0):
    torch.manual_seed(seed)
    base = torch.nn.Conv2d(in_c, out_c, 3, padding=1, bias=False).to(torch.bfloat16)
    base.weight.requires_grad_(False)
    return base


def _set_mask(module, r):
    """Bind a non-trivial T-LoRA mask (alternating 1/0) so the gate path is
    actually exercised — the default all-ones mask is a no-op multiply."""
    mask = (torch.arange(r) % 2 == 0).to(torch.float32).unsqueeze(0)
    module._timestep_mask = mask


def _x_seq(in_dim=32, seed=100):
    torch.manual_seed(seed)
    return torch.randn(2, 8, in_dim, dtype=torch.bfloat16)


def _x_img(in_c=8, hw=12, seed=100):
    torch.manual_seed(seed)
    return torch.randn(2, in_c, hw, hw, dtype=torch.bfloat16)


def _channel_scale(in_features, seed=7):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(in_features, generator=g, dtype=torch.float32) * 2.0 + 0.5


# ---------------------------------------------------------------------------
# Per-variant builders. Each returns (base, module) with ΔW ≠ 0 (perturbed
# from the zero-init so the forward is a meaningful guard) and a non-trivial
# timestep mask bound where the variant honors one.
# ---------------------------------------------------------------------------


def _b_lora(channel_scale=None, conv=False):
    from networks.lora_modules.lora import LoRAModule

    base = _conv_base() if conv else _linear_base()
    with _force_cpu():
        module = LoRAModule(
            "m", base, multiplier=1.0, lora_dim=4, alpha=4, channel_scale=channel_scale
        )
    with torch.no_grad():
        module.lora_up.weight.copy_(torch.randn_like(module.lora_up.weight) * 0.1)
    module.apply_to()
    _set_mask(module, 4)
    return base, module


def _b_step_expert():
    from networks.lora_modules.step_expert import StepExpertLoRAModule

    base = _linear_base()
    with _force_cpu():
        module = StepExpertLoRAModule(
            "m", base, multiplier=1.0, lora_dim=4, alpha=4, step_expert_K=2
        )
    with torch.no_grad():
        for up in module.lora_ups:
            up.weight.copy_(torch.randn_like(up.weight) * 0.1)
    module.set_step(1)
    module.apply_to()
    _set_mask(module, 4)
    return base, module


def _b_ortho_init(channel_scale=None):
    from networks.lora_modules.ortho import OrthoInitLoRAModule

    base = _linear_base()
    with _force_cpu():
        module = OrthoInitLoRAModule(
            "m", base, multiplier=1.0, lora_dim=4, alpha=4, channel_scale=channel_scale
        )
    with torch.no_grad():
        module.lambda_layer.copy_(torch.randn_like(module.lambda_layer) * 0.3)
    module.apply_to()
    _set_mask(module, 4)
    return base, module


def _b_ortho(channel_scale=None):
    from networks.lora_modules.ortho import OrthoLoRAModule

    base = _linear_base()
    with _force_cpu():
        module = OrthoLoRAModule(
            "m", base, multiplier=1.0, lora_dim=4, alpha=4, channel_scale=channel_scale
        )
    with torch.no_grad():
        module.S_p.copy_(torch.randn_like(module.S_p) * 0.05)
        module.S_q.copy_(torch.randn_like(module.S_q) * 0.05)
        module.lambda_layer.copy_(torch.randn_like(module.lambda_layer) * 0.3)
    module.apply_to()
    _set_mask(module, 4)
    return base, module


def _b_ortho_hydra():
    from networks.lora_modules.ortho import OrthoHydraLoRAModule

    base = _linear_base()
    with _force_cpu():
        module = OrthoHydraLoRAModule(
            "m", base, multiplier=1.0, lora_dim=4, alpha=4, num_experts=3
        )
    with torch.no_grad():
        module.S_p.copy_(torch.randn_like(module.S_p) * 0.05)
        module.S_q.copy_(torch.randn_like(module.S_q) * 0.05)
        module.lambda_layer.copy_(torch.randn_like(module.lambda_layer) * 0.3)
    module.apply_to()
    _set_mask(module, 4)
    return base, module


def _b_hydra(channel_scale=None):
    from networks.lora_modules.hydra import HydraLoRAModule

    base = _linear_base()
    with _force_cpu():
        module = HydraLoRAModule(
            "m",
            base,
            multiplier=1.0,
            lora_dim=4,
            alpha=4,
            num_experts=3,
            channel_scale=channel_scale,
        )
    with torch.no_grad():
        module.lora_up_weight.copy_(torch.randn_like(module.lora_up_weight) * 0.1)
        # Break the router out of its near-uniform init so routing is exercised.
        module.router.weight.copy_(torch.randn_like(module.router.weight) * 0.1)
    module.apply_to()
    _set_mask(module, 4)
    return base, module


def _b_stacked(ortho=False):
    from networks.lora_modules.stacked_experts import StackedExpertsLoRAModule

    base = _linear_base()
    with _force_cpu():
        module = StackedExpertsLoRAModule(
            "m", base, multiplier=1.0, lora_dim=4, alpha=4, num_experts=3, ortho=ortho
        )
    with torch.no_grad():
        if ortho:
            module.lambda_layer.copy_(torch.randn_like(module.lambda_layer) * 0.3)
        else:
            module.lora_up_weight.copy_(torch.randn_like(module.lora_up_weight) * 0.1)
    module.apply_to()
    module.set_routing_weights(torch.tensor([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]]))
    _set_mask(module, 4)
    return base, module


def _b_chimera(use_ortho_init=False, channel_scale=None):
    from networks.lora_modules.chimera import ChimeraHydraLoRAModule

    base = _linear_base()
    with _force_cpu():
        module = ChimeraHydraLoRAModule(
            "m",
            base,
            multiplier=1.0,
            lora_dim=4,
            alpha=4,
            num_experts_content=3,
            num_experts_freq=2,
            lambda_init=0.1,
            channel_scale=channel_scale,
            use_ortho_init=use_ortho_init,
        )
    with torch.no_grad():
        module.lambda_c.copy_(torch.randn_like(module.lambda_c) * 0.3)
        module.lambda_f.copy_(torch.randn_like(module.lambda_f) * 0.3)
        if not use_ortho_init:
            module.S_p_c.copy_(torch.randn_like(module.S_p_c) * 0.05)
            module.S_p_f.copy_(torch.randn_like(module.S_p_f) * 0.05)
            module.S_q_c.copy_(torch.randn_like(module.S_q_c) * 0.05)
            module.S_q_f.copy_(torch.randn_like(module.S_q_f) * 0.05)
    module.apply_to()
    module.set_content_routing_weights(torch.tensor([[0.5, 0.3, 0.2]]))
    module.set_freq_routing_weights(torch.tensor([[0.6, 0.4]]))
    _set_mask(module, 4)
    return base, module


def _b_chimera_inf(channel_scale=None):
    from networks.lora_modules.chimera import ChimeraHydraInferenceModule

    base = _linear_base()
    with _force_cpu():
        module = ChimeraHydraInferenceModule(
            "m",
            base,
            multiplier=1.0,
            lora_dim=4,
            alpha=4,
            num_experts_content=3,
            num_experts_freq=2,
            channel_scale=channel_scale,
        )
    with torch.no_grad():
        module.lora_up_c_weight.copy_(torch.randn_like(module.lora_up_c_weight) * 0.1)
        module.lora_up_f_weight.copy_(torch.randn_like(module.lora_up_f_weight) * 0.1)
    module.apply_to()
    module.set_content_routing_weights(torch.tensor([[0.5, 0.3, 0.2]]))
    module.set_freq_routing_weights(torch.tensor([[0.6, 0.4]]))
    return base, module


# name → (builder, modes, x_factory). ``modes`` lists which forward modes to
# capture; conv LoRA is eval-only (the T-LoRA mask multiply is shaped for the
# Linear rank axis, never exercised on conv in production).
_VARIANTS = {
    "lora": (_b_lora, ("train", "eval"), _x_seq),
    "lora_channel_scale": (
        lambda: _b_lora(channel_scale=_channel_scale(32)),
        ("train", "eval"),
        _x_seq,
    ),
    "lora_conv2d": (lambda: _b_lora(conv=True), ("eval",), lambda: _x_img()),
    "step_expert": (_b_step_expert, ("train", "eval"), _x_seq),
    "ortho_init": (_b_ortho_init, ("train", "eval"), _x_seq),
    "ortho_init_channel_scale": (
        lambda: _b_ortho_init(channel_scale=_channel_scale(32)),
        ("train", "eval"),
        _x_seq,
    ),
    "ortho": (_b_ortho, ("train", "eval"), _x_seq),
    "ortho_channel_scale": (
        lambda: _b_ortho(channel_scale=_channel_scale(32)),
        ("train", "eval"),
        _x_seq,
    ),
    "ortho_hydra": (_b_ortho_hydra, ("train", "eval"), _x_seq),
    "hydra": (_b_hydra, ("train", "eval"), _x_seq),
    "hydra_channel_scale": (
        lambda: _b_hydra(channel_scale=_channel_scale(32)),
        ("train", "eval"),
        _x_seq,
    ),
    "stacked_free": (lambda: _b_stacked(ortho=False), ("train", "eval"), _x_seq),
    "stacked_ortho": (lambda: _b_stacked(ortho=True), ("train", "eval"), _x_seq),
    "chimera_frozen": (
        lambda: _b_chimera(use_ortho_init=False),
        ("train", "eval"),
        _x_seq,
    ),
    "chimera_ortho_init": (
        lambda: _b_chimera(use_ortho_init=True),
        ("train", "eval"),
        _x_seq,
    ),
    "chimera_inference": (_b_chimera_inf, ("eval",), _x_seq),
}

# Variants whose eval forward calls ``nn.Linear`` submodules directly (no
# internal dtype cast), so the adapter params must be bf16 to match the base.
_EVAL_BF16 = {"lora", "lora_channel_scale", "lora_conv2d", "step_expert"}


def _capture(name: str) -> dict:
    """Run every captured mode for ``name`` and return a tensor dict.

    For train mode the input carries grad so the backward path (grad wrt x) is
    pinned alongside the forward output.
    """
    builder, modes, x_factory = _VARIANTS[name]
    out: dict[str, torch.Tensor] = {}
    for mode in modes:
        _, module = builder()
        x = x_factory()
        if mode == "train":
            # Training keeps the fp32 master params (the dtype-policy contract);
            # the forward casts to the bf16 base compute dtype internally, so no
            # autocast is needed for a faithful train capture.
            module.train()
            x = x.clone().requires_grad_(True)
            y = module.forward(x)
            y.float().sum().backward()
            out["train_out"] = y.detach().clone()
            out["train_grad_x"] = x.grad.detach().clone()
        else:
            module.eval()
            # The raw-nn.Linear eval paths (LoRA / step-expert) require the
            # adapter params to match the bf16 base + activations — inference
            # loads them in the model dtype. The other variants' eval forwards
            # cast internally (and keep fp32 buffers like ``_eye_r`` the Cayley
            # solve needs), so a blanket ``.to(bf16)`` would break them.
            if name in _EVAL_BF16:
                module.to(torch.bfloat16)
            with torch.no_grad():
                y = module.forward(x)
            out["eval_out"] = y.detach().clone()
    return out


def _golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.pt"


def _write_goldens():
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for name in _VARIANTS:
        torch.save(_capture(name), _golden_path(name))
        print(f"wrote {_golden_path(name).relative_to(GOLDEN_DIR.parent.parent)}")


@pytest.mark.parametrize("name", list(_VARIANTS))
def test_module_forward_matches_golden(name):
    path = _golden_path(name)
    if not path.exists():
        pytest.skip(f"golden missing ({path.name}); run `python {__file__} --write`")
    golden = torch.load(path)
    captured = _capture(name)
    assert captured.keys() == golden.keys(), (
        f"{name}: captured tensors {set(captured)} != golden {set(golden)}"
    )
    for key, ref in golden.items():
        got = captured[key]
        assert got.shape == ref.shape, f"{name}/{key}: shape {got.shape} != {ref.shape}"
        assert got.dtype == ref.dtype, f"{name}/{key}: dtype {got.dtype} != {ref.dtype}"
        assert torch.equal(got, ref), (
            f"{name}/{key}: forward/backward diverged from golden "
            f"(max abs diff {(got.float() - ref.float()).abs().max().item():.3e})"
        )


if __name__ == "__main__":
    if "--write" in sys.argv:
        _write_goldens()
    else:
        print("pass --write to (re)generate goldens under tests/golden/")
