"""Step-expert LoRA (turbo per-step head split) invariants.

Covers:
  * zero-init up-heads → ΔW = 0 at init (base preserved).
  * hard step selection: ``set_step(k)`` routes through ``lora_ups[k]`` only;
    a perturbation to head k is visible at step k and invisible at step j != k.
  * ``LoRANetwork.set_step_index`` broadcasts to every adapted module and
    no-ops on non-step-expert modules.
  * save → load roundtrip preserves per-step outputs (fused-key layout, the
    bespoke kept-live format written by ``TurboDMDNetwork.save_student``).
"""

from __future__ import annotations

import torch

from networks.lora_modules import LoRAModule, StepExpertLoRAModule


def _make_module(in_dim=8, out_dim=16, r=4, K=2) -> StepExpertLoRAModule:
    lin = torch.nn.Linear(in_dim, out_dim, bias=False)
    m = StepExpertLoRAModule("lora_unet_test", lin, 1.0, r, r, step_expert_K=K)
    m.apply_to()  # reroutes lin.forward → m.forward, stashes org_forward
    m.eval()
    return m


def test_zero_init_delta_is_zero():
    m = _make_module()
    x = torch.randn(3, 8)
    base = m.org_forward(x)
    for k in range(m.K):
        m.set_step(k)
        assert torch.allclose(m.forward(x), base), f"head {k} not zero at init"


def test_hard_step_selection_isolates_heads():
    m = _make_module()
    x = torch.randn(3, 8)
    # Perturb only head 1's up-weight.
    with torch.no_grad():
        m.lora_ups[1].weight.add_(torch.randn_like(m.lora_ups[1].weight))
    base = m.org_forward(x)
    m.set_step(0)
    assert torch.allclose(m.forward(x), base), "head 0 changed by head-1 perturbation"
    m.set_step(1)
    assert not torch.allclose(m.forward(x), base), "head 1 perturbation not visible"


def test_set_step_index_broadcasts_and_skips_plain():
    from networks.lora_anima.network import LoRANetwork

    net = LoRANetwork.__new__(LoRANetwork)
    torch.nn.Module.__init__(net)
    se = [_make_module(), _make_module(K=2)]
    plain_lin = torch.nn.Linear(8, 16, bias=False)
    plain = LoRAModule("lora_unet_plain", plain_lin, 1.0, 4, 4)
    net.unet_loras = [*se, plain]
    net.text_encoder_loras = []

    net.set_step_index(1)
    assert all(m._step == 1 for m in se)
    # Plain LoRA has no set_step → silently skipped (no crash, no attr added).
    assert not hasattr(plain, "_step")


def test_save_load_roundtrip_preserves_per_step():
    """Bake the multi-head fused-key layout and rebuild it; outputs must match."""
    from networks.lora_modules.lora import bake_inv_scale

    m = _make_module(K=2)
    # Give both heads + the shared down distinct nonzero weights.
    with torch.no_grad():
        m.lora_down.weight.copy_(torch.randn_like(m.lora_down.weight))
        for up in m.lora_ups:
            up.weight.copy_(torch.randn_like(up.weight))

    x = torch.randn(3, 8)
    outs = {}
    for k in range(m.K):
        m.set_step(k)
        outs[k] = m.forward(x).clone()

    # Emit the on-disk layout TurboDMDNetwork.save_student writes (prefix + the
    # module's own state_dict keys), bake inv_scale (no-op here), then rebuild.
    sd = {f"lora_unet_test.{kk}": vv for kk, vv in m.state_dict().items()}
    sd = {kk: vv for kk, vv in sd.items() if ".lora_" in kk or ".alpha" in kk}
    bake_inv_scale(sd)  # must not corrupt the multi-head layout

    lin2 = torch.nn.Linear(8, 16, bias=False)
    lin2.load_state_dict({"weight": m.org_forward.__self__.weight})  # share base
    m2 = StepExpertLoRAModule("lora_unet_test", lin2, 1.0, 4, 4, step_expert_K=2)
    m2.apply_to()
    m2.eval()
    info = m2.load_state_dict(
        {kk[len("lora_unet_test.") :]: vv for kk, vv in sd.items()}, strict=False
    )
    assert not info.unexpected_keys, info.unexpected_keys
    for k in range(m2.K):
        m2.set_step(k)
        assert torch.allclose(m2.forward(x), outs[k], atol=1e-5), f"step {k} mismatch"
