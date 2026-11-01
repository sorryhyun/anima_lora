"""Warm-start round-trip: extracted delta file → student LoRA init.

Covers the two pure pieces of the ``student_init_weights`` path
(networks/methods/turbo_dmd.py):

* ``_plain_lora_module_delta`` — re-fusing defused q/k/v components must
  reconstruct the exact fused ΔW that ``defuse_standard_qkv`` split at save.
* ``_warm_start_module`` — writing a delta into a plain LoRAModule (fixed
  scale = alpha/rank) must make the module's applied delta equal the
  rank-truncated ΔW.
"""

import pytest
import torch

from networks.lora_modules.lora import LoRAModule, defuse_standard_qkv
from networks.methods.turbo_dmd import _plain_lora_module_delta, _warm_start_module

OUT, IN, RANK = 24, 16, 4


def _fused_file_sd(lora_name: str, up: torch.Tensor, down: torch.Tensor, alpha: float):
    sd = {
        f"{lora_name}.lora_up.weight": up.clone(),
        f"{lora_name}.lora_down.weight": down.clone(),
        f"{lora_name}.alpha": torch.tensor(alpha),
    }
    defuse_standard_qkv(sd)
    return sd


@pytest.mark.core
def test_module_delta_refuses_qkv_split_exactly():
    torch.manual_seed(0)
    name = "lora_unet_blocks_0_self_attn_qkv_proj"
    up = torch.randn(3 * OUT, RANK)
    down = torch.randn(RANK, IN)
    alpha = 2.0
    sd = _fused_file_sd(name, up, down, alpha)
    # save-side really split it (q/k/v keys, fused key gone)
    assert f"{name}.lora_up.weight" not in sd
    assert "lora_unet_blocks_0_self_attn_q_proj.lora_up.weight" in sd

    A, B = _plain_lora_module_delta(sd, name)
    expected = (alpha / RANK) * up @ down
    assert torch.allclose(A @ B, expected, atol=1e-5)


@pytest.mark.core
def test_warm_start_module_applies_exact_truncated_delta():
    torch.manual_seed(1)
    org = torch.nn.Linear(IN, OUT, bias=False)
    module = LoRAModule("lora_unet_blocks_0_mlp_layer1", org, lora_dim=RANK, alpha=2.5)

    # Rank-RANK delta must round-trip exactly despite the non-1 module scale.
    fa, fb = torch.randn(OUT, RANK), torch.randn(RANK, IN)
    energy = _warm_start_module(module, fa, fb)
    applied = module.scale * module.lora_up.weight @ module.lora_down.weight
    assert torch.allclose(applied, fa @ fb, atol=1e-4)
    assert energy > 0.9999

    # Full-rank delta (factored as ΔW @ I): applied == best rank-RANK
    # approximation (SVD truncation).
    full = torch.randn(OUT, IN)
    energy = _warm_start_module(module, full, torch.eye(IN))
    U, S, Vh = torch.linalg.svd(full, full_matrices=False)
    best = U[:, :RANK] @ torch.diag(S[:RANK]) @ Vh[:RANK]
    applied = module.scale * module.lora_up.weight @ module.lora_down.weight
    assert torch.allclose(applied, best, atol=1e-4)
    assert 0.0 < energy < 1.0


def test_warm_start_module_file_rank_below_module_rank():
    # File rank 2 < module rank 4: top-2 filled, remaining rank stays zero.
    torch.manual_seed(2)
    org = torch.nn.Linear(IN, OUT, bias=False)
    module = LoRAModule("lora_unet_blocks_0_mlp_layer1", org, lora_dim=RANK, alpha=2.5)
    fa, fb = torch.randn(OUT, 2), torch.randn(2, IN)
    energy = _warm_start_module(module, fa, fb)
    applied = module.scale * module.lora_up.weight @ module.lora_down.weight
    assert torch.allclose(applied, fa @ fb, atol=1e-4)
    assert energy > 0.9999
