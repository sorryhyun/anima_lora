"""Concat/rename hook coverage for TensorWeightAdapter (library/io/safetensors.py).

Every DiT checkpoint load (library/anima/weights.py::load_anima_model) runs
through _dit_rename_hook + _dit_concat_hook via TensorWeightAdapter, so the
fused-key enumeration and the concat ordering are load-bearing: q/k/v must fuse
in (q, k, v) order and the AdaLN down-projections in (self_attn, cross_attn,
mlp) order regardless of the on-disk key order.
"""

import pytest
import torch
from safetensors.torch import save_file

from library.anima.weights import _dit_concat_hook, _dit_rename_hook
from library.io.safetensors import (
    MemoryEfficientSafeOpen,
    TensorWeightAdapter,
    WeightTransformHooks,
)


def _const(v):
    return torch.full((4, 8), float(v))


# Source keys deliberately out of fusion order — ordering must come from the
# hook's name lookup, not from on-disk key order.
_SOURCE = {
    "net.blocks.0.self_attn.v_proj.weight": _const(3),
    "net.blocks.0.self_attn.q_proj.weight": _const(1),
    "net.blocks.0.self_attn.k_proj.weight": _const(2),
    "net.blocks.0.cross_attn.v_proj.weight": _const(5),
    "net.blocks.0.cross_attn.k_proj.weight": _const(4),
    "net.blocks.0.adaln_modulation_mlp.1.weight": _const(8),
    "net.blocks.0.adaln_modulation_self_attn.1.weight": _const(6),
    "net.blocks.0.adaln_modulation_cross_attn.1.weight": _const(7),
    # rename-only paths (no fusion)
    "net.blocks.0.adaln_modulation_self_attn.2.weight": _const(9),
    "net.blocks.0.mlp.fc1.weight": _const(10),
}

pytestmark = pytest.mark.core

_DIT_HOOKS = WeightTransformHooks(
    rename_hook=_dit_rename_hook,
    concat_hook=_dit_concat_hook,
)


def test_dit_concat_and_rename_hooks(tmp_path):
    path = tmp_path / "dit.safetensors"
    save_file(_SOURCE, str(path))

    with MemoryEfficientSafeOpen(str(path)) as f:
        adapter = TensorWeightAdapter(_DIT_HOOKS, f)

        assert set(adapter.keys()) == {
            "blocks.0.self_attn.qkv_proj.weight",
            "blocks.0.cross_attn.kv_proj.weight",
            "blocks.0.adaln_fused_down.1.weight",
            "blocks.0.adaln_up_self_attn.weight",
            "blocks.0.mlp.fc1.weight",
        }

        qkv = adapter.get_tensor("blocks.0.self_attn.qkv_proj.weight")
        assert torch.equal(qkv, torch.cat([_const(1), _const(2), _const(3)], dim=0))

        kv = adapter.get_tensor("blocks.0.cross_attn.kv_proj.weight")
        assert torch.equal(kv, torch.cat([_const(4), _const(5)], dim=0))

        adaln = adapter.get_tensor("blocks.0.adaln_fused_down.1.weight")
        assert torch.equal(adaln, torch.cat([_const(6), _const(7), _const(8)], dim=0))

        assert torch.equal(
            adapter.get_tensor("blocks.0.adaln_up_self_attn.weight"), _const(9)
        )
        assert torch.equal(adapter.get_tensor("blocks.0.mlp.fc1.weight"), _const(10))
