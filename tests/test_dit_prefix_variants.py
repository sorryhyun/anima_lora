"""Official Anima releases ship the same DiT under two state-dict prefixes.

``anima-base-v1.0`` uses ``net.``; every later release (aesthetic-v1.0/1.0b/1.1,
turbo-v1.0, preview3-base) uses ComfyUI's ``model.diffusion_model.``. The key
sets are otherwise identical (685 keys, matching shapes), so the *only* thing
standing between those checkpoints and LoRA training is prefix stripping — and
``load_anima_model`` raises on any unexpected key, so a missed prefix is a hard
load failure rather than a silent degradation.

These tests pin the strip where a miss is silent: the AdaLN fusion and
``load_llm_adapter``'s own prefix sniffing.
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from library.anima.weights import (
    _DIT_PREFIXES,
    _dit_concat_hook,
    _strip_net_prefix,
    load_llm_adapter,
)


def test_strip_leaves_unrelated_keys_alone():
    # Must not chew into a key that merely *contains* a prefix substring.
    assert _strip_net_prefix("x_embedder.net.weight") == "x_embedder.net.weight"


@pytest.mark.parametrize("prefix", _DIT_PREFIXES)
def test_adaln_down_fusion_concatenates_in_branch_order(prefix):
    """Phase 2 matches candidate tensors by stripped key — a prefix the strip
    doesn't know would silently produce a short/empty concat rather than raise,
    so assert the full 3-branch fusion actually happened, in order."""
    branches = ("self_attn", "cross_attn", "mlp")
    tensors = {
        f"{prefix}blocks.0.adaln_modulation_{b}.1.weight": torch.full((2, 4), float(i))
        for i, b in enumerate(branches)
    }
    key = f"{prefix}blocks.0.adaln_modulation_self_attn.1.weight"
    fused_key, fused = _dit_concat_hook(key, tensors)

    assert fused_key == "blocks.0.adaln_fused_down.1.weight"
    assert fused.shape == (6, 4)
    # Rows must be self_attn(0), cross_attn(1), mlp(2) — order is positional.
    assert [fused[i * 2, 0].item() for i in range(3)] == [0.0, 1.0, 2.0]


@pytest.mark.core
@pytest.mark.parametrize("prefix", _DIT_PREFIXES)
def test_load_llm_adapter_finds_weights_under_either_prefix(tmp_path, prefix):
    """load_llm_adapter sniffs prefixes independently of the DiT hooks. If it
    misses one it falls through to ``prefix=None`` and treats the whole DiT as
    adapter weights — no exception, just a silently untrained adapter."""
    path = tmp_path / f"dit_{prefix.replace('.', '_')}.safetensors"
    sd = {
        # A real adapter tensor, plus a non-adapter DiT key that must be ignored.
        f"{prefix}llm_adapter.blocks.0.cross_attn.k_norm.weight": torch.ones(64),
        f"{prefix}blocks.0.self_attn.q_proj.weight": torch.zeros(2048, 2048),
    }
    save_file(sd, str(path))

    adapter = load_llm_adapter(
        dit_path=str(path), dtype=torch.float32, device=torch.device("cpu")
    )

    loaded = adapter.blocks[0].cross_attn.k_norm.weight
    assert torch.allclose(loaded, torch.ones_like(loaded)), (
        "adapter weight was not loaded from the checkpoint — prefix sniffing "
        f"missed {prefix!r}"
    )
