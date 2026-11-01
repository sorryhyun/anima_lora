"""DiT depth/width is read off the checkpoint, not assumed.

``anima-base-v1.0`` and every official variant are 28 blocks wide 2048, but
community depth-expansions ship the same architecture at a different depth
(Anima-2.9B: 40 blocks, LLaMA-Pro-style interleaved block insertion). The model
class has always been parametric; the *loader* used to fall through to the 28
default, so a 40-block checkpoint failed with 12 blocks' worth of unexpected
keys. ``probe_dit_arch`` closes that by counting blocks in the safetensors
header.

The count must key off the top-level ``blocks.N.`` stack only — the LLM adapter
carries its own ``llm_adapter.blocks.0..5``, and letting those through would
inflate the depth and produce a model that no checkpoint fits.
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from library.anima.weights import _DIT_PREFIXES, probe_dit_arch


def _write_dit(path, *, num_blocks: int, width: int = 2048, prefix: str = "net."):
    """Minimal header-only stand-in: the keys probe_dit_arch actually reads."""
    sd = {f"{prefix}x_embedder.proj.1.weight": torch.zeros(width, 68)}
    for i in range(num_blocks):
        sd[f"{prefix}blocks.{i}.mlp.layer1.weight"] = torch.zeros(1, 1)
    # The 6-layer LLM adapter is present in every real Anima checkpoint.
    for i in range(6):
        sd[f"{prefix}llm_adapter.blocks.{i}.mlp.layer1.weight"] = torch.zeros(1, 1)
    save_file(sd, str(path))
    return path


@pytest.mark.core
@pytest.mark.parametrize("prefix", _DIT_PREFIXES)
@pytest.mark.parametrize("num_blocks", [28, 40])
def test_depth_is_counted_under_either_prefix(tmp_path, prefix, num_blocks):
    path = _write_dit(
        tmp_path / f"d{num_blocks}_{prefix.replace('.', '_')}.safetensors",
        num_blocks=num_blocks,
        prefix=prefix,
    )
    assert probe_dit_arch(str(path))["num_blocks"] == num_blocks


def test_width_maps_to_head_count(tmp_path):
    path = _write_dit(tmp_path / "w.safetensors", num_blocks=40, width=2048)
    arch = probe_dit_arch(str(path))
    assert arch == {"num_blocks": 40, "model_channels": 2048, "num_heads": 16}


def test_unknown_width_raises(tmp_path):
    """A width with no known head count would otherwise build a model whose
    head_dim silently differs from the checkpoint's."""
    path = _write_dit(tmp_path / "odd.safetensors", num_blocks=28, width=1536)
    with pytest.raises(RuntimeError, match="Unsupported DiT width"):
        probe_dit_arch(str(path))
