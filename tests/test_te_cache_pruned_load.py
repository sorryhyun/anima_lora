"""Regression: pruned (crossattn-only) TE caches must load regardless of the
run's ``cache_llm_adapter_outputs`` flag.

v1.12.1 started pruning the unused Qwen ``prompt_embeds`` tuple from
adapter-output caches (writing only ``crossattn_emb``). The loader still gated
the crossattn *read* on ``self.cache_llm_adapter_outputs``, so a run with the
flag off (or a preprocess/training flag mismatch) fell through to
``f.get_tensor("prompt_embeds_v2")`` and hard-crashed in safetensors:

    SafetensorError: File does not contain tensor prompt_embeds_v2

The downstream consumer (library/training/forward/text_conds.py) switches on
tuple shape, not the flag, so the loader must serve whatever the file holds.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


from library.anima.strategy import (
    AnimaTextEncoderOutputsCachingStrategy,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

S, D = 4, 8  # tiny seq / hidden dims


def _write_pruned_variant_cache(path: Path, num_variants: int = 3) -> None:
    """Adapter-output cache: crossattn_emb_v* + t5_attn_mask_v* only — the
    Qwen prompt_embeds / attn_mask / t5_input_ids are pruned."""
    save_dict = {
        "num_variants": torch.tensor(num_variants, dtype=torch.int64),
        "v0_intact": torch.tensor(1, dtype=torch.int8),
        "caption_dropout_rate": torch.tensor(0.1),
    }
    for vi in range(num_variants):
        save_dict[f"crossattn_emb_v{vi}"] = torch.randn(S, D, dtype=torch.float32)
        save_dict[f"t5_attn_mask_v{vi}"] = torch.ones(S, dtype=torch.int32)
    save_file(save_dict, str(path))


def _strategy(**kw) -> AnimaTextEncoderOutputsCachingStrategy:
    return AnimaTextEncoderOutputsCachingStrategy(
        cache_to_disk=True,
        batch_size=1,
        skip_disk_cache_validity_check=True,
        **kw,
    )


@pytest.mark.core
@pytest.mark.parametrize("flag", [False, True])
def test_pruned_cache_loads_regardless_of_flag(tmp_path, flag):
    """The exact crash: crossattn-only cache, shuffled variants, any suffix.

    With the flag off (the failing case) the loader must still serve
    crossattn_emb and report prompt_embeds as None — never read the pruned
    prompt_embeds_v* key.
    """
    cache = tmp_path / "x_anima_te.safetensors"
    _write_pruned_variant_cache(cache, num_variants=3)

    strat = _strategy(
        cache_llm_adapter_outputs=flag,
        use_shuffled_caption_variants=True,
    )

    # Hammer the random suffix selection so v0/v1/v2 are all exercised.
    for seed in range(25):
        random.seed(seed)
        out = strat.load_outputs_npz(str(cache))
        # 6-element list → [pe, am, t5i, t5m, crossattn, dropout_rate]
        assert len(out) == 6
        prompt_embeds, attn_mask, t5_input_ids, _t5m, crossattn, _dr = out
        assert crossattn is not None
        assert crossattn.shape == (S, D)
        # Pruned tensors must come back None, not crash.
        assert prompt_embeds is None
        assert attn_mask is None
        assert t5_input_ids is None


@pytest.mark.core
def test_plain_cache_still_loads_with_flag_on(tmp_path):
    """Reverse mismatch: a plain (prompt_embeds-only) cache read by a flag-on
    run must serve the Qwen tuple (4+1 shape), not demand crossattn."""
    cache = tmp_path / "y_anima_te.safetensors"
    save_dict = {
        "prompt_embeds": torch.randn(S, D, dtype=torch.float32),
        "attn_mask": torch.ones(S, dtype=torch.int32),
        "t5_input_ids": torch.ones(S, dtype=torch.int64),
        "t5_attn_mask": torch.ones(S, dtype=torch.int32),
        "caption_dropout_rate": torch.tensor(0.1),
    }
    save_file(save_dict, str(cache))

    strat = _strategy(cache_llm_adapter_outputs=True)
    out = strat.load_outputs_npz(str(cache))
    assert len(out) == 5  # plain path, no crossattn
    assert out[0] is not None  # prompt_embeds present
