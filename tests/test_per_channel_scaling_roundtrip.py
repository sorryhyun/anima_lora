"""Round-trip test for the plain-LoRA per_channel_scaling save/load path.

Regression for the silent-drop bug where ``defuse_standard_qkv`` (save side)
and ``_refuse_unfused_attn_lora_keys`` (load side) ignored ``inv_scale`` on
fused qkv/kv modules. With channel scaling on, the absorbed ``lora_down``
columns ship to disk while ``inv_scale`` is dropped — reload then treats
``lora_down`` as un-absorbed and silently produces wrong output.

Coverage:

* End-to-end: build a real ``LoRAModule`` with ``channel_scale``, run defuse,
  refuse, and verify the rebuilt fused module produces identical output to the
  in-memory original.
* Bake: the production ``defuse_and_bake_standard`` path now folds
  ``inv_scale`` into ``lora_down`` and drops the key, so a plain (no
  channel-scale) consumer reproduces the channel-scaled forward bitwise.
"""

from __future__ import annotations

import pytest
import torch

from networks.lora_anima.loading import _refuse_unfused_attn_lora_keys
from networks.lora_modules.lora import (
    LoRAModule,
    defuse_and_bake_standard,
    defuse_standard_qkv,
)


def _make_calibration(in_dim: int, seed: int = 0) -> torch.Tensor:
    """Synthetic mean_abs vector with a single dominant channel.

    Matches the real bench: most channels small, one outlier that dominates.
    """
    torch.manual_seed(seed)
    stats = torch.rand(in_dim, dtype=torch.float32) * 0.3 + 0.05
    stats[7] = 50.0  # the dominant channel
    return stats


def test_end_to_end_roundtrip_preserves_forward():
    """Build a real LoRAModule with channel_scale, run save defuse + load refuse,
    rebuild a fused LoRAModule, confirm forward output matches the original."""
    in_dim = 32
    out_dim_per_comp = 16
    n = 3  # qkv
    out_dim = out_dim_per_comp * n
    rank = 4

    torch.manual_seed(42)
    base = torch.nn.Linear(in_dim, out_dim, bias=False)
    calibration = _make_calibration(in_dim)

    lora = LoRAModule(
        "lora_unet_blocks_0_self_attn_qkv_proj",
        base,
        multiplier=1.0,
        lora_dim=rank,
        alpha=rank,
        channel_scale=calibration,
    )
    # Mimic the post-apply training state: nonzero lora_up so the adapter
    # actually contributes (zero-init would make the test trivially pass).
    torch.nn.init.normal_(lora.lora_up.weight, std=0.05)
    lora.eval()

    # Capture the original adapter contribution on a fresh input.
    x = torch.randn(2, 7, in_dim) * 4.0
    x[..., 7] += 30.0  # exercise the dominant channel
    with torch.no_grad():
        lora_only_orig = lora.lora_up(lora.lora_down(lora._rebalance(x))) * lora.scale

    sd = {f"{lora.lora_name}.{k}": v for k, v in lora.state_dict().items()}
    assert f"{lora.lora_name}.inv_scale" in sd, (
        "channel-scaled LoRAModule must register inv_scale as a persistent buffer"
    )
    on_disk_inv = sd[f"{lora.lora_name}.inv_scale"].clone()

    # 1. Save defuse → on-disk split layout.
    defuse_standard_qkv(sd)
    for letter in ("q", "k", "v"):
        assert f"lora_unet_blocks_0_self_attn_{letter}_proj.inv_scale" in sd, (
            "round-trip lost inv_scale on save"
        )

    # 2. Load refuse → back to fused layout.
    _refuse_unfused_attn_lora_keys(sd)
    fused_prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    assert f"{fused_prefix}.inv_scale" in sd
    assert torch.equal(sd[f"{fused_prefix}.inv_scale"], on_disk_inv)

    # 3. Rebuild a fresh LoRAModule and load the round-tripped state_dict.
    base_rebuilt = torch.nn.Linear(in_dim, out_dim, bias=False)
    # Use the same base weight so the org_forward contribution matches.
    base_rebuilt.load_state_dict(base.state_dict())
    lora_rebuilt = LoRAModule(
        "lora_unet_blocks_0_self_attn_qkv_proj",
        base_rebuilt,
        multiplier=1.0,
        lora_dim=rank,
        alpha=rank,
        channel_scale=calibration,
    )
    rebuilt_sd = {
        k[len(fused_prefix) + 1 :]: v
        for k, v in sd.items()
        if k.startswith(fused_prefix + ".")
    }
    missing, unexpected = lora_rebuilt.load_state_dict(rebuilt_sd, strict=False)
    assert not unexpected, f"unexpected keys after round-trip: {unexpected}"
    lora_rebuilt.eval()

    with torch.no_grad():
        lora_only_new = (
            lora_rebuilt.lora_up(lora_rebuilt.lora_down(lora_rebuilt._rebalance(x)))
            * lora_rebuilt.scale
        )

    # The pre-fused detection in _refuse_unfused_attn_lora_keys collapses
    # identical-down splits back into the original fused layout, so output
    # must match bitwise after the rebuild (fp32 throughout).
    assert torch.allclose(lora_only_new, lora_only_orig, atol=1e-6), (
        "round-tripped LoRA adapter output diverged from the original"
    )


@pytest.mark.core
def test_baked_save_reproduces_channel_scaled_forward():
    """The production save path (defuse_and_bake_standard) bakes
    inv_scale, so a *plain* LoRAModule (no channel_scale) loading the baked
    weights reproduces the original channel-scaled forward bitwise — this is
    what stock ComfyUI / merge_to_dit now apply correctly."""
    in_dim = 32
    out_dim_per_comp = 16
    n = 3
    out_dim = out_dim_per_comp * n
    rank = 4

    torch.manual_seed(7)
    base = torch.nn.Linear(in_dim, out_dim, bias=False)
    calibration = _make_calibration(in_dim)

    lora = LoRAModule(
        "lora_unet_blocks_0_self_attn_qkv_proj",
        base,
        multiplier=1.0,
        lora_dim=rank,
        alpha=rank,
        channel_scale=calibration,
    )
    torch.nn.init.normal_(lora.lora_up.weight, std=0.05)
    lora.eval()

    x = torch.randn(2, 7, in_dim) * 4.0
    x[..., 7] += 30.0
    with torch.no_grad():
        orig = lora.lora_up(lora.lora_down(lora._rebalance(x))) * lora.scale

    # Production save: defuse → bake. After this there are NO inv_scale keys.
    sd = {f"{lora.lora_name}.{k}": v for k, v in lora.state_dict().items()}
    defuse_and_bake_standard(sd)
    assert not any(k.endswith(".inv_scale") for k in sd), (
        "baked save must not ship inv_scale keys"
    )

    # Load refuse (re-fuse split q/k/v) and rebuild a PLAIN module (channel
    # scaling off — the absence of inv_scale keys is the inference signal).
    _refuse_unfused_attn_lora_keys(sd)
    fused = "lora_unet_blocks_0_self_attn_qkv_proj"
    base_rebuilt = torch.nn.Linear(in_dim, out_dim, bias=False)
    base_rebuilt.load_state_dict(base.state_dict())
    plain = LoRAModule(fused, base_rebuilt, multiplier=1.0, lora_dim=rank, alpha=rank)
    rebuilt_sd = {
        k[len(fused) + 1 :]: v for k, v in sd.items() if k.startswith(fused + ".")
    }
    _, unexpected = plain.load_state_dict(rebuilt_sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    plain.eval()
    assert not plain._has_channel_scale

    with torch.no_grad():
        new = plain.lora_up(plain.lora_down(plain._rebalance(x))) * plain.scale

    assert torch.allclose(new, orig, atol=1e-6), (
        "baked plain forward diverged from the channel-scaled original"
    )
