"""Static-merge path (``load_safetensors_with_lora``) must apply comfy-layout
adaln LoRA keys — the layout every shipped ``train_adaln`` checkpoint carries.

Regression for 2026-09-02: the merge hook sees the DiT's runtime key names
(``adaln_up_{br}``, post ``_dit_rename_hook``) while the file carries
``adaln_modulation_{br}_2``; without the rename the adaln rows were silently
skipped (only a "not all LoRA keys are used" warning) on ``--lora_weight``.
"""

from __future__ import annotations

import torch
from safetensors.torch import save_file

from library.anima import weights as anima_weights
from networks.lora_utils import load_safetensors_with_lora


def _merge(tmp_path, lora_sd):
    base = torch.randn(8, 4)
    save_file(
        {"net.blocks.0.adaln_modulation_mlp.2.weight": base},
        str(tmp_path / "dit.safetensors"),
    )
    hooks = anima_weights.WeightTransformHooks(
        rename_hook=anima_weights._dit_rename_hook,
        concat_hook=anima_weights._dit_concat_hook,
    )
    sd = load_safetensors_with_lora(
        model_files=str(tmp_path / "dit.safetensors"),
        lora_weights_list=[lora_sd],
        lora_multipliers=[1.0],
        calc_device=torch.device("cpu"),
        weight_transform_hooks=hooks,
    )
    return base, sd["blocks.0.adaln_up_mlp.weight"]


def _lora(prefix):
    down, up = torch.randn(2, 4), torch.randn(8, 2)
    return {
        f"{prefix}.lora_down.weight": down,
        f"{prefix}.lora_up.weight": up,
        f"{prefix}.alpha": torch.tensor(2.0),
    }, up @ down


def test_comfy_layout_adaln_keys_merge(tmp_path):
    lora_sd, delta = _lora("lora_unet_blocks_0_adaln_modulation_mlp_2")
    base, merged = _merge(tmp_path, lora_sd)
    torch.testing.assert_close(merged, base + delta)


def test_runtime_layout_adaln_keys_still_merge(tmp_path):
    lora_sd, delta = _lora("lora_unet_blocks_0_adaln_up_mlp")
    base, merged = _merge(tmp_path, lora_sd)
    torch.testing.assert_close(merged, base + delta)
