"""Why a LoRA renders differently on ComfyUI's split and fused int8 MOSS-Image: both
against the bf16 model with the same LoRA.

    make daemon-run ARGS="--stall-timeout 300 project/int8_train/src/comfy_qkv_lora_check.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from comfy_qkv_check import GPTQ, LORA, predict, toolkit  # noqa: E402

import comfy.model_management  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def state(patcher):
    dm = patcher.model.diffusion_model
    proj = dm.layers[0].proj
    name = "qkv" if "qkv" in proj else "q"
    m = proj[name]
    return {
        "linear": name,
        "class": type(m).__name__,
        "weight_function": len(m.weight_function),
        "weight": type(m.weight).__name__,
        "scale": getattr(
            getattr(m.weight, "_params", None), "scale", torch.zeros(1)
        ).shape,
        "patches": len(patcher.patches),
    }


def main() -> None:
    loader, _stream = toolkit()
    device = comfy.model_management.get_torch_device()
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(1, 128, 64, 64, generator=gen).to(device)
    ctx = torch.randn(1, 300, 4096, generator=gen).to(device)
    t = torch.tensor([0.5], device=device)
    lora = comfy.utils.load_torch_file(LORA, safe_load=True)

    models = {
        "split": comfy.sd.load_diffusion_model(GPTQ),
        "fused": loader.load_file_dit(GPTQ, {}),
        "bf16": loader.load_hf_dit(
            str(Path(GPTQ).parent / "MOSS-Image-Base"), "default"
        ),
    }
    plain, lora_out = {}, {}
    for name, patcher in models.items():
        plain[name] = predict(patcher, x, t, ctx)
        comfy.model_management.unload_all_models()
        with_lora, _ = comfy.sd.load_lora_for_models(patcher, None, lora, 1.0, 0.0)
        comfy.model_management.load_models_gpu([with_lora])
        print(name, state(with_lora), flush=True)
        lora_out[name] = predict(with_lora, x, t, ctx)
        comfy.model_management.unload_all_models()
    print(
        "no LoRA, against bf16:",
        {n: round(rel(plain[n], plain["bf16"]), 5) for n in plain},
    )
    print(
        "LoRA, against bf16:",
        {n: round(rel(lora_out[n], lora_out["bf16"]), 5) for n in lora_out},
    )
    print("LoRA's own move:", {n: round(rel(lora_out[n], plain[n]), 5) for n in plain})
    print("fused vs split, LoRA:", round(rel(lora_out["fused"], lora_out["split"]), 5))


def seed_floor() -> None:
    """The same split model twice, the requantization's stochastic rounding reseeded:
    the distance a different key alone gives."""
    device = comfy.model_management.get_torch_device()
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(1, 128, 64, 64, generator=gen).to(device)
    ctx = torch.randn(1, 300, 4096, generator=gen).to(device)
    t = torch.tensor([0.5], device=device)
    lora = comfy.utils.load_torch_file(LORA, safe_load=True)
    patcher = comfy.sd.load_diffusion_model(GPTQ)
    seed = comfy.utils.string_to_seed
    outs = []
    for offset in (0, 1):
        comfy.utils.string_to_seed = lambda key: seed(key) + offset
        with_lora, _ = comfy.sd.load_lora_for_models(patcher, None, lora, 1.0, 0.0)
        outs.append(predict(with_lora, x, t, ctx))
        comfy.model_management.unload_all_models()
    comfy.utils.string_to_seed = seed
    print("split vs split, another rounding seed:", round(rel(outs[1], outs[0]), 5))


if __name__ == "__main__":
    seed_floor() if "--seed_floor" in sys.argv else main()
