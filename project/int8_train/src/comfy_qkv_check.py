"""ComfyUI's MOSS-Image on the GPTQ file, split and fused by the MOSS Toolkit's loader:
same output, with and without a LoRA and on the fp32 stream, and the forward time.

    make daemon-run ARGS="--stall-timeout 300 project/int8_train/src/comfy_qkv_check.py"
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
import time
from pathlib import Path

import torch

from library.env import resolve_under_home

COMFY = Path(resolve_under_home("../comfy")).resolve()
TOOLKIT = "comfyui-moss-toolkit"
sys.path.insert(0, str(COMFY))

import comfy.model_management  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402

GPTQ = str(
    resolve_under_home("models/moss_image/moss_image_base_int8_gptq.safetensors")
)
LORA = str(resolve_under_home("output/ckpt_moss/sincos_lay08p02_lr5e5_s1.safetensors"))


def toolkit():
    spec = importlib.util.spec_from_file_location(
        "moss_toolkit",
        COMFY / "custom_nodes" / TOOLKIT / "__init__.py",
        submodule_search_locations=[str(COMFY / "custom_nodes" / TOOLKIT)],
    )
    pack = importlib.util.module_from_spec(spec)
    sys.modules["moss_toolkit"] = pack
    spec.loader.exec_module(pack)
    return sys.modules["moss_toolkit.loader"], sys.modules["moss_toolkit.stream"]


def predict(patcher, x, t, ctx):
    comfy.model_management.load_models_gpu([patcher])
    with torch.inference_mode():
        return patcher.model.apply_model(x, t, c_crossattn=ctx).float().cpu()


def timed(patcher, x, t, ctx, n=12):
    comfy.model_management.load_models_gpu([patcher])
    out = []
    with torch.inference_mode():
        for _ in range(n):
            torch.cuda.synchronize()
            start = time.perf_counter()
            patcher.model.apply_model(x, t, c_crossattn=ctx)
            torch.cuda.synchronize()
            out.append((time.perf_counter() - start) * 1e3)
    return statistics.median(out[2:])


def main() -> None:
    loader, stream = toolkit()
    device = comfy.model_management.get_torch_device()
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(1, 128, 64, 64, generator=gen).to(device)
    ctx = torch.randn(1, 300, 4096, generator=gen).to(device)
    t = torch.tensor([0.5], device=device)

    split = comfy.sd.load_diffusion_model(GPTQ)
    fused = loader.load_file_dit(GPTQ, {})
    proj = fused.get_model_object("diffusion_model").layers[0].proj
    assert "qkv" in proj and "q" not in proj
    print(
        "kernels:", proj["qkv"].quant_format, proj["qkv"]._full_precision_mm, flush=True
    )

    outs = {}
    for name, patcher in (("split", split), ("fused", fused)):
        outs[name] = predict(patcher, x, t, ctx)
        comfy.model_management.unload_all_models()
    print("fused == split:", torch.equal(outs["split"], outs["fused"]), flush=True)

    ms = {"split": [], "fused": []}
    for _ in range(3):  # alternating, for the card's drift
        for name, patcher in (("split", split), ("fused", fused)):
            ms[name].append(timed(patcher, x, t, ctx))
            comfy.model_management.unload_all_models()
    print("forward ms:", ms, flush=True)

    lora = comfy.utils.load_torch_file(LORA, safe_load=True)
    patched = {}
    for name, patcher in (("split", split), ("fused", fused)):
        with_lora, _ = comfy.sd.load_lora_for_models(patcher, None, lora, 1.0, 0.0)
        patched[name] = predict(with_lora, x, t, ctx)
        comfy.model_management.unload_all_models()
    d = (patched["fused"] - patched["split"]).norm() / patched["split"].norm()
    moved = (patched["fused"] - outs["fused"]).norm() / outs["fused"].norm()
    print(
        f"LoRA: fused vs split {d:.6f}, LoRA moved the output {moved:.4f}", flush=True
    )

    streamed = {
        name: predict(stream.stream_fp32_core(p.clone()), x, t, ctx)
        for name, p in (("split", split), ("fused", fused))
    }
    d = (streamed["fused"] - streamed["split"]).norm() / streamed["split"].norm()
    print(f"fp32 stream: fused vs split {d:.6f}", flush=True)


if __name__ == "__main__":
    main()
