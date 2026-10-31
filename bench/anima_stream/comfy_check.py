"""The ComfyUI side of Anima's fp32 stream (``~/ComfyUI-Anima_lora-Adapter/stream.py``).

``numeric``: one step through ComfyUI's ``apply_model``, patched and not, against the library
forward without RoPE, the gap the node has. ``compat``: a short ``common_ksampler`` per row of
the proposal's table, off and on, with the dtypes that reach the GEMMs and the head.

    make daemon-run ARGS="--stall-timeout 900 bench/anima_stream/comfy_check.py"
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

from library.runtime.allocator import default_expandable_segments

default_expandable_segments()

import torch  # noqa: E402

from bench._anima import add_common_args, add_model_args, build_anima  # noqa: E402
from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402
from library.anima.int8 import convert_dit_, forward_file  # noqa: E402
from bench.anima_stream.run_bench import (  # noqa: E402
    fp32_reference,
    load_cell,
    pick_images,
    predict,
    set_arm,
)
from library.env import resolve_under_home  # noqa: E402

COMFY = Path(resolve_under_home("../comfy")).resolve()
sys.path.insert(0, str(COMFY))

import comfy.model_management  # noqa: E402
import comfy.sd  # noqa: E402
import folder_paths  # noqa: E402
import nodes  # noqa: E402

CASES = "base,lora,vocab_pack,dave,spectrum,fp8,block_compile".split(",")
LORA = "output/ckpt-artist/int8ab_bf16.safetensors"


def load_pack(name: str):
    path = COMFY / "custom_nodes" / name
    module = f"node_{name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(
        module, path / "__init__.py", submodule_search_locations=[str(path)]
    )
    pack = importlib.util.module_from_spec(spec)
    sys.modules[module] = pack
    spec.loader.exec_module(pack)
    return module, pack


adapter_module, adapter_pack = load_pack("comfyui-anima-lora-adapter")
adapter_nodes = sys.modules[f"{adapter_module}.nodes"]
stream_module = sys.modules[f"{adapter_module}.stream"]
stream_fp32_ = stream_module.stream_fp32_
INT8_FILE = "anima-base-v1.0_int8_gptq.safetensors"
folder_paths.add_model_folder_path(
    "diffusion_models", str(resolve_under_home("models/diffusion_models"))
)


def comfy_v(patcher, x, sigma, context):
    comfy.model_management.load_models_gpu([patcher])
    device = comfy.model_management.get_torch_device()
    x, sigma = x.to(device), sigma.to(device)
    denoised = patcher.model.apply_model(
        x,
        sigma,
        c_crossattn=context.to(device),
        transformer_options={"wrappers": patcher.wrappers},
    )
    return ((x - denoised) / sigma).squeeze(2).float().cpu()


def run_numeric(args, out):
    bundle = build_anima(args, adapter=None, train_mode=False)
    anima, device = bundle.anima, bundle.device
    images = pick_images(
        Path(resolve_under_home(args.cache_root)),
        args.n_images,
        args.min_tokens,
        args.max_tokens,
        args.seed,
    )
    sigmas = [float(s) for s in args.sigmas.split(",")]
    cells, ref, lib = {}, {}, {"bf16": {}, "fp32_norope": {}}
    with fp32_reference(anima):
        for s in sigmas:
            for j, ci in enumerate(images):
                cells[s, j] = load_cell(ci, j, args.seed, device)
                ref[s, j] = predict(anima, cells[s, j], s, "ref").cpu()
    for arm, preds in lib.items():
        set_arm(anima, arm)
        for key, cell in cells.items():
            preds[key] = predict(anima, cell, key[0], arm).cpu()
    set_arm(anima, "bf16")
    del anima, bundle
    torch.cuda.empty_cache()

    base = comfy.sd.load_diffusion_model(str(resolve_under_home(args.dit)))
    print("comfy compute dtype:", base.model.get_dtype_inference(), flush=True)
    patched = base.clone()
    assert stream_fp32_(patched)
    comfy_preds = {"bf16": {}, "fp32": {}}
    for arm, patcher in (("bf16", base), ("fp32", patched)):
        for (s, j), (latents, crossattn, noise) in cells.items():
            x = ((1.0 - s) * latents + s * noise).unsqueeze(2).float()
            comfy_preds[arm][s, j] = comfy_v(
                patcher, x, torch.tensor([s]), crossattn.float()
            )
        comfy.model_management.unload_all_models()

    def dist(a, b, key):
        return float((a[key] - b[key]).norm() / b[key].norm() * 100)

    for s in sigmas:
        keys = [k for k in cells if k[0] == s]
        row = {
            "library bf16": [dist(lib["bf16"], ref, k) for k in keys],
            "library fp32, no rope": [dist(lib["fp32_norope"], ref, k) for k in keys],
            "comfy bf16": [dist(comfy_preds["bf16"], ref, k) for k in keys],
            "comfy fp32": [dist(comfy_preds["fp32"], ref, k) for k in keys],
            "comfy fp32 to library": [
                dist(comfy_preds["fp32"], lib["fp32_norope"], k) for k in keys
            ],
            "comfy bf16 to library": [
                dist(comfy_preds["bf16"], lib["bf16"], k) for k in keys
            ],
        }
        out[str(s)] = {k: statistics.mean(v) for k, v in row.items()}
        print(f"RESULT σ={s} {json.dumps(out[str(s)], ensure_ascii=False)}", flush=True)


def run_int8(args, out):
    """ComfyUI's INT8 path as it ships, with the kernel's fp32 input, and with the stream,
    against the library's int8 base, whose kernel has always read fp32."""
    bundle = build_anima(args, adapter=None, train_mode=False)
    anima, device = bundle.anima, bundle.device
    images = pick_images(
        Path(resolve_under_home(args.cache_root)),
        args.n_images,
        args.min_tokens,
        args.max_tokens,
        args.seed,
    )
    sigmas = [float(s) for s in args.sigmas.split(",")]
    cells, ref = {}, {}
    with fp32_reference(anima):
        for s in sigmas:
            for j, ci in enumerate(images):
                cells[s, j] = load_cell(ci, j, args.seed, device)
                ref[s, j] = predict(anima, cells[s, j], s, "ref").cpu()
    path = forward_file(None, args.dit)
    convert_dit_(anima, "infer", path)
    arms = {"library int8": {}, "library int8 fp32, no rope": {}}
    for arm, tag in zip(arms, ("int8", "int8_fp32_norope")):
        set_arm(anima, tag)
        for key, cell in cells.items():
            arms[arm][key] = predict(anima, cell, key[0], tag).cpu()
    set_arm(anima, "int8")
    del anima, bundle
    torch.cuda.empty_cache()

    loader = adapter_nodes.AnimaInt8Loader()
    stock = comfy.sd.load_diffusion_model(path)
    fixed = loader.load(INT8_FILE, "bf16")[0]
    streamed = loader.load(INT8_FILE, "fp32")[0]
    q_proj = stock.get_model_object("diffusion_model").blocks[0].cross_attn.q_proj
    print("int8 kernels:", q_proj.quant_format, q_proj._full_precision_mm, flush=True)
    times = {}
    for arm, patcher in (
        ("comfy int8", stock),
        ("comfy int8, fp32 kernel input", fixed),
        ("comfy int8 fp32 stream", streamed),
    ):
        arms[arm] = {}
        t0 = time.perf_counter()
        for (s, j), (latents, crossattn, noise) in cells.items():
            x = ((1.0 - s) * latents + s * noise).unsqueeze(2).float()
            arms[arm][s, j] = comfy_v(patcher, x, torch.tensor([s]), crossattn.float())
        times[arm] = (time.perf_counter() - t0) / len(cells) * 1000
        comfy.model_management.unload_all_models()

    for s in sigmas:
        keys = [k for k in cells if k[0] == s]
        row = {
            arm: statistics.mean(
                float((p[k] - ref[k]).norm() / ref[k].norm() * 100) for k in keys
            )
            for arm, p in arms.items()
        }
        out[str(s)] = row
        print(f"RESULT σ={s} {json.dumps(row, ensure_ascii=False)}", flush=True)
    out["ms_per_forward"] = times
    print(f"RESULT ms/forward {json.dumps(times, ensure_ascii=False)}", flush=True)


def apply_case(name, model):
    """A patched clone of ``model`` for this case, with the sampler to run it on."""
    adapter = sys.modules[f"{adapter_module}.adapter"]
    if name in ("base", "fp8"):
        return model.clone(), None
    if name in ("lora", "int8_lora"):
        m = model.clone()
        adapter.apply_adapter(m, str(resolve_under_home(LORA)), 1.0, 1.0)
        return m, None
    if name == "vocab_pack":
        vocab = sys.modules[f"{adapter_module}.vocab_pack"]
        pack = resolve_under_home("models/vocab_packs/anima_cjk_vocab_pack_preview")
        table, _ = vocab.load_vocab_pack(f"{pack}.safetensors")
        m = model.clone()
        vocab.apply_vocab_pack(m, table)
        return m, None
    if name == "dave":
        module, _ = load_pack("comfyui-anima-dave")
        return sys.modules[f"{module}.nodes"].AnimaDAVE().patch(
            model.clone(), "auto", 1.0, 0.5
        )[0], None
    if name == "spectrum":
        module, _ = load_pack("comfyui-spectrum-ksampler")
        spectrum_nodes = sys.modules[f"{module}.nodes"]
        sampler = spectrum_nodes.SpectrumKSampler()
        required = sampler.INPUT_TYPES()["required"]
        ratio = required["refresh_ratio"][1]["default"]

        def run(m, seed, positive, negative, latent):
            return sampler.sample(
                m,
                seed,
                12,
                4.0,
                "euler",
                "simple",
                positive,
                negative,
                latent,
                "",
                "off",
                ratio,
            )[0]

        return model.clone(), run
    if name == "block_compile":
        module, _ = load_pack("comfyui-anima-blockcompile")
        return model.clone(), "compile"
    raise ValueError(name)


@contextlib.contextmanager
def dtypes_seen(dit):
    seen = {}
    if not hasattr(dit, "final_layer"):
        yield seen
        return
    names = {
        id(dit.blocks[-1]): "block",
        id(dit.blocks[0].self_attn.q_proj): "q_proj",
        id(dit.blocks[0].cross_attn.k_proj): "k_proj",
        id(dit.blocks[0].mlp.layer1): "layer1",
        id(dit.final_layer.linear): "head",
    }

    def hook(module, inputs, _output):
        if (key := names.get(id(module))) is not None:
            seen[key] = str(inputs[0].dtype).removeprefix("torch.")

    handle = torch.nn.modules.module.register_module_forward_hook(hook)
    try:
        yield seen
    finally:
        handle.remove()


def conds(context, case):
    positive = [[context, {}]]
    negative = [[torch.zeros_like(context), {}]]
    if (
        case == "vocab_pack"
    ):  # the per-step path: ids in the cond, outside inference mode
        options = {
            "t5xxl_ids": torch.randint(0, 32128, (64,)),
            "t5xxl_weights": torch.ones(64),
        }
        positive = [[context[:, :64], options]]
        negative = [[torch.zeros_like(context[:, :64]), dict(options)]]
    return positive, negative


def sample(case, run, patcher, context, seed):
    latent = {"samples": torch.zeros(1, 16, 1, 64, 64)}
    positive, negative = conds(context, case)
    if callable(run):
        return run(patcher, seed, positive, negative, latent)["samples"]
    return nodes.common_ksampler(
        patcher, seed, 8, 4.0, "euler", "simple", positive, negative, latent
    )[0]["samples"]


def run_case(case, base_path, context):
    options = {"dtype": torch.float8_e4m3fn} if case == "fp8" else {}
    if case == "int8_lora":
        base_path = str(Path(base_path).with_name(INT8_FILE))
    model = comfy.sd.load_diffusion_model(base_path, model_options=options)
    row = {"compute": str(model.model.get_dtype_inference())}
    latents = {}
    grad = contextlib.nullcontext() if case == "vocab_pack" else torch.inference_mode()
    for stream in (False, True):
        patcher, run = apply_case(case, model)
        if stream:
            row["applied"] = stream_fp32_(patcher)
        if run == "compile":
            compile_nodes = sys.modules["node_comfyui_anima_blockcompile.nodes"]
            patcher = compile_nodes.AnimaBlockCompile().patch(patcher)[0]
            run = None
        t0 = time.perf_counter()
        with grad, dtypes_seen(patcher.get_model_object("diffusion_model")) as seen:
            z = sample(case, run, patcher, context, 7)
        label = "fp32" if stream else "bf16"
        latents[label] = z.float()
        row[f"{label}_s"] = round(time.perf_counter() - t0, 1)
        row[f"{label}_dtypes"] = seen
        comfy.model_management.unload_all_models()
        del patcher
    row["finite"] = bool(torch.isfinite(latents["fp32"]).all())
    row["moved_pct"] = float(
        (latents["fp32"] - latents["bf16"]).norm() / latents["bf16"].norm() * 100
    )
    del model
    torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(ap, vae=False, text_encoder=False)
    add_common_args(ap, include_compile=False, include_checkpointing=False)
    ap.add_argument("--mode", default="numeric,compat")
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--cache_root", default="post_image_dataset/lora")
    ap.add_argument("--sigmas", default="0.1,0.5,0.9")
    ap.add_argument("--n_images", type=int, default=6)
    ap.add_argument("--min_tokens", type=int, default=3500)
    ap.add_argument("--max_tokens", type=int, default=4608)
    args = ap.parse_args()
    start_heartbeat(label="comfy_check")
    run_dir = make_run_dir("anima_stream", label=args.label or "comfy")
    metrics = {"numeric": {}, "int8": {}, "compat": {}}

    with torch.no_grad():
        if "numeric" in args.mode:
            run_numeric(args, metrics["numeric"])
        if "int8" in args.mode:
            run_int8(args, metrics["int8"])
    if "compat" in args.mode:
        ci = pick_images(
            Path(resolve_under_home(args.cache_root)),
            1,
            args.min_tokens,
            args.max_tokens,
            args.seed,
        )[0]
        context = load_cell(ci, 0, args.seed, "cpu")[1].float()
        path = str(resolve_under_home(args.dit))
        for case in args.cases.split(","):
            print(f"=== {case}", flush=True)
            try:
                row = run_case(case, path, context)
            except Exception as exc:  # the table records it, the rest still runs
                traceback.print_exc()
                row = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                comfy.model_management.unload_all_models()
                torch.cuda.empty_cache()
            metrics["compat"][case] = row
            print(f"RESULT {case}: {json.dumps(row, ensure_ascii=False)}", flush=True)
    write_result(run_dir, script=__file__, args=args, metrics=metrics)
    print(f"wrote {run_dir}", flush=True)


if __name__ == "__main__":
    main()
