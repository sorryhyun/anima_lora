"""How much of a compiled MOSS-Image LoRA step the convertible GEMMs take (the MOSS
counterpart of ``profile_step.py``).

Runs ``library.moss.train.run_train`` (TrainRequest argv after ours) and wraps
``training_loss``, which is called once per step: the interval between two calls is
one full step (forward, backward, optimizer, logging). Steps [warmup, warmup+timed)
are wall-timed with a sync on each side, the next ``profiled`` steps run under
torch.profiler with input shapes recorded. Then it writes the report JSON and
hard-exits (no checkpoint save).

    make daemon-run ARGS="--stall-timeout 1200 project/int8_train/src/profile_step_moss.py \\
        -- --config configs/moss/lora_bf16.toml --cache 'post_image_dataset/moss/channel_(caststation)_layout'"

GEMMs are classified by the second operand of ``aten::mm``: the converted linears are
the six per-block projections ``q, k, v, o`` (3072²), ``ffn_up_gate`` (3072 → 16384)
and ``ffn_down`` (8192 → 3072), forward (M,K)@(K,N) and dX (M,N)@(N,K), M >= 1024.
The text projection (M = text tokens), modulation and LoRA rank GEMMs fall outside.
"""

import argparse
import dataclasses
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--timed", type=int, default=20)
ap.add_argument("--profiled", type=int, default=6)
ap.add_argument("--out", default="project/int8_train/assets/moss_step_profile.json")
own, train_argv = ap.parse_known_args()
if train_argv and train_argv[0] == "--":
    train_argv = train_argv[1:]

from library.moss.requests import TrainRequest  # noqa: E402

req = TrainRequest.from_argv(train_argv)
if req.expandable_segments:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from library.env import resolve_under_home  # noqa: E402
from library.moss import train as moss_train  # noqa: E402

D, UG, FF = 3072, 16384, 8192
CONVERTED_KN = {(D, D), (D, UG), (UG, D), (FF, D), (D, FF)}
MIN_M = 1024
# gemm_bench.py 4608 moss per-block totals (ms), bf16 / int8, forward + dX.
R = (10.374 + 10.811) / (3.642 + 3.981)

T0 = own.warmup
T1 = T0 + own.timed
T2 = T1 + own.profiled
n_step = 0
t_prev = None
step_ms, tokens = [], []
prof = None

_orig_loss = moss_train.training_loss


def training_loss(transformer, latent, embeds, *args, **kwargs):
    global n_step, t_prev, prof
    torch.cuda.synchronize()
    now = time.perf_counter()
    if t_prev is not None:
        ms = (now - t_prev) * 1e3
        if T0 < n_step <= T1:
            step_ms.append(ms)
        print(f"[profile_step] step {n_step} {ms:.0f} ms", flush=True)
    t_prev = now
    if n_step == T1:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CUDA,
                torch.profiler.ProfilerActivity.CPU,
            ],
            record_shapes=True,
        )
        prof.__enter__()
    if n_step == T2:
        prof.__exit__(None, None, None)
        _report()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    if T0 <= n_step < T1:
        tokens.append(
            {"img": latent.shape[-2] * latent.shape[-1], "text": embeds.shape[1]}
        )
    n_step += 1
    return _orig_loss(transformer, latent, embeds, *args, **kwargs)


# First match wins: the quantize kernel's signature names `signed char` too, so it goes
# before the int8 GEMM.
KERNEL_CATEGORIES = [
    ("rotate_quant", r"quantize_int8|convrot"),
    ("int8_gemm", r"signed char"),
    ("attention", r"flash|fmha|attention"),
    ("memcpy", r"^Memcpy|^Memset"),
    ("gemm", r"gemm|gemv|cutlass|sm\d+_xmma"),
    ("optimizer", r"multi_tensor|foreach"),
    ("triton_fused", r"^triton_"),
    ("copy_cast", r"copy|elementwise_kernel"),
]


def _kernel_breakdown():
    by_name = defaultdict(lambda: [0.0, 0])
    for e in prof.events():
        if e.device_type != torch.autograd.DeviceType.CUDA or getattr(
            e, "is_user_annotation", False
        ):
            continue
        by_name[e.name][0] += e.device_time
        by_name[e.name][1] += 1
    cats = defaultdict(float)
    kernels = []
    for name, (us, cnt) in by_name.items():
        cat = next((c for c, pat in KERNEL_CATEGORIES if re.search(pat, name)), "other")
        cats[cat] += us / 1e3 / own.profiled
        kernels.append(
            {
                "name": name[:160],
                "category": cat,
                "ms": us / 1e3 / own.profiled,
                "calls_per_step": cnt / own.profiled,
            }
        )
    kernels.sort(key=lambda r: -r["ms"])
    return dict(sorted(cats.items(), key=lambda kv: -kv[1])), kernels[:80]


def _report():
    by_shape = defaultdict(lambda: [0.0, 0])  # (name, shapes) -> [device us, count]
    for ev in prof.key_averages(group_by_input_shape=True):
        dev_us = ev.device_time_total
        if dev_us <= 0:
            continue
        key = (ev.key, str(ev.input_shapes))
        by_shape[key][0] += dev_us
        by_shape[key][1] += ev.count

    rows = []
    for (name, shapes), (us, cnt) in by_shape.items():
        if name not in ("aten::mm", "aten::addmm", "aten::bmm", "aten::_scaled_mm"):
            continue
        s = eval(shapes)  # noqa: S307 -- profiler's own repr of nested int lists
        a, b = (s[-2], s[-1]) if name == "aten::addmm" else (s[0], s[1])
        conv = (
            len(a) == 2 and len(b) == 2 and a[0] >= MIN_M and tuple(b) in CONVERTED_KN
        )
        rows.append(
            {
                "op": name,
                "shapes": shapes,
                "ms": us / 1e3 / own.profiled,
                "calls_per_step": cnt / own.profiled,
                "kind": "converted" if conv else "other",
            }
        )
    conv_ms = sum(r["ms"] for r in rows if r["kind"] == "converted")
    other_mm = sum(r["ms"] for r in rows if r["kind"] == "other")
    rows.sort(key=lambda r: -r["ms"])

    categories, kernels = _kernel_breakdown()
    compute_ms = sum(v for k, v in categories.items() if k != "memcpy")
    step = statistics.median(step_ms)
    saving = conv_ms * (1 - 1 / R)
    out = {
        "argv": train_argv,
        "steps": {"warmup": own.warmup, "timed": own.timed, "profiled": own.profiled},
        "step_ms_median": step,
        "step_ms": step_ms,
        "tokens": tokens,
        "compute_kernel_ms_per_step": compute_ms,
        "converted_gemm_ms": conv_ms,
        "other_gemm_ms": other_mm,
        "converted_share_of_compute": conv_ms / compute_ms,
        "converted_share_of_step": conv_ms / step,
        "predicted_saving_ms": saving,
        "predicted_speedup_if_compute_bound": step / (step - saving),
        "kernel_category_ms": categories,
        "cuda_kernels": kernels,
        "gemm_rows": rows[:40],
    }
    path = resolve_under_home(own.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(
        json.dumps(
            {
                k: v
                for k, v in out.items()
                if k not in ("gemm_rows", "cuda_kernels", "step_ms", "tokens")
            },
            indent=1,
        )
    )
    print(f"[profile_step] wrote {path}")


moss_train.training_loss = training_loss
moss_train.run_train(
    dataclasses.replace(req, output="output/ckpt_moss/int8_profile.safetensors")
)
