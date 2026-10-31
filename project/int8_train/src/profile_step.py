"""Phase 1a: how much of a real compiled Anima LoRA step the convertible GEMMs take.

Runs the stock trainer (train.py argv after ours) and patches the loop's per-step
profiler hooks: steps [warmup, warmup+timed) are wall-timed with a sync on each side,
the next `profiled` steps run under torch.profiler with input shapes recorded. Then it
writes the report JSON and hard-exits (no checkpoint save).

    make daemon-run ARGS="--stall-timeout 1200 project/int8_train/src/profile_step.py \
        --warmup 15 --timed 20 --profiled 8 -- --method lora --preset default"

GEMMs are classified by the second operand of `aten::mm`: the converted linears are
self qkv/out, cross q/out, MLP 1/2 (dims 2048 / 6144 / 8192), forward (M,K)@(K,N) and
dX (M,N)@(N,K), M >= 1024. Cross kv (K=1024 context), adaLN (M=batch) and LoRA rank
GEMMs fall outside.
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict

from library.runtime.allocator import default_expandable_segments

default_expandable_segments()  # train.py sets this before torch loads; match it

import torch  # noqa: E402

from library.env import resolve_under_home  # noqa: E402

D, FF = 2048, 8192
CONVERTED_KN = {(D, 3 * D), (3 * D, D), (D, D), (D, FF), (FF, D)}
MIN_M = 1024
# phase 0 per-block totals (ms), bf16 / int8, forward + dX at M=4096. Shapes cannot
# tell fwd from dX (mlp1 fwd == mlp2 dX), and the two ratios are 2.62 / 2.57 anyway.
R = (4.881 + 4.879) / (1.861 + 1.897)

ap = argparse.ArgumentParser()
ap.add_argument("--warmup", type=int, default=15)
ap.add_argument("--timed", type=int, default=20)
ap.add_argument("--profiled", type=int, default=8)
ap.add_argument("--out", default="project/int8_train/assets/step_profile.json")
ap.add_argument("--int8", choices=("off", "fwd", "full"), default="off")
ap.add_argument("--int8_file", default=None)
own, train_argv = ap.parse_known_args()
if train_argv and train_argv[0] == "--":
    train_argv = train_argv[1:]

from library.training import loop  # noqa: E402

T0 = own.warmup
T1 = T0 + own.timed
T2 = T1 + own.profiled
n_step = 0
t_begin = None
t_prev_begin = None
step_ms, iter_ms, tokens = [], [], []
prof = None
cur_tokens = None

_orig_run_step = loop._run_step


def _run_step(trainer, state, batch):
    global cur_tokens
    lat = batch.get("latents") if isinstance(batch, dict) else None
    cur_tokens = (
        None if lat is None else int(lat.shape[0] * lat.shape[-2] * lat.shape[-1] // 4)
    )
    return _orig_run_step(trainer, state, batch)


def _begin(state):
    global t_begin, t_prev_begin, prof
    torch.cuda.synchronize()
    now = time.perf_counter()
    if T0 < n_step <= T1 and t_prev_begin is not None:
        iter_ms.append((now - t_prev_begin) * 1e3)
    t_prev_begin = now
    if n_step == T1:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CUDA,
                torch.profiler.ProfilerActivity.CPU,
            ],
            record_shapes=True,
        )
        prof.__enter__()
    t_begin = time.perf_counter()


def _end(state):
    global n_step
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t_begin) * 1e3
    if T0 <= n_step < T1:
        step_ms.append(ms)
        tokens.append(cur_tokens)
    n_step += 1
    print(f"[profile_step] step {n_step} {ms:.0f} ms tokens={cur_tokens}", flush=True)
    if n_step == T2:
        prof.__exit__(None, None, None)
        _report()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


# First match wins: the quantize kernel's signature names `signed char` too, so it goes
# before the int8 GEMM; inductor's fused kernels by the ops in their names.
KERNEL_CATEGORIES = [
    ("rotate_quant", r"quantize_int8|convrot"),
    ("int8_gemm", r"signed char"),
    ("attention", r"^void flash::"),
    ("gemm", r"gemm|gemv|cutlass"),
    ("optimizer", r"multi_tensor"),
    ("norm", r"^triton_.*layer_norm"),
    ("gelu", r"^triton_.*gelu"),
    ("rope_qknorm", r"^triton_.*(rsqrt|neg|pow)"),
    ("triton_other", r"^triton_"),
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

    total_kernel_us = sum(
        e.device_time
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA
        and not getattr(e, "is_user_annotation", False)  # compiled-graph ranges
    )
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

    kernel_ms = total_kernel_us / 1e3 / own.profiled
    step = statistics.median(step_ms)
    saving = conv_ms * (1 - 1 / R)
    pred = {f"r={r}": step / (step - conv_ms * (1 - 1 / r)) for r in (2.0, 2.3)}
    pred[f"phase0 r={R:.2f}"] = step / (step - saving)
    categories, kernels = _kernel_breakdown()
    out = {
        "argv": train_argv,
        "int8": own.int8,
        "steps": {"warmup": own.warmup, "timed": own.timed, "profiled": own.profiled},
        "step_ms_median": step,
        "step_ms": step_ms,
        "iter_ms_median": statistics.median(iter_ms) if iter_ms else None,
        "tokens": tokens,
        "profiled_kernel_ms_per_step": kernel_ms,
        "converted_gemm_ms": conv_ms,
        "other_gemm_ms": other_mm,
        "converted_share_of_kernel_time": conv_ms / kernel_ms,
        "converted_share_of_step": conv_ms / step,
        "predicted_saving_ms": saving,
        "predicted_speedup": pred,
        "kernel_category_ms": categories,
        "cuda_kernels": kernels,
        "gemm_rows": rows[:40],
        "top_kernels": [
            {"name": e.key[:120], "ms": e.device_time_total / 1e3 / own.profiled}
            for e in sorted(prof.key_averages(), key=lambda e: -e.device_time_total)[
                :30
            ]
            if e.device_time_total > 0
        ],
    }
    path = resolve_under_home(own.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(
        json.dumps(
            {
                k: v
                for k, v in out.items()
                if k
                not in ("gemm_rows", "top_kernels", "cuda_kernels", "step_ms", "tokens")
            },
            indent=1,
        )
    )
    print(f"[profile_step] wrote {path}")


loop._run_step = _run_step
loop._profiler_step_begin = _begin
loop._profiler_step_end = _end

from library.anima.trainer.cli import main  # noqa: E402

train_argv += ["--int8", own.int8, "--output_name", "int8_profile"]
if own.int8_file:
    train_argv += ["--int8_file", own.int8_file]
main(train_argv)
