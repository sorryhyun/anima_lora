"""Two backward-side arms on the MOSS production plan (``gptq_fused_f32s``): ``dy32``
feeds the int8 dX kernel an fp32 ``dY`` (the forward's comfy-kitchen#289 cast, mirrored),
``noreduce`` turns cuBLAS's bf16 split-K reduction off, which on the int8 plan touches the
LoRA-side GEMMs alone. ``rep`` repeats the plan for FA2's non-determinism floor.

    make daemon-run ARGS="--stall-timeout 900 project/finished/int8_train/src/moss_dy32_noreduce.py"
"""

import contextlib
import json
import statistics
import sys

from library.env import resolve_under_home

import torch  # noqa: E402

import grad_error_moss as G  # noqa: E402
from library.int8 import linear as L  # noqa: E402

BASE = "gptq_fused_f32s"
ARMS = ("plan_rep", "plan_dy32", "plan_noreduce")
real_mm = L.int8_mm


def mm_fp32_in(x, wq, s):
    return real_mm(x.float(), wq, s)


orig_arm = G.arm


@contextlib.contextmanager
def arm(tr, mods, mode):
    if mode not in ARMS:
        with orig_arm(tr, mods, mode):
            yield
        return
    matmul = torch.backends.cuda.matmul
    if mode == "plan_dy32":
        L.int8_mm = mm_fp32_in
    elif mode == "plan_noreduce":
        matmul.allow_bf16_reduced_precision_reduction = False
    try:
        with orig_arm(tr, mods, BASE):
            yield
    finally:
        L.int8_mm = real_mm
        matmul.allow_bf16_reduced_precision_reduction = True


G.arm = arm
G.MODES = (*G.MODES, *ARMS)
out = "project/finished/int8_train/assets/moss_dy32_noreduce.json"
sys.argv = [
    "grad_error_moss.py",
    "--modes",
    ",".join(("bf16", BASE, *ARMS)),
    "--int8_file",
    "models/moss_image/moss_image_base_int8_gptq.safetensors",
    "--exclude_report",
    "models/moss_image/moss_image_base_int8_gptq.report.json",
    "--out",
    out,
]
G.main()

report = json.loads(resolve_under_home(out).read_text())
for sigma, s in report["sigmas"].items():
    print(f"σ={sigma} median e32/noise", flush=True)
    for m in (BASE, *ARMS):
        print(f"  {m}: {s['median'][f'e32_{m}_vs_noise']:.4f}", flush=True)
    print(
        "  pred",
        {k: round(v, 5) for k, v in s["pred_err"].items() if "vs_pred" in k},
        flush=True,
    )
    print("  loss", {k: round(v, 5) for k, v in s["loss_mean"].items()}, flush=True)
for m in ARMS:
    depths = sorted({int(g) for p in report["per_image"] for g in p["err"][m]})
    line = []
    for g in depths:
        vals = [
            (p["err"][m][str(g)] / p["err"][BASE][str(g)]) ** 0.5
            for p in report["per_image"]
        ]
        line.append(f"{statistics.median(vals):.3f}[{max(vals):.2f}]")
    print(f"{m}/{BASE} by depth, median[max]:", " ".join(line), flush=True)
