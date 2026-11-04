"""Triton fp16acc forward arms on MOSS: time on MOSS's self shape, then the LoRA gradient on the
production plan, with FA2 swapped for the Triton pair in ``library.moss.model``.

    make daemon-run ARGS="--stall-timeout 900 project/finished/int8_train/src/moss_fwd16_triton.py"
"""

import contextlib
import json
import statistics
import sys
import time

from library.env import resolve_under_home


import torch  # noqa: E402

import grad_error_moss as G  # noqa: E402
import library.moss.model as M  # noqa: E402
from int8_attn import attn_bwd, attn_fwd  # noqa: E402

A, GAIN = "fp16acc", "fp16acc_gain"
F, FPV, F16 = ("bf16", "bf16"), ("bf16", A), (A, A)
BASE = "gptq_fused_f32s"
# name: (forward (qk, pv), s, dp, dv, dqk)
TRI = {
    "tri_nos": (F, "bf16", A, GAIN, A),
    "tri_nos_rep": (F, "bf16", A, GAIN, A),
    "tri_pv": (FPV, "bf16", A, GAIN, A),
    "tri_all": (F16, A, A, GAIN, A),
}
s_peak = {"max": 0.0}
counts = {"calls": 0, "nan_grad": 0}


class TriAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, arm):
        o, lse = attn_fwd(q, k, v, *arm[0])
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.arm = arm
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        arm = ctx.arm
        got = attn_bwd(do, q, k, v, o, lse, None, arm[2], arm[3], arm[4], s=arm[1])
        counts["nan_grad"] += sum(int(not g.isfinite().all()) for g in got)
        return (*got, None)


def tri_func(arm):
    def call(q, k, v):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        counts["calls"] += 1
        if arm[1] == A and counts["calls"] % 7 == 0:
            with torch.no_grad():
                s = q[0, :, 0].float() @ k[0, :, 0].float().T
                s_peak["max"] = max(s_peak["max"], s.abs().max().item())
        return TriAttn.apply(q, k, v, arm)

    return call


def timing():
    torch.manual_seed(0)
    shape = (1, 4600, 24, 128)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in "qkv")
    do = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 1e-7
    rows = {}
    for name in ("tri_nos", "tri_pv", "tri_all"):
        arm = TRI[name]
        fwd, bwd = [], []
        for i in range(45):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            o, lse = attn_fwd(q, k, v, *arm[0])
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            attn_bwd(do, q, k, v, o, lse, None, arm[2], arm[3], arm[4], s=arm[1])
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            if i >= 15:
                fwd.append((t1 - t0) * 1e3)
                bwd.append((t2 - t1) * 1e3)
        rows[name] = {"fwd": statistics.median(fwd), "bwd": statistics.median(bwd)}
        print(name, rows[name], flush=True)
    del q, k, v, do, o, lse
    torch.cuda.empty_cache()
    return rows


orig_arm, fa2 = G.arm, M.flash_attn_func


@contextlib.contextmanager
def arm(tr, mods, mode):
    if mode not in TRI:
        with orig_arm(tr, mods, mode):
            yield
        return
    M.flash_attn_func = tri_func(TRI[mode])
    try:
        with orig_arm(tr, mods, BASE):
            yield
    finally:
        M.flash_attn_func = fa2


torch.backends.cuda.matmul.allow_tf32 = False
times = timing()
G.arm = arm
G.MODES = (*G.MODES, *TRI)
out = "project/finished/int8_train/assets/moss_fwd16_triton.json"
sys.argv = [
    "grad_error_moss.py",
    "--modes",
    ",".join(("bf16", BASE, *TRI)),
    "--int8_file",
    "models/moss_image/moss_image_base_int8_gptq.safetensors",
    "--out",
    out,
]
G.main()

report = json.loads(resolve_under_home(out).read_text())
arms = (BASE, *TRI)
print("times", json.dumps(times), flush=True)
print("s_peak", s_peak, "counts", counts, flush=True)
for sigma, s in report["sigmas"].items():
    print(f"σ={sigma} median e32/noise", flush=True)
    for m in arms:
        print(f"  {m}: {s['median'][f'e32_{m}_vs_noise']:.4f}", flush=True)
    print(
        "  pred",
        {k: round(v, 5) for k, v in s["pred_err"].items() if "vs_pred" in k},
        flush=True,
    )
    print("  loss", {k: round(v, 5) for k, v in s["loss_mean"].items()}, flush=True)
ref = "tri_nos"
for m in ("tri_nos_rep", "tri_pv", "tri_all", BASE):
    depths = sorted({int(g) for p in report["per_image"] for g in p["err"][m]})
    line = []
    for g in depths:
        vals = [
            (p["err"][m][str(g)] / p["err"][ref][str(g)]) ** 0.5
            for p in report["per_image"]
        ]
        line.append(f"{statistics.median(vals):.3f}[{max(vals):.2f}]")
    print(f"{m}/{ref} by depth, median[max]:", " ".join(line), flush=True)
