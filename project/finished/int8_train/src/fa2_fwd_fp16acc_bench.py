"""FA2's forward with fp16 accumulation (PV, or QKᵀ + PV) against its bf16 forward, in one process.

``--fa2_dir`` holds a build with flash-attention-fp16's ``patches/fwd_fp16acc.patch``.
``FLASH_ATTN_FP16ACC_FWD`` picks the arm per call (``0`` / ``pv`` / ``all``); the
backward stays on the shipped fp16acc. Error of O and of dQ / dK / dV against fp32 SDPA on the
Anima capture, speed on the capture's first self and cross calls and on MOSS's self shape.

    make daemon-run ARGS="--stall-timeout 900 project/finished/int8_train/src/fa2_fwd_fp16acc_bench.py --fa2_dir <dir>"
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

from library.runtime.allocator import default_expandable_segments

default_expandable_segments()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from attn_bwd_bench import GRADS, relerr  # noqa: E402
from bench._common import start_heartbeat  # noqa: E402
from fa2_fp16acc_bench import timeit  # noqa: E402
from library.env import resolve_under_home  # noqa: E402

ARMS = {"stock": "0", "pv": "pv", "all": "all"}


def set_arm(name):
    os.environ["FLASH_ATTN_FP16ACC_FWD"] = ARMS[name]


def ref(q, k, v, do):
    leaves = [x.float().requires_grad_() for x in (q, k, v)]
    o = F.scaled_dot_product_attention(*(x.transpose(1, 2) for x in leaves)).transpose(
        1, 2
    )
    return o.detach(), torch.autograd.grad(o, leaves, do.float())


def run(fa, q, k, v, do):
    leaves = [x.detach().requires_grad_() for x in (q, k, v)]
    o = fa(*leaves)
    return o.detach(), torch.autograd.grad(o, leaves, do)


def speed(fa, q, k, v, do):
    row = {}
    for name in ARMS:
        set_arm(name)
        leaves = [x.detach().requires_grad_() for x in (q, k, v)]
        row[f"{name}.fwd"] = timeit(lambda: fa(q, k, v))
        row[f"{name}.fwd_bwd"] = timeit(
            lambda: torch.autograd.grad(fa(*leaves), leaves, do)
        )
    for name in ("pv", "all"):
        row[f"{name}.fwd_gain"] = row["stock.fwd"] / row[f"{name}.fwd"]
        row[f"{name}.fwd_bwd_gain"] = row["stock.fwd_bwd"] / row[f"{name}.fwd_bwd"]
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--capture", default="output/int8_train/attn_capture.pt")
    ap.add_argument("--out", default="project/finished/int8_train/assets/fa2_fwd_fp16acc.json")
    ap.add_argument("--fa2_dir", required=True)
    ap.add_argument("--moss_tokens", type=int, default=4608)
    args = ap.parse_args()
    fa2_dir = Path(resolve_under_home(args.fa2_dir)).resolve()
    assert "flash_attn_2_cuda" not in sys.modules, (
        "flash_attn imported before --fa2_dir"
    )
    sys.path.insert(0, str(fa2_dir))
    start_heartbeat(label="fa2_fwd_fp16acc_bench")
    t0 = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False
    import flash_attn_2_cuda
    from flash_attn import flash_attn_func as fa

    print("fa2:", flash_attn_2_cuda.__file__, flush=True)
    assert Path(flash_attn_2_cuda.__file__).parent == fa2_dir
    os.environ.pop("FLASH_ATTN_FP16ACC", None)
    sets = torch.load(resolve_under_home(args.capture), mmap=True)
    dev = "cuda"

    errs, nonfinite = [], 0
    for i, s in enumerate(sets):
        q, k, v, do = (s[n].to(dev).contiguous() for n in ("q", "k", "v", "do"))
        o_ref, g_ref = ref(q, k, v, do)
        row = {n: s[n] for n in ("kind", "block", "sigma", "image")}
        for name in ARMS:
            set_arm(name)
            o, g = run(fa, q, k, v, do)
            nonfinite += sum(int(not t.isfinite().all()) for t in (o, *g))
            row[f"{name}.o"] = relerr(o, o_ref)
            for gn, x, r in zip(GRADS, g, g_ref):
                row[f"{name}.{gn}"] = relerr(x, r)
        errs.append(row)
        print(
            f"err {i + 1}/{len(sets)} {row['kind']} b{row['block']} σ={row['sigma']} "
            + " ".join(
                f"{a}=" + "/".join(f"{row[f'{a}.{g}']:.4f}" for g in ("o", *GRADS))
                for a in ARMS
            ),
            flush=True,
        )
        del q, k, v, do, o_ref, g_ref, o, g
        torch.cuda.empty_cache()

    keys = ("o", *GRADS)
    ratio = {}
    for kind in sorted({r["kind"] for r in errs}):
        rows = [r for r in errs if r["kind"] == kind]
        ratio[kind] = {
            a: {
                g: {
                    "median": statistics.median(
                        r[f"{a}.{g}"] / r[f"stock.{g}"] for r in rows
                    ),
                    "max": max(r[f"{a}.{g}"] / r[f"stock.{g}"] for r in rows),
                }
                for g in keys
            }
            for a in ("pv", "all")
        }

    t = {}
    for kind in ("self", "cross"):
        s = next(x for x in sets if x["kind"] == kind and x["image"] == 0)
        qkvd = [s[n].to(dev).contiguous() for n in ("q", "k", "v", "do")]
        t[kind] = {
            "shape_q": list(qkvd[0].shape),
            "shape_k": list(qkvd[1].shape),
            **speed(fa, *qkvd),
        }
    q, k, v = (
        torch.randn(1, args.moss_tokens, 24, 128, device=dev).bfloat16()
        for _ in range(3)
    )
    do = (torch.randn_like(q, dtype=torch.float32) * 1e-7).bfloat16()
    t["moss_random"] = {"shape_q": list(q.shape), **speed(fa, q, k, v, do)}

    result = {
        "ratio_over_stock": ratio,
        "speed_ms": t,
        "nonfinite_tensors": nonfinite,
        "errors": errs,
        "wall_s": time.perf_counter() - t0,
    }
    Path(resolve_under_home(args.out)).write_text(json.dumps(result, indent=1))
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("ratio_over_stock", "speed_ms", "nonfinite_tensors")
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
