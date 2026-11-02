"""FA2's backward with fp16 accumulation against its stock backward, in one process.

``--fa2_dir`` holds a build with ``patches/fp16acc.patch`` of flash-attention-fp16.
``FLASH_ATTN_FP16ACC=0`` takes the stock bf16 backward. Same capture and error measure
as ``attn_bwd_fp16acc_bench.py``, here on self- and cross-attention, against FA2. Speed
on the capture's first self and cross calls and on MOSS's self shape (random input).

    make daemon-run ARGS="--stall-timeout 600 project/int8_train/src/fa2_fp16acc_bench.py --fa2_dir <dir>"
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

from attn_bwd_bench import GRADS, ref_grads, relerr  # noqa: E402
from bench._common import start_heartbeat  # noqa: E402
from library.env import resolve_under_home  # noqa: E402

ARMS = {"stock": "0", "fp16acc": "1"}


def timeit(fn, warm=10, iters=50, rounds=5):
    best = []
    for _ in range(warm):
        fn()
    for _ in range(rounds):
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        best.append(s.elapsed_time(e) / iters)
    return statistics.median(best)


def speed(flash_attn_func, q, k, v, do):
    leaves = [x.detach().requires_grad_() for x in (q, k, v)]
    o = flash_attn_func(*leaves)
    row = {"fwd": timeit(lambda: flash_attn_func(q, k, v))}
    for name, flag in ARMS.items():
        os.environ["FLASH_ATTN_FP16ACC"] = flag
        row[name] = timeit(
            lambda: torch.autograd.grad(o, leaves, do, retain_graph=True)
        )
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--capture", default="output/int8_train/attn_capture.pt")
    ap.add_argument("--out", default="project/int8_train/assets/fa2_fp16acc.json")
    ap.add_argument("--fa2_dir", required=True)
    ap.add_argument("--moss_tokens", type=int, default=4600)
    args = ap.parse_args()
    sys.path.insert(0, args.fa2_dir)
    start_heartbeat(label="fa2_fp16acc_bench")
    t0 = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import flash_attn_2_cuda
    from flash_attn import flash_attn_func

    print("fa2:", flash_attn_2_cuda.__file__, flush=True)
    sets = torch.load(resolve_under_home(args.capture))
    dev = "cuda"

    errs = []
    for i, s in enumerate(sets):
        q, k, v, do = (s[n].to(dev).contiguous() for n in ("q", "k", "v", "do"))
        refs = dict(zip(GRADS, ref_grads(q, k, v, do)))
        row = {n: s[n] for n in ("kind", "block", "sigma", "image")}
        leaves = [x.detach().requires_grad_() for x in (q, k, v)]
        o = flash_attn_func(*leaves)
        for name, flag in ARMS.items():
            os.environ["FLASH_ATTN_FP16ACC"] = flag
            got = torch.autograd.grad(o, leaves, do, retain_graph=True)
            for g, x in zip(GRADS, got):
                row[f"{name}.{g}"] = relerr(x, refs[g])
        errs.append(row)
        print(
            f"err {i + 1}/{len(sets)} {row['kind']} b{row['block']} σ={row['sigma']} "
            + " ".join(
                f"{a}=" + "/".join(f"{row[f'{a}.{g}']:.4f}" for g in GRADS)
                for a in ARMS
            ),
            flush=True,
        )
        del q, k, v, do, refs, leaves, o, got
        torch.cuda.empty_cache()

    medians = {}
    for kind in sorted({r["kind"] for r in errs}):
        for sigma in sorted({r["sigma"] for r in errs}):
            rows = [r for r in errs if r["kind"] == kind and r["sigma"] == sigma]
            medians[f"{kind}.{sigma}"] = {
                a: {g: statistics.median(r[f"{a}.{g}"] for r in rows) for g in GRADS}
                for a in ARMS
            }

    t = {}
    for kind in ("self", "cross"):
        s = next(x for x in sets if x["kind"] == kind and x["image"] == 0)
        qkvd = [s[n].to(dev).contiguous() for n in ("q", "k", "v", "do")]
        t[kind] = {"shape_q": list(qkvd[0].shape), **speed(flash_attn_func, *qkvd)}
    q, k, v = (
        torch.randn(1, args.moss_tokens, 24, 128, device=dev).bfloat16()
        for _ in range(3)
    )
    do = (torch.randn_like(q, dtype=torch.float32) * 1e-7).bfloat16()
    t["moss_random"] = {"shape_q": list(q.shape), **speed(flash_attn_func, q, k, v, do)}
    for row in t.values():
        row["bwd_speedup"] = row["stock"] / row["fp16acc"]
        row["fwd_bwd_speedup"] = (row["fwd"] + row["stock"]) / (
            row["fwd"] + row["fp16acc"]
        )

    worst = {}
    for kind in ("self", "cross"):
        rows = [r for r in errs if r["kind"] == kind]
        worst[kind] = {
            g: max(r[f"fp16acc.{g}"] / r[f"stock.{g}"] for r in rows) for g in GRADS
        }
    result = {
        "medians": medians,
        "worst_call_ratio": worst,
        "speed_ms": t,
        "errors": errs,
        "wall_s": time.perf_counter() - t0,
    }
    Path(resolve_under_home(args.out)).write_text(json.dumps(result, indent=1))
    print(
        json.dumps({k: result[k] for k in ("worst_call_ratio", "speed_ms")}, indent=1)
    )
    for key, m in medians.items():
        print(
            key,
            " / ".join(f"{m['fp16acc'][g] / m['stock'][g]:.2f}" for g in GRADS),
            flush=True,
        )


if __name__ == "__main__":
    main()
