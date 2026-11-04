"""fp16-accumulate attention backward against the Triton 16-bit baseline.

Candidate 2 of _archive/proposals/step_headroom.md. Same capture, error measure and
timing as ``attn_bwd_bench.py``; the gate is the one in reports.md, in multiples of
``tri_fp16``: dQ, dK, dV each at most 2x its error (median per σ) and at least 1.3x
its speed. FA2 is reported for reference only.

    make daemon-run ARGS="--stall-timeout 1200 project/finished/int8_train/src/attn_bwd_fp16acc_bench.py"
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from library.runtime.allocator import default_expandable_segments

default_expandable_segments()

import torch  # noqa: E402

from attn_bwd_bench import GRADS, fa2_fwd_bwd, ref_grads, relerr, timeit  # noqa: E402
from bench._common import start_heartbeat  # noqa: E402
from int8_attn import attn_bwd, attn_fwd  # noqa: E402
from library.env import resolve_under_home  # noqa: E402

BASE = "tri_fp16"
A, R, G = "fp16acc", "fp16acc_row", "fp16acc_gain"
F, FPV, F16 = ("bf16", "bf16"), ("bf16", A), (A, A)  # forward (qk, pv)
# name: (forward, s, dp, dv, dqk)
ARMS = {
    BASE: (F, "bf16", "fp16", "fp16", "fp16"),
    "dp": (F, "bf16", A, "fp16", "fp16"),
    "dv": (F, "bf16", "fp16", A, "fp16"),
    "dv_row": (F, "bf16", "fp16", R, "fp16"),
    "s": (F, A, "fp16", "fp16", "fp16"),
    "dqk": (F, "bf16", "fp16", "fp16", A),
    "dqk_row": (F, "bf16", "fp16", "fp16", R),
    "all": (F, A, A, A, A),
    "all_row": (F, A, A, R, R),
    "all_row_fwd16": (F16, A, A, R, R),
    "all_fwd16": (F16, A, A, A, A),
    "all_dvrow_fwd16": (F16, A, A, R, A),
    "no_s": (F, "bf16", A, A, A),
    "no_s_dvrow": (F, "bf16", A, R, A),
    "no_s_dvrow_fwdpv": (FPV, "bf16", A, R, A),
    "no_s_fwdpv": (FPV, "bf16", A, A, A),
    "no_s_dvgain": (F, "bf16", A, G, A),
    "no_s_dvgain_fwdpv": (FPV, "bf16", A, G, A),
    # what the FA2 port can run: Q and K share fp16 smem with dK, dQ
    "port": (FPV, "fp16", A, G, A),
    "port_bwd": (F, "fp16", A, G, A),
}


def arm_fwd(q, k, v, arm):
    return attn_fwd(q, k, v, *arm[0])


def arm_bwd(do, q, k, v, o, lse, arm):
    return attn_bwd(do, q, k, v, o, lse, None, arm[2], arm[3], arm[4], s=arm[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--capture", default="output/int8_train/attn_capture.pt")
    ap.add_argument("--out", default="project/finished/int8_train/assets/attn_bwd_fp16acc.json")
    ap.add_argument("--arms", nargs="*", help="subset; the baseline always runs")
    args = ap.parse_args()
    arms_run = {
        n: a for n, a in ARMS.items() if not args.arms or n in args.arms or n == BASE
    }
    start_heartbeat(label="attn_bwd_fp16acc_bench")
    t0 = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sets = [
        s for s in torch.load(resolve_under_home(args.capture)) if s["kind"] == "self"
    ]
    dev = "cuda"
    arms = ["fa2", *arms_run]

    errs = []
    for i, s in enumerate(sets):
        q, k, v, do = (s[n].to(dev).contiguous() for n in ("q", "k", "v", "do"))
        refs = dict(zip(GRADS, ref_grads(q, k, v, do)))
        row = {n: s[n] for n in ("block", "sigma", "image")}
        o, leaves = fa2_fwd_bwd(q, k, v)
        got = torch.autograd.grad(o, leaves, do)
        for g, x in zip(GRADS, got):
            row[f"fa2.{g}"] = relerr(x, refs[g])
        del o, leaves, got
        for name, arm in arms_run.items():
            o, lse = arm_fwd(q, k, v, arm)
            got = arm_bwd(do, q, k, v, o, lse, arm)
            for g, x in zip(GRADS, got):
                row[f"{name}.{g}"] = relerr(x, refs[g])
        errs.append(row)
        print(
            f"err {i + 1}/{len(sets)} b{row['block']} σ={row['sigma']} "
            + " ".join(
                f"{a}=" + "/".join(f"{row[f'{a}.{g}']:.4f}" for g in GRADS)
                for a in arms
            ),
            flush=True,
        )
        del q, k, v, do, refs
        torch.cuda.empty_cache()

    medians = defaultdict(dict)
    for sigma in sorted({r["sigma"] for r in errs}):
        rows = [r for r in errs if r["sigma"] == sigma]
        for a in arms:
            medians[str(sigma)][a] = {
                g: statistics.median(r[f"{a}.{g}"] for r in rows) for g in GRADS
            }

    s = next(x for x in sets if x["image"] == 0)
    q, k, v, do = (s[n].to(dev).contiguous() for n in ("q", "k", "v", "do"))
    o, leaves = fa2_fwd_bwd(q, k, v)
    from flash_attn import flash_attn_func

    t = {
        "fa2.fwd": timeit(lambda: flash_attn_func(q, k, v)),
        "fa2.bwd": timeit(
            lambda: torch.autograd.grad(o, leaves, do, retain_graph=True)
        ),
    }
    for name, arm in arms_run.items():
        fo, flse = arm_fwd(q, k, v, arm)
        t[f"{name}.fwd"] = timeit(lambda: arm_fwd(q, k, v, arm))
        t[f"{name}.bwd"] = timeit(lambda: arm_bwd(do, q, k, v, fo, flse, arm))
    print(
        f"speed {list(q.shape)}: " + " ".join(f"{a}={ms:.3f}ms" for a, ms in t.items()),
        flush=True,
    )

    gate = {}
    for a in arms_run:
        ratios = {g: max(m[a][g] / m[BASE][g] for m in medians.values()) for g in GRADS}
        speed = t[f"{BASE}.bwd"] / t[f"{a}.bwd"]
        gate[a] = {
            "worst_ratio_to_baseline": ratios,
            "bwd_speedup": speed,
            "fwd_bwd_speedup": (t[f"{BASE}.fwd"] + t[f"{BASE}.bwd"])
            / (t[f"{a}.fwd"] + t[f"{a}.bwd"]),
            "pass": max(ratios.values()) <= 2.0 and speed >= 1.3,
        }
    result = {
        "shape_q": list(q.shape),
        "errors": errs,
        "medians": medians,
        "speed_ms": t,
        "gate": gate,
        "wall_s": time.perf_counter() - t0,
    }
    out = Path(resolve_under_home(args.out))
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"medians": medians, "gate": gate}, indent=1))


if __name__ == "__main__":
    main()
