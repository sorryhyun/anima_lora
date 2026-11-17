"""QKᵀ from two int8 planes per operand, in Triton: error and forward + backward time.

Q and K are each held as 16-bit fixed point per token, ``scale · (a + b / 254)`` with ``a``
and ``b`` int8. q·k ≈ a·a + (a·b + b·a) / 254 is three int8 GEMMs whose sums are exact in
int32 (the b·b term, about 1e-4 of the logit scale, is dropped). The backward recomputes S
from the same planes. Arms: ``port`` (what ships, fp32-accumulate QKᵀ), ``all`` (fp16
accumulation), ``int8_1p`` (one rotated int8 plane, the 5a forward) and ``int8_2p``.

Criteria, fixed before the run: dK and dV at or under 1.10x ``port`` by median and 1.21x on
the worst call; and ``int8_2p``'s forward + backward under ``port``'s by more than its
quantization (``prep``; torch ops here, the one-plane Triton kernel's time is beside it).

    make daemon-run ARGS="--stall-timeout 1200 project/finished/int8_train/src/attn_int8x2_bench.py"
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("INT8_ATTN_BWD_L", "64")  # FA2's backward tile

from library.runtime.allocator import default_expandable_segments  # noqa: E402

default_expandable_segments()

import torch  # noqa: E402

from bench._common import start_heartbeat  # noqa: E402
from fa2_kcenter_bench import OUTS, ref_all, relerr, timeit  # noqa: E402
from int8_attn import attn_bwd, attn_fwd, k_center, quant_planes, quant_tok  # noqa: E402
from library.env import resolve_under_home  # noqa: E402

A, G = "fp16acc", "fp16acc_gain"
ARMS = ("port", "all", "int8_1p", "int8_2p")
ROUNDS = 7


def prep(q, k, arm):
    if arm == "int8_1p":
        qr, qs = quant_tok(q, None)
        mean = k_center(k)
        kr, ks = quant_tok(k, mean)
        return (qr, qs, kr, ks, mean), None
    if arm == "int8_2p":
        qa, qb, qs = quant_planes(q)
        ka, kb, ks = quant_planes(k)
        return (qa, qs, ka, ks, None), (qb, kb)
    return None, None


def fwd(q, k, v, arm, saved, planes2):
    if arm == "port":
        return attn_fwd(q, k, v, "bf16", A)
    if arm == "all":
        return attn_fwd(q, k, v, A, A)
    return attn_fwd(
        q, k, v, "int8", A, None, "token", planes2 is None,
        pre=saved[:4], planes2=planes2,
    )  # fmt: skip


def bwd(do, q, k, v, o, lse, arm, saved, planes2):
    s = {"port": "fp16", "all": A}.get(arm, "bf16")
    return attn_bwd(do, q, k, v, o, lse, saved, A, G, A, s=s, planes2=planes2)


def speed(q, k, v, do):
    """ms per call, median of ROUNDS rounds with the arms interleaved."""
    fns = {}
    for arm in ARMS:
        saved, planes2 = prep(q, k, arm)
        o, lse = fwd(q, k, v, arm, saved, planes2)
        bwd(do, q, k, v, o, lse, arm, saved, planes2)  # autotune before any timing
        fns[f"{arm}.fwd"] = lambda a=arm, s=saved, p=planes2: fwd(q, k, v, a, s, p)
        fns[f"{arm}.bwd"] = lambda a=arm, s=saved, p=planes2, o=o, lse=lse: bwd(
            do, q, k, v, o, lse, a, s, p
        )
        if saved is not None:
            fns[f"{arm}.prep"] = lambda a=arm: prep(q, k, a)
    runs = {n: [] for n in fns}
    for _ in range(ROUNDS):
        for n, fn in fns.items():
            runs[n].append(timeit(fn, warm=5, iters=30))
    out = {n: statistics.median(x) for n, x in runs.items()}
    out.update({f"{n}.spread": max(x) - min(x) for n, x in runs.items()})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--capture", default="output/int8_train/attn_capture.pt")
    ap.add_argument(
        "--out", default="project/finished/int8_train/assets/attn_int8x2.json"
    )
    args = ap.parse_args()
    start_heartbeat(label="attn_int8x2_bench")
    t0 = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sets = [
        s for s in torch.load(resolve_under_home(args.capture)) if s["kind"] == "self"
    ]
    dev = "cuda"

    rows = []
    for i, s in enumerate(sets):
        q, k, v, do = (s[n].to(dev).contiguous() for n in ("q", "k", "v", "do"))
        refs = dict(zip(OUTS, ref_all(q, k, v, do)))
        row = {n: s[n] for n in ("block", "sigma", "image")}
        for arm in ARMS:
            saved, planes2 = prep(q, k, arm)
            o, lse = fwd(q, k, v, arm, saved, planes2)
            got = (o, *bwd(do, q, k, v, o, lse, arm, saved, planes2))
            for g, x in zip(OUTS, got):
                row[f"{arm}.{g}"] = relerr(x, refs[g])
        rows.append(row)
        print(
            f"{i + 1}/{len(sets)} b{row['block']} σ={row['sigma']} "
            + " ".join(
                f"{a}=" + "/".join(f"{row[f'{a}.{g}']:.4f}" for g in OUTS) for a in ARMS
            ),
            flush=True,
        )
        del q, k, v, do, refs
        torch.cuda.empty_cache()
    ratio = {
        a: {
            g: {
                "median": statistics.median(
                    r[f"{a}.{g}"] / r[f"port.{g}"] for r in rows
                ),
                "max": max(r[f"{a}.{g}"] / r[f"port.{g}"] for r in rows),
            }
            for g in OUTS
        }
        for a in ARMS
    }
    print(json.dumps(ratio, indent=1), flush=True)

    s = next(x for x in sets if x["image"] == 0)
    q, k, v, do = (s[n].to(dev).contiguous() for n in ("q", "k", "v", "do"))
    speeds = {"anima_self": {"shape": list(q.shape), **speed(q, k, v, do)}}
    del q, k, v, do
    torch.cuda.empty_cache()
    g = torch.Generator(device=dev).manual_seed(0)
    q, k, v = (
        torch.randn(1, 4600, 24, 128, device=dev, generator=g).to(torch.bfloat16)
        for _ in range(3)
    )
    do = (torch.randn(1, 4600, 24, 128, device=dev, generator=g) * 1e-7).to(
        torch.bfloat16
    )
    speeds["moss_self_random"] = {"shape": list(q.shape), **speed(q, k, v, do)}

    result = {
        "calls": len(rows),
        "ratio_to_port": ratio,
        "speed_ms": speeds,
        "rows": rows,
        "wall_s": time.perf_counter() - t0,
    }
    out = Path(resolve_under_home(args.out))
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(speeds, indent=1))


if __name__ == "__main__":
    main()
