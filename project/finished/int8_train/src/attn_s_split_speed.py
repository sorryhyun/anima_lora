"""Forward + backward time of the four-way fp16 QKᵀ split, in Triton.

``attn_s_split_bench.py`` gave the split's error from arms that reload Q and K per chunk.
Here the chunks are cut in registers (``split_reg``: the 128 dims in four groups by index mod
4, each an fp16 GEMM, summed in fp32), so the kernels do the work a real split does and the
time counts. Arms: ``port`` (what ships, fp32-accumulate QKᵀ), ``all`` (fp16 over all 128
dims) and ``all_q4``. Error on the Anima capture is repeated for ``all_q4``, since its chunks
are not the contiguous ones measured before.

Triton's backward computes S in both of its kernels and FA2 in one, so a backward difference
between arms counts about twice here. The installed FA2's ``pv`` and ``all`` are timed beside
them.

    make daemon-run ARGS="--stall-timeout 1200 project/finished/int8_train/src/attn_s_split_speed.py"
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
from flash_attn.flash_attn_interface import flash_attn_f16in_func  # noqa: E402

from bench._common import start_heartbeat  # noqa: E402
from fa2_kcenter_bench import OUTS, ref_all, relerr, set_mode, timeit  # noqa: E402
from int8_attn import attn_bwd, attn_fwd  # noqa: E402
from library.env import resolve_under_home  # noqa: E402

A, G = "fp16acc", "fp16acc_gain"
# name: (forward qk, s, register split)
TRI = {
    "port": ("bf16", "fp16", False),
    "all": (A, A, False),
    "all_q4": (A, A, True),
}
FA2 = {"fa2_pv": None, "fa2_all": "all"}
ROUNDS = 7


def tri_fwd(q, k, v, arm):
    return attn_fwd(q, k, v, arm[0], A, split_reg=arm[2])


def tri_bwd(do, q, k, v, o, lse, arm):
    return attn_bwd(do, q, k, v, o, lse, None, A, G, A, s=arm[1], split_reg=arm[2])


def speed(q, k, v, do):
    """ms per call, median of ROUNDS rounds with the arms interleaved."""
    fns = {}
    for name, arm in TRI.items():
        o, lse = tri_fwd(q, k, v, arm)
        tri_bwd(do, q, k, v, o, lse, arm)  # autotune before any timing
        fns[f"{name}.fwd"] = lambda arm=arm: tri_fwd(q, k, v, arm)
        fns[f"{name}.bwd"] = lambda arm=arm, o=o, lse=lse: tri_bwd(
            do, q, k, v, o, lse, arm
        )
    for name, fwd in FA2.items():
        set_mode(None, fwd)
        leaves = [x.detach().requires_grad_() for x in (q, k, v)]
        o = flash_attn_f16in_func(*leaves)

        def f(fwd=fwd):
            set_mode(None, fwd)
            return flash_attn_f16in_func(q, k, v)

        def b(fwd=fwd, o=o, leaves=leaves):
            set_mode(None, fwd)
            return torch.autograd.grad(o, leaves, do, retain_graph=True)

        fns[f"{name}.fwd"], fns[f"{name}.bwd"] = f, b
    runs = {n: [] for n in fns}
    for _ in range(ROUNDS):
        for n, fn in fns.items():
            runs[n].append(timeit(fn, warm=5, iters=30))
    set_mode(None, None)
    out = {n: statistics.median(x) for n, x in runs.items()}
    out.update({f"{n}.spread": max(x) - min(x) for n, x in runs.items()})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--capture", default="output/int8_train/attn_capture.pt")
    ap.add_argument(
        "--out", default="project/finished/int8_train/assets/attn_s_split_speed.json"
    )
    args = ap.parse_args()
    start_heartbeat(label="attn_s_split_speed")
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
        for name, arm in TRI.items():
            o, lse = tri_fwd(q, k, v, arm)
            got = (o, *tri_bwd(do, q, k, v, o, lse, arm))
            for g, x in zip(OUTS, got):
                row[f"{name}.{g}"] = relerr(x, refs[g])
        rows.append(row)
        print(
            f"{i + 1}/{len(sets)} b{row['block']} σ={row['sigma']} "
            + " ".join(
                f"{a}=" + "/".join(f"{row[f'{a}.{g}']:.4f}" for g in OUTS) for a in TRI
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
        for a in TRI
    }

    s = next(x for x in sets if x["image"] == 0)
    q, k, v, do = (s[n].to(dev).contiguous() for n in ("q", "k", "v", "do"))
    speeds = {"anima_self": {"shape": list(q.shape), **speed(q, k, v, do)}}
    print(json.dumps(speeds, indent=1), flush=True)
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
    print(json.dumps({k: result[k] for k in ("ratio_to_port", "speed_ms")}, indent=1))


if __name__ == "__main__":
    main()
