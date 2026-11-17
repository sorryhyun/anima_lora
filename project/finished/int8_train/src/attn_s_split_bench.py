"""fp16-accumulate QKᵀ with the partial sum handed to fp32 inside the head dim.

``all`` accumulates q·k over all 128 dims in one fp16 register. ``all_s2`` / ``s4`` / ``s8``
hand it to fp32 every 64 / 32 / 16 dims, on the forward and on the backward's S alike, so the
logsumexp still matches. Triton stand-ins (``int8_attn.py``); the split arms reload Q and K
per chunk, which gives the split's rounding and not its speed. The installed FA2's ``pv`` and ``all`` run beside them as a cross-check.

Criterion, fixed before the run: a split is worth porting to the CUDA kernels if dK and dV
sit at or under 1.10x ``port`` by median and 1.21x on the worst call, the level the shipped
backward passed at (attn_bwd_fp16acc.md). Error is against fp32 SDPA on the Anima capture.

    make daemon-run ARGS="--stall-timeout 1200 project/finished/int8_train/src/attn_s_split_bench.py"
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
from fa2_kcenter_bench import OUTS, ref_all, relerr, run_arm  # noqa: E402
from int8_attn import attn_bwd, attn_fwd  # noqa: E402
from library.env import resolve_under_home  # noqa: E402

A, G = "fp16acc", "fp16acc_gain"
# name: (forward qk, forward pv, s, split)
TRI = {
    "port": ("bf16", A, "fp16", 1),  # what ships: fp32-accumulate QKᵀ
    "all": (A, A, A, 1),
    "all_s2": (A, A, A, 2),
    "all_s4": (A, A, A, 4),
    "all_s8": (A, A, A, 8),
}
FA2 = {"fa2_pv": (None, None, "raw"), "fa2_all": (None, "all", "raw")}
BASE = {**{n: "port" for n in TRI}, **{n: "fa2_pv" for n in FA2}}


def tri_arm(q, k, v, do, arm):
    qk, pv, s, split = arm
    o, lse = attn_fwd(q, k, v, qk, pv, qk_split=split)
    return (o, *attn_bwd(do, q, k, v, o, lse, None, A, G, A, s=s, s_split=split))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--capture", default="output/int8_train/attn_capture.pt")
    ap.add_argument(
        "--out", default="project/finished/int8_train/assets/attn_s_split.json"
    )
    args = ap.parse_args()
    start_heartbeat(label="attn_s_split_bench")
    t0 = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sets = [
        s for s in torch.load(resolve_under_home(args.capture)) if s["kind"] == "self"
    ]
    dev = "cuda"
    names = [*TRI, *FA2]

    rows = []
    for i, s in enumerate(sets):
        q, k, v, do = (s[n].to(dev).contiguous() for n in ("q", "k", "v", "do"))
        refs = dict(zip(OUTS, ref_all(q, k, v, do)))
        row = {n: s[n] for n in ("block", "sigma", "image")}
        for name, arm in TRI.items():
            for g, x in zip(OUTS, tri_arm(q, k, v, do, arm)):
                row[f"{name}.{g}"] = relerr(x, refs[g])
        for name, arm in FA2.items():
            for g, x in zip(OUTS, run_arm(q, k, v, do, arm)):
                row[f"{name}.{g}"] = relerr(x, refs[g])
        rows.append(row)
        print(
            f"{i + 1}/{len(sets)} b{row['block']} σ={row['sigma']} "
            + " ".join(
                f"{a}=" + "/".join(f"{row[f'{a}.{g}']:.4f}" for g in OUTS)
                for a in names
            ),
            flush=True,
        )
        del q, k, v, do, refs
        torch.cuda.empty_cache()

    def ratios(sub):
        return {
            a: {
                g: {
                    "median": statistics.median(
                        r[f"{a}.{g}"] / r[f"{BASE[a]}.{g}"] for r in sub
                    ),
                    "max": max(r[f"{a}.{g}"] / r[f"{BASE[a]}.{g}"] for r in sub),
                }
                for g in OUTS
            }
            for a in names
        }

    result = {
        "calls": len(rows),
        "ratio_to_base": ratios(rows),
        "median_error": {
            a: {g: statistics.median(r[f"{a}.{g}"] for r in rows) for g in OUTS}
            for a in names
        },
        "by_block": {
            str(b): ratios([r for r in rows if r["block"] == b])
            for b in sorted({r["block"] for r in rows})
        },
        "pass": {},
        "rows": rows,
        "wall_s": time.perf_counter() - t0,
    }
    for a in TRI:
        r = result["ratio_to_base"][a]
        result["pass"][a] = all(
            r[g]["median"] <= 1.10 and r[g]["max"] <= 1.21 for g in ("dk", "dv")
        )
    out = Path(resolve_under_home(args.out))
    out.write_text(json.dumps(result, indent=1))
    print(
        json.dumps(
            {k: result[k] for k in ("ratio_to_base", "median_error", "pass")}, indent=1
        )
    )


if __name__ == "__main__":
    main()
