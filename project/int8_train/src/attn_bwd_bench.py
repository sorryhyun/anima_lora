"""Phase 5b: Triton int8 attention backward against FA2 (reports.md).

Reads phase 5a's capture (``src/attn_bench.py`` writes it: q, k, v and dO of real
training calls) and scores the self-attention calls only; cross-attention stays FA2.
Every arm's dQ, dK, dV and FA2's are compared with fp32 SDPA autograd (TF32 off) on
the same bf16 inputs, as ``‖g − g_fp32‖ / ‖g_fp32‖``. Gate, per gradient: median over
calls, per σ, at most 2x FA2's; and the backward at least 1.3x FA2's on the first
image's shape, preparation kernels included.

    make daemon-run ARGS="--stall-timeout 1200 project/int8_train/src/attn_bwd_bench.py"
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
import torch.nn.functional as F  # noqa: E402

from bench._common import start_heartbeat  # noqa: E402
from library.env import resolve_under_home  # noqa: E402

# name: (int8 forward, dp, dv, dqk)
ARMS = {
    "tri_fp16": (False, "fp16", "fp16", "fp16"),
    "int8qk_fp16": (True, "fp16", "fp16", "fp16"),
    "int8qk_fp16acc": (True, "fp16acc", "fp16acc", "fp16"),
    "int8_dp_fp16": (True, "fp16", "int8", "int8"),
    "int8_dp_fp16acc": (True, "fp16acc", "int8", "int8"),
    "int8_all": (True, "int8", "int8", "int8"),
    # int8 forward's O and logsumexp into FA2's CUDA backward (bf16 recompute)
    "int8fwd_fa2bwd": (True, "fa2", "fa2", "fa2"),
}
GRADS = ("dq", "dk", "dv")


def relerr(a, ref):
    return float((a.float() - ref).norm() / ref.norm())


def timeit(fn, warm=10, iters=50):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def ref_grads(q, k, v, do):
    leaves = [x.float().requires_grad_() for x in (q, k, v)]
    o = F.scaled_dot_product_attention(*(x.transpose(1, 2) for x in leaves))
    return torch.autograd.grad(o.transpose(1, 2), leaves, do.float())


def fa2_fwd_bwd(q, k, v):
    from flash_attn import flash_attn_func

    leaves = [x.detach().requires_grad_() for x in (q, k, v)]
    o = flash_attn_func(*leaves)
    return o, leaves


def arm_fwd(q, k, v, int8_fwd):
    from int8_attn import attn_fwd, attn_fwd_train  # sibling module

    if int8_fwd:
        return attn_fwd_train(q, k, v)
    return (*attn_fwd(q, k, v, "bf16", "bf16"), None)


def arm_bwd(do, q, k, v, fwd, arm):
    from int8_attn import attn_bwd, lse_natural  # sibling module

    o, lse, saved = fwd
    if arm[1] == "fa2":
        from flash_attn.flash_attn_interface import _flash_attn_backward

        dq, dk, dv = (torch.empty_like(x) for x in (q, k, v))
        lse_n = lse_natural(q, lse, saved[4])
        _flash_attn_backward(
            do, q, k, v, o, lse_n, dq, dk, dv, 0.0, q.shape[-1] ** -0.5, False,
            -1, -1, 0.0, None, False,
        )  # fmt: skip
        return dq, dk, dv
    return attn_bwd(do, q, k, v, o, lse, saved, *arm[1:])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--capture", default="output/int8_train/attn_capture.pt")
    ap.add_argument("--out", default="project/int8_train/assets/attn_bwd.json")
    args = ap.parse_args()
    start_heartbeat(label="attn_bwd_bench")
    t0 = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sets = [
        s for s in torch.load(resolve_under_home(args.capture)) if s["kind"] == "self"
    ]
    dev = "cuda"

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
        for name, arm in ARMS.items():
            got = arm_bwd(do, q, k, v, arm_fwd(q, k, v, arm[0]), arm)
            for g, x in zip(GRADS, got):
                row[f"{name}.{g}"] = relerr(x, refs[g])
        errs.append(row)
        print(
            f"err {i + 1}/{len(sets)} b{row['block']} σ={row['sigma']} "
            + " ".join(
                f"{a}=" + "/".join(f"{row[f'{a}.{g}']:.4f}" for g in GRADS)
                for a in ["fa2", *ARMS]
            ),
            flush=True,
        )
        del q, k, v, do, refs
        torch.cuda.empty_cache()

    arms = ["fa2", *ARMS]
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
    for name, arm in ARMS.items():
        fwd = arm_fwd(q, k, v, arm[0])
        t[f"{name}.fwd"] = timeit(lambda: arm_fwd(q, k, v, arm[0]))
        t[f"{name}.bwd"] = timeit(lambda: arm_bwd(do, q, k, v, fwd, arm))
    from int8_attn import quant_t, quant_tok  # sibling module

    t["prep.quant_tok"] = timeit(lambda: quant_tok(q, None))
    t["prep.quant_t"] = timeit(lambda: quant_t(q))
    print(
        f"speed {list(q.shape)}: " + " ".join(f"{a}={ms:.3f}ms" for a, ms in t.items()),
        flush=True,
    )

    gate = {}
    for a in ARMS:
        ratios = {
            g: max(m[a][g] / m["fa2"][g] for m in medians.values()) for g in GRADS
        }
        speed = t["fa2.bwd"] / t[f"{a}.bwd"]
        total = (t["fa2.fwd"] + t["fa2.bwd"]) / (t[f"{a}.fwd"] + t[f"{a}.bwd"])
        gate[a] = {
            "worst_ratio_to_fa2": ratios,
            "bwd_speedup": speed,
            "fwd_bwd_speedup": total,
            "saved_ms_per_step_28_blocks": 28
            * (t["fa2.fwd"] + t["fa2.bwd"] - t[f"{a}.fwd"] - t[f"{a}.bwd"]),
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
