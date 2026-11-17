"""fp16-accumulate QKᵀ on the shipped FA2 build, with the key mean removed.

Softmax over keys is unchanged by q·c added to a whole row, so K − c gives the same O, dQ, dK
and dV for any constant c. With c the mean key per (batch, head) the logits get smaller, and
fp16 accumulation rounds a smaller partial sum. Arms, all on the installed wheel's kernels
through ``flash_attn_f16in_func`` (the mode variables are read on every call):

- ``stock``: bf16 forward and stock backward.
- ``pv``: what ships (fp32-accumulate QKᵀ).
- ``all``: QKᵀ in fp16 accumulation, forward and backward.
- ``all_kc`` / ``pv_kc``: the same two on K − mean, rounded to fp16.
- ``all_kc_bf16``: K − mean rounded to bf16.
- ``kc_floor``: fp32 attention on the fp16-rounded K − mean, the rounding's own cost.

Error is against fp32 SDPA on the captured tensors, as in ``fa2_fwd_fp16acc_bench.py``.

    make daemon-run ARGS="--stall-timeout 600 project/finished/int8_train/src/fa2_kcenter_bench.py"
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from library.runtime.allocator import default_expandable_segments

default_expandable_segments()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from flash_attn import flash_attn_func  # noqa: E402
from flash_attn.flash_attn_interface import flash_attn_f16in_func  # noqa: E402

from bench._common import start_heartbeat  # noqa: E402
from library.env import resolve_under_home  # noqa: E402

OUTS = ("o", "dq", "dk", "dv")
# name: (FLASH_ATTN_FP16ACC, FLASH_ATTN_FP16ACC_FWD, keys)
ARMS = {
    "stock": ("0", "0", "raw"),
    "pv": (None, None, "raw"),
    "all": (None, "all", "raw"),
    "pv_kc": (None, None, "kc"),
    "all_kc": (None, "all", "kc"),
    "all_kc_bf16": (None, "all", "kc_bf16"),
}


def set_mode(fp16acc: str | None, fwd: str | None) -> None:
    for name, val in (("FLASH_ATTN_FP16ACC", fp16acc), ("FLASH_ATTN_FP16ACC_FWD", fwd)):
        if val is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = val


def center(k: torch.Tensor) -> torch.Tensor:
    """``k [B, L, H, D]`` minus its mean over keys, fp16."""
    return (k.float() - k.mean(1, keepdim=True, dtype=torch.float32)).half()


def relerr(a, ref):
    return float((a.float() - ref).norm() / ref.norm())


def ref_all(q, k, v, do):
    leaves = [x.float().requires_grad_() for x in (q, k, v)]
    o = F.scaled_dot_product_attention(*(x.transpose(1, 2) for x in leaves)).transpose(
        1, 2
    )
    return (o.detach(), *torch.autograd.grad(o, leaves, do.float()))


def run_arm(q, k, v, do, arm):
    fp16acc, fwd, _ = arm
    set_mode(fp16acc, fwd)
    # an fp16 leaf would get its gradient cast to fp16, and dK (about 1e-7) underflows there
    leaves = [
        (x.float() if x.dtype == torch.float16 else x).detach().requires_grad_()
        for x in (q, k, v)
    ]
    fn = flash_attn_func if fp16acc == "0" else flash_attn_f16in_func
    o = fn(*leaves)
    grads = torch.autograd.grad(o, leaves, do)
    set_mode(None, None)
    return (o.detach(), *grads)


def logit_stats(q, k):
    """Raw q·k before the 1/√D scale: mean |S|, P-weighted mean |S|, max |S|."""
    d = q.shape[-1]
    s = q.float().transpose(1, 2) @ k.float().transpose(1, 2).transpose(-1, -2)
    p = torch.softmax(s * d**-0.5, -1)
    a = s.abs()
    return {
        "abs_mean": float(a.mean()),
        "p_weighted_abs": float((p * a).sum(-1).mean()),
        "abs_max": float(a.max()),
    }


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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--capture", default="output/int8_train/attn_capture.pt")
    ap.add_argument(
        "--out", default="project/finished/int8_train/assets/fa2_kcenter.json"
    )
    args = ap.parse_args()
    start_heartbeat(label="fa2_kcenter_bench")
    t0 = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sets = [
        s for s in torch.load(resolve_under_home(args.capture)) if s["kind"] == "self"
    ]
    dev = "cuda"
    names = [*ARMS, "kc_floor"]

    rows = []
    for i, s in enumerate(sets):
        q, k, v, do = (s[n].to(dev).contiguous() for n in ("q", "k", "v", "do"))
        keys = {"raw": k, "kc": center(k)}
        keys["kc_bf16"] = keys["kc"].float().to(torch.bfloat16)
        refs = dict(zip(OUTS, ref_all(q, k, v, do)))
        row = {n: s[n] for n in ("block", "sigma", "image")}
        for name, arm in ARMS.items():
            for g, x in zip(OUTS, run_arm(q, keys[arm[2]], v, do, arm)):
                row[f"{name}.{g}"] = relerr(x, refs[g])
        for g, x in zip(OUTS, ref_all(q, keys["kc"], v, do)):
            row[f"kc_floor.{g}"] = relerr(x, refs[g])
        row["logits_raw"] = logit_stats(q, k)
        row["logits_kc"] = logit_stats(q, keys["kc"])
        rows.append(row)
        print(
            f"{i + 1}/{len(sets)} b{row['block']} σ={row['sigma']} "
            f"|S| p-weighted {row['logits_raw']['p_weighted_abs']:.1f} → "
            f"{row['logits_kc']['p_weighted_abs']:.1f} "
            + " ".join(
                f"{a}=" + "/".join(f"{row[f'{a}.{g}']:.4f}" for g in OUTS)
                for a in names
            ),
            flush=True,
        )
        del q, k, v, do, refs, keys
        torch.cuda.empty_cache()

    # per-call ratio over stock, median and max, as fa2_fwd_fp16acc.md reports it
    ratios = {
        a: {
            g: {
                "median": statistics.median(
                    r[f"{a}.{g}"] / r[f"stock.{g}"] for r in rows
                ),
                "max": max(r[f"{a}.{g}"] / r[f"stock.{g}"] for r in rows),
            }
            for g in OUTS
        }
        for a in names
    }
    medians = {
        a: {g: statistics.median(r[f"{a}.{g}"] for r in rows) for g in OUTS}
        for a in names
    }
    by_block = {}
    for b in sorted({r["block"] for r in rows}):
        rb = [r for r in rows if r["block"] == b]
        by_block[str(b)] = {
            "p_weighted_abs_raw": statistics.median(
                r["logits_raw"]["p_weighted_abs"] for r in rb
            ),
            "p_weighted_abs_kc": statistics.median(
                r["logits_kc"]["p_weighted_abs"] for r in rb
            ),
            **{
                f"{a}.{g}": statistics.median(
                    r[f"{a}.{g}"] / r[f"stock.{g}"] for r in rb
                )
                for a in ("all", "all_kc")
                for g in OUTS
            },
        }

    s = next(x for x in sets if x["image"] == 0)
    q, k, v, do = (s[n].to(dev).contiguous() for n in ("q", "k", "v", "do"))
    kc = center(k)
    speed = {"center": timeit(lambda: center(k))}
    for name, keys in (("pv", k), ("all", k), ("all_kc", kc)):
        arm = ARMS[name]
        set_mode(arm[0], arm[1])
        leaves = [x.detach().requires_grad_() for x in (q, keys, v)]
        o = flash_attn_f16in_func(*leaves)
        speed[f"{name}.fwd"] = timeit(lambda: flash_attn_f16in_func(q, keys, v))
        speed[f"{name}.bwd"] = timeit(
            lambda: torch.autograd.grad(o, leaves, do, retain_graph=True)
        )
        set_mode(None, None)

    result = {
        "shape_q": list(q.shape),
        "calls": len(rows),
        "ratio_to_stock": ratios,
        "median_error": medians,
        "by_block": by_block,
        "speed_ms": speed,
        "rows": rows,
        "wall_s": time.perf_counter() - t0,
    }
    out = Path(resolve_under_home(args.out))
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in list(result)[:6]}, indent=1))


if __name__ == "__main__":
    main()
