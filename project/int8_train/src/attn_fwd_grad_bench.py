"""Phase 5a′: forward variants judged on the gradients they yield through FA2's backward.

Each arm produces O and a natural-log logsumexp for the 24 self-attention calls of the
5a capture; FA2's CUDA backward (``_flash_attn_backward``) takes them with the captured
dO, and dQ, dK, dV are compared with fp32 SDPA autograd (TF32 off) on the same bf16
inputs. Gate (proposal.md, 5a′): each gradient at most 2x FA2's distance from fp32,
median per σ; kernel arms also at least 1.3x FA2's forward.

Kernel arms run ``int8_attn``. ``emu_*`` arms are fp32 torch emulations of the int8
rounding (the kernel's centering, bf16 cast, Hadamard and round-half-away), used to
split the error by operand before writing a kernel: ``emu_exact`` is an exact fp32
forward (the floor FA2's backward reaches), ``emu_q`` / ``emu_k`` round one operand,
``emu_int8`` both, ``emu_qsmooth`` both with Q's token mean removed first and its
``q̄·kᵀ`` term added back per key in fp32 (SageAttention2's Q smoothing).

    make daemon-run ARGS="--stall-timeout 1200 project/int8_train/src/attn_fwd_grad_bench.py"
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

from bench._common import start_heartbeat  # noqa: E402
from library.env import resolve_under_home  # noqa: E402

GRADS = ("dq", "dk", "dv")
KERNEL_ARMS = ("int8", "tri_bf16_pv16acc", "qk16acc")
EMU_ARMS = ("emu_exact", "emu_q", "emu_k", "emu_int8", "emu_qsmooth")


def emu_round(x, mean, hm):
    """Dequantized token + rot int8 copy of ``x - mean``, as ``_quant_k_kernel`` makes it."""
    xf = x.float()
    if mean is not None:
        xf = xf - mean[:, None]
    h = hm.float()
    xr = xf.bfloat16().float() @ h
    s = xr.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127.0
    t = xr / s
    return (torch.sign(t) * torch.floor(t.abs() + 0.5) * s) @ h.T / h.shape[0]


def emu_attn(qh, kh, v, q, kmean, sm, bias=None):
    """fp32 attention of ``qh`` on centered keys ``kh`` (+ a per-key ``bias``) → O bf16
    and the natural logsumexp over the raw keys."""
    s = torch.einsum("blhd,bmhd->bhlm", qh, kh) * sm
    if bias is not None:
        s += bias[:, :, None, :] * sm
    lse = torch.logsumexp(s, -1)
    o = torch.einsum("bhlm,bmhd->blhd", torch.softmax(s, -1), v.float())
    lse += sm * torch.einsum("blhd,bhd->bhl", q.float(), kmean)
    return o.bfloat16(), lse


def arm_forward(name, q, k, v):
    from int8_attn import (  # sibling module
        LOG2E,
        attn_fwd,
        attn_fwd_train,
        hadamard,
        k_center,
        lse_natural,
    )

    sm = q.shape[-1] ** -0.5
    if name == "int8":
        o, lse, saved = attn_fwd_train(q, k, v)
        return o, lse_natural(q, lse, saved[4])
    if name == "tri_bf16_pv16acc":
        o, lse = attn_fwd(q, k, v, "bf16", "fp16acc")
        return o, lse / LOG2E
    if name == "qk16acc":
        o, lse = attn_fwd(q, k, v, "fp16acc", "fp16acc")
        return o, lse / LOG2E
    hm = hadamard(q.shape[-1], q.device)
    kmean = k_center(k)
    kc = k.float() - kmean[:, None]
    if name == "emu_exact":
        return emu_attn(q.float(), kc, v, q, kmean, sm)
    if name == "emu_q":
        return emu_attn(emu_round(q, None, hm), kc, v, q, kmean, sm)
    if name == "emu_k":
        return emu_attn(q.float(), emu_round(k, kmean, hm), v, q, kmean, sm)
    if name == "emu_int8":
        return emu_attn(
            emu_round(q, None, hm), emu_round(k, kmean, hm), v, q, kmean, sm
        )
    if name == "emu_qsmooth":
        qmean = q.mean(1, dtype=torch.float32)
        bias = torch.einsum("bhd,bmhd->bhm", qmean, kc)
        return emu_attn(
            emu_round(q, qmean, hm), emu_round(k, kmean, hm), v, q, kmean, sm, bias
        )
    raise ValueError(name)


def fa2_bwd(do, q, k, v, o, lse):
    from flash_attn.flash_attn_interface import _flash_attn_backward

    dq, dk, dv = (torch.empty_like(x) for x in (q, k, v))
    _flash_attn_backward(
        do, q, k, v, o, lse.contiguous(), dq, dk, dv, 0.0, q.shape[-1] ** -0.5,
        False, -1, -1, 0.0, None, False,
    )  # fmt: skip
    return dq, dk, dv


def mean_ratio(x):
    """‖token mean‖ / RMS token norm, averaged over heads."""
    xf = x.float()
    return float(
        (xf.mean(1).norm(dim=-1) / xf.norm(dim=-1).pow(2).mean(1).sqrt()).mean()
    )


def main():
    from attn_bwd_bench import ref_grads, relerr, timeit  # sibling module
    from flash_attn import flash_attn_func

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--capture", default="output/int8_train/attn_capture.pt")
    ap.add_argument("--out", default="project/int8_train/assets/attn_fwd_grad.json")
    args = ap.parse_args()
    start_heartbeat(label="attn_fwd_grad_bench")
    t0 = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sets = [
        s for s in torch.load(resolve_under_home(args.capture)) if s["kind"] == "self"
    ]
    dev = "cuda"
    arms = ("fa2", *KERNEL_ARMS, *EMU_ARMS)

    errs = []
    for i, s in enumerate(sets):
        q, k, v, do = (s[n].to(dev).contiguous() for n in ("q", "k", "v", "do"))
        refs = dict(zip(GRADS, ref_grads(q, k, v, do)))
        o_ref = torch.nn.functional.scaled_dot_product_attention(
            *(x.float().transpose(1, 2) for x in (q, k, v))
        ).transpose(1, 2)
        row = {n: s[n] for n in ("block", "sigma", "image")}
        row["q_mean_ratio"], row["k_mean_ratio"] = mean_ratio(q), mean_ratio(k)
        for a in arms:
            if a == "fa2":
                leaves = [x.detach().requires_grad_() for x in (q, k, v)]
                o = flash_attn_func(*leaves)
                got = torch.autograd.grad(o, leaves, do)
            else:
                o, lse = arm_forward(a, q, k, v)
                got = fa2_bwd(do, q, k, v, o, lse)
            row[f"{a}.o"] = relerr(o, o_ref)
            for g, x in zip(GRADS, got):
                row[f"{a}.{g}"] = relerr(x, refs[g])
            del o, got
            torch.cuda.empty_cache()
        errs.append(row)
        print(
            f"err {i + 1}/{len(sets)} b{row['block']} σ={row['sigma']} "
            f"q̄={row['q_mean_ratio']:.2f} k̄={row['k_mean_ratio']:.2f} "
            + " ".join(
                f"{a}="
                + "/".join(f"{row[f'{a}.{g}'] / row[f'fa2.{g}']:.2f}" for g in GRADS)
                for a in arms[1:]
            ),
            flush=True,
        )
        del q, k, v, do, refs, o_ref

    keys = ("o", *GRADS)
    medians = defaultdict(dict)
    for sigma in sorted({r["sigma"] for r in errs}):
        rows = [r for r in errs if r["sigma"] == sigma]
        for a in arms:
            medians[str(sigma)][a] = {
                g: statistics.median(r[f"{a}.{g}"] for r in rows) for g in keys
            }

    s = next(x for x in sets if x["image"] == 0)
    q, k, v = (s[n].to(dev).contiguous() for n in "qkv")
    t = {"fa2": timeit(lambda: flash_attn_func(q, k, v))}
    for a in KERNEL_ARMS:
        t[a] = timeit(lambda: arm_forward(a, q, k, v))
    print(
        f"speed {list(q.shape)}: " + " ".join(f"{a}={ms:.3f}ms" for a, ms in t.items()),
        flush=True,
    )

    gate = {}
    for a in arms[1:]:
        ratios = {g: max(m[a][g] / m["fa2"][g] for m in medians.values()) for g in keys}
        entry = {"worst_ratio_to_fa2": ratios}
        ok = max(ratios[g] for g in GRADS) <= 2.0
        if a in t:
            entry["fwd_speedup"] = t["fa2"] / t[a]
            ok = ok and entry["fwd_speedup"] >= 1.3
        entry["pass"] = ok
        gate[a] = entry
    result = {
        "shape_q": list(q.shape),
        "errors": errs,
        "medians": medians,
        "speed_ms": t,
        "gate": gate,
        "wall_s": time.perf_counter() - t0,
    }
    Path(resolve_under_home(args.out)).write_text(json.dumps(result, indent=1))
    print(json.dumps(gate, indent=1))


if __name__ == "__main__":
    main()
