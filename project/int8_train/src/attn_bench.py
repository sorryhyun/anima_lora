"""Phase 5a: Triton int8-QKᵀ attention forward against FA2 (proposal.md §Phase 5).

Capture: the bf16 DiT (flash, eager, non-reentrant checkpointing) runs one training
forward and backward per (σ, image) on real cached batches. A wrapper on ``dispatch_attention``
keeps q, k, v and the output gradient dO of four self-attention and four
cross-attention calls (blocks 0, n/3, 2n/3, n−1), saved to ``--capture`` for phase 5b.

Kernels, after the DiT is freed. Every arm of ``int8_attn.attn_fwd`` and FA2 is scored
against fp32 SDPA (TF32 off) as ``‖o − o_fp32‖ / ‖o_fp32‖``, per captured call. Gate:
median over calls, per kind and σ, at most 2x FA2's. Speed on the first image's shapes,
with the K/V quantization included, and the quantization alone.

    make daemon-run ARGS="--stall-timeout 1200 project/int8_train/src/attn_bench.py"
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

# run 1 (20261030-123855-73488c) also had pv fp16 (as accurate as bf16, slower) and
# pv int8 (20–80x FA2's error); run 2 (20261030-124331-70b381) pv bf16 at both
# granularities (as accurate as fp16acc, slower). All dropped.
ARMS = [
    ("bf16", "bf16", "block", False),
    ("int8", "fp16acc", "block", False),
    ("int8", "fp16acc", "token", False),
    ("int8", "fp16acc", "block", True),
    ("int8", "fp16acc", "token", True),
]


def arm_name(qk: str, pv: str, gran: str, rot: bool) -> str:
    if qk != "int8":
        return f"qk_{qk}.pv_{pv}"
    return f"qk_{qk}.pv_{pv}.{gran}" + (".rot" if rot else "")


def capture(args) -> None:
    from bench._anima import build_anima
    from grad_error import load_batch, pick_images  # sibling module
    from networks import attention_dispatch

    bundle = build_anima(args, adapter=None, train_mode=True)
    anima, device = bundle.anima, bundle.device
    n = len(anima.blocks)
    keep = sorted({0, n // 3, 2 * n // 3, n - 1})
    images = pick_images(
        Path(resolve_under_home(args.cache_root)),
        args.n_images,
        args.min_tokens,
        args.max_tokens,
        args.seed,
    )
    orig = attention_dispatch.dispatch_attention
    state = {"self": 0, "cross": 0, "tag": None, "recompute": False}
    sets: list[dict] = []

    def wrapped(qkv_or_q, k=None, v=None, attn_params=None, drop_rate=0.0):
        q, k, v = qkv_or_q if isinstance(qkv_or_q, list) else (qkv_or_q, k, v)
        if state["recompute"]:  # checkpoint recompute in backward: dO hooks are set
            return orig([q, k, v], attn_params=attn_params, drop_rate=drop_rate)
        kind = "self" if k.shape[1] == q.shape[1] else "cross"
        block = state[kind]
        state[kind] += 1
        rec = None
        if block in keep:
            rec = dict(
                kind=kind,
                block=block,
                **state["tag"],
                q=q.detach().cpu(),
                k=k.detach().cpu(),
                v=v.detach().cpu(),
            )
            sets.append(rec)
        x = orig([q, k, v], attn_params=attn_params, drop_rate=drop_rate)
        if rec is not None and x.requires_grad:
            x.register_hook(lambda g: rec.__setitem__("do", g.reshape(q.shape).cpu()))
        return x

    attention_dispatch.dispatch_attention = wrapped
    for sigma_v in [float(s) for s in args.sigmas.split(",")]:
        sigma = torch.tensor([sigma_v], device=device)
        for j, ci in enumerate(images):
            state.update(
                self=0, cross=0, tag={"sigma": sigma_v, "image": j}, recompute=False
            )
            latents, crossattn, noise = load_batch(ci, j, args.seed, device)
            s4 = sigma.view(-1, 1, 1, 1)
            noisy = ((1.0 - s4) * latents + s4 * noise).unsqueeze(2).bfloat16()
            noisy.requires_grad_(True)
            pad = torch.zeros(
                1, 1, *latents.shape[-2:], dtype=torch.bfloat16, device=device
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = anima(noisy, sigma, crossattn.bfloat16(), padding_mask=pad)
            state["recompute"] = True
            F.mse_loss(pred.squeeze(2).float(), noise - latents).backward()
            print(f"captured σ={sigma_v} image {j}: {len(sets)} calls", flush=True)
    attention_dispatch.dispatch_attention = orig
    missing = [i for i, s in enumerate(sets) if "do" not in s]
    if missing:
        raise SystemExit(f"{len(missing)} captured calls got no dO")
    path = Path(resolve_under_home(args.capture))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sets, path)
    print(f"saved {len(sets)} calls to {path}", flush=True)
    del anima, bundle
    torch.cuda.empty_cache()


def ref_fp32(q, k, v):
    o = F.scaled_dot_product_attention(
        q.float().transpose(1, 2), k.float().transpose(1, 2), v.float().transpose(1, 2)
    )
    return o.transpose(1, 2)


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


def kernels(args) -> dict:
    from flash_attn import flash_attn_func

    from int8_attn import attn_fwd, quant_k, quant_vt  # sibling module

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sets = torch.load(resolve_under_home(args.capture))
    dev = "cuda"

    errs = []
    for i, s in enumerate(sets):
        q, k, v = (s[n].to(dev).contiguous() for n in "qkv")
        ref = ref_fp32(q, k, v)
        row = {n: s[n] for n in ("kind", "block", "sigma", "image")}
        row["fa2"] = relerr(flash_attn_func(q, k, v), ref)
        for qk, pv, gran, rot in ARMS:
            o = attn_fwd(q, k, v, qk, pv, gran=gran, rot=rot)[0]
            row[arm_name(qk, pv, gran, rot)] = relerr(o, ref)
        errs.append(row)
        print(
            f"err {i + 1}/{len(sets)} {row['kind']} b{row['block']} σ={row['sigma']}: "
            + " ".join(
                f"{a}={row[a]:.4f}" for a in ["fa2"] + [arm_name(*x) for x in ARMS]
            ),
            flush=True,
        )
        del q, k, v, ref

    arms = ["fa2"] + [arm_name(*x) for x in ARMS]
    medians = defaultdict(dict)
    for kind in ("self", "cross"):
        for sigma in sorted({r["sigma"] for r in errs}):
            rows = [r for r in errs if r["kind"] == kind and r["sigma"] == sigma]
            medians[kind][str(sigma)] = {
                a: statistics.median(r[a] for r in rows) for a in arms
            }
    gate = {}
    for a in arms[1:]:
        ratios = [m[a] / m["fa2"] for kind in medians.values() for m in kind.values()]
        gate[a] = {"worst_ratio_to_fa2": max(ratios), "pass": max(ratios) <= 2.0}

    speed = {}
    for kind in ("self", "cross"):
        s = next(x for x in sets if x["kind"] == kind and x["image"] == 0)
        q, k, v = (s[n].to(dev).contiguous() for n in "qkv")
        flops = 4 * q.shape[0] * q.shape[2] * q.shape[1] * k.shape[1] * q.shape[3]
        t = {"fa2": timeit(lambda: flash_attn_func(q, k, v))}
        for qk, pv, gran, rot in ARMS:
            t[arm_name(qk, pv, gran, rot)] = timeit(
                lambda: attn_fwd(q, k, v, qk, pv, gran=gran, rot=rot)
            )
        t["prep.quant_k"] = timeit(lambda: quant_k(k))
        t["prep.quant_k.token.rot"] = timeit(lambda: quant_k(k, "token", True))
        t["prep.quant_vt"] = timeit(lambda: quant_vt(v))
        speed[kind] = {
            "shape_q": list(q.shape),
            "lk": k.shape[1],
            "ms": t,
            "tflops": {a: flops / (ms * 1e9) for a, ms in t.items() if "prep" not in a},
            "speedup_vs_fa2": {
                a: t["fa2"] / ms for a, ms in t.items() if "prep" not in a
            },
        }
        print(
            f"speed {kind} {list(q.shape)} lk={k.shape[1]}: "
            + " ".join(f"{a}={ms:.3f}ms" for a, ms in t.items()),
            flush=True,
        )
    return {"errors": errs, "medians": medians, "gate": gate, "speed": speed}


def main():
    from bench._anima import add_common_args, add_model_args

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(ap, vae=False, text_encoder=False)
    add_common_args(ap, include_compile=False, include_checkpointing=False)
    ap.add_argument("--cache_root", default="post_image_dataset/lora")
    ap.add_argument("--sigmas", default="0.1,0.5,0.9")
    ap.add_argument("--n_images", type=int, default=2)
    ap.add_argument("--min_tokens", type=int, default=3500)
    ap.add_argument("--max_tokens", type=int, default=4608)
    ap.add_argument("--capture", default="output/int8_train/attn_capture.pt")
    ap.add_argument("--recapture", action="store_true")
    ap.add_argument("--out", default="project/int8_train/reports/attn_fwd.json")
    args = ap.parse_args()
    start_heartbeat(label="attn_bench")
    t0 = time.perf_counter()

    if args.recapture or not Path(resolve_under_home(args.capture)).exists():
        # eager 4k-token activations do not fit in 16 GB without it
        args.gradient_checkpointing = True
        capture(args)
    result = kernels(args)
    result["wall_s"] = time.perf_counter() - t0
    out = Path(resolve_under_home(args.out))
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"gate": result["gate"], "speed": result["speed"]}, indent=1))


if __name__ == "__main__":
    main()
