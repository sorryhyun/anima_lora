"""One int8 linear for MOSS ``proj.q/k/v`` against three (step_headroom.md, candidate 3).

Block 10's GPTQ forward copies, kernel ``comfy_kitchen.int8_linear``:

- forward: one 3072→9216 call on the concatenated rows against three 3072→3072 calls,
  and whether the outputs are bit-identical;
- dX: one ``dY (9216) @ W`` call on a concatenated, round-to-nearest ``wq_t`` against three
  ``dY (3072) @ W`` calls summed, with the error against fp32. dY is synthetic here, and its
  q / k / v parts carry the ``--dv_ratio`` scale between them.

    make daemon-run ARGS="project/int8_train/src/qkv_fuse_bench.py [--tokens 4600]"
"""

from __future__ import annotations

import argparse
import json

import comfy_kitchen
import torch
from safetensors import safe_open

from library.env import resolve_under_home
from library.int8.linear import quantize
from library.moss.dit_int8.base import forward_file, load_copies
from library.moss.requests import TrainRequest

GROUP = 256


def kernel(x, wq, s):
    return comfy_kitchen.int8_linear(
        x,
        wq,
        s,
        out_dtype=torch.bfloat16,
        convrot=True,
        convrot_groupsize=GROUP,
    )


def timed(fn, warm=10, iters=50):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def relerr(a, ref):
    return ((a.float() - ref).norm() / ref.norm()).item()


def weights(block: int) -> list[torch.Tensor]:
    root = resolve_under_home("models/moss_image/MOSS-Image-Base/transformer")
    index = json.loads(
        (root / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )["weight_map"]
    out = []
    for name in "qkv":
        key = f"layers.{block}.proj.{name}.weight"
        with safe_open(str(root / index[key]), "pt", device="cuda") as f:
            out.append(f.get_tensor(key).to(torch.bfloat16))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=4600)
    ap.add_argument("--block", type=int, default=10)
    ap.add_argument("--dv_ratio", type=float, default=1.0)
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda"

    w = weights(args.block)
    copies, _meta = load_copies(forward_file(TrainRequest()))
    fwd = [
        tuple(t.to(dev) for t in copies[f"layers.{args.block}.proj.{p}"]) for p in "qkv"
    ]
    fused_wq = torch.cat([c[0] for c in fwd])
    fused_s = torch.cat([c[1] for c in fwd])

    K = w[0].shape[1]
    x = torch.randn(args.tokens, K, device=dev)
    x[:, torch.randperm(K, device=dev)[:8]] *= 30
    x = x.float()

    three_f = lambda: [kernel(x, wq, s) for wq, s in fwd]  # noqa: E731
    one_f = lambda: kernel(x, fused_wq, fused_s)  # noqa: E731
    t3f, t1f = timed(three_f), timed(one_f)
    a = torch.cat(three_f(), -1)
    b = one_f()
    print(
        json.dumps(
            {
                "fwd_three_ms": round(t3f, 3),
                "fwd_one_ms": round(t1f, 3),
                "fwd_ratio": round(t3f / t1f, 3),
                "fwd_saved_42_blocks_ms": round(42 * (t3f - t1f), 1),
                "fwd_bit_identical": bool(torch.equal(a, b)),
                "fwd_max_abs_diff": (a.float() - b.float()).abs().max().item(),
            }
        ),
        flush=True,
    )

    bwd = [quantize(wi.t().contiguous()) for wi in w]
    fused_w = torch.cat(w).t().contiguous()  # (K, 9216)
    fused_bwd = quantize(fused_w)

    dy = (
        torch.distributions.StudentT(3.0)
        .sample((args.tokens, 3 * K))
        .to(dev)
        .bfloat16()
        * 1e-3
    )
    dy[:, 2 * K :] *= args.dv_ratio
    parts = dy.chunk(3, -1)

    def three_b():
        dx = kernel(parts[0], *bwd[0])
        dx = dx + kernel(parts[1], *bwd[1])
        return dx + kernel(parts[2], *bwd[2])

    one_b = lambda: kernel(dy, *fused_bwd)  # noqa: E731
    t3b, t1b = timed(three_b), timed(one_b)
    ref = dy.float() @ fused_w.float().t()
    print(
        json.dumps(
            {
                "dx_three_ms": round(t3b, 3),
                "dx_one_ms": round(t1b, 3),
                "dx_ratio": round(t3b / t1b, 3),
                "dx_saved_42_blocks_ms": round(42 * (t3b - t1b), 1),
                "dx_err_three": round(relerr(three_b(), ref), 5),
                "dx_err_one": round(relerr(one_b(), ref), 5),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
