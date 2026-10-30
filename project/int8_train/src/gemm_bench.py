"""Phase 0 microbench (reports/gemm_microbench.md). Anima block GEMMs: bf16 vs int8 (raw _int_mm, ConvRot W8A8 via comfy_kitchen) vs fp8.
``gemm_bench.py <M> moss`` runs MOSS-Image block 10's six linears instead.

Forward Y = X W^T and backward dX = dY W (frozen base, LoRA training: no dW).
Backward int8 reuses the same kernel with a second int8 copy of W^T, rotated on N.
"""

import json
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

import comfy_kitchen
from library.env import resolve_under_home
from library.moss.int8 import quantize_weight

torch.manual_seed(0)
dev = "cuda"
M = int(sys.argv[1]) if len(sys.argv) > 1 else 4096  # 1024^2 -> 64x64 tokens
LINE = sys.argv[2] if len(sys.argv) > 2 else "anima"
CTX = 512


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


def relerr(a, ref):
    return ((a.float() - ref).norm() / ref.norm()).item()


if LINE == "moss":
    MOSS = resolve_under_home("models/moss_image/MOSS-Image-Base/transformer")
    wmap = json.loads(
        (MOSS / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )["weight_map"]

    def get(key):
        with safe_open(str(MOSS / wmap[key]), "pt", device="cpu") as f:
            return f.get_tensor(key)

    layers = {
        name: (get(f"layers.10.proj.{name}.weight"), M, True)
        for name in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")
    }
else:
    CKPT = str(
        resolve_under_home("models/diffusion_models/anima-base-v1.0.safetensors")
    )
    f = safe_open(CKPT, "pt", device="cpu")
    P = "net.blocks.10."
    layers = {
        "sa.qkv": (
            torch.cat([f.get_tensor(P + f"self_attn.{c}_proj.weight") for c in "qkv"]),
            M,
            True,
        ),
        "sa.out": (f.get_tensor(P + "self_attn.output_proj.weight"), M, True),
        "ca.q": (f.get_tensor(P + "cross_attn.q_proj.weight"), M, True),
        "ca.kv": (
            torch.cat([f.get_tensor(P + f"cross_attn.{c}_proj.weight") for c in "kv"]),
            CTX,
            False,
        ),
        "ca.out": (f.get_tensor(P + "cross_attn.output_proj.weight"), M, True),
        "mlp.1": (f.get_tensor(P + "mlp.layer1.weight"), M, True),
        "mlp.2": (f.get_tensor(P + "mlp.layer2.weight"), M, True),
    }

rows = []
tot = {
    k: 0.0
    for k in ("bf16_f", "ck_f", "bf16_b", "ck_b", "raw_bf16", "raw_int8", "raw_fp8")
}
for name, (W, m, needs_dx) in layers.items():
    W = W.to(dev, torch.bfloat16)
    N, K = W.shape
    # activations with outlier channels; gradients heavy-tailed (student-t)
    x = torch.randn(m, K, device=dev)
    x[:, torch.randperm(K, device=dev)[:8]] *= 30
    x = x.bfloat16()
    dy = torch.distributions.StudentT(3.0).sample((m, N)).to(dev).bfloat16() * 1e-3

    wq, ws = quantize_weight(W)
    wq, ws = wq.to(dev), ws.to(dev).float()
    wtq, wts = quantize_weight(W.t().contiguous())
    wtq, wts = wtq.to(dev), wts.to(dev).float()

    def ck_f(x=x, wq=wq, ws=ws):
        return comfy_kitchen.int8_linear(
            x, wq, ws, out_dtype=torch.bfloat16, convrot=True, convrot_groupsize=256
        )

    def ck_b(dy=dy, wtq=wtq, wts=wts):
        return comfy_kitchen.int8_linear(
            dy, wtq, wts, out_dtype=torch.bfloat16, convrot=True, convrot_groupsize=256
        )

    def bf_f(x=x, W=W):
        return F.linear(x, W)

    def bf_b(dy=dy, W=W):
        return dy @ W

    ref_f = x.float() @ W.float().t()
    ref_b = dy.float() @ W.float()
    r = dict(
        layer=name,
        M=m,
        K=K,
        N=N,
        bf16_f=timeit(bf_f),
        ck_f=timeit(ck_f),
        err_bf16_f=relerr(bf_f(), ref_f),
        err_ck_f=relerr(ck_f(), ref_f),
    )
    if needs_dx:
        r.update(
            bf16_b=timeit(bf_b),
            ck_b=timeit(ck_b),
            err_bf16_b=relerr(bf_b(), ref_b),
            err_ck_b=relerr(ck_b(), ref_b),
        )
    # raw GEMM throughput (pre-quantized operands, no scale/rotation work)
    xi = torch.randint(-127, 127, (m, K), device=dev, dtype=torch.int8)
    wi = torch.randint(-127, 127, (N, K), device=dev, dtype=torch.int8)
    r["raw_bf16"] = timeit(lambda: x @ W.t())
    r["raw_int8"] = timeit(lambda: torch._int_mm(xi, wi.t()))
    try:
        x8 = x.to(torch.float8_e4m3fn)
        w8 = W.to(torch.float8_e4m3fn)
        sa = torch.ones(m, 1, device=dev)
        sb = torch.ones(1, N, device=dev)
        r["raw_fp8"] = timeit(
            lambda: torch._scaled_mm(x8, w8.t(), sa, sb, out_dtype=torch.bfloat16)
        )
    except Exception as ex:  # noqa: BLE001
        r["raw_fp8"] = float("nan")
        r["fp8_err"] = str(ex)[:120]
    for k in tot:
        tot[k] += r.get(k, 0.0)
    rows.append(r)

for r in rows:
    print(
        json.dumps(
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
        )
    )
flops = sum(2 * r["M"] * r["K"] * r["N"] for r in rows)
print("TOTAL ms per block:", {k: round(v, 3) for k, v in tot.items()})
print(
    "raw TFLOPS bf16 %.1f int8 %.1f fp8 %.1f"
    % tuple(flops / (tot[k] * 1e9) for k in ("raw_bf16", "raw_int8", "raw_fp8"))
)
print(
    "speedup fwd %.2fx  bwd-dX %.2fx"
    % (tot["bf16_f"] / tot["ck_f"], tot["bf16_b"] / tot["ck_b"])
)
