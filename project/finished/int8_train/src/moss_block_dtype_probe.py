"""MOSS block dtype trace (eager and compiled) and fp16 GEMM timing for the LoRA / int8 output."""

import json
import statistics
import time

import comfy_kitchen
import torch
import torch.nn.functional as F

import library.moss.model as M
from library.int8 import linear as L
from library.moss.dit_int8 import base as B

dev = torch.device("cuda")
print("flash", M.flash_attn_func, flush=True)
torch.manual_seed(0)
DIM, FFN, HEADS, TOK = 3072, 8192, 24, 4608  # 4608 tokens: 72 x 64 grid

block = M._Block(DIM, FFN, HEADS).to(dev, torch.bfloat16).eval().requires_grad_(False)
for p in block.parameters():
    p.data = p.data.float() if p.ndim == 1 else p.data
for n, m in block.proj.items():
    torch.nn.init.normal_(m.weight, std=1 / DIM**0.5)
    m.weight.data = m.weight.data.to(torch.bfloat16)
B.fuse_qkv_(block, "full", {}, "layers.0")
for p in B.REST:
    B.convert_moss_(block.proj[p], "full", None)
block.norm1.weight.data = block.norm1.weight.data.float()
block.norm2.weight.data = block.norm2.weight.data.float()
block.q_norm.weight.data = block.q_norm.weight.data.float()
block.k_norm.weight.data = block.k_norm.weight.data.float()

kernel_in = []
_real = comfy_kitchen.int8_linear


def _spy(x, *a, **k):
    kernel_in.append(
        {
            "dtype": str(x.dtype),
            "bf16_representable": (x == x.to(torch.bfloat16).float())
            .float()
            .mean()
            .item(),
            "out": str(k.get("out_dtype")),
        }
    )
    return _real(x, *a, **k)


L.comfy_kitchen = type("K", (), {"int8_linear": staticmethod(_spy)})
fa_log = []
_fa = M.flash_attn_func


def _fa_spy(q, k, v, *a, **kw):
    o = _fa(q, k, v, *a, **kw)
    fa_log.append(
        {"q": str(q.dtype), "k": str(k.dtype), "v": str(v.dtype), "o": str(o.dtype)}
    )
    return o


M.flash_attn_func = _fa_spy

x = torch.randn(1, TOK, DIM, device=dev, dtype=torch.float32).requires_grad_(True)
mod = (torch.randn(1, 6, DIM, device=dev) * 0.05).float()
rope = (torch.randn(1, TOK, 128, device=dev)).float()
rope = torch.cat((rope.sin()[..., :64], rope.cos()[..., :64]), -1)

kernel_in.clear()
out = block(x, rope, mod)
eager = {"kernels": list(kernel_in), "fa2": list(fa_log), "out": str(out.dtype)}
kernel_in.clear()
out.square().mean().backward()
eager["backward_kernels"] = list(kernel_in)

fa_log.clear()
kernel_in.clear()
compiled = torch.compile(block.forward)
out = compiled(x, rope, mod)
comp = {"kernels": list(kernel_in), "out": str(out.dtype)}
kernel_in.clear()
out.square().mean().backward()
comp["backward_kernels"] = list(kernel_in)
L.comfy_kitchen = comfy_kitchen
M.flash_attn_func = _fa
print(json.dumps({"eager": eager, "compiled": comp}, indent=1), flush=True)


def timed(fn, n=200, warm=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3


matmul = torch.backends.cuda.matmul
rank = 32
xr = torch.randn(TOK, DIM, device=dev)
xr[:, :8] *= 30
down = torch.randn(DIM, rank, device=dev) / DIM**0.5
up = torch.randn(rank, DIM, device=dev) / rank**0.5
lxr = xr @ down
dy = (torch.randn(TOK, DIM, device=dev) * 1e-7).float()


def rel(a, ref):
    return ((a.float() - ref).norm() / ref.norm()).item()


matmul.allow_bf16_reduced_precision_reduction = False
ref_lx = xr @ down
ref_d = lxr @ up
matmul.allow_bf16_reduced_precision_reduction = True
rows = []
for name, dt, acc16 in (
    ("bf16", torch.bfloat16, False),
    ("fp16_acc32", torch.float16, False),
    ("fp16_acc16", torch.float16, True),
):
    matmul.allow_fp16_accumulation = acc16
    X, D, LX, U = xr.to(dt), down.to(dt), lxr.to(dt), up.to(dt)
    rows.append(
        {
            "arm": name,
            "down_ms": timed(lambda: F.linear(X, D.T)),
            "up_ms": timed(lambda: F.linear(LX, U.T)),
            "down_err": rel(F.linear(X, D.T), ref_lx),
            "up_err": rel(F.linear(LX, U.T), ref_d),
        }
    )
matmul.allow_fp16_accumulation = False
print(json.dumps({"lora_fwd_gemm": rows}, indent=1), flush=True)

# LoRA backward's M-long GEMMs, bf16, for the share of "other GEMM"
dY = (dy * 1e7).to(torch.bfloat16)
back = {
    "dU_K=M_ms": timed(lambda: dY.T @ lxr.to(torch.bfloat16)),
    "dD_K=M_ms": timed(lambda: xr.to(torch.bfloat16).T @ lxr.to(torch.bfloat16)),
    "dlx_K=N_ms": timed(lambda: dY @ up.T.to(torch.bfloat16)),
}
print(json.dumps({"lora_bwd_gemm_bf16": back}, indent=1), flush=True)

# int8 kernel output dtype, forward shape of block.qkv
w = block.qkv.wq
s = block.qkv.s
xf = torch.randn(TOK, DIM, device=dev)
int8_rows = {}
for name, dt in (
    ("bf16", torch.bfloat16),
    ("fp16", torch.float16),
    ("fp32", torch.float32),
):
    try:
        int8_rows[name] = timed(
            lambda: comfy_kitchen.int8_linear(
                xf, w, s, out_dtype=dt, convrot=True, convrot_groupsize=256
            )
        )
    except Exception as e:
        int8_rows[name] = repr(e)[:200]
print(json.dumps({"int8_out_ms": int8_rows}, indent=1), flush=True)
