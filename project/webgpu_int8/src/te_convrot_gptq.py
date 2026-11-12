"""ConvRot GPTQ for the Anima Qwen3-0.6B TE on ORT-web's MatMulNBits grid.

Grid (what the browser computes, ORT web 1.30 WebGPU EP):
  - input x rotated by blockdiag(H256) (Sylvester Hadamard / 16), in-graph;
  - weight int8 (uint8 + default zero point 128), one fp16 scale per (row, wblock) of K;
  - DP4A path (accuracy_level 4): the rotated activation rounded to int8 per token per
    128-element K block (absmax / 127).  Without DP4A (Apple, no subgroups) the weight
    is dequantised and the activation stays fp16 -> mode "w8".

Sequential layer-by-layer GPTQ (static block scales, act order), optional act-aware
least-squares target as in project/finished/moss_lora/src/te_gptq.py. Eval on held-out
captions: per-token rel L2 of last_hidden_state vs fp32.

    make daemon-run ARGS="--stall-timeout 1800 project/webgpu_int8/src/te_convrot_gptq.py \
        --variants aa32k --save aa32k"     # -> output/webgpu_int8/te_int8.aa32k.safetensors

Shipped: aa32k (act-aware, 32-wide weight blocks, layer 2 down_proj kept fp16).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time

import torch
from safetensors.torch import save_file

from library.anima.weights import load_qwen3_text_encoder
from library.env import resolve_under_home

TE = "models/text_encoders/qwen_3_06b_base.safetensors"
LINS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
ROT = 256
ABLK = 128


def hadamard(n: int) -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / n**0.5


H256 = None


def rot(x: torch.Tensor) -> torch.Tensor:
    s = x.shape
    return (x.reshape(*s[:-1], s[-1] // ROT, ROT) @ H256).reshape(s)


def act_qdq(x: torch.Tensor) -> torch.Tensor:
    s = x.shape
    g = x.reshape(*s[:-1], s[-1] // ABLK, ABLK)
    sc = (g.abs().amax(-1, keepdim=True) / 127).clamp(min=1e-30)
    return ((g / sc).round().clamp(-127, 127) * sc).reshape(s)


def block_scale(w: torch.Tensor, b: int) -> torch.Tensor:
    o, k = w.shape
    return w.reshape(o, k // b, b).abs().amax(-1).clamp(min=1e-12) / 127


def qdq_w(w: torch.Tensor, sc: torch.Tensor, b: int) -> torch.Tensor:
    o, k = w.shape
    s = sc.repeat_interleave(b, 1)
    return (w / s).round().clamp(-128, 127) * s


def gptq(W, H, sc, b, damp=0.01, blk=128):
    """Static block scales ``sc`` (out, K/b); act order. Returns dequantised Q."""
    W, H = W.clone(), H.clone()
    n = W.shape[1]
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    perm = torch.argsort(torch.diag(H), descending=True)
    W, H = W[:, perm], H[perm][:, perm]
    S = sc.repeat_interleave(b, 1)[:, perm]
    H.diagonal().add_(damp * H.diagonal().mean())
    Hinv = torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True
    )
    Q = torch.zeros_like(W)
    for i1 in range(0, n, blk):
        i2 = min(i1 + blk, n)
        W1, Hi = W[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
        Q1, E1 = torch.zeros_like(W1), torch.zeros_like(W1)
        for j in range(i2 - i1):
            s = S[:, i1 + j]
            qj = (W1[:, j] / s).round().clamp(-128, 127) * s
            e = (W1[:, j] - qj) / Hi[j, j]
            Q1[:, j], E1[:, j] = qj, e
            W1[:, j:] -= e[:, None] * Hi[j, j:][None, :]
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q[:, torch.argsort(perm)]


class QLin(torch.nn.Module):
    """Rotated-basis dequantised weight; mode w8 / w8a8 set on the class."""

    mode = "w8a8"

    def __init__(self, q_rot: torch.Tensor):
        super().__init__()
        self.register_buffer("w", q_rot)

    def forward(self, x):
        xr = rot(x.float())
        if QLin.mode == "w8a8":
            xr = act_qdq(xr)
        return (xr @ self.w.T).to(x.dtype)


def captions(seed: int):
    caps = sorted(
        glob.glob(
            str(resolve_under_home("post_image_dataset/resized")) + "/**/*.txt",
            recursive=True,
        )
    )
    caps = [c for c in caps if not c.endswith(".variants.txt")]
    random.Random(seed).shuffle(caps)
    out = []
    for c in caps:
        t = open(c, encoding="utf-8").read().strip()
        if t:
            out.append(t)
    return out


@torch.no_grad()
def capture(model, ids_list):
    got = {}

    def stub(hidden_states, **kw):
        got["h"], got["kw"] = hidden_states, kw
        raise StopIteration

    model.layers[0].forward = stub
    out = []
    for ids in ids_list:
        try:
            model(input_ids=ids, use_cache=False)
        except StopIteration:
            pass
        out.append((got["h"], got["kw"]))
    del model.layers[0].forward
    return out


def layer_out(r):
    return r[0] if isinstance(r, tuple) else r


@torch.no_grad()
def final(model, ids_list):
    return [
        model(input_ids=i, use_cache=False).last_hidden_state.float()[0]
        for i in ids_list
    ]


def stats(ref, out):
    rel, rel1, r0, cos = [], [], [], []
    for r, o in zip(ref, out):
        d = (o - r).norm(dim=-1) / r.norm(dim=-1)
        rel.append(d.mean())
        rel1.append(d[1:].mean())
        r0.append(d[0])
        cos.append(torch.nn.functional.cosine_similarity(o, r, dim=-1)[1:].mean())
    f = lambda x: round(torch.stack(x).mean().item(), 5)  # noqa: E731
    return {"rel": f(rel), "rel_tok1+": f(rel1), "rel_tok0": f(r0), "cos_tok1+": f(cos)}


@torch.no_grad()
def quantize(model, orig, calib, var, dev):
    """Fresh copy of ``orig`` weights, sequential GPTQ; returns {key: (q int8, scale)}."""
    for i, layer in enumerate(model.layers):
        for n in LINS:
            p, _, c = n.rpartition(".")
            setattr(layer.get_submodule(p), c, orig_lin[(i, n)])
    model.load_state_dict(orig)
    b = var["wblock"]
    samples = capture(model, calib)
    out = {}
    report = []
    for i, layer in enumerate(model.layers):
        lins = {n: layer.get_submodule(n) for n in LINS}
        acc = {}
        hooks = []

        def mk(n):
            def hook(_m, a):
                x = a[0].reshape(-1, a[0].shape[-1]).float()
                xr = rot(x)
                d = acc.setdefault(n, {})
                am = x.abs().amax(-1)
                d["r0"] = max(
                    d.get("r0", 0.0),
                    (am[0] / am[1:].amax().clamp(min=1e-12)).item()
                    if am.numel() > 1
                    else 0.0,
                )
                if var["act_aware"]:
                    xq = act_qdq(xr)
                    for k, (u, v) in {
                        "Hq": (xq, xq),
                        "G": (xq, xr),
                        "Hr": (xr, xr),
                    }.items():
                        d[k] = d.get(k, 0) + u.T @ v
                else:
                    d["Hr"] = d.get("Hr", 0) + xr.T @ xr

            return hook

        for n, m in lins.items():
            hooks.append(m.register_forward_pre_hook(mk(n)))
        for h, kw in samples:
            layer(h, **kw)
        for hk in hooks:
            hk.remove()
        for n, m in lins.items():
            if n == "mlp.down_proj" or acc[n]["r0"] > 4:
                print(f"L{i} {n} tok0/rest amax {acc[n]['r0']:.1f}", flush=True)
            if acc[n]["r0"] > var.get("keep", 1e9):
                report.append((i, n, -1.0))
                continue
            Wr = rot(m.weight.float())
            d = acc[n]
            if var["act_aware"]:
                Hq = d["Hq"].double()
                lam, eye = 1e-4, torch.eye(Wr.shape[1], device=dev, dtype=torch.float64)
                while True:
                    L, info = torch.linalg.cholesky_ex(
                        Hq + lam * Hq.diagonal().mean() * eye
                    )
                    if info == 0:
                        break
                    lam *= 10
                if lam > 1e-4:
                    print(f"L{i} {n} ls ridge raised to {lam:g}", flush=True)
                W = torch.cholesky_solve(d["G"].double() @ Wr.T.double(), L).T.float()
                H = d["Hq"]
            else:
                W, H = Wr, d["Hr"]
            sc = block_scale(W, b)
            Q = qdq_w(W, sc, b) if var["rtn"] else gptq(W, H, sc, b)
            ref = ((Wr @ d["Hr"]) * Wr).sum()
            if var["act_aware"]:
                err = (
                    ((Q @ d["Hq"]) * Q).sum() - 2 * ((Q @ d["G"]) * Wr).sum() + ref
                ) / ref
            else:
                D = Q - Wr
                err = ((D @ d["Hr"]) * D).sum() / ref
            report.append((i, n, round(err.item(), 6)))
            q = (Q / sc.repeat_interleave(b, 1)).round().clamp(-128, 127).to(torch.int8)
            out[f"layers.{i}.{n}"] = (q.cpu(), sc.half().cpu())
            p, _, c = n.rpartition(".")
            setattr(layer.get_submodule(p), c, QLin(Q))
        QLin.mode = "w8a8"
        samples = [(layer_out(layer(h, **kw)), kw) for h, kw in samples]
    return out, report


orig_lin = {}


def main():
    global H256
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_calib", type=int, default=256)
    ap.add_argument("--n_eval", type=int, default=64)
    ap.add_argument("--variants", default="rtn128,gptq128,aa128,aa32")
    ap.add_argument("--save", default="", help="variant name to save")
    ap.add_argument("--out", default="output/webgpu_int8/te_int8")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    args.out = str(resolve_under_home(args.out))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dev = torch.device("cuda")
    H256 = hadamard(ROT).to(dev)
    t0 = time.time()
    model, tok = load_qwen3_text_encoder(
        str(resolve_under_home(TE)), dtype=torch.float32, device="cpu"
    )
    model = model.to(dev).eval()
    caps = captions(0)
    enc = lambda t: tok(t, return_tensors="pt").input_ids[:, :512].to(dev)  # noqa: E731
    ev = [enc(c) for c in caps[: args.n_eval]]
    cal = [enc(c) for c in caps[args.n_eval : args.n_eval + args.n_calib]]
    print(
        f"calib {len(cal)} caps {sum(i.shape[1] for i in cal)} tok; eval {len(ev)}",
        flush=True,
    )
    orig = {k: v.clone() for k, v in model.state_dict().items()}
    for i, layer in enumerate(model.layers):
        for n in LINS:
            orig_lin[(i, n)] = layer.get_submodule(n)
    ref = final(model, ev)
    res = {}
    for name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        model.to(dt)
        res[name] = stats(ref, final(model, ev))
        model.to(torch.float32)
        print(name, res[name], flush=True)
    model.load_state_dict(orig)

    table = {
        "rtn128": dict(wblock=128, act_aware=False, rtn=True),
        "rtn32": dict(wblock=32, act_aware=False, rtn=True),
        "gptq128": dict(wblock=128, act_aware=False, rtn=False),
        "gptq32": dict(wblock=32, act_aware=False, rtn=False),
        "aa128": dict(wblock=128, act_aware=True, rtn=False),
        "aa32": dict(wblock=32, act_aware=True, rtn=False),
        "gptq128k": dict(wblock=128, act_aware=False, rtn=False, keep=20),
        "aa128k": dict(wblock=128, act_aware=True, rtn=False, keep=20),
        "aa32k": dict(wblock=32, act_aware=True, rtn=False, keep=20),
    }
    reports = {}
    for v in args.variants.split(","):
        qs, rep = quantize(model, orig, cal, table[v], dev)
        for mode in ("w8a8", "w8"):
            QLin.mode = mode
            res[f"{v}/{mode}"] = stats(ref, final(model, ev))
            print(v, mode, res[f"{v}/{mode}"], f"{time.time() - t0:.0f}s", flush=True)
        worst = sorted(rep, key=lambda r: -r[2])[:6]
        reports[v] = rep
        print(v, "worst linears", worst, flush=True)
        if v in args.save.split(","):
            flat = {}
            for k, (q, sc) in qs.items():
                flat[f"{k}.q"] = q
                flat[f"{k}.scale"] = sc
            save_file(
                flat,
                f"{args.out}.{v}.safetensors",
                metadata={
                    "variant": v,
                    **{k: str(x) for k, x in table[v].items()},
                    "rot": str(ROT),
                },
            )
            print(
                "saved", v, "kept fp16:", [r[:2] for r in rep if r[2] < 0], flush=True
            )
    print(json.dumps(res, indent=1))
    with open(args.out + ".report.json", "w") as f:
        json.dump({"res": res, "linears": reports}, f)


if __name__ == "__main__":
    main()
