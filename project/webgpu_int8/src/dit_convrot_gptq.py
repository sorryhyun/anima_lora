"""ConvRot GPTQ for the Anima DiT on ORT-web's MatMulNBits grid (the browser DiT).

The grid is te_convrot_gptq.py's: the input rotated by blockdiag(H256) in-graph, the
weight int8 with one fp16 scale per (row, 32-wide K block), and on the DP4A path the
rotated activation rounded per token per 128-wide K block. The converted linears are
library/anima/int8.py's 168 (self-attn qkv / out, cross-attn q / out, mlp layer1 / 2);
the fused qkv is split back into q / k / v rows on write, which the per-row scales allow.

Calibration is dit_gptq.py's (cached images per artist, one σ each from the inference
schedule), here on the 4-step shift-3 grid and the caption pass only (a cfg 1 turbo).
Blocks are quantized in order, each quantized block (W8A8 on the DP4A grid) feeding the
next. Per linear: act-aware least-squares target W* from the DP4A-rounded input, then
GPTQ with static block scales and act order (variant ``aa32``); ``gptq32`` and ``rtn32``
are scored alongside on the same grid.

    make daemon-run ARGS="--queue --stall-timeout 1800 project/webgpu_int8/src/dit_convrot_gptq.py \
        --dit models/diffusion_models/anima_turbo_4step_v4_preview.safetensors"

Writes ``output/webgpu_int8/dit_int8.<stem>.safetensors``: per linear ``<key>.q`` int8
(out, in) in the rotated basis and ``<key>.scale`` fp16 (out, in / 32), keys in the
checkpoint's own naming (``blocks.N.self_attn.q_proj`` …), plus ``.report.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "finished/int8_train/src"))

from dit_gptq import (  # noqa: E402  (finished line's script, read-only reuse)
    add_common_args,
    add_model_args,
    build_anima,
    capture,
    load_cached_latents,
    load_cached_text_features,
    pick_images,
    schedule_sigmas,
    tree,
)
from library.anima.int8 import CONVERT_SUFFIXES, convert_targets  # noqa: E402
from library.env import resolve_under_home  # noqa: E402

ROT = 256
ABLK = 128
H256: torch.Tensor | None = None


def hadamard(n: int) -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / n**0.5


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
    s = sc.repeat_interleave(b, 1)
    return (w / s).round().clamp(-128, 127) * s


def gptq(W, H, sc, b, damp=0.01, blk=128):
    """te_convrot_gptq.gptq: static block scales ``sc`` (out, K/b), act order."""
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
    """The DP4A path: rotated input rounded per 128 block, dequantised weight."""

    def __init__(self, q_rot: torch.Tensor, bias):
        super().__init__()
        self.register_buffer("w", q_rot)
        self.bias = bias

    def forward(self, x):
        with torch.autocast("cuda", enabled=False):
            y = act_qdq(rot(x.float())) @ self.w.T
            if self.bias is not None:
                y = y + self.bias.float()
        return y.to(x.dtype)


class Acc:
    """``H_r``, ``H̃``, ``G`` of one linear's rotated input (fp32 sums, as dit_gptq)."""

    def __init__(self, k: int, device):
        self.Hr, self.Hq, self.G = (torch.zeros(k, k, device=device) for _ in range(3))

    def hook(self, _mod, inputs):
        with torch.autocast("cuda", enabled=False):
            xr = rot(inputs[0].reshape(-1, inputs[0].shape[-1]).float())
            xq = act_qdq(xr)
            self.Hr.addmm_(xr.T, xr)
            self.Hq.addmm_(xq.T, xq)
            self.G.addmm_(xq.T, xr)


def w8a8_err(Q, Wr, acc: Acc) -> float:
    ref = ((Wr @ acc.Hr) * Wr).sum().double()
    e = ((Q @ acc.Hq) * Q).sum().double() - 2 * ((Q @ acc.G) * Wr).sum().double() + ref
    return (e / ref).item()


def w8_err(Q, Wr, acc: Acc) -> float:
    D = Q - Wr
    return (((D @ acc.Hr) * D).sum() / ((Wr @ acc.Hr) * Wr).sum()).item()


def act_aware_target(Wr, acc: Acc, lam=1e-4):
    Hq = acc.Hq.double()
    eye = torch.eye(Wr.shape[1], device=Wr.device, dtype=torch.float64)
    while True:
        L, info = torch.linalg.cholesky_ex(Hq + lam * Hq.diagonal().mean() * eye)
        if info == 0:
            break
        lam *= 10
    return torch.cholesky_solve(acc.G.double() @ Wr.T.double(), L).T.float(), lam


def main():
    global H256
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(ap, vae=False, text_encoder=True)
    add_common_args(ap, include_compile=False, include_checkpointing=False)
    ap.add_argument("--cache_root", default="post_image_dataset/lora")
    ap.add_argument("--per_artist", type=int, default=4)
    ap.add_argument("--min_tokens", type=int, default=3000)
    ap.add_argument("--max_tokens", type=int, default=4608)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--flow_shift", type=float, default=3.0)
    ap.add_argument("--wblock", type=int, default=32)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    t0 = time.perf_counter()
    src = resolve_under_home(args.dit)
    out = resolve_under_home(
        args.out or f"output/webgpu_int8/dit_int8.{src.stem}.safetensors"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    bundle = build_anima(args, adapter=None, train_mode=False)
    anima, device = bundle.anima, bundle.device
    H256 = hadamard(ROT).to(device)
    b = args.wblock

    picked = pick_images(
        resolve_under_home(args.cache_root),
        args.per_artist,
        args.min_tokens,
        args.max_tokens,
        args.seed,
    )
    sigmas = schedule_sigmas(args.steps, args.flow_shift)
    import random

    rng = random.Random(args.seed)
    samples, attn_params, n = [], None, 0
    for pairs in picked.values():
        for ci in pairs:
            latents = load_cached_latents(ci.npz_path)[0].unsqueeze(0).to(device)
            crossattn, _ = load_cached_text_features(ci.te_path, variant=0)
            gen = torch.Generator(device=device).manual_seed(args.seed * 100003 + n)
            noise = torch.randn(latents.shape, generator=gen, device=device)
            sigma = torch.tensor([rng.choice(sigmas)], device=device)
            s4 = sigma.view(-1, 1, 1, 1)
            noisy = ((1.0 - s4) * latents + s4 * noise).unsqueeze(2).bfloat16()
            pad = torch.zeros(
                1, 1, *latents.shape[-2:], dtype=torch.bfloat16, device=device
            )
            a, kw, attn_params = capture(
                anima,
                noisy,
                sigma,
                crossattn.unsqueeze(0).to(device, torch.bfloat16),
                pad,
            )
            samples.append((a, kw))
            n += 1
    calib = {
        "artists": len(picked),
        "images": n,
        "tokens_per_linear": sum(a[0].numel() // a[0].shape[-1] for a, _ in samples),
        "sigma_schedule": f"{args.steps} steps, shift {args.flow_shift}",
        "seed": args.seed,
    }
    print(f"[calib] {json.dumps(calib)} {time.perf_counter() - t0:.0f}s", flush=True)

    def run_block(block, a, kw):
        a, kw = tree(a, lambda t: t.to(device)), tree(kw, lambda t: t.to(device))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return block(a[0], a[1], a[2], attn_params, **kw)

    targets = dict(convert_targets(anima))
    flat, report = {}, {}
    for i, block in enumerate(anima.blocks):
        lins = {suf: targets[f"blocks.{i}.{suf}"] for suf in CONVERT_SUFFIXES}
        accs = {suf: Acc(m.in_features, device) for suf, m in lins.items()}
        hooks = [m.register_forward_pre_hook(accs[s].hook) for s, m in lins.items()]
        for a, kw in samples:
            run_block(block, a, kw)
        for h in hooks:
            h.remove()
        for suf, m in lins.items():
            acc = accs[suf]
            Wr = rot(m.weight.detach().float())
            Hr32, Hq32 = acc.Hr, acc.Hq
            sc_r = block_scale(Wr, b)
            Wstar, lam = act_aware_target(Wr, acc)
            sc_a = block_scale(Wstar, b)
            rounds = {
                "rtn32": (qdq_w(Wr, sc_r, b), sc_r),
                "gptq32": (gptq(Wr, Hr32, sc_r, b, args.damp), sc_r),
                "aa32": (gptq(Wstar, Hq32, sc_a, b, args.damp), sc_a),
            }
            row = {f"w8a8_{k}": w8a8_err(Q, Wr, acc) for k, (Q, _s) in rounds.items()}
            row.update({f"w8_{k}": w8_err(Q, Wr, acc) for k, (Q, _s) in rounds.items()})
            row["ls_ridge"] = lam
            report[f"blocks.{i}.{suf}"] = row
            Q, sc = rounds["aa32"]
            q = (Q / sc.repeat_interleave(b, 1)).round().clamp(-128, 127).to(torch.int8)
            if suf == "self_attn.qkv_proj":
                rows = q.shape[0] // 3
                for j, name in enumerate(("q_proj", "k_proj", "v_proj")):
                    key = f"blocks.{i}.self_attn.{name}"
                    flat[f"{key}.q"] = q[j * rows : (j + 1) * rows].cpu().contiguous()
                    flat[f"{key}.scale"] = (
                        sc[j * rows : (j + 1) * rows].half().cpu().contiguous()
                    )
            else:
                flat[f"blocks.{i}.{suf}.q"] = q.cpu().contiguous()
                flat[f"blocks.{i}.{suf}.scale"] = sc.half().cpu().contiguous()
            parent, _, child = suf.rpartition(".")
            setattr(block.get_submodule(parent), child, QLin(Q, m.bias))
        del accs
        samples = [
            ((tree(run_block(block, a, kw), lambda t: t.cpu()), *a[1:]), kw)
            for a, kw in samples
        ]
        torch.cuda.empty_cache()
        rows = [report[f"blocks.{i}.{s}"] for s in CONVERT_SUFFIXES]
        mean = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
        print(
            f"[gptq] block {i:2d} w8a8 rtn {mean['w8a8_rtn32']:.2e} gptq "
            f"{mean['w8a8_gptq32']:.2e} aa {mean['w8a8_aa32']:.2e} | w8 rtn "
            f"{mean['w8_rtn32']:.2e} gptq {mean['w8_gptq32']:.2e} aa {mean['w8_aa32']:.2e}"
            f" | {time.perf_counter() - t0:.0f}s",
            flush=True,
        )

    keys = next(iter(report.values())).keys()
    total = {k: sum(r[k] for r in report.values()) / len(report) for k in keys}
    print("[gptq] mean over linears " + json.dumps(total), flush=True)
    meta = {
        "source": src.name,
        "variant": "aa32",
        "wblock": str(b),
        "rot": str(ROT),
        "act_block": str(ABLK),
        "calibration": json.dumps(calib),
    }
    save_file(flat, str(out), metadata=meta)
    out.with_suffix(".report.json").write_text(
        json.dumps({"meta": meta, "mean": total, "linears": report}, indent=1)
    )
    print(
        f"[gptq] wrote {out} ({out.stat().st_size / 2**30:.2f} GiB) in "
        f"{time.perf_counter() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
