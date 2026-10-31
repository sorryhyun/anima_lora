"""Anima: where the bf16 stream error and the mixed LoRA gradient of
``anima_stream_error.py`` come from. An arm is ``<bf16|int8>[+flag...]``:

    s      x at each block and the adaln_up outputs in fp32 (that script's ``*_s32``)
    head   final_layer.linear in fp32
    mod    t_embedder, adaln_* and final_layer.adaln_modulation in fp32
    embed  x_embedder in fp32
    rope   RoPE and the q / k norms in fp32, rounded only on entry to FA2
    y32    int8 kernel output and dY in fp32, both directions
    dx     dX of the block linears = dY.float() @ W.float(), TF32 off
    sdpa   attention in fp32 SDPA, both directions
    wide   s+head+mod+embed+rope

``fp32_jit<n>`` is the fp32 reference with relative Gaussian noise of 2^-9 on ``x`` at each
block, seed n: rounding-sized forward noise with no bf16 anywhere.

    make daemon-run ARGS="--stall-timeout 600 project/int8_train/src/anima_stream_attrib.py"

reports/anima_stream_attrib.md.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from library.runtime.allocator import default_expandable_segments

default_expandable_segments()

import comfy_kitchen  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.autograd import Function  # noqa: E402

import library.anima.models as am  # noqa: E402
from bench._anima import add_common_args, add_model_args, build_anima  # noqa: E402
from bench._common import start_heartbeat  # noqa: E402
from bench.grad_init.probe_subspace import enumerate_targets, latent_tokens  # noqa: E402
from library.anima.int8 import convert_dit_, forward_file  # noqa: E402
from library.env import resolve_under_home  # noqa: E402
from library.int8.linear import (  # noqa: E402
    GROUP,
    HeldCopy,
    Int8TrainLinear,
    int8_mm,
    quantize,
)
from library.io.cache import (  # noqa: E402
    discover_cached_pairs,
    load_cached_latents,
    load_cached_text_features,
)
from networks import attention_dispatch  # noqa: E402

GRAD_ARMS = (
    "bf16,bf16+s,int8,int8+s,bf16+s+dx,int8+s+dx,bf16+s+sdpa,bf16+s+sdpa+dx,"
    "bf16+wide,int8+wide,int8+wide+y32"
)
PRED_ARMS = (
    "bf16,bf16+s,bf16+s+head,bf16+s+mod,bf16+s+embed,bf16+s+rope,bf16+wide,"
    "int8,int8+s,int8+s+head,int8+s+mod,int8+s+embed,int8+s+rope,int8+wide,"
    "int8+y32,int8+s+y32,int8+wide+y32"
)
FLAGS: set[str] = set()
JIT: list = [None]
TRACE_Q = 16
FA2 = ("flash", "flash4")


@contextlib.contextmanager
def no_tf32():
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


class Wide(nn.Linear):
    def forward(self, x):
        with torch.autocast("cuda", enabled=False), no_tf32():
            bias = None if self.bias is None else self.bias.float()
            return F.linear(x.float(), self.weight.float(), bias)


class ExactDx(Function):
    @staticmethod
    def forward(ctx, x, y, w):
        ctx.w, ctx.x_dtype = w, x.dtype
        return y.clone()

    @staticmethod
    def backward(ctx, dy):
        with no_tf32():
            return (dy.float() @ ctx.w.float()).to(ctx.x_dtype), None, None


def mm32(x, wq, s):
    y = comfy_kitchen.int8_linear(
        x.reshape(-1, x.shape[-1]),
        wq,
        s,
        out_dtype=torch.float32,
        convrot=True,
        convrot_groupsize=GROUP,
    )
    return y.reshape(*x.shape[:-1], wq.shape[0])


class Y32(Function):
    @staticmethod
    def forward(ctx, x, wq, s, wq_t, s_t):
        ctx.wq_t, ctx.s_t, ctx.x_dtype = wq_t, s_t, x.dtype
        return mm32(x, wq, s)

    @staticmethod
    def backward(ctx, dy):
        return mm32(dy.float(), ctx.wq_t, ctx.s_t).to(ctx.x_dtype), *[None] * 4


class BfDx(nn.Linear):
    def forward(self, x):
        with torch.no_grad():
            y = super().forward(x)
        return ExactDx.apply(x, y, self.weight)


class Int8Probe(Int8TrainLinear):
    def forward(self, x):
        y32, dx = "y32" in FLAGS, "dx" in FLAGS
        if not (y32 or dx):
            return super().forward(x)
        x = x.float()
        if dx:
            with torch.no_grad():
                y = (mm32 if y32 else int8_mm)(x, self.wq, self.s)
            return ExactDx.apply(x, y, self.weight)
        return Y32.apply(x, self.wq, self.s, self.wq_t, self.s_t)


orig_block = am.Block.forward
orig_rms = am.RMSNorm.forward
orig_prepare = am.Anima.prepare_embedded_sequence
orig_dispatch = attention_dispatch.dispatch_attention


def block_forward(self, x, *a, **k):
    if "s" in FLAGS:
        x = x.float()
    if JIT[0] is not None:
        noise = torch.randn(x.shape, generator=JIT[0], device=x.device, dtype=x.dtype)
        x = x * (1 + 2.0**-9 * noise)
    return orig_block(self, x, *a, **k)


def rms_forward(self, x):
    if "rope" in FLAGS:
        return self._norm(x.float()) * self.weight.float()
    return orig_rms(self, x)


def prepare(self, *a, **k):
    x, rope = orig_prepare(self, *a, **k)
    return (x.float(), rope) if "rope" in FLAGS else (x, rope)


def dispatch(qkv, *a, attn_params=None, **k):
    if "sdpa" in FLAGS:
        with torch.autocast("cuda", enabled=False):
            return orig_dispatch(
                [t.float() for t in qkv], *a, attn_params=attn_params, **k
            )
    if attn_params.attn_mode in FA2:
        qkv = [t.to(torch.bfloat16) for t in qkv]
    return orig_dispatch(qkv, *a, attn_params=attn_params, **k)


am.Block.forward = block_forward
am.RMSNorm.forward = rms_forward
am.Anima.prepare_embedded_sequence = prepare
attention_dispatch.dispatch_attention = dispatch


def to_fp32(_mod, _inputs, y):
    return y.float() if "s" in FLAGS else y


def parse(arm):
    base, *flags = arm.split("+")
    out = set(flags)
    if "wide" in out:
        out = (out - {"wide"}) | {"s", "head", "mod", "embed", "rope"}
    return base, out


def set_arm(anima, mods, groups, attn_mode, arm):
    base, flags = parse(arm)
    FLAGS.clear()
    FLAGS.update(flags)
    anima.attn_mode = "torch" if "sdpa" in flags else attn_mode
    for m in mods:
        if base == "int8":
            m.__class__ = Int8Probe
        else:
            m.__class__ = BfDx if "dx" in flags else nn.Linear
    for name, linears in groups.items():
        for m in linears:
            m.__class__ = Wide if name in flags else nn.Linear


class Sketcher:
    """``S = (dY Ω)ᵀ X`` per target, plus per-token ``‖dY‖²`` and the first
    ``TRACE_Q`` columns of ``dY Ω``."""

    def __init__(self, targets, q, device, seed):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.omega = {
            ln: torch.randn(mod.out_features, q, generator=gen).to(device)
            for ln, _o, mod in targets
        }
        self.reset()
        self.capturing = False
        for ln, _o, mod in targets:
            mod.register_forward_hook(self._hook(ln))

    def reset(self):
        self.S, self.E, self.D = {}, {}, {}

    def _hook(self, ln):
        def hook(_mod, inputs, y):
            if not self.capturing or not y.requires_grad:
                return
            x = inputs[0].detach().to("cpu")

            def on_grad(dy):
                flat = dy.reshape(-1, dy.shape[-1]).float()
                d = flat @ self.omega[ln]
                xx = x.to(dy.device).reshape(-1, x.shape[-1]).float()
                self.S[ln] = (d.T @ xx).cpu()
                self.E[ln] = flat.pow(2).sum(-1).cpu()
                self.D[ln] = d[:, :TRACE_Q].cpu()

            y.register_hook(on_grad)

        return hook

    def take(self):
        out = (self.S, self.E, self.D)
        self.reset()
        return out


class NormProbe:
    """Smallest per-token, per-head RMS each block's attention norms see, the input to
    ``RMSNorm``'s gain ``1 / sqrt(rms² + 1e-6)``."""

    def __init__(self, anima):
        self.capturing = False
        self.rec = {}
        for i, b in enumerate(anima.blocks):
            for key, norm in (
                ("self_q", b.self_attn.q_norm),
                ("self_k", b.self_attn.k_norm),
                ("cross_q", b.cross_attn.q_norm),
            ):
                norm.register_forward_pre_hook(self._hook(f"{i}.{key}"))

    def _hook(self, key):
        def hook(_mod, inputs):
            if not self.capturing:
                return
            r = inputs[0].detach().float().square().mean(-1).sqrt().amin(-1)[0]
            top = int(r.argmin())
            self.rec[key] = (float(r[top]), top, float(r.median()))

        return hook

    def take(self):
        out, self.rec = self.rec, {}
        return out


def pick_images(root, n, lo, hi, seed):
    pairs = discover_cached_pairs(str(root))
    gen = torch.Generator().manual_seed(seed)
    picked = []
    for i in torch.randperm(len(pairs), generator=gen).tolist():
        lat = load_cached_latents(pairs[i].npz_path)[0]
        if lo <= latent_tokens(lat) <= hi:
            picked.append(pairs[i])
        if len(picked) == n:
            break
    return picked


def load_batch(ci, j, seed, device):
    latents = load_cached_latents(ci.npz_path)[0].unsqueeze(0).to(device)
    crossattn, _ = load_cached_text_features(ci.te_path, variant=0)
    crossattn = crossattn.unsqueeze(0).to(device, torch.float32)
    gen = torch.Generator(device=device).manual_seed(seed * 1000 + j)
    noise = torch.randn(latents.shape, generator=gen, device=device)
    return latents, crossattn, noise


def one_pass(anima, sk, np_, batch, sigma, *, fp32=False, grad=True):
    latents, crossattn, noise = batch
    dtype = torch.float32 if fp32 else torch.bfloat16
    s4 = sigma.view(-1, 1, 1, 1)
    noisy = ((1.0 - s4) * latents + s4 * noise).unsqueeze(2).to(dtype)
    noisy.requires_grad_(grad)
    pad = torch.zeros(1, 1, *latents.shape[-2:], dtype=dtype, device=latents.device)
    sk.capturing = np_.capturing = True
    with (
        torch.set_grad_enabled(grad),
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=not fp32),
    ):
        pred = anima(noisy, sigma, crossattn.to(dtype), padding_mask=pad)
    sk.capturing = np_.capturing = False
    pred = pred.squeeze(2).float()
    loss = F.mse_loss(pred, noise - latents)
    if grad:
        loss.backward()
    S, E, D = sk.take()
    return {
        "S": S,
        "E": E,
        "D": D,
        "loss": float(loss),
        "pred": pred.detach(),
    }, np_.take()


def energy(E, names):
    t = sum(E[ln].double() for ln in names)
    top = int(t.argmax())
    return {"e": float(t.sum()), "share": float(t[top] / t.sum()), "tok": top}


def compare(rec, ref, block_of, kind_of, by_group, by_block):
    """Distance from fp32 per block group and per (group, kind) on S, token concentration
    of the error on the dY trace, and the arm's own energy per block."""
    out = {"dist": {}, "kind": defaultdict(dict), "err": {}, "energy": {}}
    for g, names in by_group.items():
        names = [ln for ln in names if ln in ref["S"] and ln in rec["S"]]
        kinds = defaultdict(float)
        for ln in names:
            kinds[kind_of[ln]] += float((rec["S"][ln] - ref["S"][ln]).pow(2).sum())
        out["dist"][g] = sum(kinds.values())
        out["kind"][g] = dict(kinds)
        e_t = sum((rec["D"][ln] - ref["D"][ln]).double().pow(2).sum(-1) for ln in names)
        total = float(e_t.sum())
        vals, idx = e_t.topk(10)
        out["err"][g] = {
            "e": total,
            "top1": float(vals[0]) / total,
            "top10": float(vals.sum()) / total,
            "tok": int(idx[0]),
        }
    for b, names in by_block.items():
        names = [ln for ln in names if ln in rec["E"]]
        if names:
            out["energy"][b] = energy(rec["E"], names)
    return out


def ref_stats(ref, by_group, by_block):
    out = {"S2": {}, "energy": {}}
    for g, names in by_group.items():
        out["S2"][g] = sum(
            float(ref["S"][ln].pow(2).sum()) for ln in names if ln in ref["S"]
        )
    for b, names in by_block.items():
        names = [ln for ln in names if ln in ref["E"]]
        if names:
            out["energy"][b] = energy(ref["E"], names)
    return out


def check_y32(mods):
    """Does the kernel's fp32 output carry more digits than its bf16 one?"""
    m = next(m for m in mods if m.in_features == m.out_features or True)
    x = torch.randn(64, m.in_features, device=m.wq.device)
    y32 = mm32(x, m.wq, m.s)
    y16 = int8_mm(x, m.wq, m.s)
    frac = (y32 - y16.float()).abs().div(y32.abs().clamp_min(1e-9)).median()
    print(
        f"y32 check: dtype {y32.dtype}, median |y32 - y16| / |y| {float(frac):.2e} "
        f"(bf16 half ulp up to {2.0**-8:.2e})",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    add_model_args(ap, vae=False, text_encoder=False)
    add_common_args(ap, include_compile=False, include_checkpointing=False)
    ap.add_argument("--cache_root", default="post_image_dataset/lora")
    ap.add_argument("--sigmas", default="0.1,0.5,0.9")
    ap.add_argument("--n_images", type=int, default=6)
    ap.add_argument("--sketch_q", type=int, default=64)
    ap.add_argument("--min_tokens", type=int, default=3500)
    ap.add_argument("--max_tokens", type=int, default=4608)
    ap.add_argument("--arms", default=None)
    ap.add_argument("--jit", type=int, default=2, help="fp32_jit seeds")
    ap.add_argument("--no_grad", action="store_true", help="prediction arms only")
    ap.add_argument(
        "--out", default="project/int8_train/assets/anima_stream_attrib.json"
    )
    args = ap.parse_args()
    args.gradient_checkpointing = True
    grad = not args.no_grad
    arms = (args.arms or (GRAD_ARMS if grad else PRED_ARMS)).split(",")
    start_heartbeat(label="stream_attrib")
    for _ in range(90):
        if torch.cuda.mem_get_info()[0] > 14e9:
            break
        time.sleep(1)
    print(f"free VRAM {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB", flush=True)

    bundle = build_anima(args, adapter=None, train_mode=True)
    anima, device = bundle.anima, bundle.device
    targets = enumerate_targets(anima)
    sk = Sketcher(targets, args.sketch_q, device, args.seed)
    np_ = NormProbe(anima)
    images = pick_images(
        Path(resolve_under_home(args.cache_root)),
        args.n_images,
        args.min_tokens,
        args.max_tokens,
        args.seed,
    )
    sigmas = [float(s) for s in args.sigmas.split(",")]
    print("images:", [Path(c.npz_path).stem for c in images], flush=True)

    block_of, kind_of = {}, {}
    by_group, by_block = defaultdict(list), defaultdict(list)
    for ln, o, _ in targets:
        parts = o.split(".")
        if parts[0] == "blocks":
            b = int(parts[1])
            block_of[ln], kind_of[ln] = b, ".".join(parts[2:])
            by_group[b // 7].append(ln)
            by_block[b].append(ln)

    t0 = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtypes = {
        n: t.dtype for n, t in [*anima.named_parameters(), *anima.named_buffers()]
    }
    attn_mode = anima.attn_mode
    anima.float()
    anima.attn_mode = "torch"
    refs = {}
    cells = {}
    for sigma_v in sigmas:
        sigma = torch.tensor([sigma_v], device=device)
        for j, ci in enumerate(images):
            batch = load_batch(ci, j, args.seed, device)
            ref, norms = one_pass(anima, sk, np_, batch, sigma, fp32=True, grad=grad)
            ref["pred"] = ref["pred"].cpu()
            refs[sigma_v, j] = ref
            cell = {
                "sigma": sigma_v,
                "image": j,
                "tokens": latent_tokens(batch[0]),
                "loss_fp32": ref["loss"],
                "norms": {"fp32": norms},
                "arms": {},
            }
            if grad:
                cell["ref"] = ref_stats(ref, by_group, by_block)
            cells[sigma_v, j] = cell
            for n in range(args.jit):
                JIT[0] = torch.Generator(device=device).manual_seed(n + 1)
                rec, norms = one_pass(
                    anima, sk, np_, batch, sigma, fp32=True, grad=grad
                )
                JIT[0] = None
                row = {
                    "pred": float(
                        (rec["pred"].cpu() - ref["pred"]).norm() / ref["pred"].norm()
                    ),
                    "loss": rec["loss"],
                }
                if grad:
                    row |= compare(rec, ref, block_of, kind_of, by_group, by_block)
                cell["arms"][f"fp32_jit{n}"] = row
            print(
                f"fp32 σ={sigma_v} {j + 1}/{len(images)} loss={ref['loss']:.4f} "
                f"{time.perf_counter() - t0:.0f}s",
                flush=True,
            )
    for n, t in [*anima.named_parameters(), *anima.named_buffers()]:
        t.data = t.data.to(dtypes[n])
    anima.attn_mode = attn_mode
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    path = forward_file(None, args.dit)
    print(f"forward copies: {path}", flush=True)
    names = convert_dit_(anima, "fwd", path or None)
    mods = [anima.get_submodule(n) for n in names]
    for m in mods:
        wq_t, s_t = quantize(m.weight.detach().t().contiguous())
        m.held_wq_t = HeldCopy(wq_t)
        m.register_buffer("s_t", s_t, persistent=False)
        m.int8_mode = "full"
    check_y32(mods)
    mod_linears = [
        *(m for m in anima.t_embedder.modules() if isinstance(m, nn.Linear)),
        *(
            m
            for m in anima.final_layer.adaln_modulation.modules()
            if isinstance(m, nn.Linear)
        ),
    ]
    for block in anima.blocks:
        mod_linears.append(block.adaln_fused_down[1])
        for mod in (
            block.adaln_up_self_attn,
            block.adaln_up_cross_attn,
            block.adaln_up_mlp,
        ):
            mod_linears.append(mod)
            mod.register_forward_hook(to_fp32)
    groups = {
        "mod": mod_linears,
        "head": [anima.final_layer.linear],
        "embed": [anima.x_embedder.proj[1]],
    }

    for sigma_v in sigmas:
        sigma = torch.tensor([sigma_v], device=device)
        for j, ci in enumerate(images):
            batch = load_batch(ci, j, args.seed, device)
            ref = refs[sigma_v, j]
            pred32 = ref["pred"].to(device)
            cell = cells[sigma_v, j]
            for arm in arms:
                set_arm(anima, mods, groups, attn_mode, arm)
                rec, norms = one_pass(anima, sk, np_, batch, sigma, grad=grad)
                row = {
                    "pred": float((rec["pred"] - pred32).norm() / pred32.norm()),
                    "loss": rec["loss"],
                }
                if grad:
                    row |= compare(rec, ref, block_of, kind_of, by_group, by_block)
                cell["norms"][arm] = norms
                cell["arms"][arm] = row
            FLAGS.clear()
            print(
                f"σ={sigma_v} {j + 1}/{len(images)} "
                + " ".join(
                    f"{a}={cell['arms'][a]['pred'] * 100:.3f}" for a in cell["arms"]
                )
                + f" {time.perf_counter() - t0:.0f}s",
                flush=True,
            )
            Path(resolve_under_home(args.out)).write_text(
                json.dumps(list(cells.values()))
            )

    all_arms = list(next(iter(cells.values()))["arms"])
    print("\nRESULT prediction error, % of ‖v_fp32‖ (mean over images)")
    for s in sigmas:
        row = [c for (sv, _j), c in cells.items() if sv == s]
        print(
            f"RESULT σ={s} "
            + " ".join(
                f"{a}={statistics.mean(c['arms'][a]['pred'] for c in row) * 100:.3f}"
                for a in all_arms
            )
        )
    if grad and "bf16" in all_arms:
        print("RESULT ‖S - S_fp32‖ / ‖S_fp32‖, %, median over images, groups 0..3")
        for s in sigmas:
            row = [c for (sv, _j), c in cells.items() if sv == s]
            for a in all_arms:
                cols = []
                for g in sorted(row[0]["ref"]["S2"]):
                    v = [
                        (c["arms"][a]["dist"][g] / c["ref"]["S2"][g]) ** 0.5
                        for c in row
                    ]
                    cols.append(f"{statistics.median(v) * 100:.1f}")
                print(f"RESULT σ={s} {a}: " + " | ".join(cols))


if __name__ == "__main__":
    main()
