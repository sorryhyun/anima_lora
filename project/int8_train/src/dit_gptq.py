"""Offline GPTQ for the Anima DiT's int8 ConvRot linears (the 168 ``int8_linear``
converts), written as a full DiT checkpoint in ComfyUI's ``int8_tensorwise`` +
``convrot`` layout.

Calibration is broad across artists: up to ``--per_artist`` cached images from every
artist folder of ``--cache_root`` (token count in range), each noised at one σ drawn
from the inference schedule (``--steps`` Euler steps, ``--flow_shift``), and run twice,
with its own caption and with ``--negative_prompt`` (the CFG uncond of every step).

Blocks are quantized in order, each quantized block (W8A8, the kernel) feeding the
next. Per converted linear, three roundings onto the same grid (the weight rotated per
256 columns, per-row scale at the rotated weight's amax / 127):

- ``rtn``: round to nearest, what ``convert_`` does without a file.
- ``gptq``: GPTQ against ``H_r = X_rᵀX_r`` (the rotated bf16 input).
- ``act_aware``: GPTQ of ``W* = W_r·Gᵀ·(H̃ + λI)⁻¹`` against ``H̃ = X̃ᵀX̃``, where
  ``X̃`` is the per-token int8-rounded rotated input and ``G = X̃ᵀX_r``: the
  least-squares weight from ``X̃`` to the exact output (``te_gptq.py --act_aware``).

All three are scored by the W8A8 output error ``‖X̃Qᵀ − X_r W_rᵀ‖² / ‖X_r W_rᵀ‖²``,
and ``rtn`` / ``gptq`` also by the weight-only error. ``--method`` picks the rounding
that is written and that feeds the next block. GPTQ, ``w8a8_err`` and ``out_err``
are ``project/finished/moss_lora/src/te_gptq.py``'s.

The file holds every source tensor; a converted linear's ``weight`` is int8 with its
fp32 ``weight_scale`` (out, 1) and ``comfy_quant``, the fused ``qkv_proj`` split back
into q / k / v. Metadata: ``int8_id`` (sha256 over the int8 tensors and scales, what
LoRAs are stamped with), ``int8_modules``, ``base_fingerprint`` (checked at load), the
source file's sha256, the calibration.
``int8_linear.load_int8_file`` reads it back.

    make daemon-run ARGS="--stall-timeout 1800 project/int8_train/src/dit_gptq.py"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path

from library.runtime.allocator import default_expandable_segments

default_expandable_segments()

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from bench._anima import add_common_args, add_model_args, build_anima  # noqa: E402
from library.env import resolve_under_home  # noqa: E402
from library.io.cache import (  # noqa: E402
    discover_cached_pairs,
    load_cached_latents,
    load_cached_text_features,
)
from library.moss import int8  # noqa: E402
from int8_linear import (  # noqa: E402  (sibling module)
    CONVERT_SUFFIXES,
    base_fingerprint,
    ckpt_keys,
    convert_,
    convert_targets,
)

NEGATIVE = (
    "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, "
    "sepia"
)
_WXH = re.compile(r"_(\d+)x(\d+)_anima\.npz$")
METHODS = ("rtn", "gptq", "act_aware")


# --- te_gptq.py's GPTQ and error measures -------------------------------------


def act_qdq(x: torch.Tensor) -> torch.Tensor:
    """Per-row int8 round trip, ``comfy_kitchen``'s activation quantizer."""
    s = (x.abs().amax(-1, keepdim=True) / int8.INT8_MAX).clamp(min=1e-30)
    return (x / s).round().clamp(-int8.INT8_MAX - 1, int8.INT8_MAX) * s


def w8a8_err(Q, Wr, Hq, G, Hr) -> float:
    """``‖X̃Qᵀ − X_r W_rᵀ‖² / ‖X_r W_rᵀ‖²`` from ``H̃ = X̃ᵀX̃``, ``G = X̃ᵀX_r``,
    ``H_r = X_rᵀX_r``."""
    ref = ((Wr @ Hr) * Wr).sum()
    return (((Q @ Hq) * Q).sum() - 2 * ((Q @ G) * Wr).sum() + ref).item() / ref.item()


def out_err(delta: torch.Tensor, H: torch.Tensor, W: torch.Tensor) -> float:
    """``tr(Δ H Δᵀ) / tr(W H Wᵀ)``: relative output energy of the error."""
    return ((delta @ H) * delta).sum().item() / ((W @ H) * W).sum().item()


def gptq(W: torch.Tensor, H: torch.Tensor, q, block: int, damp: float, act_order: bool):
    """``W`` (out, in) fp32 rounded column by column with ``q`` (a column → its
    dequantised rounding); returns the dequantised weight. Columns no calibration
    input reached keep plain rounding."""
    W, H = W.clone(), H.clone()
    n = W.shape[1]
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    perm = None
    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
        W, H = W[:, perm], H[perm][:, perm]
    H.diagonal().add_(damp * H.diagonal().mean())
    Hinv = torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True
    )
    Q = torch.zeros_like(W)
    for i1 in range(0, n, block):
        i2 = min(i1 + block, n)
        W1, Hi = W[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
        Q1, E1 = torch.zeros_like(W1), torch.zeros_like(W1)
        for j in range(i2 - i1):
            qj = q(W1[:, j])
            e = (W1[:, j] - qj) / Hi[j, j]
            Q1[:, j], E1[:, j] = qj, e
            W1[:, j:] -= e[:, None] * Hi[j, j:][None, :]
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    if perm is not None:
        Q = Q[:, torch.argsort(perm)]
    return Q


# --- calibration --------------------------------------------------------------


def pick_images(root: Path, per_artist: int, min_tokens: int, max_tokens: int, seed):
    """Up to ``per_artist`` cached images from every artist folder under ``root``."""
    rng = random.Random(seed)
    picked = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        pairs = []
        for ci in discover_cached_pairs(str(d)):
            m = _WXH.search(ci.npz_path)
            if m and min_tokens <= (int(m[1]) // 16) * (int(m[2]) // 16) <= max_tokens:
                pairs.append(ci)
        if pairs:
            picked[d.name] = rng.sample(pairs, min(per_artist, len(pairs)))
    return picked


def schedule_sigmas(steps: int, shift: float) -> list[float]:
    ts = [1 - k / steps for k in range(steps)]
    return [shift * t / (1 + (shift - 1) * t) for t in ts]


def tree(obj, fn):
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, (tuple, list)):
        return type(obj)(tree(o, fn) for o in obj)
    if isinstance(obj, dict):
        return {k: tree(v, fn) for k, v in obj.items()}
    return obj


class _Stop(Exception):
    pass


@torch.no_grad()
def capture(anima, noisy, sigma, crossattn, pad):
    """``(x, t_emb, crossattn_emb, block kwargs)`` entering block 0, on the CPU."""
    got = {}
    skip = ("capture_blocks", "feature_sink", "stop_after_block")

    def run_blocks(x, t_emb, c, attn_params, **kw):
        got["args"] = (x, t_emb, c)
        got["kw"] = {k: v for k, v in kw.items() if k not in skip}
        got["attn_params"] = attn_params
        raise _Stop

    anima._run_blocks = run_blocks
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            anima(noisy, sigma, crossattn, padding_mask=pad)
    except _Stop:
        pass
    finally:
        del anima._run_blocks
    return (
        tree(got["args"], lambda t: t.cpu()),
        tree(got["kw"], lambda t: t.cpu()),
        got["attn_params"],
    )


def encode_negative(anima, args, device) -> torch.Tensor:
    from library.inference.text import prepare_text_inputs

    arg_c, _ = prepare_text_inputs(
        device=device,
        anima=anima,
        prompt=args.negative_prompt,
        negative_prompt="",
        text_encoder_path=args.text_encoder,
    )
    return arg_c["embed"][0].float()


class Acc:
    """``H_r``, ``H̃`` and ``G`` of one linear's input, in the rotated basis."""

    def __init__(self, k: int, device):
        self.Hr, self.Hq, self.G = (torch.zeros(k, k, device=device) for _ in range(3))
        self.tokens = 0

    def hook(self, _mod, inputs):
        with torch.autocast("cuda", enabled=False):
            xr = int8.rotate(inputs[0].reshape(-1, inputs[0].shape[-1]).float())
            xq = act_qdq(xr)
            self.Hr.addmm_(xr.T, xr)
            self.Hq.addmm_(xq.T, xq)
            self.G.addmm_(xq.T, xr)
        self.tokens += xr.shape[0]


def chol_ridge(H: torch.Tensor, damp: float, tries: int = 6):
    """Cholesky of ``H + λI``, ``λ`` = ``damp`` of the mean diagonal, raised 10x per
    try until positive definite. Returns ``(L, damp used)``."""
    eye = torch.eye(H.shape[0], device=H.device)
    for k in range(tries):
        d = damp * 10**k
        L, info = torch.linalg.cholesky_ex(H + d * H.diagonal().mean() * eye)
        if not info:
            return L, d
    raise ValueError(f"H + λI not positive definite up to λ = {d} of the mean diagonal")


def roundings(W, acc: Acc, args) -> tuple[dict, float]:
    """Dequantised ``Q`` and its row scale per method, in the rotated basis, and the
    ridge ``act_aware`` needed."""
    Wr = int8.rotate(W)
    scale = int8.row_scale(Wr)
    out = {"rtn": (int8.qdq(Wr, scale), scale)}

    def grid(s):
        return lambda c: int8.qdq(c, s)

    out["gptq"] = (
        gptq(Wr, acc.Hr, grid(scale), args.block, args.damp, not args.no_act_order),
        scale,
    )
    L, ls_damp = chol_ridge(acc.Hq, args.ls_damp)
    Wstar = torch.cholesky_solve(acc.G @ Wr.T, L).T
    s_star = int8.row_scale(Wstar)
    out["act_aware"] = (
        gptq(Wstar, acc.Hq, grid(s_star), args.block, args.damp, not args.no_act_order),
        s_star,
    )
    return out, ls_damp


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def write_file(
    srcs: list[Path], out: Path, copies: dict, meta: dict, name_keys=ckpt_keys
) -> str:
    """Every source tensor (``srcs``: the checkpoint's shards), the converted linears
    replaced by their int8 copies. Returns ``int8_id``."""
    conf = torch.tensor(list(json.dumps(int8.QUANT_CONF).encode()), dtype=torch.uint8)
    replaced, digest = {}, hashlib.sha256()
    for name in sorted(copies):
        wq, s = copies[name]
        keys = name_keys(name)
        rows = wq.shape[0] // len(keys)
        for j, key in enumerate(keys):
            q = wq[j * rows : (j + 1) * rows].contiguous()
            sc = s[j * rows : (j + 1) * rows].contiguous()
            prefix = key.removesuffix("weight")
            replaced[key] = q
            replaced[f"{prefix}weight_scale"] = sc
            replaced[f"{prefix}comfy_quant"] = conf.clone()
            digest.update(key.encode())
            digest.update(q.numpy().tobytes())
            digest.update(sc.numpy().tobytes())
    src_keys, tensors = set(), {}
    for src in srcs:
        with safe_open(str(src), framework="pt") as f:
            keys = set(f.keys())
            src_keys |= keys
            tensors.update({k: f.get_tensor(k) for k in keys if k not in replaced})
    stray = [k for k in replaced if k.endswith(".weight") and k not in src_keys]
    if stray:
        raise ValueError(f"{len(stray)} converted keys not in {srcs}, e.g. {stray[0]}")
    tensors.update(replaced)
    int8_id = digest.hexdigest()
    meta = {
        **meta,
        "int8_id": int8_id,
        "int8_modules": json.dumps(sorted(copies)),
        "quantization": json.dumps(int8.QUANT_CONF),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata=meta)
    return int8_id


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(ap, vae=False, text_encoder=True)
    add_common_args(ap, include_compile=False, include_checkpointing=False)
    ap.add_argument("--cache_root", default="post_image_dataset/lora")
    ap.add_argument("--per_artist", type=int, default=2)
    ap.add_argument("--min_tokens", type=int, default=3000)
    ap.add_argument("--max_tokens", type=int, default=4608)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--flow_shift", type=float, default=3.0)
    ap.add_argument("--negative_prompt", default=NEGATIVE)
    ap.add_argument("--method", choices=METHODS[1:], default="gptq")
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--ls_damp", type=float, default=1e-4, help="act_aware's ridge")
    ap.add_argument("--no_act_order", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    t0 = time.perf_counter()
    src = resolve_under_home(args.dit)
    out = resolve_under_home(
        args.out or f"models/diffusion_models/{src.stem}_int8_{args.method}.safetensors"
    )

    bundle = build_anima(args, adapter=None, train_mode=False)
    anima, device = bundle.anima, bundle.device
    fingerprint = base_fingerprint(anima)
    neg = encode_negative(anima, args, device)

    picked = pick_images(
        resolve_under_home(args.cache_root),
        args.per_artist,
        args.min_tokens,
        args.max_tokens,
        args.seed,
    )
    sigmas = schedule_sigmas(args.steps, args.flow_shift)
    rng = random.Random(args.seed)
    samples, attn_params, n = [], None, 0
    for artist, pairs in picked.items():
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
            for c in (crossattn.unsqueeze(0), neg):
                a, kw, attn_params = capture(
                    anima, noisy, sigma, c.to(device, torch.bfloat16), pad
                )
                samples.append((a, kw))
            n += 1
    tokens = sum(a[0].numel() // a[0].shape[-1] for a, _ in samples)
    calib = {
        "artists": len(picked),
        "images": n,
        "samples": len(samples),
        "tokens_per_linear": tokens,
        "per_artist": args.per_artist,
        "token_range": [args.min_tokens, args.max_tokens],
        "sigma_schedule": f"{args.steps} steps, shift {args.flow_shift}",
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
    }
    print(f"[calib] {json.dumps(calib)} {time.perf_counter() - t0:.0f}s", flush=True)

    def run_block(block, a, kw):
        a, kw = tree(a, lambda t: t.to(device)), tree(kw, lambda t: t.to(device))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return block(a[0], a[1], a[2], attn_params, **kw)

    targets = dict(convert_targets(anima))
    copies, report = {}, {}
    for i, block in enumerate(anima.blocks):
        lins = {suf: targets[f"blocks.{i}.{suf}"] for suf in CONVERT_SUFFIXES}
        accs = {suf: Acc(m.in_features, device) for suf, m in lins.items()}
        hooks = [m.register_forward_pre_hook(accs[s].hook) for s, m in lins.items()]
        for a, kw in samples:
            run_block(block, a, kw)
        for h in hooks:
            h.remove()
        for suf, m in lins.items():
            acc, W = accs[suf], m.weight.detach().float()
            Wr = int8.rotate(W)
            rounds, ls_damp = roundings(W, acc, args)
            row = {
                f"w8a8_{k}": w8a8_err(Q, Wr, acc.Hq, acc.G, acc.Hr)
                for k, (Q, _s) in rounds.items()
            }
            for k in ("rtn", "gptq"):
                row[f"w8a16_{k}"] = out_err(rounds[k][0] - Wr, acc.Hr, Wr)
            row["ls_damp"] = ls_damp
            report[f"blocks.{i}.{suf}"] = row
            Q, scale = rounds[args.method]
            wq = (Q / scale[:, None]).round().clamp(-128, 127).to(torch.int8)
            copies[f"blocks.{i}.{suf}"] = (wq.cpu(), scale[:, None].cpu())
            convert_(m, "infer", (wq, scale[:, None]))
        del accs
        samples = [
            ((tree(run_block(block, a, kw), lambda t: t.cpu()), *a[1:]), kw)
            for a, kw in samples
        ]
        torch.cuda.empty_cache()
        rows = [report[f"blocks.{i}.{s}"] for s in CONVERT_SUFFIXES]
        mean = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
        print(
            f"[gptq] block {i:2d} w8a8 rtn {mean['w8a8_rtn']:.2e} gptq "
            f"{mean['w8a8_gptq']:.2e} act_aware {mean['w8a8_act_aware']:.2e} | w8a16 "
            f"rtn {mean['w8a16_rtn']:.2e} gptq {mean['w8a16_gptq']:.2e} | "
            f"{time.perf_counter() - t0:.0f}s",
            flush=True,
        )

    keys = next(iter(report.values())).keys()
    total = {k: sum(r[k] for r in report.values()) / len(report) for k in keys}
    print("[gptq] mean over linears " + json.dumps(total), flush=True)
    del anima, bundle, samples
    torch.cuda.empty_cache()
    meta = {
        "source": src.name,
        "source_sha256": sha256_file(src),
        "base_fingerprint": fingerprint,
        "method": args.method,
        "calibration": json.dumps(calib),
        "gptq": json.dumps(
            {
                "block": args.block,
                "damp": args.damp,
                "ls_damp": args.ls_damp,
                "act_order": not args.no_act_order,
            }
        ),
    }
    int8_id = write_file([src], out, copies, meta)
    out.with_suffix(".report.json").write_text(
        json.dumps(
            {"meta": meta, "int8_id": int8_id, "mean": total, "linears": report},
            indent=1,
        )
    )
    print(
        f"[gptq] wrote {out} ({out.stat().st_size / 2**30:.2f} GiB) int8_id "
        f"{int8_id[:16]} in {time.perf_counter() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
