"""Phase 1b: does int8 change the LoRA gradient by more than minibatch noise?

Every LoRA gradient on a target linear is a linear map of ``G = dYᵀX`` (up: ``G Aᵀ``,
down: ``Bᵀ G``), so this compares ``G`` itself, sketched to ``S = (dY Ω)ᵀ X``
(Ω Gaussian, ``q`` columns, norms preserved in expectation). No LoRA is applied.

For each σ and each of ``n_images`` real cached batches (B=1, one fixed ε), the base
DiT (gradient checkpointing on) runs five passes: fp32 (SDPA, no autocast, TF32 off;
run first, before any int8 copy exists), bf16, bf16 again (the run-to-run floor), int8
``fwd`` (int8 forward, bf16 dX), int8 ``full`` (int8 forward and dX), and the two halves
of the ``fwd`` forward in bf16 emulation: ``w8a16`` (int8 weight, bf16 activation) and
``w16a8`` (bf16 weight, int8 per-token activation). Both split arms take dX from the
bf16 W, as ``fwd`` does. Per module:

    e32_mode = RMS_i ‖S_mode,i − S_fp32,i‖       (distance from the fp32 gradient)
    e16_mode = RMS_i ‖S_mode,i − S_bf16,i‖       (distance from the bf16 gradient)
    noise    = RMS_i ‖S_fp32,i − S_fp32,i-1‖

Gate (proposal.md): median over modules of e16_full / noise < 0.1 at every σ. The fp32
columns ask the calibrating question: is int8 much further from fp32 than bf16 is?
The prediction itself is compared too, against ‖pred‖ and against the residual
‖pred − target‖ that drives the gradient.

    make daemon-run ARGS="--stall-timeout 900 project/int8_train/src/grad_error.py"
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
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from bench._anima import add_common_args, add_model_args, build_anima  # noqa: E402
from bench._common import start_heartbeat  # noqa: E402
from bench.grad_init.probe_subspace import (  # noqa: E402
    enumerate_targets,
    latent_tokens,
    module_kind,
)
from library.env import resolve_under_home  # noqa: E402
from library.io.cache import (  # noqa: E402
    discover_cached_pairs,
    load_cached_latents,
    load_cached_text_features,
)
from comfy_kitchen.backends.eager.quantization import (  # noqa: E402
    _build_hadamard,
    _rotate_activation,
    quantize_int8_rowwise,
)
from library.moss.int8 import rotate  # noqa: E402
from int8_linear import (  # noqa: E402  (sibling module; script dir is on sys.path)
    GROUP,
    Int8TrainLinear,
    convert_dit_,
    quantize,
)

MODES = ("bf16", "bf16_rep", "fwd", "full", "w8a16", "w16a8")  # after the fp32 phase
INT8_MODES = MODES[2:]


def qdq_tokens(x: torch.Tensor) -> torch.Tensor:
    """The kernel's activation side: comfy_kitchen's eager reference (bf16 rotation,
    per-token int8 at amax), dequantised and rotated back in fp32. Autocast off: the
    rotation is a matmul, which would drop to bf16."""
    with torch.autocast("cuda", enabled=False):
        xb = x.to(torch.bfloat16).reshape(-1, x.shape[-1])
        h = _build_hadamard(GROUP, device=x.device, dtype=torch.bfloat16)
        q, sc = quantize_int8_rowwise(_rotate_activation(xb, h, GROUP))
        return rotate(q.float() * sc).reshape(x.shape)


class _SplitArm(torch.autograd.Function):
    """fp32 GEMM on one emulated int8 side and one bf16 side, so the emulation adds no
    rounding of its own; ``dX = dY W`` with the bf16 W, as mode ``fwd``."""

    @staticmethod
    def forward(ctx, x, w_fwd, w, quant_x):
        ctx.save_for_backward(w)
        ctx.x_dtype = x.dtype
        with torch.autocast("cuda", enabled=False):
            xf = qdq_tokens(x) if quant_x else x.to(torch.bfloat16).float()
            y = F.linear(xf, w_fwd.float())
        return y.to(torch.bfloat16)

    @staticmethod
    def backward(ctx, dy):
        (w,) = ctx.saved_tensors
        return (dy.to(w.dtype) @ w).to(ctx.x_dtype), None, None, None


class W8A16Linear(nn.Linear):
    def forward(self, x):
        with torch.autocast("cuda", enabled=False):
            w_hat = rotate(self.wq.float() * self.s)  # dequantised, in W's basis
        return _SplitArm.apply(x, w_hat, self.weight, False)


class W16A8Linear(nn.Linear):
    def forward(self, x):
        return _SplitArm.apply(x, self.weight, self.weight, True)


SPLIT = {"w8a16": W8A16Linear, "w16a8": W16A8Linear}


def set_mode(mods: list[nn.Linear], mode: str) -> None:
    for m in mods:
        if mode.startswith("bf16"):
            m.__class__ = nn.Linear
        elif mode in SPLIT:
            m.__class__ = SPLIT[mode]
        else:
            m.__class__ = Int8TrainLinear
            m.int8_mode = mode


class Sketcher:
    """``S = (dY Ω)ᵀ X`` per target linear, collected on the backward pass."""

    def __init__(self, targets, q, device, seed):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.omega = {
            ln: torch.randn(mod.out_features, q, generator=gen).to(device)
            for ln, _o, mod in targets
        }
        self.out: dict[str, torch.Tensor] = {}
        self.capturing = False
        for ln, _o, mod in targets:
            mod.register_forward_hook(self._hook(ln))

    def _hook(self, ln):
        def hook(_mod, inputs, y):
            # only the first forward; checkpoint recompute re-runs the hooks
            if not self.capturing or not y.requires_grad:
                return
            x = inputs[0].detach().to("cpu")

            def on_grad(dy):
                d = dy.reshape(-1, dy.shape[-1]).float() @ self.omega[ln]
                xx = x.to(dy.device).reshape(-1, x.shape[-1]).float()
                self.out[ln] = d.T @ xx

            y.register_hook(on_grad)

        return hook

    def take(self) -> dict[str, torch.Tensor]:
        out, self.out = self.out, {}
        return out


def pick_images(cache_root: Path, n: int, min_tokens: int, max_tokens: int, seed):
    pairs = discover_cached_pairs(str(cache_root))
    gen = torch.Generator().manual_seed(seed)
    picked = []
    for i in torch.randperm(len(pairs), generator=gen).tolist():
        lat = load_cached_latents(pairs[i].npz_path)[0]
        if min_tokens <= latent_tokens(lat) <= max_tokens:
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


def one_pass(anima, sk, latents, crossattn, noise, sigma, fp32=False):
    dtype = torch.float32 if fp32 else torch.bfloat16
    s4 = sigma.view(-1, 1, 1, 1)
    noisy = ((1.0 - s4) * latents + s4 * noise).unsqueeze(2).to(dtype)
    noisy.requires_grad_(True)
    pad = torch.zeros(1, 1, *latents.shape[-2:], dtype=dtype, device=latents.device)
    sk.capturing = True
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not fp32):
        pred = anima(noisy, sigma, crossattn.to(dtype), padding_mask=pad)
    sk.capturing = False
    pred = pred.squeeze(2).float()
    loss = torch.nn.functional.mse_loss(pred, noise - latents)
    loss.backward()
    return sk.take(), float(loss), pred.detach()


def fp32_phase(anima, sk, images, sigmas, seed, device, t0):
    """fp32 sketches and predictions for every (σ, image), kept on the CPU."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtypes = {
        n: t.dtype for n, t in [*anima.named_parameters(), *anima.named_buffers()]
    }
    attn_mode = anima.attn_mode
    anima.float()
    anima.attn_mode = "torch"  # flash has no fp32 kernel
    ref = {}
    for sigma_v in sigmas:
        sigma = torch.tensor([sigma_v], device=device)
        for j, ci in enumerate(images):
            latents, crossattn, noise = load_batch(ci, j, seed, device)
            S, loss, pred = one_pass(
                anima, sk, latents, crossattn, noise, sigma, fp32=True
            )
            ref[sigma_v, j] = ({k: v.cpu() for k, v in S.items()}, pred.cpu(), loss)
            print(
                f"fp32 σ={sigma_v} image {j + 1}/{len(images)} loss={loss:.4f} "
                f"{time.perf_counter() - t0:.0f}s",
                flush=True,
            )
    for n, t in [*anima.named_parameters(), *anima.named_buffers()]:
        t.data = t.data.to(dtypes[n])
    anima.attn_mode = attn_mode
    return ref


def sq(a, b):
    return float((a - b).pow(2).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(ap, vae=False, text_encoder=False)
    add_common_args(ap, include_compile=False, include_checkpointing=False)
    ap.add_argument("--cache_root", default="post_image_dataset/lora")
    ap.add_argument("--sigmas", default="0.1,0.5,0.9")
    ap.add_argument("--n_images", type=int, default=6)
    ap.add_argument("--sketch_q", type=int, default=64)
    ap.add_argument("--min_tokens", type=int, default=3500)
    ap.add_argument("--max_tokens", type=int, default=4608)
    ap.add_argument("--out", default="project/int8_train/assets/grad_error_fp32.json")
    args = ap.parse_args()
    # torch non-reentrant: eager activations at 4k tokens next to the bf16 weights
    # and two int8 copies do not fit in 16 GB. The sketch hooks park X on the CPU.
    args.gradient_checkpointing = True
    start_heartbeat(label="grad_error")

    bundle = build_anima(args, adapter=None, train_mode=True)
    anima, device = bundle.anima, bundle.device
    targets = enumerate_targets(anima)
    sk = Sketcher(targets, args.sketch_q, device, args.seed)
    images = pick_images(
        Path(resolve_under_home(args.cache_root)),
        args.n_images,
        args.min_tokens,
        args.max_tokens,
        args.seed,
    )
    sigmas = [float(s) for s in args.sigmas.split(",")]
    t0 = time.perf_counter()
    ref32 = fp32_phase(anima, sk, images, sigmas, args.seed, device, t0)

    names = convert_dit_(anima, "fwd")
    mods = [anima.get_submodule(n) for n in names]
    for m in mods:
        wq_t, s_t = quantize(m.weight.detach().t().contiguous())
        m.register_buffer("wq_t", wq_t, persistent=False)
        m.register_buffer("s_t", s_t, persistent=False)
    print(f"converted {len(mods)} linears, {len(targets)} LoRA targets", flush=True)

    # acc[σ][module][stat] = Σ of squared norms over images
    acc = {s: defaultdict(lambda: defaultdict(float)) for s in sigmas}
    pred_err = {s: defaultdict(list) for s in sigmas}
    losses = {s: defaultdict(list) for s in sigmas}
    for sigma_v in sigmas:
        sigma = torch.tensor([sigma_v], device=device)
        for j, ci in enumerate(images):
            latents, crossattn, noise = load_batch(ci, j, args.seed, device)
            S32, pred32, loss32 = ref32[sigma_v, j]
            pred32 = pred32.to(device)
            resid = float((pred32 - (noise - latents)).norm())
            losses[sigma_v]["fp32"].append(loss32)
            S = {}
            for mode in MODES:
                set_mode(mods, mode)
                S[mode], loss, pred = one_pass(
                    anima, sk, latents, crossattn, noise, sigma
                )
                losses[sigma_v][mode].append(loss)
                d = float((pred - pred32).norm())
                pred_err[sigma_v][f"{mode}_vs_pred"].append(d / float(pred32.norm()))
                pred_err[sigma_v][f"{mode}_vs_resid"].append(d / resid)
            for ln, r16 in S["bf16"].items():
                a = acc[sigma_v][ln]
                r32 = S32[ln].to(device)
                a["ref"] += float(r32.pow(2).sum())
                for mode in MODES:
                    a[f"e32_{mode}"] += sq(S[mode][ln], r32)
                for mode in MODES[1:]:
                    a[f"e16_{mode}"] += sq(S[mode][ln], r16)
                if j:
                    a["noise"] += sq(r32, ref32[sigma_v, j - 1][0][ln].to(device))
                    a["n_noise"] += 1
                a["n"] += 1
            del S
            print(
                f"σ={sigma_v} image {j + 1}/{len(images)} "
                f"tokens={latent_tokens(latents)} "
                f"loss fp32={loss32:.4f} "
                + " ".join(f"{m}={losses[sigma_v][m][-1]:.4f}" for m in MODES)
                + f" {time.perf_counter() - t0:.0f}s",
                flush=True,
            )

    stats = [f"e32_{m}" for m in MODES] + [f"e16_{m}" for m in MODES[1:]]
    report = {"args": vars(args), "images": [ci.npz_path for ci in images]}
    report["sigmas"] = {}
    for sigma_v in sigmas:
        rows = []
        for ln, orig, _mod in targets:
            a = acc[sigma_v][ln]
            if not a["n"] or not a["n_noise"]:
                continue
            noise = (a["noise"] / a["n_noise"]) ** 0.5
            row = {
                "module": orig,
                "kind": module_kind(orig),
                "ref": (a["ref"] / a["n"]) ** 0.5,
                "noise": noise,
            }
            for k in stats:
                row[f"{k}_vs_noise"] = (a[k] / a["n"]) ** 0.5 / noise
            for mode in INT8_MODES:
                row[f"{mode}_over_bf16_e32"] = (a[f"e32_{mode}"] / a["e32_bf16"]) ** 0.5
            rows.append(row)
        keys = [f"{k}_vs_noise" for k in stats] + [
            f"{m}_over_bf16_e32" for m in INT8_MODES
        ]
        med = {k: statistics.median(r[k] for r in rows) for k in keys}
        by_depth = defaultdict(lambda: defaultdict(list))
        for r in rows:
            parts = r["module"].split(".")
            if parts[0] == "blocks":
                g = f"blocks {int(parts[1]) // 7 * 7}-{int(parts[1]) // 7 * 7 + 6}"
                for k in (
                    "e32_bf16_vs_noise",
                    "e32_full_vs_noise",
                    *(f"{m}_over_bf16_e32" for m in INT8_MODES),
                ):
                    by_depth[g][k].append(r[k])
        report["sigmas"][str(sigma_v)] = {
            "median": med,
            "gate_pass": med["e16_full_vs_noise"] < 0.1,
            "by_depth": {
                g: {k: statistics.median(v) for k, v in d.items()}
                for g, d in sorted(
                    by_depth.items(), key=lambda kv: int(kv[0].split()[1].split("-")[0])
                )
            },
            "pred_err": {k: statistics.mean(v) for k, v in pred_err[sigma_v].items()},
            "loss_mean": {m: statistics.mean(v) for m, v in losses[sigma_v].items()},
            "worst_full_over_bf16": sorted(
                rows, key=lambda r: -r["full_over_bf16_e32"]
            )[:10],
            "rows": rows,
        }
        print(
            f"σ={sigma_v} " + " ".join(f"{k}={v:.4f}" for k, v in med.items()),
            flush=True,
        )
        print(
            f"σ={sigma_v} pred "
            + " ".join(
                f"{k}={v:.4f}"
                for k, v in report["sigmas"][str(sigma_v)]["pred_err"].items()
            ),
            flush=True,
        )
    path = resolve_under_home(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
