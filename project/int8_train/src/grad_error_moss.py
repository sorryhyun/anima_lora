"""The MOSS-Image counterpart of ``grad_error.py``: how far the int8 base moves the
LoRA gradient, measured against fp32 and against bf16's own distance from fp32.

Same passes per (σ, image) as ``grad_error.py``: fp32 first, then bf16, bf16 again,
int8 ``fwd``, ``full`` and the split arms ``w8a16`` / ``w16a8``, on the 252 block
linears ``layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}`` (the set
``TrainRequest.int8`` converts). Three changes for a 5B DiT on 16 GB:

- fp32 reference: every linear keeps its bf16 weight (``w16``) and runs an fp32 GEMM
  on an upcast copy per call, exact since the checkpoint is bf16; its ``weight`` is an
  empty fp32 buffer, so the model casts activations to fp32. Biases and norms are cast
  to fp32. SDPA (FA2 has no fp32 kernel), TF32 off.
- The int8 copies are quantized from the bf16 weight on every call (round to nearest,
  as ``library/moss/dit_int8/base.py``) instead of held: bf16 plus both int8 copies is ~19 GB.
- Both sides are sketched, ``S = (dY Ω)ᵀ (X Ψ)``, so a linear's input is not parked
  until its gradient arrives. Ω and Ψ are Gaussian, one per width.

Loss as ``library.moss.train.training_loss`` (plain flow-matching MSE, timestep
``1000·σ``), B=1, the full caption, eager, torch non-reentrant checkpointing per block.

    make daemon-run ARGS="--stall-timeout 900 project/int8_train/src/grad_error_moss.py"
"""

from __future__ import annotations

from library.runtime.allocator import default_expandable_segments

default_expandable_segments()

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from bench._common import start_heartbeat  # noqa: E402
from library.env import resolve_under_home  # noqa: E402
from library.moss import cache as moss_cache  # noqa: E402
from library.moss.int8 import rotate  # noqa: E402
from library.moss.loader import load_transformer  # noqa: E402
from library.moss.train import NUM_TRAIN_TIMESTEPS, checkpoint_blocks  # noqa: E402

# sibling modules; the script dir is on sys.path
from grad_error import _SplitArm, sq  # noqa: E402
from int8_linear import linear_full, linear_fwd, quantize  # noqa: E402
from library.moss.dit_int8.base import check_base, load_copies, moss_targets  # noqa: E402

# after the fp32 phase. ``bf16_f32s``: every parameter but the block linears' weights in
# fp32, so the residual stream, modulation and gates are fp32 around bf16 GEMMs.
# ``bf16_noreduce``: cuBLAS bf16 GEMMs without reduced-precision split-K reduction.
MODES = (
    "bf16",
    "bf16_rep",
    "fwd",
    "full",
    "w8a16",
    "w16a8",
    "bf16_f32s",
    "bf16_noreduce",
    "gptq",
    "gptq_w8a16",
    "full_f32s",
    "gptq_f32s",
)
DEFAULT_MODES = "bf16,bf16_rep,fwd,full,w8a16,w16a8"
GROUP_BLOCKS = 7


class F32Linear(nn.Linear):
    """fp32 GEMM on the bf16 weight held in ``w16``."""

    def forward(self, x):
        return F.linear(x.float(), self.w16.float(), self.bias)


class Int8Arm(nn.Linear):
    """One int8 arm, its copies quantized from the bf16 ``weight`` per call. The
    ``gptq`` arms read the forward copy from ``--int8_file`` (``self.copy``, held on
    the CPU); the backward copy stays round to nearest."""

    arm: str

    def forward(self, x):
        w = self.weight
        if self.arm == "w16a8":
            return _SplitArm.apply(x, w, w, True)
        if self.arm.startswith("gptq"):
            wq, s = (t.to(w.device) for t in self.copy)
        else:
            wq, s = quantize(w)
        if self.arm.endswith("w8a16"):
            return _SplitArm.apply(x, rotate(wq.float() * s), w, False)
        x = x.float()
        if self.arm == "fwd":
            return linear_fwd(x, wq, s, w)
        return linear_full(x, wq, s, *quantize(w.t().contiguous()))


def set_mode(mods: list[nn.Linear], mode: str) -> None:
    for m in mods:
        if mode.startswith("bf16"):
            m.__class__ = nn.Linear
        else:
            m.__class__ = Int8Arm
            m.arm = mode.removesuffix("_f32s")


def cast_fp32_except(tr, linears) -> dict:
    """Every parameter but ``linears``' weights to fp32; returns the old dtypes."""
    keep = {id(m.weight) for m in linears}
    dtypes = {}
    for n, p in tr.named_parameters():
        if id(p) not in keep:
            dtypes[n] = p.dtype
            p.data = p.data.float()
    return dtypes


def restore_dtypes(tr, dtypes) -> None:
    for n, p in tr.named_parameters():
        if n in dtypes:
            p.data = p.data.to(dtypes[n])


def enter_fp32(tr, linears) -> dict:
    dtypes = cast_fp32_except(tr, linears)
    for m in linears:
        w = m._parameters.pop("weight")
        m._buffers["w16"] = w
        m._buffers["weight"] = torch.empty(0, device=w.device)
        m.__class__ = F32Linear
    return dtypes


def exit_fp32(tr, linears, dtypes) -> None:
    for m in linears:
        del m._buffers["weight"]
        m._parameters["weight"] = m._buffers.pop("w16")
        m.__class__ = nn.Linear
    restore_dtypes(tr, dtypes)


@contextlib.contextmanager
def arm(tr, mods, mode):
    set_mode(mods, mode)
    if mode.endswith("_f32s"):
        dtypes = cast_fp32_except(tr, mods)
        try:
            yield
        finally:
            restore_dtypes(tr, dtypes)
    elif mode == "bf16_noreduce":
        matmul = torch.backends.cuda.matmul
        matmul.allow_bf16_reduced_precision_reduction = False
        try:
            yield
        finally:
            matmul.allow_bf16_reduced_precision_reduction = True
    else:
        yield


class Sketcher:
    """``S = (dY Ω)ᵀ (X Ψ)`` per target linear, collected on the backward pass."""

    def __init__(self, targets, q, q_in, device, seed):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.omega, self.psi = {}, {}
        for _name, m in targets:
            for d, n, k in (
                (self.omega, m.out_features, q),
                (self.psi, m.in_features, q_in),
            ):
                if n not in d:
                    d[n] = torch.randn(n, k, generator=gen).to(device)
        self.out: dict[str, torch.Tensor] = {}
        self.capturing = False
        for name, m in targets:
            m.register_forward_hook(self._hook(name, m.out_features, m.in_features))

    def _hook(self, name, n_out, n_in):
        omega, psi = self.omega[n_out], self.psi[n_in]

        def hook(_mod, inputs, y):
            # only the first forward; checkpoint recompute re-runs the hooks
            if not self.capturing or not y.requires_grad:
                return
            xp = (inputs[0].detach().reshape(-1, n_in).float() @ psi).cpu()

            def on_grad(dy):
                d = dy.reshape(-1, n_out).float() @ omega
                self.out[name] = (d.T @ xp.to(d.device)).cpu()

            y.register_hook(on_grad)

        return hook

    def take(self) -> dict[str, torch.Tensor]:
        out, self.out = self.out, {}
        return out


def pick_items(cache_dir, n, min_tokens, max_tokens, seed, exclude=frozenset()):
    items = [
        it
        for it in moss_cache.list_items(cache_dir)
        if min_tokens <= (it.width // 16) * (it.height // 16) <= max_tokens
        and it.stem not in exclude
    ]
    gen = torch.Generator().manual_seed(seed)
    return [items[i] for i in torch.randperm(len(items), generator=gen)[:n].tolist()]


def centre_crop(t, n):
    top, left = (t.shape[-2] - n) // 2, (t.shape[-1] - n) // 2
    return t[..., top : top + n, left : left + n] if n else t


def load_batch(item, j, seed, device, crop=0):
    latent, embeds = moss_cache.load_item(item)
    latent = centre_crop(latent.unsqueeze(0), crop).to(device)
    ref = moss_cache.load_ref(item)
    ref = None if ref is None else centre_crop(ref.unsqueeze(0), crop).to(device)
    gen = torch.Generator(device=device).manual_seed(seed * 1000 + j)
    noise = torch.randn(latent.shape, generator=gen, device=device)
    return latent, embeds.unsqueeze(0).to(device), noise, ref


def one_pass(tr, sk, latent, embeds, noise, ref, sigma):
    noisy = ((1.0 - sigma) * latent + sigma * noise).requires_grad_(True)
    timestep = torch.tensor([NUM_TRAIN_TIMESTEPS * sigma], device=latent.device)
    sk.capturing = True
    pred = tr(noisy, timestep, embeds, return_dict=False, ref=ref)[0]
    sk.capturing = False
    loss = F.mse_loss(pred.float(), noise - latent)
    loss.backward()
    return sk.take(), float(loss), pred.detach().float()


def module_kind(name: str) -> str:
    return name.rsplit(".", 1)[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model_dir", default="")
    ap.add_argument("--model", default="base")
    ap.add_argument(
        "--cache", default="post_image_dataset/moss/channel_(caststation)_layout"
    )
    ap.add_argument("--sigmas", default="0.1,0.5,0.9")
    ap.add_argument("--n_images", type=int, default=6)
    ap.add_argument("--sketch_q", type=int, default=64)
    ap.add_argument("--sketch_q_in", type=int, default=256)
    ap.add_argument("--min_tokens", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=1 << 30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modes", default=DEFAULT_MODES, help=f"bf16 first; of {MODES}")
    ap.add_argument("--int8_file", default="", help="dit_gptq_moss.py file (gptq arms)")
    ap.add_argument(
        "--crop",
        type=int,
        default=0,
        help="centre crop of the latent and reference, N×N tokens (i2i: 48 fits 16 GB)",
    )
    ap.add_argument(
        "--exclude_report",
        default="",
        help="its .report.json: skip the stems it calibrated on",
    )
    ap.add_argument("--out", default="project/int8_train/assets/moss_grad_error.json")
    args = ap.parse_args()
    modes = args.modes.split(",")
    if modes[0] != "bf16" or not set(modes) <= set(MODES):
        raise SystemExit(f"--modes: bf16 first, then any of {MODES}")
    others = modes[1:]
    start_heartbeat(label="grad_error_moss")
    device = torch.device("cuda")

    tr = load_transformer(args.model_dir, args.model, device=device)
    tr.requires_grad_(False)
    checkpoint_blocks(tr.layers)
    tr.ref_timestep, tr.ref_isolate = 0.0, True  # every i2i recipe; t2i ignores them
    targets = moss_targets(tr)
    mods = [m for _n, m in targets]
    if args.int8_file:
        copies, meta = load_copies(args.int8_file)
        check_base(meta, dict(targets))
        for n, m in targets:
            m.copy = copies[n]
    sk = Sketcher(targets, args.sketch_q, args.sketch_q_in, device, args.seed)
    exclude = frozenset()
    if args.exclude_report:
        report = json.loads(resolve_under_home(args.exclude_report).read_text())
        exclude = frozenset(s for v in report["stems"].values() for s in v)
    items = pick_items(
        resolve_under_home(args.cache),
        args.n_images,
        args.min_tokens,
        args.max_tokens,
        args.seed,
        exclude,
    )
    sigmas = [float(s) for s in args.sigmas.split(",")]
    print(
        f"{len(targets)} linears, images: "
        + ", ".join(
            f"{it.stem} ({(it.width // 16) * (it.height // 16)}+{it.n_text}+{it.n_ref})"
            for it in items
        ),
        flush=True,
    )
    t0 = time.perf_counter()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    linears = [m for m in tr.modules() if type(m) is nn.Linear]
    dtypes = enter_fp32(tr, linears)
    ref32 = {}
    for sigma in sigmas:
        for j, it in enumerate(items):
            latent, embeds, noise, ref = load_batch(it, j, args.seed, device, args.crop)
            S, loss, pred = one_pass(tr, sk, latent, embeds, noise, ref, sigma)
            ref32[sigma, j] = (S, pred.cpu(), loss)
            print(
                f"fp32 σ={sigma} image {j + 1}/{len(items)} loss={loss:.4f} "
                f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB "
                f"{time.perf_counter() - t0:.0f}s",
                flush=True,
            )
    exit_fp32(tr, linears, dtypes)
    torch.cuda.empty_cache()

    acc = {s: defaultdict(lambda: defaultdict(float)) for s in sigmas}
    pred_err = {s: defaultdict(list) for s in sigmas}
    losses = {s: defaultdict(list) for s in sigmas}
    per_image = []
    for sigma in sigmas:
        for j, it in enumerate(items):
            latent, embeds, noise, ref = load_batch(it, j, args.seed, device, args.crop)
            S32, pred32, loss32 = ref32[sigma, j]
            pred32 = pred32.to(device)
            resid = float((pred32 - (noise - latent)).norm())
            losses[sigma]["fp32"].append(loss32)
            S = {}
            for mode in modes:
                with arm(tr, mods, mode):
                    S[mode], loss, pred = one_pass(
                        tr, sk, latent, embeds, noise, ref, sigma
                    )
                losses[sigma][mode].append(loss)
                d = float((pred - pred32).norm())
                pred_err[sigma][f"{mode}_vs_pred"].append(d / float(pred32.norm()))
                pred_err[sigma][f"{mode}_vs_resid"].append(d / resid)
            set_mode(mods, "bf16")
            depth_err = {m: defaultdict(float) for m in modes}
            for mode in modes:
                for name, s in S[mode].items():
                    g = int(name.split(".")[1]) // GROUP_BLOCKS * GROUP_BLOCKS
                    depth_err[mode][g] += sq(s, S32[name])
            per_image.append({"sigma": sigma, "image": it.stem, "err": depth_err})
            for mode in others:
                print(
                    f"σ={sigma} {it.stem} {mode}/bf16 by depth: "
                    + " ".join(
                        f"{(depth_err[mode][g] / depth_err['bf16'][g]) ** 0.5:.2f}"
                        for g in sorted(depth_err[mode])
                    ),
                    flush=True,
                )
            for name, r16 in S["bf16"].items():
                a = acc[sigma][name]
                r32 = S32[name]
                a["ref"] += float(r32.pow(2).sum())
                for mode in modes:
                    a[f"e32_{mode}"] += sq(S[mode][name], r32)
                for mode in others:
                    a[f"e16_{mode}"] += sq(S[mode][name], r16)
                if j:
                    a["noise"] += sq(r32, ref32[sigma, j - 1][0][name])
                    a["n_noise"] += 1
                a["n"] += 1
            print(
                f"σ={sigma} image {j + 1}/{len(items)} loss fp32={loss32:.4f} "
                + " ".join(f"{m}={losses[sigma][m][-1]:.4f}" for m in modes)
                + f" peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB"
                + f" {time.perf_counter() - t0:.0f}s",
                flush=True,
            )

    stats = [f"e32_{m}" for m in modes] + [f"e16_{m}" for m in others]
    ratios = [f"{m}_over_bf16_e32" for m in others]
    report = {
        "args": vars(args),
        "images": [it.stem for it in items],
        "per_image": per_image,
        "sigmas": {},
    }
    for sigma in sigmas:
        rows = []
        for name, _m in targets:
            a = acc[sigma][name]
            if not a["n"] or not a["n_noise"]:
                continue
            noise = (a["noise"] / a["n_noise"]) ** 0.5
            row = {
                "module": name,
                "kind": module_kind(name),
                "block": int(name.split(".")[1]),
                "ref": (a["ref"] / a["n"]) ** 0.5,
                "noise": noise,
            }
            for k in stats:
                row[f"{k}_vs_noise"] = (a[k] / a["n"]) ** 0.5 / noise
            for mode in others:
                row[f"{mode}_over_bf16_e32"] = (a[f"e32_{mode}"] / a["e32_bf16"]) ** 0.5
            rows.append(row)
        keys = [f"{k}_vs_noise" for k in stats] + ratios
        med = {k: statistics.median(r[k] for r in rows) for k in keys}
        group_keys = (*(f"e32_{m}_vs_noise" for m in modes), *ratios)
        by_depth = defaultdict(lambda: defaultdict(list))
        by_kind = defaultdict(lambda: defaultdict(list))
        for r in rows:
            g0 = r["block"] // GROUP_BLOCKS * GROUP_BLOCKS
            for k in group_keys:
                by_depth[f"blocks {g0}-{g0 + GROUP_BLOCKS - 1}"][k].append(r[k])
                by_kind[r["kind"]][k].append(r[k])

        def medians(groups):
            return {
                g: {k: statistics.median(v) for k, v in d.items()}
                for g, d in groups.items()
            }

        report["sigmas"][str(sigma)] = {
            "median": med,
            "gate_pass": med["full_over_bf16_e32"] <= 2.0 if "full" in modes else None,
            "by_depth": medians(
                dict(
                    sorted(
                        by_depth.items(),
                        key=lambda kv: int(kv[0].split()[1].split("-")[0]),
                    )
                )
            ),
            "by_kind": medians(by_kind),
            "pred_err": {k: statistics.mean(v) for k, v in pred_err[sigma].items()},
            "loss_mean": {m: statistics.mean(v) for m, v in losses[sigma].items()},
            "worst_full_over_bf16": sorted(
                rows, key=lambda r: -r["full_over_bf16_e32"]
            )[:10]
            if "full" in modes
            else None,
            "rows": rows,
        }
        print(
            f"σ={sigma} " + " ".join(f"{k}={v:.4f}" for k, v in med.items()), flush=True
        )
        print(
            f"σ={sigma} pred "
            + " ".join(
                f"{k}={v:.4f}"
                for k, v in report["sigmas"][str(sigma)]["pred_err"].items()
            ),
            flush=True,
        )
    path = resolve_under_home(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
