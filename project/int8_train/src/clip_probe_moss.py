"""What clipping the per-token activation scale would give the int8 base: held-out
prediction error against fp32, for the scale ``ratio · amax / 127`` in place of the
kernel's ``amax / 127``.

``comfy_kitchen`` has no such argument, so each arm replays the kernel in fp32: rotate
the bf16 input, round per token to int8 with the clipped scale, GEMM on the dequantised
GPTQ copy, bf16 out. Ratio 1 is the kernel on an fp32 input, and should read
``grad_error_moss.py``'s ``gptq_f32s`` on the same images. The stream is fp32 (the
default plan), the weights are not refitted to the ratio, and the sampling is
``grad_error_moss.py``'s at its seed. Forward only.

``--arms`` takes ``r<ratio>`` and the kernel itself, ``kernel_bf16`` / ``kernel_fp32``
for its input dtype: it rounds around its activation division to that dtype.

    make daemon-run ARGS="--stall-timeout 900 project/int8_train/src/clip_probe_moss.py"
"""

from __future__ import annotations

from library.runtime.allocator import default_expandable_segments

default_expandable_segments()

import argparse  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

import comfy_kitchen  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from bench._common import start_heartbeat  # noqa: E402
from library.env import resolve_under_home  # noqa: E402
from library.moss.dit_int8.base import check_base, load_copies, moss_targets  # noqa: E402
from library.moss.int8 import GROUP, INT8_MAX, rotate  # noqa: E402
from library.moss.loader import load_transformer  # noqa: E402
from library.moss.train import NUM_TRAIN_TIMESTEPS  # noqa: E402

# sibling modules
from grad_error_moss import (  # noqa: E402
    cast_fp32_except,
    enter_fp32,
    exit_fp32,
    load_batch,
    pick_items,
    restore_dtypes,
)

BASE_FILE = "models/moss_image/moss_image_base_int8_gptq.safetensors"


class ClipArm(nn.Linear):
    """The kernel's forward with the activation scale clipped to ``ratio``."""

    ratio: float

    def forward(self, x):
        wq, s = (t.to(x.device) for t in self.copy)
        xr = rotate(x.to(torch.bfloat16).reshape(-1, x.shape[-1]).float())
        step = (xr.abs().amax(-1, keepdim=True) * self.ratio / INT8_MAX).clamp(
            min=1e-30
        )
        xq = (xr / step).round().clamp(-INT8_MAX - 1, INT8_MAX) * step
        y = F.linear(xq, wq.float() * s).to(torch.bfloat16)
        return y.reshape(*x.shape[:-1], wq.shape[0])


class KernelArm(nn.Linear):
    """``comfy_kitchen.int8_linear`` on an input cast to ``in_dtype``."""

    in_dtype: torch.dtype

    def forward(self, x):
        wq, s = (t.to(x.device) for t in self.copy)
        y = comfy_kitchen.int8_linear(
            x.to(self.in_dtype).reshape(-1, x.shape[-1]),
            wq,
            s,
            out_dtype=torch.bfloat16,
            convrot=True,
            convrot_groupsize=GROUP,
        )
        return y.reshape(*x.shape[:-1], wq.shape[0])


def set_arm(mods, arm: str) -> None:
    for m in mods:
        if arm.startswith("kernel_"):
            m.__class__ = KernelArm
            m.in_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[arm[7:]]
        else:
            m.__class__ = ClipArm
            m.ratio = float(arm[1:])


def forward(tr, latent, embeds, noise, ref, sigma):
    noisy = (1.0 - sigma) * latent + sigma * noise
    timestep = torch.tensor([NUM_TRAIN_TIMESTEPS * sigma], device=latent.device)
    return tr(noisy, timestep, embeds, return_dict=False, ref=ref)[0].float()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model_dir", default="")
    ap.add_argument("--model", default="base")
    ap.add_argument(
        "--cache", default="post_image_dataset/moss/channel_(caststation)_layout"
    )
    ap.add_argument("--sigmas", default="0.1,0.5,0.9")
    ap.add_argument("--n_images", type=int, default=6)
    ap.add_argument("--arms", default="r1,r0.9,r0.8,r0.7,r0.6")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--int8_file", default=BASE_FILE)
    ap.add_argument("--out", default="project/int8_train/assets/moss_clip_probe.json")
    args = ap.parse_args()
    start_heartbeat(label="clip_probe_moss")
    torch.set_grad_enabled(False)
    device = torch.device("cuda")

    tr = load_transformer(args.model_dir, args.model, device=device)
    tr.ref_timestep, tr.ref_isolate = 0.0, True
    targets = moss_targets(tr)
    mods = [m for _n, m in targets]
    copies, meta = load_copies(args.int8_file)
    check_base(meta, dict(targets))
    for n, m in targets:
        m.copy = copies[n]
    path = resolve_under_home(args.int8_file).with_suffix(".report.json")
    report = json.loads(path.read_text())
    exclude = frozenset(s for v in report["stems"].values() for s in v)
    items = pick_items(
        resolve_under_home(args.cache), args.n_images, 0, 1 << 30, args.seed, exclude
    )
    sigmas = [float(s) for s in args.sigmas.split(",")]
    arms = args.arms.split(",")
    print("images: " + ", ".join(it.stem for it in items), flush=True)
    t0 = time.perf_counter()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    linears = [m for m in tr.modules() if type(m) is nn.Linear]
    dtypes = enter_fp32(tr, linears)
    ref32 = {}
    for sigma in sigmas:
        for j, it in enumerate(items):
            batch = load_batch(it, j, args.seed, device, 0)
            ref32[sigma, j] = forward(tr, *batch, sigma).cpu()
    exit_fp32(tr, linears, dtypes)
    torch.cuda.empty_cache()
    print(f"fp32 done, {time.perf_counter() - t0:.0f}s", flush=True)

    dtypes = cast_fp32_except(tr, mods)
    errs = {s: {a: [] for a in arms} for s in sigmas}
    for sigma in sigmas:
        for j, it in enumerate(items):
            batch = load_batch(it, j, args.seed, device, 0)
            pred32 = ref32[sigma, j].to(device)
            for a in arms:
                set_arm(mods, a)
                pred = forward(tr, *batch, sigma)
                errs[sigma][a].append(
                    float((pred - pred32).norm()) / float(pred32.norm())
                )
            print(
                f"σ={sigma} image {j + 1}/{len(items)} "
                + " ".join(f"{a}={errs[sigma][a][-1]:.4f}" for a in arms)
                + f" peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB "
                f"{time.perf_counter() - t0:.0f}s",
                flush=True,
            )
    restore_dtypes(tr, dtypes)

    means = {str(s): {a: statistics.mean(errs[s][a]) for a in arms} for s in sigmas}
    for s, row in means.items():
        print(f"σ={s} " + " ".join(f"{k}={v:.5f}" for k, v in row.items()), flush=True)
    out = resolve_under_home(args.out)
    out.write_text(
        json.dumps(
            {
                "args": vars(args),
                "images": [it.stem for it in items],
                "mean": means,
                "per_image": {str(s): d for s, d in errs.items()},
            },
            indent=1,
        )
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
