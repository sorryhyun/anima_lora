"""Where a one-image gradient spike enters: per block, ``‖dY‖`` of ``o`` and the share
of its energy in the top token, beside the smallest per-token input RMS each
``_RMSNorm`` sees (its gain is ``1 / sqrt(rms² + 1e-6)``), per arm, on one (image, σ)
of ``grad_error_moss.py`` (same pick, same ε). Arm ``fp32`` is its reference.

    make daemon-run ARGS="project/int8_train/src/grad_spike.py --image channel4 \\
        --int8_file <file> --exclude_report <report>"
"""

from __future__ import annotations

from library.runtime.allocator import default_expandable_segments

default_expandable_segments()

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from library.env import resolve_under_home  # noqa: E402
from library.moss.loader import load_transformer  # noqa: E402
from library.moss.train import NUM_TRAIN_TIMESTEPS, checkpoint_blocks  # noqa: E402

# sibling modules
from grad_error_moss import arm, enter_fp32, exit_fp32, load_batch, pick_items  # noqa: E402
from library.moss.dit_int8.base import check_base, load_copies, moss_targets  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--cache", default="post_image_dataset/moss/channel_(caststation)_layout"
    )
    ap.add_argument("--image", default="channel4")
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--arms", default="fp32,bf16,bf16_f32s,gptq_w8a16,gptq_f32s")
    ap.add_argument("--n_images", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--int8_file", required=True)
    ap.add_argument("--exclude_report", required=True)
    ap.add_argument(
        "--out", default="project/int8_train/assets/moss_grad_spike_trace.json"
    )
    args = ap.parse_args()
    device = torch.device("cuda")

    tr = load_transformer("", "base", device=device)
    tr.requires_grad_(False)
    checkpoint_blocks(tr.layers)
    targets = moss_targets(tr)
    mods = [m for _n, m in targets]
    copies, meta = load_copies(args.int8_file)
    check_base(meta, dict(targets))
    for n, m in targets:
        m.copy = copies[n]
    report = json.loads(resolve_under_home(args.exclude_report).read_text())
    exclude = frozenset(s for v in report["stems"].values() for s in v)
    items = pick_items(
        resolve_under_home(args.cache), args.n_images, 0, 1 << 30, args.seed, exclude
    )
    j = next(k for k, it in enumerate(items) if it.stem == args.image)
    latent, embeds, noise, ref = load_batch(items[j], j, args.seed, device)
    n_img = latent.shape[-2] * latent.shape[-1]
    print(f"{args.image}: {n_img} image + {embeds.shape[1]} text tokens", flush=True)

    rec, capturing = {}, {"on": False}

    def hook_of(name):
        def hook(_mod, _inputs, y):
            if not capturing["on"] or not y.requires_grad:
                return

            def on_grad(dy):
                t = dy.float().reshape(-1, dy.shape[-1]).pow(2).sum(-1)
                top = int(t.argmax())
                rec[name] = (float(t.sum()) ** 0.5, float(t[top] / t.sum()), top)

            y.register_hook(on_grad)

        return hook

    for n, m in targets:
        m.register_forward_hook(hook_of(n))

    rms = {}

    def rms_hook(key):
        def hook(_mod, inputs):
            if not capturing["on"]:
                return
            r = inputs[0].detach().float().square().mean(-1).sqrt()
            r = r.reshape(r.shape[1], -1).amin(-1)
            top = int(r.argmin())
            rms[key] = (float(r[top]), top, float(r.median()))

        return hook

    for i, layer in enumerate(tr.layers):
        for key in ("norm1", "norm2", "q_norm", "k_norm"):
            getattr(layer, key).register_forward_pre_hook(rms_hook(f"{i}.{key}"))

    sigma = args.sigma
    noisy = ((1.0 - sigma) * latent + sigma * noise).requires_grad_(True)
    timestep = torch.tensor([NUM_TRAIN_TIMESTEPS * sigma], device=device)
    runs = {}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    linears = [m for m in tr.modules() if type(m) is nn.Linear]
    for name in args.arms.split(","):
        rec, rms = {}, {}
        fp32 = name == "fp32"
        dtypes = enter_fp32(tr, linears) if fp32 else None
        with contextlib.nullcontext() if fp32 else arm(tr, mods, name):
            capturing["on"] = True
            pred = tr(noisy, timestep, embeds, return_dict=False)[0]
            capturing["on"] = False
            loss = F.mse_loss(pred.float(), noise - latent)
            loss.backward()
        noisy.grad = None
        if fp32:
            exit_fp32(tr, linears, dtypes)
        runs[name] = {"dy": rec, "rms": rms}
        print(f"{name} loss {loss.item():.5f}", flush=True)

    print("arm: block ‖dY_o‖ top@tok | min rms q@tok (median) k norm1 norm2")
    for name, r in runs.items():
        print(f"--- {name}")
        for i in (0, 6, 12, 18, 22, 23, 24, 25, 26, 28, 34, 41):
            d = r["dy"][f"layers.{i}.proj.o"]
            m = r["rms"]
            cols = " ".join(
                f"{k[0]} {m[f'{i}.{k}'][0]:.2e}@{m[f'{i}.{k}'][1]}({m[f'{i}.{k}'][2]:.2g})"
                for k in ("q_norm", "k_norm", "norm1", "norm2")
            )
            print(f"{i:2d} {d[0]:.2e} {d[1]:.2f}@{d[2]} | {cols}", flush=True)
    path = resolve_under_home(args.out)
    path.write_text(json.dumps(runs))


if __name__ == "__main__":
    main()
