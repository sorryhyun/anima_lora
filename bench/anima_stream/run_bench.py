"""Anima's fp32 stream on the path users generate on: no autocast and a bf16 sampler state
on ``--stream bf16``, against ``library.anima.stream`` on ``--stream fp32``.

Prediction error at fixed noised images, final-latent distance over full trajectories
through ``generate_body``, each against a true fp32 forward (SDPA, no autocast), plus
ms/step and peak. ``*_norope`` leaves the q / k norms and RoPE in bf16; ``bf16_ac`` is the trainer's
forward, the base of ``project/int8_train/reports/anima_stream_attrib.md``.

    make daemon-run ARGS="--stall-timeout 600 bench/anima_stream/run_bench.py"
"""

from __future__ import annotations

import argparse
import contextlib
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

from library.runtime.allocator import default_expandable_segments

default_expandable_segments()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from diffusers.utils.torch_utils import randn_tensor  # noqa: E402

import library.anima.models as am  # noqa: E402
from bench._anima import add_common_args, add_model_args, build_anima  # noqa: E402
from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402
from bench.grad_init.probe_subspace import latent_tokens  # noqa: E402
from library.anima.int8 import convert_dit_, forward_file  # noqa: E402
from library.anima.stream import (  # noqa: E402
    Fp32Linear,
    Fp32RMSNorm,
    latent_dtype,
    stream_fp32_,
)
from library.anima.vocab_pack import attach_vocab_pack, resolve_active_pack  # noqa: E402
from library.env import resolve_under_home  # noqa: E402
from library.inference.args import build_default_args  # noqa: E402
from library.inference.generation import generate_body  # noqa: E402
from library.inference.text import prepare_text_inputs  # noqa: E402
from library.io.cache import (  # noqa: E402
    discover_cached_pairs,
    load_cached_latents,
    load_cached_text_features,
)

PRED_ARMS = (
    "bf16,bf16_ac,fp32,fp32_norope,int8,int8_ac,int8_fp32,int8_fp32_norope".split(",")
)
TRAJ_ARMS = "bf16,fp32,fp32_norope,int8,int8_fp32,int8_fp32_norope".split(",")
NEGATIVE = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia"


def unstream(dit):
    for m in dit.modules():
        if type(m) is Fp32Linear:
            m.__class__ = nn.Linear
        elif type(m) is Fp32RMSNorm:
            m.__class__ = am.RMSNorm
    dit.__dict__.pop("forward_mini_train_dit", None)
    dit.__dict__.pop("stream", None)


def set_arm(dit, arm):
    unstream(dit)
    if "fp32" in arm:
        stream_fp32_(dit, rope="norope" not in arm)


@contextlib.contextmanager
def fp32_reference(anima):
    """A true fp32 forward; hands every arm's context through unchanged."""
    dtypes = {
        n: t.dtype for n, t in [*anima.named_parameters(), *anima.named_buffers()]
    }
    attn_mode = anima.attn_mode

    def to_fp32(_mod, args, kwargs):
        return tuple(a.float() if a.is_floating_point() else a for a in args), kwargs

    anima.float()
    anima.attn_mode = "torch"
    anima.stream = "fp32"  # x, σ and the sampler state in fp32; no conversion
    hook = anima.register_forward_pre_hook(to_fp32, with_kwargs=True)
    try:
        yield
    finally:
        hook.remove()
        del anima.stream
        for n, t in [*anima.named_parameters(), *anima.named_buffers()]:
            t.data = t.data.to(dtypes[n])
        anima.attn_mode = attn_mode


def pick_images(root, n, lo, hi, seed):
    """``anima_stream_attrib.py``'s pick, so the images match its tables."""
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


def load_cell(ci, j, seed, device):
    latents = load_cached_latents(ci.npz_path)[0].unsqueeze(0).to(device)
    crossattn, _ = load_cached_text_features(ci.te_path, variant=0)
    crossattn = crossattn.unsqueeze(0).to(device, torch.bfloat16)
    gen = torch.Generator(device=device).manual_seed(seed * 1000 + j)
    noise = torch.randn(latents.shape, generator=gen, device=device)
    return latents, crossattn, noise


@torch.no_grad()
def predict(anima, cell, sigma_v, arm):
    latents, crossattn, noise = cell
    dtype = latent_dtype(anima)
    x = ((1.0 - sigma_v) * latents + sigma_v * noise).unsqueeze(2).to(dtype)
    t = torch.tensor([sigma_v], device=x.device, dtype=dtype)
    pad = torch.zeros(1, 1, *latents.shape[-2:], dtype=dtype, device=x.device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=arm.endswith("_ac")):
        return anima(x, t, crossattn, padding_mask=pad).squeeze(2).float()


def run_pred(anima, cells, sigmas, arms, out):
    for arm in arms:
        set_arm(anima, arm)
        for (s, j), (cell, ref) in cells.items():
            v = predict(anima, cell, s, arm)
            out[arm, s].append(float((v - ref).norm() / ref.norm() * 100))
    unstream(anima)


def trajectory(anima, args, ctx, noise, device, label):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    z = generate_body(
        args,
        anima,
        ctx[0],
        ctx[1],
        device,
        args.seed,
        latents=noise.to(device, latent_dtype(anima)),
    ).float()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"{label} {seconds:.1f}s peak {peak:.2f} GiB", flush=True)
    return z.cpu(), seconds, peak


def run_traj(anima, gens, refs, arms, out, device, steps):
    for arm in arms:
        set_arm(anima, arm)
        for (name, gen_args, ctx, noise), ref in zip(gens, refs):
            z, seconds, peak = trajectory(
                anima, gen_args, ctx, noise, device, f"{arm} {name}"
            )
            out[arm]["dist"].append(float((z - ref).norm() / ref.norm() * 100))
            out[arm]["ms_step"].append(seconds / steps * 1000)
            out[arm]["peak"].append(peak)
    unstream(anima)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(ap, vae=False, text_encoder=True)
    add_common_args(ap, include_compile=False, include_checkpointing=False)
    ap.add_argument("--cache_root", default="post_image_dataset/lora")
    ap.add_argument("--sigmas", default="0.1,0.5,0.9")
    ap.add_argument("--n_images", type=int, default=6)
    ap.add_argument("--min_tokens", type=int, default=3500)
    ap.add_argument("--max_tokens", type=int, default=4608)
    ap.add_argument("--prompts", default="project/int8_train/src/ab_prompts.txt")
    ap.add_argument("--n_prompts", type=int, default=3)
    ap.add_argument("--seeds", default="11,22")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--image_size", type=int, default=1024)
    ap.add_argument("--pred_arms", default=",".join(PRED_ARMS))
    ap.add_argument("--traj_arms", default=",".join(TRAJ_ARMS))
    ap.add_argument("--no_int8", action="store_true")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    start_heartbeat(label="anima_stream")

    bundle = build_anima(args, adapter=None, train_mode=False)
    anima, device = bundle.anima, bundle.device
    pack = resolve_active_pack(args)
    if pack is not None:
        attach_vocab_pack(anima, pack)

    sigmas = [float(s) for s in args.sigmas.split(",")]
    images = pick_images(
        Path(resolve_under_home(args.cache_root)),
        args.n_images,
        args.min_tokens,
        args.max_tokens,
        args.seed,
    )
    print("images:", [Path(c.npz_path).stem for c in images], flush=True)

    lines = resolve_under_home(args.prompts).read_text().splitlines()
    prompts = []
    for line in lines:
        p = re.sub(r"\s--d\s+\d+\s*$", "", line).strip()
        if p and p not in prompts:
            prompts.append(p)
    prompts = prompts[: args.n_prompts]
    contexts = [
        prepare_text_inputs(
            device=device,
            anima=anima,
            prompt=p,
            negative_prompt=NEGATIVE,
            text_encoder_path=args.text_encoder,
        )
        for p in prompts
    ]

    gens = []
    for pi, ctx in enumerate(contexts):
        for seed in (int(s) for s in args.seeds.split(",")):
            gen_args = build_default_args(
                [
                    "--text_encoder",
                    args.text_encoder,
                    "--save_path",
                    "unused",
                    "--prompt",
                    prompts[pi],
                    "--image_size",
                    str(args.image_size),
                    str(args.image_size),
                    "--infer_steps",
                    str(args.steps),
                    "--flow_shift",
                    "3.0",
                    "--guidance_scale",
                    "4.0",
                    "--sampler",
                    "euler",
                    "--seed",
                    str(seed),
                ]
            )
            side = args.image_size // 8
            noise = randn_tensor(
                (1, 16, 1, side, side),
                generator=torch.Generator("cpu").manual_seed(seed),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            gens.append((f"p{pi}s{seed}", gen_args, ctx, noise))

    cells, refs, ref_stats = {}, [], defaultdict(list)
    with fp32_reference(anima):
        for s in sigmas:
            for j, ci in enumerate(images):
                cell = load_cell(ci, j, args.seed, device)
                cells[s, j] = (cell, predict(anima, cell, s, "fp32_ref"))
        for name, gen_args, ctx, noise in gens:
            z, seconds, peak = trajectory(
                anima, gen_args, ctx, noise, device, f"ref {name}"
            )
            refs.append(z)
            ref_stats["ms_step"].append(seconds / args.steps * 1000)
            ref_stats["peak"].append(peak)

    pred_arms = args.pred_arms.split(",")
    traj_arms = args.traj_arms.split(",")
    pred, traj = defaultdict(list), defaultdict(lambda: defaultdict(list))
    run_pred(anima, cells, sigmas, [a for a in pred_arms if "int8" not in a], pred)
    run_traj(
        anima,
        gens,
        refs,
        [a for a in traj_arms if "int8" not in a],
        traj,
        device,
        args.steps,
    )
    if not args.no_int8:
        names = convert_dit_(anima, "infer", forward_file(None, args.dit) or None)
        print(f"int8 infer: {len(names)} linears", flush=True)
        run_pred(anima, cells, sigmas, [a for a in pred_arms if "int8" in a], pred)
        run_traj(
            anima,
            gens,
            refs,
            [a for a in traj_arms if "int8" in a],
            traj,
            device,
            args.steps,
        )

    print("\nRESULT prediction error, % of ‖v_fp32‖, mean over images")
    for arm in dict.fromkeys(a for a, _ in pred):
        row = " ".join(f"σ{s}={statistics.mean(pred[arm, s]):.3f}" for s in sigmas)
        print(f"RESULT {arm}: {row}")
    print("RESULT trajectory: final latent % from fp32, ms/step, peak GiB")
    print(
        f"RESULT fp32 reference: {statistics.median(ref_stats['ms_step']):.0f} ms "
        f"{max(ref_stats['peak']):.2f} GiB"
    )
    for arm, r in traj.items():
        print(
            f"RESULT {arm}: {statistics.mean(r['dist']):.3f} "
            f"[{min(r['dist']):.3f}–{max(r['dist']):.3f}] "
            f"{statistics.median(r['ms_step']):.0f} ms {max(r['peak']):.2f} GiB"
        )

    run_dir = make_run_dir("anima_stream", label=args.label)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={
            "images": [Path(c.npz_path).stem for c in images],
            "prompts": prompts,
            "pred": {f"{a}@{s}": v for (a, s), v in pred.items()},
            "traj": {a: dict(r) for a, r in traj.items()},
            "reference": dict(ref_stats),
        },
    )
    print(f"wrote {run_dir}", flush=True)


if __name__ == "__main__":
    main()
