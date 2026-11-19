#!/usr/bin/env python3
"""End-to-end decode cost: the P2 head against the VAE, on the shipped `make test` sampler.

Each of P3's 16 prompts is rendered once with the `make test SPECTRUM=1` settings (anima-base,
Euler 28 steps, shift 3, CFG 4, the Spectrum flags of ``scripts/tasks/inference.py``; its last
step is an actual forward, so the head's tap is there). The final latent is then decoded four
ways, each timed with a CUDA sync and its peak VRAM over the resident DiT recorded:

- ``head``: head(f_u + g·(f_c − f_u), x_t) at the last step, head resident on the card.
- ``vae_swap``: the interactive path — VAE kept on CPU, moved to the card for the decode,
  moved back, ``empty_cache``.
- ``vae_disk``: the single-prompt path of ``inference.py`` — VAE loaded from disk per image.
- ``vae_resident``: decode only, VAE already on the card (reference).

Every arm ends with the pixels on the CPU in fp32. The head was trained at σ 0.0577 (50-step
grid); this grid's last σ is 0.1, so the head's PSNR against VAE(x_final) is recorded too.

Usage::

    make daemon-run ARGS="--queue --stall-timeout 900 bench/pixel_readout/e2e_speed.py \
        --compile --label e2e_mmdd_hhmm"
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

import networks.spectrum  # noqa: F401  (registers the Spectrum runner)
from bench._anima import (
    DEFAULT_NEG,
    DEFAULT_TEXT_ENCODER,
    add_common_args,
    add_model_args,
    build_anima,
)
from bench._common import make_run_dir, start_heartbeat, write_result
from bench.pixel_readout.convnext_head import lat_patches, load_head
from bench.pixel_readout.linear_probe import PATCH, psnr
from bench.pixel_readout.p3_generated import PROMPTS
from library.anima.vocab_pack import default_vocab_pack
from library.anima.weights import _attach_vocab_pack_if_set
from library.env import resolve_under_home
from library.inference.args import build_default_args
from library.inference.generation import generate_body
from library.inference.models import load_text_encoder
from library.inference.sampling import get_timesteps_sigmas
from library.inference.text import prepare_text_inputs
from library.models import qwen_vae
from scripts.tasks.inference import _spectrum_flags

HEAD_RUN = "bench/pixel_readout/results/20261119-1151-convnext_1119_1106"
MB = 1 << 20


def sync_time(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def peak_over(fn):
    """(result, seconds, peak MB above what was allocated before the call)."""
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out, dt = sync_time(fn)
    return out, dt, (torch.cuda.max_memory_allocated() - base) / MB


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(p, text_encoder=False)
    add_common_args(p)
    p.add_argument("--head_run", default=HEAD_RUN)
    p.add_argument("--text_encoder", default=DEFAULT_TEXT_ENCODER)
    p.add_argument("--infer_steps", type=int, default=28)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--vae_chunk_size", type=int, default=64)
    p.add_argument("--only", nargs="*", default=None, help="prompt names (smoke)")
    args = p.parse_args()
    start_heartbeat(label="e2e")
    device = torch.device(args.device)
    prompts = [x for x in PROMPTS if not args.only or x[0] in args.only]
    g = args.guidance_scale

    _, sigmas = get_timesteps_sigmas(args.infer_steps, args.flow_shift, device)
    sigma_last = float(sigmas[-2])

    compile_blocks, args.compile = args.compile, False
    bundle = build_anima(args, adapter=None, train_mode=False)
    anima = bundle.anima
    _attach_vocab_pack_if_set(anima, default_vocab_pack())
    if compile_blocks:
        toks = sorted({(h // PATCH) * (w // PATCH) for _, _, (h, w) in prompts})
        anima.compile_blocks(dynamic_seq=True, seq_bands=[(toks[0], toks[-1])])

    calls: list[dict] = []

    def pre(_m, a, kw):
        t = float(a[1].flatten()[0])
        if calls and calls[-1]["t"] != t:
            calls.clear()
        calls.append({"t": t, "x": a[0].detach()})

    anima.register_forward_pre_hook(pre, with_kwargs=True)
    anima.final_layer.linear.register_forward_pre_hook(
        lambda _m, inp: calls and calls[-1].__setitem__("f", inp[0].detach())
    )

    head = load_head(args.head_run, anima.final_layer.hidden_size, device)
    head_mb = sum(t.numel() * t.element_size() for t in head.state_dict().values()) / MB

    vae_kw = dict(
        disable_mmap=True,
        spatial_chunk_size=args.vae_chunk_size,
        disable_cache=True,
        vae_2d=True,
    )

    def load_vae_cpu():
        v = qwen_vae.load_vae(args.vae, device="cpu", **vae_kw)
        return v.to(torch.bfloat16).eval()

    vae_cpu = load_vae_cpu()
    vae_gpu = load_vae_cpu().to(device)
    vae_mb = (
        sum(t.numel() * t.element_size() for t in vae_gpu.state_dict().values()) / MB
    )

    def decode(vae, lat):
        with torch.no_grad():
            return vae.decode_to_pixels(lat.to(device, vae.dtype))[0].float().cpu()

    def arm_head(f_c, f_u, x_t):
        def run():
            f = f_u.float() + g * (f_c.float() - f_u.float())
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                img = head(f[0, 0], lat_patches(x_t).float())
            return img.float().clamp(-1, 1).cpu()

        return run

    def arm_swap(lat):
        def run():
            vae_cpu.to(device)
            img = decode(vae_cpu, lat)
            vae_cpu.to("cpu")
            torch.cuda.empty_cache()
            return img

        return run

    def arm_disk(lat):
        def run():
            v = qwen_vae.load_vae(args.vae, device="cpu", **vae_kw)
            v.to(device, dtype=torch.bfloat16).eval()
            img = decode(v, lat)
            del v
            torch.cuda.empty_cache()
            return img

        return run

    # --- text: encode every prompt up front (same cost for every arm) ---
    te_path = str(resolve_under_home(args.text_encoder))
    te = load_text_encoder(text_encoder=te_path, dtype=torch.bfloat16, device=device)
    shared: dict = {"text_encoder": te.eval()}
    contexts, te_s = [], []
    for _, prompt, _ in prompts:
        ctx, dt = sync_time(
            lambda: prepare_text_inputs(
                device=device,
                anima=anima,
                shared_models=shared,
                prompt=prompt,
                negative_prompt=DEFAULT_NEG,
                text_encoder_path=te_path,
            )
        )
        contexts.append(ctx)
        te_s.append(dt)
    shared.clear()
    del te
    torch.cuda.empty_cache()

    gen_base = [
        "--prompt", "x", "--negative_prompt", DEFAULT_NEG,
        "--dit", str(args.dit), "--vae", str(args.vae),
        "--text_encoder", str(args.text_encoder), "--save_path", "unused",
        "--infer_steps", str(args.infer_steps), "--flow_shift", str(args.flow_shift),
        "--guidance_scale", str(g), "--sampler", "euler", *_spectrum_flags(),
    ]  # fmt: skip

    def render(k, seed):
        _, _, (h, w) = prompts[k]
        ctx, ctx_null = contexts[k]
        gargs = build_default_args(
            gen_base + ["--image_size", str(h), str(w), "--seed", str(seed)]
        )
        with torch.no_grad():
            return generate_body(gargs, anima, ctx, ctx_null, device, seed)

    def decode_all(final, x_t, f_c, f_u):
        lat = final.float().reshape(x_t.shape)
        out = {}
        for name, fn in (
            ("head", arm_head(f_c, f_u, x_t)),
            ("vae_resident", lambda: decode(vae_gpu, lat)),
            ("vae_swap", arm_swap(lat)),
            ("vae_disk", arm_disk(lat)),
        ):
            out[name] = peak_over(fn)
        return out

    # --- warmup: one render + every decode arm per resolution (compile, cudnn) ---
    seen = set()
    for k, (_, _, hw) in enumerate(prompts):
        if hw in seen:
            continue
        seen.add(hw)
        final = render(k, 1000)
        cond, unc = calls
        decode_all(final, cond["x"].float().squeeze(2), cond["f"], unc["f"])
        print(f"warmup {hw} done", flush=True)

    rows = []
    for k, (name, _, (h, w)) in enumerate(prompts):
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        final, dit_s = sync_time(lambda: render(k, args.seed))
        dit_peak = (torch.cuda.max_memory_allocated() - base) / MB
        assert len(calls) == 2 and abs(calls[0]["t"] - sigma_last) < 1e-4, [
            c["t"] for c in calls
        ]
        cond, unc = calls
        x_t = cond["x"].float().squeeze(2)
        res = decode_all(final, x_t, cond["f"], unc["f"])
        row = {
            "name": name,
            "size": [h, w],
            "te_s": te_s[k],
            "dit_s": dit_s,
            "dit_peak_mb": dit_peak,
            "psnr_head_vs_vae": psnr(res["head"][0], res["vae_resident"][0]),
        }
        for arm, (_, dt, pk) in res.items():
            row[f"{arm}_s"] = dt
            row[f"{arm}_peak_mb"] = pk
        rows.append(row)
        calls.clear()
        print(
            f"  {name} {h}x{w}: dit {dit_s:.2f}s  head {row['head_s'] * 1e3:.0f}ms "
            f"swap {row['vae_swap_s']:.2f}s disk {row['vae_disk_s']:.2f}s "
            f"resident {row['vae_resident_s']:.2f}s  psnr {row['psnr_head_vs_vae']:.2f}",
            flush=True,
        )

    keys = [k for k in rows[0] if k.endswith(("_s", "_mb")) or k.startswith("psnr")]
    summary = {
        k: {
            "median": statistics.median(r[k] for r in rows),
            "mean": statistics.fmean(r[k] for r in rows),
            "sum": sum(r[k] for r in rows),
        }
        for k in keys
    }
    for k, v in summary.items():
        print(
            f"{k}: median {v['median']:.4f}  mean {v['mean']:.4f}  sum {v['sum']:.2f}"
        )
    run_dir = make_run_dir("pixel_readout", label=args.label)
    out = write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={
            "sigma_last": sigma_last,
            "head_weights_mb": head_mb,
            "vae_weights_mb": vae_mb,
            "summary": summary,
            "per_image": rows,
        },
        label=args.label,
        device=device,
    )
    print(f"envelope: {out}")


if __name__ == "__main__":
    main()
