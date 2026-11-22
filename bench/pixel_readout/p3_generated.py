#!/usr/bin/env python3
"""P3: the P2 head on generated images, base DiT and a LoRA on it (read-only).

Each (prompt, seed) is rendered with the standard sampler (``generate_body``: Euler,
50 steps, shift 3, CFG 3.5, the bench negative prompt). Hooks keep the last step's
sampler state x_t (σ_last), the cond and uncond pre-head features and both velocities.
The last Euler step is x_final = x_t − σ_last·v_cfg, so the target is VAE(x_final),
the decode of what the sampler produced (FastFrames' target). Three head reads per image:

- ``cfg``: head(f_u + g·(f_c − f_u), x_t). The tap is the input of ``final_layer.linear``,
  so this feature maps to v_cfg exactly; the head was trained on cond features only.
- ``cond``: head(f_c, x_t), the feature the head was trained on.
- ``tf``: teacher-forced on the generated latent (fresh noise to σ_last, one cond forward),
  the P2 train/test setting on generated content.

``--adapter`` runs every prompt twice, multiplier 0 (base) then 1 (LoRA); the head stays
the base-trained one (``--arms lora`` skips the base pass). ``cfg_check`` is the relative
error of x_t − σ·(v_u + g(v_c − v_u)) against the sampler's x_final, which confirms the
combine is plain CFG. At ``--guidance_scale 1`` the sampler runs cond only and ``cfg`` =
``cond``.

``tf`` is read against two targets: VAE(x_final) (``psnr_tf``) and VAE(x̂0) of its own
forward, x_tf − σ·v_tf (``psnr_tf_xhat0``, what the head is trained on). At σ_last 0.06
the two nearly coincide; at turbo's σ_last 0.5 they do not.

``--x2``: ×2 heads (``head_x2.safetensors`` of an ``experiments/x2_head.py`` run, or a file path) read
through the library tap's path (the CFG feature, ``PixelHeadX2``). The reference is the
chain the head replaces, the released ResShift ×2 student on VAE(x_final); per image:
×2 against it (dB, DINO distance, whole-image and dark-pixel luma), ×1 against VAE(x_final) (empty for a file with
no ×1 output), and
the time of VAE decode + ResShift against the head's ×2 readout. Sheets: two columns
``VAE → ResShift ×2 | ×2 head | bicubic ×2 of VAE``, full image and a 768-px crop at
native pixels.

Usage::

    make daemon-run ARGS="--queue --stall-timeout 900 bench/pixel_readout/p3_generated.py \
        --compile --adapter output/ckpt/anima_merged15.safetensors --label p3_mmdd_hhmm"
    # T2, turbo 4-step (shift 3, CFG 1, last forward σ 0.5):
    make daemon-run ARGS="--queue --stall-timeout 900 bench/pixel_readout/p3_generated.py \
        --compile --adapter models/loras/anima_turbo_4step.safetensors --arms lora \
        --infer_steps 4 --guidance_scale 1.0 --head_run <run> --label t2probe_mmdd_hhmm"
    # ×2 read, base 28 steps CFG 4 (turbo: --adapter …v4_longer --arms lora
    # --infer_steps 4 --guidance_scale 1.0):
    make daemon-run ARGS="--queue --stall-timeout 900 bench/pixel_readout/p3_generated.py \
        --compile --x2 --head_run <x2 run> --arms base --infer_steps 28 \
        --guidance_scale 4.0 --label x2gen_mmdd_hhmm"
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

from bench._anima import (
    DEFAULT_NEG,
    DEFAULT_TEXT_ENCODER,
    add_common_args,
    add_model_args,
    build_anima,
)
from bench._common import make_run_dir, start_heartbeat, write_result
from bench.pixel_readout._prompts import PROMPTS
from bench.pixel_readout._shared import bicubic2, load_head, resolve_run
from bench.pixel_readout._teacher import TEACHER_DIR, TEACHER_REPO, load_teacher
from library.anima.pixel_head import (
    PATCH,
    PixelHeadX2,
    head_scales,
    lat_patches,
    load_pixel_head,
)
from library.anima.pixel_head_train import (
    DinoFeat,
    attach_adapter,
    compile_dit,
    densest_window,
    pair_sheet,
    psnr,
    tone_stats,
    zoom_sheet,
)
from library.anima.vocab_pack import default_vocab_pack
from library.anima.weights import _attach_vocab_pack_if_set
from library.env import resolve_under_home
from scripts.pixel_head._setup import DINO_REPO
from library.inference.args import build_default_args
from library.inference.generation import generate_body
from library.inference.models import load_text_encoder
from library.inference.sampling import get_timesteps_sigmas
from library.inference.text import prepare_text_inputs
from library.models import qwen_vae

HEAD_RUN = "bench/pixel_readout/results/20261119-0959-convnext_1119_0938"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(p, text_encoder=False)
    add_common_args(p)
    p.add_argument(
        "--head_run",
        nargs="+",
        default=[HEAD_RUN],
        help="one or more heads (path or label); each reads the same renders",
    )
    p.add_argument("--adapter", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--only", nargs="*", default=None, help="prompt names (smoke)")
    p.add_argument("--text_encoder", default=DEFAULT_TEXT_ENCODER)
    p.add_argument("--infer_steps", type=int, default=50)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument(
        "--arms", nargs="+", default=["base", "lora"], choices=["base", "lora"]
    )
    p.add_argument(
        "--stream",
        choices=["fp32", "bf16"],
        default="fp32",
        help="DiT residual stream; fp32 is inference.py's default and the head's since "
        "replay_1119_1225",
    )
    p.add_argument(
        "--int8",
        action="store_true",
        help="render on the int8 base (inference.py --int8: GPTQ forward copies beside "
        "the DiT); the heads were trained on the bf16 GEMMs",
    )
    p.add_argument(
        "--x2",
        action="store_true",
        help="×2 heads (head_x2.safetensors) read against VAE → ResShift ×2 (see the "
        "module docstring)",
    )
    p.add_argument("--crop", type=int, default=768, help="--x2 sheet crop, ×2 pixels")
    args = p.parse_args()
    start_heartbeat(label="p3")
    device = torch.device(args.device)
    prompts = [x for x in PROMPTS if not args.only or x[0] in args.only]

    _, sigmas = get_timesteps_sigmas(args.infer_steps, args.flow_shift, device)
    sigma_last = float(sigmas[-2])
    vae = qwen_vae.load_vae(
        args.vae, device="cpu", dtype=torch.bfloat16, eval=True, vae_2d=True
    ).to(device)
    compile_blocks, args.compile = args.compile, False
    anima = build_anima(args, adapter=None, train_mode=False).anima
    if args.int8:  # inference.py's order: convert, stream change, adapters
        from library.anima.int8 import convert_for_inference

        convert_for_inference(
            anima, argparse.Namespace(dit=args.dit, int8_file=None, lora_weight=[])
        )
    if args.stream == "fp32":
        from library.anima.stream import stream_fp32_

        stream_fp32_(anima)
    network = attach_adapter(anima, args.adapter, device) if args.adapter else None
    # the prompt side tokenizes with the base.toml vocab pack (as `make test`); the
    # DiT needs the pack rows or a CJK prompt indexes past the stock T5 table
    _attach_vocab_pack_if_set(anima, default_vocab_pack())
    if compile_blocks:
        toks = sorted({(h // PATCH) * (w // PATCH) for _, _, (h, w) in prompts})
        compile_dit(anima, [(toks[0], toks[-1])], tag="int8" if args.int8 else "")

    # --- taps: every forward appends (t, x_in, feature, v); reset when t changes ---
    calls: list[dict] = []

    def pre(_m, a, kw):
        t = float(a[1].flatten()[0])
        if calls and calls[-1]["t"] != t:
            calls.clear()
        calls.append({"t": t, "x": a[0].detach()})

    def post(_m, _a, _kw, out):
        calls[-1]["v"] = out.detach()

    anima.register_forward_pre_hook(pre, with_kwargs=True)
    anima.register_forward_hook(post, with_kwargs=True)
    anima.final_layer.linear.register_forward_pre_hook(
        lambda _m, inp: calls[-1].__setitem__("f", inp[0].detach())
    )

    def head_name(r) -> str:
        r = Path(str(r))
        return (r.parent.name if r.suffix == ".safetensors" else r.name).split("-")[-1]

    def load_x2(r):
        r = str(r)
        path = Path(resolve_under_home(r)) if r.endswith(".safetensors") else None
        path = path or resolve_run(r) / "head_x2.safetensors"
        head = load_pixel_head(path, device, anima.final_layer.hidden_size)
        assert isinstance(head, PixelHeadX2), f"{path}: not a ×2 head (no x2_depth)"
        return head

    if args.x2:
        heads = {head_name(r): load_x2(r) for r in args.head_run}
        up2 = load_teacher(
            Path(resolve_under_home(TEACHER_REPO)),
            Path(resolve_under_home(TEACHER_DIR)),
            device,
        )
        dn = DinoFeat(device, DINO_REPO)

        def timed(fn):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn()
            torch.cuda.synchronize()
            return out, (time.perf_counter() - t0) * 1e3
    else:
        heads = {
            head_name(r): load_head(r, anima.final_layer.hidden_size, device)
            for r in args.head_run
        }

    def read(head, f, x_t):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return head(f[0, 0].float(), lat_patches(x_t).float()).float().clamp(-1, 1)

    gen_base = [
        "--prompt", "x", "--negative_prompt", DEFAULT_NEG,
        "--dit", str(args.dit), "--vae", str(args.vae),
        "--text_encoder", str(args.text_encoder), "--save_path", "unused",
        "--infer_steps", str(args.infer_steps), "--flow_shift", str(args.flow_shift),
        "--guidance_scale", str(args.guidance_scale), "--sampler", "euler",
    ]  # fmt: skip

    te_path = str(resolve_under_home(args.text_encoder))
    te = load_text_encoder(text_encoder=te_path, dtype=torch.bfloat16, device=device)
    shared: dict = {"text_encoder": te.eval()}
    contexts = [
        prepare_text_inputs(
            device=device,
            anima=anima,
            shared_models=shared,
            prompt=prompt,
            negative_prompt=DEFAULT_NEG,
            text_encoder_path=te_path,
        )
        for _, prompt, _ in prompts
    ]
    shared.clear()
    del te
    torch.cuda.empty_cache()

    run_dir = make_run_dir("pixel_readout", label=args.label)
    arms = [a for a in args.arms if a == "base" or network is not None]
    assert arms, "--arms lora needs --adapter"
    n_calls = 2 if args.guidance_scale != 1.0 else 1
    rows, artifacts = [], []
    for arm in arms:
        if network is not None:
            network.set_multiplier(1.0 if arm == "lora" else 0.0)
        for k, (name, prompt, (h, w)) in enumerate(prompts):
            ctx, ctx_null = contexts[k]
            for seed in args.seeds:
                gargs = build_default_args(
                    gen_base + ["--image_size", str(h), str(w), "--seed", str(seed)]
                )
                final = generate_body(gargs, anima, ctx, ctx_null, device, seed)
                assert (
                    len(calls) == n_calls and abs(calls[0]["t"] - sigma_last) < 1e-4
                ), [c["t"] for c in calls]
                cond, unc = calls if n_calls == 2 else (calls[0], calls[0])
                x_t = cond["x"].float().squeeze(2)  # (1, C, h, w)
                g = args.guidance_scale
                v_cfg = unc["v"].float() + g * (cond["v"].float() - unc["v"].float())
                x_chk = x_t - sigma_last * v_cfg.squeeze(2)
                fin = final.float().reshape(x_t.shape)
                cfg_check = float((x_chk - fin).norm() / fin.norm())
                f_cfg = unc["f"].float() + g * (cond["f"].float() - unc["f"].float())

                with torch.no_grad():
                    tgt, ms_vae = (
                        timed(lambda: vae.decode_to_pixels(fin)[0].float())
                        if args.x2
                        else (vae.decode_to_pixels(fin)[0].float(), None)
                    )
                f_cond = cond["f"].float()

                if args.x2:
                    calls.clear()
                    t2, ms_teacher = timed(lambda: up2(tgt.clamp(-1, 1), seed))
                    cx, cy = densest_window(t2, args.crop)
                    cols = [("VAE -> ResShift x2", t2)]
                    for hname, head in heads.items():
                        lat_p = lat_patches(x_t).float()

                        def run_head(head=head, lat_p=lat_p):
                            with (
                                torch.no_grad(),
                                torch.autocast("cuda", dtype=torch.bfloat16),
                            ):
                                return head(f_cfg[0, 0].float(), lat_p, scale=2)

                        run_head()  # warm
                        x2_img, ms_head = timed(run_head)
                        x2_img = x2_img.float().clamp(-1, 1)
                        psnr_x1 = None  # a file stamped scales = 2 has no ×1 output
                        if 1 in head_scales(head):
                            with (
                                torch.no_grad(),
                                torch.autocast("cuda", dtype=torch.bfloat16),
                            ):
                                x1_img = head(f_cfg[0, 0].float(), lat_p, scale=1)
                            psnr_x1 = psnr(x1_img.float().clamp(-1, 1), tgt)
                        lum, dark = tone_stats(x2_img, t2)
                        row = {
                            "arm": arm,
                            "head": hname,
                            "name": name,
                            "seed": seed,
                            "size": [h, w],
                            "psnr_x2": psnr(x2_img, t2),
                            "dino_x2": dn.image(x2_img, t2),
                            "luma_x2": lum,
                            "dark_x2": dark,
                            "psnr_x1": psnr_x1,
                            "psnr_bicubic": psnr(bicubic2(tgt), t2),
                            "ms_vae": ms_vae,
                            "ms_teacher": ms_teacher,
                            "ms_head_x2": ms_head,
                            "cfg_check": cfg_check,
                        }
                        rows.append(row)
                        print(
                            f"  {arm} {hname} {name} s{seed}: ×2 {row['psnr_x2']:.2f} "
                            f"dino {1e3 * row['dino_x2']:.3f}e-3 dark {dark if dark is None else round(dark, 1)}"
                            f"  ×1 {'-' if psnr_x1 is None else f'{psnr_x1:.2f}'}"
                            f"  bicubic {row['psnr_bicubic']:.2f}  "
                            f"ms vae {ms_vae:.0f} + rs {ms_teacher:.0f} vs head "
                            f"{ms_head:.1f}",
                            flush=True,
                        )
                        cols.append((f"x2 head {hname}", x2_img))
                    cols.append(("bicubic x2 of VAE", bicubic2(tgt)))
                    for tag, at in (
                        ("full", None),
                        ("crop", (cx, cy, args.crop)),
                    ):
                        fn = f"x2_{arm}_{k:02d}_{name}_s{seed}_{tag}.png"
                        pair_sheet(cols, run_dir / fn, at=at)
                        artifacts.append(fn)
                    continue

                # teacher-forced on the generated latent (P2 setting)
                gn = torch.Generator(device=device).manual_seed(
                    seed * 100003 + 30_000 + k
                )
                noise = torch.randn(fin.shape, generator=gn, device=device)
                x_tf = (1 - sigma_last) * fin + sigma_last * noise
                calls.clear()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    anima(
                        x_tf.unsqueeze(2).to(torch.bfloat16),
                        torch.tensor([sigma_last], device=device),
                        ctx["embed"][0].to(device, torch.bfloat16),
                        padding_mask=torch.zeros(
                            1, 1, *fin.shape[-2:], dtype=torch.bfloat16, device=device
                        ),
                    )
                f_tf = calls[0]["f"].float()
                x0_tf = x_tf - sigma_last * calls[0]["v"].float().squeeze(2)
                calls.clear()
                with torch.no_grad():
                    tgt_tf = vae.decode_to_pixels(x0_tf)[0].float()

                cols = [("VAE decode", tgt)]
                for hname, head in heads.items():
                    pred_cfg = read(head, f_cfg, x_t)
                    pred_cond = read(head, f_cond, x_t)
                    pred_tf = read(head, f_tf, x_tf)
                    row = {
                        "arm": arm,
                        "head": hname,
                        "name": name,
                        "seed": seed,
                        "size": [h, w],
                        "psnr_cfg": psnr(pred_cfg, tgt),
                        "psnr_cond": psnr(pred_cond, tgt),
                        "psnr_tf": psnr(pred_tf, tgt),
                        "psnr_tf_xhat0": psnr(pred_tf, tgt_tf),
                        "psnr_xhat0_tf_vs_final": psnr(tgt_tf, tgt),
                        "cfg_check": cfg_check,
                    }
                    rows.append(row)
                    print(
                        f"  {arm} {hname} {name} s{seed}: cfg {row['psnr_cfg']:.2f}  "
                        f"cond {row['psnr_cond']:.2f}  tf {row['psnr_tf']:.2f} (own x̂0 "
                        f"{row['psnr_tf_xhat0']:.2f})  chk {cfg_check:.1e}",
                        flush=True,
                    )
                    if len(heads) > 1:
                        cols.append((hname, pred_cfg))
                    elif n_calls == 2:
                        cols += [
                            ("head, CFG feature", pred_cfg),
                            ("head, cond feature", pred_cond),
                        ]
                    else:
                        cols.append(("head", pred_cond))
                fn = f"sheet_{arm}_{k:02d}_{name}_s{seed}.png"
                zoom_sheet(
                    cols,
                    run_dir / fn,
                    at=densest_window(tgt, 384),
                )
                artifacts.append(fn)

    summary = {}
    keys = (
        (
            "psnr_x2",
            "dino_x2",
            "luma_x2",
            "dark_x2",
            "psnr_x1",
            "psnr_bicubic",
            "ms_vae",
            "ms_teacher",
            "ms_head_x2",
        )
        if args.x2
        else (
            "psnr_cfg",
            "psnr_cond",
            "psnr_tf",
            "psnr_tf_xhat0",
            "psnr_xhat0_tf_vs_final",
        )
    )
    for arm, hname in ((a, hn) for a in arms for hn in heads):
        rs = [r for r in rows if r["arm"] == arm and r["head"] == hname]
        key_ = f"{arm}/{hname}"
        summary[key_] = {}
        for key in keys:
            vals = [r[key] for r in rs if r[key] is not None]
            summary[key_][key] = sum(vals) / len(vals) if vals else None
        summary[key_]["cfg_check_max"] = max(r["cfg_check"] for r in rs)
        print(f"{key_}: {summary[key_]}", flush=True)
    out = write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={"sigma_last": sigma_last, "summary": summary, "per_image": rows},
        label=args.label,
        artifacts=artifacts,
        device=device,
    )
    print(f"envelope: {out}")


if __name__ == "__main__":
    main()
