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
the base-trained one. ``cfg_check`` is the relative error of x_t − σ·(v_u + g(v_c − v_u))
against the sampler's x_final, which confirms the combine is plain CFG.

Usage::

    make daemon-run ARGS="--queue --stall-timeout 900 bench/pixel_readout/p3_generated.py \
        --compile --adapter output/ckpt/anima_merged15.safetensors --label p3_mmdd_hhmm"
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

from bench._anima import (
    DEFAULT_NEG,
    DEFAULT_PROMPT,
    DEFAULT_TEXT_ENCODER,
    add_common_args,
    add_model_args,
    build_anima,
)
from bench._common import make_run_dir, start_heartbeat, write_result
from bench.pixel_readout.convnext_head import (
    load_head,
    densest_window,
    lat_patches,
    sheet,
)
from bench.pixel_readout.linear_probe import PATCH, psnr
from library.anima.vocab_pack import default_vocab_pack
from library.anima.weights import _attach_vocab_pack_if_set
from library.env import resolve_under_home
from library.inference.args import build_default_args
from library.inference.generation import generate_body
from library.inference.models import load_text_encoder
from library.inference.sampling import get_timesteps_sigmas
from library.inference.text import prepare_text_inputs
from library.models import qwen_vae

HEAD_RUN = "bench/pixel_readout/results/20261119-0959-convnext_1119_0938"
Q = "masterpiece, best quality, score_7, safe. "
PROMPTS = [  # (name, prompt, (h, w))
    ("anima_sign", DEFAULT_PROMPT, (1024, 1024)),
    (
        "sakura",
        Q
        + "1girl, solo, silver hair, long hair, red eyes, school uniform, cherry blossoms, falling petals, wind, looking back, upper body.",
        (1216, 832),
    ),
    (
        "neon_rain",
        Q
        + "1boy, black jacket, city street at night, neon signs, rain, puddle reflections, cinematic lighting.",
        (832, 1216),
    ),
    (
        "castle",
        Q
        + "scenery, no humans, fantasy castle on a cliff, sunset, dramatic clouds, detailed background.",
        (832, 1216),
    ),
    (
        "cafe",
        Q
        + "2girls, cafe interior, sitting at table, cake, coffee cup, window light, smile.",
        (1024, 1024),
    ),
    (
        "maid",
        Q
        + "1girl, maid, frills, lace, detailed clothes, holding tray, full body, white background.",
        (1216, 832),
    ),
    (
        "chibi",
        Q + "chibi, 1girl, flat color, thick outlines, sticker, white background.",
        (1024, 1024),
    ),
    (
        "candle",
        Q
        + "1girl, dark room, candlelight, low key lighting, dramatic shadows, portrait.",
        (1024, 1024),
    ),
    (
        "beach",
        Q + "1girl, swimsuit, beach, blue sky, ocean, sunlight, splashing water.",
        (832, 1216),
    ),
    (
        "battle",
        Q + "1girl, armor, sword, battle, sparks, dynamic pose, motion lines.",
        (1216, 832),
    ),
    (
        "kotatsu",
        Q + "cat, kotatsu, mandarin oranges, cozy room, tatami, detailed background.",
        (1024, 1024),
    ),
    (
        "library",
        Q + "1girl, glasses, library, bookshelf, many books, reading, soft light.",
        (1216, 832),
    ),
    (
        "shop_sign_ja",
        Q
        + '1girl, apron, shop front, holding sign, japanese text, the sign reads "ありがとう".',
        (1024, 1024),
    ),
    (
        "manga_page",
        Q
        + "monochrome, greyscale, comic, manga, panels, 1girl, speech bubble, japanese text, screentone.",
        (1216, 832),
    ),
    (
        "manga_action",
        Q
        + "monochrome, greyscale, comic, 1boy, 1girl, speech bubble, japanese text, action lines, sound effects.",
        (1216, 832),
    ),
    (
        "sketch",
        Q + "monochrome, greyscale, sketch, lineart, hatching, 1girl, portrait.",
        (1024, 1024),
    ),
]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(p, text_encoder=False)
    add_common_args(p)
    p.add_argument("--head_run", default=HEAD_RUN)
    p.add_argument("--adapter", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--only", nargs="*", default=None, help="prompt names (smoke)")
    p.add_argument("--text_encoder", default=DEFAULT_TEXT_ENCODER)
    p.add_argument("--infer_steps", type=int, default=50)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument("--guidance_scale", type=float, default=3.5)
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
    bundle = build_anima(args, adapter=args.adapter, train_mode=False)
    anima, network = bundle.anima, bundle.network
    # the prompt side tokenizes with the base.toml vocab pack (as `make test`); the
    # DiT needs the pack rows or a CJK prompt indexes past the stock T5 table
    _attach_vocab_pack_if_set(anima, default_vocab_pack())
    if compile_blocks:
        toks = sorted({(h // PATCH) * (w // PATCH) for _, _, (h, w) in prompts})
        anima.compile_blocks(dynamic_seq=True, seq_bands=[(toks[0], toks[-1])])

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

    head = load_head(args.head_run, anima.final_layer.hidden_size, device)

    def read(f, x_t):
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
    arms = ["base"] + (["lora"] if network is not None else [])
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
                assert len(calls) == 2 and abs(calls[0]["t"] - sigma_last) < 1e-4, [
                    c["t"] for c in calls
                ]
                cond, unc = calls
                x_t = cond["x"].float().squeeze(2)  # (1, C, h, w)
                g = args.guidance_scale
                v_cfg = unc["v"].float() + g * (cond["v"].float() - unc["v"].float())
                x_chk = x_t - sigma_last * v_cfg.squeeze(2)
                fin = final.float().reshape(x_t.shape)
                cfg_check = float((x_chk - fin).norm() / fin.norm())
                f_cfg = unc["f"].float() + g * (cond["f"].float() - unc["f"].float())

                with torch.no_grad():
                    tgt = vae.decode_to_pixels(fin)[0].float()
                pred_cfg = read(f_cfg, x_t)
                pred_cond = read(cond["f"].float(), x_t)

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
                pred_tf = read(calls[0]["f"].float(), x_tf)
                calls.clear()

                row = {
                    "arm": arm,
                    "name": name,
                    "seed": seed,
                    "size": [h, w],
                    "psnr_cfg": psnr(pred_cfg, tgt),
                    "psnr_cond": psnr(pred_cond, tgt),
                    "psnr_tf": psnr(pred_tf, tgt),
                    "cfg_check": cfg_check,
                }
                rows.append(row)
                print(
                    f"  {arm} {name} s{seed}: cfg {row['psnr_cfg']:.2f}  cond "
                    f"{row['psnr_cond']:.2f}  tf {row['psnr_tf']:.2f}  chk {cfg_check:.1e}",
                    flush=True,
                )
                fn = f"sheet_{arm}_{k:02d}_{name}_s{seed}.png"
                sheet(
                    [
                        ("VAE decode", tgt),
                        ("head, CFG feature", pred_cfg),
                        ("head, cond feature", pred_cond),
                    ],
                    run_dir / fn,
                    at=densest_window(tgt, 384),
                )
                artifacts.append(fn)

    summary = {}
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        summary[arm] = {
            key: sum(r[key] for r in rs) / len(rs)
            for key in ("psnr_cfg", "psnr_cond", "psnr_tf")
        }
        summary[arm]["cfg_check_max"] = max(r["cfg_check"] for r in rs)
        print(f"{arm}: {summary[arm]}", flush=True)
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
