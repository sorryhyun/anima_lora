"""Render server for the preference-RL pick rounds: one daemon job that keeps the policy loaded.

Loads the DiT (int8 by default, blocks compiled on a dynamic token axis), the text encoder and
the VAE once, then renders every request the GUI drops into ``<session>/requests/`` until the
GUI writes ``<session>/stop`` or nothing arrives for ``--idle_exit`` seconds. Blind rounds swap
the adapters between arm A and arm B.

    make daemon-run ARGS="--stall-timeout 900 project/anima_turbo/pref_rl/server.py --session output/turbo_pref/<name>"

The GUI submits exactly this job; the CLI form is for a server started by hand.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from store import Session, _write_json  # noqa: E402

HEARTBEAT_PRINT_S = 120.0


def _status(session: Session, state: str, **extra) -> None:
    _write_json(
        session.server_path,
        {"state": state, "pid": os.getpid(), "t": time.time(), **extra},
    )


def build_args(cfg, loras: list[str]):
    from library.inference import GenerationRequest

    req = GenerationRequest(
        prompt="placeholder",
        dit=cfg.dit,
        text_encoder="models/text_encoders/qwen_3_06b_base.safetensors",
        vae="models/vae/qwen_image_vae.safetensors",
        vae_chunk_size=64,
        vae_disable_cache=True,
        attn_mode="flash",
        infer_steps=cfg.infer_steps,
        guidance_scale=cfg.guidance_scale,
        flow_shift=cfg.flow_shift,
        sampler=cfg.sampler,
        lora_weight=list(loras),
        lora_multiplier=[1.0] * len(loras),
        save_path="unused",
        extra_argv=(["--int8"] if cfg.int8 else [])
        + (["--compile_dynamic_seq"] if cfg.compile else []),
    )
    return req.to_args()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument(
        "--idle_exit",
        type=float,
        default=1200.0,
        help="seconds with no request before exit",
    )
    ap.add_argument("--poll", type=float, default=0.5)
    opts = ap.parse_args()

    session = Session(Path(opts.session).resolve())
    cfg = session.config()
    session.stop_path.unlink(missing_ok=True)
    _status(session, "loading")
    print(f"[pref-server] session {session.root}", flush=True)

    import torch
    from diffusers.utils.torch_utils import randn_tensor

    from library.anima import models as anima_models
    from library.anima.stream import latent_dtype
    from library.inference import (
        generate_body,
        get_generation_settings,
        load_dit_model,
        load_text_encoder,
        prepare_text_inputs,
        swap_adapters,
    )
    from library.inference.output import decode_latent, pixels_to_pil
    from library.models import qwen_vae

    arm_loras = {"A": list(cfg.loras), "B": list(cfg.blind_loras)}
    args = build_args(cfg, arm_loras["A"])
    gen_settings = get_generation_settings(args)
    device = gen_settings.device

    t0 = time.time()
    anima = load_dit_model(args, device, torch.bfloat16)
    attached = "A"
    te = load_text_encoder(args, dtype=torch.bfloat16, device=torch.device("cpu")).to(
        device
    )
    shared = {"text_encoder": te, "conds_cache": {}}
    vae = qwen_vae.load_vae(
        args.vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=args.vae_chunk_size,
        disable_cache=args.vae_disable_cache,
        vae_2d=args.vae_2d,
    )
    vae.to(device, dtype=torch.bfloat16).eval()
    print(f"[pref-server] loaded in {time.time() - t0:.0f}s", flush=True)

    policy = {
        "dit": cfg.dit,
        "A": arm_loras["A"],
        "B": arm_loras["B"],
        "int8": cfg.int8,
        "steps": cfg.infer_steps,
        "cfg": cfg.guidance_scale,
        "shift": cfg.flow_shift,
        "sampler": cfg.sampler,
    }
    last_work = time.time()
    last_print = 0.0
    n_done = 0
    _status(session, "ready", rounds=n_done)

    while True:
        if session.stop_path.exists():
            print("[pref-server] stop requested", flush=True)
            break
        pending = session.pending_rids()
        now = time.time()
        if not pending:
            if now - last_work > opts.idle_exit:
                print(f"[pref-server] idle {opts.idle_exit:.0f}s, exiting", flush=True)
                break
            if now - last_print > HEARTBEAT_PRINT_S:
                print(f"[pref-server] idle, {n_done} round(s) rendered", flush=True)
                last_print = now
            _status(session, "ready", rounds=n_done)
            time.sleep(opts.poll)
            continue

        rid = pending[0]
        req = session.request(rid)
        if req is None:  # withdrawn by the GUI between the scan and the read
            continue
        out = session.round_dir(rid)
        out.mkdir(parents=True, exist_ok=True)
        _status(session, "rendering", rid=rid, rounds=n_done)
        t_round = time.time()
        h, w = req["size"]
        pa = copy.copy(args)
        pa.prompt = req["caption"]
        pa.image_size = [h, w]
        with torch.no_grad():
            context, context_null = prepare_text_inputs(pa, device, anima, shared)
            slots = list(range(len(req["seeds"])))
            by_arm: dict[str, list[int]] = {}
            for i in slots:
                by_arm.setdefault(req["arms"][i], []).append(i)
            for arm in sorted(by_arm):
                if arm != attached:
                    if not arm_loras.get(arm):
                        raise SystemExit(
                            f"round {rid} asks for arm {arm}, which has no LoRA"
                        )
                    swap_adapters(anima, pa, device, arm_loras[arm])
                    attached = arm
                idxs = by_arm[arm]
                for b0 in range(0, len(idxs), cfg.batch):
                    chunk = idxs[b0 : b0 + cfg.batch]
                    seeds = [req["seeds"][i] for i in chunk]
                    noise = []
                    for s in seeds:
                        g = torch.Generator(device=device)
                        g.manual_seed(s)
                        noise.append(
                            randn_tensor(
                                (
                                    1,
                                    anima_models.Anima.LATENT_CHANNELS,
                                    1,
                                    h // 8,
                                    w // 8,
                                ),
                                generator=g,
                                device=device,
                                dtype=latent_dtype(anima),
                            )
                        )
                    pa.seed = seeds[0]
                    lat = generate_body(
                        pa,
                        anima,
                        context,
                        context_null,
                        device,
                        seeds,
                        latents=torch.cat(noise, 0),
                    )
                    for j, i in enumerate(chunk):
                        img = pixels_to_pil(decode_latent(vae, lat[j : j + 1], device))
                        tmp = out / f".{i:02d}.png"
                        img.save(tmp)
                        os.replace(tmp, out / f"{i:02d}.png")
        if attached != "A":
            swap_adapters(anima, pa, device, arm_loras["A"])
            attached = "A"
        dt = time.time() - t_round
        _write_json(
            out / "done.json", {"render_s": dt, "policy": policy, "t": time.time()}
        )
        n_done += 1
        last_work = last_print = time.time()
        print(
            f"[pref-server] round {rid}: {len(req['seeds'])} images in {dt:.1f}s",
            flush=True,
        )

    _status(session, "stopped", rounds=n_done)


if __name__ == "__main__":
    main()
