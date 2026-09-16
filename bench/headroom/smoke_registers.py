"""Plumbing smoke test for the register-token adapter (Phase-0.5).

NOT a training/eval run — this only proves the register mechanism is wired
correctly on the real Anima DiT, before any training:

  1. attach an untrained adapter (arm B, zero-init LoRA) and confirm a full
     generate_body forward runs and decodes without shape errors (strip works);
  2. confirm the stripped latent matches the base latent's shape exactly;
  3. matched-seed diff: base image vs base+untrained-registers should be a
     MODEST perturbation (registers add attention mass; LoRA is a no-op at init),
     not garbage — a sanity floor that the plumbing didn't corrupt the forward;
  4. report the per-forward register-vs-patch residual norm readout.

Run: uv run python bench/headroom/smoke_registers.py --seeds 2 --size 1024
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anima_lora import GenerationRequest, load_vae  # noqa: E402
from bench._anima import add_common_args, add_model_args  # noqa: E402
from bench.headroom.register_adapter import RegisterAdapter  # noqa: E402
from library.inference.generation import generate_body  # noqa: E402
from library.inference.models import load_dit_model, load_text_encoder  # noqa: E402
from library.inference.output import decode_latent, pixels_to_pil  # noqa: E402
from library.inference.text import ensure_text_strategies, prepare_text_inputs  # noqa: E402
from library.runtime.device import clean_memory_on_device  # noqa: E402

_PROMPT = (
    "1girl, solo, long hair, looking at viewer, standing, simple background, "
    "detailed face, masterpiece, best quality"
)


def build_args():
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    add_model_args(p)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--infer_steps", type=int, default=20)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument("--num_registers", type=int, default=16)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--save_images", action="store_true")
    return p.parse_args()


def main():
    args = build_args()
    device = torch.device(args.device)

    req = GenerationRequest(
        prompt="placeholder",
        dit=args.dit,
        vae=args.vae,
        text_encoder=args.text_encoder,
        negative_prompt="",
        image_size=(args.size, args.size),
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg,
        flow_shift=args.flow_shift,
        attn_mode=args.attn_mode,
        device=args.device,
        seed=0,
    )
    gen_args = req.to_args()
    gen_args.compile_blocks = False  # eager: register wrap must not be traced away

    ensure_text_strategies(args.text_encoder)
    text_encoder = load_text_encoder(gen_args, dtype=torch.bfloat16, device=device)
    text_encoder.eval()
    shared = {"text_encoder": text_encoder, "conds_cache": {}}
    anima = load_dit_model(gen_args, device, torch.bfloat16)
    shared["model"] = anima

    vae = load_vae(args.vae, device="cpu", vae_2d=True)
    vae.to(torch.bfloat16).eval()

    ctx, ctx_null = prepare_text_inputs(gen_args, device, anima, shared, prompt=_PROMPT)

    ok = True
    for s in range(args.seeds):
        # --- base (no adapter) ---
        with torch.no_grad():
            base_latent = generate_body(gen_args, anima, ctx, ctx_null, device, seed=s)
        base_img = decode_latent(vae, base_latent, device)
        clean_memory_on_device(device)

        # --- attach untrained arm-B adapter ---
        adapter = RegisterAdapter(
            anima, num_registers=args.num_registers, arm="B", lora_rank=args.lora_rank
        ).to(device=device, dtype=torch.bfloat16)
        adapter.apply_to()
        try:
            with torch.no_grad():
                reg_latent = generate_body(
                    gen_args, anima, ctx, ctx_null, device, seed=s
                )
        finally:
            adapter.remove()
        reg_img = decode_latent(vae, reg_latent, device)
        clean_memory_on_device(device)

        # --- checks ---
        shape_ok = tuple(reg_latent.shape) == tuple(base_latent.shape)
        finite_ok = torch.isfinite(reg_latent).all().item()
        # matched-seed pixel L1 (normalized [-1,1] space) — expect a modest,
        # non-catastrophic change from untrained registers.
        l1 = (reg_img.float() - base_img.float()).abs().mean().item()
        trainable = adapter.num_trainable()

        print(
            f"[seed {s}] shape_ok={shape_ok} finite={finite_ok} "
            f"pixel_L1={l1:.4f}  reg_norm={adapter.last_reg_norm:.2f} "
            f"patch_norm={adapter.last_patch_norm:.2f}  trainable_params={trainable}"
        )
        if not (shape_ok and finite_ok):
            ok = False
        if not (0.0 < l1 < 1.0):
            print(f"  WARN: pixel_L1={l1:.4f} outside sane (0,1) — plumbing suspect")

        if args.save_images:
            out = (
                Path(__file__).resolve().parent
                / "results"
                / f"smoke_{args.label or 'reg'}"
            )
            out.mkdir(parents=True, exist_ok=True)
            pixels_to_pil(base_img).save(out / f"s{s}_base.png")
            pixels_to_pil(reg_img).save(out / f"s{s}_reg_untrained.png")

    print("SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
