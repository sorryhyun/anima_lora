#!/usr/bin/env python
"""Generate an image conditioned on a reference image via the img2emb resampler.

Pipeline:
  ref image → siglip2 vision tower → AnchoredResampler (phase-2a ckpt) →
  per-group anchor injection at default slots → DiT cross-attention →
  flow-matching euler denoise → VAE decode → png.

Replaces the text-encoder path entirely — no prompt, no negative prompt.

Usage:
  python scripts/img2emb/test_img2emb.py --ref_image path/to/ref.png
  python scripts/img2emb/test_img2emb.py --ref_image ref.png --resampler_ckpt other.safetensors
  python scripts/img2emb/test_img2emb.py --ref_image ref.png --slot_override rating=0,girl_count=2
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.img2emb.anchors import (  # noqa: E402
    AnchorSpec,
    AnchoredResampler,
    inject_anchors,
    load_anchor_spec,
)
from scripts.img2emb.extract_features import ENCODERS, load_encoder  # noqa: E402
from library.anima import weights as anima_utils  # noqa: E402
from library.inference.output import save_images  # noqa: E402
from library.inference.sampling import get_timesteps_sigmas, step as euler_step  # noqa: E402
from library.log import setup_logging  # noqa: E402
from library.models.qwen_vae import load_vae  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


DEFAULT_ANCHORS_YAML = Path(__file__).parent / "anchors.yaml"

# Matches the phase 1.5 save naming.
DEFAULT_RESAMPLER_CKPT = REPO_ROOT / "bench" / "img2emb" / "results" / "phase2a" / "siglip2_phase2a_warm.safetensors"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--ref_image", required=True, help="Path to the reference PNG/JPG.")
    p.add_argument(
        "--resampler_ckpt",
        default=str(DEFAULT_RESAMPLER_CKPT),
        help="Path to the img2emb resampler ckpt (phase-2a warm).",
    )
    p.add_argument(
        "--tag_slot_dir",
        default=str(REPO_ROOT / "bench" / "inversionv2" / "results" / "tag_slot"),
        help="Directory with class prototype tables (needed to rebuild AnchoredResampler).",
    )
    p.add_argument(
        "--anchors_yaml",
        default=str(DEFAULT_ANCHORS_YAML),
        help="YAML spec listing anchor groups + classes.",
    )
    p.add_argument("--encoder", default="siglip2", choices=list(ENCODERS.keys()))
    p.add_argument("--d_model", type=int, default=1024)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_slots", type=int, default=512)

    p.add_argument(
        "--slot_override",
        default="",
        help="Per-group slot override, e.g. 'rating=0,girl_count=2'. "
             "Unspecified groups use default_slot from anchors.yaml.",
    )
    p.add_argument(
        "--skip_anchors",
        action="store_true",
        help="Skip anchor injection (debug: use raw resampler output).",
    )

    # DiT / VAE
    p.add_argument("--dit", default="models/diffusion_models/anima-preview3-base.safetensors")
    p.add_argument("--vae", default="models/vae/qwen_image_vae.safetensors")
    p.add_argument("--attn_mode", default="flash")
    p.add_argument("--blocks_to_swap", type=int, default=0)

    # sampling
    p.add_argument("--image_size", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W"))
    p.add_argument("--infer_steps", type=int, default=30)
    p.add_argument("--flow_shift", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance_scale", type=float, default=4.0,
                   help="CFG scale; 1.0 disables CFG. Unconditional uses zero-ctx.")

    # output
    p.add_argument("--save_path", default="output/tests")
    p.add_argument("--no_metadata", action="store_true")
    p.add_argument("--output_type", default="images", choices=["images", "latent", "latent_images"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _parse_slot_override(raw: str) -> dict[str, int]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    out: dict[str, int] = {}
    for chunk in raw.split(","):
        if not chunk.strip():
            continue
        k, v = chunk.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out


def _encode_ref_image(
    path: str, encoder_name: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (tokens[1, T, D_enc] bf16, pooled[1, D_enc] f32)."""
    model, processor = load_encoder(encoder_name, device)
    img = Image.open(path).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt")["pixel_values"].to(device, dtype=torch.bfloat16)
    with torch.no_grad():
        out = model(pixel_values=pixel_values)
    tokens = out.last_hidden_state.to(torch.bfloat16)                    # (1, T, D_enc)
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        pooled = out.pooler_output.float()                                # (1, D_enc)
    else:
        pooled = out.last_hidden_state.float().mean(dim=1)
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return tokens, pooled


def _load_resampler(args, spec: AnchorSpec, device: torch.device) -> AnchoredResampler:
    # siglip2 large: D_enc = 1024, pooled same. (See ENCODERS in extract_features.py.)
    model = AnchoredResampler(
        spec=spec, d_enc=1024, d_pool=1024,
        d_model=args.d_model, n_heads=args.n_heads, n_slots=args.n_slots, n_layers=args.n_layers,
    ).to(device)

    sd = load_file(args.resampler_ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning(f"resampler missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        logger.warning(f"resampler unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    model.eval()
    return model


def _build_ctx(
    model: AnchoredResampler,
    spec: AnchorSpec,
    tokens: torch.Tensor,
    pooled: torch.Tensor,
    args,
    device: torch.device,
) -> torch.Tensor:
    """Resampler + per-group anchor injection at default slots. Returns ctx[1, S, 1024] bf16."""
    with torch.no_grad():
        fwd = model(tokens, pooled)
    pred = fwd["pred"]
    slot_overrides = _parse_slot_override(args.slot_override)

    # Log top-1 class per group.
    for g in spec.groups:
        lg = fwd["logits"][g.name][0]
        if g.mutex:
            idx = int(lg.argmax(dim=-1).item())
            name = g.classes[idx] if idx < g.n_classes else "<unknown>"
            logger.info(f"  [{g.name}] top1={name} (idx={idx})")
        else:
            probs = torch.sigmoid(lg[: g.n_classes])
            pos = [(g.classes[i], float(probs[i].item())) for i in range(g.n_classes) if probs[i].item() > 0.5]
            logger.info(f"  [{g.name}] positives={pos}")

    if args.skip_anchors:
        return pred.to(torch.bfloat16)

    B = pred.shape[0]
    for g in spec.groups:
        anchor_emb = fwd["anchor_emb"][g.name]
        if g.mutex:
            slot = slot_overrides.get(g.name, g.default_slot)
            slots = torch.full((B,), int(slot), dtype=torch.long, device=device)
            inject_anchors(pred, anchor_emb, slots, mutex=True, mode="replace")
        else:
            # Multi-label: use per-class default_slots; mask by sigmoid > 0.5.
            base_slot = slot_overrides.get(g.name)
            if base_slot is None:
                per_class = torch.tensor(g.default_slots, dtype=torch.long, device=device)
            else:
                per_class = torch.full((g.n_classes,), int(base_slot), dtype=torch.long, device=device)
            slots = per_class.unsqueeze(0).expand(B, -1).contiguous()
            logits = fwd["logits"][g.name][..., : g.n_classes]
            mask = (torch.sigmoid(logits) > 0.5)
            inject_anchors(pred, anchor_emb, slots, mutex=False, mode="replace", mask=mask)

    return pred.to(torch.bfloat16)


@torch.no_grad()
def _denoise(
    anima, ctx: torch.Tensor, pooled_text: torch.Tensor, args, device: torch.device
) -> torch.Tensor:
    """Flow-matching euler denoise with optional CFG. Returns (1, C, 1, H_lat, W_lat) bf16.

    CFG unconditional: zero-ctx, same pooled_text. Matches the cache-time
    zero-pad convention (crossattn_emb_v0 padding is zeros).
    """
    H, W = args.image_size
    h_lat, w_lat = H // 8, W // 8
    C = 16  # Anima.LATENT_CHANNELS

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    latents = torch.randn((1, C, 1, h_lat, w_lat), generator=g, dtype=torch.bfloat16).to(device)

    timesteps, sigmas = get_timesteps_sigmas(args.infer_steps, args.flow_shift, device)
    timesteps = (timesteps / 1000).to(dtype=torch.bfloat16)

    padding_mask = torch.zeros(1, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)
    kwargs = {"padding_mask": padding_mask, "pooled_text_override": pooled_text}

    do_cfg = args.guidance_scale != 1.0
    uncond_ctx = torch.zeros_like(ctx) if do_cfg else None

    logger.info(
        f"denoising: {args.infer_steps} steps, {H}x{W}, flow_shift={args.flow_shift}, "
        f"cfg={args.guidance_scale}"
    )
    for i, t in enumerate(timesteps):
        t_expand = t.expand(1)
        noise_pred = anima(latents, t_expand, ctx, **kwargs)
        if do_cfg:
            uncond_pred = anima(latents, t_expand, uncond_ctx, **kwargs)
            noise_pred = uncond_pred + args.guidance_scale * (noise_pred - uncond_pred)
        latents = euler_step(latents, noise_pred, sigmas, i).to(torch.bfloat16)
    return latents


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.save_path, exist_ok=True)

    # 0) Anchor spec.
    spec = load_anchor_spec(Path(args.anchors_yaml), Path(args.tag_slot_dir))

    # 1) ref image → siglip2 features (loaded → used → freed).
    logger.info(f"encoding ref image: {args.ref_image}")
    tokens, pooled = _encode_ref_image(args.ref_image, args.encoder, device)
    logger.info(f"siglip2 tokens={tuple(tokens.shape)}  pooled={tuple(pooled.shape)}")

    # 2) Resampler → ctx.
    logger.info(f"loading resampler ckpt: {args.resampler_ckpt}")
    resampler = _load_resampler(args, spec, device)
    ctx = _build_ctx(resampler, spec, tokens, pooled, args, device)
    logger.info(f"ctx shape: {tuple(ctx.shape)}")
    del resampler
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 3) DiT.
    logger.info(f"loading DiT: {args.dit}")
    is_swapping = args.blocks_to_swap > 0
    anima = anima_utils.load_anima_model(
        device="cpu" if is_swapping else device,
        dit_path=args.dit,
        attn_mode=args.attn_mode,
        split_attn=True,
        loading_device="cpu" if is_swapping else device,
        dit_weight_dtype=torch.bfloat16,
    )
    anima.to(torch.bfloat16)
    anima.requires_grad_(False)
    anima.eval()
    anima.split_attn = False
    if is_swapping:
        anima.enable_block_swap(args.blocks_to_swap, device)
        anima.move_to_device_except_swap_blocks(device)
        anima.prepare_block_swap_before_forward()
    else:
        anima.to(device)

    # 4) Pooled text + denoise. Pooled = resampler ctx.amax(dim=1) — matches training.
    pooled_text = ctx.float().amax(dim=1).to(torch.bfloat16)
    latents = _denoise(anima, ctx, pooled_text, args, device)
    del anima
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 5) VAE decode + save.
    logger.info(f"loading VAE: {args.vae}")
    vae = load_vae(args.vae, device=device, disable_cache=True)
    with torch.no_grad():
        pixels = vae.decode_to_pixels(latents.to(device, dtype=vae.dtype))
    if pixels.ndim == 5:
        pixels = pixels.squeeze(2)
    pixels = pixels[0].to("cpu", dtype=torch.float32)

    # Reuse save_images — expects args.seed, args.save_path; pretend prompt is the ref path.
    args.prompt = f"[img2emb:{Path(args.ref_image).name}]"
    args.negative_prompt = None
    out = save_images(pixels, args, original_base_name=Path(args.ref_image).stem)
    logger.info(f"saved: {out}.png")

    # 6) Drop the ref image alongside the generated image for easy side-by-side.
    ref_copy = Path(f"{out}_ref{Path(args.ref_image).suffix}")
    shutil.copyfile(args.ref_image, ref_copy)
    logger.info(f"ref image copied to: {ref_copy}")


if __name__ == "__main__":
    main()
