#!/usr/bin/env python
"""Generate an image conditioned on a reference image via the img2emb resampler.

Pipeline:
  ref image → siglip2 vision tower → AnchoredResampler (phase-2a ckpt) →
  anchor injection at default (rating, count, artist) slots → DiT cross-attention →
  flow-matching euler denoise → VAE decode → png.

Replaces the text-encoder path entirely — no prompt, no negative prompt.

Usage:
  python scripts/img2emb/test_img2emb.py --ref_image path/to/ref.png
  python scripts/img2emb/test_img2emb.py --ref_image ref.png --resampler_ckpt other.safetensors
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file, save_file
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.img2emb.extract_features import ENCODERS, load_encoder  # noqa: E402
from scripts.img2emb.phase1_5_anchored import (  # noqa: E402
    COUNT_CLASSES,
    RATING_CLASSES,
    AnchoredResampler,
    _load_artist_prototypes,
    _load_prototypes,
    inject_anchors,
)
from library.anima import weights as anima_utils  # noqa: E402
from library.inference.output import save_images  # noqa: E402
from library.inference.sampling import get_timesteps_sigmas, step as euler_step  # noqa: E402
from library.log import setup_logging  # noqa: E402
from library.models.qwen_vae import load_vae  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


# Training-data modes from bench/inversionv2/results/tag_slot/phase1_positions.json:
# rating tags sit at slot 0 (always), 1girl/count at slot 2 (dominant), artist at slot 10.
# See phase2_proposal.md for the shuffle-prefix policy that makes these stable.
DEFAULT_SLOTS = {"rating": 0, "count": 2, "artist": 10}

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
    p.add_argument("--encoder", default="siglip2", choices=list(ENCODERS.keys()))
    p.add_argument("--d_model", type=int, default=1024)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_slots", type=int, default=512)

    p.add_argument("--rating_slot", type=int, default=DEFAULT_SLOTS["rating"])
    p.add_argument("--count_slot",  type=int, default=DEFAULT_SLOTS["count"])
    p.add_argument("--artist_slot", type=int, default=DEFAULT_SLOTS["artist"])
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
    p.add_argument(
        "--pooled_text_source",
        default="dataset_mean",
        choices=["dataset_mean", "resampler", "skip"],
        help=("dataset_mean = variant-mean pooled averaged over the training cache "
              "(closest to training distribution, default); resampler = ctx.max(dim=1); "
              "skip = bypass pooled_text_proj."),
    )
    p.add_argument("--image_dir", default="post_image_dataset",
                   help="Training image_dir (for dataset_mean pooled scan). Cached to mean_pooled.safetensors.")
    p.add_argument("--pooled_cache",
                   default=str(REPO_ROOT / "output" / "img2embs" / "mean_pooled.safetensors"),
                   help="Path to the cached dataset-mean pooled vector; computed on first use.")

    # output
    p.add_argument("--save_path", default="output/tests")
    p.add_argument("--no_metadata", action="store_true")
    p.add_argument("--output_type", default="images", choices=["images", "latent", "latent_images"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


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


def _load_resampler(args, device: torch.device) -> AnchoredResampler:
    tag_slot_dir = Path(args.tag_slot_dir)
    rating_protos, _ = _load_prototypes(
        tag_slot_dir, RATING_CLASSES, "phase2_class_prototypes.safetensors", key_prefix="rating=",
    )
    count_protos, _ = _load_prototypes(
        tag_slot_dir, COUNT_CLASSES, "phase2_class_prototypes.safetensors",
    )
    artist_protos, _ = _load_artist_prototypes(tag_slot_dir)

    # siglip2 large: D_enc = 1024, pooled same. (See ENCODERS in extract_features.py.)
    model = AnchoredResampler(
        d_enc=1024, d_pool=1024,
        rating_protos=rating_protos, count_protos=count_protos, artist_protos=artist_protos,
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
    tokens: torch.Tensor,
    pooled: torch.Tensor,
    args,
    device: torch.device,
) -> torch.Tensor:
    """Resampler + anchor injection at default slots. Returns ctx[1, S, 1024] bf16."""
    with torch.no_grad():
        fwd = model(tokens, pooled)
    pred = fwd["pred"]
    r_logits, c_logits, a_logits = fwd["logits"]
    r_emb, c_emb, a_emb = fwd["anchor_emb"]

    def _pred_name(logits, names):
        idx = int(logits.argmax(dim=-1).item())
        name = names[idx] if idx < len(names) else f"<idx={idx}>"
        return idx, name

    r_idx, r_name = _pred_name(r_logits[0], RATING_CLASSES)
    c_idx, c_name = _pred_name(c_logits[0], COUNT_CLASSES)
    logger.info(
        f"classifier predictions: rating={r_name} ({r_idx})  count={c_name} ({c_idx})  "
        f"artist_idx={int(a_logits.argmax(dim=-1).item())}"
    )

    if not args.skip_anchors:
        r_slot = torch.tensor([args.rating_slot], device=device, dtype=torch.long)
        c_slot = torch.tensor([args.count_slot],  device=device, dtype=torch.long)
        a_slot = torch.tensor([args.artist_slot], device=device, dtype=torch.long)
        inject_anchors(pred, r_emb, r_slot, mode="replace")
        inject_anchors(pred, c_emb, c_slot, mode="replace")
        inject_anchors(pred, a_emb, a_slot, mode="replace")

    return pred.to(torch.bfloat16)


def _compute_or_load_mean_pooled(image_dir: Path, cache_path: Path) -> torch.Tensor:
    """Variant-mean-then-dataset-mean pooled over cached *_anima_te.safetensors.

    Matches phase2_flow.py's `t5_pooled` computation: per-image = mean over variants
    of crossattn_emb[v].amax(dim=0); then average across images. Result is (1024,) f32.
    """
    if cache_path.exists():
        sd = load_file(str(cache_path))
        if "mean_pooled" in sd:
            logger.info(f"dataset_mean pooled: cached at {cache_path}")
            return sd["mean_pooled"].float()

    te_paths = sorted(image_dir.glob("*_anima_te.safetensors"))
    if not te_paths:
        raise FileNotFoundError(
            f"No *_anima_te.safetensors under {image_dir}. "
            f"Run preprocessing first, or pass --pooled_text_source resampler."
        )
    logger.info(f"computing dataset_mean pooled over {len(te_paths)} TE caches")

    running = None
    n = 0
    for p in tqdm(te_paths, desc="scan TE"):
        sd = load_file(str(p))
        vkeys = sorted(k for k in sd if k.startswith("crossattn_emb_v"))
        if not vkeys and "crossattn_emb" in sd:
            vkeys = ["crossattn_emb"]
        if not vkeys:
            continue
        stack = torch.stack([sd[k].float().amax(dim=0) for k in vkeys], dim=0)
        per_image = stack.mean(dim=0)  # (1024,)
        running = per_image if running is None else running + per_image
        n += 1
    mean_pooled = (running / n).contiguous()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"mean_pooled": mean_pooled}, str(cache_path))
    logger.info(f"saved dataset_mean pooled → {cache_path}  (n={n}, ||={mean_pooled.norm():.3f})")
    return mean_pooled


def _build_pooled_text(
    ctx: torch.Tensor, args, device: torch.device
) -> torch.Tensor | None:
    """Return the positive-side pooled_text_override (or None for skip)."""
    if args.pooled_text_source == "skip":
        return None
    if args.pooled_text_source == "resampler":
        return ctx.float().amax(dim=1).to(torch.bfloat16)
    # dataset_mean
    mean = _compute_or_load_mean_pooled(Path(args.image_dir), Path(args.pooled_cache))
    return mean.unsqueeze(0).to(device=device, dtype=torch.bfloat16)


@torch.no_grad()
def _denoise(
    anima, ctx: torch.Tensor, pooled_text: torch.Tensor | None, args, device: torch.device
) -> torch.Tensor:
    """Flow-matching euler denoise with optional CFG. Returns (1, C, 1, H_lat, W_lat) bf16.

    CFG unconditional: zero-ctx, same pooled_text policy. Matches the cache-time
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

    skip = args.pooled_text_source == "skip"
    kwargs = {"padding_mask": padding_mask}
    if skip:
        kwargs["skip_pooled_text_proj"] = True
    elif pooled_text is not None:
        kwargs["pooled_text_override"] = pooled_text

    do_cfg = args.guidance_scale != 1.0
    uncond_ctx = torch.zeros_like(ctx) if do_cfg else None

    logger.info(
        f"denoising: {args.infer_steps} steps, {H}x{W}, flow_shift={args.flow_shift}, "
        f"cfg={args.guidance_scale}, pooled={args.pooled_text_source}"
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

    # 1) ref image → siglip2 features (loaded → used → freed).
    logger.info(f"encoding ref image: {args.ref_image}")
    tokens, pooled = _encode_ref_image(args.ref_image, args.encoder, device)
    logger.info(f"siglip2 tokens={tuple(tokens.shape)}  pooled={tuple(pooled.shape)}")

    # 2) Resampler → ctx.
    logger.info(f"loading resampler ckpt: {args.resampler_ckpt}")
    resampler = _load_resampler(args, device)
    ctx = _build_ctx(resampler, tokens, pooled, args, device)
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

    # 4) Pooled text + denoise.
    pooled_text = _build_pooled_text(ctx, args, device)
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
