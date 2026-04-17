"""Interpret inverted embeddings: generate verification images and find nearest text captions."""

import argparse
import glob
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm import tqdm

from library import (
    anima_models,
    anima_utils,
)
from library.models import qwen_vae as qwen_image_autoencoder_kl
from library.io.cache import TE_CACHE_SUFFIX, load_cached_crossattn_emb
from library.runtime.device import clean_memory_on_device
from library.inference import sampling as inference_utils
from library.utils import setup_logging

setup_logging()

logger = logging.getLogger(__name__)


def find_nearest_captions(inverted_path, dataset_dir, top_k=10):
    """Compare inverted embedding against all cached TE embeddings in the dataset."""
    inv_sd = load_file(inverted_path)
    inv_emb = inv_sd["crossattn_emb"].float()  # (512, 1024)

    results = []
    te_files = sorted(glob.glob(os.path.join(dataset_dir, f"*{TE_CACHE_SUFFIX}")))

    for te_path in tqdm(te_files, desc="Comparing embeddings"):
        stem = os.path.basename(te_path).removesuffix(TE_CACHE_SUFFIX)
        caption_path = os.path.join(dataset_dir, f"{stem}.txt")
        if not os.path.exists(caption_path):
            continue

        ref_emb = load_cached_crossattn_emb(te_path)
        if ref_emb is None:
            continue

        # Cosine similarity on flattened embeddings
        cos_sim = F.cosine_similarity(
            inv_emb.flatten().unsqueeze(0), ref_emb.flatten().unsqueeze(0)
        ).item()

        # Also compute per-token cosine sim and average (captures token-level structure)
        per_token_sim = F.cosine_similarity(inv_emb, ref_emb, dim=-1).mean().item()

        with open(caption_path) as f:
            caption = f.read().strip()

        results.append(
            {
                "stem": stem,
                "cos_sim": cos_sim,
                "token_sim": per_token_sim,
                "caption": caption,
            }
        )

    results.sort(key=lambda x: x["cos_sim"], reverse=True)

    return results[:top_k]


def generate_from_embedding(
    inverted_path,
    dit_path,
    vae_path,
    device,
    h=1024,
    w=1024,
    steps=50,
    flow_shift=5.0,
    seed=42,
    vae_chunk_size=64,
    blocks_to_swap=0,
    attn_mode="flash",
):
    """Generate an image from an inverted embedding."""
    inv_sd = load_file(inverted_path)
    embed = inv_sd["crossattn_emb"].unsqueeze(0).to(device, dtype=torch.bfloat16)

    # Try to recover original image size from metadata
    try:
        from safetensors import safe_open

        with safe_open(inverted_path, framework="pt") as f:
            metadata = f.metadata()
        if metadata and "image_size" in metadata:
            h_str, w_str = metadata["image_size"].split("x")
            h, w = int(h_str), int(w_str)
            logger.info(f"Using image size from metadata: {h}x{w}")
    except Exception:
        pass

    logger.info("Loading DiT...")
    is_swapping = blocks_to_swap > 0
    anima = anima_utils.load_anima_model(
        device="cpu" if is_swapping else device,
        dit_path=dit_path,
        attn_mode=attn_mode,
        split_attn=True,
        loading_device="cpu" if is_swapping else device,
        dit_weight_dtype=torch.bfloat16,
    )
    anima.to(torch.bfloat16)
    anima.requires_grad_(False)

    if is_swapping:
        anima.enable_block_swap(blocks_to_swap, device)
        anima.move_to_device_except_swap_blocks(device)
        anima.prepare_block_swap_before_forward()
    else:
        anima.to(device)

    logger.info("Loading VAE...")
    vae = qwen_image_autoencoder_kl.load_vae(
        vae_path, device="cpu", disable_mmap=True, spatial_chunk_size=vae_chunk_size
    )
    vae.to(device, dtype=torch.bfloat16)
    vae.eval()

    h_lat, w_lat = h // 8, w // 8
    padding_mask = torch.zeros(1, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)

    gen = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        1,
        anima_models.Anima.LATENT_CHANNELS,
        1,
        h_lat,
        w_lat,
        device=device,
        dtype=torch.bfloat16,
        generator=gen,
    )

    timesteps, sigmas = inference_utils.get_timesteps_sigmas(steps, flow_shift, device)
    timesteps = (timesteps / 1000).to(device, dtype=torch.bfloat16)

    if hasattr(anima, "switch_block_swap_for_inference"):
        anima.switch_block_swap_for_inference()

    logger.info(f"Denoising {steps} steps at {h}x{w}...")
    with torch.no_grad():
        for step_i, t in enumerate(tqdm(timesteps, desc="Denoising", leave=False)):
            if hasattr(anima, "prepare_block_swap_before_forward"):
                anima.prepare_block_swap_before_forward()
            noise_pred = anima(
                latents, t.unsqueeze(0), embed, padding_mask=padding_mask
            )
            latents = inference_utils.step(latents, noise_pred, sigmas, step_i).to(
                torch.bfloat16
            )

    logger.info("Decoding latents...")
    with torch.no_grad():
        from PIL import Image

        pixels = vae.decode_to_pixels(latents.squeeze(2))
    pixels = (
        ((pixels + 1.0) / 2.0)
        .clamp(0, 1)
        .squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .float()
        .numpy()
    )
    pixels = (pixels * 255).clip(0, 255).astype("uint8")

    del anima, vae
    clean_memory_on_device(device)

    return Image.fromarray(pixels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inversions_dir", type=str, default="inversions/results")
    p.add_argument("--dataset_dir", type=str, default="post_image_dataset")
    p.add_argument(
        "--dit",
        type=str,
        default="models/diffusion_models/anima-preview3-base.safetensors",
    )
    p.add_argument("--vae", type=str, default="models/vae/qwen_image_vae.safetensors")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--verify", action="store_true", help="Generate verification images")
    p.add_argument("--verify_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--flow_shift", type=float, default=5.0)
    p.add_argument("--attn_mode", type=str, default="flash")
    p.add_argument("--blocks_to_swap", type=int, default=0)
    p.add_argument(
        "--name",
        type=str,
        default=None,
        help="Process only this stem (e.g. '10042360'). 'latest' picks the most recent file.",
    )
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    all_inv_files = sorted(
        glob.glob(os.path.join(args.inversions_dir, "*_inverted.safetensors"))
    )
    if not all_inv_files:
        logger.error(f"No inverted embeddings found in {args.inversions_dir}")
        return

    if args.name == "latest":
        inv_files = [max(all_inv_files, key=os.path.getmtime)]
    elif args.name:
        target = os.path.join(args.inversions_dir, f"{args.name}_inverted.safetensors")
        if not os.path.exists(target):
            logger.error(f"Not found: {target}")
            return
        inv_files = [target]
    else:
        inv_files = all_inv_files

    for inv_path in inv_files:
        stem = os.path.basename(inv_path).replace("_inverted.safetensors", "")
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Interpreting: {stem}")
        logger.info(f"{'=' * 60}")

        # Symlink source image for easy comparison
        source_path = os.path.join(args.dataset_dir, f"{stem}.png")
        link_path = os.path.join(args.inversions_dir, f"{stem}_source.png")
        if os.path.exists(source_path):
            if os.path.islink(link_path):
                os.remove(link_path)
            if not os.path.exists(link_path):
                os.symlink(os.path.relpath(source_path, args.inversions_dir), link_path)
                logger.info(f"Linked source: {link_path}")

        # Original caption
        caption_path = os.path.join(args.dataset_dir, f"{stem}.txt")
        if os.path.exists(caption_path):
            with open(caption_path) as f:
                logger.info(f"Original caption: {f.read().strip()}")

        # Find nearest text embeddings
        logger.info(f"\nTop-{args.top_k} nearest captions by embedding similarity:")
        nearest = find_nearest_captions(inv_path, args.dataset_dir, top_k=args.top_k)
        for i, r in enumerate(nearest):
            is_self = " ← SELF" if r["stem"] == stem else ""
            logger.info(
                f"  {i + 1}. [{r['cos_sim']:.4f} / tok:{r['token_sim']:.4f}] {r['stem']}{is_self}"
            )
            logger.info(f"     {r['caption'][:120]}")

        # Generate verification image
        if args.verify:
            # Read source image dimensions directly (metadata may have legacy WxH order)
            kwargs = {}
            if os.path.exists(source_path):
                from PIL import Image

                with Image.open(source_path) as src_img:
                    sw, sh = src_img.size
                kwargs = {"h": sh, "w": sw}
            img = generate_from_embedding(
                inv_path,
                args.dit,
                args.vae,
                device,
                steps=args.verify_steps,
                seed=args.seed,
                flow_shift=args.flow_shift,
                attn_mode=args.attn_mode,
                blocks_to_swap=args.blocks_to_swap,
                **kwargs,
            )
            out_path = os.path.join(args.inversions_dir, f"{stem}_verify.png")
            img.save(out_path)
            logger.info(f"Saved verification: {out_path}")


if __name__ == "__main__":
    main()
