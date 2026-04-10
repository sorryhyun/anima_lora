"""Embedding inversion: find the optimal text embedding for a target image.

Given a target image (or a directory of preprocessed images), optimizes a text
embedding (crossattn_emb) in the post-T5, pre-DiT space to minimize flow-matching
loss through the frozen DiT. This reveals "how the DiT interprets the image" in
embedding space.

Usage:
    # Single image (encodes via VAE on the fly)
    python scripts/invert_embedding.py \
        --image path/to/image.png \
        --dit models/diffusion_models/anima-preview3-base.safetensors \
        --vae models/vae/qwen_image_vae.safetensors \
        --save_path output/inverted.safetensors

    # Batch from post_image_dataset/ (uses cached latents + cached TE for init)
    python scripts/invert_embedding.py \
        --image_dir post_image_dataset \
        --dit models/diffusion_models/anima-preview3-base.safetensors \
        --save_path output/inversions/ \
        --num_images 10 --steps 500
"""

import argparse
import csv
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file, save_file
from torchvision import transforms
from tqdm import tqdm

from library import anima_models, anima_utils, qwen_image_autoencoder_kl, strategy_anima, strategy_base
from library.device_utils import clean_memory_on_device
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def parse_args():
    p = argparse.ArgumentParser(description="Embedding inversion for Anima DiT")

    # Model paths
    p.add_argument("--dit", type=str, required=True, help="DiT checkpoint path")
    p.add_argument("--vae", type=str, default=None, help="VAE checkpoint path (only needed for --image mode)")
    p.add_argument("--text_encoder", type=str, default=None, help="Text encoder path (for --init_prompt)")
    p.add_argument("--attn_mode", type=str, default="flash", help="Attention backend")

    # Target — single image or directory
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--image", type=str, default=None, help="Single target image path")
    target.add_argument("--image_dir", type=str, default=None, help="Directory with preprocessed images (cached latents + TE outputs)")
    p.add_argument("--image_size", type=int, nargs=2, default=None, help="Resize to H W (--image mode only)")
    p.add_argument("--num_images", type=int, default=None, help="Process N random images from --image_dir (default: all)")
    p.add_argument("--shuffle", action="store_true", help="Shuffle image order in --image_dir mode")

    # Initialization
    p.add_argument("--init_prompt", type=str, default=None, help="Initialize embedding from this prompt (requires --text_encoder)")
    p.add_argument("--init_embedding", type=str, default=None, help="Initialize from a saved embedding .safetensors")
    p.add_argument("--init_from_cache", action="store_true", default=True, help="Initialize from cached crossattn_emb in _anima_te.safetensors (default: True, --image_dir mode)")
    p.add_argument("--init_zeros", action="store_true", help="Initialize from zeros instead of cached embedding")

    # Optimization
    p.add_argument("--steps", type=int, default=100, help="Optimization steps per image")
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    p.add_argument("--lr_schedule", type=str, default="cosine", choices=["cosine", "constant"], help="LR schedule")
    p.add_argument("--timesteps_per_step", type=int, default=1, help="Random timesteps sampled per optimization step (batched forward)")
    p.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps (total timesteps per update = timesteps_per_step × grad_accum)")
    p.add_argument("--sigma_sampling", type=str, default="uniform", choices=["uniform", "sigmoid"], help="Sigma sampling strategy")
    p.add_argument("--sigmoid_scale", type=float, default=1.0, help="Scale for sigmoid sigma sampling")

    # Output
    p.add_argument("--output_dir", type=str, default="inversions", help="Output directory (results/ and logs/ created inside)")
    p.add_argument("--log_every", type=int, default=10, help="Log loss every N steps")
    p.add_argument("--log_block_grads", action="store_true", help="Log per-block gradient norms (for analyzing block sensitivity to embedding)")

    # Verification
    p.add_argument("--verify", action="store_true", help="Generate an image from the inverted embedding after optimization")
    p.add_argument("--verify_steps", type=int, default=50, help="Inference steps for verification")
    p.add_argument("--verify_seed", type=int, default=42, help="Seed for verification generation")
    p.add_argument("--flow_shift", type=float, default=5.0, help="Flow shift for verification sampling")

    # Device / VRAM
    p.add_argument("--device", type=str, default=None, help="Device (default: cuda if available)")
    p.add_argument("--vae_chunk_size", type=int, default=64, help="VAE spatial chunk size")
    p.add_argument("--blocks_to_swap", type=int, default=0, help="Number of transformer blocks to swap to CPU (0 = use gradient checkpointing instead)")

    args = p.parse_args()

    if args.image is not None and args.vae is None:
        p.error("--vae is required when using --image mode")

    # Setup output subdirectories
    args.results_dir = os.path.join(args.output_dir, "results")
    args.logs_dir = os.path.join(args.output_dir, "logs")

    return args


# region Data loading


def discover_images(image_dir):
    """Find all images in a preprocessed dataset directory that have cached latents."""
    images = []
    for png_path in sorted(glob.glob(os.path.join(image_dir, "*.png"))):
        stem = os.path.splitext(png_path)[0]
        # Find the corresponding cached latent NPZ
        npz_files = glob.glob(f"{stem}_*_anima.npz")
        if not npz_files:
            continue
        te_path = f"{stem}_anima_te.safetensors"
        if not os.path.exists(te_path):
            te_path = None
        images.append({
            "image_path": png_path,
            "npz_path": npz_files[0],  # use first bucket resolution
            "te_path": te_path,
            "stem": os.path.basename(stem),
        })
    return images


def load_cached_latents(npz_path, device):
    """Load cached latents from a preprocessed NPZ file."""
    data = np.load(npz_path)
    # Keys are like 'latents_180x90', find the latents key
    latent_key = [k for k in data.keys() if k.startswith("latents_")][0]
    latents = torch.from_numpy(data[latent_key]).unsqueeze(0)  # (1, 16, H/8, W/8)
    latents = latents.to(device, dtype=torch.bfloat16)

    # Extract original size from the matching key
    size_suffix = latent_key[len("latents_"):]  # e.g. "180x90"
    size_key = f"original_size_{size_suffix}"
    if size_key in data:
        # original_size is stored as [W, H] in the preprocessing pipeline
        orig_w, orig_h = int(data[size_key][0]), int(data[size_key][1])
    else:
        # Derive from latent shape: latent is (16, H/8, W/8)
        orig_h = latents.shape[-2] * 8
        orig_w = latents.shape[-1] * 8

    return latents, orig_h, orig_w


def load_cached_embedding(te_path, device):
    """Load cached crossattn_emb from a preprocessed TE safetensors file."""
    sd = load_file(te_path)
    # Use variant 0 (original caption, no shuffle)
    if "crossattn_emb_v0" in sd:
        emb = sd["crossattn_emb_v0"]
    elif "crossattn_emb" in sd:
        emb = sd["crossattn_emb"]
    else:
        return None
    return emb.unsqueeze(0).to(device, dtype=torch.float32)  # (1, 512, 1024)


def load_and_encode_image(args, device):
    """Load a single target image, encode to latents via VAE, then free VAE."""
    logger.info(f"Loading target image: {args.image}")
    img = Image.open(args.image).convert("RGB")

    if args.image_size is not None:
        h, w = args.image_size
    else:
        w, h = img.size
        h = (h // 32) * 32
        w = (w // 32) * 32

    if img.size != (w, h):
        img = img.resize((w, h), Image.LANCZOS)
        logger.info(f"Resized to {w}x{h}")

    logger.info("Loading VAE...")
    vae = qwen_image_autoencoder_kl.load_vae(
        args.vae, device="cpu", disable_mmap=True, spatial_chunk_size=args.vae_chunk_size
    )
    vae.to(device, dtype=torch.bfloat16)
    vae.eval()

    img_tensor = IMAGE_TRANSFORMS(img).unsqueeze(0).to(device)
    with torch.no_grad():
        latents = vae.encode_pixels_to_latents(img_tensor)
    latents = latents.to(torch.bfloat16)
    logger.info(f"Encoded latents shape: {latents.shape}")

    del vae
    clean_memory_on_device(device)

    return latents, h, w


# endregion


# region Embedding initialization


def create_initial_embedding(args, device, anima, te_path=None):
    """Create the initial embedding tensor to optimize.

    Priority: --init_embedding > --init_prompt > cached TE > zeros
    """
    if args.init_embedding is not None:
        logger.info(f"Init from saved embedding: {args.init_embedding}")
        sd = load_file(args.init_embedding)
        embed = sd["crossattn_emb"].to(device, dtype=torch.float32)
        if embed.ndim == 2:
            embed = embed.unsqueeze(0)
        return embed

    if args.init_prompt is not None and args.text_encoder is not None:
        logger.info(f"Init from prompt: {args.init_prompt}")
        return _encode_prompt(args, device, anima)

    if not args.init_zeros and te_path is not None:
        emb = load_cached_embedding(te_path, device)
        if emb is not None:
            logger.info(f"Init from cached TE: {te_path}")
            return emb

    logger.info("Init from zeros")
    return torch.zeros(1, 512, 1024, dtype=torch.float32, device=device)


def _encode_prompt(args, device, anima):
    """Encode a text prompt to crossattn_emb via text encoder + LLM adapter."""
    tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
        qwen3_path=args.text_encoder,
        t5_tokenizer_path=None,
        qwen3_max_length=512,
        t5_max_length=512,
    )
    strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
    encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()
    strategy_base.TextEncodingStrategy.set_strategy(encoding_strategy)

    text_encoder, _ = anima_utils.load_qwen3_text_encoder(
        args.text_encoder, dtype=torch.bfloat16, device=device,
    )
    text_encoder.eval()

    with torch.no_grad():
        tokens = tokenize_strategy.tokenize(args.init_prompt)
        embed = encoding_strategy.encode_tokens(tokenize_strategy, [text_encoder], tokens)
        crossattn_emb, _ = anima._preprocess_text_embeds(
            source_hidden_states=embed[0].to(device),
            target_input_ids=embed[2].to(device),
            target_attention_mask=embed[3].to(device),
            source_attention_mask=embed[1].to(device),
        )
        crossattn_emb[~embed[3].bool()] = 0
        if crossattn_emb.shape[1] < 512:
            crossattn_emb = F.pad(crossattn_emb, (0, 0, 0, 512 - crossattn_emb.shape[1]))

    del text_encoder
    clean_memory_on_device(device)

    return crossattn_emb.to(torch.float32)


# endregion


# region Optimization core


def sample_sigmas(args, batch_size, device):
    """Sample noise levels for the optimization step."""
    if args.sigma_sampling == "sigmoid":
        sigmas = torch.sigmoid(args.sigmoid_scale * torch.randn(batch_size, device=device))
    else:
        sigmas = torch.rand(batch_size, device=device)
    return sigmas


def inversion_step(anima, latents, embed_bf16, sigmas, padding_mask):
    """One optimization step: compute flow-matching loss."""
    n_t = sigmas.shape[0]

    lat = latents.expand(n_t, -1, -1, -1)
    noise = torch.randn_like(lat)

    sigmas_view = sigmas.view(-1, 1, 1, 1)
    noisy = (1.0 - sigmas_view) * lat + sigmas_view * noise
    noisy_5d = noisy.to(torch.bfloat16).unsqueeze(2)

    timesteps = sigmas.to(torch.bfloat16)
    emb = embed_bf16.expand(n_t, -1, -1)
    pm = padding_mask.expand(n_t, -1, -1, -1)

    pred = anima(noisy_5d, timesteps, emb, padding_mask=pm)
    pred = pred.squeeze(2)

    target = noise - lat
    return F.mse_loss(pred.float(), target.float())


def _install_block_grad_hooks(anima):
    """Register hooks to capture gradient norms at each block's cross-attention output.

    Hooks the o_proj (output projection) of each block's cross_attn module.
    Returns (handle_list, grad_norms_dict) where grad_norms_dict maps
    block index -> list of gradient norms (one per backward call).
    """
    handles = []
    grad_norms = {}
    num_blocks = len(anima.blocks)  # type: ignore[arg-type]

    for block_idx in range(num_blocks):
        grad_norms[block_idx] = []
        block = anima.blocks[block_idx]  # type: ignore[index]

        # Hook onto cross_attn.o_proj — the output projection of cross-attention
        target_module = getattr(getattr(block, "cross_attn", None), "o_proj", None)
        if target_module is None:
            continue

        def make_hook(idx):
            def hook_fn(_module, _grad_input, grad_output):
                if grad_output[0] is not None:
                    grad_norms[idx].append(grad_output[0].norm().item())
            return hook_fn

        h = target_module.register_full_backward_hook(make_hook(block_idx))
        handles.append(h)

    return handles, grad_norms


def optimize_embedding(args, anima, latents, init_embed, device, log_path=None):
    """Run the optimization loop for a single image. Returns (best_embed, best_loss)."""
    embed = torch.nn.Parameter(init_embed.clone())
    optimizer = torch.optim.AdamW([embed], lr=args.lr, weight_decay=0.0)

    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.steps, eta_min=args.lr * 0.01
        )
    else:
        scheduler = None

    h_lat, w_lat = latents.shape[-2], latents.shape[-1]
    padding_mask = torch.zeros(1, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)

    # Per-block gradient hooks
    block_grad_handles, block_grad_norms = [], {}
    if args.log_block_grads:
        block_grad_handles, block_grad_norms = _install_block_grad_hooks(anima)

    # CSV log
    csv_file = None
    csv_writer = None
    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        csv_file = open(log_path, "w", newline="")
        fieldnames = ["step", "loss", "best_loss", "lr", "grad_norm"]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    best_loss = float("inf")
    best_embed = None
    grad_accum = args.grad_accum

    pbar = tqdm(range(args.steps), desc="Inverting", leave=False)
    for step in pbar:
        optimizer.zero_grad()

        accum_loss = 0.0
        for _ in range(grad_accum):
            sigmas = sample_sigmas(args, args.timesteps_per_step, device)
            embed_bf16 = embed.to(torch.bfloat16)

            loss = inversion_step(anima, latents, embed_bf16, sigmas, padding_mask)
            (loss / grad_accum).backward()
            accum_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_([embed], max_norm=1.0).item()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_val = accum_loss / grad_accum
        if loss_val < best_loss:
            best_loss = loss_val
            best_embed = embed.detach().clone()

        if step % args.log_every == 0 or step == args.steps - 1:
            lr_now = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss_val:.6f}", best=f"{best_loss:.6f}", lr=f"{lr_now:.2e}")

        # Write every step to CSV
        if csv_writer is not None:
            csv_writer.writerow({
                "step": step,
                "loss": f"{loss_val:.6f}",
                "best_loss": f"{best_loss:.6f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                "grad_norm": f"{grad_norm:.6f}",
            })

    # Finalize logging
    if csv_file is not None:
        csv_file.close()

    # Save per-block gradient summary
    if args.log_block_grads and log_path is not None:
        block_summary = {}
        for idx, norms in block_grad_norms.items():
            if norms:
                block_summary[f"block_{idx:02d}"] = {
                    "mean": float(np.mean(norms)),
                    "std": float(np.std(norms)),
                    "max": float(np.max(norms)),
                    "min": float(np.min(norms)),
                }
        block_log_path = log_path.replace(".csv", "_block_grads.json")
        with open(block_log_path, "w") as f:
            json.dump(block_summary, f, indent=2)
        logger.info(f"  Block gradient summary: {block_log_path}")

    # Cleanup hooks
    for h in block_grad_handles:
        h.remove()

    return best_embed, best_loss


# endregion


# region Verification


def verify_embedding(args, anima, embed, h, w, device, save_path):
    """Generate an image from the inverted embedding to verify quality."""
    from library import inference_utils

    logger.info("Generating verification image...")

    vae = qwen_image_autoencoder_kl.load_vae(
        args.vae, device="cpu", disable_mmap=True, spatial_chunk_size=args.vae_chunk_size
    )
    vae.to(device, dtype=torch.bfloat16)
    vae.eval()

    embed_bf16 = embed.to(device=device, dtype=torch.bfloat16)
    if embed_bf16.ndim == 2:
        embed_bf16 = embed_bf16.unsqueeze(0)

    h_lat, w_lat = h // 8, w // 8
    padding_mask = torch.zeros(1, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)

    gen = torch.Generator(device=device).manual_seed(args.verify_seed)
    latents = torch.randn(
        1, anima_models.Anima.LATENT_CHANNELS, 1, h_lat, w_lat,
        device=device, dtype=torch.bfloat16, generator=gen,
    )

    timesteps, sigmas = inference_utils.get_timesteps_sigmas(args.verify_steps, args.flow_shift, device)
    timesteps = (timesteps / 1000).to(device, dtype=torch.bfloat16)

    if hasattr(anima, "switch_block_swap_for_inference"):
        anima.switch_block_swap_for_inference()

    with torch.no_grad():
        for step_i, t in enumerate(tqdm(timesteps, desc="Denoising", leave=False)):
            if hasattr(anima, "prepare_block_swap_before_forward"):
                anima.prepare_block_swap_before_forward()
            t_expand = t.unsqueeze(0)
            noise_pred = anima(latents, t_expand, embed_bf16, padding_mask=padding_mask)
            latents = inference_utils.step(latents, noise_pred, sigmas, step_i).to(torch.bfloat16)

    with torch.no_grad():
        pixels = vae.decode_to_pixels(latents.squeeze(2))
    pixels = ((pixels + 1.0) / 2.0).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    pixels = (pixels * 255).clip(0, 255).astype("uint8")
    Image.fromarray(pixels).save(save_path)
    logger.info(f"Saved verification image: {save_path}")

    del vae
    clean_memory_on_device(device)


# endregion


# region Main


def process_single(args, anima, device):
    """Process a single image (--image mode)."""
    stem = os.path.splitext(os.path.basename(args.image))[0]
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    latents, h, w = load_and_encode_image(args, device)
    init_embed = create_initial_embedding(args, device, anima)

    log_path = os.path.join(args.logs_dir, f"{stem}.csv")
    best_embed, best_loss = optimize_embedding(args, anima, latents, init_embed, device, log_path=log_path)

    out_path = os.path.join(args.results_dir, f"{stem}_inverted.safetensors")
    save_dict = {"crossattn_emb": best_embed.squeeze(0).to(torch.bfloat16)}
    metadata = {
        "source_image": os.path.basename(args.image),
        "image_size": f"{h}x{w}",
        "steps": str(args.steps),
        "lr": str(args.lr),
        "best_loss": f"{best_loss:.6f}",
        "init_prompt": args.init_prompt or "",
    }
    save_file(save_dict, out_path, metadata=metadata)
    logger.info(f"Saved: {out_path} (loss={best_loss:.6f})")

    if args.verify and args.vae:
        verify_path = os.path.join(args.results_dir, f"{stem}_verify.png")
        verify_embedding(args, anima, best_embed, h, w, device, verify_path)


def process_batch(args, anima, device):
    """Process a directory of preprocessed images (--image_dir mode)."""
    images = discover_images(args.image_dir)
    if not images:
        logger.error(f"No images with cached latents found in {args.image_dir}")
        return

    if args.shuffle:
        random.shuffle(images)

    if args.num_images is not None:
        images = images[:args.num_images]

    logger.info(f"Processing {len(images)} images from {args.image_dir}")
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    for i, img_info in enumerate(images):
        stem = img_info["stem"]
        logger.info(f"[{i+1}/{len(images)}] {stem}")

        out_path = os.path.join(args.results_dir, f"{stem}_inverted.safetensors")
        if os.path.exists(out_path):
            logger.info(f"  Skipping (already exists): {out_path}")
            continue

        # Load cached latents
        latents, orig_h, orig_w = load_cached_latents(img_info["npz_path"], device)

        # Initialize from cached TE embedding
        init_embed = create_initial_embedding(args, device, anima, te_path=img_info["te_path"])

        # Optimize
        log_path = os.path.join(args.logs_dir, f"{stem}.csv")
        best_embed, best_loss = optimize_embedding(args, anima, latents, init_embed, device, log_path=log_path)

        # Save
        save_dict = {"crossattn_emb": best_embed.squeeze(0).to(torch.bfloat16)}
        metadata = {
            "source_image": stem,
            "image_size": f"{orig_h}x{orig_w}",
            "steps": str(args.steps),
            "lr": str(args.lr),
            "best_loss": f"{best_loss:.6f}",
        }
        save_file(save_dict, out_path, metadata=metadata)
        logger.info(f"  Saved: {out_path} (loss={best_loss:.6f})")

        if args.verify and args.vae:
            verify_path = os.path.join(args.results_dir, f"{stem}_verify.png")
            verify_embedding(args, anima, best_embed, orig_h, orig_w, device, verify_path)


def main():
    args = parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Device: {device}")

    # Load frozen DiT
    logger.info("Loading DiT model...")
    is_swapping = args.blocks_to_swap > 0
    grad_ckpt = args.blocks_to_swap < 0
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
    anima.split_attn = False  # all batch elements share the same image; avoids data-dependent graph breaks

    if is_swapping:
        logger.info(f"Enabling block swap: {args.blocks_to_swap} blocks to CPU")
        anima.enable_block_swap(args.blocks_to_swap, device)
        anima.move_to_device_except_swap_blocks(device)
        anima.prepare_block_swap_before_forward()
    else:
        anima.to(device)
        if grad_ckpt:
            # Gradient checkpointing: recompute block activations during backward.
            # Blocks must be in train mode for checkpointing to activate.
            logger.info("Enabling gradient checkpointing")
            anima.enable_gradient_checkpointing()
            for block in anima.blocks:  # type: ignore[union-attr]
                block.train()

        logger.info("Compiling DiT with torch.compile...")
        anima = torch.compile(anima)

    if args.image_dir is not None:
        process_batch(args, anima, device)
    else:
        process_single(args, anima, device)

    logger.info("Done!")


# endregion


if __name__ == "__main__":
    main()
