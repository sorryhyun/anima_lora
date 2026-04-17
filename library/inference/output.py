"""Output handling for Anima inference: decoding, saving latents and images."""

import argparse
import datetime
import logging
import os
import time
from typing import Optional, Tuple

import torch
from safetensors.torch import save_file
from PIL import Image

from library.models import qwen_vae as qwen_image_autoencoder_kl

logger = logging.getLogger(__name__)


def get_time_flag():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S-%f")[
        :-3
    ]


def check_inputs(args) -> Tuple[int, int]:
    """Validate image size.

    Returns:
        Tuple[int, int]: (height, width)
    """
    height = args.image_size[0]
    width = args.image_size[1]

    if height % 32 != 0 or width % 32 != 0:
        raise ValueError(
            f"`height` and `width` have to be divisible by 32 but are {height} and {width}."
        )

    return height, width


def decode_latent(
    vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage,
    latent: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    logger.info(f"Decoding image. Latent shape {latent.shape}, device {device}")

    vae.to(device)
    with torch.no_grad():
        pixels = vae.decode_to_pixels(latent.to(device, dtype=vae.dtype))
    if (
        pixels.ndim == 5
    ):  # remove frame dimension if exists, [B, C, F, H, W] -> [B, C, H, W]
        pixels = pixels.squeeze(2)

    pixels = pixels.to(
        "cpu", dtype=torch.float32
    )  # move to CPU and convert to float32 (bfloat16 is not supported by numpy)
    vae.to("cpu")

    logger.info(f"Decoded. Pixel shape {pixels.shape}")
    return pixels[0]  # remove batch dimension


def save_latent(
    latent: torch.Tensor, args: argparse.Namespace, height: int, width: int
) -> str:
    """Save latent to file.

    Returns:
        str: Path to saved latent file
    """
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = get_time_flag()

    seed = args.seed

    latent_path = f"{save_path}/{time_flag}_{seed}_latent.safetensors"

    if args.no_metadata:
        metadata = None
    else:
        metadata = {
            "seeds": f"{seed}",
            "prompt": f"{args.prompt}",
            "height": f"{height}",
            "width": f"{width}",
            "infer_steps": f"{args.infer_steps}",
            "guidance_scale": f"{args.guidance_scale}",
        }
        if args.negative_prompt is not None:
            metadata["negative_prompt"] = f"{args.negative_prompt}"

    sd = {"latent": latent.contiguous()}
    save_file(sd, latent_path, metadata=metadata)
    logger.info(f"Latent saved to: {latent_path}")

    return latent_path


def save_images(
    sample: torch.Tensor,
    args: argparse.Namespace,
    original_base_name: Optional[str] = None,
) -> str:
    """Save images to directory.

    Returns:
        str: Path to saved image
    """
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = get_time_flag()

    seed = args.seed
    original_name = "" if original_base_name is None else f"_{original_base_name}"
    image_name = f"{time_flag}_{seed}{original_name}"

    x = torch.clamp(sample, -1.0, 1.0)
    x = ((x + 1.0) * 127.5).to(torch.uint8).cpu().numpy()
    x = x.transpose(1, 2, 0)  # C, H, W -> H, W, C

    image = Image.fromarray(x)
    image.save(os.path.join(save_path, f"{image_name}.png"))

    logger.info(f"Sample images saved to: {save_path}/{image_name}")

    return f"{save_path}/{image_name}"


def save_output(
    args: argparse.Namespace,
    vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage,
    latent: torch.Tensor,
    device: torch.device,
    original_base_name: Optional[str] = None,
) -> None:
    """Decode latent and save output (latent/images based on args.output_type)."""
    height, width = latent.shape[-2], latent.shape[-1]  # BCTHW
    height *= 8
    width *= 8
    if args.output_type == "latent" or args.output_type == "latent_images":
        save_latent(latent, args, height, width)
    if args.output_type == "latent":
        return

    if vae is None:
        logger.error("VAE is None, cannot decode latents for saving video/images.")
        return

    if latent.ndim == 2:  # S,C. For packed latents from other inference scripts
        latent = latent.unsqueeze(0)
        height, width = check_inputs(args)
        latent = latent.view(
            1,
            vae.latent_channels,
            1,  # Frame dimension
            height // 8,
            width // 8,
        )

    image = decode_latent(vae, latent, device)

    if args.output_type == "images" or args.output_type == "latent_images":
        if original_base_name is None:
            original_name = ""
        else:
            original_name = f"_{original_base_name}"
        save_images(image, args, original_name)
