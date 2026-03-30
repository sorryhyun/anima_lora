#!/usr/bin/env python3
"""Generate text segmentation masks for training images.

Model: https://huggingface.co/a-b-c-x-y-z/Manga-Text-Segmentation-2025
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Allow running from scripts/ subdirectory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library.ctd import detect_mask, load_model

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def get_image_files(image_dir: Path) -> list[Path]:
    return sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def save_mask(path: Path, alpha_mask: np.ndarray) -> None:
    Image.fromarray(alpha_mask, mode="L").save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=str, required=True, help="Image directory")
    parser.add_argument(
        "--mask-dir", type=str, required=True, help="Output mask directory"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model.pth (downloads from HuggingFace if not specified)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Regenerate existing masks"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (default: cuda)"
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.8,
        help="Text segmentation threshold (default: 0.7)",
    )
    parser.add_argument(
        "--dilate", type=int, default=5, help="Mask dilation in pixels (default: 5)"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="I/O workers (default: 4)"
    )
    args = parser.parse_args()

    dilate_kernel = (
        np.ones((args.dilate, args.dilate), dtype=np.uint8) if args.dilate > 0 else None
    )

    print("Loading text segmentation model...")
    model = load_model(args.model_path, device=args.device)

    image_dir = Path(args.image_dir)
    masks_dir = Path(args.mask_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Filter to work items
    work_items = []
    for image_path in get_image_files(image_dir):
        mask_path = masks_dir / f"{image_path.stem}_mask.png"
        if mask_path.exists() and not args.force:
            continue
        work_items.append((image_path, mask_path))

    total = len(work_items)
    if total == 0:
        print("No images to process.")
        return

    pool = ThreadPoolExecutor(max_workers=args.workers)

    pbar = tqdm(total=total, desc="Generating masks")
    for image_path, mask_path in work_items:
        # Load image as RGB numpy array
        pil_image = Image.open(image_path).convert("RGB")
        img_np = np.array(pil_image)

        # Run detection — returns mask where higher values = more likely text
        mask = detect_mask(
            model,
            img_np,
            device=args.device,
            text_threshold=args.text_threshold,
        )

        pbar.update(1)

        if mask is None or not mask.any():
            pbar.set_postfix_str(f"{image_path.name}: skipped")
            continue

        # Binarize
        combined_mask = (mask > 127).astype(np.uint8)

        if dilate_kernel is not None:
            combined_mask = cv2.dilate(combined_mask, dilate_kernel, iterations=1)

        # Invert: detected=1 → alpha=0 (ignore), no detection → alpha=255 (train)
        alpha_mask = ((1 - combined_mask) * 255).astype(np.uint8)

        pool.submit(save_mask, mask_path, alpha_mask)

        h, w = img_np.shape[:2]
        masked_pct = 100 * np.count_nonzero(combined_mask) / (w * h)
        pbar.set_postfix_str(f"{image_path.name}: {masked_pct:.1f}%")

    pbar.close()
    pool.shutdown(wait=True)
    print(f"Masks saved to {masks_dir}/")


if __name__ == "__main__":
    main()
