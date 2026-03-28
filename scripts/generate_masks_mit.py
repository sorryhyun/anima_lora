#!/usr/bin/env python3
"""Generate text/speech-bubble masks for training images using manga-image-translator's ComicTextDetector."""

import argparse
import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def get_image_files(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def save_mask(path: Path, alpha_mask: np.ndarray) -> None:
    Image.fromarray(alpha_mask, mode="L").save(path)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=str, required=True, help="Image directory")
    parser.add_argument("--mask-dir", type=str, required=True, help="Output mask directory")
    parser.add_argument("--model-path", type=str, default=None, help="Path to comictextdetector.pt (auto-detected if omitted)")
    parser.add_argument("--force", action="store_true", help="Regenerate existing masks")
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    parser.add_argument("--detect-size", type=int, default=1024, help="Detection input size (default: 1024)")
    parser.add_argument("--text-threshold", type=float, default=0.5, help="Text segmentation threshold (default: 0.5)")
    parser.add_argument("--box-threshold", type=float, default=0.6, help="Box detection threshold (default: 0.6)")
    parser.add_argument("--unclip-ratio", type=float, default=1.5, help="Unclip ratio for text regions (default: 1.5)")
    parser.add_argument("--dilate", type=int, default=5, help="Mask dilation in pixels (default: 5)")
    parser.add_argument("--workers", type=int, default=4, help="I/O workers (default: 4)")
    args = parser.parse_args()

    dilate_kernel = np.ones((args.dilate, args.dilate), dtype=np.uint8) if args.dilate > 0 else None

    # Block the top-level manga_translator __init__ from importing the full translator
    # (which transitively requires pydensecrf). We only need the detection sub-package.
    import importlib
    import types

    mit_root = Path(__file__).resolve().parents[2] / "manga-image-translator"
    if str(mit_root) not in sys.path:
        sys.path.insert(0, str(mit_root))

    # Insert a stub manga_translator package that doesn't run __init__.py's re-exports
    sys.modules["manga_translator"] = types.ModuleType("manga_translator")
    sys.modules["manga_translator"].__path__ = [str(mit_root / "manga_translator")]

    from manga_translator.detection.ctd import ComicTextDetector

    image_dir = Path(args.image_dir)
    masks_dir = Path(args.mask_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Symlink model files if a custom path is provided
    if args.model_path:
        model_path = Path(args.model_path)
        mit_model_dir = Path(os.path.dirname(__file__), "..", "..", "manga-image-translator", "models", "detection")
        mit_model_dir.mkdir(parents=True, exist_ok=True)
        for name in ("comictextdetector.pt", "comictextdetector.pt.onnx"):
            src = model_path / name if model_path.is_dir() else model_path
            dst = mit_model_dir / name
            if src.exists() and not dst.exists():
                dst.symlink_to(src.resolve())

    print("Loading ComicTextDetector...")
    detector = ComicTextDetector()
    await detector.load(device=args.device)

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

    pbar = tqdm(total=total, desc="Generating masks (MIT)")
    for image_path, mask_path in work_items:
        # Load image as RGB numpy array
        pil_image = Image.open(image_path).convert("RGB")
        img_np = np.array(pil_image)

        # Run detection
        _textlines, mask_raw, mask_refined = await detector.detect(
            image=img_np,
            detect_size=args.detect_size,
            text_threshold=args.text_threshold,
            box_threshold=args.box_threshold,
            unclip_ratio=args.unclip_ratio,
            invert=False,
            gamma_correct=False,
            rotate=False,
        )

        # Use refined mask if available, otherwise raw
        mask = mask_refined if mask_refined is not None else mask_raw

        pbar.update(1)

        if mask is None or not mask.any():
            pbar.set_postfix_str(f"{image_path.name}: skipped")
            continue

        # Ensure binary uint8
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
    await detector.unload()
    print(f"Masks saved to {masks_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
