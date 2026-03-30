"""Vendored ComicTextDetector from manga-image-translator (GPL-3.0).

Provides text/speech-bubble segmentation masks for training image masking.
Only the model architecture and mask inference are included; box detection
and mask refinement (which require pyclipper/shapely) are omitted.

Model weights: https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.3
Source: https://github.com/zyddnys/manga-image-translator
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from .model import TextDetBase
from .utils import letterbox


def load_model(model_path: str, device: str = "cuda") -> TextDetBase:
    """Load ComicTextDetector model from checkpoint."""
    model = TextDetBase(model_path, device=device, act="leaky")
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def detect_mask(
    model: TextDetBase,
    image: np.ndarray,
    detect_size: int = 1024,
    device: str = "cuda",
    text_threshold: float | None = None,
) -> np.ndarray:
    """Run text detection and return segmentation mask.

    Args:
        model: Loaded TextDetBase model.
        image: RGB uint8 image (H, W, 3).
        detect_size: Input size for the detector.
        device: Torch device string.
        text_threshold: Optional binarization threshold for the mask.

    Returns:
        Grayscale mask (H, W) uint8. Higher values = more likely text.
        Values 0-255 (continuous unless text_threshold is set).
    """
    im_h, im_w = image.shape[:2]
    input_size = (detect_size, detect_size)

    # Preprocess: letterbox resize + normalize
    img_in, _ratio, (dw, dh) = letterbox(
        image, new_shape=input_size, auto=False, stride=64
    )
    img_in = img_in.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img_in = np.ascontiguousarray(img_in).astype(np.float32) / 255.0
    img_in = torch.from_numpy(img_in[None]).to(device)

    # Forward pass
    _blks, mask, _lines = model(img_in)

    # Postprocess mask
    mask = mask.squeeze().cpu().numpy()
    # Crop padding
    mask = mask[: mask.shape[0] - dh, : mask.shape[1] - dw] if dh or dw else mask

    if text_threshold is not None:
        mask = (mask > text_threshold).astype(np.float32)

    mask = (mask * 255).astype(np.uint8)

    # Resize to original image dimensions
    if mask.shape[:2] != (im_h, im_w):
        mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)

    return mask
