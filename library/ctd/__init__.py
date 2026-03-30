"""Manga text segmentation using UNet++ with EfficientNetV2 backbone.

Provides text segmentation masks for training image masking.
Model: https://huggingface.co/a-b-c-x-y-z/Manga-Text-Segmentation-2025
"""

from __future__ import annotations

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2

ENCODER = "tu-efficientnetv2_rw_m"
HF_REPO = "a-b-c-x-y-z/Manga-Text-Segmentation-2025"
HF_FILENAME = "model.pth"


def _convert_batchnorm_to_groupnorm(module: nn.Module) -> None:
    """Replace BatchNorm2d with GroupNorm in decoder (matches training setup)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            num_groups = 8
            if num_channels < num_groups or num_channels % num_groups != 0:
                for i in range(min(num_channels, 8), 1, -1):
                    if num_channels % i == 0:
                        num_groups = i
                        break
                else:
                    num_groups = 1
            setattr(
                module,
                name,
                nn.GroupNorm(num_groups=num_groups, num_channels=num_channels),
            )
        else:
            _convert_batchnorm_to_groupnorm(child)


def load_model(model_path: str | None = None, device: str = "cuda") -> nn.Module:
    """Load UNet++ text segmentation model.

    Args:
        model_path: Path to model.pth. If None, downloads from HuggingFace Hub.
        device: Torch device string.
    """
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
        decoder_attention_type="scse",
    )
    _convert_batchnorm_to_groupnorm(model.decoder)

    if model_path is None:
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


_transform = Compose(
    [
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


@torch.no_grad()
def detect_mask(
    model: nn.Module,
    image: np.ndarray,
    device: str = "cuda",
    text_threshold: float | None = None,
) -> np.ndarray:
    """Run text segmentation and return mask.

    Args:
        model: Loaded UNet++ model.
        image: RGB uint8 image (H, W, 3).
        device: Torch device string.
        text_threshold: Optional binarization threshold for the mask.

    Returns:
        Grayscale mask (H, W) uint8. Higher values = more likely text.
        Values 0-255 (continuous unless text_threshold is set).
    """
    h, w = image.shape[:2]

    # Pad to multiple of 32
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32

    tensor = _transform(image=image)["image"].unsqueeze(0).to(device)

    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)

    # Forward pass
    if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
        with torch.amp.autocast("cuda"):
            logits = model(tensor)
    else:
        logits = model(tensor)

    prob_map = logits.sigmoid()[0, 0, :h, :w].cpu().numpy()

    if text_threshold is not None:
        prob_map = (prob_map > text_threshold).astype(np.float32)

    mask = (prob_map * 255).astype(np.uint8)
    return mask
