# Custom training functions for Anima LoRA training
# Stripped version: only apply_masked_loss and add_custom_train_arguments

import argparse
import torch
from .utils import setup_logging

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)


def add_custom_train_arguments(
    parser: argparse.ArgumentParser, support_weighted_captions: bool = True
):
    parser.add_argument(
        "--min_snr_gamma",
        type=float,
        default=None,
        help="gamma for reducing the weight of high loss timesteps. Lower numbers have stronger effect. 5 is recommended by paper.",
    )
    parser.add_argument(
        "--scale_v_pred_loss_like_noise_pred",
        action="store_true",
        help="scale v-prediction loss like noise prediction loss",
    )
    parser.add_argument(
        "--v_pred_like_loss",
        type=float,
        default=None,
        help="add v-prediction like loss multiplied by this value",
    )
    parser.add_argument(
        "--debiased_estimation_loss",
        action="store_true",
        help="debiased estimation loss",
    )
    if support_weighted_captions:
        parser.add_argument(
            "--weighted_captions",
            action="store_true",
            default=False,
            help="Enable weighted captions in the standard style (token:1.3).",
        )


def apply_masked_loss(loss, batch) -> torch.FloatTensor:
    if "conditioning_images" in batch:
        mask_image = (
            batch["conditioning_images"].to(dtype=loss.dtype)[:, 0].unsqueeze(1)
        )  # use R channel
        mask_image = mask_image / 2 + 0.5
    elif "alpha_masks" in batch and batch["alpha_masks"] is not None:
        mask_image = (
            batch["alpha_masks"].to(dtype=loss.dtype).unsqueeze(1)
        )  # add channel dimension
    else:
        return loss

    # resize to the same size as the loss
    mask_image = torch.nn.functional.interpolate(
        mask_image, size=loss.shape[2:], mode="area"
    )
    loss = loss * mask_image
    return loss
