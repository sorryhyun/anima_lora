import glob
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from library.datasets.buckets import BucketManager
from library.datasets.subsets import ImageInfo
from library.device_utils import clean_memory_on_device
from library.utils import resize_image

logger = logging.getLogger(__name__)


IMAGE_EXTENSIONS = [
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".PNG",
    ".JPG",
    ".JPEG",
    ".WEBP",
    ".BMP",
]


def load_mask_from_dir(
    mask_dir: str, image_path: str, size: Tuple[int, int]
) -> Optional[torch.Tensor]:
    """Load a mask from a separate file in mask_dir matching the image stem.

    Args:
        mask_dir: Directory containing {stem}_mask.png files.
        image_path: Path to the source image (used for stem matching).
        size: (width, height) to resize the mask to if needed.

    Returns:
        Float tensor [H, W] in [0, 1] range, or None if no mask file found.
    """
    stem = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join(mask_dir, f"{stem}_mask.png")
    if not os.path.exists(mask_path):
        return None
    mask = Image.open(mask_path).convert("L")
    if (mask.width, mask.height) != size:
        mask = mask.resize(size, Image.LANCZOS)
    mask_np = np.array(mask, dtype=np.float32) / 255.0
    return torch.FloatTensor(mask_np)


try:
    import pillow_avif  # noqa: F401

    IMAGE_EXTENSIONS.extend([".avif", ".AVIF"])
except Exception:
    pass

# JPEG-XL on Linux
try:
    from jxlpy import JXLImagePlugin  # noqa: F401
    from library.jpeg_xl_util import get_jxl_size  # noqa: F401

    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])
except Exception:
    pass

# JPEG-XL on Linux and Windows
try:
    import pillow_jxl  # noqa: F401
    from library.jpeg_xl_util import get_jxl_size  # noqa: F401

    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])
except Exception:
    pass

IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX = "_te_outputs.npz"
TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX_SD3 = "_sd3_te.npz"


def load_image(image_path, alpha=False):
    try:
        with Image.open(image_path) as image:
            if alpha:
                if not image.mode == "RGBA":
                    image = image.convert("RGBA")
            else:
                if not image.mode == "RGB":
                    image = image.convert("RGB")
            img = np.array(image, np.uint8)
            return img
    except (IOError, OSError) as e:
        logger.error(f"Error loading file: {image_path}")
        raise e


def trim_and_resize_if_required(
    random_crop: bool,
    image: np.ndarray,
    reso,
    resized_size: Tuple[int, int],
    resize_interpolation: Optional[str] = None,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int, int, int]]:
    image_height, image_width = image.shape[0:2]
    original_size = (image_width, image_height)

    if image_width != resized_size[0] or image_height != resized_size[1]:
        image = resize_image(
            image,
            image_width,
            image_height,
            resized_size[0],
            resized_size[1],
            resize_interpolation,
        )

    image_height, image_width = image.shape[0:2]

    if image_width > reso[0]:
        trim_size = image_width - reso[0]
        import random

        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        image = image[:, p : p + reso[0]]
    if image_height > reso[1]:
        trim_size = image_height - reso[1]
        import random

        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        image = image[p : p + reso[1]]

    crop_ltrb = BucketManager.get_crop_ltrb(reso, original_size)

    assert image.shape[0] == reso[1] and image.shape[1] == reso[0], (
        f"internal error, illegal trimmed size: {image.shape}, {reso}"
    )
    return image, original_size, crop_ltrb


# for new_cache_latents
def load_images_and_masks_for_caching(
    image_infos: List[ImageInfo], use_alpha_mask: bool, random_crop: bool
) -> Tuple[
    torch.Tensor,
    List[np.ndarray],
    List[Tuple[int, int]],
    List[Tuple[int, int, int, int]],
]:
    r"""
    requires image_infos to have: [absolute_path or image], bucket_reso, resized_size

    returns: image_tensor, alpha_masks, original_sizes, crop_ltrbs

    image_tensor: torch.Tensor = torch.Size([B, 3, H, W]), ...], normalized to [-1, 1]
    alpha_masks: List[np.ndarray] = [np.ndarray([H, W]), ...], normalized to [0, 1]
    original_sizes: List[Tuple[int, int]] = [(W, H), ...]
    crop_ltrbs: List[Tuple[int, int, int, int]] = [(L, T, R, B), ...]
    """
    images: List[torch.Tensor] = []
    alpha_masks: List[np.ndarray] = []
    original_sizes: List[Tuple[int, int]] = []
    crop_ltrbs: List[Tuple[int, int, int, int]] = []
    for info in image_infos:
        has_mask_file = getattr(info, "mask_path", None) is not None
        image = (
            load_image(info.absolute_path, use_alpha_mask and not has_mask_file)
            if info.image is None
            else np.array(info.image, np.uint8)
        )
        image, original_size, crop_ltrb = trim_and_resize_if_required(
            random_crop,
            image,
            info.bucket_reso,
            info.resized_size,
            resize_interpolation=info.resize_interpolation,
        )

        original_sizes.append(original_size)
        crop_ltrbs.append(crop_ltrb)

        if has_mask_file:
            alpha_mask = load_mask_from_dir(
                os.path.dirname(info.mask_path),
                info.absolute_path,
                (image.shape[1], image.shape[0]),
            )
            if alpha_mask is None:
                alpha_mask = torch.ones(
                    (image.shape[0], image.shape[1]), dtype=torch.float32
                )
        elif use_alpha_mask:
            if image.shape[2] == 4:
                alpha_mask = image[:, :, 3]
                alpha_mask = alpha_mask.astype(np.float32) / 255.0
                alpha_mask = torch.FloatTensor(alpha_mask)
            else:
                alpha_mask = torch.ones_like(image[:, :, 0], dtype=torch.float32)
        else:
            alpha_mask = None
        alpha_masks.append(alpha_mask)

        image = image[:, :, :3]
        image = IMAGE_TRANSFORMS(image)
        images.append(image)

    img_tensor = torch.stack(images, dim=0)
    return img_tensor, alpha_masks, original_sizes, crop_ltrbs


def cache_batch_latents(
    vae,
    cache_to_disk: bool,
    image_infos: List[ImageInfo],
    flip_aug: bool,
    use_alpha_mask: bool,
    random_crop: bool,
) -> None:
    r"""
    requires image_infos to have: absolute_path, bucket_reso, resized_size, latents_npz
    optionally requires image_infos to have: image
    if cache_to_disk is True, set info.latents_npz
        flipped latents is also saved if flip_aug is True
    if cache_to_disk is False, set info.latents
        latents_flipped is also set if flip_aug is True
    latents_original_size and latents_crop_ltrb are also set
    """
    images = []
    alpha_masks: List[np.ndarray] = []
    for info in image_infos:
        has_mask_file = getattr(info, "mask_path", None) is not None
        image = (
            load_image(info.absolute_path, use_alpha_mask and not has_mask_file)
            if info.image is None
            else np.array(info.image, np.uint8)
        )
        image, original_size, crop_ltrb = trim_and_resize_if_required(
            random_crop,
            image,
            info.bucket_reso,
            info.resized_size,
            resize_interpolation=info.resize_interpolation,
        )

        info.latents_original_size = original_size
        info.latents_crop_ltrb = crop_ltrb

        if has_mask_file:
            alpha_mask = load_mask_from_dir(
                os.path.dirname(info.mask_path),
                info.absolute_path,
                (image.shape[1], image.shape[0]),
            )
            if alpha_mask is None:
                alpha_mask = torch.ones(
                    (image.shape[0], image.shape[1]), dtype=torch.float32
                )
        elif use_alpha_mask:
            if image.shape[2] == 4:
                alpha_mask = image[:, :, 3]
                alpha_mask = alpha_mask.astype(np.float32) / 255.0
                alpha_mask = torch.FloatTensor(alpha_mask)
            else:
                alpha_mask = torch.ones_like(image[:, :, 0], dtype=torch.float32)
        else:
            alpha_mask = None
        alpha_masks.append(alpha_mask)

        image = image[:, :, :3]
        image = IMAGE_TRANSFORMS(image)
        images.append(image)

    img_tensors = torch.stack(images, dim=0)
    img_tensors = img_tensors.to(device=vae.device, dtype=vae.dtype)

    with torch.no_grad():
        latents = vae.encode(img_tensors).latent_dist.sample().to("cpu")

    if flip_aug:
        img_tensors = torch.flip(img_tensors, dims=[3])
        with torch.no_grad():
            flipped_latents = vae.encode(img_tensors).latent_dist.sample().to("cpu")
    else:
        flipped_latents = [None] * len(latents)

    for info, latent, flipped_latent, alpha_mask in zip(
        image_infos, latents, flipped_latents, alpha_masks
    ):
        if torch.isnan(latents).any() or (
            flipped_latent is not None and torch.isnan(flipped_latent).any()
        ):
            raise RuntimeError(f"NaN detected in latents: {info.absolute_path}")

        if cache_to_disk:
            pass
        else:
            info.latents = latent
            if flip_aug:
                info.latents_flipped = flipped_latent
            info.alpha_mask = alpha_mask

    # import here to avoid circular dependency at module level
    from library.datasets.base import HIGH_VRAM

    if not HIGH_VRAM:
        clean_memory_on_device(vae.device)


def save_text_encoder_outputs_to_disk(npz_path, hidden_state1, hidden_state2, pool2):
    np.savez(
        npz_path,
        hidden_state1=hidden_state1.cpu().float().numpy(),
        hidden_state2=hidden_state2.cpu().float().numpy(),
        pool2=pool2.cpu().float().numpy(),
    )


def load_text_encoder_outputs_from_disk(npz_path):
    with np.load(npz_path) as f:
        hidden_state1 = torch.from_numpy(f["hidden_state1"])
        hidden_state2 = (
            torch.from_numpy(f["hidden_state2"]) if "hidden_state2" in f else None
        )
        pool2 = torch.from_numpy(f["pool2"]) if "pool2" in f else None
    return hidden_state1, hidden_state2, pool2


def is_disk_cached_latents_is_expected(
    reso, npz_path: str, flip_aug: bool, alpha_mask: bool
):
    expected_latents_size = (reso[1] // 8, reso[0] // 8)

    if not os.path.exists(npz_path):
        return False

    try:
        npz = np.load(npz_path)
        if "latents" not in npz or "original_size" not in npz or "crop_ltrb" not in npz:
            return False
        if npz["latents"].shape[1:3] != expected_latents_size:
            return False

        if flip_aug:
            if "latents_flipped" not in npz:
                return False
            if npz["latents_flipped"].shape[1:3] != expected_latents_size:
                return False

        if alpha_mask:
            if "alpha_mask" not in npz:
                return False
            if (npz["alpha_mask"].shape[1], npz["alpha_mask"].shape[0]) != reso:
                return False
        else:
            if "alpha_mask" in npz:
                return False
    except Exception as e:
        logger.error(f"Error loading file: {npz_path}")
        raise e

    return True


def glob_images(directory, base="*"):
    img_paths = []
    for ext in IMAGE_EXTENSIONS:
        if base == "*":
            img_paths.extend(
                glob.glob(os.path.join(glob.escape(directory), base + ext))
            )
        else:
            img_paths.extend(
                glob.glob(glob.escape(os.path.join(directory, base + ext)))
            )
    img_paths = list(set(img_paths))
    img_paths.sort()
    return img_paths


def glob_images_pathlib(dir_path, recursive):
    image_paths = []
    if recursive:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.rglob("*" + ext))
    else:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.glob("*" + ext))
    image_paths = list(set(image_paths))
    image_paths.sort()
    return image_paths


class ImageLoadingDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            tensor_pil = transforms.functional.pil_to_tensor(image)
        except Exception:
            logger.error("Could not load image path")
            return None

        return (tensor_pil, img_path)
