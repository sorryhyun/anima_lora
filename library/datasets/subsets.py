import logging
import math
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from library.datasets.path_filter import filter_paths_by_glob as filter_paths_by_glob

logger = logging.getLogger(__name__)


def _resolve_default_mask_dir() -> Optional[str]:
    """Resolve the default mask directory.

    Prefers the new ``post_image_dataset/masks/`` layout produced by
    ``make mask``; falls back to the legacy ``masks/{merged,sam,mit}/``
    triple so users who haven't re-run masking after the consolidation
    keep training without manual intervention.

    Returned path is relative, matching how other paths are resolved from the
    training CWD (anima_lora/).
    """
    candidates = (
        "post_image_dataset/masks",
        "masks/merged",
        "masks/sam",
        "masks/mit",
    )
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return None


def resolve_configured_mask_dir(mask_dir: Optional[str]) -> Optional[str]:
    """Gate a *config-level* ``mask_dir`` (``configs/preprocess.toml`` →
    preset → method → ``--mask_dir``) on the directory existing.

    The config default names where ``make mask`` *would* write, which is not
    the same as masks being present. Handing a nonexistent root to every
    subset would flip ``alpha_mask`` on for maskless checkouts and suppress
    the legacy ``masks/{merged,sam,mit}`` auto-resolution, so drop it here and
    let ``_resolve_default_mask_dir`` run. A subset that names its own
    ``mask_dir`` in the dataset blueprint bypasses this and keeps the loud
    "resolved 0 masks" warning from the dataset build.
    """
    if not mask_dir:
        return None
    if os.path.isdir(mask_dir):
        return mask_dir
    logger.debug("Configured mask_dir %s does not exist; ignoring.", mask_dir)
    return None


_FOLDER_REPEAT_RE = re.compile(r"^(\d+)_")


def folder_repeat_count(image_path: str, image_dir: str) -> Optional[int]:
    """Kohya-style per-folder repeat count for ``image_path``.

    Walks the directory components between ``image_dir`` and the image
    (deepest first) and returns ``n`` from the first component named
    ``{n}_...`` — so ``image_dir/5_mychar/img.png`` → 5, and a nested
    ``5_mychar/extra/img.png`` still resolves to 5. Returns ``None`` when no
    component matches or the image sits directly in ``image_dir``. ``0_...``
    is a valid result (0) — the loader drops such images from training.
    """
    try:
        rel = os.path.relpath(os.path.dirname(image_path), image_dir)
    except ValueError:
        return None
    if not rel or rel == "." or rel.startswith(".."):
        return None
    # glob can mix `/` and `\` on Windows — split on both.
    for part in reversed(re.split(r"[\\/]", rel)):
        m = _FOLDER_REPEAT_RE.match(part)
        if m:
            return int(m.group(1))
    return None


def split_train_val(
    paths: List[str],
    sizes: List[Optional[Tuple[int, int]]],
    is_training_dataset: bool,
    validation_split: float,
    validation_seed: int | None,
    validation_split_num: int = 0,
) -> Tuple[List[str], List[Optional[Tuple[int, int]]]]:
    """
    Split the dataset into train and validation.

    Shuffle the dataset based on ``validation_seed`` (or current RNG when
    None), then carve off a validation slice. When ``validation_split_num > 0``
    the count-based split wins over the fractional ``validation_split``;
    otherwise the original fraction-based split is used. For example, with
    ``validation_split=0.2`` on 100 paths: [0:80] = 80 training, [80:] = 20
    validation.

    Guardrail: when ``validation_split_num >= len(paths)`` (or
    ``validation_split >= 1.0``) the requested val slice would consume the
    entire training pool — surface that as a warning and disable validation
    for this subset (training gets all paths, val gets none). The val-side
    caller's empty return is dropped by ``load_dreambooth_dir``'s "no images
    found" branch, so the trainer simply runs without a val pass.
    """
    dataset = list(zip(paths, sizes))
    if validation_seed is not None:
        logging.info(f"Using validation seed: {validation_seed}")
        prevstate = random.getstate()
        random.seed(validation_seed)
        random.shuffle(dataset)
        random.setstate(prevstate)
    else:
        random.shuffle(dataset)

    paths, sizes = zip(*dataset)
    paths = list(paths)
    sizes = list(sizes)

    val_would_exhaust = (
        validation_split_num
        and validation_split_num > 0
        and validation_split_num >= len(paths)
    ) or (validation_split and validation_split >= 1.0)
    if val_would_exhaust:
        if is_training_dataset:
            # Log only on the training pass so the warning surfaces once per
            # subset (split_train_val is called twice — once per is_training).
            logger.warning(
                "validation_split_num=%s / validation_split=%s would consume the "
                "entire subset (size=%d); disabling validation for this subset.",
                validation_split_num,
                validation_split,
                len(paths),
            )
            return paths, sizes
        return [], []

    if validation_split_num and validation_split_num > 0:
        n_val = int(validation_split_num)
        split = len(paths) - n_val
    elif is_training_dataset:
        split = math.ceil(len(paths) * (1 - validation_split))
    else:
        split = len(paths) - round(len(paths) * validation_split)

    if is_training_dataset:
        return paths[0:split], sizes[0:split]
    return paths[split:], sizes[split:]


class ImageInfo:
    def __init__(
        self,
        image_key: str,
        num_repeats: int,
        caption: str,
        is_reg: bool,
        absolute_path: str,
        caption_dropout_rate: float = 0.0,
    ) -> None:
        self.image_key: str = image_key
        self.num_repeats: int = num_repeats
        self.caption: str = caption
        self.is_reg: bool = is_reg
        self.absolute_path: str = absolute_path
        self.caption_dropout_rate: float = caption_dropout_rate
        self.image_size: Tuple[int, int] = None
        self.resized_size: Tuple[int, int] = None
        self.bucket_reso: Tuple[int, int] = None
        self.latents: Optional[torch.Tensor] = None
        self.latents_flipped: Optional[torch.Tensor] = None
        self.latents_npz: Optional[str] = None  # set in cache_latents
        self.latents_original_size: Optional[Tuple[int, int]] = (
            None  # original image size, not latents size
        )
        self.latents_crop_ltrb: Optional[Tuple[int, int]] = (
            None  # crop left top right bottom in original pixel size, not latents size
        )
        self.cond_img_path: Optional[str] = None
        self.image: Optional[Any] = None  # optional, original PIL Image
        self.text_encoder_outputs_npz: Optional[str] = (
            None  # filename. set in cache_text_encoder_outputs
        )

        self.text_encoder_outputs: Optional[List[torch.Tensor]] = None
        self.text_encoder_outputs1: Optional[torch.Tensor] = None
        self.text_encoder_outputs2: Optional[torch.Tensor] = None
        self.text_encoder_pool2: Optional[torch.Tensor] = None

        self.alpha_mask: Optional[torch.Tensor] = (
            None  # alpha mask can be flipped in runtime
        )
        self.mask_path: Optional[str] = (
            None  # path to separate mask file (from mask_dir)
        )
        # Preloaded uint8 [H, W] mask at bucket_reso, populated once at dataset
        # init by BaseDataset._preload_alpha_masks(). Avoids per-fetch PNG decode.
        self.preloaded_alpha_mask: Optional[torch.Tensor] = None
        self.resize_interpolation: Optional[str] = None


class AugHelper:
    def __init__(self):
        pass

    def color_aug(self, image: np.ndarray):
        hue_shift_limit = 8

        # remove dependency to albumentations
        if random.random() <= 0.33:
            if random.random() > 0.5:
                hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                hue_shift = random.uniform(-hue_shift_limit, hue_shift_limit)
                if hue_shift < 0:
                    hue_shift = 180 + hue_shift
                hsv_img[:, :, 0] = (hsv_img[:, :, 0] + hue_shift) % 180
                image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
            else:
                gamma = random.uniform(0.95, 1.05)
                image = np.clip(image**gamma, 0, 255).astype(np.uint8)

        return {"image": image}

    def get_augmentor(
        self, use_color_aug: bool
    ):  # -> Optional[Callable[[np.ndarray], Dict[str, np.ndarray]]]:
        return self.color_aug if use_color_aug else None


class BaseSubset:
    def __init__(
        self,
        image_dir: Optional[str],
        alpha_mask: Optional[bool],
        num_repeats: int,
        sample_ratio: float,
        caption_separator: str,
        keep_tokens_separator: str,
        secondary_separator: Optional[str],
        enable_wildcard: bool,
        color_aug: bool,
        flip_aug: bool,
        face_crop_aug_range: Optional[Tuple[float, float]],
        random_crop: bool,
        caption_dropout_rate: float,
        caption_dropout_every_n_epochs: int,
        caption_tag_dropout_rate: float,
        caption_prefix: Optional[str],
        caption_suffix: Optional[str],
        token_warmup_min: int,
        token_warmup_step: float | int,
        custom_attributes: Optional[Dict[str, Any]] = None,
        validation_seed: Optional[int] = None,
        validation_split: Optional[float] = 0.0,
        validation_split_num: int = 0,
        resize_interpolation: Optional[str] = None,
        recursive: bool = False,
        path_pattern: Optional[str] = None,
        repeat_by_folder_name: bool = False,
    ) -> None:
        self.image_dir = image_dir
        self.alpha_mask = alpha_mask if alpha_mask is not None else False
        self.num_repeats = num_repeats
        # Kohya-style folder repeats: a `{n}_...` directory component under
        # image_dir overrides num_repeats with n for the images inside it.
        self.repeat_by_folder_name = repeat_by_folder_name
        self.recursive = recursive
        # fnmatch glob applied to each image's path-relative-to-image_dir at
        # enumeration time; `*` / None / empty = no filtering.
        self.path_pattern = path_pattern or "*"
        self.sample_ratio = sample_ratio
        self.caption_separator = caption_separator
        self.keep_tokens_separator = keep_tokens_separator
        self.secondary_separator = secondary_separator
        self.enable_wildcard = enable_wildcard
        self.color_aug = color_aug
        self.flip_aug = flip_aug
        self.face_crop_aug_range = face_crop_aug_range
        self.random_crop = random_crop
        self.caption_dropout_rate = caption_dropout_rate
        self.caption_dropout_every_n_epochs = caption_dropout_every_n_epochs
        self.caption_tag_dropout_rate = caption_tag_dropout_rate
        self.caption_prefix = caption_prefix
        self.caption_suffix = caption_suffix

        self.token_warmup_min = token_warmup_min
        self.token_warmup_step = token_warmup_step

        self.custom_attributes = (
            custom_attributes if custom_attributes is not None else {}
        )

        self.img_count = 0

        self.validation_seed = validation_seed
        self.validation_split = validation_split
        self.validation_split_num = int(validation_split_num or 0)

        self.resize_interpolation = resize_interpolation


class DreamBoothSubset(BaseSubset):
    def __init__(
        self,
        image_dir: str,
        is_reg: bool,
        class_tokens: Optional[str],
        caption_extension: str,
        cache_info: bool,
        alpha_mask: bool,
        num_repeats,
        sample_ratio,
        caption_separator: str,
        keep_tokens_separator,
        secondary_separator,
        enable_wildcard,
        color_aug,
        flip_aug,
        face_crop_aug_range,
        random_crop,
        caption_dropout_rate,
        caption_dropout_every_n_epochs,
        caption_tag_dropout_rate,
        caption_prefix,
        caption_suffix,
        token_warmup_min,
        token_warmup_step,
        custom_attributes: Optional[Dict[str, Any]] = None,
        validation_seed: Optional[int] = None,
        validation_split: Optional[float] = 0.0,
        validation_split_num: int = 0,
        resize_interpolation: Optional[str] = None,
        mask_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        cond_cache_dir: Optional[str] = None,
        text_cache_dir: Optional[str] = None,
        latent_cache_dir: Optional[str] = None,
        recursive: bool = False,
        path_pattern: Optional[str] = None,
        repeat_by_folder_name: bool = False,
    ) -> None:
        assert image_dir is not None, "image_dir must be specified"

        super().__init__(
            image_dir,
            alpha_mask,
            num_repeats,
            sample_ratio,
            caption_separator,
            keep_tokens_separator,
            secondary_separator,
            enable_wildcard,
            color_aug,
            flip_aug,
            face_crop_aug_range,
            random_crop,
            caption_dropout_rate,
            caption_dropout_every_n_epochs,
            caption_tag_dropout_rate,
            caption_prefix,
            caption_suffix,
            token_warmup_min,
            token_warmup_step,
            custom_attributes=custom_attributes,
            validation_seed=validation_seed,
            validation_split=validation_split,
            validation_split_num=validation_split_num,
            resize_interpolation=resize_interpolation,
            recursive=recursive,
            path_pattern=path_pattern,
            repeat_by_folder_name=repeat_by_folder_name,
        )

        self.is_reg = is_reg
        self.class_tokens = class_tokens
        self.caption_extension = caption_extension
        if self.caption_extension and not self.caption_extension.startswith("."):
            self.caption_extension = "." + self.caption_extension
        self.cache_info = cache_info
        if mask_dir is None:
            mask_dir = _resolve_default_mask_dir()
            if mask_dir:
                logger.info(f"Auto-resolved mask_dir: {mask_dir}")
        self.mask_dir = mask_dir
        if mask_dir:
            self.alpha_mask = (
                True  # enable alpha mask pipeline when using separate mask files
            )
        # Optional redirect for VAE / text-encoder / PE caches. When set, all
        # caches for this subset live under cache_dir/ with stem-mirrored
        # filenames; when None (default) they sit alongside the source image.
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        # Parallel cache for the *condition* latent (cond≠target tasks, e.g.
        # colorization). Read-only at train time — do NOT makedirs it; the prep
        # step populates it. None → cond falls back to the target latent.
        self.cond_cache_dir = cond_cache_dir
        # Optional redirect for the *text-encoder* cache only (latents stay in
        # cache_dir). Read-only at train time; the prep step populates it with
        # re-encoded captions. None → TE caches come from cache_dir.
        self.text_cache_dir = text_cache_dir
        # Optional redirect for the *target VAE latent* cache only (TE/PE stay
        # in text_cache_dir/cache_dir). A prep step populates it with re-encoded
        # targets (e.g. white-balanced colorize targets); any latent still
        # missing at train time is encoded from the ORIGINAL image into this dir
        # (with a warning — see new_cache_latents). None → latents from cache_dir.
        self.latent_cache_dir = latent_cache_dir

    def __eq__(self, other) -> bool:
        if not isinstance(other, DreamBoothSubset):
            return NotImplemented
        return self.image_dir == other.image_dir
