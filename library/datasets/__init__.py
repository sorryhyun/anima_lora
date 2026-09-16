# Dataset classes and utilities for Anima LoRA training.
# Re-exports all public names so `from library.datasets import X` works.
#
# Each name resolves lazily (PEP 562) on first access, so importing the package —
# or a light submodule like `library.datasets.buckets` — does not import torch
# (image_utils / base / dreambooth / cache / loss_recorder do). The GUI reads the
# bucket table on its UI thread and relies on this.

from __future__ import annotations

import importlib as _importlib

# export name -> dotted module that defines it
_ATTR_TO_MODULE: dict[str, str] = {
    # buckets
    "make_bucket_resolutions": "library.datasets.buckets",
    "BucketManager": "library.datasets.buckets",
    "BucketBatchIndex": "library.datasets.buckets",
    # subsets
    "split_train_val": "library.datasets.subsets",
    "ImageInfo": "library.datasets.subsets",
    "AugHelper": "library.datasets.subsets",
    "BaseSubset": "library.datasets.subsets",
    "DreamBoothSubset": "library.datasets.subsets",
    # image_utils
    "IMAGE_EXTENSIONS": "library.datasets.image_utils",
    "IMAGE_TRANSFORMS": "library.datasets.image_utils",
    "TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX": "library.datasets.image_utils",
    "TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX_SD3": "library.datasets.image_utils",
    "load_image": "library.datasets.image_utils",
    "trim_and_resize_if_required": "library.datasets.image_utils",
    "load_images_and_masks_for_caching": "library.datasets.image_utils",
    "cache_batch_latents": "library.datasets.image_utils",
    "save_text_encoder_outputs_to_disk": "library.datasets.image_utils",
    "load_text_encoder_outputs_from_disk": "library.datasets.image_utils",
    "glob_images": "library.datasets.image_utils",
    "glob_images_pathlib": "library.datasets.image_utils",
    "is_disk_cached_latents_is_expected": "library.datasets.image_utils",
    "ImageLoadingDataset": "library.datasets.image_utils",
    # base / concrete datasets
    "BaseDataset": "library.datasets.base",
    "DreamBoothDataset": "library.datasets.dreambooth",
    "DatasetGroup": "library.datasets.group",
    "MinimalDataset": "library.datasets.minimal",
    "load_arbitrary_dataset": "library.datasets.minimal",
    "debug_dataset": "library.datasets.minimal",
    "collator_class": "library.datasets.collator",
    # cache (general train-cache reader)
    "BucketBatchSampler": "library.datasets.cache",
    "CachedDataset": "library.datasets.cache",
    "LossRecorder": "library.training.loss_recorder",
}


def __getattr__(name: str):
    module = _ATTR_TO_MODULE.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(_importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = list(_ATTR_TO_MODULE.keys())
