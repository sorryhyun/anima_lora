import glob
import importlib
import json
import logging
import math
import os
import random
import re
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imagesize
import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image
from tqdm import tqdm

from library.runtime.device import clean_memory_on_device
from library.strategy_base import (
    LatentsCachingStrategy,
    TextEncoderOutputsCachingStrategy,
    TextEncodingStrategy,
    TokenizeStrategy,
)
from library.datasets.buckets import BucketBatchIndex, BucketManager
from library.datasets.image_utils import (
    resize_image,
    validate_interpolation_fn,
    IMAGE_TRANSFORMS,
    glob_images,
    is_disk_cached_latents_is_expected,
    load_image,
    trim_and_resize_if_required,
)
from library.datasets.subsets import (
    AugHelper,
    BaseSubset,
    DreamBoothSubset,
    ImageInfo,
    split_train_val,
)

logger = logging.getLogger(__name__)

HIGH_VRAM = False


def enable_high_vram():
    global HIGH_VRAM
    HIGH_VRAM = True


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        resolution: Optional[Tuple[int, int]],
        network_multiplier: float,
        debug_dataset: bool,
        resize_interpolation: Optional[str] = None,
    ) -> None:
        super().__init__()

        # width/height is used when enable_bucket==False
        self.width, self.height = (None, None) if resolution is None else resolution
        self.network_multiplier = network_multiplier
        self.debug_dataset = debug_dataset

        self.subsets: List[DreamBoothSubset] = []

        self.token_padding_disabled = False
        self.tag_frequency = {}
        self.XTI_layers = None
        self.token_strings = None

        self.enable_bucket = False
        self.bucket_manager: BucketManager = None  # not initialized
        self.min_bucket_reso = None
        self.max_bucket_reso = None
        self.bucket_reso_steps = None
        self.bucket_no_upscale = None
        self.bucket_info = None  # for metadata

        self.current_epoch: int = 0

        self.current_step: int = 0
        self.max_train_steps: int = 0
        self.seed: int = 0

        # augmentation
        self.aug_helper = AugHelper()

        self.image_transforms = IMAGE_TRANSFORMS

        self.custom_shuffle_caption_fn = (
            None  # optional: fn(flex_tokens) -> shuffled list
        )

        if resize_interpolation is not None:
            assert validate_interpolation_fn(resize_interpolation), (
                f'Resize interpolation "{resize_interpolation}" is not a valid interpolation'
            )
        self.resize_interpolation = resize_interpolation

        self.image_data: Dict[str, ImageInfo] = {}
        self.image_to_subset: Dict[str, DreamBoothSubset] = {}

        self.replacements = {}

        # Functional-loss inversion supervision (postfix-func).
        # Set via `dataset.inversion_dir = ...` after construction; None disables.
        self.inversion_dir: Optional[str] = None
        self.inversion_num_runs: int = 3

        # caching
        self.caching_mode = None  # None, 'latents', 'text'

        self.tokenize_strategy = None
        self.text_encoder_output_caching_strategy = None
        self.latents_caching_strategy = None

    def set_current_strategies(self):
        self.tokenize_strategy = TokenizeStrategy.get_strategy()
        self.text_encoder_output_caching_strategy = (
            TextEncoderOutputsCachingStrategy.get_strategy()
        )
        self.latents_caching_strategy = LatentsCachingStrategy.get_strategy()

    def adjust_min_max_bucket_reso_by_steps(
        self,
        resolution: Tuple[int, int],
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
    ) -> Tuple[int, int]:
        # make min/max bucket reso to be multiple of bucket_reso_steps
        if min_bucket_reso % bucket_reso_steps != 0:
            adjusted_min_bucket_reso = (
                min_bucket_reso - min_bucket_reso % bucket_reso_steps
            )
            logger.warning(
                "min_bucket_reso is adjusted to be multiple of bucket_reso_steps"
            )
            min_bucket_reso = adjusted_min_bucket_reso
        if max_bucket_reso % bucket_reso_steps != 0:
            adjusted_max_bucket_reso = (
                max_bucket_reso
                + bucket_reso_steps
                - max_bucket_reso % bucket_reso_steps
            )
            logger.warning(
                "max_bucket_reso is adjusted to be multiple of bucket_reso_steps"
            )
            max_bucket_reso = adjusted_max_bucket_reso

        assert min(resolution) >= min_bucket_reso, (
            "min_bucket_reso must be equal or less than resolution"
        )
        assert max(resolution) <= max_bucket_reso, (
            "max_bucket_reso must be equal or greater than resolution"
        )

        return min_bucket_reso, max_bucket_reso

    def set_seed(self, seed):
        self.seed = seed

    def set_caching_mode(self, mode):
        self.caching_mode = mode

    def set_current_epoch(self, epoch):
        if not self.current_epoch == epoch:
            if epoch > self.current_epoch:
                logger.info(
                    "epoch is incremented. current_epoch: {}, epoch: {}".format(
                        self.current_epoch, epoch
                    )
                )
                num_epochs = epoch - self.current_epoch
                for _ in range(num_epochs):
                    self.current_epoch += 1
                    self.shuffle_buckets()
            else:
                logger.warning(
                    "epoch is not incremented. current_epoch: {}, epoch: {}".format(
                        self.current_epoch, epoch
                    )
                )
                self.current_epoch = epoch

    def set_current_step(self, step):
        self.current_step = step

    def set_max_train_steps(self, max_train_steps):
        self.max_train_steps = max_train_steps

    def set_tag_frequency(self, dir_name, captions):
        frequency_for_dir = self.tag_frequency.get(dir_name, {})
        self.tag_frequency[dir_name] = frequency_for_dir
        for caption in captions:
            for tag in caption.split(","):
                tag = tag.strip()
                if tag:
                    tag = tag.lower()
                    frequency = frequency_for_dir.get(tag, 0)
                    frequency_for_dir[tag] = frequency + 1

    def disable_token_padding(self):
        self.token_padding_disabled = True

    def enable_XTI(self, layers=None, token_strings=None):
        self.XTI_layers = layers
        self.token_strings = token_strings

    def add_replacement(self, str_from, str_to):
        self.replacements[str_from] = str_to

    def process_caption(self, subset: BaseSubset, caption):
        if subset.caption_prefix:
            caption = subset.caption_prefix + " " + caption
        if subset.caption_suffix:
            caption = caption + " " + subset.caption_suffix

        is_drop_out = (
            subset.caption_dropout_rate > 0
            and random.random() < subset.caption_dropout_rate
        )
        is_drop_out = (
            is_drop_out
            or subset.caption_dropout_every_n_epochs > 0
            and self.current_epoch % subset.caption_dropout_every_n_epochs == 0
        )

        if is_drop_out:
            caption = ""
        else:
            # process wildcards
            if subset.enable_wildcard:
                # if caption is multiline, random choice one line
                if "\n" in caption:
                    caption = random.choice(caption.split("\n"))

                # wildcard is like '{aaa|bbb|ccc...}'
                # escape the curly braces like {{ or }}
                replacer1 = "⦅"
                replacer2 = "⦆"
                while replacer1 in caption or replacer2 in caption:
                    replacer1 += "⦅"
                    replacer2 += "⦆"

                caption = caption.replace("{{", replacer1).replace("}}", replacer2)

                # replace the wildcard
                def replace_wildcard(match):
                    return random.choice(match.group(1).split("|"))

                caption = re.sub(r"\{([^}]+)\}", replace_wildcard, caption)

                # unescape the curly braces
                caption = caption.replace(replacer1, "{").replace(replacer2, "}")
            else:
                # if caption is multiline, use the first line
                caption = caption.split("\n")[0]

            if (
                subset.shuffle_caption
                or subset.token_warmup_step > 0
                or subset.caption_tag_dropout_rate > 0
            ):
                fixed_tokens = []
                flex_tokens = []
                fixed_suffix_tokens = []
                if (
                    hasattr(subset, "keep_tokens_separator")
                    and subset.keep_tokens_separator
                    and subset.keep_tokens_separator in caption
                ):
                    fixed_part, flex_part = caption.split(
                        subset.keep_tokens_separator, 1
                    )
                    if subset.keep_tokens_separator in flex_part:
                        flex_part, fixed_suffix_part = flex_part.split(
                            subset.keep_tokens_separator, 1
                        )
                        fixed_suffix_tokens = [
                            t.strip()
                            for t in fixed_suffix_part.split(subset.caption_separator)
                            if t.strip()
                        ]

                    fixed_tokens = [
                        t.strip()
                        for t in fixed_part.split(subset.caption_separator)
                        if t.strip()
                    ]
                    flex_tokens = [
                        t.strip()
                        for t in flex_part.split(subset.caption_separator)
                        if t.strip()
                    ]
                else:
                    tokens = [
                        t.strip()
                        for t in caption.strip().split(subset.caption_separator)
                    ]
                    flex_tokens = tokens[:]
                    if subset.keep_tokens > 0:
                        fixed_tokens = flex_tokens[: subset.keep_tokens]
                        flex_tokens = tokens[subset.keep_tokens :]

                if subset.token_warmup_step < 1:
                    subset.token_warmup_step = math.floor(
                        subset.token_warmup_step * self.max_train_steps
                    )
                if (
                    subset.token_warmup_step
                    and self.current_step < subset.token_warmup_step
                ):
                    tokens_len = (
                        math.floor(
                            (self.current_step)
                            * (
                                (len(flex_tokens) - subset.token_warmup_min)
                                / (subset.token_warmup_step)
                            )
                        )
                        + subset.token_warmup_min
                    )
                    flex_tokens = flex_tokens[:tokens_len]

                def dropout_tags(tokens):
                    if subset.caption_tag_dropout_rate <= 0:
                        return tokens
                    filtered = []
                    for token in tokens:
                        if random.random() >= subset.caption_tag_dropout_rate:
                            filtered.append(token)
                    return filtered

                if subset.shuffle_caption:
                    if self.custom_shuffle_caption_fn is not None:
                        flex_tokens = self.custom_shuffle_caption_fn(flex_tokens)
                    else:
                        random.shuffle(flex_tokens)

                flex_tokens = dropout_tags(flex_tokens)

                caption = ", ".join(fixed_tokens + flex_tokens + fixed_suffix_tokens)

            # process secondary separator
            if subset.secondary_separator:
                caption = caption.replace(
                    subset.secondary_separator, subset.caption_separator
                )

            for str_from, str_to in self.replacements.items():
                if str_from == "":
                    # replace all
                    if isinstance(str_to, list):
                        caption = random.choice(str_to)
                    else:
                        caption = str_to
                else:
                    caption = caption.replace(str_from, str_to)

        return caption

    def get_input_ids(self, caption, tokenizer=None):
        if tokenizer is None:
            tokenizer = self.tokenizers[0]

        input_ids = tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_max_length,
            return_tensors="pt",
        ).input_ids

        if self.tokenizer_max_length > tokenizer.model_max_length:
            input_ids = input_ids.squeeze(0)
            iids_list = []
            if tokenizer.pad_token_id == tokenizer.eos_token_id:
                # v1
                for i in range(
                    1,
                    self.tokenizer_max_length - tokenizer.model_max_length + 2,
                    tokenizer.model_max_length - 2,
                ):  # (1, 152, 75)
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),
                        input_ids[i : i + tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )
                    ids_chunk = torch.cat(ids_chunk)
                    iids_list.append(ids_chunk)
            else:
                # v2 or SDXL
                for i in range(
                    1,
                    self.tokenizer_max_length - tokenizer.model_max_length + 2,
                    tokenizer.model_max_length - 2,
                ):
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),  # BOS
                        input_ids[i : i + tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )  # PAD or EOS
                    ids_chunk = torch.cat(ids_chunk)

                    if (
                        ids_chunk[-2] != tokenizer.eos_token_id
                        and ids_chunk[-2] != tokenizer.pad_token_id
                    ):
                        ids_chunk[-1] = tokenizer.eos_token_id
                    if ids_chunk[1] == tokenizer.pad_token_id:
                        ids_chunk[1] = tokenizer.eos_token_id

                    iids_list.append(ids_chunk)

            input_ids = torch.stack(iids_list)  # 3,77
        return input_ids

    def register_image(self, info: ImageInfo, subset: BaseSubset):
        self.image_data[info.image_key] = info
        self.image_to_subset[info.image_key] = subset

    def make_buckets(self, constant_token_buckets: bool = False):
        """
        bucketingbucket
        min_size and max_size are ignored when enable_bucket is False
        """
        logger.info("loading image sizes.")
        for info in tqdm(self.image_data.values()):
            if info.image_size is None:
                info.image_size = self.get_image_size(info.absolute_path)

        if self.enable_bucket:
            logger.info("make buckets")
        else:
            logger.info("prepare dataset")

        if self.enable_bucket:
            if self.bucket_manager is None:
                self.bucket_manager = BucketManager(
                    self.bucket_no_upscale,
                    (self.width, self.height),
                    self.min_bucket_reso,
                    self.max_bucket_reso,
                    self.bucket_reso_steps,
                )
                if not self.bucket_no_upscale:
                    self.bucket_manager.make_buckets(
                        constant_token_buckets=constant_token_buckets
                    )
                else:
                    logger.warning(
                        "min_bucket_reso and max_bucket_reso are ignored if bucket_no_upscale is set, because bucket reso is defined by image size automatically"
                    )

            img_ar_errors = []
            for image_info in self.image_data.values():
                image_width, image_height = image_info.image_size
                image_info.bucket_reso, image_info.resized_size, ar_error = (
                    self.bucket_manager.select_bucket(image_width, image_height)
                )

                img_ar_errors.append(abs(ar_error))

            self.bucket_manager.sort()
        else:
            self.bucket_manager = BucketManager(
                False, (self.width, self.height), None, None, None
            )
            self.bucket_manager.set_predefined_resos([(self.width, self.height)])
            for image_info in self.image_data.values():
                image_width, image_height = image_info.image_size
                image_info.bucket_reso, image_info.resized_size, _ = (
                    self.bucket_manager.select_bucket(image_width, image_height)
                )

        for image_info in self.image_data.values():
            for _ in range(image_info.num_repeats):
                self.bucket_manager.add_image(
                    image_info.bucket_reso, image_info.image_key
                )

        if self.enable_bucket:
            self.bucket_info = {"buckets": {}}
            logger.info("number of images (including repeats)")
            for i, (reso, bucket) in enumerate(
                zip(self.bucket_manager.resos, self.bucket_manager.buckets)
            ):
                count = len(bucket)
                if count > 0:
                    self.bucket_info["buckets"][i] = {
                        "resolution": reso,
                        "count": len(bucket),
                    }
                    logger.info(f"bucket {i}: resolution {reso}, count: {len(bucket)}")

            if len(img_ar_errors) == 0:
                mean_img_ar_error = 0  # avoid NaN
            else:
                img_ar_errors = np.array(img_ar_errors)
                mean_img_ar_error = np.mean(np.abs(img_ar_errors))
            self.bucket_info["mean_img_ar_error"] = mean_img_ar_error
            logger.info(f"mean ar error (without repeats): {mean_img_ar_error}")

        # Drop incomplete last batches to keep batch dim constant for torch.compile,
        # but only when no subset uses sample_ratio (where every image matters more).
        has_sample_ratio = any(s.sample_ratio < 1.0 for s in self.subsets)
        self.buckets_indices: List[BucketBatchIndex] = []
        for bucket_index, bucket in enumerate(self.bucket_manager.buckets):
            if has_sample_ratio:
                batch_count = int(math.ceil(len(bucket) / self.batch_size))
            else:
                batch_count = len(bucket) // self.batch_size
            for batch_index in range(batch_count):
                self.buckets_indices.append(
                    BucketBatchIndex(bucket_index, self.batch_size, batch_index)
                )

        self.shuffle_buckets()
        self._length = len(self.buckets_indices)

    def shuffle_buckets(self):
        # set random seed for this epoch
        random.seed(self.seed + self.current_epoch)

        random.shuffle(self.buckets_indices)
        self.bucket_manager.shuffle()

    def verify_bucket_reso_steps(self, min_steps: int):
        assert (
            self.bucket_reso_steps is None or self.bucket_reso_steps % min_steps == 0
        ), (
            f"bucket_reso_steps is {self.bucket_reso_steps}. it must be divisible by {min_steps}.\n"
            + f"bucket_reso_steps{self.bucket_reso_steps}{min_steps}"
        )

    def is_latent_cacheable(self):
        return all(
            [not subset.color_aug and not subset.random_crop for subset in self.subsets]
        )

    def is_text_encoder_output_cacheable(self, cache_supports_dropout: bool = False):
        return all(
            [
                not (
                    subset.caption_dropout_rate > 0
                    and not cache_supports_dropout
                    or subset.shuffle_caption
                    or subset.token_warmup_step > 0
                    or subset.caption_tag_dropout_rate > 0
                )
                for subset in self.subsets
            ]
        )

    def new_cache_latents(self, model: Any, accelerator: Accelerator):
        r"""
        a brand new method to cache latents. This method caches latents with caching strategy.
        normal cache_latents method is used by default, but this method is used when caching strategy is specified.
        """
        logger.info("caching latents with caching strategy.")
        caching_strategy = LatentsCachingStrategy.get_strategy()
        image_infos = list(self.image_data.values())

        # sort by resolution
        image_infos.sort(key=lambda info: info.bucket_reso[0] * info.bucket_reso[1])

        # split by resolution and some conditions
        class Condition:
            def __init__(self, reso, flip_aug, alpha_mask, random_crop):
                self.reso = reso
                self.flip_aug = flip_aug
                self.alpha_mask = alpha_mask
                self.random_crop = random_crop

            def __eq__(self, other):
                return (
                    other is not None
                    and self.reso == other.reso
                    and self.flip_aug == other.flip_aug
                    and self.alpha_mask == other.alpha_mask
                    and self.random_crop == other.random_crop
                )

        batch: List[ImageInfo] = []
        current_condition = None

        # support multiple-gpus
        num_processes = accelerator.num_processes
        process_index = accelerator.process_index

        # define a function to submit a batch to cache
        def submit_batch(batch, cond):
            for info in batch:
                if info.image is not None and isinstance(info.image, Future):
                    info.image = info.image.result()  # future to image
            caching_strategy.cache_batch_latents(
                model, batch, cond.flip_aug, cond.alpha_mask, cond.random_crop
            )

            # remove image from memory
            for info in batch:
                info.image = None

        # define ThreadPoolExecutor to load images in parallel
        max_workers = min(os.cpu_count(), len(image_infos))
        max_workers = max(1, max_workers // num_processes)  # consider multi-gpu
        max_workers = min(
            max_workers, caching_strategy.batch_size
        )  # max_workers should be less than batch_size
        executor = ThreadPoolExecutor(max_workers)

        try:
            # iterate images
            logger.info("caching latents...")
            for i, info in enumerate(tqdm(image_infos)):
                subset = self.image_to_subset[info.image_key]

                if info.latents_npz is not None:  # fine tuning dataset
                    continue

                # check disk cache exists and size of latents
                if caching_strategy.cache_to_disk:
                    info.latents_npz = caching_strategy.get_latents_npz_path(
                        info.absolute_path, info.image_size
                    )

                    # if the modulo of num_processes is not equal to process_index, skip caching
                    if i % num_processes != process_index:
                        continue

                    cache_available = caching_strategy.is_disk_cached_latents_expected(
                        info.bucket_reso,
                        info.latents_npz,
                        subset.flip_aug,
                        subset.alpha_mask,
                    )
                    if cache_available:  # do not add to batch
                        continue

                # if batch is not empty and condition is changed, flush the batch.
                condition = Condition(
                    info.bucket_reso,
                    subset.flip_aug,
                    subset.alpha_mask,
                    subset.random_crop,
                )
                if len(batch) > 0 and current_condition != condition:
                    submit_batch(batch, current_condition)
                    batch = []
                if condition != current_condition and HIGH_VRAM:
                    clean_memory_on_device(accelerator.device)

                if info.image is None:
                    # load image in parallel
                    info.image = executor.submit(
                        load_image, info.absolute_path, condition.alpha_mask
                    )

                batch.append(info)
                current_condition = condition

                # if number of data in batch is enough, flush the batch
                if len(batch) >= caching_strategy.batch_size:
                    submit_batch(batch, current_condition)
                    batch = []

            if len(batch) > 0:
                submit_batch(batch, current_condition)

        finally:
            executor.shutdown()

    def cache_latents(
        self,
        vae,
        vae_batch_size=1,
        cache_to_disk=False,
        is_main_process=True,
        file_suffix=".npz",
    ):
        logger.info("caching latents.")

        image_infos = list(self.image_data.values())

        # sort by resolution
        image_infos.sort(key=lambda info: info.bucket_reso[0] * info.bucket_reso[1])

        # split by resolution and some conditions
        class Condition:
            def __init__(self, reso, flip_aug, alpha_mask, random_crop):
                self.reso = reso
                self.flip_aug = flip_aug
                self.alpha_mask = alpha_mask
                self.random_crop = random_crop

            def __eq__(self, other):
                return (
                    self.reso == other.reso
                    and self.flip_aug == other.flip_aug
                    and self.alpha_mask == other.alpha_mask
                    and self.random_crop == other.random_crop
                )

        batches: List[Tuple[Any, List[ImageInfo]]] = []
        batch: List[ImageInfo] = []
        current_condition = None

        logger.info("checking cache validity...")
        for info in tqdm(image_infos):
            subset = self.image_to_subset[info.image_key]

            if info.latents_npz is not None:  # fine tuning dataset
                continue

            # check disk cache exists and size of latents
            if cache_to_disk:
                info.latents_npz = os.path.splitext(info.absolute_path)[0] + file_suffix
                if not is_main_process:  # store to info only
                    continue

                cache_available = is_disk_cached_latents_is_expected(
                    info.bucket_reso,
                    info.latents_npz,
                    subset.flip_aug,
                    subset.alpha_mask,
                )

                if cache_available:  # do not add to batch
                    continue

            # if batch is not empty and condition is changed, flush the batch.
            condition = Condition(
                info.bucket_reso, subset.flip_aug, subset.alpha_mask, subset.random_crop
            )
            if len(batch) > 0 and current_condition != condition:
                batches.append((current_condition, batch))
                batch = []

            batch.append(info)
            current_condition = condition

            # if number of data in batch is enough, flush the batch
            if len(batch) >= vae_batch_size:
                batches.append((current_condition, batch))
                batch = []
                current_condition = None

        if len(batch) > 0:
            batches.append((current_condition, batch))

        if cache_to_disk and not is_main_process:
            return

        from library.datasets.image_utils import (
            cache_batch_latents as _cache_batch_latents,
        )

        # iterate batches: batch doesn't have image, image will be loaded in cache_batch_latents and discarded
        logger.info("caching latents...")
        for condition, batch in tqdm(batches, smoothing=1, total=len(batches)):
            _cache_batch_latents(
                vae,
                cache_to_disk,
                batch,
                condition.flip_aug,
                condition.alpha_mask,
                condition.random_crop,
            )

    def new_cache_text_encoder_outputs(
        self, models: List[Any], accelerator: Accelerator
    ):
        r"""
        a brand new method to cache text encoder outputs. This method caches text encoder outputs with caching strategy.
        """
        tokenize_strategy = TokenizeStrategy.get_strategy()
        text_encoding_strategy = TextEncodingStrategy.get_strategy()
        caching_strategy = TextEncoderOutputsCachingStrategy.get_strategy()
        batch_size = caching_strategy.batch_size or self.batch_size

        logger.info("caching Text Encoder outputs with caching strategy.")
        image_infos = list(self.image_data.values())

        # split by resolution
        batches = []
        batch = []

        # support multiple-gpus
        num_processes = accelerator.num_processes
        process_index = accelerator.process_index

        logger.info("checking cache validity...")
        for i, info in enumerate(tqdm(image_infos)):
            # check disk cache exists and size of text encoder outputs
            if caching_strategy.cache_to_disk:
                te_out_npz = caching_strategy.get_outputs_npz_path(info.absolute_path)
                info.text_encoder_outputs_npz = te_out_npz

                if i % num_processes != process_index:
                    continue

                cache_available = caching_strategy.is_disk_cached_outputs_expected(
                    te_out_npz
                )
                if cache_available:
                    continue

            batch.append(info)

            if len(batch) >= batch_size:
                batches.append(batch)
                batch = []

        if len(batch) > 0:
            batches.append(batch)

        if len(batches) == 0:
            logger.info("no Text Encoder outputs to cache")
            return

        # iterate batches
        logger.info("caching Text Encoder outputs...")
        for batch in tqdm(batches, smoothing=1, total=len(batches)):
            caching_strategy.cache_batch_outputs(
                tokenize_strategy, models, text_encoding_strategy, batch
            )

    def cache_text_encoder_outputs(
        self,
        tokenizers,
        text_encoders,
        device,
        output_dtype,
        cache_to_disk=False,
        is_main_process=True,
    ):
        assert len(tokenizers) == 2, "only support SDXL"
        return self.cache_text_encoder_outputs_common(
            tokenizers,
            text_encoders,
            [device, device],
            output_dtype,
            [output_dtype],
            cache_to_disk,
            is_main_process,
        )

    def cache_text_encoder_outputs_sd3(
        self,
        tokenizer,
        text_encoders,
        devices,
        output_dtype,
        te_dtypes,
        cache_to_disk=False,
        is_main_process=True,
        batch_size=None,
    ):
        from library.datasets.image_utils import TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX_SD3

        return self.cache_text_encoder_outputs_common(
            [tokenizer],
            text_encoders,
            devices,
            output_dtype,
            te_dtypes,
            cache_to_disk,
            is_main_process,
            TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX_SD3,
            batch_size,
        )

    def cache_text_encoder_outputs_common(
        self,
        tokenizers,
        text_encoders,
        devices,
        output_dtype,
        te_dtypes,
        cache_to_disk=False,
        is_main_process=True,
        file_suffix=None,
        batch_size=None,
    ):
        from library.datasets.image_utils import TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX

        if file_suffix is None:
            file_suffix = TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX

        logger.info("caching text encoder outputs.")

        tokenize_strategy = TokenizeStrategy.get_strategy()

        if batch_size is None:
            batch_size = self.batch_size

        image_infos = list(self.image_data.values())

        logger.info("checking cache existence...")
        image_infos_to_cache = []
        for info in tqdm(image_infos):
            if cache_to_disk:
                te_out_npz = os.path.splitext(info.absolute_path)[0] + file_suffix
                info.text_encoder_outputs_npz = te_out_npz

                if not is_main_process:
                    continue

                if os.path.exists(te_out_npz):
                    continue

            image_infos_to_cache.append(info)

        if cache_to_disk and not is_main_process:
            return

        # prepare tokenizers and text encoders
        for text_encoder, device, te_dtype in zip(text_encoders, devices, te_dtypes):
            text_encoder.to(device)
            if te_dtype is not None:
                text_encoder.to(dtype=te_dtype)

        # create batch
        is_sd3 = len(tokenizers) == 1
        batch = []
        batches = []
        for info in image_infos_to_cache:
            if not is_sd3:
                input_ids1 = self.get_input_ids(info.caption, tokenizers[0])
                input_ids2 = self.get_input_ids(info.caption, tokenizers[1])
                batch.append((info, input_ids1, input_ids2))
            else:
                l_tokens, g_tokens, t5_tokens = tokenize_strategy.tokenize(info.caption)
                batch.append((info, l_tokens, g_tokens, t5_tokens))

            if len(batch) >= batch_size:
                batches.append(batch)
                batch = []

        if len(batch) > 0:
            batches.append(batch)

        # iterate batches: call text encoder and cache outputs for memory or disk
        logger.info("caching text encoder outputs...")
        # Note: SD/SDXL/SD3 specific batch caching functions are not included in this stripped version.
        # Anima uses new_cache_text_encoder_outputs with caching strategy instead.

    def get_image_size(self, image_path):
        if image_path.endswith(".jxl") or image_path.endswith(".JXL"):
            from library.jpeg_xl_util import get_jxl_size

            return get_jxl_size(image_path)
        image_size = imagesize.get(image_path)
        if image_size[0] <= 0:
            try:
                with Image.open(image_path) as img:
                    image_size = img.size
            except Exception as e:
                logger.warning(f"failed to get image size: {image_path}, error: {e}")
                image_size = (0, 0)
        return image_size

    def load_image_with_face_info(
        self, subset: BaseSubset, image_path: str, alpha_mask=False
    ):
        img = load_image(image_path, alpha_mask)

        face_cx = face_cy = face_w = face_h = 0
        if subset.face_crop_aug_range is not None:
            tokens = os.path.splitext(os.path.basename(image_path))[0].split("_")
            if len(tokens) >= 5:
                face_cx = int(tokens[-4])
                face_cy = int(tokens[-3])
                face_w = int(tokens[-2])
                face_h = int(tokens[-1])

        return img, face_cx, face_cy, face_w, face_h

    def crop_target(self, subset: BaseSubset, image, face_cx, face_cy, face_w, face_h):
        height, width = image.shape[0:2]
        if height == self.height and width == self.width:
            return image

        face_size = max(face_w, face_h)
        size = min(self.height, self.width)
        min_scale = max(self.height / height, self.width / width)
        min_scale = min(
            1.0, max(min_scale, size / (face_size * subset.face_crop_aug_range[1]))
        )
        max_scale = min(
            1.0, max(min_scale, size / (face_size * subset.face_crop_aug_range[0]))
        )
        if min_scale >= max_scale:
            scale = min_scale
        else:
            scale = random.uniform(min_scale, max_scale)

        nh = int(height * scale + 0.5)
        nw = int(width * scale + 0.5)
        assert nh >= self.height and nw >= self.width, (
            f"internal error. small scale {scale}, {width}*{height}"
        )
        image = resize_image(image, width, height, nw, nh, subset.resize_interpolation)
        face_cx = int(face_cx * scale + 0.5)
        face_cy = int(face_cy * scale + 0.5)
        height, width = nh, nw

        for axis, (target_size, length, face_p) in enumerate(
            zip((self.height, self.width), (height, width), (face_cy, face_cx))
        ):
            p1 = face_p - target_size // 2

            if subset.random_crop:
                range_ = max(length - face_p, face_p)
                p1 = (
                    p1
                    + (random.randint(0, range_) + random.randint(0, range_))
                    - range_
                )
            else:
                if subset.face_crop_aug_range[0] != subset.face_crop_aug_range[1]:
                    if face_size > size // 10 and face_size >= 40:
                        p1 = p1 + random.randint(-face_size // 20, +face_size // 20)

            p1 = max(0, min(p1, length - target_size))

            if axis == 0:
                image = image[p1 : p1 + target_size, :]
            else:
                image = image[:, p1 : p1 + target_size]

        return image

    def __len__(self):
        return self._length

    def _try_load_inversion_runs(self, image_abs_path: str) -> Optional[torch.Tensor]:
        """Load <stem>_inverted_run{0..N-1}.safetensors from self.inversion_dir.

        Returns a [N_runs, S, D] tensor, or None if any of the expected runs is missing
        (caller masks samples without inversions out of the functional loss).
        """
        if not self.inversion_dir:
            return None
        stem = os.path.splitext(os.path.basename(image_abs_path))[0]
        from safetensors.torch import load_file

        runs = []
        for i in range(self.inversion_num_runs):
            p = os.path.join(self.inversion_dir, f"{stem}_inverted_run{i}.safetensors")
            if not os.path.exists(p):
                return None
            sd = load_file(p)
            t = sd.get("crossattn_emb")
            if t is None:
                return None
            runs.append(t.float())
        return torch.stack(runs, dim=0)  # [N_runs, S, D]

    def __getitem__(self, index):
        bucket = self.bucket_manager.buckets[self.buckets_indices[index].bucket_index]
        bucket_batch_size = self.buckets_indices[index].bucket_batch_size
        image_index = self.buckets_indices[index].batch_index * bucket_batch_size

        if (
            self.caching_mode is not None
        ):  # return batch for latents/text encoder outputs caching
            return self.get_item_for_caching(bucket, bucket_batch_size, image_index)

        loss_weights = []
        captions = []
        input_ids_list = []
        latents_list = []
        alpha_mask_list = []
        images = []
        original_sizes_hw = []
        crop_top_lefts = []
        target_sizes_hw = []
        flippeds = []
        text_encoder_outputs_list = []
        custom_attributes = []
        inversion_runs_list: List[Optional[torch.Tensor]] = []

        for image_key in bucket[image_index : image_index + bucket_batch_size]:
            image_info = self.image_data[image_key]
            subset = self.image_to_subset[image_key]

            custom_attributes.append(subset.custom_attributes)

            loss_weights.append(self.prior_loss_weight if image_info.is_reg else 1.0)

            flipped = subset.flip_aug and random.random() < 0.5

            if image_info.latents is not None:
                original_size = image_info.latents_original_size
                crop_ltrb = image_info.latents_crop_ltrb
                if not flipped:
                    latents = image_info.latents
                    alpha_mask = image_info.alpha_mask
                else:
                    latents = image_info.latents_flipped
                    alpha_mask = (
                        None
                        if image_info.alpha_mask is None
                        else torch.flip(image_info.alpha_mask, [1])
                    )

                image = None
            elif image_info.latents_npz is not None:
                latents, original_size, crop_ltrb, flipped_latents, alpha_mask = (
                    self.latents_caching_strategy.load_latents_from_disk(
                        image_info.latents_npz, image_info.bucket_reso
                    )
                )
                if flipped:
                    latents = flipped_latents
                    alpha_mask = (
                        None if alpha_mask is None else alpha_mask[:, ::-1].copy()
                    )
                    del flipped_latents
                latents = torch.FloatTensor(latents)
                if alpha_mask is not None:
                    alpha_mask = torch.FloatTensor(alpha_mask)

                image = None
            else:
                img, face_cx, face_cy, face_w, face_h = self.load_image_with_face_info(
                    subset, image_info.absolute_path, subset.alpha_mask
                )
                im_h, im_w = img.shape[0:2]

                if self.enable_bucket:
                    img, original_size, crop_ltrb = trim_and_resize_if_required(
                        subset.random_crop,
                        img,
                        image_info.bucket_reso,
                        image_info.resized_size,
                        resize_interpolation=image_info.resize_interpolation,
                    )
                else:
                    if face_cx > 0:
                        img = self.crop_target(
                            subset, img, face_cx, face_cy, face_w, face_h
                        )
                    elif im_h > self.height or im_w > self.width:
                        assert subset.random_crop, (
                            "image too large, but cropping and bucketing are disabled"
                        )
                        if im_h > self.height:
                            p = random.randint(0, im_h - self.height)
                            img = img[p : p + self.height]
                        if im_w > self.width:
                            p = random.randint(0, im_w - self.width)
                            img = img[:, p : p + self.width]

                    im_h, im_w = img.shape[0:2]
                    assert im_h == self.height and im_w == self.width, (
                        "image size is small"
                    )

                    original_size = [im_w, im_h]
                    crop_ltrb = (0, 0, 0, 0)

                aug = self.aug_helper.get_augmentor(subset.color_aug)
                if aug is not None:
                    img_rgb = img[:, :, :3]
                    img_rgb = aug(image=img_rgb)["image"]
                    img[:, :, :3] = img_rgb

                if flipped:
                    img = img[:, ::-1, :].copy()

                if image_info.mask_path is not None:
                    from library.datasets.image_utils import load_mask_from_dir

                    alpha_mask = load_mask_from_dir(
                        os.path.dirname(image_info.mask_path),
                        image_info.absolute_path,
                        (img.shape[1], img.shape[0]),
                    )
                    if alpha_mask is None:
                        alpha_mask = torch.ones(
                            (img.shape[0], img.shape[1]), dtype=torch.float32
                        )
                    if flipped:
                        alpha_mask = torch.flip(alpha_mask, [1])
                elif subset.alpha_mask:
                    if img.shape[2] == 4:
                        alpha_mask = img[:, :, 3]
                        alpha_mask = alpha_mask.astype(np.float32) / 255.0
                        alpha_mask = torch.FloatTensor(alpha_mask)
                    else:
                        alpha_mask = torch.ones(
                            (img.shape[0], img.shape[1]), dtype=torch.float32
                        )
                else:
                    alpha_mask = None

                img = img[:, :, :3]

                latents = None
                image = self.image_transforms(img)
                del img

            images.append(image)
            latents_list.append(latents)
            alpha_mask_list.append(alpha_mask)

            target_size = (
                (image.shape[2], image.shape[1])
                if image is not None
                else (latents.shape[2] * 8, latents.shape[1] * 8)
            )

            if not flipped:
                crop_left_top = (crop_ltrb[0], crop_ltrb[1])
            else:
                crop_left_top = (target_size[0] - crop_ltrb[2], crop_ltrb[1])

            original_sizes_hw.append((int(original_size[1]), int(original_size[0])))
            crop_top_lefts.append((int(crop_left_top[1]), int(crop_left_top[0])))
            target_sizes_hw.append((int(target_size[1]), int(target_size[0])))
            flippeds.append(flipped)

            caption = image_info.caption

            tokenization_required = (
                self.text_encoder_output_caching_strategy is None
                or self.text_encoder_output_caching_strategy.is_partial
            )
            text_encoder_outputs = None
            input_ids = None

            if image_info.text_encoder_outputs is not None:
                text_encoder_outputs = image_info.text_encoder_outputs
            elif image_info.text_encoder_outputs_npz is not None:
                text_encoder_outputs = (
                    self.text_encoder_output_caching_strategy.load_outputs_npz(
                        image_info.text_encoder_outputs_npz
                    )
                )
            else:
                tokenization_required = True
            text_encoder_outputs_list.append(text_encoder_outputs)

            if tokenization_required:
                caption = self.process_caption(subset, image_info.caption)
                input_ids = [ids[0] for ids in self.tokenize_strategy.tokenize(caption)]

            input_ids_list.append(input_ids)
            captions.append(caption)

            if self.inversion_dir:
                inversion_runs_list.append(
                    self._try_load_inversion_runs(image_info.absolute_path)
                )
            else:
                inversion_runs_list.append(None)

        def none_or_stack_elements(tensors_list, converter):
            if (
                len(tensors_list) == 0
                or tensors_list[0] is None
                or len(tensors_list[0]) == 0
                or tensors_list[0][0] is None
            ):
                return None

            result = []
            for i in range(len(tensors_list[0])):
                tensors = [x[i] for x in tensors_list]
                if tensors[0] is None:
                    result.append(None)
                    continue
                if tensors[0].ndim == 0:
                    result.append(torch.stack([converter(x[i]) for x in tensors_list]))
                    continue

                min_len = min([len(x) for x in tensors])
                max_len = max([len(x) for x in tensors])

                if min_len == max_len:
                    result.append(torch.stack([converter(x) for x in tensors]))
                else:
                    tensors = [converter(x) for x in tensors]
                    if tensors[0].ndim == 1:
                        result.append(
                            torch.stack(
                                [
                                    (
                                        torch.nn.functional.pad(
                                            x, (0, max_len - x.shape[0])
                                        )
                                    )
                                    for x in tensors
                                ]
                            )
                        )
                    else:
                        result.append(
                            torch.stack(
                                [
                                    (
                                        torch.nn.functional.pad(
                                            x, (0, 0, 0, max_len - x.shape[0])
                                        )
                                    )
                                    for x in tensors
                                ]
                            )
                        )
            return result

        example = {}
        example["custom_attributes"] = custom_attributes
        example["loss_weights"] = torch.FloatTensor(loss_weights)
        example["text_encoder_outputs_list"] = none_or_stack_elements(
            text_encoder_outputs_list,
            lambda x: (
                x
                if isinstance(x, torch.Tensor)
                else torch.tensor(x, dtype=torch.float32)
            ),
        )
        example["input_ids_list"] = none_or_stack_elements(input_ids_list, lambda x: x)

        none_or_not = [x is None for x in alpha_mask_list]
        if all(none_or_not):
            example["alpha_masks"] = None
        elif any(none_or_not):
            for i in range(len(alpha_mask_list)):
                if alpha_mask_list[i] is None:
                    if images[i] is not None:
                        alpha_mask_list[i] = torch.ones(
                            (images[i].shape[1], images[i].shape[2]),
                            dtype=torch.float32,
                        )
                    else:
                        alpha_mask_list[i] = torch.ones(
                            (
                                latents_list[i].shape[1] * 8,
                                latents_list[i].shape[2] * 8,
                            ),
                            dtype=torch.float32,
                        )
            example["alpha_masks"] = torch.stack(alpha_mask_list)
        else:
            example["alpha_masks"] = torch.stack(alpha_mask_list)

        if images[0] is not None:
            images = torch.stack(images)
            images = images.to(memory_format=torch.contiguous_format).float()
        else:
            images = None
        example["images"] = images

        example["latents"] = (
            torch.stack(latents_list) if latents_list[0] is not None else None
        )
        example["captions"] = captions

        example["original_sizes_hw"] = torch.stack(
            [torch.LongTensor(x) for x in original_sizes_hw]
        )
        example["crop_top_lefts"] = torch.stack(
            [torch.LongTensor(x) for x in crop_top_lefts]
        )
        example["target_sizes_hw"] = torch.stack(
            [torch.LongTensor(x) for x in target_sizes_hw]
        )
        example["flippeds"] = flippeds

        example["network_multipliers"] = torch.FloatTensor(
            [self.network_multiplier] * len(captions)
        )

        # Inversion runs for functional-loss supervision (postfix-func).
        # If any sample in the batch has inversions loaded, stack them; samples
        # without matching inversions get zero-tensor placeholders and mask=False.
        valid_inversions = [t for t in inversion_runs_list if t is not None]
        if valid_inversions:
            ref_shape = valid_inversions[0].shape  # [N_runs, S, D]
            stacked = torch.stack(
                [
                    t if t is not None else torch.zeros(ref_shape, dtype=torch.float32)
                    for t in inversion_runs_list
                ],
                dim=0,
            )
            mask = torch.tensor(
                [t is not None for t in inversion_runs_list], dtype=torch.bool
            )
            example["inversion_runs"] = stacked  # [B, N_runs, S, D]
            example["inversion_mask"] = mask  # [B]
        else:
            example["inversion_runs"] = None
            example["inversion_mask"] = None

        if self.debug_dataset:
            example["image_keys"] = bucket[image_index : image_index + self.batch_size]
        return example

    def get_item_for_caching(self, bucket, bucket_batch_size, image_index):
        captions = []
        images = []
        input_ids1_list = []
        input_ids2_list = []
        absolute_paths = []
        resized_sizes = []
        bucket_reso = None
        flip_aug = None
        alpha_mask = None
        random_crop = None

        for image_key in bucket[image_index : image_index + bucket_batch_size]:
            image_info = self.image_data[image_key]
            subset = self.image_to_subset[image_key]

            if flip_aug is None:
                flip_aug = subset.flip_aug
                alpha_mask = subset.alpha_mask
                random_crop = subset.random_crop
                bucket_reso = image_info.bucket_reso
            else:
                assert flip_aug == subset.flip_aug, "flip_aug must be same in a batch"
                assert alpha_mask == subset.alpha_mask, (
                    "alpha_mask must be same in a batch"
                )
                assert random_crop == subset.random_crop, (
                    "random_crop must be same in a batch"
                )
                assert bucket_reso == image_info.bucket_reso, (
                    "bucket_reso must be same in a batch"
                )

            caption = image_info.caption

            if self.caching_mode == "latents":
                image = load_image(image_info.absolute_path)
            else:
                image = None

            if self.caching_mode == "text":
                input_ids1 = self.get_input_ids(caption, self.tokenizers[0])
                input_ids2 = self.get_input_ids(caption, self.tokenizers[1])
            else:
                input_ids1 = None
                input_ids2 = None

            captions.append(caption)
            images.append(image)
            input_ids1_list.append(input_ids1)
            input_ids2_list.append(input_ids2)
            absolute_paths.append(image_info.absolute_path)
            resized_sizes.append(image_info.resized_size)

        example = {}

        if images[0] is None:
            images = None
        example["images"] = images

        example["captions"] = captions
        example["input_ids1_list"] = input_ids1_list
        example["input_ids2_list"] = input_ids2_list
        example["absolute_paths"] = absolute_paths
        example["resized_sizes"] = resized_sizes
        example["flip_aug"] = flip_aug
        example["alpha_mask"] = alpha_mask
        example["random_crop"] = random_crop
        example["bucket_reso"] = bucket_reso
        return example


class DreamBoothDataset(BaseDataset):
    IMAGE_INFO_CACHE_FILE = "metadata_cache.json"

    def __init__(
        self,
        subsets: Sequence[DreamBoothSubset],
        is_training_dataset: bool,
        batch_size: int,
        resolution,
        network_multiplier: float,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        prior_loss_weight: float,
        debug_dataset: bool,
        validation_split: float,
        validation_seed: Optional[int],
        resize_interpolation: Optional[str],
    ) -> None:
        super().__init__(
            resolution, network_multiplier, debug_dataset, resize_interpolation
        )

        assert resolution is not None, "resolution is required"

        self.batch_size = batch_size
        self.size = min(self.width, self.height)
        self.prior_loss_weight = prior_loss_weight
        self.latents_cache = None
        self.is_training_dataset = is_training_dataset
        self.validation_seed = validation_seed
        self.validation_split = validation_split

        self.enable_bucket = enable_bucket
        if self.enable_bucket:
            min_bucket_reso, max_bucket_reso = self.adjust_min_max_bucket_reso_by_steps(
                resolution, min_bucket_reso, max_bucket_reso, bucket_reso_steps
            )
            self.min_bucket_reso = min_bucket_reso
            self.max_bucket_reso = max_bucket_reso
            self.bucket_reso_steps = bucket_reso_steps
            self.bucket_no_upscale = bucket_no_upscale
        else:
            self.min_bucket_reso = None
            self.max_bucket_reso = None
            self.bucket_reso_steps = None
            self.bucket_no_upscale = False

        def read_caption(img_path, caption_extension, enable_wildcard):
            base_name = os.path.splitext(img_path)[0]
            base_name_face_det = base_name
            tokens = base_name.split("_")
            if len(tokens) >= 5:
                base_name_face_det = "_".join(tokens[:-4])
            cap_paths = [
                base_name + caption_extension,
                base_name_face_det + caption_extension,
            ]

            caption = None
            for cap_path in cap_paths:
                if os.path.isfile(cap_path):
                    with open(cap_path, "rt", encoding="utf-8") as f:
                        try:
                            lines = f.readlines()
                        except UnicodeDecodeError as e:
                            logger.error("illegal char in file (not UTF-8)")
                            raise e
                        assert len(lines) > 0, "caption file is empty"
                        if enable_wildcard:
                            caption = "\n".join(
                                [line.strip() for line in lines if line.strip() != ""]
                            )
                        else:
                            caption = lines[0].strip()
                    break
            return caption

        def load_dreambooth_dir(subset: DreamBoothSubset):
            if not os.path.isdir(subset.image_dir):
                logger.warning(f"not directory: {subset.image_dir}")
                return [], [], []

            info_cache_file = os.path.join(subset.image_dir, self.IMAGE_INFO_CACHE_FILE)
            use_cached_info_for_subset = subset.cache_info
            if use_cached_info_for_subset:
                logger.info("using cached image info for this subset")
                if not os.path.isfile(info_cache_file):
                    logger.warning(
                        "image info file not found. You can ignore this warning if this is the first time to use this subset"
                        + ""
                    )
                    use_cached_info_for_subset = False

            if use_cached_info_for_subset:
                with open(info_cache_file, "r", encoding="utf-8") as f:
                    metas = json.load(f)
                img_paths = list(metas.keys())
                sizes: List[Optional[Tuple[int, int]]] = [
                    meta["resolution"] for meta in metas.values()
                ]
            else:
                img_paths = glob_images(subset.image_dir, "*")
                sizes: List[Optional[Tuple[int, int]]] = [None] * len(img_paths)

                strategy = LatentsCachingStrategy.get_strategy()
                if strategy is not None:
                    logger.info("get image size from name of cache files")

                    npz_paths = glob.glob(
                        os.path.join(subset.image_dir, "*" + strategy.cache_suffix)
                    )
                    npz_paths.sort(key=lambda item: item.rsplit("_", maxsplit=2)[0])
                    npz_path_index = 0

                    size_set_count = 0
                    for i, img_path in enumerate(tqdm(img_paths)):
                        stem_len = len(os.path.splitext(img_path)[0])
                        found = False
                        while npz_path_index < len(npz_paths):
                            if (
                                npz_paths[npz_path_index][:stem_len]
                                > img_path[:stem_len]
                            ):
                                break
                            if (
                                npz_paths[npz_path_index][:stem_len]
                                == img_path[:stem_len]
                            ):
                                found = True
                                break
                            npz_path_index += 1

                        if found:
                            w, h = strategy.get_image_size_from_disk_cache_path(
                                img_path, npz_paths[npz_path_index]
                            )
                        else:
                            w, h = None, None

                        if w is not None and h is not None:
                            sizes[i] = (w, h)
                            size_set_count += 1
                    logger.info(
                        f"set image size from cache files: {size_set_count}/{len(img_paths)}"
                    )

            if self.validation_split > 0.0:
                if subset.is_reg is True:
                    if self.is_training_dataset is False:
                        img_paths = []
                        sizes = []
                else:
                    img_paths, sizes = split_train_val(
                        img_paths,
                        sizes,
                        self.is_training_dataset,
                        self.validation_split,
                        self.validation_seed,
                    )

            if subset.sample_ratio < 1.0 and len(img_paths) > 0:
                sample_count = max(1, int(len(img_paths) * subset.sample_ratio))
                dataset = list(zip(img_paths, sizes))
                prevstate = random.getstate()
                random.seed(self.validation_seed)
                random.shuffle(dataset)
                random.setstate(prevstate)
                img_paths, sizes = zip(*dataset[:sample_count])
                img_paths = list(img_paths)
                sizes = list(sizes)
                logger.info(
                    f"sampled {sample_count} images (sample_ratio={subset.sample_ratio}) from {subset.image_dir}"
                )

            logger.info(
                f"found directory {subset.image_dir} contains {len(img_paths)} image files"
            )

            if use_cached_info_for_subset:
                captions = [meta["caption"] for meta in metas.values()]
                missing_captions = [
                    img_path
                    for img_path, caption in zip(img_paths, captions)
                    if caption is None or caption == ""
                ]
            else:
                captions = []
                missing_captions = []
                for img_path in tqdm(img_paths, desc="read caption"):
                    cap_for_img = read_caption(
                        img_path, subset.caption_extension, subset.enable_wildcard
                    )
                    if cap_for_img is None and subset.class_tokens is None:
                        logger.warning(
                            f"neither caption file nor class tokens are found. use empty caption for {img_path}"
                        )
                        captions.append("")
                        missing_captions.append(img_path)
                    else:
                        if cap_for_img is None:
                            captions.append(subset.class_tokens)
                            missing_captions.append(img_path)
                        else:
                            captions.append(cap_for_img)

            self.set_tag_frequency(os.path.basename(subset.image_dir), captions)

            if missing_captions:
                number_of_missing_captions = len(missing_captions)
                number_of_missing_captions_to_show = 5
                remaining_missing_captions = (
                    number_of_missing_captions - number_of_missing_captions_to_show
                )

                logger.warning(
                    f"No caption file found for {number_of_missing_captions} images. Training will continue without captions for these images. If class token exists, it will be used."
                )
                for i, missing_caption in enumerate(missing_captions):
                    if i >= number_of_missing_captions_to_show:
                        logger.warning(
                            missing_caption
                            + f"... and {remaining_missing_captions} more"
                        )
                        break
                    logger.warning(missing_caption)

            if not use_cached_info_for_subset and subset.cache_info:
                logger.info("cache image info for")
                sizes = [
                    self.get_image_size(img_path)
                    for img_path in tqdm(img_paths, desc="get image size")
                ]
                matas = {}
                for img_path, caption, size in zip(img_paths, captions, sizes):
                    matas[img_path] = {"caption": caption, "resolution": list(size)}
                with open(info_cache_file, "w", encoding="utf-8") as f:
                    json.dump(matas, f, ensure_ascii=False, indent=2)
                logger.info("cache image info done for")

            return img_paths, captions, sizes

        logger.info("prepare images.")
        num_train_images = 0
        num_reg_images = 0
        reg_infos: List[Tuple[ImageInfo, DreamBoothSubset]] = []
        for subset in subsets:
            num_repeats = subset.num_repeats if self.is_training_dataset else 1
            if num_repeats < 1:
                logger.warning(
                    f"ignore subset with image_dir='{subset.image_dir}': num_repeats is less than 1"
                )
                continue

            if subset in self.subsets:
                logger.warning(
                    f"ignore duplicated subset with image_dir='{subset.image_dir}': use the first one"
                )
                continue

            img_paths, captions, sizes = load_dreambooth_dir(subset)
            if len(img_paths) < 1:
                logger.warning(
                    f"ignore subset with image_dir='{subset.image_dir}': no images found"
                )
                continue

            if subset.is_reg:
                num_reg_images += num_repeats * len(img_paths)
            else:
                num_train_images += num_repeats * len(img_paths)

            for img_path, caption, size in zip(img_paths, captions, sizes):
                info = ImageInfo(
                    img_path,
                    num_repeats,
                    caption,
                    subset.is_reg,
                    img_path,
                    subset.caption_dropout_rate,
                )
                info.resize_interpolation = (
                    subset.resize_interpolation
                    if subset.resize_interpolation is not None
                    else self.resize_interpolation
                )
                if getattr(subset, "mask_dir", None):
                    stem = os.path.splitext(os.path.basename(img_path))[0]
                    mask_path = os.path.join(subset.mask_dir, f"{stem}_mask.png")
                    if os.path.exists(mask_path):
                        info.mask_path = mask_path
                if size is not None:
                    info.image_size = size
                if subset.is_reg:
                    reg_infos.append((info, subset))
                else:
                    self.register_image(info, subset)

            subset.img_count = len(img_paths)
            self.subsets.append(subset)

        images_split_name = "train" if self.is_training_dataset else "validation"
        logger.info(f"{num_train_images} {images_split_name} images with repeats.")

        self.num_train_images = num_train_images

        logger.info(f"{num_reg_images} reg images with repeats.")
        if num_train_images < num_reg_images:
            logger.warning("some of reg images are not used")

        if num_reg_images == 0:
            logger.warning("no regularization images")
        else:
            n = 0
            first_loop = True
            while n < num_train_images:
                for info, subset in reg_infos:
                    if first_loop:
                        self.register_image(info, subset)
                        n += info.num_repeats
                    else:
                        info.num_repeats += 1
                        n += 1
                    if n >= num_train_images:
                        break
                first_loop = False

        self.num_reg_images = num_reg_images


# behave as Dataset mock
class DatasetGroup(torch.utils.data.ConcatDataset):
    def __init__(self, datasets: Sequence[DreamBoothDataset]):
        self.datasets: List[DreamBoothDataset]

        super().__init__(datasets)

        self.image_data = {}
        self.num_train_images = 0
        self.num_reg_images = 0

        for dataset in datasets:
            self.image_data.update(dataset.image_data)
            self.num_train_images += dataset.num_train_images
            self.num_reg_images += dataset.num_reg_images

    def add_replacement(self, str_from, str_to):
        for dataset in self.datasets:
            dataset.add_replacement(str_from, str_to)

    def set_text_encoder_output_caching_strategy(
        self, strategy: TextEncoderOutputsCachingStrategy
    ):
        for dataset in self.datasets:
            dataset.set_text_encoder_output_caching_strategy(strategy)

    def enable_XTI(self, *args, **kwargs):
        for dataset in self.datasets:
            dataset.enable_XTI(*args, **kwargs)

    def cache_latents(
        self,
        vae,
        vae_batch_size=1,
        cache_to_disk=False,
        is_main_process=True,
        file_suffix=".npz",
    ):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.cache_latents(
                vae, vae_batch_size, cache_to_disk, is_main_process, file_suffix
            )

    def new_cache_latents(self, model: Any, accelerator: Accelerator):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.new_cache_latents(model, accelerator)
        accelerator.wait_for_everyone()

    def cache_text_encoder_outputs(
        self,
        tokenizers,
        text_encoders,
        device,
        weight_dtype,
        cache_to_disk=False,
        is_main_process=True,
    ):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.cache_text_encoder_outputs(
                tokenizers,
                text_encoders,
                device,
                weight_dtype,
                cache_to_disk,
                is_main_process,
            )

    def cache_text_encoder_outputs_sd3(
        self,
        tokenizer,
        text_encoders,
        device,
        output_dtype,
        te_dtypes,
        cache_to_disk=False,
        is_main_process=True,
        batch_size=None,
    ):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.cache_text_encoder_outputs_sd3(
                tokenizer,
                text_encoders,
                device,
                output_dtype,
                te_dtypes,
                cache_to_disk,
                is_main_process,
                batch_size,
            )

    def new_cache_text_encoder_outputs(
        self, models: List[Any], accelerator: Accelerator
    ):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.new_cache_text_encoder_outputs(models, accelerator)
        accelerator.wait_for_everyone()

    def set_caching_mode(self, caching_mode):
        for dataset in self.datasets:
            dataset.set_caching_mode(caching_mode)

    def verify_bucket_reso_steps(self, min_steps: int):
        for dataset in self.datasets:
            dataset.verify_bucket_reso_steps(min_steps)

    def get_resolutions(self) -> List[Tuple[int, int]]:
        return [(dataset.width, dataset.height) for dataset in self.datasets]

    def is_latent_cacheable(self) -> bool:
        return all([dataset.is_latent_cacheable() for dataset in self.datasets])

    def is_text_encoder_output_cacheable(
        self, cache_supports_dropout: bool = False
    ) -> bool:
        return all(
            [
                dataset.is_text_encoder_output_cacheable(cache_supports_dropout)
                for dataset in self.datasets
            ]
        )

    def set_current_strategies(self):
        for dataset in self.datasets:
            dataset.set_current_strategies()

    def set_current_epoch(self, epoch):
        for dataset in self.datasets:
            dataset.set_current_epoch(epoch)

    def set_current_step(self, step):
        for dataset in self.datasets:
            dataset.set_current_step(step)

    def set_max_train_steps(self, max_train_steps):
        for dataset in self.datasets:
            dataset.set_max_train_steps(max_train_steps)

    def disable_token_padding(self):
        for dataset in self.datasets:
            dataset.disable_token_padding()


class MinimalDataset(BaseDataset):
    def __init__(self, resolution, network_multiplier, debug_dataset=False):
        super().__init__(resolution, network_multiplier, debug_dataset)

        self.num_train_images = 0
        self.num_reg_images = 0
        self.datasets = [self]
        self.batch_size = 1

        self.subsets = [self]
        self.num_repeats = 1
        self.img_count = 1
        self.bucket_info = {}
        self.is_reg = False
        self.image_dir = "dummy"

    def verify_bucket_reso_steps(self, min_steps: int):
        pass

    def is_latent_cacheable(self) -> bool:
        return False

    def __len__(self):
        raise NotImplementedError

    def set_current_epoch(self, epoch):
        self.current_epoch = epoch

    def __getitem__(self, idx):
        raise NotImplementedError

    def get_resolutions(self) -> List[Tuple[int, int]]:
        return []


def load_arbitrary_dataset(args, tokenizer=None) -> MinimalDataset:
    module = ".".join(args.dataset_class.split(".")[:-1])
    dataset_class = args.dataset_class.split(".")[-1]
    module = importlib.import_module(module)
    dataset_class = getattr(module, dataset_class)
    train_dataset_group: MinimalDataset = dataset_class(
        tokenizer, args.max_token_length, args.resolution, args.debug_dataset
    )
    return train_dataset_group


def debug_dataset(train_dataset, show_input_ids=False):
    import cv2

    logger.info("Total dataset length (steps)")
    logger.info("`S` for next step, `E` for next epoch no. , Escape for exit.")

    epoch = 1
    while True:
        logger.info("")
        logger.info(f"epoch: {epoch}")

        steps = (epoch - 1) * len(train_dataset) + 1
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)

        k = 0
        for i, idx in enumerate(indices):
            train_dataset.set_current_epoch(epoch)
            train_dataset.set_current_step(steps)
            logger.info(f"steps: {steps} ({i + 1}/{len(train_dataset)})")

            example = train_dataset[idx]
            if example["latents"] is not None:
                logger.info(
                    f"sample has latents from npz file: {example['latents'].size()}"
                )
            for j, (ik, cap, lw, orgsz, crptl, trgsz, flpdz) in enumerate(
                zip(
                    example["image_keys"],
                    example["captions"],
                    example["loss_weights"],
                    example["original_sizes_hw"],
                    example["crop_top_lefts"],
                    example["target_sizes_hw"],
                    example["flippeds"],
                )
            ):
                logger.info(
                    f'{ik}, size: {train_dataset.image_data[ik].image_size}, loss weight: {lw}, caption: "{cap}", original size: {orgsz}, crop top left: {crptl}, target size: {trgsz}, flipped: {flpdz}'
                )
                if "network_multipliers" in example:
                    logger.info(
                        f"network multiplier: {example['network_multipliers'][j]}"
                    )
                if "custom_attributes" in example:
                    logger.info(f"custom attributes: {example['custom_attributes'][j]}")

                if example["images"] is not None:
                    im = example["images"][j]
                    logger.info(f"image size: {im.size()}")
                    im = ((im.numpy() + 1.0) * 127.5).astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))  # c,H,W -> H,W,c
                    im = im[:, :, ::-1]  # RGB -> BGR (OpenCV)

                    if "conditioning_images" in example:
                        cond_img = example["conditioning_images"][j]
                        logger.info(f"conditioning image size: {cond_img.size()}")
                        cond_img = ((cond_img.numpy() + 1.0) * 127.5).astype(np.uint8)
                        cond_img = np.transpose(cond_img, (1, 2, 0))
                        cond_img = cond_img[:, :, ::-1]
                        if os.name == "nt":
                            cv2.imshow("cond_img", cond_img)

                    if "alpha_masks" in example and example["alpha_masks"] is not None:
                        alpha_mask = example["alpha_masks"][j]
                        logger.info(f"alpha mask size: {alpha_mask.size()}")
                        alpha_mask = (alpha_mask.numpy() * 255.0).astype(np.uint8)
                        if os.name == "nt":
                            cv2.imshow("alpha_mask", alpha_mask)

                    if os.name == "nt":  # only windows
                        cv2.imshow("img", im)
                        k = cv2.waitKey()
                        cv2.destroyAllWindows()
                    if k == 27 or k == ord("s") or k == ord("e"):
                        break
            steps += 1

            if k == ord("e"):
                break
            if k == 27 or (example["images"] is None and i >= 8):
                k = 27
                break
        if k == 27:
            break

        epoch += 1


class collator_class:
    def __init__(self, epoch, step, dataset):
        self.current_epoch = epoch
        self.current_step = step
        self.dataset = dataset

    def __call__(self, examples):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            dataset = worker_info.dataset
        else:
            dataset = self.dataset

        dataset.set_current_epoch(self.current_epoch.value)
        dataset.set_current_step(self.current_step.value)
        return examples[0]


class LossRecorder:
    def __init__(self):
        self.loss_list: List[float] = []
        self.loss_total: float = 0.0

    def add(self, *, epoch: int, step: int, loss: float) -> None:
        if epoch == 0:
            self.loss_list.append(loss)
        else:
            while len(self.loss_list) <= step:
                self.loss_list.append(0.0)
            self.loss_total -= self.loss_list[step]
            self.loss_list[step] = loss
        self.loss_total += loss

    @property
    def moving_average(self) -> float:
        losses = len(self.loss_list)
        if losses == 0:
            return 0
        return self.loss_total / losses
