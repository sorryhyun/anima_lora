import glob
import json
import logging
import os
import random
from typing import List, Optional, Sequence, Tuple

from tqdm import tqdm

from library.anima.text_strategies import LatentsCachingStrategy
from library.datasets import base as _base
from library.datasets.base import BaseDataset
from library.datasets.image_utils import (
    _assert_unique_stems,
    glob_images,
)
from library.io.walk import safe_walk
from library.datasets.subsets import (
    DreamBoothSubset,
    ImageInfo,
    filter_paths_by_glob,
    folder_repeat_count,
    split_train_val,
)

logger = logging.getLogger(__name__)


# Colorization (and other cond≠target tasks) pair a synthetic *condition* latent
# with each target. Staging (easycontrol_adapters/colorization/prep.py) decides
# which targets get a condition synthesized via the descriptor's
# `[staging].only_data_includes` / `exclude_data_includes` tag filters (net set =
# only − exclude). The tag scan below backs that staging-time selection (the colorize
# targets live under post_image_dataset/resized/ with no .txt sidecar, so tags are
# read from the caption master in image_dataset/). At *train* time the loader does
# NOT re-scan tags — it simply keeps the targets that actually have a cached cond
# latent (see the cond_cache_dir branch below), so whatever staging selected is
# paired out automatically.
_tag_stems_cache: dict = {}


def stems_with_any_tag(caption_master_dir: str, tags) -> frozenset:
    """Stems whose caption sidecar under ``caption_master_dir`` carries any of the
    standalone comma-separated ``tags`` (case-insensitive). Scanned once per
    (dir, tags) via ``os.walk`` (follows the symlinked master) and memoized; returns
    an empty set when the dir is absent or ``tags`` is empty."""
    wanted = frozenset(t.strip().lower() for t in (tags or ()) if t and t.strip())
    if not wanted:
        return frozenset()
    key = (caption_master_dir, wanted)
    cached = _tag_stems_cache.get(key)
    if cached is not None:
        return cached
    stems: set = set()
    if os.path.isdir(caption_master_dir):
        for dirpath, _dirnames, filenames in safe_walk(
            caption_master_dir, followlinks=True
        ):
            for name in filenames:
                if not name.endswith(".txt"):
                    continue
                try:
                    with open(os.path.join(dirpath, name), encoding="utf-8") as f:
                        line = f.readline()
                except (OSError, UnicodeDecodeError):
                    continue
                if any(t.strip().lower() in wanted for t in line.split(",")):
                    stems.add(os.path.splitext(name)[0])
    result = frozenset(stems)
    _tag_stems_cache[key] = result
    return result


def read_caption(img_path, caption_extension, enable_wildcard):
    """Read the caption sidecar for ``img_path``.

    Returns ``None`` when no sidecar file exists. An empty (or
    whitespace-only) caption file is a valid *explicit empty caption*
    (unconditional / style-LoRA training) and resolves to ``""`` rather than
    raising — callers treat ``""`` as a real caption, not a missing one.
    """
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
                if enable_wildcard:
                    caption = "\n".join(
                        [line.strip() for line in lines if line.strip() != ""]
                    )
                else:
                    caption = lines[0].strip() if lines else ""
            break
    return caption


class DreamBoothDataset(BaseDataset):
    IMAGE_INFO_CACHE_FILE = "metadata_cache.json"

    def __init__(
        self,
        subsets: Sequence[DreamBoothSubset],
        is_training_dataset: bool,
        batch_size: int,
        network_multiplier: float,
        prior_loss_weight: float,
        debug_dataset: bool,
        validation_split: float,
        validation_seed: Optional[int],
        resize_interpolation: Optional[str],
        validation_split_num: int = 0,
    ) -> None:
        super().__init__(network_multiplier, debug_dataset, resize_interpolation)

        self.batch_size = batch_size
        self.prior_loss_weight = prior_loss_weight
        self.latents_cache = None
        self.is_training_dataset = is_training_dataset
        self.validation_seed = validation_seed
        self.validation_split = validation_split
        self.validation_split_num = int(validation_split_num or 0)

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
                pattern = getattr(subset, "path_pattern", "*") or "*"
                if pattern != "*":
                    meta_paths = list(metas.keys())
                    keep = filter_paths_by_glob(meta_paths, subset.image_dir, pattern)
                    metas = {p: metas[p] for p, k in zip(meta_paths, keep) if k}
                    logger.info(
                        f"path_pattern={pattern!r} kept {len(metas)}/"
                        f"{len(meta_paths)} cached entries from "
                        f"{subset.image_dir}"
                    )
                img_paths = list(metas.keys())
                sizes: List[Optional[Tuple[int, int]]] = [
                    meta["resolution"] for meta in metas.values()
                ]
            else:
                recursive = getattr(subset, "recursive", False)
                img_paths = glob_images(subset.image_dir, "*", recursive=recursive)
                if recursive:
                    _assert_unique_stems(img_paths, source_label=subset.image_dir)
                pattern = getattr(subset, "path_pattern", "*") or "*"
                if pattern != "*":
                    keep = filter_paths_by_glob(img_paths, subset.image_dir, pattern)
                    pre_n = len(img_paths)
                    img_paths = [p for p, k in zip(img_paths, keep) if k]
                    logger.info(
                        f"path_pattern={pattern!r} kept {len(img_paths)}/"
                        f"{pre_n} images from {subset.image_dir}"
                    )
                sizes: List[Optional[Tuple[int, int]]] = [None] * len(img_paths)

                strategy = LatentsCachingStrategy.get_strategy()
                if strategy is not None:
                    logger.info("get image size from name of cache files")

                    if recursive:
                        npz_paths = glob.glob(
                            os.path.join(
                                subset.image_dir, "**", "*" + strategy.cache_suffix
                            ),
                            recursive=True,
                        )
                    else:
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

            if self.validation_split > 0.0 or self.validation_split_num > 0:
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
                        validation_split_num=self.validation_split_num,
                    )

            # sample_ratio shrinks only the training pool. The validation pool
            # is pinned by validation_split_num / validation_split — applying
            # sample_ratio there would silently reduce the user-requested val
            # count (e.g. PRESET=half + validation_split_num=16 → 8 items),
            # which is surprising for CMMD where val size controls estimator
            # variance.
            if (
                subset.sample_ratio < 1.0
                and len(img_paths) > 0
                and self.is_training_dataset
            ):
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

            # cond≠target task (cond_cache_dir set): keep only the targets that
            # actually have a cached condition latent — BEFORE reading captions.
            # Staging's tag filters ([staging].only_data_includes /
            # exclude_data_includes) synthesize a cond + text cache for only a
            # subset of the shared `resized` corpus, but image_dir still
            # enumerates the whole corpus. Filtering here (rather than after the
            # caption read, as this used to) keeps the TE-cache check below from
            # spamming "neither caption file nor class tokens" for the thousands
            # of unpaired targets we're about to drop anyway. A target with no
            # cond latent would also crash at load time (_load_cond_latent raises).
            if getattr(subset, "cond_cache_dir", None):
                from library.io.cache import discover_latents_by_stem

                cond_stems = set(discover_latents_by_stem(subset.cond_cache_dir))
                if not cond_stems:
                    raise FileNotFoundError(
                        f"No condition latents under {subset.cond_cache_dir!r} — run "
                        f"the cond prep step first (e.g. `make easycontrol-preprocess "
                        f"EASYADAPTER=colorize`)."
                    )
                pre = len(img_paths)
                kept = [
                    (p, s)
                    for p, s in zip(img_paths, sizes)
                    if os.path.splitext(os.path.basename(p))[0] in cond_stems
                ]
                img_paths, sizes = (
                    (list(t) for t in zip(*kept)) if kept else ([], [])
                )
                if len(img_paths) != pre:
                    logger.info(
                        f"colorize: kept {len(img_paths)}/{pre} targets with a cached "
                        f"condition latent (dropped {pre - len(img_paths)} unpaired) "
                        f"from {subset.image_dir}"
                    )

            if use_cached_info_for_subset:
                captions = [metas[p]["caption"] for p in img_paths]
                missing_captions = [
                    img_path
                    for img_path, caption in zip(img_paths, captions)
                    if caption is None or caption == ""
                ]
            else:
                # Subset may redirect TE caches to a separate `cache_dir` (e.g.
                # captions live in image_dataset/ but resized images live in
                # post_image_dataset/resized/ with caches in
                # post_image_dataset/lora/). When a TE cache exists for the
                # image's relative path, missing .txt sidecars are expected,
                # not an error — training reads the cached prompt embeddings.
                # Key by (rel_subdir, stem) to support nested cache layouts
                # where two images can share a stem in different subdirs.
                # text_cache_dir (when set) holds the TE caches instead of
                # cache_dir, so scan there to suppress missing-.txt warnings.
                cache_dir = getattr(subset, "text_cache_dir", None) or getattr(
                    subset, "cache_dir", None
                )
                te_suffix = "_anima_te.safetensors"
                te_cached_keys: set[tuple[str, str]] = set()
                if cache_dir and os.path.isdir(cache_dir):
                    cache_root = os.fspath(cache_dir)
                    for dirpath, _dirnames, filenames in os.walk(cache_root):
                        try:
                            rel_dir = os.path.relpath(dirpath, cache_root)
                        except ValueError:
                            rel_dir = ""
                        if rel_dir == ".":
                            rel_dir = ""
                        rel_dir = rel_dir.replace(os.sep, "/")
                        for name in filenames:
                            if name.endswith(te_suffix):
                                te_cached_keys.add(
                                    (rel_dir, name.removesuffix(te_suffix))
                                )

                image_root = getattr(subset, "image_dir", None)
                captions = []
                missing_captions = []
                for img_path in tqdm(img_paths, desc="read caption"):
                    cap_for_img = read_caption(
                        img_path, subset.caption_extension, subset.enable_wildcard
                    )
                    if image_root:
                        try:
                            img_rel_dir = os.path.relpath(
                                os.path.dirname(img_path), image_root
                            )
                        except ValueError:
                            img_rel_dir = ""
                        if img_rel_dir == ".":
                            img_rel_dir = ""
                        img_rel_dir = img_rel_dir.replace(os.sep, "/")
                    else:
                        img_rel_dir = ""
                    img_stem = os.path.splitext(os.path.basename(img_path))[0]
                    has_te_cache = (img_rel_dir, img_stem) in te_cached_keys
                    if cap_for_img is None and subset.class_tokens is None:
                        if not has_te_cache:
                            logger.warning(
                                f"neither caption file nor class tokens are found. use empty caption for {img_path}"
                            )
                            missing_captions.append(img_path)
                        captions.append("")
                    else:
                        if cap_for_img is None:
                            captions.append(subset.class_tokens)
                            if not has_te_cache:
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

            if _base._ARTIST_FILTER is not None:
                pre = len(img_paths)
                kept = [
                    (p, c, s)
                    for p, c, s in zip(img_paths, captions, sizes)
                    if _base._caption_has_artist(c, _base._ARTIST_FILTER)
                ]
                if kept:
                    img_paths, captions, sizes = (list(t) for t in zip(*kept))
                else:
                    img_paths, captions, sizes = [], [], []
                logger.info(
                    f"artist_filter='{_base._ARTIST_FILTER}' → kept {len(img_paths)}/{pre} "
                    f"images from {subset.image_dir}"
                )

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

            # Kohya-style folder repeats: a `{n}_...` directory component
            # under image_dir overrides num_repeats with n for the images
            # inside it; `0_...` drops them from the training pool. Training
            # only — the validation pool stays at 1 repeat (num_repeats above).
            repeats = [num_repeats] * len(img_paths)
            if self.is_training_dataset and getattr(
                subset, "repeat_by_folder_name", False
            ):
                repeats = [
                    n
                    if (n := folder_repeat_count(p, subset.image_dir)) is not None
                    else num_repeats
                    for p in img_paths
                ]
                n_overridden = sum(1 for r in repeats if r != num_repeats)
                n_dropped = sum(1 for r in repeats if r == 0)
                if n_overridden:
                    logger.info(
                        f"repeat_by_folder_name: {n_overridden}/{len(img_paths)} images "
                        f"take their repeat count from a {{n}}_* folder under "
                        f"{subset.image_dir}"
                        + (
                            f" ({n_dropped} in 0_* folders dropped)"
                            if n_dropped
                            else ""
                        )
                    )
                if n_dropped:
                    kept = [
                        (p, c, s, r)
                        for p, c, s, r in zip(img_paths, captions, sizes, repeats)
                        if r > 0
                    ]
                    if not kept:
                        logger.warning(
                            f"ignore subset with image_dir='{subset.image_dir}': "
                            f"all images sit in 0_* folders"
                        )
                        continue
                    img_paths, captions, sizes, repeats = (list(t) for t in zip(*kept))

            if subset.is_reg:
                num_reg_images += sum(repeats)
            else:
                num_train_images += sum(repeats)

            for img_path, caption, size, image_repeats in zip(
                img_paths, captions, sizes, repeats
            ):
                info = ImageInfo(
                    img_path,
                    image_repeats,
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
                    # Prefer the nested path that mirrors subset.image_dir →
                    # mask_dir; fall back to the flat layout so legacy
                    # masks/merged/ etc. caches keep working.
                    candidates: list[str] = []
                    image_dir = getattr(subset, "image_dir", None)
                    if image_dir:
                        try:
                            rel = os.path.relpath(os.path.dirname(img_path), image_dir)
                        except ValueError:
                            rel = ""
                        if rel and rel != "." and not rel.startswith(".."):
                            candidates.append(
                                os.path.join(subset.mask_dir, rel, f"{stem}_mask.png")
                            )
                    candidates.append(os.path.join(subset.mask_dir, f"{stem}_mask.png"))
                    for mask_path in candidates:
                        if os.path.exists(mask_path):
                            info.mask_path = mask_path
                            break
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
