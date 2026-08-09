import glob
import logging
import math
import os
import random
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import imagesize
import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image
from tqdm import tqdm

from library.runtime.device import clean_memory_on_device
from library.anima.text_strategies import (
    LatentsCachingStrategy,
    TextEncoderOutputsCachingStrategy,
    TextEncodingStrategy,
    TokenizeStrategy,
)
from library.datasets.buckets import BucketBatchIndex, BucketManager
from library.datasets.image_utils import (
    validate_interpolation_fn,
    IMAGE_TRANSFORMS,
    is_disk_cached_latents_is_expected,
    load_image,
    trim_and_resize_if_required,
)
from library.datasets.subsets import (
    AugHelper,
    BaseSubset,
    DreamBoothSubset,
    ImageInfo,
)

logger = logging.getLogger(__name__)

HIGH_VRAM = False

# Module-level artist filter — set from train.py (`--artist_filter`). When non-empty,
# `load_dreambooth_dir` keeps only images whose caption contains the `@<artist>` tag.
_ARTIST_FILTER: Optional[str] = None


def set_artist_filter(artist: Optional[str]) -> None:
    global _ARTIST_FILTER
    if artist is None or artist == "":
        _ARTIST_FILTER = None
        return
    _ARTIST_FILTER = artist if artist.startswith("@") else f"@{artist}"


def _caption_has_artist(caption: Optional[str], needle: str) -> bool:
    if not caption:
        return False
    needle_lc = needle.lower()
    for tag in caption.split(","):
        if tag.strip().lower() == needle_lc:
            return True
    return False


def enable_high_vram():
    global HIGH_VRAM
    HIGH_VRAM = True


def none_or_stack_elements(tensors_list, converter):
    """Stack a list of per-sample tuples element-wise, or return None.

    Each entry of ``tensors_list`` is a per-sample tuple/list of tensors (e.g.
    the multi-output of a text encoder). Returns a list whose i-th entry stacks
    the i-th element across all samples (right-padding ragged lengths with
    zeros), or None when the batch carries no such outputs. Pure helper — closes
    over nothing, hoisted out of ``BaseDataset.__getitem__``.
    """
    if (
        len(tensors_list) == 0
        or tensors_list[0] is None
        or len(tensors_list[0]) == 0
        or all(t is None for t in tensors_list[0])
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
                            (torch.nn.functional.pad(x, (0, max_len - x.shape[0])))
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


@dataclass
class SidecarSpec:
    """One per-image sidecar channel: a loader plus its batch collation policy.

    Generalizes the load/append/stack/None dance that used to be hand-copied
    per feature (inversion runs, BYG tuples, REPA PE). Register via
    ``BaseDataset.register_sidecar``; ``__getitem__`` then owns the per-sample
    loop and collation for every registered spec.

    Policies:
      - ``"all_or_nothing"``: ``example[out_key]`` is the stacked ``[B, ...]``
        tensor when every sample in the batch loaded (and, with
        ``uniform_shape``, all shapes match); ``None`` otherwise.
      - ``"masked"``: missing samples get zero placeholders and
        ``example[mask_key]`` carries a ``[B]`` bool validity mask; both keys
        are ``None`` when no sample loaded.
      - ``"dict"``: the loader returns a ``dict[str, Tensor]`` per sample;
        every key present in *all* samples' dicts is stacked into
        ``example[f"{out_key}{key}"]``. Nothing is emitted (keys absent, not
        ``None``) unless all samples loaded.
    """

    name: str
    loader: Callable[[ImageInfo], Optional[Any]]
    out_key: str
    policy: str = "all_or_nothing"
    # Name of the dataset attribute gating this channel: truthy → active,
    # ``None`` → always on. A plain attribute name (not a closure) keeps the
    # spec picklable so the dataset can be shipped to DataLoader workers under
    # Windows/`spawn` (a stored lambda raised "Can't get local object
    # ...<locals>.<lambda>" at first iteration).
    enabled_attr: Optional[str] = None
    mask_key: Optional[str] = None
    uniform_shape: bool = False


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        network_multiplier: float,
        debug_dataset: bool,
        resize_interpolation: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.network_multiplier = network_multiplier
        self.debug_dataset = debug_dataset

        self.subsets: List[DreamBoothSubset] = []

        self.token_padding_disabled = False
        self.tag_frequency = {}
        self.XTI_layers = None
        self.token_strings = None

        self.bucket_manager: BucketManager = None  # not initialized
        self.bucket_info = None  # for metadata

        self.current_epoch: int = 0

        self.current_step: int = 0
        self.max_train_steps: int = 0
        self.seed: int = 0

        self.aug_helper = AugHelper()

        self.image_transforms = IMAGE_TRANSFORMS

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

        # BYG unpaired-editing per-image text conditionings. Set via
        # `dataset.byg_text_dir = ...` after construction; None disables. Loads
        # <stem>_byg.safetensors holding the 4 role embeddings + masks (built by
        # scripts/byg/build_edit_tuples.py).
        self.byg_text_dir: Optional[str] = None
        self._byg_roles = (
            "src_caption",
            "tgt_caption",
            "instruction",
            "reverse_instruction",
        )

        # REPA v2 PE feature loading. Set via `dataset.load_repa_pe = True`
        # (+ `repa_pe_encoder`) after construction; off disables. Loads the
        # cached {stem}_anima_{encoder}.safetensors patch tokens into
        # batch["repa_pe_features"] for REPAMethodAdapter. Sidecar resolution
        # walks a fallback chain (TE-cache dir → subset latent-cache dir →
        # image dir) — see _repa_pe_sidecar_candidates.
        self.load_repa_pe: bool = False
        self.repa_pe_encoder: str = "pe_spatial"

        # sigma_lowres Phase 1b (σ-conditional low-res training). Set via
        # ``enable_sigma_demote(native, demote)`` after construction; None
        # disables. When set, each batch carries ``demoted_latents`` — the
        # demote-tier sibling latent stored as a ``demoted_{H}x{W}`` key inside
        # the native npz (``make preprocess-demote``) — and the trainer swaps
        # it in when the step's σ draw clears the gate. Train-group only; the
        # validation group stays native so val loss remains arm-comparable.
        self._sigma_demote: Optional[Tuple[int, int]] = None
        # Warned-route set, NOT a single bool: each route reports its own
        # missing emit (see ``_demote_warn_once``).
        self._sigma_demote_warned: set = set()
        # Secondary demote route (E16 stacked router: e.g. 1024→768 on the
        # measured window, priority over the primary 1024→896 rule). Set via
        # ``enable_sigma_demote2``; batches then also carry
        # ``demoted_latents2``.
        self._sigma_demote2: Optional[Tuple[int, int]] = None
        # One-slot (path, NpzFile) memo shared by both routes' loaders.
        self._demote_npz_cache = None

        # Soft-tokens contrastive negatives. When a sampler is attached via
        # ``setup_contrastive_negatives`` each example carries
        # ``neg_crossattn_emb`` of shape (B, k, S, D): k cached text embeddings
        # of *unrelated* images, used as InfoNCE negatives. Reuses the
        # IdentityPairSampler's ``shuffled`` policy (Phase 1). Decoupled from the
        # VAE target — an unrelated stem's cached feature replaces the target's,
        # but the swapped feature is the text embedding, not a vision feature. See
        # docs/proposal/soft_tokens_contrastive.md.
        self.contrastive_neg_sampler = None  # IdentityPairSampler | None
        self.contrastive_neg_k: int = 1
        self.contrastive_neg_mode: str = "shuffled"

        # Per-image sidecar registry: each spec bundles a loader with its batch
        # collation policy, so a new sidecar channel is one register_sidecar()
        # call instead of a hand-copied loop/stack/None dance in __getitem__.
        # The toggles above (inversion_dir / byg_text_dir / load_repa_pe) stay
        # the public enable surface; `enabled` closures read them live. The
        # soft-tokens contrastive negatives stay bespoke — they are drawn by a
        # sampler from *other* stems, not loaded from a per-image file.
        self._sidecar_specs: List[SidecarSpec] = []
        # Loaders are bound methods (not lambdas) and `enabled_attr` names a
        # dataset attribute rather than capturing one in a closure, so every
        # spec pickles cleanly for Windows/`spawn` DataLoader workers.
        self.register_sidecar(
            SidecarSpec(
                name="inversion_runs",
                loader=self._try_load_inversion_runs,
                out_key="inversion_runs",
                policy="masked",
                mask_key="inversion_mask",
                enabled_attr="inversion_dir",
            )
        )
        self.register_sidecar(
            SidecarSpec(
                name="byg",
                loader=self._try_load_byg_tuple,
                out_key="byg_",
                policy="dict",
                enabled_attr="byg_text_dir",
            )
        )
        self.register_sidecar(
            SidecarSpec(
                name="sigma_demote",
                loader=self._try_load_demoted_latent,
                out_key="demoted_latents",
                policy="all_or_nothing",
                # Same-bucket samples share (W, H) → same demoted grid, so
                # uniform shapes are structural; the guard only fires if a
                # stale emit left mismatched keys, degrading that batch to
                # native instead of crashing the epoch.
                uniform_shape=True,
                enabled_attr="_sigma_demote",
            )
        )
        self.register_sidecar(
            SidecarSpec(
                name="sigma_demote2",
                loader=self._try_load_demoted_latent2,
                out_key="demoted_latents2",
                policy="all_or_nothing",
                uniform_shape=True,
                enabled_attr="_sigma_demote2",
            )
        )
        self.register_sidecar(
            SidecarSpec(
                name="repa_pe",
                loader=self._try_load_repa_pe,
                out_key="repa_pe_features",
                policy="all_or_nothing",
                # All samples in a bucket share the latent resolution → same
                # aspect → same encoder bucket, so equal token counts are the
                # norm; the shape guard skips the term on the rare
                # aspect-rounding edge instead of crashing the epoch.
                uniform_shape=True,
                enabled_attr="load_repa_pe",
            )
        )

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
            if subset.enable_wildcard:
                if "\n" in caption:
                    caption = random.choice(caption.split("\n"))

                # wildcard is like '{aaa|bbb|ccc...}'; escape literal {{ }}
                replacer1 = "⦅"
                replacer2 = "⦆"
                while replacer1 in caption or replacer2 in caption:
                    replacer1 += "⦅"
                    replacer2 += "⦆"

                caption = caption.replace("{{", replacer1).replace("}}", replacer2)

                def replace_wildcard(match):
                    return random.choice(match.group(1).split("|"))

                caption = re.sub(r"\{([^}]+)\}", replace_wildcard, caption)

                caption = caption.replace(replacer1, "{").replace(replacer2, "}")
            else:
                caption = caption.split("\n")[0]

            if subset.token_warmup_step > 0 or subset.caption_tag_dropout_rate > 0:
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

                flex_tokens = dropout_tags(flex_tokens)

                caption = ", ".join(fixed_tokens + flex_tokens + fixed_suffix_tokens)

            if subset.secondary_separator:
                caption = caption.replace(
                    subset.secondary_separator, subset.caption_separator
                )

            for str_from, str_to in self.replacements.items():
                if str_from == "":
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

    def make_buckets(
        self,
        target_res=None,
    ):
        """Assign every image to its nearest bucket resolution.

        Free-fit is the only resize mode: the predefined bucket set is the **union
        of the distinct on-disk resized sizes** (read here as ``info.image_size``),
        so each latent exact-matches its own (W, H) and nothing AR-snaps at load.
        The on-disk caches are the source of truth for which shapes/tiers are
        present; ``target_res`` is preprocess-only and inert here. The compile
        token-family budget is derived from the populated buckets (train.py) — all
        within one tier's band, so ``compile_dynamic_seq`` keeps them at one graph.
        """
        logger.info("loading image sizes.")
        for info in tqdm(self.image_data.values()):
            if info.image_size is None:
                info.image_size = self.get_image_size(info.absolute_path)

        logger.info("make buckets")

        # Remember the tier set so a later rebuild (e.g. restrict_to_byg_tuples)
        # re-buckets identically.
        self._target_res = target_res

        if self.bucket_manager is None:
            self.bucket_manager = BucketManager()
            freefit_resos = {
                tuple(info.image_size)
                for info in self.image_data.values()
                if info.image_size is not None
            }
            self.bucket_manager.make_buckets(freefit_resos=freefit_resos)

        img_ar_errors = []
        for image_info in self.image_data.values():
            image_width, image_height = image_info.image_size
            image_info.bucket_reso, image_info.resized_size, ar_error = (
                self.bucket_manager.select_bucket(image_width, image_height)
            )

            img_ar_errors.append(abs(ar_error))

        self.bucket_manager.sort()

        for image_info in self.image_data.values():
            for _ in range(image_info.num_repeats):
                self.bucket_manager.add_image(
                    image_info.bucket_reso, image_info.image_key
                )

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

        self._preload_alpha_masks()

    def _preload_alpha_masks(self):
        """Load mask PNGs into memory once as uint8 [H, W] tensors at
        bucket_reso, so the dataloader hot path doesn't re-decode + resize a
        PNG on every fetch. Mask files generated by `make mask` are already at
        post-resize resolution (matches bucket_reso), so no resize is needed
        in the common case; we only resize as a safety net for stale masks.
        Skipped for subsets with random_crop=True since image size varies per
        fetch in that case.
        """
        targets = [
            info
            for info in self.image_data.values()
            if info.mask_path is not None
            and not self.image_to_subset[info.image_key].random_crop
        ]
        if not targets:
            return
        logger.info(f"preloading {len(targets)} alpha masks into memory...")
        n_resized = 0
        n_missing = 0
        for info in tqdm(targets, desc="preload masks"):
            if not os.path.exists(info.mask_path):
                n_missing += 1
                continue
            mask = Image.open(info.mask_path).convert("L")
            target_w, target_h = info.bucket_reso  # (W, H)
            if (mask.width, mask.height) != (target_w, target_h):
                mask = mask.resize((target_w, target_h), Image.NEAREST)
                n_resized += 1
            info.preloaded_alpha_mask = torch.from_numpy(np.array(mask, dtype=np.uint8))
        if n_missing:
            logger.warning(f"  {n_missing} mask files missing on disk")
        if n_resized:
            logger.info(
                f"  {n_resized} masks needed runtime resize (size != bucket_reso)"
            )

    def shuffle_buckets(self):
        random.seed(self.seed + self.current_epoch)

        random.shuffle(self.buckets_indices)
        self.bucket_manager.shuffle()
        self._largest_bucket_first()

    def _largest_bucket_first(self):
        """Pin one batch of EACH token-count family to the front of the epoch
        order, largest-token-first.

        Under native-shape bucketing each distinct token count traces its own
        ``torch.compile`` block graph. The first time a graph runs it both peaks
        the caching-allocator activations AND loads its inductor kernel module +
        cuBLAS/cuDNN/flash workspaces into the CUDA context (the latter is
        ``nvidia-smi``-visible but invisible to ``torch.cuda.memory_reserved``).
        Front-loading only the single biggest bucket warms one graph; the others
        compile lazily as the shuffle reaches them, so context VRAM ramps up
        ~mid-epoch then plateaus. Warming one batch of every family up front
        forces all graphs to compile in the first few steps — peak (allocator +
        context) lands at start, so a too-tight budget fails fast instead of
        creeping up mid-run. See [[project_compile_context_vram_climb]].

        Equal token count ⟺ equal pixel area (each native bucket exactly fills
        its count), so distinct areas == distinct graph families. Only the
        leading batches are reordered; the rest of the epoch stays shuffled.
        """
        if not self.buckets_indices:
            return
        # resos are (W, H); pixel area uniquely keys each token-count family.
        if getattr(self, "_warmup_bucket_indices", None) is None:
            resos = self.bucket_manager.resos
            by_area: dict[int, int] = {}
            for bbi in self.buckets_indices:
                bi = bbi.bucket_index
                area = resos[bi][0] * resos[bi][1]
                by_area.setdefault(area, bi)  # one representative per family
            # Largest-token (largest-area) family first.
            self._warmup_bucket_indices = [
                by_area[a] for a in sorted(by_area, reverse=True)
            ]
        # Move one batch of each family to the front. Iterate smallest-first and
        # insert at 0 so the largest family ends up at index 0 (worst case first).
        for target in reversed(self._warmup_bucket_indices):
            for i, bbi in enumerate(self.buckets_indices):
                if bbi.bucket_index == target:
                    if i:
                        self.buckets_indices.insert(0, self.buckets_indices.pop(i))
                    break

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
                    or subset.token_warmup_step > 0
                    or subset.caption_tag_dropout_rate > 0
                )
                for subset in self.subsets
            ]
        )

    def is_latents_cache_complete(self) -> bool:
        """True iff every image already has a valid on-disk latents cache.

        Read-only probe (no model, no GPU) used by the trainer to decide
        whether the VAE needs loading at all. Mirrors the per-file skip
        condition inside ``new_cache_latents``; honours ``skip_cache_check``
        via the strategy's ``is_disk_cached_latents_expected``.
        """
        caching_strategy = LatentsCachingStrategy.get_strategy()
        if caching_strategy is None or not caching_strategy.cache_to_disk:
            return False
        for info in self.image_data.values():
            if info.latents_npz is not None:  # fine tuning dataset: pre-set path
                continue
            subset = self.image_to_subset[info.image_key]
            # latent_cache_dir (when set) redirects only the target-latent
            # cache — TE/PE still resolve via text_cache_dir/cache_dir. Lets
            # colorize read prep-corrected (white-balanced) target latents.
            npz_path = caching_strategy.get_latents_npz_path(
                info.absolute_path,
                info.image_size,
                cache_dir=getattr(subset, "latent_cache_dir", None)
                or getattr(subset, "cache_dir", None),
                image_dir=getattr(subset, "image_dir", None),
            )
            if not caching_strategy.is_disk_cached_latents_expected(
                info.bucket_reso,
                npz_path,
                subset.flip_aug,
                subset.alpha_mask,
            ):
                return False
        return True

    def is_text_encoder_outputs_cache_complete(self) -> bool:
        """True iff every image already has a valid on-disk text-encoder cache.

        Read-only probe (no model, no GPU) used by the trainer to decide
        whether the text encoder needs loading at all. Mirrors the per-file
        skip condition inside ``new_cache_text_encoder_outputs``.
        """
        caching_strategy = TextEncoderOutputsCachingStrategy.get_strategy()
        if caching_strategy is None or not caching_strategy.cache_to_disk:
            return False
        for info in self.image_data.values():
            subset = self.image_to_subset.get(info.image_key)
            npz_path = caching_strategy.get_outputs_npz_path(
                info.absolute_path,
                cache_dir=getattr(subset, "cache_dir", None),
                image_dir=getattr(subset, "image_dir", None),
            )
            if not caching_strategy.is_disk_cached_outputs_expected(npz_path):
                return False
        return True

    def count_repa_pe_sidecars(self) -> Tuple[int, int]:
        """Return ``(present, total)`` REPA PE sidecars for this dataset.

        Read-only probe (no model, no GPU). ``present`` counts images whose
        ``{stem}_anima_{repa_pe_encoder}.safetensors`` resolves on disk via the
        same fallback chain ``_try_load_repa_pe`` uses; ``total`` is the image
        count. The trainer calls this when ``use_repa`` is on to catch a
        fully-absent / partial PE cache before it silently disables the REPA
        alignment term. Returns ``(0, 0)`` when PE loading is off.
        """
        if not self.load_repa_pe:
            return (0, 0)
        present = 0
        total = 0
        for info in self.image_data.values():
            total += 1
            if any(os.path.exists(p) for p in self._repa_pe_sidecar_candidates(info)):
                present += 1
        return (present, total)

    def new_cache_latents(self, model: Any, accelerator: Accelerator):
        r"""
        a brand new method to cache latents. This method caches latents with caching strategy.
        normal cache_latents method is used by default, but this method is used when caching strategy is specified.
        """
        logger.info("caching latents with caching strategy.")
        caching_strategy = LatentsCachingStrategy.get_strategy()
        image_infos = list(self.image_data.values())

        image_infos.sort(key=lambda info: info.bucket_reso[0] * info.bucket_reso[1])

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

        num_processes = accelerator.num_processes
        process_index = accelerator.process_index

        def submit_batch(batch, cond):
            for info in batch:
                if info.image is not None and isinstance(info.image, Future):
                    info.image = info.image.result()
            caching_strategy.cache_batch_latents(
                model, batch, cond.flip_aug, cond.alpha_mask, cond.random_crop
            )

            for info in batch:
                info.image = None

        max_workers = min(os.cpu_count(), len(image_infos))
        max_workers = max(1, max_workers // num_processes)  # consider multi-gpu
        max_workers = min(
            max_workers, caching_strategy.batch_size
        )  # max_workers should be less than batch_size
        executor = ThreadPoolExecutor(max_workers)

        try:
            logger.info("caching latents...")
            for i, info in enumerate(tqdm(image_infos)):
                subset = self.image_to_subset[info.image_key]

                if info.latents_npz is not None:  # fine tuning dataset
                    continue

                if caching_strategy.cache_to_disk:
                    # latent_cache_dir redirects only the target-latent cache
                    # (see is_latents_cache_complete).
                    info.latents_npz = caching_strategy.get_latents_npz_path(
                        info.absolute_path,
                        info.image_size,
                        cache_dir=getattr(subset, "latent_cache_dir", None)
                        or getattr(subset, "cache_dir", None),
                        image_dir=getattr(subset, "image_dir", None),
                    )

                    # multi-GPU: each rank caches only its modulo-strided shard
                    if i % num_processes != process_index:
                        continue

                    cache_available = caching_strategy.is_disk_cached_latents_expected(
                        info.bucket_reso,
                        info.latents_npz,
                        subset.flip_aug,
                        subset.alpha_mask,
                    )
                    if cache_available:
                        continue

                    # latent_cache_dir is prep-populated with *corrected*
                    # targets (e.g. white-balanced colorize targets); a latent
                    # missing here gets encoded from the ORIGINAL image, which
                    # silently skips the correction. Warn loudly, once per stem.
                    if getattr(subset, "latent_cache_dir", None):
                        logger.warning(
                            f"latent_cache_dir is set but {info.latents_npz} is "
                            "missing — encoding from the ORIGINAL (uncorrected) "
                            "image. Re-run the prep step that populates "
                            f"{subset.latent_cache_dir!r} to fix this."
                        )

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

        image_infos.sort(key=lambda info: info.bucket_reso[0] * info.bucket_reso[1])

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

                if cache_available:
                    continue

            condition = Condition(
                info.bucket_reso, subset.flip_aug, subset.alpha_mask, subset.random_crop
            )
            if len(batch) > 0 and current_condition != condition:
                batches.append((current_condition, batch))
                batch = []

            batch.append(info)
            current_condition = condition

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

        batches = []
        batch = []

        num_processes = accelerator.num_processes
        process_index = accelerator.process_index

        logger.info("checking cache validity...")
        for i, info in enumerate(tqdm(image_infos)):
            subset = self.image_to_subset.get(info.image_key)
            if caching_strategy.cache_to_disk:
                # text_cache_dir (when set) redirects only the TE cache —
                # latents still resolve under cache_dir. Lets colorization read
                # re-encoded color-only captions without re-caching latents.
                te_out_npz = caching_strategy.get_outputs_npz_path(
                    info.absolute_path,
                    cache_dir=(
                        getattr(subset, "text_cache_dir", None)
                        or getattr(subset, "cache_dir", None)
                    ),
                    image_dir=getattr(subset, "image_dir", None),
                )
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

        for text_encoder, device, te_dtype in zip(text_encoders, devices, te_dtypes):
            text_encoder.to(device)
            if te_dtype is not None:
                text_encoder.to(dtype=te_dtype)

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

    def __len__(self):
        return self._length

    def _load_cond_latent(
        self, subset, image_info, flipped: bool
    ) -> Optional[torch.Tensor]:
        """Load the *condition* latent for cond≠target tasks (e.g. colorization).

        Resolves a stem-matched ``{stem}_{WxH}_anima.npz`` under
        ``subset.cond_cache_dir`` (nested under the source subpath, mirroring the
        target cache). Returns the bucket-matched tensor, flip-aligned to the
        target, or ``None`` when the subset has no ``cond_cache_dir`` configured
        (→ caller falls back to the ref==target latent)."""
        cond_dir = getattr(subset, "cond_cache_dir", None)
        if not cond_dir:
            return None
        npz_path = self.latents_caching_strategy.get_latents_npz_path(
            image_info.absolute_path,
            image_info.bucket_reso,
            cache_dir=str(cond_dir),
            image_dir=subset.image_dir,
        )
        # The cond is loaded at the target's bucket by default (the common case:
        # cond and target share a shape). Under free-fit the cond is a paired but
        # *distinct* image (e.g. the near-twin _tags member) that can land at a
        # DIFFERENT free-fit shape, so the same-bucket path won't exist — fall back
        # to the cond filed at its own shape (glob by stem; exactly one cond per
        # target) and read its reso from the filename. cond≠target shapes are fine
        # downstream: the EasyControl cond stream encodes at the cond's native token
        # count, and cond_diff_loss self-skips on a shape mismatch.
        cond_reso = image_info.bucket_reso
        if not os.path.exists(npz_path):
            suffix = self.latents_caching_strategy.ANIMA_LATENTS_NPZ_SUFFIX
            stem_prefix = npz_path[: -len(suffix)].rsplit("_", 1)[0]  # drop "_WxH"
            matches = sorted(glob.glob(f"{stem_prefix}_*{suffix}"))
            if not matches:
                raise FileNotFoundError(
                    f"Condition latent cache missing for "
                    f"{image_info.absolute_path!r}: {npz_path} (and no free-fit "
                    f"sibling {stem_prefix}_*{suffix}). Run the cond prep step "
                    f"first (e.g. `make easycontrol-preprocess EASYADAPTER=colorize`)."
                )
            npz_path = matches[0]
            wxh = npz_path[: -len(suffix)].rsplit("_", 1)[1]  # "WWWWxHHHH"
            cw, ch = wxh.split("x")
            cond_reso = (int(cw), int(ch))
        cond, _, _, cond_flipped, _ = (
            self.latents_caching_strategy.load_latents_from_disk(npz_path, cond_reso)
        )
        if flipped:
            if cond_flipped is None:
                raise ValueError(
                    f"flip_aug is on but the cond cache {npz_path} has no flipped "
                    f"latent. Set flip_aug=false for cond≠target subsets (latents "
                    f"can't be flipped post-hoc) or regenerate the cond cache."
                )
            cond = cond_flipped
        return torch.FloatTensor(cond)

    def _caption_key(self, info: ImageInfo) -> str:
        """Caption-index key for an image (posix relpath under its subset's
        ``image_dir``, extension stripped — see :func:`library.io.cache.caption_key`).

        Matches the key ``build_caption_index`` writes: the resized subdir tree
        mirrors the original caption tree, so relativizing this image's path
        against ``subset.image_dir`` reproduces the same ``subdir/stem`` key and
        keeps duplicate bare stems across folders (``en/1`` vs ``ew/1``) distinct.
        """
        from library.io.cache import caption_key

        subset = self.image_to_subset.get(info.image_key)
        image_dir = getattr(subset, "image_dir", None) if subset is not None else None
        return caption_key(info.absolute_path, image_dir)

    def setup_contrastive_negatives(
        self,
        index_path: str,
        *,
        k: int,
        mode: str,
        is_validation: bool,
    ) -> None:
        """Attach an IdentityPairSampler so ``__getitem__`` surfaces ``k``
        cached negative text embeddings (``neg_crossattn_emb``) per example for
        the soft-tokens contrastive objective.

        ``mode`` (docs/proposal/soft_tokens_contrastive.md):
          - ``shuffled``    — an unrelated image (no character/copyright overlap).
          - ``jaccard``     — shuffled sourcing + a per-negative tag-overlap
            weight (``neg_jaccard``) the loss uses to down-weight near-misses.
          - ``hard``        — a same-artist / different-character sibling (falls
            back to shuffled for orphan artists).
          - ``hard_backoff`` — tiered hard negative: same-artist/different-
            character → same-copyright/different-character → shuffled. The
            copyright tier rescues most of ``hard``'s ~71% orphan fallback.

        The candidate pool is restricted to this dataset's registered stems so
        negatives never leak in from another split."""
        if mode not in ("shuffled", "jaccard", "hard", "hard_backoff"):
            raise ValueError(
                "contrastive_negative_mode must be shuffled/jaccard/hard/"
                f"hard_backoff, got {mode!r}"
            )
        from library.datasets.identity_pairs import IdentityPairSampler

        registered = {self._caption_key(info) for info in self.image_data.values()}
        self.contrastive_neg_sampler = IdentityPairSampler(
            index_path,
            min_level="artist",
            cross_artist=False,
            restrict_stems=registered,
        )
        self.contrastive_neg_k = int(k)
        self.contrastive_neg_mode = str(mode)
        n_missing = sum(
            1 for s in registered if not self.contrastive_neg_sampler.has(s)
        )
        if n_missing:
            logger.warning(
                f"[contrastive] {n_missing}/{len(registered)} registered stems "
                f"are absent from {index_path} (will skip negatives for those). "
                f"Re-run `make caption-index` if the dataset changed."
            )

        # One-shot hardness diagnostic: tally the negative *level* each registered
        # stem would draw under this mode (one deterministic draw per stem). Lets
        # you read the strict-vs-shuffled mix before committing to a run — e.g.
        # how much of `hard`'s shuffled fallback the `hard_backoff` copyright tier
        # actually rescues. Skipped for shuffled/jaccard (every draw is shuffled).
        if mode in ("hard", "hard_backoff"):
            from collections import Counter

            diag_rng = random.Random(0)
            hist: Counter[str] = Counter()
            for s in sorted(registered):
                if self.contrastive_neg_sampler.has(s):
                    _, lvl = self.contrastive_neg_sampler.draw(s, mode, diag_rng)
                    hist[lvl] += 1
            total = sum(hist.values())
            if total:
                breakdown = ", ".join(
                    f"{lvl}={n} ({100 * n / total:.0f}%)"
                    for lvl, n in sorted(hist.items(), key=lambda kv: -kv[1])
                )
                logger.info(
                    f"[contrastive] negative-level mix ({mode}, n={total}): {breakdown}"
                )

    def _load_te_for_stem(
        self, stem: str, subset, rel_dir: str
    ) -> Optional[torch.Tensor]:
        """Load a *negative* stem's cached text embedding (post-LLM-adapter
        ``crossattn_emb``) by reconstructing its nested cache path
        (``cache_dir/<rel_dir>/<stem>_anima_te.safetensors``, with a flat
        fallback). Returns ``(S, D)`` or None."""
        from safetensors import safe_open

        suffix = "_anima_te.safetensors"
        cache_dir = getattr(subset, "cache_dir", None) if subset is not None else None
        candidates: list[str] = []
        if cache_dir:
            if rel_dir:
                candidates.append(os.path.join(str(cache_dir), rel_dir, stem + suffix))
            candidates.append(os.path.join(str(cache_dir), stem + suffix))
        cache_path = next((c for c in candidates if os.path.exists(c)), None)
        if cache_path is None:
            raise FileNotFoundError(
                f"TE cache missing for contrastive negative stem {stem!r}. "
                f"Looked in: {candidates}. Run `make preprocess-te` with "
                f"cache_llm_adapter_outputs=true."
            )
        with safe_open(cache_path, framework="pt") as f:
            keys = set(f.keys())
            # Prefer the pristine v0 variant; fall back to single-variant cache.
            for key in ("crossattn_emb_v0", "crossattn_emb"):
                if key in keys:
                    return f.get_tensor(key)
        raise KeyError(
            f"TE cache {cache_path} has no 'crossattn_emb' key — the negative "
            f"requires cache_llm_adapter_outputs=true. Re-run `make preprocess-te`."
        )

    def register_sidecar(self, spec: SidecarSpec) -> None:
        """Attach a per-image sidecar channel (see ``SidecarSpec``)."""
        self._sidecar_specs.append(spec)

    @staticmethod
    def _collate_sidecar(
        spec: SidecarSpec, values: List[Optional[Any]], example: dict
    ) -> None:
        loaded = bool(values) and all(v is not None for v in values)
        if spec.policy == "masked":
            valid = [v for v in values if v is not None]
            if valid:
                ref_shape = valid[0].shape
                example[spec.out_key] = torch.stack(
                    [
                        v
                        if v is not None
                        else torch.zeros(ref_shape, dtype=torch.float32)
                        for v in values
                    ],
                    dim=0,
                )
                example[spec.mask_key] = torch.tensor(
                    [v is not None for v in values], dtype=torch.bool
                )
            else:
                example[spec.out_key] = None
                example[spec.mask_key] = None
        elif spec.policy == "dict":
            if loaded:
                for key in values[0]:
                    if all(key in v for v in values):
                        example[f"{spec.out_key}{key}"] = torch.stack(
                            [v[key] for v in values], dim=0
                        )
        else:  # "all_or_nothing"
            ok = loaded
            if ok and spec.uniform_shape:
                ok = len({tuple(v.shape) for v in values}) == 1
            example[spec.out_key] = torch.stack(values, dim=0) if ok else None

    def enable_sigma_demote(self, native_edge: int, demote_edge: int) -> None:
        """sigma_lowres Phase 1b: activate the σ-demote sidecar channel.

        Every image whose bucket sits in ``native_edge``'s token band gets its
        demote-tier sibling latent loaded onto the batch (``demoted_latents``);
        images native to other tiers ride through with ``None`` (their batches
        always train native). Call on the *train* datasets only.
        """
        self._sigma_demote = (int(native_edge), int(demote_edge))

    def enable_sigma_demote2(self, native_edge: int, demote_edge: int) -> None:
        """E16 stacked router: activate the SECONDARY σ-demote sidecar
        (``demoted_latents2``) for a second, deeper route — the trainer's
        rule 2 (own gate/window/span, priority over rule 1)."""
        self._sigma_demote2 = (int(native_edge), int(demote_edge))

    def _try_load_demoted_latent(self, info: ImageInfo) -> Optional[torch.Tensor]:
        """σ-demote sibling latent: the ``demoted_{H}x{W}`` key inside the
        native npz (emitted by ``make preprocess-demote``). None when the image
        is off the demote route (e.g. a native-896 original on the 1024→896
        route), latents aren't npz-cached, or the npz predates the emit — the
        batch then trains native, so a partial emit degrades instead of
        crashing."""
        return self._load_demoted_sibling(info, self._sigma_demote)

    def _try_load_demoted_latent2(self, info: ImageInfo) -> Optional[torch.Tensor]:
        """Secondary-route sibling (``demoted_latents2``) — same npz, its own
        ``demoted_{H}x{W}`` key (the two routes' buckets never collide)."""
        return self._load_demoted_sibling(info, self._sigma_demote2)

    def _demote_warn_once(self, route: Tuple[int, int], msg: str, *fmt) -> None:
        """Warn once PER ROUTE. A single shared flag would let a stale image on
        the primary route swallow the warning for a wholly un-emitted secondary
        route — the stacked router would then quietly degrade to the primary
        rule, whose only symptom is a few points of wall-clock."""
        if route in self._sigma_demote_warned:
            return
        self._sigma_demote_warned.add(route)
        logger.warning(msg, *fmt)

    def _load_demoted_sibling(
        self, info: ImageInfo, route: Optional[Tuple[int, int]]
    ) -> Optional[torch.Tensor]:
        if not route or info.latents_npz is None or info.bucket_reso is None:
            return None
        from library.datasets.buckets import demote_bucket_for
        from library.io.cache_names import demoted_latents_key

        bucket = demote_bucket_for(
            int(info.bucket_reso[0]), int(info.bucket_reso[1]), route[0], route[1]
        )
        if bucket is None:
            return None
        key = demoted_latents_key(*bucket)
        try:
            npz = self._open_demote_npz(info.latents_npz)
            if key not in npz:
                self._demote_warn_once(
                    route,
                    "sigma_lowres: no '%s' key (route %d:%d) in %s — run `make "
                    "preprocess-demote` to emit the sibling latents; affected "
                    "batches train native (warned once per route).",
                    key,
                    route[0],
                    route[1],
                    info.latents_npz,
                )
                return None
            return torch.from_numpy(npz[key].copy()).float()
        except Exception as e:  # truncated/corrupt npz → native, not a crash
            self._demote_npz_cache = None
            self._demote_warn_once(
                route,
                "sigma_lowres: failed reading %s (%s) — batch trains native.",
                info.latents_npz,
                e,
            )
            return None

    def _open_demote_npz(self, path):
        """One-slot memo over the sample's npz. ``__getitem__`` runs every
        sidecar loader back-to-back for the same image, so a stacked router
        would otherwise re-open (and re-read the zip directory of) the same
        archive once per route."""
        cached = self._demote_npz_cache
        if cached is not None and cached[0] == path:
            return cached[1]
        if cached is not None:
            cached[1].close()
        npz = np.load(path)
        self._demote_npz_cache = (path, npz)
        return npz

    def __getstate__(self):
        # An open NpzFile is not picklable, and the dataset is shipped to
        # DataLoader workers under Windows/`spawn` — drop the memo (workers
        # refill it on their first __getitem__).
        state = self.__dict__.copy()
        state["_demote_npz_cache"] = None
        return state

    def _try_load_inversion_runs(self, info: ImageInfo) -> Optional[torch.Tensor]:
        """Load <stem>_inverted_run{0..N-1}.safetensors from self.inversion_dir.

        Returns a [N_runs, S, D] tensor, or None if any of the expected runs is missing
        (caller masks samples without inversions out of the functional loss).
        """
        if not self.inversion_dir:
            return None
        stem = os.path.splitext(os.path.basename(info.absolute_path))[0]
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

    def _repa_pe_sidecar_candidates(self, info: ImageInfo) -> List[str]:
        """Candidate paths for the ``{stem}_anima_{encoder}.safetensors``
        sidecar, in resolution order (issues.md P2.2):

        1. next to the TE cache (the common case — TE and PE caches share
           ``subset.cache_dir``, and the TE npz path already encodes any
           nested-subdir mirroring),
        2. the subset latent-cache dir, replicating the writer's nesting rule
           (``make preprocess-pe`` resolves via ``resolve_cache_path`` with
           ``cache_dir``+``image_dir``) — needed when ``text_cache_dir``
           redirects only the TE cache (colorize) so candidate 1 lands in a
           directory with no sidecars,
        3. next to the image (legacy no-cache_dir layout).
        """
        stem = os.path.splitext(os.path.basename(info.absolute_path))[0]
        name = f"{stem}_anima_{self.repa_pe_encoder}.safetensors"
        candidates: List[str] = []
        if info.text_encoder_outputs_npz:
            candidates.append(
                os.path.join(os.path.dirname(info.text_encoder_outputs_npz), name)
            )
        subset = self.image_to_subset.get(info.image_key)
        cache_dir = getattr(subset, "cache_dir", None) if subset is not None else None
        if cache_dir:
            from library.io.cache import resolve_cache_path

            candidates.append(
                resolve_cache_path(
                    info.absolute_path,
                    f"_anima_{self.repa_pe_encoder}.safetensors",
                    cache_dir=cache_dir,
                    image_dir=getattr(subset, "image_dir", None),
                )
            )
        candidates.append(os.path.join(os.path.dirname(info.absolute_path), name))
        return list(dict.fromkeys(candidates))

    def _try_load_repa_pe(self, info: ImageInfo) -> Optional[torch.Tensor]:
        """Load cached ``{stem}_anima_{encoder}.safetensors`` patch tokens.

        Returns the ``[T, d_enc]`` feature tensor (CLS still at index 0), or
        None when loading is off / the sidecar is missing (the adapter then
        skips the REPA term for that batch). Resolution walks the fallback
        chain in ``_repa_pe_sidecar_candidates`` because the TE cache can be
        redirected away from where ``make preprocess-pe`` wrote the sidecar
        (``text_cache_dir`` — colorize). Mirrors
        ``library.training.cmmd.resolve_pe_sidecar`` for candidate 1.
        """
        if not self.load_repa_pe:
            return None
        for p in self._repa_pe_sidecar_candidates(info):
            if not os.path.exists(p):
                continue
            from safetensors.torch import load_file

            sd = load_file(p)
            feats = sd.get("image_features")
            return feats.float() if feats is not None else None
        return None

    def restrict_to_byg_tuples(self) -> tuple[int, int]:
        """Drop images lacking a BYG edit-tuple sidecar, then rebuild buckets.

        ``build_edit_tuples`` only emits a ``<stem>_byg.safetensors`` for captions
        containing a swappable tag (v1: a color word); images without one carry no
        BYG training signal. Collation silently omits ``byg_*_emb`` whenever *any*
        sample in a bucket-batch lacks a tuple (see ``__getitem__``), so the
        adapter raises mid-epoch the first time such a batch is drawn. Filtering
        the registry up front guarantees every batch is fully-tupled.

        Returns ``(kept, dropped)`` image counts.
        """
        if not self.byg_text_dir:
            return (0, 0)
        kept: Dict[str, ImageInfo] = {}
        dropped = 0
        for key, info in self.image_data.items():
            stem = os.path.splitext(os.path.basename(info.absolute_path))[0]
            p = os.path.join(self.byg_text_dir, f"{stem}_byg.safetensors")
            if os.path.exists(p):
                kept[key] = info
            else:
                dropped += 1
        if dropped == 0:
            return (len(kept), 0)
        self.image_data = kept
        # Keep the repeat-weighted train-image count in sync (matches the
        # num_repeats * num_images definition in DreamBoothDataset).
        self.num_train_images = sum(info.num_repeats for info in kept.values())
        # bucket_manager.add_image accumulates, so reset before re-bucketing.
        self.bucket_manager = None
        self.make_buckets(target_res=getattr(self, "_target_res", None))
        return (len(kept), dropped)

    def _try_load_byg_tuple(self, info: ImageInfo) -> Optional[dict]:
        """Load <stem>_byg.safetensors from self.byg_text_dir.

        Returns ``{f"{role}_emb": Tensor[S,D], f"{role}_mask": Tensor[S]}`` for
        the four edit-tuple roles, or None if the sidecar is missing / malformed
        (the BYG adapter raises a clear error if a sample lacks its tuple).
        """
        if not self.byg_text_dir:
            return None
        stem = os.path.splitext(os.path.basename(info.absolute_path))[0]
        from safetensors.torch import load_file

        p = os.path.join(self.byg_text_dir, f"{stem}_byg.safetensors")
        if not os.path.exists(p):
            return None
        sd = load_file(p)
        out = {}
        for role in self._byg_roles:
            emb = sd.get(f"{role}_emb")
            if emb is None:
                return None
            out[f"{role}_emb"] = emb.float()
            mask = sd.get(f"{role}_mask")
            if mask is not None:
                out[f"{role}_mask"] = mask
        return out

    def _load_sample(self, subset, image_info, flipped):
        """Load one sample's pixels-or-latents plus its alpha mask.

        Serves from the in-memory or npz latent cache when present, otherwise
        decodes the image from disk (with crop/resize, color aug, and mask
        resolution). Honors ``flipped`` throughout and applies the
        ``preloaded_alpha_mask`` override (mask_dir is the source of truth).

        Returns ``(image, latents, alpha_mask, original_size, crop_ltrb)``;
        exactly one of ``image`` / ``latents`` is non-None.
        """
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
                alpha_mask = None if alpha_mask is None else alpha_mask[:, ::-1].copy()
                del flipped_latents
            latents = torch.FloatTensor(latents)
            if alpha_mask is not None:
                alpha_mask = torch.FloatTensor(alpha_mask)

            image = None
        else:
            img, _, _, _, _ = self.load_image_with_face_info(
                subset, image_info.absolute_path, subset.alpha_mask
            )

            img, original_size, crop_ltrb = trim_and_resize_if_required(
                subset.random_crop,
                img,
                image_info.bucket_reso,
                image_info.resized_size,
                resize_interpolation=image_info.resize_interpolation,
            )

            aug = self.aug_helper.get_augmentor(subset.color_aug)
            if aug is not None:
                img_rgb = img[:, :, :3]
                img_rgb = aug(image=img_rgb)["image"]
                img[:, :, :3] = img_rgb

            if flipped:
                img = img[:, ::-1, :].copy()

            if image_info.mask_path is not None:
                if image_info.preloaded_alpha_mask is not None:
                    # Will be filled in by the post-branch override below.
                    alpha_mask = None
                else:
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

        if image_info.preloaded_alpha_mask is not None:
            # mask_dir is the source of truth: override any alpha_mask coming
            # from the latent cache (npz / in-memory) or the raw-image branch.
            alpha_mask = image_info.preloaded_alpha_mask.float() / 255.0
            if flipped:
                alpha_mask = torch.flip(alpha_mask, [1])

        return image, latents, alpha_mask, original_size, crop_ltrb

    def _draw_contrastive_negatives(self, target_stem, subset):
        """Draw k soft-tokens contrastive negatives for one target, or (None, None).

        Draws k unrelated stems and loads their cached text embeddings.
        Deterministic per target on the rare chance this dataset is used for
        validation; random in training. Returns ``(neg_crossattn, neg_jaccard)``
        — a (k, S, D) stack of negative embeddings and, in jaccard mode, a (k,)
        tag-overlap weight vector. Either is None when no sampler is attached,
        the target is absent from the index, or k distinct negatives couldn't be
        reached (the adapter then skips the contrastive forward).
        """
        neg_sampler = self.contrastive_neg_sampler
        if neg_sampler is None or not neg_sampler.has(target_stem):
            return None, None

        k = self.contrastive_neg_k
        mode = self.contrastive_neg_mode
        nrng = random.Random(self.seed ^ (hash(target_stem) & 0xFFFFFFFF))
        neg_feats: List[torch.Tensor] = []
        neg_jacc: List[float] = []
        for _ in range(k):
            neg_stem, _lvl = neg_sampler.draw(target_stem, mode, nrng)
            if neg_stem == target_stem:
                continue  # no distinct negative reachable
            # neg_stem is a caption-index key (posix ``subdir/stem``); the TE
            # cache filename uses the *bare* stem while rel_dir supplies the
            # subdir, so split them here to avoid a doubled subdir in the path.
            feat = self._load_te_for_stem(
                os.path.basename(neg_stem), subset, neg_sampler.rel_dir(neg_stem)
            )
            if feat is not None:
                neg_feats.append(feat)
                neg_jacc.append(
                    neg_sampler.tag_jaccard(target_stem, neg_stem)
                    if mode == "jaccard"
                    else 0.0
                )
        ok = len(neg_feats) == k
        neg_crossattn = torch.stack(neg_feats, dim=0) if ok else None
        neg_jaccard = (
            torch.tensor(neg_jacc, dtype=torch.float32)
            if (ok and mode == "jaccard")
            else None
        )
        return neg_crossattn, neg_jaccard

    @staticmethod
    def _guard_item_load(image_info, what: str, thunk):
        """Run a per-item disk load, turning a bare loader exception (corrupt
        npz, missing/mismatched safetensors key, …) into a clear error that
        names the offending image and its cache files.

        Without this, a single bad cache raises deep inside safetensors/numpy
        with no file context; under a Windows DataLoader worker that surfaces
        as an opaque "training froze at epoch 1" rather than an actionable
        error. Naming the file makes it debuggable and re-fixable (re-run
        ``make preprocess-*`` for that image)."""
        try:
            return thunk()
        except Exception as e:
            raise RuntimeError(
                f"DataLoader failed to load {what} for image "
                f"{image_info.absolute_path!r} "
                f"(latents={image_info.latents_npz!r}, "
                f"TE={image_info.text_encoder_outputs_npz!r}): "
                f"{type(e).__name__}: {e}"
            ) from e

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
        # Optional condition latent (cond≠target tasks, e.g. colorization);
        # stays all-None when no subset sets cond_cache_dir.
        cond_latents_list: List[Optional[torch.Tensor]] = []
        alpha_mask_list = []
        images = []
        original_sizes_hw = []
        crop_top_lefts = []
        target_sizes_hw = []
        flippeds = []
        text_encoder_outputs_list = []
        custom_attributes = []
        # Soft-tokens contrastive negatives: per-image (k, S, D) stack of cached
        # negative text embeddings, or None when no sampler is attached.
        neg_crossattn_list: List[Optional[torch.Tensor]] = []
        # Per-image (k,) tag-overlap weights for jaccard mode; None otherwise.
        neg_jaccard_list: List[Optional[torch.Tensor]] = []
        # Registered per-image sidecars (inversion runs, BYG tuples, REPA PE…).
        sidecar_values: Dict[str, List[Optional[Any]]] = {
            spec.name: [] for spec in self._sidecar_specs
        }

        for image_key in bucket[image_index : image_index + bucket_batch_size]:
            image_info = self.image_data[image_key]
            subset = self.image_to_subset[image_key]

            custom_attributes.append(subset.custom_attributes)

            loss_weights.append(self.prior_loss_weight if image_info.is_reg else 1.0)

            flipped = subset.flip_aug and random.random() < 0.5

            image, latents, alpha_mask, original_size, crop_ltrb = (
                self._guard_item_load(
                    image_info,
                    "latents",
                    lambda: self._load_sample(subset, image_info, flipped),
                )
            )

            images.append(image)
            latents_list.append(latents)
            cond_latents_list.append(
                self._guard_item_load(
                    image_info,
                    "cond latents",
                    lambda: self._load_cond_latent(subset, image_info, flipped),
                )
            )
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

            target_stem = self._caption_key(image_info)

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
                text_encoder_outputs = self._guard_item_load(
                    image_info,
                    "text-encoder cache",
                    lambda: self.text_encoder_output_caching_strategy.load_outputs_npz(
                        image_info.text_encoder_outputs_npz
                    ),
                )
            else:
                tokenization_required = True
            text_encoder_outputs_list.append(text_encoder_outputs)

            if tokenization_required:
                caption = self.process_caption(subset, image_info.caption)
                input_ids = [ids[0] for ids in self.tokenize_strategy.tokenize(caption)]

            input_ids_list.append(input_ids)
            captions.append(caption)

            for spec in self._sidecar_specs:
                enabled = spec.enabled_attr is None or bool(
                    getattr(self, spec.enabled_attr)
                )
                sidecar_values[spec.name].append(
                    spec.loader(image_info) if enabled else None
                )

            neg_crossattn, neg_jaccard = self._draw_contrastive_negatives(
                target_stem, subset
            )
            neg_crossattn_list.append(neg_crossattn)
            neg_jaccard_list.append(neg_jaccard)

        return self._collate_examples(
            bucket=bucket,
            image_index=image_index,
            custom_attributes=custom_attributes,
            loss_weights=loss_weights,
            text_encoder_outputs_list=text_encoder_outputs_list,
            input_ids_list=input_ids_list,
            alpha_mask_list=alpha_mask_list,
            images=images,
            latents_list=latents_list,
            cond_latents_list=cond_latents_list,
            captions=captions,
            original_sizes_hw=original_sizes_hw,
            crop_top_lefts=crop_top_lefts,
            target_sizes_hw=target_sizes_hw,
            flippeds=flippeds,
            sidecar_values=sidecar_values,
            neg_crossattn_list=neg_crossattn_list,
            neg_jaccard_list=neg_jaccard_list,
        )

    def _collate_examples(
        self,
        *,
        bucket,
        image_index,
        custom_attributes,
        loss_weights,
        text_encoder_outputs_list,
        input_ids_list,
        alpha_mask_list,
        images,
        latents_list,
        cond_latents_list,
        captions,
        original_sizes_hw,
        crop_top_lefts,
        target_sizes_hw,
        flippeds,
        sidecar_values,
        neg_crossattn_list,
        neg_jaccard_list,
    ):
        """Stack the per-sample lists gathered in ``__getitem__`` into one batch.

        Owns the tensor-stacking / None-handling for every channel: pixels,
        latents, cond latents, alpha masks (ragged-fill), text encoder outputs
        and input ids (via :func:`none_or_stack_elements`), size/crop metadata,
        registered sidecars, and soft-tokens contrastive negatives. Returns the
        ``example`` dict consumed by the collator.
        """
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
        # Condition latent for cond≠target tasks (colorization / near-twin removal).
        # Targets in a batch share a bucket, so same-shape conds stack plainly. Under
        # free-fit a cond can land at a different shape than its target, so conds in
        # one batch may be ragged — fine at batch_size=1 (the EasyControl default; a
        # 1-element stack is shape-agnostic). For batch_size>1 a ragged batch can't
        # stack, so fail with a clear message instead of a cryptic torch error.
        if cond_latents_list and cond_latents_list[0] is not None:
            shapes = {tuple(c.shape) for c in cond_latents_list}
            if len(shapes) > 1:
                raise RuntimeError(
                    "cond_latents have mixed shapes within a batch "
                    f"({sorted(shapes)}) — free-fit can pair a cond at a different "
                    "shape than its target. Use batch_size=1 for cond≠target "
                    "subsets under free-fit (the EasyControl default)."
                )
            example["cond_latents"] = torch.stack(cond_latents_list)
        else:
            example["cond_latents"] = None
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

        # Registered sidecars: inversion_runs/_mask (masked — placeholder zeros
        # for absent samples), byg_{role}_emb/_mask (dict — keys absent unless
        # every sample carries a tuple, so the BYG adapter fails loudly),
        # repa_pe_features (all-or-nothing + shape guard). See SidecarSpec.
        for spec in self._sidecar_specs:
            self._collate_sidecar(spec, sidecar_values[spec.name], example)

        # Soft-tokens contrastive negatives: (B, k, S, D) cached text embeddings.
        # All cached crossattn_emb share the padded sequence length, so a plain
        # stack works. None when no sampler is attached (or any target in the
        # bucket couldn't reach k distinct negatives).
        if neg_crossattn_list and all(t is not None for t in neg_crossattn_list):
            example["neg_crossattn_emb"] = torch.stack(neg_crossattn_list, dim=0)
        else:
            example["neg_crossattn_emb"] = None
        # Per-negative tag-overlap weights (B, k) for jaccard mode; None for
        # shuffled / hard (the loss then runs plain InfoNCE).
        if neg_jaccard_list and all(t is not None for t in neg_jaccard_list):
            example["neg_jaccard"] = torch.stack(neg_jaccard_list, dim=0)
        else:
            example["neg_jaccard"] = None

        # Per-item identity (image_data keys, i.e. absolute paths). Always
        # present: the debug walker and the online memorization Δ-gap tracker
        # (library/training/mem_reweight.py) key their state off these; a
        # short string list rides the batch like ``captions``.
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
