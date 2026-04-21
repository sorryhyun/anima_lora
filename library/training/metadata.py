# Metadata construction for LoRA training checkpoints.
#
# Extracted from train.py to keep the training loop focused on training.

from __future__ import annotations

import json
import os
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Metadata key constants
# ---------------------------------------------------------------------------

SS_METADATA_KEY_V2 = "ss_v2"
SS_METADATA_KEY_BASE_MODEL_VERSION = "ss_base_model_version"
SS_METADATA_KEY_NETWORK_MODULE = "ss_network_module"
SS_METADATA_KEY_NETWORK_DIM = "ss_network_dim"
SS_METADATA_KEY_NETWORK_ALPHA = "ss_network_alpha"
SS_METADATA_KEY_NETWORK_ARGS = "ss_network_args"

SS_METADATA_MINIMUM_KEYS = [
    SS_METADATA_KEY_V2,
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_KEY_NETWORK_MODULE,
    SS_METADATA_KEY_NETWORK_DIM,
    SS_METADATA_KEY_NETWORK_ALPHA,
    SS_METADATA_KEY_NETWORK_ARGS,
]


def build_minimum_network_metadata(
    v2: Optional[str],
    base_model: Optional[str],
    network_module: str,
    network_dim: str,
    network_alpha: str,
    network_args: Optional[dict],
) -> dict[str, str]:
    metadata = {
        SS_METADATA_KEY_NETWORK_MODULE: network_module,
        SS_METADATA_KEY_NETWORK_DIM: network_dim,
        SS_METADATA_KEY_NETWORK_ALPHA: network_alpha,
    }
    if v2 is not None:
        metadata[SS_METADATA_KEY_V2] = v2
    if base_model is not None:
        metadata[SS_METADATA_KEY_BASE_MODEL_VERSION] = base_model
    if network_args is not None:
        metadata[SS_METADATA_KEY_NETWORK_ARGS] = json.dumps(network_args)
    return metadata


# ---------------------------------------------------------------------------
# Full training metadata
# ---------------------------------------------------------------------------


def build_training_metadata(
    args,
    *,
    session_id: int,
    training_started_at: float,
    text_encoder_lr: Any,
    optimizer_name: str,
    optimizer_args: str,
    model_version: str,
    num_train_images: int,
    num_val_images: int,
    num_reg_images: int,
    num_batches_per_epoch: int,
    num_train_epochs: int,
) -> dict[str, Any]:
    """Build the base training-session metadata dict from args and locals."""
    from library.training.hashing import get_git_revision_hash

    return {
        "ss_session_id": session_id,
        "ss_training_started_at": training_started_at,
        "ss_output_name": args.output_name,
        "ss_learning_rate": args.learning_rate,
        "ss_text_encoder_lr": text_encoder_lr,
        "ss_unet_lr": args.unet_lr,
        "ss_num_train_images": num_train_images,
        "ss_num_validation_images": num_val_images,
        "ss_num_reg_images": num_reg_images,
        "ss_num_batches_per_epoch": num_batches_per_epoch,
        "ss_num_epochs": num_train_epochs,
        "ss_gradient_checkpointing": args.gradient_checkpointing,
        "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
        "ss_max_train_steps": args.max_train_steps,
        "ss_lr_warmup_steps": args.lr_warmup_steps,
        "ss_lr_scheduler": args.lr_scheduler,
        "ss_network_module": args.network_module,
        "ss_network_dim": args.network_dim,
        "ss_network_alpha": args.network_alpha,
        "ss_network_dropout": args.network_dropout,
        "ss_mixed_precision": args.mixed_precision,
        "ss_full_fp16": bool(args.full_fp16),
        "ss_v2": bool(args.v2),
        "ss_base_model_version": model_version,
        "ss_clip_skip": args.clip_skip,
        "ss_max_token_length": args.max_token_length,
        "ss_cache_latents": bool(args.cache_latents),
        "ss_seed": args.seed,
        "ss_lowram": args.lowram,
        "ss_noise_offset": args.noise_offset,
        "ss_multires_noise_iterations": args.multires_noise_iterations,
        "ss_multires_noise_discount": args.multires_noise_discount,
        "ss_adaptive_noise_scale": args.adaptive_noise_scale,
        "ss_zero_terminal_snr": args.zero_terminal_snr,
        "ss_training_comment": args.training_comment,
        "ss_sd_scripts_commit_hash": get_git_revision_hash(),
        "ss_optimizer": optimizer_name
        + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
        "ss_max_grad_norm": args.max_grad_norm,
        "ss_caption_dropout_rate": args.caption_dropout_rate,
        "ss_caption_dropout_every_n_epochs": args.caption_dropout_every_n_epochs,
        "ss_caption_tag_dropout_rate": args.caption_tag_dropout_rate,
        "ss_face_crop_aug_range": args.face_crop_aug_range,
        "ss_prior_loss_weight": args.prior_loss_weight,
        "ss_min_snr_gamma": args.min_snr_gamma,
        "ss_scale_weight_norms": args.scale_weight_norms,
        "ss_ip_noise_gamma": args.ip_noise_gamma,
        "ss_debiased_estimation": bool(args.debiased_estimation_loss),
        "ss_noise_offset_random_strength": args.noise_offset_random_strength,
        "ss_ip_noise_gamma_random_strength": args.ip_noise_gamma_random_strength,
        "ss_loss_type": args.loss_type,
        "ss_huber_schedule": args.huber_schedule,
        "ss_huber_scale": args.huber_scale,
        "ss_huber_c": args.huber_c,
        "ss_pseudo_huber_c": args.pseudo_huber_c,
        "ss_multiscale_loss_weight": args.multiscale_loss_weight,
        "ss_fp8_base": bool(args.fp8_base),
        "ss_fp8_base_unet": bool(args.fp8_base_unet),
        "ss_validation_seed": args.validation_seed,
        "ss_validation_split": args.validation_split,
        "ss_max_validation_steps": args.max_validation_steps,
        "ss_validate_every_n_epochs": args.validate_every_n_epochs,
        "ss_validate_every_n_steps": args.validate_every_n_steps,
        "ss_resize_interpolation": args.resize_interpolation,
    }


# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------


def _build_subset_metadata(subset) -> dict[str, Any]:
    meta = {
        "img_count": subset.img_count,
        "num_repeats": subset.num_repeats,
        "color_aug": bool(subset.color_aug),
        "flip_aug": bool(subset.flip_aug),
        "random_crop": bool(subset.random_crop),
        "shuffle_caption": bool(subset.shuffle_caption),
        "keep_tokens": subset.keep_tokens,
        "keep_tokens_separator": subset.keep_tokens_separator,
        "secondary_separator": subset.secondary_separator,
        "enable_wildcard": bool(subset.enable_wildcard),
        "caption_prefix": subset.caption_prefix,
        "caption_suffix": subset.caption_suffix,
        "resize_interpolation": subset.resize_interpolation,
    }
    return meta


def _build_dataset_metadata_user_config(datasets) -> tuple[list, dict, dict]:
    """Build dataset metadata when using user_config (multiple datasets)."""
    datasets_metadata = []
    tag_frequency: dict = {}
    dataset_dirs_info: dict = {}

    for dataset in datasets:
        dataset_metadata = {
            "is_dreambooth": True,
            "batch_size_per_device": dataset.batch_size,
            "num_train_images": dataset.num_train_images,
            "num_reg_images": dataset.num_reg_images,
            "resolution": (dataset.width, dataset.height),
            "enable_bucket": bool(dataset.enable_bucket),
            "min_bucket_reso": dataset.min_bucket_reso,
            "max_bucket_reso": dataset.max_bucket_reso,
            "tag_frequency": dataset.tag_frequency,
            "bucket_info": dataset.bucket_info,
            "resize_interpolation": dataset.resize_interpolation,
        }

        subsets_metadata = []
        for subset in dataset.subsets:
            subset_metadata = _build_subset_metadata(subset)

            image_dir_or_metadata_file = None
            if subset.image_dir:
                image_dir = os.path.basename(subset.image_dir)
                subset_metadata["image_dir"] = image_dir
                image_dir_or_metadata_file = image_dir

            subset_metadata["class_tokens"] = subset.class_tokens
            subset_metadata["is_reg"] = subset.is_reg
            if subset.is_reg:
                image_dir_or_metadata_file = None

            subsets_metadata.append(subset_metadata)

            if image_dir_or_metadata_file is not None:
                v = image_dir_or_metadata_file
                i = 2
                while v in dataset_dirs_info:
                    v = image_dir_or_metadata_file + f" ({i})"
                    i += 1
                image_dir_or_metadata_file = v

                dataset_dirs_info[image_dir_or_metadata_file] = {
                    "n_repeats": subset.num_repeats,
                    "img_count": subset.img_count,
                }

        dataset_metadata["subsets"] = subsets_metadata
        datasets_metadata.append(dataset_metadata)

        for ds_dir_name, ds_freq_for_dir in dataset.tag_frequency.items():
            if ds_dir_name in tag_frequency:
                continue
            tag_frequency[ds_dir_name] = ds_freq_for_dir

    return datasets_metadata, tag_frequency, dataset_dirs_info


def add_dataset_metadata(
    metadata: dict[str, Any],
    train_dataset_group,
    args,
    *,
    use_user_config: bool,
    use_dreambooth_method: bool,
    total_batch_size: int,
) -> None:
    """Add dataset-related metadata to *metadata* in place."""
    if use_user_config:
        datasets_metadata, tag_frequency, dataset_dirs_info = (
            _build_dataset_metadata_user_config(train_dataset_group.datasets)
        )
        metadata["ss_datasets"] = json.dumps(datasets_metadata)
        metadata["ss_tag_frequency"] = json.dumps(tag_frequency)
        metadata["ss_dataset_dirs"] = json.dumps(dataset_dirs_info)
    else:
        assert len(train_dataset_group.datasets) == 1, (
            f"There should be a single dataset but {len(train_dataset_group.datasets)} found. "
            "This seems to be a bug."
        )

        dataset = train_dataset_group.datasets[0]

        dataset_dirs_info: dict = {}
        reg_dataset_dirs_info: dict = {}
        if use_dreambooth_method:
            for subset in dataset.subsets:
                info = reg_dataset_dirs_info if subset.is_reg else dataset_dirs_info
                info[os.path.basename(subset.image_dir)] = {
                    "n_repeats": subset.num_repeats,
                    "img_count": subset.img_count,
                }
        else:
            for subset in dataset.subsets:
                dataset_dirs_info[os.path.basename(subset.metadata_file)] = {
                    "n_repeats": subset.num_repeats,
                    "img_count": subset.img_count,
                }

        metadata.update(
            {
                "ss_batch_size_per_device": args.train_batch_size,
                "ss_total_batch_size": total_batch_size,
                "ss_resolution": args.resolution,
                "ss_color_aug": bool(args.color_aug),
                "ss_flip_aug": bool(args.flip_aug),
                "ss_random_crop": bool(args.random_crop),
                "ss_shuffle_caption": bool(args.shuffle_caption),
                "ss_enable_bucket": bool(dataset.enable_bucket),
                "ss_bucket_no_upscale": bool(dataset.bucket_no_upscale),
                "ss_min_bucket_reso": dataset.min_bucket_reso,
                "ss_max_bucket_reso": dataset.max_bucket_reso,
                "ss_keep_tokens": args.keep_tokens,
                "ss_dataset_dirs": json.dumps(dataset_dirs_info),
                "ss_reg_dataset_dirs": json.dumps(reg_dataset_dirs_info),
                "ss_tag_frequency": json.dumps(dataset.tag_frequency),
                "ss_bucket_info": json.dumps(dataset.bucket_info),
            }
        )


# ---------------------------------------------------------------------------
# Model / VAE hash metadata
# ---------------------------------------------------------------------------


def add_model_hash_metadata(metadata: dict[str, Any], args) -> None:
    """Add model name/hash and VAE name/hash entries to *metadata* in place."""
    from library.training.hashing import model_hash, calculate_sha256

    if args.pretrained_model_name_or_path is not None:
        sd_model_name = args.pretrained_model_name_or_path
        if os.path.exists(sd_model_name):
            metadata["ss_sd_model_hash"] = model_hash(sd_model_name)
            metadata["ss_new_sd_model_hash"] = calculate_sha256(sd_model_name)
            sd_model_name = os.path.basename(sd_model_name)
        metadata["ss_sd_model_name"] = sd_model_name

    if args.vae is not None:
        vae_name = args.vae
        if os.path.exists(vae_name):
            metadata["ss_vae_hash"] = model_hash(vae_name)
            metadata["ss_new_vae_hash"] = calculate_sha256(vae_name)
            vae_name = os.path.basename(vae_name)
        metadata["ss_vae_name"] = vae_name


# ---------------------------------------------------------------------------
# Finalization
# ---------------------------------------------------------------------------


def finalize_metadata(
    metadata: dict[str, Any],
    *,
    net_kwargs: Optional[dict] = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Stringify all values and extract minimum metadata for filtering.

    Returns (metadata, minimum_metadata) — both with string values.
    """
    if net_kwargs:
        metadata["ss_network_args"] = json.dumps(net_kwargs)

    metadata = {k: str(v) for k, v in metadata.items()}

    minimum_metadata = {
        k: metadata[k] for k in SS_METADATA_MINIMUM_KEYS if k in metadata
    }
    return metadata, minimum_metadata
