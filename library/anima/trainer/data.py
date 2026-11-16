"""Dataset setup for the Anima trainer: blueprint → train/val groups, the
per-feature dataset switches, the compile token budget, and the TE-cache
pass (plus sample-prompt encoding)."""

from __future__ import annotations

import logging
import os
from multiprocessing import Value
from typing import Optional, Union

import torch
from accelerate import Accelerator

from library import train_util
from library.anima import text_strategies
from library.anima.trainer import sigma_route
from library.anima.trainer.cli import resolve_network_kwargs
from library.config import loader as config_util
from library.config.io import load_dataset_config_from_base
from library.config.loader import BlueprintGenerator, ConfigSanitizer
from library.datasets import (
    DatasetGroup,
    MinimalDataset,
    collator_class,
    load_arbitrary_dataset,
)
from library.runtime.device import clean_memory_on_device
from library.training import DatasetBundle

logger = logging.getLogger(__name__)


def build_datasets(args) -> DatasetBundle:
    """Build train/val dataset groups and the collator shared by both loaders."""
    from library.datasets.subsets import resolve_configured_mask_dir

    use_dreambooth_method = args.in_json is None
    use_user_config = args.dataset_config is not None
    caption_dropout_enabled = False

    # `mask_dir` from the config chain names where `make mask` *would* write;
    # gate it on the directory existing so a maskless checkout falls back to
    # the legacy auto-resolution instead of flipping alpha_mask on.
    args.mask_dir = resolve_configured_mask_dir(getattr(args, "mask_dir", None))

    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(support_dropout=True))
        if use_user_config:
            logger.info(f"Loading dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "reg_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning(
                    "ignoring the following options because config file is found: {0}".format(
                        ", ".join(ignored)
                    )
                )
        else:
            base_ds = load_dataset_config_from_base(
                overrides=vars(args),
                method=getattr(args, "method", None),
                methods_subdir=getattr(args, "methods_subdir", None) or "methods",
                config_file=(
                    getattr(args, "config_file", None)
                    if getattr(args, "method", None) is None
                    else None
                ),
            )
            if base_ds is not None:
                if getattr(args, "method", None) is None and getattr(
                    args, "config_file", None
                ):
                    logger.info("Loading dataset config from config_file")
                else:
                    logger.info("Loading dataset config from configs/base.toml")
                user_config = base_ds
                use_user_config = True
            elif use_dreambooth_method:
                logger.info("Using DreamBooth method.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                args.train_data_dir, args.reg_data_dir
                            )
                        }
                    ]
                }
            else:
                logger.info("Training with captions.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": [
                                {
                                    "image_dir": args.train_data_dir,
                                    "metadata_file": args.in_json,
                                }
                            ]
                        }
                    ]
                }

        # Global --sample_ratio override; 1.0 is inert so a per-subset
        # sample_ratio in a custom blueprint stays authoritative.
        sample_ratio = getattr(args, "sample_ratio", None)
        if sample_ratio is not None and sample_ratio != 1.0:
            for ds in user_config.get("datasets", []):
                for sub in ds.get("subsets", []):
                    sub["sample_ratio"] = sample_ratio
            logger.info(f"Applied --sample_ratio={sample_ratio} to all subsets")

        # --artists_shard k_N: one round-robin shard of the artist subdirs,
        # expanded into each subset's path_pattern before the blueprint.
        artists_shard = getattr(args, "artists_shard", None)
        if artists_shard:
            if getattr(args, "path_pattern", None):
                raise ValueError(
                    "--artists_shard and --path_pattern are mutually "
                    "exclusive (the shard expands to a path_pattern)."
                )
            from library.datasets.artist_shard import apply_artist_shard
            from library.env import resolve_under_home

            shard_info = apply_artist_shard(
                user_config, artists_shard, resolve=resolve_under_home
            )
            for image_dir, meta in shard_info.items():
                logger.info(
                    "Applied --artists_shard=%s to %s: %d/%d artists [%s]",
                    artists_shard,
                    image_dir,
                    meta["n_shard"],
                    meta["n_artists"],
                    ", ".join(meta["artists"]),
                )

        blueprint = blueprint_generator.generate(user_config, args)
        # `masked_loss` is the one masking switch: a mask tree on disk must
        # not re-enable masking. Gated before the datasets are built because
        # construction bakes mask paths and preloads the PNGs.
        if not getattr(args, "masked_loss", False):
            ignored_mask_dirs = config_util.disable_masks_in_blueprint(
                blueprint.dataset_group
            )
            if ignored_mask_dirs:
                logger.info(
                    "masked_loss = false: masks under %s are ignored "
                    "(set masked_loss = true to train with them)",
                    ", ".join(ignored_mask_dirs),
                )
            else:
                logger.info("masked_loss = false: training unmasked")
        train_dataset_group, val_dataset_group = (
            config_util.generate_dataset_group_by_blueprint(
                blueprint.dataset_group,
                # Free-fit: buckets are the on-disk resized sizes;
                # target_res is preprocess-only and inert here.
                target_res=getattr(args, "target_res", None),
            )
        )

        rates = [
            subset.caption_dropout_rate
            for ds in train_dataset_group.datasets
            for subset in ds.subsets
        ]
        caption_dropout_enabled = bool(rates) and any(r > 0 for r in rates)
        if caption_dropout_enabled:
            logger.info(f"caption dropout ENABLED -- per-subset rates: {rates}")
        else:
            logger.info("caption dropout DISABLED (rate=0.0 on all subsets)")
    else:
        train_dataset_group = load_arbitrary_dataset(args)
        val_dataset_group = None  # validation is not supported for arbitrary datasets

    sigma_route.enable_on_datasets(args, train_dataset_group)

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = (
        train_dataset_group if args.max_data_loader_n_workers == 0 else None
    )
    collator = collator_class(current_epoch, current_step, ds_for_collator)

    return DatasetBundle(
        train_group=train_dataset_group,
        val_group=val_dataset_group,
        current_epoch=current_epoch,
        current_step=current_step,
        collator=collator,
        use_user_config=use_user_config,
        use_dreambooth_method=use_dreambooth_method,
        caption_dropout_enabled=caption_dropout_enabled,
    )


def configure_dataset_features(
    args,
    train_dataset_group: Union[DatasetGroup, MinimalDataset],
    val_dataset_group: Optional[DatasetGroup],
) -> None:
    """Check the TE-cache / offload flags against the datasets and switch on
    per-feature batch loading (inversion runs, REPA PE features, soft-token
    negatives). May flip ``args`` (cache_llm_adapter_outputs,
    gradient_checkpointing)."""
    # use_text_cache is expanded to cache_text_encoder_outputs in
    # verify_training_args (runs first).
    if args.cache_text_encoder_outputs:
        assert train_dataset_group.is_text_encoder_output_cacheable(
            cache_supports_dropout=True
        ), (
            "when caching Text Encoder output, token_warmup_step or caption_tag_dropout_rate cannot be used"
        )
        if getattr(args, "cache_llm_adapter_outputs", False):
            # Adapter output caching is only valid when the adapter is frozen.
            # The flag is store_true, so a method TOML that sets it can't be
            # switched off from a descriptor: auto-disable instead.
            if args.network_args is not None and any(
                "train_llm_adapter" in a and "true" in a.lower()
                for a in args.network_args
            ):
                logger.warning(
                    "network_args train_llm_adapter=true needs the llm_adapter "
                    "live; disabling cache_llm_adapter_outputs."
                )
                args.cache_llm_adapter_outputs = False
    elif getattr(args, "cache_llm_adapter_outputs", False):
        logger.warning(
            "cache_llm_adapter_outputs=true has no effect without text-encoder "
            "caching (use_text_cache=false / live text encoding); disabling it."
        )
        args.cache_llm_adapter_outputs = False

    if args.unsloth_offload_checkpointing:
        if not args.gradient_checkpointing:
            logger.warning(
                "unsloth_offload_checkpointing is enabled, so gradient_checkpointing is also enabled"
            )
            args.gradient_checkpointing = True
        assert args.blocks_to_swap is None or args.blocks_to_swap == 0, (
            "blocks_to_swap is not supported with unsloth_offload_checkpointing"
        )

    # Propagate inversion_dir to datasets for functional-loss supervision (postfix-func).
    inversion_dir = getattr(args, "inversion_dir", None)
    if inversion_dir:
        num_runs = getattr(args, "functional_loss_num_runs", 3)
        for dataset in train_dataset_group.datasets:
            dataset.inversion_dir = inversion_dir
            dataset.inversion_num_runs = num_runs
        if val_dataset_group is not None:
            for dataset in val_dataset_group.datasets:
                dataset.inversion_dir = inversion_dir
                dataset.inversion_num_runs = num_runs

    # REPA: load cached PE-Spatial patch tokens into batches.
    net_kwargs = resolve_network_kwargs(args)
    if net_kwargs.get("use_repa", "").lower() in ("true", "1", "yes"):
        repa_encoder = net_kwargs.get("repa_encoder") or "pe_spatial"
        for dataset in train_dataset_group.datasets:
            dataset.load_repa_pe = True
            dataset.repa_pe_encoder = repa_encoder
        # A missing PE cache makes the REPA term a silent no-op (the loss
        # skips batches without repa_pe_features): fail on none, warn on partial.
        present, total = train_dataset_group.count_repa_pe_sidecars()
        if total > 0 and present == 0:
            raise RuntimeError(
                f"use_repa is enabled but none of the {total} training "
                f"images have a {repa_encoder} PE feature cache "
                f"(*_anima_{repa_encoder}.safetensors) — the REPA "
                f"alignment loss would be a silent no-op. Run "
                f"`make preprocess-pe ARGS='--encoder {repa_encoder}'` "
                f"first, or disable use_repa."
            )
        if present < total:
            logger.warning(
                f"REPA: only {present}/{total} training images have a "
                f"{repa_encoder} PE sidecar; the alignment term is skipped "
                f"for batches missing one. Run `make preprocess-pe "
                f"ARGS='--encoder {repa_encoder}'` to cover the rest."
            )
        logger.info(
            f"REPA: PE feature loading enabled (encoder={repa_encoder}); "
            f"batches carry repa_pe_features ({present}/{total} cached)."
        )

    # Soft-tokens contrastive negatives, on when contrastive_weight > 0.
    if str(getattr(args, "network_module", "") or "") == (
        "networks.methods.soft_tokens"
    ):
        con_weight = float(net_kwargs.get("contrastive_weight", 0.0) or 0.0)
        if con_weight > 0.0:
            con_k = int(net_kwargs.get("contrastive_k", 1) or 1)
            con_mode = str(net_kwargs.get("contrastive_negative_mode", "shuffled"))
            con_index = "post_image_dataset/captions/caption_index.json"
            if not os.path.exists(con_index):
                raise FileNotFoundError(
                    f"contrastive_index not found: {con_index}. Run `make caption-index`."
                )
            if not getattr(args, "cache_llm_adapter_outputs", False):
                raise ValueError(
                    "soft_tokens contrastive requires "
                    "cache_llm_adapter_outputs=true (negatives are cached "
                    "crossattn_emb swapped off disk)."
                )
            # Train only: validation FM-MSE stays a clean baseline.
            for dataset in train_dataset_group.datasets:
                dataset.setup_contrastive_negatives(
                    con_index, k=con_k, mode=con_mode, is_validation=False
                )
            logger.info(
                f"Soft-tokens contrastive: weight={con_weight} k={con_k} "
                f"mode={con_mode} index={con_index}"
            )


def derive_token_budget(args, train_group, val_group):
    """(n_token_families, seq_range, seq_bands) from the buckets the
    datasets populate, unioned with sample-prompt token counts
    (:func:`sample_prompt_token_counts`). Sizes ``compile_blocks``' dynamo
    cache to exactly the tiers on disk, independent of ``args.target_res``.
    ``seq_bands`` is the per-tier clustering of the same count set
    (``cluster_token_bands``), consumed only under ``--compile_seq_bands``.
    ``(None, None, None)`` when no bucketed resos are available.
    """
    from library.datasets.buckets import cluster_token_bands, token_counts_for_resos

    resos: set = set()
    for group in (train_group, val_group):
        if group is None:
            continue
        for dataset in getattr(group, "datasets", []):
            bm = getattr(dataset, "bucket_manager", None)
            if bm is not None:
                resos.update(bm.resos)
    if not resos:
        return None, None, None
    counts = (
        token_counts_for_resos(resos)
        | sample_prompt_token_counts(args)
        | sigma_route.demoted_token_counts(args, resos)
    )
    return len(counts), (min(counts), max(counts)), cluster_token_bands(counts)


def sample_prompt_token_counts(args) -> set:
    """Token counts the sample prompts will request; empty when sampling
    is off. Samples run through the same compiled blocks as training, so an
    out-of-bucket resolution would raise ConstraintViolationError unless folded
    into the budget up front. Resolutions added mid-run are skipped with a
    warning at sample time.
    """
    from library.datasets.buckets import token_counts_for_sample_prompts

    if not getattr(args, "sample_prompts", None):
        return set()
    will_sample = (
        getattr(args, "sample_at_first", False)
        or getattr(args, "sample_every_n_steps", None)
        or getattr(args, "sample_every_n_epochs", None)
    )
    if not will_sample:
        return set()
    try:
        prompts = train_util.load_prompts(args.sample_prompts)
    except Exception as e:
        logger.warning(
            f"Could not parse sample prompts ({args.sample_prompts}) for the "
            f"compile token budget: {e}. Sample resolutions outside the "
            "training buckets may be skipped under torch_compile."
        )
        return set()
    return token_counts_for_sample_prompts(prompts)


def cache_text_encoder_outputs(
    args, accelerator: Accelerator, text_encoders, groups
) -> dict | None:
    """Populate each group's ``ImageInfo.text_encoder_outputs_npz`` and encode
    the sample prompts once. Returns the sample-prompt TE outputs, or None
    when nothing was encoded.

    With caching on, the on-disk cache is guaranteed complete (asserted in
    ``train()``), so no dataset encoding happens here. Live-encoding mode
    (``cache_text_encoder_outputs=false``) just moves the TE to the device.
    """
    if not args.cache_text_encoder_outputs:
        text_encoders[0].to(accelerator.device)
        return None

    for group in groups:
        if group is not None:
            group.new_cache_text_encoder_outputs([None], accelerator)

    sample_prompts_te_outputs = None
    if text_encoders[0] is not None and args.sample_prompts is not None:
        logger.info(
            f"cache Text Encoder outputs for sample prompts: {args.sample_prompts}"
        )
        logger.info("move text encoder to gpu")
        text_encoders[0].to(accelerator.device)

        tokenize_strategy = text_strategies.TokenizeStrategy.get_strategy()
        text_encoding_strategy = text_strategies.TextEncodingStrategy.get_strategy()

        prompts = train_util.load_prompts(args.sample_prompts)
        sample_prompts_te_outputs = {}
        with accelerator.autocast(), torch.no_grad():
            for prompt_dict in prompts:
                for p in [
                    prompt_dict.get("prompt", ""),
                    prompt_dict.get("negative_prompt", ""),
                ]:
                    if p not in sample_prompts_te_outputs:
                        logger.info(f"  cache TE outputs for: {p}")
                        tokens_and_masks = tokenize_strategy.tokenize(p)
                        sample_prompts_te_outputs[p] = (
                            text_encoding_strategy.encode_tokens(
                                tokenize_strategy,
                                text_encoders,
                                tokens_and_masks,
                            )
                        )

        logger.info("move text encoder back to cpu")
        text_encoders[0].to("cpu")
        clean_memory_on_device(accelerator.device)

    accelerator.wait_for_everyone()
    return sample_prompts_te_outputs
