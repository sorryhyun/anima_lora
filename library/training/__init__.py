# Training utilities for Anima LoRA training.
# Re-exports all public names so `from library.training import X` works.

from library.training.optimizers import (
    get_optimizer,
    get_optimizer_train_eval_fn,
    is_schedulefree_optimizer,
)
from library.training.schedulers import (
    get_scheduler_fix,
    get_dummy_scheduler,
)
from library.training.metadata import (
    SS_METADATA_KEY_V2,
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_KEY_NETWORK_MODULE,
    SS_METADATA_KEY_NETWORK_DIM,
    SS_METADATA_KEY_NETWORK_ALPHA,
    SS_METADATA_KEY_NETWORK_ARGS,
    SS_METADATA_MINIMUM_KEYS,
    build_minimum_network_metadata,
    build_training_metadata,
    add_dataset_metadata,
    add_model_hash_metadata,
    finalize_metadata,
)
from library.training.samplers import (
    SamplerContext,
    SamplerOut,
    SAMPLER_REGISTRY,
)
from library.training.losses import (
    LossContext,
    LossComposer,
    LOSS_REGISTRY,
    build_loss_composer,
)
from library.training.metrics import (
    MetricContext,
    METRIC_REGISTRY,
    collect_metrics,
)
from library.training.checkpoints import (
    EPOCH_STATE_NAME,
    EPOCH_FILE_NAME,
    EPOCH_DIFFUSERS_DIR_NAME,
    LAST_STATE_NAME,
    DEFAULT_EPOCH_NAME,
    DEFAULT_LAST_OUTPUT_NAME,
    DEFAULT_STEP_NAME,
    STEP_STATE_NAME,
    STEP_FILE_NAME,
    STEP_DIFFUSERS_DIR_NAME,
    default_if_none,
    get_epoch_ckpt_name,
    get_step_ckpt_name,
    get_last_ckpt_name,
    get_remove_epoch_no,
    get_remove_step_no,
    save_sd_model_on_epoch_end_or_stepwise_common,
    save_and_remove_state_on_epoch_end,
    save_and_remove_state_stepwise,
    save_state_on_train_end,
    save_sd_model_on_train_end_common,
    get_checkpoint_state_dir,
    get_checkpoint_ckpt_name,
    save_checkpoint_state,
)

__all__ = [
    # samplers
    "SamplerContext",
    "SamplerOut",
    "SAMPLER_REGISTRY",
    # losses
    "LossContext",
    "LossComposer",
    "LOSS_REGISTRY",
    "build_loss_composer",
    # metrics
    "MetricContext",
    "METRIC_REGISTRY",
    "collect_metrics",
    # metadata
    "SS_METADATA_KEY_V2",
    "SS_METADATA_KEY_BASE_MODEL_VERSION",
    "SS_METADATA_KEY_NETWORK_MODULE",
    "SS_METADATA_KEY_NETWORK_DIM",
    "SS_METADATA_KEY_NETWORK_ALPHA",
    "SS_METADATA_KEY_NETWORK_ARGS",
    "SS_METADATA_MINIMUM_KEYS",
    "build_minimum_network_metadata",
    "build_training_metadata",
    "add_dataset_metadata",
    "add_model_hash_metadata",
    "finalize_metadata",
    # optimizers
    "get_optimizer",
    "get_optimizer_train_eval_fn",
    "is_schedulefree_optimizer",
    # schedulers
    "get_scheduler_fix",
    "get_dummy_scheduler",
    # checkpoints
    "EPOCH_STATE_NAME",
    "EPOCH_FILE_NAME",
    "EPOCH_DIFFUSERS_DIR_NAME",
    "LAST_STATE_NAME",
    "DEFAULT_EPOCH_NAME",
    "DEFAULT_LAST_OUTPUT_NAME",
    "DEFAULT_STEP_NAME",
    "STEP_STATE_NAME",
    "STEP_FILE_NAME",
    "STEP_DIFFUSERS_DIR_NAME",
    "default_if_none",
    "get_epoch_ckpt_name",
    "get_step_ckpt_name",
    "get_last_ckpt_name",
    "get_remove_epoch_no",
    "get_remove_step_no",
    "save_sd_model_on_epoch_end_or_stepwise_common",
    "save_and_remove_state_on_epoch_end",
    "save_and_remove_state_stepwise",
    "save_state_on_train_end",
    "save_sd_model_on_train_end_common",
    "get_checkpoint_state_dir",
    "get_checkpoint_ckpt_name",
    "save_checkpoint_state",
]
