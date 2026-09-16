# Training utilities for Anima LoRA training.
#
# This is a *curated* facade: it re-exports the public, stable surface of the
# training package, grouped by concern below. Two things are deliberately NOT
# re-exported here and must be imported by their submodule path:
#   - The argparse / CLI surface, which lives in `library.config.cli_args`
#     (it is config, not training).
#   - Loop-internal machinery only `train.py`/`loop.py` compose:
#       library.training.loop          (build_loop_state, run_training_loop)
#       library.training.validation    (run_validation)
#       library.training.log_dispatch  (dispatch_logs)
#       library.training.forward       (per-step forward helpers)
# Everything below is part of the supported import-from-the-package surface.

from library.training.contexts import (
    AcceleratedBundle,
    DatasetBundle,
    NetworkBundle,
    OptimizerBundle,
    RuntimeState,
    TrainCtx,
    ValCtx,
)

# Per-method extension protocol (public — used by networks/methods/*).
from library.training.method_adapter import (
    ComputeLossCtx,
    ForwardArtifacts,
    MethodAdapter,
    SetupCtx,
    StepCtx,
    ValidationBaseline,
    resolve_adapters,
)

from library.training.samplers import (
    SamplerContext,
    SamplerOut,
    SAMPLER_REGISTRY,
)

# The default rectified-flow training step as plain kwargs — the bench-friendly
# entry point. Home is library.runtime.noise; re-exported here so
# training-adjacent callers find it next to the sampler surface.
from library.runtime.noise import fm_training_batch

from library.training.losses import (
    LivenessLedger,
    LossContext,
    LossComposer,
    LOSS_REGISTRY,
    build_loss_composer,
    add_custom_train_arguments,
    apply_masked_loss,
    conditional_loss,
    get_huber_threshold_if_needed,
)

from library.training.metrics import (
    MetricContext,
    MetricProducer,
    collect_metrics,
)

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

from library.training.hashing import (
    model_hash,
    calculate_sha256,
    addnet_hash_legacy,
    addnet_hash_safetensors,
    precalculate_safetensors_hashes,
    get_git_revision_hash,
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
    CheckpointSaver,
    default_if_none,
    get_epoch_ckpt_name,
    get_step_ckpt_name,
    get_last_ckpt_name,
    save_sd_model_on_epoch_end_or_stepwise_common,
    save_state_on_epoch_end,
    save_state_stepwise,
    save_state_on_train_end,
    save_sd_model_on_train_end_common,
    get_checkpoint_state_dir,
    get_checkpoint_ckpt_name,
    save_checkpoint_state,
)

from library.training.progress import (
    ProgressSink,
    run_scope,
)

from library.training.loss_recorder import LossRecorder

__all__ = [
    "AcceleratedBundle",
    "DatasetBundle",
    "NetworkBundle",
    "OptimizerBundle",
    "RuntimeState",
    "TrainCtx",
    "ValCtx",
    "ComputeLossCtx",
    "ForwardArtifacts",
    "MethodAdapter",
    "SetupCtx",
    "StepCtx",
    "ValidationBaseline",
    "resolve_adapters",
    "SamplerContext",
    "SamplerOut",
    "SAMPLER_REGISTRY",
    "fm_training_batch",
    "LivenessLedger",
    "LossContext",
    "LossComposer",
    "LOSS_REGISTRY",
    "build_loss_composer",
    "add_custom_train_arguments",
    "apply_masked_loss",
    "conditional_loss",
    "get_huber_threshold_if_needed",
    "MetricContext",
    "MetricProducer",
    "collect_metrics",
    "get_optimizer",
    "get_optimizer_train_eval_fn",
    "is_schedulefree_optimizer",
    "get_scheduler_fix",
    "get_dummy_scheduler",
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
    "model_hash",
    "calculate_sha256",
    "addnet_hash_legacy",
    "addnet_hash_safetensors",
    "precalculate_safetensors_hashes",
    "get_git_revision_hash",
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
    "CheckpointSaver",
    "default_if_none",
    "get_epoch_ckpt_name",
    "get_step_ckpt_name",
    "get_last_ckpt_name",
    "save_sd_model_on_epoch_end_or_stepwise_common",
    "save_state_on_epoch_end",
    "save_state_stepwise",
    "save_state_on_train_end",
    "save_sd_model_on_train_end_common",
    "get_checkpoint_state_dir",
    "get_checkpoint_ckpt_name",
    "save_checkpoint_state",
    "ProgressSink",
    "run_scope",
    "LossRecorder",
]
