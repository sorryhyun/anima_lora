"""Shared training/validation context dataclasses.

``TrainCtx``/``ValCtx`` are frozen bundles built once in ``train()`` and
threaded through per-step/per-batch methods; ``RuntimeState`` is the mutable
per-run counterpart. Live here (not train.py) so ``loop.py`` and other
entrypoints can import them directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from accelerate import Accelerator


@dataclass(frozen=True)
class TrainCtx:
    """Training-wide state, fixed for the whole run, passed to per-step /
    per-batch methods. Per-call values
    (epoch, global_step, progress_bar, …) stay explicit at call sites."""

    args: Any
    accelerator: Accelerator
    network: Any
    unet: Any
    vae: Any
    text_encoders: list
    noise_scheduler: Any
    text_encoding_strategy: Any
    tokenize_strategy: Any
    vae_dtype: torch.dtype
    weight_dtype: torch.dtype
    train_text_encoder: bool
    train_unet: bool
    optimizer_eval_fn: Callable
    optimizer_train_fn: Callable
    is_tracking: bool


@dataclass(frozen=True)
class ValCtx:
    """Validation-wide state fixed for the entire run. ``val_loss_recorder``
    (step vs epoch) stays explicit per call site; everything else is shared."""

    dataloader: Any
    sigmas: list
    steps: int
    total_steps: int
    train_loss_recorder: Any
    original_t_min: float
    original_t_max: float
    # Held so CMMD-style validation can enumerate held-out items for paired
    # sample generation against the cached PE reference pool.
    dataset_group: Any = None


@dataclass(frozen=True)
class DatasetBundle:
    """Return of ``AnimaTrainer._prepare_dataset``: the train/val dataset groups
    plus the shared collator and the ``multiprocessing.Value`` step/epoch
    counters the collator and loop read."""

    train_group: Any
    val_group: Any
    current_epoch: Any  # multiprocessing.Value("i", ...)
    current_step: Any
    collator: Any
    use_user_config: bool
    use_dreambooth_method: bool


@dataclass(frozen=True)
class NetworkBundle:
    """Return of ``AnimaTrainer._create_and_apply_network``: the built+applied
    adapter network and the train/eval flags derived while applying it."""

    network: Any
    net_kwargs: dict
    train_unet: bool
    train_text_encoder: bool


@dataclass(frozen=True)
class OptimizerBundle:
    """Return of ``AnimaTrainer._setup_optimizer_and_dataloader``: optimizer (+
    its name/args and train/eval fns), LR schedule, and the train/val
    dataloaders, all pre-``accelerator.prepare``."""

    optimizer: Any
    optimizer_name: str
    optimizer_args: Any
    optimizer_train_fn: Callable
    optimizer_eval_fn: Callable
    text_encoder_lr: Any
    lr_descriptions: Any
    train_dataloader: Any
    val_dataloader: Any
    lr_scheduler: Any


@dataclass(frozen=True)
class AcceleratedBundle:
    """Return of ``AnimaTrainer._prepare_with_accelerator``: the
    ``accelerator.prepare``-wrapped network/optimizer/dataloaders/scheduler plus
    the re-resolved model handles (the wrapped DiT, the cast/prepared text
    encoders, and the chosen unet weight dtype)."""

    network: Any
    optimizer: Any
    train_dataloader: Any
    val_dataloader: Any
    lr_scheduler: Any
    training_model: Any
    unet: Any
    text_encoders: list
    text_encoder: Any
    unet_weight_dtype: torch.dtype


@dataclass
class RuntimeState:
    """Per-run mutable state threaded across trainer methods (unlike the
    frozen ``*Ctx`` bundles above)."""

    # Merged from adapters' `extra_forwards` in get_noise_pred_and_target;
    # consumed by the loss composer in _process_batch_inner.
    extras_for_step: dict = field(default_factory=dict)
    # EMA λ for the flow_matching_vr AsymFlow §5.2 control variate (frozen
    # reference = trainable DiT with network.set_multiplier(0)).
    vr: dict = field(default_factory=lambda: {"lambda_ema": None})
    # T5("") crossattn sidecar (1, S, 1024) bf16, staged by
    # ensure_uncond_crossattn when caption dropout is on; prepare_text_conds
    # uses it for dropped rows instead of zeros, matching CFG-uncond inference.
    uncond_crossattn_1: torch.Tensor | None = None
    # Gates whether ensure_uncond_crossattn stages the sidecar above.
    caption_dropout_enabled: bool = False
    # Online memorization Δ-gap tracker (library/training/mem_reweight.py),
    # created when --mem_reweight_mode is set; updated in _attach_aux_losses,
    # consumed (EMA + loss_weights multiply) in _process_batch_inner.
    mem_tracker: object | None = None
