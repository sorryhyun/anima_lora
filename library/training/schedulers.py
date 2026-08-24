import ast
import importlib
import logging
from typing import Any, Optional

import torch
from torch.optim import Optimizer

from library.training.optimizers import is_schedulefree_optimizer

# transformers (~1.3s) and diffusers (~2s) are imported lazily inside
# get_scheduler_fix so that merely importing this module (and, transitively,
# library.train_util) doesn't pay for them. Only the actual scheduler-build
# path needs them, and only the piecewise_constant branch reaches into
# diffusers.optimization.

logger = logging.getLogger(__name__)


def get_dummy_scheduler(optimizer: Optimizer) -> Any:
    class DummyScheduler:
        def __init__(self, optimizer: Optimizer):
            self.optimizer = optimizer

        def step(self):
            pass

        def get_last_lr(self):
            return [group["lr"] for group in self.optimizer.param_groups]

    return DummyScheduler(optimizer)


def make_warmup_cosine_scheduler(
    optimizer: Optimizer,
    total_steps: int,
    lr: float,
    *,
    warmup_steps: int,
    eta_min_ratio: float = 0.1,
):
    """Linear warmup → cosine anneal, the schedule the distillation loops share.

    Warmup ramps ``1e-6·lr → lr`` over ``warmup_steps``; cosine then anneals to
    ``eta_min_ratio·lr`` over the remaining ``total_steps − warmup_steps``.
    ``warmup_steps <= 0`` skips warmup and returns a bare ``CosineAnnealingLR``
    over all ``total_steps``.

    Open-coded identically in ``project/finished/mod_guidance`` and
    ``scripts/distill_turbo`` before being promoted here.
    """
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR,
        LinearLR,
        SequentialLR,
    )

    if warmup_steps and warmup_steps > 0:
        warmup = LinearLR(optimizer, start_factor=1e-6 / lr, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=lr * eta_min_ratio
        )
        return SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
    return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * eta_min_ratio)


def get_scheduler_fix(args, optimizer: Optimizer, num_processes: int):
    """
    Unified API to get any scheduler from its name.
    """
    if is_schedulefree_optimizer(optimizer, args):
        return get_dummy_scheduler(optimizer)

    name = args.lr_scheduler
    num_training_steps = args.max_train_steps * num_processes
    num_warmup_steps: Optional[int] = (
        int(args.lr_warmup_steps * num_training_steps)
        if isinstance(args.lr_warmup_steps, float)
        else args.lr_warmup_steps
    )
    lr_scheduler_kwargs = {}
    if args.lr_scheduler_args is not None and len(args.lr_scheduler_args) > 0:
        for arg in args.lr_scheduler_args:
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            lr_scheduler_kwargs[key] = value

    def wrap_check_needless_num_warmup_steps(return_vals):
        if num_warmup_steps is not None and num_warmup_steps != 0:
            raise ValueError(
                f"{name} does not require `num_warmup_steps`. Set None or 0."
            )
        return return_vals

    if args.lr_scheduler_type:
        lr_scheduler_type = args.lr_scheduler_type
        logger.info(f"use {lr_scheduler_type} | {lr_scheduler_kwargs} as lr_scheduler")
        if "." not in lr_scheduler_type:
            lr_scheduler_module = torch.optim.lr_scheduler
        else:
            values = lr_scheduler_type.split(".")
            lr_scheduler_module = importlib.import_module(".".join(values[:-1]))
            lr_scheduler_type = values[-1]
        lr_scheduler_class = getattr(lr_scheduler_module, lr_scheduler_type)
        lr_scheduler = lr_scheduler_class(optimizer, **lr_scheduler_kwargs)
        return wrap_check_needless_num_warmup_steps(lr_scheduler)

    # Gate on the literal value ("piecewise_constant") so the diffusers import
    # (~2s) is only paid when that scheduler is actually requested.
    if name == "piecewise_constant":
        from diffusers.optimization import (
            SchedulerType as DiffusersSchedulerType,
            TYPE_TO_SCHEDULER_FUNCTION as DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION,
        )

        name = DiffusersSchedulerType(name)
        schedule_func = DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION[name]
        return schedule_func(optimizer, **lr_scheduler_kwargs)

    from transformers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION

    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    if name == SchedulerType.CONSTANT:
        return wrap_check_needless_num_warmup_steps(
            schedule_func(optimizer, **lr_scheduler_kwargs)
        )

    if num_warmup_steps is None:
        raise ValueError(
            f"{name} requires `num_warmup_steps`, please provide that argument."
        )

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(
            optimizer, num_warmup_steps=num_warmup_steps, **lr_scheduler_kwargs
        )

    if num_training_steps is None:
        raise ValueError(
            f"{name} requires `num_training_steps`, please provide that argument."
        )

    if name == SchedulerType.LINEAR or name == SchedulerType.COSINE:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **lr_scheduler_kwargs,
        )

    if name == SchedulerType.COSINE_WITH_RESTARTS:
        # Without num_cycles the transformers default (1 hard cycle) is
        # shape-identical to plain cosine — silently degenerate (issue #69).
        lr_scheduler_kwargs.setdefault("num_cycles", args.lr_scheduler_num_cycles)
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **lr_scheduler_kwargs,
        )

    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **lr_scheduler_kwargs,
    )
