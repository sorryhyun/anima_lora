"""Turbo-only loop helpers: the LR scheduler."""

from __future__ import annotations

import torch

from library.training.schedulers import make_warmup_cosine_scheduler


def scheduler_warmup_steps(total_steps: int) -> int:
    """Linear warmup length of :func:`make_scheduler`: 2% of ``total_steps``, ≥1."""
    return max(1, int(0.02 * total_steps))


def schedule_lrs(total_steps: int, lr: float, shape: str) -> list[float]:
    """The LR of each update under :func:`make_scheduler`, from a dry run."""
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(()))], lr=lr)
    sched = make_scheduler(opt, total_steps, lr, shape)
    lrs = []
    for _ in range(total_steps):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return lrs


def make_scheduler(opt, total_steps: int, lr: float, shape: str = "cosine"):
    """Linear warmup (:func:`scheduler_warmup_steps`) → cosine to ``0.1·lr``, or
    constant at ``lr`` (``shape="constant"``).
    """
    warmup_steps = scheduler_warmup_steps(total_steps)
    # Constant is for EMA runs: without a student EMA it rendered worse than cosine.
    if shape == "constant":
        from torch.optim.lr_scheduler import ConstantLR, LinearLR, SequentialLR

        warmup = LinearLR(opt, start_factor=1e-6 / lr, total_iters=warmup_steps)
        const = ConstantLR(opt, factor=1.0, total_iters=0)
        return SequentialLR(opt, schedulers=[warmup, const], milestones=[warmup_steps])
    return make_warmup_cosine_scheduler(
        opt,
        total_steps,
        lr,
        warmup_steps=warmup_steps,
        eta_min_ratio=0.1,
    )
