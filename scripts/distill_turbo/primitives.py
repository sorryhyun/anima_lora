"""Turbo-only loop helpers: CDM grid and extrapolation, GAN ramp, LR scheduler."""

from __future__ import annotations

import torch

from library.training.schedulers import make_warmup_cosine_scheduler


def sample_dynamic_sigmas(n_min: int, n_max: int) -> list[float]:
    """One CDM-style dynamic rollout grid (arXiv:2605.06376 §3.2).

    Length N ~ U{n_min..n_max}; interior anchors drawn uniform in (0,1) and
    sorted descending, endpoints pinned to t₁=1 and 0: the static grid's (N+1)-point
    1 → 0 layout, so ``sigmas[i] - sigmas[i+1]`` is the Euler dt of step i. The
    pinned t₁=1 keeps the diversity anchor valid (v_target assumes the first
    forward sits at t=1). CPU RNG, no GPU sync.
    """
    n = int(torch.randint(n_min, n_max + 1, (1,)).item())
    if n == 1:
        return [1.0, 0.0]
    interior = torch.sort(torch.rand(n - 1), descending=True).values.tolist()
    return [1.0, *interior, 0.0]


def cdm_extrapolate(
    x: torch.Tensor, v: torch.Tensor, s_from: float, t_to: torch.Tensor
) -> torch.Tensor:
    """Velocity-driven Euler extrapolation to an off-trajectory point (CDM Eq. 7).

    ``x(t') = x + (t' − s)·v``: the rollout's Euler form with a stride to
    ``t' ~ U(0,1)`` (possibly noiseward) instead of the next grid point. Inputs are
    detached and the result is a graph-free fp32 tensor, the launch point of one
    grad-bearing student forward. ``t_to`` is per-sample ``(B,)``.
    """
    stride = t_to.detach().float().view(-1, *([1] * (x.dim() - 1))) - s_from
    return x.detach().float() + stride * v.detach().float()


def gan_effective_weight(cfg, step: int) -> float:
    """Generator-side GAN weight at ``step``: delay window, then linear ramp.

    0 for ``step < gan_delay_steps`` (DM + div re-diversify a collapsed warm start
    first; an unramped token head froze pose at the init's mode), then a linear
    0 → ``gan_loss_weight_gen`` ramp over ``gan_warmup_steps`` (instant when 0).
    The disc trains from step 0 regardless.
    """
    if cfg.gan_loss_weight_gen == 0.0 or step < cfg.gan_delay_steps:
        return 0.0
    if cfg.gan_warmup_steps > 0:
        ramp = min(1.0, (step - cfg.gan_delay_steps + 1) / cfg.gan_warmup_steps)
        return cfg.gan_loss_weight_gen * ramp
    return cfg.gan_loss_weight_gen


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

    Without a student EMA, constant rendered worse than cosine; constant is for
    EMA runs.
    """
    warmup_steps = scheduler_warmup_steps(total_steps)
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
