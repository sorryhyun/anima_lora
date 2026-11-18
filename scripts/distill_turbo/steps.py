"""Per-step DP-DMD loss terms and the fake/critic update, called from ``distill``.

The loop keeps the student's final backward and optimizer step; these are the rollout
and the loss terms around it. ``student_rollout`` and ``anchor_only_loss`` backward the
anchor term themselves, and ``fake_update`` steps the fake optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from library.training.forward import renoise
from library.training.forward import sample_sigma as sample_t

from .context import RunContext


def anchor_target(velocity, eps: torch.Tensor, sigmas: list[float], k: int):
    """Roll ``velocity(z, σ) → fp32 v`` ``k`` Euler steps from ε down ``sigmas`` and
    return the teacher's mean velocity over ``[σ_k, 1]``, ``(ε − z_k) / (1 − σ_k)``.

    The student's step 0 applies this velocity over its own interval ``[σ_1, 1]``, so
    it lands on the teacher's state ``z_k`` only when ``σ_k`` is the student's ``σ_1``
    (``config.anchor_gap`` measures the difference). The anchor cache builder rolls
    the same function."""
    z = eps
    for i in range(k):
        v = velocity(z, sigmas[i])
        z = (z.float() - (sigmas[i] - sigmas[i + 1]) * v).to(eps.dtype)
    return ((eps.float() - z.float()) / (1.0 - sigmas[k])).detach()


def teacher_anchor(
    ctx: RunContext,
    cfg,
    eps: torch.Tensor,
    crossattn_emb: torch.Tensor,
    c_null: torch.Tensor,
    B: int,
) -> torch.Tensor:
    """Teacher ``k_anchor``-step CFG anchor (no grad) → ``v_target``."""

    def velocity(z, s):
        t_b = torch.full((B,), s, device=ctx.device, dtype=ctx.dtype)
        return ctx.teacher_cfg_velocity(z, t_b, crossattn_emb, c_null)

    return anchor_target(velocity, eps, ctx.teacher_anchor_sigmas, cfg.k_anchor)


def anchor_only_loss(
    ctx: RunContext,
    eps: torch.Tensor,
    v_target: torch.Tensor,
    crossattn_emb: torch.Tensor,
    B: int,
) -> torch.Tensor:
    """``dpdmd.anchor_only``: the step-0 regression on its own, backwarded here.

    One grad-bearing student forward at σ = 1 and MSE to ``v_target``, unweighted
    (Adam and the clip make ``div_weight`` a no-op when it is the only term).
    Returns the detached loss.
    """
    x = eps.requires_grad_()  # grad-ckpt needs a grad-requiring forward input
    t_b = torch.full((B,), ctx.student_sigmas[0], device=ctx.device, dtype=ctx.dtype)
    v_first = ctx.forward("student", x, t_b, crossattn_emb, no_grad=False)
    loss = nn.functional.mse_loss(v_first.float(), v_target)
    loss.backward()
    return loss.detach()


@dataclass
class Rollout:
    """The student rollout up to its grad step g."""

    x_pred: torch.Tensor  # x_g − σ_g·v_g, attached through step g only
    s_g: float
    v_first: torch.Tensor  # step-0 velocity (metrics)
    div_loss: torch.Tensor  # pre-weight; 0 with the anchor off


def student_rollout(
    ctx: RunContext,
    cfg,
    eps: torch.Tensor,
    v_target: torch.Tensor | None,
    sigmas: list[float],
    g: int,
    crossattn_emb: torch.Tensor,
    B: int,
) -> Rollout:
    """Roll the student from ε to grad step ``g`` ≥ 1 and take its one-step
    x0-prediction.

    Step 0 regresses on ``v_target`` (``None``: anchor off) and that term is
    backwarded here. Steps 1..g−1 run under no_grad and step g starts from a fresh
    leaf, so the DM gradient never reaches step 0 (DP-DMD Fig 5).
    """
    device, dtype = ctx.device, ctx.dtype
    anchor = v_target is not None
    s0 = sigmas[0]
    x = eps
    x.requires_grad_()  # grad-ckpt needs a grad-requiring forward input
    t_b = torch.full((B,), s0, device=device, dtype=dtype)
    v_first = ctx.forward("student", x, t_b, crossattn_emb, no_grad=not anchor)
    if anchor:
        div_loss = nn.functional.mse_loss(v_first.float(), v_target)
        (cfg.div_weight * div_loss).backward()
    else:
        div_loss = torch.zeros((), device=device)

    x = (x - (s0 - sigmas[1]) * v_first).detach()
    for i in range(1, g):  # backward simulation
        t_b = torch.full((B,), sigmas[i], device=device, dtype=dtype)
        v = ctx.forward("student", x, t_b, crossattn_emb, no_grad=True)
        x = x - (sigmas[i] - sigmas[i + 1]) * v
    x_g = x.detach().requires_grad_()  # fresh leaf
    s_g = sigmas[g]
    t_b = torch.full((B,), s_g, device=device, dtype=dtype)
    v_g = ctx.forward("student", x_g, t_b, crossattn_emb, no_grad=False)
    return Rollout(
        x_pred=x_g - s_g * v_g,
        s_g=s_g,
        v_first=v_first,
        div_loss=div_loss.detach(),
    )


@dataclass
class DmdResult:
    """Outputs of the DMD surrogate."""

    grad_signal: torch.Tensor  # detached fp32
    delta_dm: torch.Tensor  # v_real − v_fake (metrics)
    tau_dm: torch.Tensor  # query level
    tau_dm_e: torch.Tensor  # the same, broadcast to (B,1,1,1)
    v_real_cond_dm: torch.Tensor
    v_fake_cond_dm: torch.Tensor


def dmd_surrogate(
    ctx: RunContext,
    cfg,
    x_pred: torch.Tensor,
    crossattn_emb: torch.Tensor,
    c_null: torch.Tensor,
    B: int,
) -> DmdResult:
    """DMD on x_θ: CFG'd teacher against the cond-only fake, at a uniform query
    level τ.

    The real score must be CFG'd: with cond-only on both sides v_real ≈ v_fake
    (dm_cos ≈ 0.9999) and the signal is noise. The signal is ``τ·Δ``; under
    ``dm_x0_norm`` it is divided by the per-sample mean ``|τ·v_real|`` (floored at
    ``norm_floor``), which cancels τ.
    """
    tau_dm = sample_t(
        B,
        distribution="uniform",
        sigmoid_scale=cfg.sigmoid_scale,
        device=ctx.device,
        dtype=ctx.dtype,
    )
    eps_dm = torch.randn_like(x_pred)
    x_renoised_dm = renoise(x_pred.detach(), tau_dm, eps_dm)
    v_real = ctx.teacher_cfg_velocity(x_renoised_dm, tau_dm, crossattn_emb, c_null)
    v_fake = ctx.forward("fake", x_renoised_dm, tau_dm, crossattn_emb, no_grad=True)
    delta_dm = v_real - v_fake
    tau_dm_e = tau_dm.view(B, 1, 1, 1).float()
    grad = tau_dm_e * delta_dm.float()
    if cfg.dm_x0_norm:
        grad = grad / (
            (tau_dm_e * v_real.float())
            .abs()
            .mean(dim=(1, 2, 3), keepdim=True)
            .clamp_min(cfg.norm_floor)
        )
    return DmdResult(
        grad_signal=grad.detach(),
        delta_dm=delta_dm,
        tau_dm=tau_dm,
        tau_dm_e=tau_dm_e,
        v_real_cond_dm=v_real,
        v_fake_cond_dm=v_fake,
    )


class ClippedGradMean:
    """Mean of the student gradient over ``n`` loop iterations (``optim.grad_accum``).

    The caller clips each iteration's gradient before :meth:`add`, so every
    iteration enters at norm ≤ ``grad_clip``. Clipping the sum instead lets a
    high-τ draw (DM grad norm 20–30 against a median near 2) outweigh the rest
    of its group.
    """

    def __init__(self, params, n: int):
        self.params = list(params)
        self.n = n
        self.count = 0
        self.sums = [torch.zeros_like(p) for p in self.params]

    def add(self) -> bool:
        """Take the current ``.grad`` into the sum. On the ``n``-th call the mean
        is written back to ``.grad`` and True is returned."""
        for s, p in zip(self.sums, self.params):
            if p.grad is not None:
                s.add_(p.grad)
        self.count += 1
        if self.count < self.n:
            return False
        for s, p in zip(self.sums, self.params):
            p.grad = s / self.n
            s.zero_()
        self.count = 0
        return True


def clip_or_norm(params, grad_clip: float) -> torch.Tensor:
    """Clip to ``grad_clip`` (off at ≤ 0) and return the pre-clip total norm."""
    max_norm = grad_clip if grad_clip > 0 else float("inf")
    return torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)


def fake_regression_step(
    ctx: RunContext,
    cfg,
    x_pred_d: torch.Tensor,
    crossattn_emb: torch.Tensor,
    tau: torch.Tensor,
    eps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One fake (critic) update: flow-matching regression on ``x_pred_d`` renoised to
    ``tau``. Returns the loss and the pre-clip gradient norm."""
    x_t = renoise(x_pred_d, tau, eps).requires_grad_()
    v_fake = ctx.forward("fake", x_t, tau, crossattn_emb, no_grad=False)
    loss = nn.functional.mse_loss(v_fake.float(), (eps - x_pred_d).float())
    loss.backward()
    gn = clip_or_norm(ctx.turbo.fake_params(), cfg.grad_clip)
    ctx.fake_opt.step()
    ctx.fake_opt.zero_grad(set_to_none=True)
    ctx.fake_sched.step()
    return loss, gn


def sample_fake_tau(ctx: RunContext, cfg, B: int) -> torch.Tensor:
    """Fake renoise level, from ``sampling.t_distribution``."""
    return sample_t(
        B,
        distribution=cfg.t_distribution,
        sigmoid_scale=cfg.sigmoid_scale,
        device=ctx.device,
        dtype=ctx.dtype,
    )


def fake_update(
    ctx: RunContext,
    cfg,
    x_pred: torch.Tensor,
    crossattn_emb: torch.Tensor,
    B: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``fake_steps_per_student_step`` fake (critic) updates against
    ``x_pred.detach()``, each on fresh (τ, ε). Returns the mean loss and the mean
    pre-clip gradient norm."""
    x_pred_d = x_pred.detach()
    loss_sum = gn_sum = torch.zeros((), device=ctx.device)
    for _ in range(cfg.fake_steps_per_student_step):
        tau = sample_fake_tau(ctx, cfg, B)
        eps = torch.randn_like(x_pred_d)
        fake_loss, gn_fake = fake_regression_step(
            ctx, cfg, x_pred_d, crossattn_emb, tau, eps
        )
        ctx.tau_profile.add(fake_loss, tau)
        loss_sum = loss_sum + fake_loss.detach()
        gn_sum = gn_sum + gn_fake
    n = cfg.fake_steps_per_student_step
    return loss_sum / n, gn_sum / n
