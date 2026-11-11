"""Per-step DP-DMD loss terms and the fake/critic update, called from ``distill.run_loop``.

The loop keeps the final backward / optimizer step inline; these are the rollout and
the loss terms around it. ``student_rollout`` backwards the diversity term when it
is severed from the grad step, ``cdm_off_trajectory_loss`` backwards its own graph,
and ``fake_update`` steps the fake and disc optimizers.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace

import torch
import torch.nn as nn

from library.anima.models import Anima
from networks.methods.turbo_dmd import gan_loss_discriminator, gan_loss_generator

from .primitives import cdm_extrapolate, renoise, sample_t
from .setup import RunContext


@contextmanager
def selective_block_grad_ckpt(model: Anima):
    """Arm per-block unsloth-offload checkpointing for one forward, then restore.

    Unsloth, not ``torch_checkpoint``: ``checkpoint(compiled_fn, use_reentrant=False)``
    diverges from the inductor forward graph on recompute (``CheckpointError``); the
    unsloth path is ``@torch._disable_dynamo`` and runs ``_forward`` eager both times.
    The flags are read per block call, so flipping them costs no recompile.

    The recompute runs at backward under the view that is current then, so every
    caller backwards before the view changes again (``cdm_off_trajectory_loss``) or
    makes its forward the last view flip of the step (``gan_generator_term``).
    """
    saved = [
        (b.gradient_checkpointing, b.unsloth_offload_checkpointing)
        for b in model.blocks
    ]
    for b in model.blocks:
        b.gradient_checkpointing = True
        b.unsloth_offload_checkpointing = True
    try:
        yield
    finally:
        for b, (g, u) in zip(model.blocks, saved):
            b.gradient_checkpointing = g
            b.unsloth_offload_checkpointing = u


def _uniform_tau(ctx: RunContext, cfg, B: int) -> torch.Tensor:
    """DM / MFD / CDM query level: uniform by design, independent of t_distribution."""
    return sample_t(
        B,
        distribution="uniform",
        sigmoid_scale=cfg.sigmoid_scale,
        device=ctx.device,
        dtype=ctx.dtype,
    )


def _dm_signal(cfg, scale_e: torch.Tensor, delta: torch.Tensor, v_real: torch.Tensor):
    """Detached fp32 ``scale·Δ``; under ``dm_x0_norm`` divided by the per-sample
    mean ``|scale·v_real|`` (floored at ``norm_floor``), which cancels the scale."""
    grad = scale_e * delta.float()
    if cfg.dm_x0_norm:
        grad = grad / (
            (scale_e * v_real.float())
            .abs()
            .mean(dim=(1, 2, 3), keepdim=True)
            .clamp_min(cfg.norm_floor)
        )
    return grad.detach()


def teacher_anchor(
    ctx: RunContext,
    cfg,
    eps: torch.Tensor,
    crossattn_emb: torch.Tensor,
    c_null: torch.Tensor,
    B: int,
) -> torch.Tensor:
    """Teacher K-step CFG anchor (no grad) → ``v_target``.

    Rolls the CFG'd teacher ``k_anchor`` Euler steps from ε and returns the mean
    velocity ε→z_tk over ``[t_k, 1]``, the target for the student's t=1 first step.
    """
    z = eps
    for i in range(cfg.k_anchor):
        s_i = ctx.teacher_anchor_sigmas[i]
        s_next = ctx.teacher_anchor_sigmas[i + 1]
        t_b = torch.full((B,), s_i, device=ctx.device, dtype=ctx.dtype)
        v = ctx.teacher_cfg_velocity(z, t_b, crossattn_emb, c_null)
        z = (z.float() - (s_i - s_next) * v).to(ctx.dtype)
    return ((eps.float() - z.float()) / (1.0 - ctx.t_k_anchor)).detach()


@dataclass
class Rollout:
    """The student rollout up to its grad step g."""

    x_pred: torch.Tensor  # x_g − σ_g·v_g, attached through step g only
    x_g: torch.Tensor  # input of step g (fresh leaf unless rollout_grad)
    v_g: torch.Tensor
    s_g: float
    v_first: torch.Tensor  # step-0 velocity (metrics)
    div_loss: torch.Tensor  # pre-weight; 0 with the anchor off
    div_pending: bool  # the caller adds div_weight·div_loss to its final backward


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
    """Roll the student from ε to grad step ``g`` and take its one-step x0-prediction.

    Step 0 regresses on ``v_target`` (``None``: anchor off, no diversity term). For
    g ≥ 1 the diversity term is backwarded here under ``detach_after_first``, steps
    1..g−1 run under no_grad and step g starts from a fresh leaf, so the DM grad
    never reaches step 0 (DP-DMD Fig 5). For g = 0 (``dm_first_step``) step 0 is the
    grad step: the DM grad lands on it, and its diversity term shares that graph, so
    it is left to the caller's backward.

    Under ``rollout_grad`` nothing is detached: x_g carries step 0's graph, so the
    DM grad on x_pred reaches step 0 through the steps after it, and the diversity
    term rides the caller's backward (superturbo.md).
    """
    device, dtype = ctx.device, ctx.dtype
    anchor = v_target is not None
    keep = cfg.rollout_grad
    s0 = sigmas[0]
    x = eps
    x.requires_grad_()  # grad-ckpt needs a grad-requiring forward input
    t_b = torch.full((B,), s0, device=device, dtype=dtype)
    v_first = ctx.forward(
        "student", x, t_b, crossattn_emb, no_grad=not (anchor or g == 0 or keep)
    )
    if anchor:
        div_loss = nn.functional.mse_loss(v_first.float(), v_target)
    else:
        div_loss = torch.zeros((), device=device)
    div_pending = anchor and (g == 0 or keep or not cfg.detach_after_first)
    if anchor and not div_pending:
        (cfg.div_weight * div_loss).backward()

    if g == 0:
        x_g, v_g, s_g = x, v_first, s0
    else:
        x = x - (s0 - sigmas[1]) * v_first
        if not keep:
            x = x.detach()
        for i in range(1, g):  # backward simulation (graph kept under rollout_grad)
            t_b = torch.full((B,), sigmas[i], device=device, dtype=dtype)
            v = ctx.forward("student", x, t_b, crossattn_emb, no_grad=not keep)
            x = x - (sigmas[i] - sigmas[i + 1]) * v
        x_g = x if keep else x.detach().requires_grad_()  # fresh leaf
        s_g = sigmas[g]
        t_b = torch.full((B,), s_g, device=device, dtype=dtype)
        v_g = ctx.forward("student", x_g, t_b, crossattn_emb, no_grad=False)
    return Rollout(
        x_pred=x_g - s_g * v_g,
        x_g=x_g,
        v_g=v_g,
        s_g=s_g,
        v_first=v_first,
        div_loss=div_loss,
        div_pending=div_pending,
    )


@dataclass
class DmdResult:
    """Outputs of the DM-slot surrogate (DMD or MFD)."""

    grad_signal: torch.Tensor  # detached fp32
    delta_dm: torch.Tensor  # v_real − v_fake (metrics)
    tau_dm: torch.Tensor  # query level; the GAN gen renoise reuses it
    tau_dm_e: torch.Tensor  # signal scale broadcast to (B,1,1,1)
    v_real_cond_dm: torch.Tensor
    v_fake_cond_dm: torch.Tensor
    eps_dm: torch.Tensor  # renoise ε; the GAN gen renoise reuses it


def dmd_surrogate(
    ctx: RunContext,
    cfg,
    x_pred: torch.Tensor,
    crossattn_emb: torch.Tensor,
    c_null: torch.Tensor,
    B: int,
) -> DmdResult:
    """DMD on x_θ: CFG'd teacher against the cond-only fake.

    The real score must be CFG'd: with cond-only on both sides v_real ≈ v_fake
    (dm_cos ≈ 0.9999) and the signal is noise.
    """
    tau_dm = _uniform_tau(ctx, cfg, B)
    eps_dm = torch.randn_like(x_pred)
    x_renoised_dm = renoise(x_pred.detach(), tau_dm, eps_dm)
    v_real = ctx.teacher_cfg_velocity(x_renoised_dm, tau_dm, crossattn_emb, c_null)
    v_fake = ctx.forward("fake", x_renoised_dm, tau_dm, crossattn_emb, no_grad=True)
    delta_dm = v_real - v_fake
    tau_dm_e = tau_dm.view(B, 1, 1, 1).float()
    return DmdResult(
        grad_signal=_dm_signal(cfg, tau_dm_e, delta_dm, v_real),
        delta_dm=delta_dm,
        tau_dm=tau_dm,
        tau_dm_e=tau_dm_e,
        v_real_cond_dm=v_real,
        v_fake_cond_dm=v_fake,
        eps_dm=eps_dm,
    )


def mfd_steps(span_max: float, dtau: float, max_steps: int) -> int:
    """Euler step count for one mean-flow integral: ceil(span / dtau), in [1, max_steps]."""
    k = max(1, int(-(-span_max // dtau)))
    return min(k, max(1, max_steps))


def _mean_velocity(v_fn, x_a, sa, sb, K) -> torch.Tensor:
    """Mean Euler velocity of ``v_fn`` from σ=sa to σ=sb (per sample, K uniform steps).

    Accumulates ``mean_k v_k`` instead of dividing the displacement by ``sa − sb``;
    at K=1 it is ``v_fn(x_a, sa)`` bit for bit.
    """
    B = x_a.shape[0]
    sa_f, sb_f = sa.float(), sb.float()
    h = ((sa_f - sb_f) / K).view(B, 1, 1, 1)
    x = x_a
    u = None
    for k in range(K):
        sig = (sa_f + (sb_f - sa_f) * (k / K)).to(sa.dtype) if k else sa
        v = v_fn(x, sig).float()
        u = v if u is None else u + v
        if k + 1 < K:
            x = (x.float() - h * v).to(x_a.dtype)
    return u if K == 1 else u / K


def mfd_surrogate(
    ctx: RunContext,
    cfg,
    x_pred: torch.Tensor,
    crossattn_emb: torch.Tensor,
    c_null: torch.Tensor,
    B: int,
) -> DmdResult:
    """Mean Flow Distillation (arXiv:2606.11155) in the DM slot.

    Renoise x_θ to σ_a, integrate the CFG'd teacher and the fake to σ_b in K Euler
    steps, and push x_θ along the endpoint gap ``−(σ_a − σ_b)·(U_real − U_fake)``.
    The ODE Jacobian is taken as identity (their §B.6). ``to_zero`` at K=1 equals
    :func:`dmd_surrogate` (same RNG draws, same values).

    Field reuse: ``tau_dm`` = σ_a, ``tau_dm_e`` = σ_a − σ_b, ``v_*_cond_dm`` = U.
    """
    sa = _uniform_tau(ctx, cfg, B)
    eps_dm = torch.randn_like(x_pred)
    if cfg.mfd_interval == "uniform":
        sb = sa * torch.rand_like(sa)
    else:
        sb = torch.zeros_like(sa)
    x_a = renoise(x_pred.detach(), sa, eps_dm)
    K = mfd_steps(float((sa - sb).max()), cfg.mfd_dtau, cfg.mfd_max_steps)

    def _v_real(x, sig):
        return ctx.teacher_cfg_velocity(x, sig, crossattn_emb, c_null)

    def _v_fake(x, sig):
        return ctx.forward("fake", x, sig, crossattn_emb, no_grad=True)

    u_real = _mean_velocity(_v_real, x_a, sa, sb, K)
    u_fake = _mean_velocity(_v_fake, x_a, sa, sb, K)
    delta = u_real - u_fake
    span_e = (sa - sb).view(B, 1, 1, 1).float()
    return DmdResult(
        grad_signal=_dm_signal(cfg, span_e, delta, u_real),
        delta_dm=delta,
        tau_dm=sa,
        tau_dm_e=span_e,
        v_real_cond_dm=u_real,
        v_fake_cond_dm=u_fake,
        eps_dm=eps_dm,
    )


def dm_slot(
    ctx: RunContext,
    cfg,
    x_pred: torch.Tensor,
    crossattn_emb: torch.Tensor,
    c_null: torch.Tensor,
    B: int,
) -> DmdResult:
    """The DM-slot surrogate (MFD when ``mfd_dtau > 0``, else DMD) over
    ``dm_draws`` independent (σ_a, ε_dm) draws on the same ``x_pred``.

    ``grad_signal`` is the mean of the per-draw signals (each keeps its own
    ``dm_x0_norm`` division); every other field is the first draw's, so the GAN
    renoise and the metrics see what a single draw gives. At ``dm_draws = 1``
    this is the surrogate call bit for bit.
    """
    fn = mfd_surrogate if cfg.mfd_dtau > 0.0 else dmd_surrogate
    first = fn(ctx, cfg, x_pred, crossattn_emb, c_null, B)
    if cfg.dm_draws <= 1:
        return first
    acc = first.grad_signal.clone()
    for _ in range(cfg.dm_draws - 1):
        acc += fn(ctx, cfg, x_pred, crossattn_emb, c_null, B).grad_signal
    return replace(first, grad_signal=acc / cfg.dm_draws)


def cdm_off_trajectory_loss(
    ctx: RunContext,
    cfg,
    cdm_src: tuple,
    crossattn_emb: torch.Tensor,
    c_null: torch.Tensor,
    latents: torch.Tensor,
    B: int,
) -> None:
    """L_CDM (arXiv:2605.06376 §3.3), backwarded here.

    Euler-extrapolates the grad step's detached (x_g, v_g, σ_g) to t' ~ U(0,1), runs
    one grad-bearing student forward there, and gives its clean estimate
    ``x0_off = x_off − t'·v_off`` the DMD signal. Supervises the off-manifold region
    few-step Euler drifts through, which on-trajectory rollouts never visit.

    Must run before the GAN gen forward. The student forward is checkpointed under
    ``cdm.grad_ckpt`` and recomputes at this backward, so the view is set back to
    'student' first; backwarding here also frees the graph before the GAN forward.
    """
    x_g_cdm, v_g_cdm, s_g_cdm = cdm_src
    t_off = torch.rand(B).to(device=ctx.device, dtype=ctx.dtype)
    # The leaf's grad flag is load-bearing: under the unsloth checkpoint, the
    # reentrant path drops the LoRA param grads when every input is detached.
    x_off = (
        cdm_extrapolate(x_g_cdm, v_g_cdm, s_g_cdm, t_off).to(ctx.dtype).requires_grad_()
    )
    cdm_ckpt = (
        selective_block_grad_ckpt(ctx.model) if cfg.cdm_grad_ckpt else nullcontext()
    )
    with cdm_ckpt:
        v_off = ctx.forward("student", x_off, t_off, crossattn_emb, no_grad=False)
    x0_off = x_off.float() - t_off.view(B, 1, 1, 1).float() * v_off.float()
    tau_cdm = _uniform_tau(ctx, cfg, B)
    eps_cdm = torch.randn_like(latents)
    x_renoised_cdm = renoise(x0_off.detach().to(ctx.dtype), tau_cdm, eps_cdm)
    v_real_cdm = ctx.teacher_cfg_velocity(
        x_renoised_cdm, tau_cdm, crossattn_emb, c_null
    )
    # Transition-aware fake: x0_off is the student's estimate at t'; σ_g is restored
    # for the fake update that follows.
    ctx.turbo.set_fake_source(t_off)
    v_fake_cdm = ctx.forward(
        "fake", x_renoised_cdm, tau_cdm, crossattn_emb, no_grad=True
    )
    ctx.turbo.set_fake_source(s_g_cdm)
    tau_cdm_e = tau_cdm.view(B, 1, 1, 1).float()
    grad_cdm = _dm_signal(cfg, tau_cdm_e, v_real_cdm - v_fake_cdm, v_real_cdm)
    cdm_loss = (grad_cdm * x0_off).mean()
    ctx.turbo.set_view("student")
    (cfg.cdm_weight * cdm_loss).backward()
    ctx.metrics.add_cdm(grad_cdm)


def _teacher_feats(
    ctx: RunContext, x_t: torch.Tensor, tau: torch.Tensor, crossattn_emb, *, no_grad
):
    """Frozen-teacher features at the GAN taps, early exit at the deepest one."""
    return ctx.turbo.features_in_order(
        ctx.forward(
            "teacher",
            x_t,
            tau,
            crossattn_emb,
            no_grad=no_grad,
            return_block_features=ctx.turbo.gan_feature_set,
            return_features_early=True,
        )
    )


def gan_generator_term(
    ctx: RunContext,
    cfg,
    x_pred: torch.Tensor,
    tau_dm: torch.Tensor,
    eps_dm: torch.Tensor,
    crossattn_emb: torch.Tensor,
    gan_weight: float,
) -> torch.Tensor:
    """GAN generator term on the frozen teacher's features of the renoised x_pred.

    The renoise keeps x_pred attached and the teacher forward is grad-enabled, so
    the grad reaches the student; the disc is frozen here. Early exit keeps it a
    half-depth forward (full depth OOM'd). Zero loss, no forward and no RNG draw
    when the GAN is off or ``gan_weight`` is still 0 in the delay window.
    """
    turbo = ctx.turbo
    if turbo.disc is None or gan_weight == 0.0:
        return torch.zeros((), device=ctx.device)

    turbo.set_disc_requires_grad(False)
    x_renoised_gan = renoise(x_pred, tau_dm, eps_dm)
    # ~3 GB of retained teacher activations otherwise.
    gan_ckpt = (
        selective_block_grad_ckpt(ctx.model) if cfg.gan_grad_ckpt else nullcontext()
    )
    with gan_ckpt:
        feats = _teacher_feats(
            ctx, x_renoised_gan, tau_dm, crossattn_emb, no_grad=False
        )
    return gan_loss_generator(turbo.disc(feats))


def clip_or_norm(params, grad_clip: float) -> torch.Tensor:
    """Clip to ``grad_clip`` (off at ≤ 0) and return the pre-clip total norm."""
    max_norm = grad_clip if grad_clip > 0 else float("inf")
    return torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)


def fake_update(
    ctx: RunContext,
    cfg,
    x_pred: torch.Tensor,
    latents: torch.Tensor,
    crossattn_emb: torch.Tensor,
    B: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Fake (critic) and disc updates against ``x_pred.detach()``.

    ``fake_steps_per_student_step`` inner updates, each on fresh (τ, ε); with the
    GAN on, the disc steps once per inner update (FastGen cadence). Returns the
    detached means of fake loss, disc loss, disc margin (mean real − mean fake
    logit: the hinge losses sit at ≈0.69 / 1.39 even while the disc is winning) and
    per-logit spread (std of the fake logits; 0 for a single-logit head), then the
    mean pre-clip gradient norms of the fake and the disc.
    """
    turbo = ctx.turbo
    device, dtype = ctx.device, ctx.dtype
    x_pred_d = x_pred.detach()
    fake_loss_sum = torch.zeros((), device=device)
    gan_disc_sum = torch.zeros((), device=device)
    gan_margin_sum = torch.zeros((), device=device)
    gan_spread_sum = torch.zeros((), device=device)
    gn_fake_sum = torch.zeros((), device=device)
    gn_disc_sum = torch.zeros((), device=device)

    def _sample_tau():
        return sample_t(
            B,
            distribution=cfg.t_distribution,
            sigmoid_scale=cfg.sigmoid_scale,
            device=device,
            dtype=dtype,
        )

    for _ in range(cfg.fake_steps_per_student_step):
        tau_fake = _sample_tau()
        eps_fake = torch.randn_like(x_pred_d)
        x_t_fake = renoise(x_pred_d, tau_fake, eps_fake).requires_grad_()
        v_fake = ctx.forward("fake", x_t_fake, tau_fake, crossattn_emb, no_grad=False)
        target_v_fake = eps_fake - x_pred_d
        fake_loss = nn.functional.mse_loss(v_fake.float(), target_v_fake.float())
        fake_loss.backward()
        gn_fake_sum = gn_fake_sum + clip_or_norm(turbo.fake_params(), cfg.grad_clip)
        ctx.fake_opt.step()
        ctx.fake_opt.zero_grad(set_to_none=True)
        ctx.fake_sched.step()
        fake_loss_sum = fake_loss_sum + fake_loss.detach()
        ctx.tau_profile.add(fake_loss, tau_fake)

        if turbo.disc is None:
            continue
        # Disc: renoised x_pred vs renoised real latents through the frozen teacher
        # (no grad), grad only to the disc head.
        turbo.set_disc_requires_grad(True)
        if cfg.gan_use_same_t_noise:
            tau_d, eps_d = tau_fake, eps_fake
        else:
            tau_d = _sample_tau()
            eps_d = torch.randn_like(x_pred_d)

        def _disc_logits(latent_in, extra_noise=None):
            x_t = renoise(latent_in, tau_d, eps_d)
            if extra_noise is not None:
                x_t = x_t + extra_noise
            return turbo.disc(
                _teacher_feats(ctx, x_t, tau_d, crossattn_emb, no_grad=True)
            )

        fake_logits_d = _disc_logits(x_pred_d)
        real_logits_d = _disc_logits(latents)
        loss_disc = gan_loss_discriminator(real_logits_d, fake_logits_d)
        with torch.no_grad():
            gan_margin_sum = gan_margin_sum + (
                real_logits_d.mean() - fake_logits_d.mean()
            )
            fl = fake_logits_d.flatten()
            if fl.numel() > 1:
                gan_spread_sum = gan_spread_sum + fl.std()

        # Approximate R1 (APT): logit change under a small perturbation of the
        # renoised real input.
        if cfg.gan_r1_weight > 0.0:
            real_logits_a = _disc_logits(
                latents, cfg.gan_r1_alpha * torch.randn_like(latents)
            )
            loss_disc = loss_disc + cfg.gan_r1_weight * nn.functional.mse_loss(
                real_logits_d, real_logits_a
            )

        loss_disc.backward()
        gn_disc_sum = gn_disc_sum + clip_or_norm(turbo.disc_params(), cfg.grad_clip)
        ctx.disc_opt.step()
        ctx.disc_opt.zero_grad(set_to_none=True)
        ctx.disc_sched.step()
        turbo.set_disc_requires_grad(False)
        gan_disc_sum = gan_disc_sum + loss_disc.detach()
    n_inner = cfg.fake_steps_per_student_step
    return (
        fake_loss_sum / n_inner,
        gan_disc_sum / n_inner,
        gan_margin_sum / n_inner,
        gan_spread_sum / n_inner,
        gn_fake_sum / n_inner,
        gn_disc_sum / n_inner,
    )
