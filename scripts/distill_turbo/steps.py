"""Per-step DP-DMD loss terms and updates called from ``distill.run_loop``.

Each function is one ``# --- … ---`` block of the training loop, reading the
built objects off the :class:`~scripts.distill_turbo.setup.RunContext`
and the resolved config. The loop keeps the algorithmic spine — the student
rollout (whose graph the non-split diversity term rides) and the final
assemble/backward/optimizer step — inline; these are the surrounding loss terms
and the fake/critic update.

Two terms carry documented in-branch side effects:

* ``cdm_off_trajectory_loss`` backwards its own graph (view restored to student
  first) and records ``metrics.add_cdm`` — it MUST run before the GAN gen forward.
* ``fake_update`` runs the fake + discriminator optimizer/scheduler steps.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn

from library.anima.models import Anima
from networks.methods.turbo_dmd import gan_loss_discriminator, gan_loss_generator

from .primitives import cdm_extrapolate, renoise, sample_t
from .setup import RunContext


@contextmanager
def selective_block_grad_ckpt(model: Anima):
    """Arm per-block gradient checkpointing for one forward, then restore.

    ``Block.forward`` self-checkpoints when ``gradient_checkpointing`` is set
    (gated on ``self.training`` + grad enabled), read eagerly per block so
    flipping it per call costs no recompile. Snapshots + restores each block's
    checkpoint flags, so this composes with a global ``--grad_ckpt`` run.

    Arms the **unsloth-offload** variant, not standard ``torch_checkpoint``:
    ``block._forward`` is ``torch.compile``'d, and
    ``checkpoint(compiled_fn, use_reentrant=False)`` diverges from the inductor
    forward graph in recompute → ``CheckpointError``. The unsloth path carries
    ``@torch._disable_dynamo``, so ``_forward`` runs eager in both forward and
    recompute, and it offloads saved tensors to CPU. The reentrant grad-drop bug
    (see ``cdm_off_trajectory_loss``) does not apply here: the frozen
    teacher view has no grad-requiring params inside the region.

    Wraps ONLY the grad-bearing GAN gen teacher forward, reclaiming the peak
    VRAM the frozen teacher retains there purely to backprop into x_pred →
    student (numerically exact — no dropout).
    """
    saved = [
        (
            b.gradient_checkpointing,
            b.unsloth_offload_checkpointing,
        )
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


def teacher_anchor(
    ctx: RunContext,
    cfg,
    eps: torch.Tensor,
    crossattn_emb: torch.Tensor,
    c_null: torch.Tensor,
    B: int,
) -> torch.Tensor:
    """Teacher K-step CFG anchor (no grad) → ``v_target`` (DP-DMD only).

    Rolls the CFG-guided teacher ``k_anchor`` Euler steps from the shared noise ε
    and returns the average velocity ε→z_tk over ``[t_k, 1]`` — exactly the target
    for the student's t=1 first step. Detached.
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
class DmdResult:
    """Outputs of the DMD-on-x_θ surrogate (all fp32 / detached as noted)."""

    grad_signal: torch.Tensor  # detached DMD signal
    delta_dm: torch.Tensor  # v_real − v_fake (metrics)
    tau_dm: torch.Tensor  # DMD query τ (reused by the GAN gen renoise)
    tau_dm_e: torch.Tensor  # τ broadcast to (B,1,1,1)
    v_real_cond_dm: torch.Tensor
    v_fake_cond_dm: torch.Tensor
    eps_dm: torch.Tensor  # DMD renoise ε (reused by the GAN gen renoise)


def dmd_surrogate(
    ctx: RunContext,
    cfg,
    x_pred: torch.Tensor,
    crossattn_emb: torch.Tensor,
    c_null: torch.Tensor,
    B: int,
) -> DmdResult:
    """DMD on x_θ against the CFG-guided teacher + fake score.

    The real score MUST be CFG-GUIDED (v_u + α·(v_c − v_u)), not cond-only:
    without guidance v_real≈v_fake (both unguided cond preds collapse,
    dm_cos≈0.9999) and the quality gradient is noise. Fake stays cond-only.
    Query τ is uniform by design (independent of t_distribution).
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
    v_real_cond_dm = ctx.teacher_cfg_velocity(
        x_renoised_dm, tau_dm, crossattn_emb, c_null
    )
    v_fake_cond_dm = ctx.forward(
        "fake", x_renoised_dm, tau_dm, crossattn_emb, no_grad=True
    ).squeeze(2)
    delta_dm = v_real_cond_dm - v_fake_cond_dm

    tau_dm_e = tau_dm.view(B, 1, 1, 1).float()
    grad_dm = tau_dm_e * delta_dm.float()
    if cfg.dm_x0_norm:
        denom = (
            (tau_dm_e * v_real_cond_dm.float())
            .abs()
            .mean(dim=(1, 2, 3), keepdim=True)
            .clamp_min(cfg.norm_floor)
        )
        grad_dm = grad_dm / denom
    return DmdResult(
        grad_signal=grad_dm.detach(),
        delta_dm=delta_dm,
        tau_dm=tau_dm,
        tau_dm_e=tau_dm_e,
        v_real_cond_dm=v_real_cond_dm,
        v_fake_cond_dm=v_fake_cond_dm,
        eps_dm=eps_dm,
    )


def mfd_steps(span_max: float, dtau: float, max_steps: int) -> int:
    """Euler step count for one mean-flow integral: ceil(span / dtau), in [1, max_steps]."""
    k = max(1, int(-(-span_max // dtau)))
    return min(k, max(1, max_steps))


def _mean_velocity(v_fn, x_a, sa, sb, K) -> torch.Tensor:
    """Average Euler velocity of ``v_fn`` from σ=sa to σ=sb (per-sample, K uniform steps).

    ``(x_a − x_b) / (sa − sb)`` without the division: the uniform-step Euler
    displacement is ``(sa − sb)·mean_k v_k``, so the mean velocity is accumulated
    directly. At K=1 it is ``v_fn(x_a, sa)`` bit-for-bit.
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
    """Mean Flow Distillation signal on x_θ (arXiv:2606.11155), in the DMD slot.

    Renoise x_θ to σ_a, integrate the CFG'd teacher and the fake from σ_a to σ_b
    (K Euler steps, K = ceil((σ_a − σ_b)/dtau) ≤ max_steps), and push x_θ along
    the endpoint gap ``x_b_real − x_b_fake = −(σ_a − σ_b)·(U_real − U_fake)``.
    The ODE Jacobian is treated as identity (their §B.6), so the signal enters
    through the same ``(grad · x_θ)`` trick as DMD. ``interval="to_zero"`` with
    K=1 reproduces :func:`dmd_surrogate` exactly (same RNG draws, same values).

    Fields reuse the DMD names for metrics/GAN: ``tau_dm`` = σ_a (the GAN gen
    renoise level), ``tau_dm_e`` = σ_a − σ_b, ``v_*_cond_dm`` = U_real / U_fake.
    """
    sa = sample_t(
        B,
        distribution="uniform",
        sigmoid_scale=cfg.sigmoid_scale,
        device=ctx.device,
        dtype=ctx.dtype,
    )
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
        return ctx.forward("fake", x, sig, crossattn_emb, no_grad=True).squeeze(2)

    u_real = _mean_velocity(_v_real, x_a, sa, sb, K)
    u_fake = _mean_velocity(_v_fake, x_a, sa, sb, K)
    delta = u_real - u_fake

    span_e = (sa - sb).view(B, 1, 1, 1).float()
    grad = span_e * delta.float()
    if cfg.dm_x0_norm:
        denom = (
            (span_e * u_real.float())
            .abs()
            .mean(dim=(1, 2, 3), keepdim=True)
            .clamp_min(cfg.norm_floor)
        )
        grad = grad / denom
    return DmdResult(
        grad_signal=grad.detach(),
        delta_dm=delta,
        tau_dm=sa,
        tau_dm_e=span_e,
        v_real_cond_dm=u_real,
        v_fake_cond_dm=u_fake,
        eps_dm=eps_dm,
    )


def cdm_off_trajectory_loss(
    ctx: RunContext,
    cfg,
    cdm_src: tuple,
    crossattn_emb: torch.Tensor,
    c_null: torch.Tensor,
    latents: torch.Tensor,
    B: int,
) -> None:
    """L_CDM off-trajectory loss.

    From the grad step's on-trajectory (x_g, v_g, σ_g), Euler-extrapolate a large
    random stride to x_off at t' ~ U(0,1) (velocity-driven; detached, so this is
    a fresh leaf — one grad forward, no second BPTT chain). The student's local
    clean estimate there, x0_off = x_off − t'·v_off, gets the same signal as the
    DM branch: the real-vs-fake DMD surrogate. Supervises the truncation-drift region
    few-step Euler traverses off-manifold, which on-trajectory rollouts never
    visit.

    ORDER MATTERS: this branch must run BEFORE the GAN gen forward — that
    forward's checkpointed recompute happens at backward under the then-current
    view, so this must stay the last view flip of the step.

    VRAM: the CDM student forward is unsloth-checkpointed under ``cdm.grad_ckpt``
    and BACKWARDED IN-BRANCH, so its graph is freed before the GAN
    forward builds. The ckpt'd forward recomputes under the then-current view,
    so the view is restored to 'student' first — backward-while-view-live, the
    same contract the GAN forward honors.
    """
    x_g_cdm, v_g_cdm, s_g_cdm = cdm_src
    # CPU RNG (seeded by torch.manual_seed, no GPU sync), per-sample.
    t_off = torch.rand(B).to(device=ctx.device, dtype=ctx.dtype)
    # requires_grad_ is LOAD-BEARING under the unsloth ckpt below:
    # the reentrant path silently drops the LoRA param grads when
    # every explicit checkpoint input is detached — the leaf's grad flag
    # is what forces the autograd node.
    x_off = (
        cdm_extrapolate(x_g_cdm, v_g_cdm, s_g_cdm, t_off).to(ctx.dtype).requires_grad_()
    )
    cdm_ckpt = (
        selective_block_grad_ckpt(ctx.model) if cfg.cdm_grad_ckpt else nullcontext()
    )
    with cdm_ckpt:
        v_off = ctx.forward(
            "student", x_off, t_off, crossattn_emb, no_grad=False
        ).squeeze(2)
    # Local clean estimate at t' (their Eq. 1, in our v-param).
    x0_off = x_off.float() - t_off.view(B, 1, 1, 1).float() * v_off.float()
    tau_cdm = sample_t(
        B,
        distribution="uniform",
        sigmoid_scale=cfg.sigmoid_scale,
        device=ctx.device,
        dtype=ctx.dtype,
    )
    eps_cdm = torch.randn_like(latents)
    x_renoised_cdm = renoise(x0_off.detach().to(ctx.dtype), tau_cdm, eps_cdm)
    v_real_cdm = ctx.teacher_cfg_velocity(
        x_renoised_cdm, tau_cdm, crossattn_emb, c_null
    )
    # Transition-aware fake: x0_off is the student's clean estimate at t', so the
    # fake is queried as for a step at σ = t'; the loop's σ_g is restored after.
    ctx.turbo.set_fake_source(t_off)
    v_fake_cdm = ctx.forward(
        "fake", x_renoised_cdm, tau_cdm, crossattn_emb, no_grad=True
    ).squeeze(2)
    ctx.turbo.set_fake_source(s_g_cdm)
    delta_cdm = v_real_cdm - v_fake_cdm
    tau_cdm_e = tau_cdm.view(B, 1, 1, 1).float()
    grad_cdm = tau_cdm_e * delta_cdm.float()
    if cfg.dm_x0_norm:
        denom_cdm = (
            (tau_cdm_e * v_real_cdm.float())
            .abs()
            .mean(dim=(1, 2, 3), keepdim=True)
            .clamp_min(cfg.norm_floor)
        )
        grad_cdm = grad_cdm / denom_cdm
    grad_cdm = grad_cdm.detach()
    cdm_loss = (grad_cdm * x0_off).mean()
    # Backward NOW, with the student view restored: the ckpt'd forward
    # above recomputes here and must see the view it was recorded
    # under (the delta forwards flipped it to teacher/fake). Frees the
    # CDM graph before the GAN forward; grads accumulate, grad_clip
    # below clips the full student gradient once.
    ctx.turbo.set_view("student")
    (cfg.cdm_weight * cdm_loss).backward()
    ctx.metrics.add_cdm(grad_cdm)


def gan_generator_term(
    ctx: RunContext,
    cfg,
    x_pred: torch.Tensor,
    tau_dm: torch.Tensor,
    eps_dm: torch.Tensor,
    crossattn_emb: torch.Tensor,
    gan_weight: float,
) -> torch.Tensor:
    """GAN generator term.

    The disc scores the frozen TEACHER's block features of the student's renoised
    x_pred. Grad must flow into x_pred → student, so this renoise keeps x_pred
    attached (unlike the DMD path) and the teacher forward is grad-enabled; the
    disc itself is frozen here. return_features_early stops after the deepest
    tapped block (half-depth grad forward — full-stack OOM'd).

    No-op when the GAN is off, or while ``gan_weight`` is still 0 inside the
    delay window — skips the grad-bearing half-depth teacher forward entirely
    (no RNG draws here, so the skip leaves the step's random stream unchanged),
    returning a zero loss.
    """
    turbo = ctx.turbo
    if turbo.disc is None or gan_weight == 0.0:
        return torch.zeros((), device=ctx.device)

    turbo.set_disc_requires_grad(False)
    x_renoised_gan = renoise(x_pred, tau_dm, eps_dm)  # grad-bearing
    # Selectively checkpoint just this forward (the only GAN extra that
    # retains a backward graph); recompute trades ~half-depth compute for
    # the ~3 GB of retained teacher activations. nullcontext when off.
    gan_ckpt = (
        selective_block_grad_ckpt(ctx.model) if cfg.gan_grad_ckpt else nullcontext()
    )
    with gan_ckpt:
        feats_gen = ctx.forward(
            "teacher",
            x_renoised_gan,
            tau_dm,
            crossattn_emb,
            no_grad=False,
            return_block_features=turbo.gan_feature_set,
            return_features_early=True,
        )
    fake_logits_gen = turbo.disc(
        turbo.features_in_order(feats_gen)
    )  # (B, taps), grad→x_pred
    return gan_loss_generator(fake_logits_gen)


def fake_update(
    ctx: RunContext,
    cfg,
    x_pred: torch.Tensor,
    latents: torch.Tensor,
    crossattn_emb: torch.Tensor,
    B: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fake (critic) + discriminator update against ``x_pred.detach()``.

    ``fake_steps_per_student_step`` inner updates, resampling (τ_fake, ε_fake)
    each iteration (keeps the fake's target ahead of the moving x_pred dist).
    When the GAN is on, the discriminator steps once per fake inner step
    (FastGen cadence). Runs the fake + disc optimizer/scheduler steps in-place;
    returns (all detached, for logging) mean fake loss, mean disc loss, mean
    disc margin (mean real − mean fake logit — the "disc winning" signal the
    hinge means hide, since both sit at equilibrium ≈0.69/1.39 even while
    per-logit structure does the damage), and mean per-logit spread (std over
    the fake branch's logits; 0 when the head emits a single logit).
    """
    turbo = ctx.turbo
    device, dtype = ctx.device, ctx.dtype
    x_pred_d = x_pred.detach()
    fake_loss_sum = torch.zeros((), device=device)
    gan_disc_sum = torch.zeros((), device=device)
    gan_margin_sum = torch.zeros((), device=device)
    gan_spread_sum = torch.zeros((), device=device)
    for _ in range(cfg.fake_steps_per_student_step):
        tau_fake = sample_t(
            B,
            distribution=cfg.t_distribution,
            sigmoid_scale=cfg.sigmoid_scale,
            device=device,
            dtype=dtype,
        )
        eps_fake = torch.randn_like(x_pred_d)
        x_t_fake = renoise(x_pred_d, tau_fake, eps_fake).requires_grad_()
        v_fake = ctx.forward(
            "fake", x_t_fake, tau_fake, crossattn_emb, no_grad=False
        ).squeeze(2)
        target_v_fake = eps_fake - x_pred_d  # flow-matching target
        fake_loss = nn.functional.mse_loss(v_fake.float(), target_v_fake.float())
        fake_loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(turbo.fake_params(), max_norm=cfg.grad_clip)
        ctx.fake_opt.step()
        ctx.fake_opt.zero_grad(set_to_none=True)
        ctx.fake_sched.step()
        fake_loss_sum = fake_loss_sum + fake_loss.detach()
        ctx.tau_profile.add(fake_loss, tau_fake)

        # Discriminator update, co-located with the fake/critic update
        # (FastGen cadence). The disc scores frozen-TEACHER block features of
        # renoised fake (x_pred) vs renoised real latents — grad only to the disc
        # head. gan_use_same_t_noise reuses (τ_fake, ε_fake) for the real branch.
        if turbo.disc is not None:
            turbo.set_disc_requires_grad(True)
            if cfg.gan_use_same_t_noise:
                tau_d, eps_d = tau_fake, eps_fake
            else:
                tau_d = sample_t(
                    B,
                    distribution=cfg.t_distribution,
                    sigmoid_scale=cfg.sigmoid_scale,
                    device=device,
                    dtype=dtype,
                )
                eps_d = torch.randn_like(x_pred_d)

            # Feature-only teacher forwards (no_grad → grad only to the disc
            # head). Early-exit at the deepest tap; each call returns its own
            # feature dict, so the fake/real captures never alias.
            def _disc_feats(latent_in):
                return turbo.features_in_order(
                    ctx.forward(
                        "teacher",
                        renoise(latent_in, tau_d, eps_d),
                        tau_d,
                        crossattn_emb,
                        no_grad=True,
                        return_block_features=turbo.gan_feature_set,
                        return_features_early=True,
                    )
                )

            fake_logits_d = turbo.disc(_disc_feats(x_pred_d))
            real_logits_d = turbo.disc(_disc_feats(latents))
            loss_disc = gan_loss_discriminator(real_logits_d, fake_logits_d)
            with torch.no_grad():
                gan_margin_sum = gan_margin_sum + (
                    real_logits_d.mean() - fake_logits_d.mean()
                )
                fl = fake_logits_d.flatten()
                if fl.numel() > 1:  # host-side shape check, no sync
                    gan_spread_sum = gan_spread_sum + fl.std()

            # Approximate-R1 (APT): penalize disc logit change under a small
            # perturbation of the real disc input. Perturb the renoised real
            # latent directly (the tensor whose features feed the disc).
            if cfg.gan_r1_weight > 0.0:
                x_t_real_a = renoise(
                    latents, tau_d, eps_d
                ) + cfg.gan_r1_alpha * torch.randn_like(latents)
                feats_a = ctx.forward(
                    "teacher",
                    x_t_real_a,
                    tau_d,
                    crossattn_emb,
                    no_grad=True,
                    return_block_features=turbo.gan_feature_set,
                    return_features_early=True,
                )
                real_logits_a = turbo.disc(turbo.features_in_order(feats_a))
                loss_disc = loss_disc + cfg.gan_r1_weight * nn.functional.mse_loss(
                    real_logits_d, real_logits_a
                )

            loss_disc.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    turbo.disc_params(), max_norm=cfg.grad_clip
                )
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
    )
