"""Fake (critic) head-start before the main loop (``optim.fake_warmup_steps``).

The student LR warmup ends at ~0.02·iterations; with an uncalibrated fake, delta_dm
is large and misaligned there and grad_signal_rms / v_student_rms spike near step 50.
The head-start trains the fake alone against the student's x_pred first.

``fake_warmup_steps`` counts fake updates, one per batch, each on a fresh no-grad
student x_pred; ``fake_steps_per_student_step`` does not scale it. The fake
scheduler is sized over ``iterations · fake_steps_per_student_step +
fake_warmup_steps``, so its LR warmup overlaps the head-start and the fake enters
the main loop at full LR.
"""

from __future__ import annotations

import logging

import torch
from tqdm import tqdm

from library.training.forward import renoise

from .metrics import TauBinCriticLoss
from .setup import RunContext
from .steps import fake_regression_step, sample_fake_tau

logger = logging.getLogger(__name__)


def run_fake_warmup(ctx: RunContext, cfg) -> None:
    steps = cfg.fake_warmup_steps
    if steps <= 0:
        return
    logger.info(f"fake (critic) head-start: {steps} fake-only updates")
    tau_profile = TauBinCriticLoss(ctx.device, prefix="warmup/fake_loss_tau")
    for i in tqdm(range(steps), desc="fake-warmup"):
        batch = ctx.next_batch()
        latents, crossattn_emb = batch.latents, batch.crossattn_emb
        B = latents.shape[0]
        torch.compiler.cudagraph_mark_step_begin()

        t = sample_fake_tau(ctx, cfg, B)
        x_t = renoise(latents, t, torch.randn_like(latents))
        with torch.no_grad():
            v_student = ctx.forward("student", x_t, t, crossattn_emb, no_grad=True)
            x_pred_d = (x_t - t.view(B, 1, 1, 1) * v_student).detach()

        tau = sample_fake_tau(ctx, cfg, B)
        eps = torch.randn_like(x_pred_d)
        fake_loss, _ = fake_regression_step(ctx, cfg, x_pred_d, crossattn_emb, tau, eps)
        tau_profile.add(fake_loss, tau)
        if (i + 1) % cfg.log_interval == 0:
            if ctx.writer is not None:
                ctx.writer.add_scalar("warmup/fake_loss", fake_loss.item(), i + 1)
            tau_profile.write(ctx.writer, i + 1)
