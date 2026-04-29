"""APEX self-adversarial distillation loss.

Implements paper Eq. 25's L_APEX = lam_p * L_fake + lam_c * L_mix
(arXiv:2604.12322). No separate L_sup term — supervision enters via the
inner mixing coefficient `lambda` inside T_mix = (1-lambda)*x + lambda*v_fake.
At lambda=0, L_mix collapses to pure FM (Theorem 1, App. B.5), which is
how the cold-start guard provides a bootstrap signal.

Shapes:
    All four aux tensors are [B, C, H, W] (the Anima latent shape, after the
    squeeze(2) in get_noise_pred_and_target).

Schedule:
    The cold-start guard ramps two things together:
      - inner lambda (T_mix mixing coefficient) from 0 -> apex_lambda_target
      - outer L_fake weight (lam_f) from 0 -> apex_lambda_p_target
    L_mix's outer weight (apex_lambda_c) is constant — its strength is
    governed by the inner-lambda ramp. During warmup, lambda=0 and lam_f=0,
    so the loss reduces to apex_lambda_c * pure_FM (since T_mix at lambda=0
    is the data target). This is the Anima-specific bootstrap; the paper has
    no warmup because at 20B scale random-init F_theta is already small
    relative to data.

    In typical use, warmup_steps / rampup_steps are computed at the call site
    as `apex_warmup_ratio * max_train_steps` / `apex_rampup_ratio * max_train_steps`
    so the schedule tracks the actual step budget; train.py resolves the
    ratio and passes absolute step counts here. Explicit step overrides
    (apex_warmup_steps / apex_rampup_steps > 0) short-circuit the ratio.

The scheduler is step-based; the caller is responsible for passing an
up-to-date global_step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ApexAux:
    """Auxiliary tensors stashed during the real+fake forwards.

    F_real is the real-branch velocity prediction (same as the normal
    model_pred). T_mix_v is the velocity-space consistency target used by
    L_mix (Eq. 23/24 after Prop. 3's t^2 equivalence). F_fake_on_fake_xt
    and target_fake are the fake-flow-fitting inputs for L_fake (Eq. 12).

    All tensors except F_real and F_fake_on_fake_xt must be detached.
    """

    F_real: "object"  # [B, C, H, W] — grad flows; same as model_pred
    T_mix_v: "object"  # [B, C, H, W] — detached (contains v_fake_sg)
    F_fake_on_fake_xt: "object"  # [B, C, H, W] — grad flows into LoRA + shift
    target_fake: "object"  # [B, C, H, W] — detached
    weighting: Optional["object"] = None  # [B] weighting for L_mix (shared t)
    weighting_fake: Optional["object"] = None  # [B] weighting for L_fake (t_fake)


def apex_schedule_weights(
    step: int,
    warmup_steps: int,
    rampup_steps: int,
    lam_inner_target: float,
    lam_f_target: float,
) -> tuple[float, float]:
    """Compute the effective inner lambda and outer L_fake weight at `step`.

    Returns (lam_inner_eff, lam_f_eff). During warmup both are 0.0 — this
    drives L_mix at lambda=0 (pure FM via T_mix=x) and zeroes L_fake. Over
    rampup, both linearly ramp to their targets. After rampup, holds.

    Phase 0 Finding B (Anima): cold-start catastrophically regresses
    (-48% NFE=1 W1) because L_fake trains the fake branch against random
    trajectories when F_theta is at init. The schedule keeps L_fake off and
    inner lambda at 0 until the real branch has learned a useful FM mapping.
    """
    warmup_steps = max(0, int(warmup_steps))
    rampup_steps = max(0, int(rampup_steps))
    if step < warmup_steps:
        return 0.0, 0.0
    if rampup_steps == 0:
        return float(lam_inner_target), float(lam_f_target)
    progress = min(1.0, (step - warmup_steps) / float(rampup_steps))
    return lam_inner_target * progress, lam_f_target * progress
