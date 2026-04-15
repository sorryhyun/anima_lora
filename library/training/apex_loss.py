"""APEX self-adversarial distillation loss.

Implements L_sup + L_mix + L_fake from the APEX paper (arXiv:2604.12322, Eq.
12, 20, 24). Gradient-equivalent to the G_APEX formulation by Theorem 1 / B.5,
verified per-sample in Phase 0 to fp32 noise.

Shapes:
    All four aux tensors are [B, C, H, W] (the Anima latent shape, after the
    squeeze(2) in get_noise_pred_and_target).

Schedule:
    lam_c and lam_f are gated by a warmup + linear rampup schedule to avoid
    the cold-start catastrophic regression observed in Phase 0. For the first
    `warmup_steps` steps, pure L_sup runs (matches standard FM). Over the
    next `rampup_steps` the two adversarial weights linearly ramp from 0 to
    their target values. After that, full APEX.

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
    weighting: Optional["object"] = None  # [B] weighting for L_sup/L_mix (shared t)
    weighting_fake: Optional["object"] = None  # [B] weighting for L_fake (t_fake)


def apex_schedule_weights(
    step: int,
    warmup_steps: int,
    rampup_steps: int,
    lam_c_target: float,
    lam_f_target: float,
) -> tuple[float, float]:
    """Compute the effective lam_c, lam_f at the given global step.

    Phase 0 Finding B: cold-start catastrophically regresses (-48% NFE=1 W1)
    because L_fake trains the fake branch against random trajectories when
    F_theta is at init. The guard is to gate both adversarial terms behind a
    warmup + linear rampup.

    Returns (0, 0) during warmup, linearly ramps to (lam_c_target, lam_f_target)
    over rampup_steps, then holds at target.
    """
    warmup_steps = max(0, int(warmup_steps))
    rampup_steps = max(0, int(rampup_steps))
    if step < warmup_steps:
        return 0.0, 0.0
    if rampup_steps == 0:
        return float(lam_c_target), float(lam_f_target)
    progress = min(1.0, (step - warmup_steps) / float(rampup_steps))
    return lam_c_target * progress, lam_f_target * progress
