"""Loss registry + composer.

M1 extraction (plan.md): the loss side of `_process_batch_inner` becomes a
registry of small callables. The composer calls the active handlers in three
phases matching the pre-refactor reduction order — break that ordering and
APEX / ortho / multiscale numerics shift.

Reduction order (must match train.py pre-refactor):
  1. Per-sample [B] stage:
       flow_match   — base FM (weighting + masked + loss_weights)
       apex_mix     — L_mix scaled by lam_c_eff (adds [B])
       apex_fake    — L_fake scaled by lam_f_eff (adds [B])
     flow_match is additionally multiplied by args.apex_lambda_p when APEX is
     active (matches the L_sup scalar in the paper).
  2. Per-sample += scalar broadcast stage (was `post_process_loss`):
       ortho_reg     — OrthoLoRA orthogonality regularizer
       hydra_balance — MoE load-balance loss
       functional    — postfix-func inversion MSE
  3. Scalar stage (after `.mean()` reduction):
       multiscale    — avg_pool2d MSE on pred/target

The composer does not own forward passes. APEX and functional-loss forwards
still happen inside the trainer (they need `anima()` and post_process_network
hooks). Those forwards stash their aux tensors on `LossContext.aux`, and the
composer consumes them.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from library.custom_train_functions import apply_masked_loss


def _conditional_loss(*args, **kwargs):
    # Lazy: library.train_util imports library.training, so a top-level import
    # here would close the cycle. Cached on first call by Python's import system.
    from library import train_util

    return train_util.conditional_loss(*args, **kwargs)


# ---------------------------------------------------------------------------
# Context + types
# ---------------------------------------------------------------------------


@dataclass
class LossContext:
    args: argparse.Namespace
    batch: dict
    model_pred: torch.Tensor
    target: torch.Tensor
    timesteps: torch.Tensor
    weighting: Optional[torch.Tensor]
    huber_c: Optional[torch.Tensor]
    loss_weights: torch.Tensor
    network: object
    aux: dict = field(default_factory=dict)


LossFn = Callable[[LossContext], torch.Tensor]


# ---------------------------------------------------------------------------
# Per-sample losses ([B])
# ---------------------------------------------------------------------------


def _flow_match_loss(ctx: LossContext) -> torch.Tensor:
    """Base rectified-flow MSE with weighting, masked loss, per-sample weight.

    Mirrors train.py lines 1041–1055 (pre-M1). Returns a [B] tensor.
    """
    loss = _conditional_loss(
        ctx.model_pred.float(),
        ctx.target.float(),
        ctx.args.loss_type,
        "none",
        ctx.huber_c,
    )
    if ctx.weighting is not None:
        loss = loss * ctx.weighting
    if ctx.args.masked_loss or (
        "alpha_masks" in ctx.batch and ctx.batch["alpha_masks"] is not None
    ):
        loss = apply_masked_loss(loss, ctx.batch)
    loss = loss.mean(dim=list(range(1, loss.ndim)))
    loss = loss * ctx.loss_weights
    return loss


def _apex_mix_loss(ctx: LossContext) -> torch.Tensor:
    """APEX L_mix: MSE(F_real, T_mix_v) gated by lam_c_eff. [B]."""
    apex_aux = ctx.aux.get("apex") or {}
    lam_c_eff = float(apex_aux.get("lam_c_eff", 0.0))
    T_mix_v = apex_aux.get("T_mix_v")
    if lam_c_eff <= 0.0 or T_mix_v is None:
        return ctx.model_pred.new_zeros(ctx.model_pred.shape[0])

    l_mix = _conditional_loss(
        ctx.model_pred.float(),
        T_mix_v.float(),
        ctx.args.loss_type,
        "none",
        ctx.huber_c,
    )
    if ctx.weighting is not None:
        l_mix = l_mix * ctx.weighting
    if ctx.args.masked_loss or (
        "alpha_masks" in ctx.batch and ctx.batch["alpha_masks"] is not None
    ):
        l_mix = apply_masked_loss(l_mix, ctx.batch)
    l_mix = l_mix.mean(dim=list(range(1, l_mix.ndim)))
    l_mix = l_mix * ctx.loss_weights
    return lam_c_eff * l_mix


def _apex_fake_loss(ctx: LossContext) -> torch.Tensor:
    """APEX L_fake: MSE(F_fake_on_fake_xt, target_fake) gated by lam_f_eff. [B]."""
    apex_aux = ctx.aux.get("apex") or {}
    lam_f_eff = float(apex_aux.get("lam_f_eff", 0.0))
    F_fake_on_fake_xt = apex_aux.get("F_fake_on_fake_xt")
    target_fake = apex_aux.get("target_fake")
    if lam_f_eff <= 0.0 or F_fake_on_fake_xt is None or target_fake is None:
        return ctx.model_pred.new_zeros(ctx.model_pred.shape[0])

    l_fake = _conditional_loss(
        F_fake_on_fake_xt.float(),
        target_fake.float(),
        ctx.args.loss_type,
        "none",
        ctx.huber_c,
    )
    w_fake = apex_aux.get("weighting_fake")
    if w_fake is not None:
        l_fake = l_fake * w_fake
    # No masked-loss: x_fake is synthetic, not tied to the input image mask.
    l_fake = l_fake.mean(dim=list(range(1, l_fake.ndim)))
    l_fake = l_fake * ctx.loss_weights
    return lam_f_eff * l_fake


# ---------------------------------------------------------------------------
# Scalar-broadcast regularizers (added to the per-sample [B] tensor)
# ---------------------------------------------------------------------------


def _ortho_reg_loss(ctx: LossContext) -> torch.Tensor:
    weight = float(getattr(ctx.network, "_ortho_reg_weight", 0.0) or 0.0)
    if weight <= 0.0:
        return ctx.model_pred.new_zeros(())
    return weight * ctx.network.get_ortho_regularization()


def _hydra_balance_loss(ctx: LossContext) -> torch.Tensor:
    weight = float(getattr(ctx.network, "_balance_loss_weight", 0.0) or 0.0)
    if weight <= 0.0:
        return ctx.model_pred.new_zeros(())
    return weight * ctx.network.get_balance_loss()


def _functional_loss(ctx: LossContext) -> torch.Tensor:
    weight = float(getattr(ctx.args, "functional_loss_weight", 0.0) or 0.0)
    func_loss = ctx.aux.get("func_loss")
    if weight <= 0.0 or func_loss is None:
        return ctx.model_pred.new_zeros(())
    # Per-sample running loss is float32 (flow_match casts inputs via .float()).
    # Match the pre-refactor cast: `func_weight * func_loss.to(loss.dtype)`.
    return weight * func_loss.float()


def _condition_shift_loss(ctx: LossContext) -> torch.Tensor:
    """Reserved slot: APEX's ConditionShift is trained implicitly via L_fake's
    autograd path. No explicit loss term today; registered for symmetry with
    the plan and in case a direct regularizer is added later.
    """
    return ctx.model_pred.new_zeros(())


# ---------------------------------------------------------------------------
# Scalar post-reduction losses (operate on the scalar mean of the per-sample)
# ---------------------------------------------------------------------------


def _multiscale_loss(ctx: LossContext) -> torch.Tensor:
    """Additional MSE term at 2x-downsampled resolution. Scalar output meant to
    be blended into the scalar mean via `(scalar + ms*ms_w) / (1 + ms_w)`.
    The composer applies that blend — this handler returns the raw MSE.
    """
    ms_weight = float(getattr(ctx.args, "multiscale_loss_weight", 0.0) or 0.0)
    if ms_weight <= 0.0:
        return ctx.model_pred.new_zeros(())
    h, w = ctx.model_pred.shape[-2:]
    side_length = math.sqrt(h * w) * 8
    if side_length < 1024 * 0.9 or h < 2 or w < 2:
        return ctx.model_pred.new_zeros(())
    pred_ds = torch.nn.functional.avg_pool2d(ctx.model_pred.float(), 2)
    target_ds = torch.nn.functional.avg_pool2d(ctx.target.float(), 2)
    return torch.nn.functional.mse_loss(pred_ds, target_ds)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


LOSS_REGISTRY: dict[str, LossFn] = {
    "flow_match": _flow_match_loss,
    "apex_mix": _apex_mix_loss,
    "apex_fake": _apex_fake_loss,
    "ortho_reg": _ortho_reg_loss,
    "hydra_balance": _hydra_balance_loss,
    "functional": _functional_loss,
    "condition_shift": _condition_shift_loss,
    "multiscale": _multiscale_loss,
}


# Which stage each registered loss runs in (see module docstring).
_STAGE_PER_SAMPLE = ("flow_match", "apex_mix", "apex_fake")
_STAGE_SCALAR_BROADCAST = (
    "ortho_reg",
    "hydra_balance",
    "functional",
    "condition_shift",
)
_STAGE_SCALAR_POST = ("multiscale",)
# _STAGE_SCALAR_POST is consulted by LossComposer.compose via the hard-coded
# multiscale branch; kept as a named constant for documentation / future
# extensibility.
__all__ = [
    "LossContext",
    "LossComposer",
    "LossFn",
    "LOSS_REGISTRY",
    "build_loss_composer",
    "_STAGE_PER_SAMPLE",
    "_STAGE_SCALAR_BROADCAST",
    "_STAGE_SCALAR_POST",
]


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------


@dataclass
class LossComposer:
    """Holds the active loss entries and composes them in-order.

    `active_losses` is a list of names that exist in LOSS_REGISTRY. The
    composer groups them by stage and applies them. `build_loss_composer`
    decides which names to include based on `args` + `network`.
    """

    active_losses: list[str]
    # APEX lambda_p target. Applied to flow_match only when ctx.aux has an
    # "apex" block — matches pre-refactor gating (is_train + apex_aux is not None).
    l_sup_scalar: float = 1.0

    def compose(self, ctx: LossContext) -> torch.Tensor:
        per_sample = ctx.model_pred.new_zeros(ctx.model_pred.shape[0])
        apex_active_this_batch = bool(ctx.aux.get("apex"))

        # Stage 1: per-sample losses.
        first = True
        for name in _STAGE_PER_SAMPLE:
            if name not in self.active_losses:
                continue
            contribution = LOSS_REGISTRY[name](ctx)
            if (
                name == "flow_match"
                and apex_active_this_batch
                and self.l_sup_scalar != 1.0
            ):
                contribution = contribution * self.l_sup_scalar
            per_sample = contribution if first else (per_sample + contribution)
            first = False
        if first:
            # flow_match should always be present; defend against a caller
            # passing an empty composer.
            raise RuntimeError(
                "LossComposer: no per-sample loss registered; "
                "'flow_match' must be among active_losses"
            )

        # Stage 2: scalar-broadcast regularizers (added to the per-sample [B]).
        for name in _STAGE_SCALAR_BROADCAST:
            if name not in self.active_losses:
                continue
            reg = LOSS_REGISTRY[name](ctx)
            if reg is None:
                continue
            per_sample = per_sample + reg  # broadcast scalar -> [B]

        scalar = per_sample.mean()

        # Stage 3: scalar-level blend (multiscale).
        if "multiscale" in self.active_losses:
            ms_weight = float(getattr(ctx.args, "multiscale_loss_weight", 0.0) or 0.0)
            if ms_weight > 0.0:
                ms_loss = LOSS_REGISTRY["multiscale"](ctx)
                if ms_loss is not None and torch.is_tensor(ms_loss) and ms_loss.numel():
                    # pre-refactor: (scalar + ms * ms_w) / (1 + ms_w), only when
                    # side_length >= 0.9 * 1024. The guard is inside
                    # _multiscale_loss and returns 0 when it shouldn't apply —
                    # check against zero to preserve exact behavior.
                    if not (ms_loss == 0).all():
                        scalar = (scalar + ms_loss * ms_weight) / (1.0 + ms_weight)

        return scalar


def build_loss_composer(args: argparse.Namespace, network: object) -> LossComposer:
    """Inspect args + network and return the active LossComposer.

    Rules (pre-refactor parity):
      - flow_match is always active.
      - apex_mix / apex_fake active iff args.method == "apex".
      - ortho_reg active iff network._ortho_reg_weight > 0.
      - hydra_balance active iff network._balance_loss_weight > 0.
      - functional active iff args.functional_loss_weight > 0.
      - multiscale active iff args.multiscale_loss_weight > 0.
      - condition_shift is included whenever apex is active (reserved).

    l_sup_scalar becomes args.apex_lambda_p when apex is active, else 1.0.
    """
    active: list[str] = ["flow_match"]
    l_sup_scalar = 1.0

    is_apex = getattr(args, "method", None) == "apex"
    if is_apex:
        active.extend(["apex_mix", "apex_fake", "condition_shift"])
        l_sup_scalar = float(getattr(args, "apex_lambda_p", 1.0))

    if float(getattr(network, "_ortho_reg_weight", 0.0) or 0.0) > 0.0:
        active.append("ortho_reg")
    if float(getattr(network, "_balance_loss_weight", 0.0) or 0.0) > 0.0:
        active.append("hydra_balance")
    if float(getattr(args, "functional_loss_weight", 0.0) or 0.0) > 0.0:
        active.append("functional")
    if float(getattr(args, "multiscale_loss_weight", 0.0) or 0.0) > 0.0:
        active.append("multiscale")

    return LossComposer(active_losses=active, l_sup_scalar=l_sup_scalar)
