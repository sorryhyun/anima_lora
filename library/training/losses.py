"""Loss registry + composer.

M1 extraction (plan.md): the loss side of `_process_batch_inner` becomes a
registry of small callables. The composer calls the active handlers in three
phases matching the pre-refactor reduction order — break that ordering and
APEX / ortho / multiscale numerics shift.

Reduction order (must match train.py pre-refactor):
  1. Per-sample [B] stage:
       flow_match   — base FM (weighting + masked + loss_weights). Skipped
                      when APEX is active — apex_mix subsumes it (T_mix at
                      inner lambda=0 is exactly v_data, so L_mix = pure FM).
       apex_mix     — L_mix scaled by args.apex_lambda_c (constant). Inner
                      lambda is ramped 0 -> apex_lambda inside T_mix_v at
                      forward time, so the warmup schedule lives there, not
                      in the outer L_mix weight. [B]
       apex_fake    — L_fake scaled by aux["apex"]["lam_f_eff"] (ramped). [B]
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


def add_custom_train_arguments(
    parser: argparse.ArgumentParser, support_weighted_captions: bool = True
):
    parser.add_argument(
        "--min_snr_gamma",
        type=float,
        default=None,
        help="gamma for reducing the weight of high loss timesteps. Lower numbers have stronger effect. 5 is recommended by paper.",
    )
    parser.add_argument(
        "--scale_v_pred_loss_like_noise_pred",
        action="store_true",
        help="scale v-prediction loss like noise prediction loss",
    )
    parser.add_argument(
        "--v_pred_like_loss",
        type=float,
        default=None,
        help="add v-prediction like loss multiplied by this value",
    )
    parser.add_argument(
        "--debiased_estimation_loss",
        action="store_true",
        help="debiased estimation loss",
    )
    if support_weighted_captions:
        parser.add_argument(
            "--weighted_captions",
            action="store_true",
            default=False,
            help="Enable weighted captions in the standard style (token:1.3).",
        )


def apply_masked_loss(loss, batch) -> torch.FloatTensor:
    if "conditioning_images" in batch:
        mask_image = (
            batch["conditioning_images"].to(dtype=loss.dtype)[:, 0].unsqueeze(1)
        )  # use R channel
        mask_image = mask_image / 2 + 0.5
    elif "alpha_masks" in batch and batch["alpha_masks"] is not None:
        mask_image = (
            batch["alpha_masks"].to(dtype=loss.dtype).unsqueeze(1)
        )  # add channel dimension
    else:
        return loss

    mask_image = torch.nn.functional.interpolate(
        mask_image, size=loss.shape[2:], mode="area"
    )
    loss = loss * mask_image
    return loss


def get_huber_threshold_if_needed(
    args, timesteps: torch.Tensor, noise_scheduler
) -> Optional[torch.Tensor]:
    if args.loss_type == "pseudo_huber":
        b_size = timesteps.shape[0]
        return torch.full((b_size,), args.pseudo_huber_c, device=timesteps.device)
    if not (args.loss_type == "huber" or args.loss_type == "smooth_l1"):
        return None

    b_size = timesteps.shape[0]
    if args.huber_schedule == "exponential":
        alpha = -math.log(args.huber_c) / noise_scheduler.config.num_train_timesteps
        result = torch.exp(-alpha * timesteps) * args.huber_scale
    elif args.huber_schedule == "snr":
        if not hasattr(noise_scheduler, "alphas_cumprod"):
            raise NotImplementedError(
                "Huber schedule 'snr' is not supported with the current model."
            )
        alphas_cumprod = torch.index_select(
            noise_scheduler.alphas_cumprod, 0, timesteps.cpu()
        )
        sigmas = ((1.0 - alphas_cumprod) / alphas_cumprod) ** 0.5
        result = (1 - args.huber_c) / (1 + sigmas) ** 2 + args.huber_c
        result = result.to(timesteps.device)
    elif args.huber_schedule == "constant":
        result = torch.full(
            (b_size,), args.huber_c * args.huber_scale, device=timesteps.device
        )
    else:
        raise NotImplementedError(f"Unknown Huber loss schedule {args.huber_schedule}!")

    return result


def conditional_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str,
    reduction: str,
    huber_c: Optional[torch.Tensor] = None,
):
    if loss_type == "l2":
        loss = torch.nn.functional.mse_loss(model_pred, target, reduction=reduction)
    elif loss_type == "l1":
        loss = torch.nn.functional.l1_loss(model_pred, target, reduction=reduction)
    elif loss_type == "huber":
        if huber_c is None:
            raise NotImplementedError("huber_c not implemented correctly")
        huber_c = huber_c.view(-1, *([1] * (model_pred.ndim - 1)))
        loss = (
            2
            * huber_c
            * (torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
        )
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    elif loss_type == "smooth_l1":
        if huber_c is None:
            raise NotImplementedError("huber_c not implemented correctly")
        huber_c = huber_c.view(-1, *([1] * (model_pred.ndim - 1)))
        loss = 2 * (torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    elif loss_type == "pseudo_huber":
        if huber_c is None:
            raise ValueError("pseudo_huber_c is required for pseudo_huber loss")
        huber_c = huber_c.view(-1, *([1] * (model_pred.ndim - 1)))
        loss = torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    else:
        raise NotImplementedError(f"Unsupported Loss Type: {loss_type}")
    return loss


# Internal alias — still referenced below by the composer stages.
_conditional_loss = conditional_loss


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
    """APEX L_mix: MSE(F_real, T_mix_v) scaled by args.apex_lambda_c. [B].

    Outer L_mix weight (paper Eq. 25 lam_c) is constant across training. The
    supervision/adversarial blend is governed by the inner lambda baked into
    T_mix_v at forward time (see ApexMethodAdapter.extra_forwards): at
    inner=0 T_mix_v == v_data so this is pure FM.
    """
    apex_aux = ctx.aux.get("apex") or {}
    T_mix_v = apex_aux.get("T_mix_v")
    if T_mix_v is None:
        return ctx.model_pred.new_zeros(ctx.model_pred.shape[0])
    lam_c = float(getattr(ctx.args, "apex_lambda_c", 1.0))
    if lam_c <= 0.0:
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
    return lam_c * l_mix


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


def _postfix_contrastive_loss(ctx: LossContext) -> torch.Tensor:
    """Inter-caption contrastive on PostfixNetwork cond_mlp outputs. Pushes
    cond_mlp toward caption-varying outputs against a MoCo-style memory queue
    — addresses the empirical collapse to a single caption-agnostic direction
    (see `archive/bench/postfix/initial_postfix_problems.md`)."""
    weight = float(getattr(ctx.network, "contrastive_weight", 0.0) or 0.0)
    if weight <= 0.0 or not hasattr(ctx.network, "get_contrastive_loss"):
        return ctx.model_pred.new_zeros(())
    loss = ctx.network.get_contrastive_loss()
    return weight * loss.to(ctx.model_pred.dtype)


def _postfix_sigma_budget_loss(ctx: LossContext) -> torch.Tensor:
    """Soft L2 budget on ‖sigma_residual‖² so the σ-branch doesn't dominate
    cond_mlp. Without this, empirical residual/base ≈ 2.5 and σ swallows the
    contrastive signal."""
    weight = float(getattr(ctx.network, "sigma_budget_weight", 0.0) or 0.0)
    if weight <= 0.0 or not hasattr(ctx.network, "get_sigma_budget_loss"):
        return ctx.model_pred.new_zeros(())
    loss = ctx.network.get_sigma_budget_loss()
    return weight * loss.to(ctx.model_pred.dtype)


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
    "postfix_contrastive": _postfix_contrastive_loss,
    "postfix_sigma_budget": _postfix_sigma_budget_loss,
}


# Which stage each registered loss runs in (see module docstring).
_STAGE_PER_SAMPLE = ("flow_match", "apex_mix", "apex_fake")
_STAGE_SCALAR_BROADCAST = (
    "ortho_reg",
    "hydra_balance",
    "functional",
    "condition_shift",
    "postfix_contrastive",
    "postfix_sigma_budget",
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

    def compose(self, ctx: LossContext) -> torch.Tensor:
        per_sample = ctx.model_pred.new_zeros(ctx.model_pred.shape[0])

        # Stage 1: per-sample losses.
        first = True
        for name in _STAGE_PER_SAMPLE:
            if name not in self.active_losses:
                continue
            contribution = LOSS_REGISTRY[name](ctx)
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

    # ---- Split-backward variants ---------------------------------------------------
    # APEX runs two autograd-disjoint DiT forwards (real branch via L_sup+L_mix,
    # fake branch via L_fake). Composing+backwarding them as one scalar keeps
    # both graphs live until the single backward, roughly doubling peak
    # activation memory. ``compose_real_branch`` returns everything except
    # ``apex_fake``; ``compose_fake_branch`` returns ``apex_fake`` alone. Their
    # sum equals ``compose`` numerically for the common case (no multiscale).
    # When multiscale is active the blend is applied to the real-branch scalar
    # only — multiscale operates on forward-1's ``model_pred``/``target`` and
    # has no fake-branch counterpart.

    def compose_real_branch(self, ctx: LossContext) -> torch.Tensor:
        per_sample = ctx.model_pred.new_zeros(ctx.model_pred.shape[0])

        first = True
        for name in _STAGE_PER_SAMPLE:
            if name == "apex_fake" or name not in self.active_losses:
                continue
            contribution = LOSS_REGISTRY[name](ctx)
            per_sample = contribution if first else (per_sample + contribution)
            first = False
        if first:
            raise RuntimeError(
                "LossComposer.compose_real_branch: no per-sample loss "
                "registered; 'flow_match' must be among active_losses"
            )

        for name in _STAGE_SCALAR_BROADCAST:
            if name not in self.active_losses:
                continue
            reg = LOSS_REGISTRY[name](ctx)
            if reg is None:
                continue
            per_sample = per_sample + reg

        scalar = per_sample.mean()

        if "multiscale" in self.active_losses:
            ms_weight = float(getattr(ctx.args, "multiscale_loss_weight", 0.0) or 0.0)
            if ms_weight > 0.0:
                ms_loss = LOSS_REGISTRY["multiscale"](ctx)
                if ms_loss is not None and torch.is_tensor(ms_loss) and ms_loss.numel():
                    if not (ms_loss == 0).all():
                        scalar = (scalar + ms_loss * ms_weight) / (1.0 + ms_weight)

        return scalar

    def compose_fake_branch(self, ctx: LossContext) -> torch.Tensor:
        if "apex_fake" not in self.active_losses:
            return ctx.model_pred.new_zeros(())
        per_sample = LOSS_REGISTRY["apex_fake"](ctx)
        return per_sample.mean()


def build_loss_composer(args: argparse.Namespace, network: object) -> LossComposer:
    """Inspect args + network and return the active LossComposer.

    Rules:
      - flow_match is active for every method except APEX. When APEX is
        active, apex_mix subsumes flow_match (T_mix at inner lambda=0 is
        v_data exactly, so L_mix is pure FM during the warmup window).
      - apex_mix / apex_fake / condition_shift active iff args.method is
        "apex" or "apex_*".
      - ortho_reg active iff network._ortho_reg_weight > 0.
      - hydra_balance active iff network._balance_loss_weight > 0.
      - functional active iff args.functional_loss_weight > 0.
      - multiscale active iff args.multiscale_loss_weight > 0.
    """
    active: list[str] = []

    method = getattr(args, "method", None) or ""
    is_apex = method == "apex" or method.startswith("apex_")
    if is_apex:
        active.extend(["apex_mix", "apex_fake", "condition_shift"])
    else:
        active.append("flow_match")

    if float(getattr(network, "_ortho_reg_weight", 0.0) or 0.0) > 0.0:
        active.append("ortho_reg")
    if float(getattr(network, "_balance_loss_weight", 0.0) or 0.0) > 0.0:
        active.append("hydra_balance")
    if float(getattr(args, "functional_loss_weight", 0.0) or 0.0) > 0.0:
        active.append("functional")
    if float(getattr(args, "multiscale_loss_weight", 0.0) or 0.0) > 0.0:
        active.append("multiscale")
    if float(getattr(network, "contrastive_weight", 0.0) or 0.0) > 0.0:
        active.append("postfix_contrastive")
    if float(getattr(network, "sigma_budget_weight", 0.0) or 0.0) > 0.0:
        active.append("postfix_sigma_budget")

    return LossComposer(active_losses=active)
