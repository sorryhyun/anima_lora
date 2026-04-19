"""Scalar derived metrics (for W&B / TB logging).

M1 extraction (plan.md): pulls derived numbers that are currently scattered
across `AnimaTrainer` (apex step counter, ortho-reg magnitude, hydra
balance loss) into a registry so new adapters can surface a metric without
editing `generate_step_logs`.

The metrics here intentionally duplicate cheap reads — they do not perform
any forward passes. They are called after `compose()` when tensorboard /
wandb trackers are active.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass
class MetricContext:
    args: object
    network: object
    # Trainer-internal state that metrics may read (apex step counter, etc.).
    trainer_state: dict


MetricFn = Callable[[MetricContext], dict[str, float]]


def _apex_step_metric(ctx: MetricContext) -> dict[str, float]:
    if getattr(ctx.args, "method", None) != "apex":
        return {}
    step = int(ctx.trainer_state.get("apex_step", 0))
    return {"apex/step": float(step)}


def _ortho_reg_magnitude(ctx: MetricContext) -> dict[str, float]:
    w = float(getattr(ctx.network, "_ortho_reg_weight", 0.0) or 0.0)
    if w <= 0.0:
        return {}
    val = ctx.network.get_ortho_regularization()
    if torch.is_tensor(val):
        val = val.detach().item()
    return {"reg/ortho": float(val), "reg/ortho_weighted": float(w * val)}


def _balance_loss_metric(ctx: MetricContext) -> dict[str, float]:
    w = float(getattr(ctx.network, "_balance_loss_weight", 0.0) or 0.0)
    if w <= 0.0:
        return {}
    val = ctx.network.get_balance_loss()
    if torch.is_tensor(val):
        val = val.detach().item()
    return {"reg/balance": float(val), "reg/balance_weighted": float(w * val)}


def _postfix_contrastive_metric(ctx: MetricContext) -> dict[str, float]:
    w = float(getattr(ctx.network, "contrastive_weight", 0.0) or 0.0)
    if w <= 0.0:
        return {}
    val = getattr(ctx.network, "_last_contrastive_value", None)
    if val is None:
        return {}
    return {
        "reg/postfix_contrastive": float(val),
        "reg/postfix_contrastive_weighted": float(w * val),
    }


def _postfix_sigma_budget_metric(ctx: MetricContext) -> dict[str, float]:
    w = float(getattr(ctx.network, "sigma_budget_weight", 0.0) or 0.0)
    if w <= 0.0:
        return {}
    val = getattr(ctx.network, "_last_sigma_budget_value", None)
    if val is None:
        return {}
    return {
        "reg/postfix_sigma_budget": float(val),
        "reg/postfix_sigma_budget_weighted": float(w * val),
    }


METRIC_REGISTRY: dict[str, MetricFn] = {
    "apex_step": _apex_step_metric,
    "ortho_reg": _ortho_reg_magnitude,
    "hydra_balance": _balance_loss_metric,
    "postfix_contrastive": _postfix_contrastive_metric,
    "postfix_sigma_budget": _postfix_sigma_budget_metric,
}


def collect_metrics(
    ctx: MetricContext, names: Optional[list[str]] = None
) -> dict[str, float]:
    """Run all (or a subset of) registered metric producers and merge results."""
    out: dict[str, float] = {}
    entries = (
        METRIC_REGISTRY.items()
        if names is None
        else ((n, METRIC_REGISTRY[n]) for n in names if n in METRIC_REGISTRY)
    )
    for _, fn in entries:
        try:
            out.update(fn(ctx))
        except Exception:
            # Metric producers must never kill a training step.
            continue
    return out
