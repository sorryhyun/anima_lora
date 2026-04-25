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


def _router_entropy_metric(ctx: MetricContext) -> dict[str, float]:
    """Hydra router diagnostics.

    Emits the single mean entropy plus quantiles (p05/p50/p95 across modules —
    dead-routing hotspots show up in p05), the mean top1-top2 margin (low =
    near-random gating), and per-expert usage rates (sum to ~1.0; a near-0
    entry means that expert is never picked).
    """
    if not getattr(ctx.network, "_use_hydra", False):
        return {}
    getter = getattr(ctx.network, "get_router_stats", None)
    if getter is not None:
        stats = getter()
        if not stats:
            return {}
        out: dict[str, float] = {
            "hydra/router_entropy": float(stats["entropy_mean"]),
            "hydra/router_entropy_p05": float(stats["entropy_p05"]),
            "hydra/router_entropy_p50": float(stats["entropy_p50"]),
            "hydra/router_entropy_p95": float(stats["entropy_p95"]),
            "hydra/router_margin": float(stats["margin_mean"]),
        }
        for i, v in enumerate(stats.get("expert_usage", [])):
            out[f"hydra/expert_usage/{i}"] = float(v)
        return out

    # Backward compat: old networks only exposed the mean scalar.
    legacy = getattr(ctx.network, "get_router_entropy", None)
    if legacy is None:
        return {}
    H = legacy()
    return {"hydra/router_entropy": float(H)} if H is not None else {}


def _expert_warmup_pick_metric(ctx: MetricContext) -> dict[str, float]:
    """Per-expert pick fraction during HydraLoRA expert-warmup.

    For random ``expert_warmup_ratio`` the value is the fraction of modules
    that picked expert ``i`` this step (≈ k/E under uniform random sampling).
    For greedy ``expert_best_warmup_ratio`` it shows the actual gradient-
    aligned distribution — flat ≈ healthy diversification, peaky = one expert
    is winning every module's top-k. Drops out of the dashboard once the
    warmup window ends (network returns None).
    """
    if not getattr(ctx.network, "_use_hydra", False):
        return {}
    getter = getattr(ctx.network, "get_expert_warmup_pick_stats", None)
    if getter is None:
        return {}
    picks = getter()
    if picks is None:
        return {}
    return {f"hydra/expert_warmup_pick/{i}": float(v) for i, v in enumerate(picks)}


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
    "hydra_router_entropy": _router_entropy_metric,
    "hydra_expert_warmup_pick": _expert_warmup_pick_metric,
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
