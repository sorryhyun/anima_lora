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
    method = getattr(ctx.args, "method", None) or ""
    if not (method == "apex" or method.startswith("apex_")):
        return {}
    out: dict[str, float] = {
        "apex/lam_inner_eff": float(ctx.trainer_state.get("apex_lam_inner_eff", 0.0)),
        "apex/lam_f_eff": float(ctx.trainer_state.get("apex_lam_f_eff", 0.0)),
    }
    # Per-component unweighted MSE means, stashed by _apex_mix_loss /
    # _apex_fake_loss. Comparable to plain FM loss on the y-axis. L_fake is
    # only emitted while lam_f_eff > 0 (during warmup it never executes).
    mix_v = getattr(ctx.network, "_last_apex_mix_value", None)
    if mix_v is not None:
        out["apex/loss_mix"] = float(mix_v)
    fake_v = getattr(ctx.network, "_last_apex_fake_value", None)
    if fake_v is not None:
        out["apex/loss_fake"] = float(fake_v)
    # MSE(v_fake_sg, F_real) — degeneracy detector. ~0 means the shifted
    # condition produces the same output as the unshifted one, so L_mix
    # collapses to its trivial self-consistency fixed point.
    div = ctx.trainer_state.get("apex_v_fake_divergence")
    if div is not None:
        out["apex/v_fake_divergence"] = float(div)
    # ConditionShift parameters — for scalar mode these are 1-element tensors;
    # for diag/full emit norms instead so the dashboard stays useful.
    cs = getattr(ctx.network, "apex_condition_shift", None)
    if cs is not None:
        mode = getattr(cs, "mode", None)
        if mode == "scalar":
            out["apex/cs_a"] = float(cs.a.detach().item())
            out["apex/cs_b"] = float(cs.b.detach().item())
        elif mode == "diag":
            out["apex/cs_a_norm"] = float(cs.a.detach().norm().item())
            out["apex/cs_b_norm"] = float(cs.b.detach().norm().item())
        elif mode == "full":
            out["apex/cs_A_norm"] = float(cs.A.detach().norm().item())
            out["apex/cs_b_norm"] = float(cs.b.detach().norm().item())
    return out


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
    near-random gating), per-expert usage rates (sum to ~1.0; a near-0 entry
    means that expert is never picked), and per-σ-bucket usage rates broken
    out as ``hydra/expert_usage_b{bucket}/{expert}`` (bucket 0 = lowest σ).
    Per-bucket sample counts are emitted as ``hydra/bucket_count/{bucket}``
    so empty-bucket rows can be filtered downstream.
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
        for b, row in enumerate(stats.get("expert_usage_per_bucket", [])):
            for i, v in enumerate(row):
                out[f"hydra/expert_usage_b{b}/{i}"] = float(v)
        for b, c in enumerate(stats.get("bucket_counts", [])):
            out[f"hydra/bucket_count/{b}"] = float(c)
        return out

    # Backward compat: old networks only exposed the mean scalar.
    legacy = getattr(ctx.network, "get_router_entropy", None)
    if legacy is None:
        return {}
    H = legacy()
    return {"hydra/router_entropy": float(H)} if H is not None else {}


def _hydra_up_grad_metric(ctx: MetricContext) -> dict[str, float]:
    """Hydra up-weight grad norms split by rank region and σ-band.

    Diagnoses the T-LoRA × σ-bucket conflict (high-σ-band experts only fire
    at high σ where T-LoRA caps rank to ``min_rank``, so columns
    ``[min_rank, R)`` of those experts' ``lora_up`` should starve). Compare
    ``hydra/up_grad/above_below_ratio/band{b}`` across bands: a healthy
    setup has comparable ratios across bands; if the high-σ band is much
    smaller than the low-σ band, the conflict is biting and capacity in
    those experts is wasted.

    Per-expert keys (``hydra/up_grad/{below,above,total}/exp{e}``) are
    L2 norms (sqrt of summed-squares across modules). Per-band keys
    aggregate experts within each σ-band. ``sp_total`` is the OrthoHydra
    fallback (per-expert ``S_p`` grad-norm, no rank-region split because
    Cayley couples all entries).

    Empty when nothing was captured this step (e.g., ``use_hydra=false``,
    or the capture call wasn't reached).
    """
    if not getattr(ctx.network, "_use_hydra", False):
        return {}
    getter = getattr(ctx.network, "get_up_grad_stats", None)
    if getter is None:
        return {}
    stats = getter()
    if not stats:
        return {}

    out: dict[str, float] = {}
    eps = 1e-12

    def _emit_per_expert(prefix: str, sq: list[float]) -> None:
        for i, v in enumerate(sq):
            out[f"hydra/up_grad/{prefix}/exp{i}"] = float(v) ** 0.5

    def _emit_per_band(prefix: str, sq: list[float]) -> None:
        for b, v in enumerate(sq):
            out[f"hydra/up_grad/{prefix}/band{b}"] = float(v) ** 0.5

    if "total" in stats:
        _emit_per_expert("total", stats["total"])
    if "below" in stats and "above" in stats:
        _emit_per_expert("below", stats["below"])
        _emit_per_expert("above", stats["above"])
        for i, (b, a) in enumerate(zip(stats["below"], stats["above"])):
            out[f"hydra/up_grad/above_below_ratio/exp{i}"] = (
                float(a) ** 0.5 / (float(b) ** 0.5 + eps)
            )
    if "sp_total" in stats:
        _emit_per_expert("sp_total", stats["sp_total"])

    if "total_band" in stats:
        _emit_per_band("total", stats["total_band"])
    if "below_band" in stats and "above_band" in stats:
        _emit_per_band("below", stats["below_band"])
        _emit_per_band("above", stats["above_band"])
        for b, (bv, av) in enumerate(zip(stats["below_band"], stats["above_band"])):
            out[f"hydra/up_grad/above_below_ratio/band{b}"] = (
                float(av) ** 0.5 / (float(bv) ** 0.5 + eps)
            )
    if "sp_total_band" in stats:
        _emit_per_band("sp_total", stats["sp_total_band"])

    return out


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
    "hydra_up_grad": _hydra_up_grad_metric,
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
    with torch.no_grad():
        for _, fn in entries:
            try:
                out.update(fn(ctx))
            except Exception:
                # Metric producers must never kill a training step.
                continue
    return out
