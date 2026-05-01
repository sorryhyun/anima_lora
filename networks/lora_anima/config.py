"""Frozen configuration object for ``LoRANetwork``.

Replaces the 25-arg ``LoRANetwork.__init__`` and the per-kwarg parse pile in
``factory.create_network`` / ``create_network_from_weights``. Two construction
sites — ``from_kwargs`` (fresh training; absorbs the str→bool/int/float casts
that train.py's ``net_kwargs`` produces) and ``from_weights`` (warm-start /
inference; values come from checkpoint key sniffing).

Frozen by intent: every field here is fixed for the run. Mutable runtime
state (``multiplier``, LoRA+ ratios, hit counters, σ caches, post-build attrs
written by ``spec.post_init``) stays as plain attributes on the network.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Type

import torch

from networks.lora_modules import LoRAModule

logger = logging.getLogger(__name__)


def _as_bool(value: Any, *, default: bool = False) -> bool:
    """Parse a kwarg that may arrive as ``"true"`` / ``"false"`` / bool / None."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _as_str_list(value: Any) -> Optional[List[str]]:
    """Parse a kwarg that's either a python-literal list, single string, or None."""
    if value is None:
        return None
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
    except (ValueError, SyntaxError):
        return [value] if isinstance(value, str) else None
    if isinstance(parsed, list):
        return parsed
    return [parsed]


def _as_float_list(value: Any) -> Optional[List[float]]:
    """Parse a kwarg that's either a TOML list, python-literal list string, or None.

    TOML arrays come through as native lists; CLI-stringified lists parse via
    ast.literal_eval. Raises on malformed input rather than silently dropping
    it, since a wrong σ-bucket boundary list would change band assignments
    without surfacing an error.
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(
                f"Could not parse list-of-floats kwarg: {value!r} ({exc})"
            ) from exc
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"Expected list of floats, got {type(value).__name__}: {value!r}"
        )
    return [float(v) for v in value]


def _validate_sigma_bucket_boundaries(
    boundaries: List[float], num_sigma_buckets: int
) -> None:
    """Validate a custom σ-bucket boundary list. Raises ValueError on any
    violation: wrong length, non-zero start, non-one end, or non-strictly-
    increasing edges.
    """
    if len(boundaries) != num_sigma_buckets + 1:
        raise ValueError(
            "sigma_bucket_boundaries must have length num_sigma_buckets + 1 = "
            f"{num_sigma_buckets + 1}, got {len(boundaries)}."
        )
    if abs(boundaries[0]) > 1e-6:
        raise ValueError(
            f"sigma_bucket_boundaries[0] must be 0.0, got {boundaries[0]}."
        )
    if abs(boundaries[-1] - 1.0) > 1e-6:
        raise ValueError(
            f"sigma_bucket_boundaries[-1] must be 1.0, got {boundaries[-1]}."
        )
    for i in range(len(boundaries) - 1):
        if boundaries[i + 1] <= boundaries[i]:
            raise ValueError(
                "sigma_bucket_boundaries must be strictly increasing; "
                f"violated at index {i}: {boundaries[i]} >= {boundaries[i + 1]}."
            )


def _parse_kv_pairs(kv_pair_str: str, *, is_int: bool) -> Dict[str, Any]:
    """Parse "key1=val1,key2=val2" into a dict, casting values to int/float."""
    pairs: Dict[str, Any] = {}
    for pair in kv_pair_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            logger.warning(f"Invalid format: {pair}, expected 'key=value'")
            continue
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            pairs[key] = int(value) if is_int else float(value)
        except ValueError:
            logger.warning(f"Invalid value for {key}: {value}")
    return pairs


# Default exclude regex appended to user-supplied excludes in `from_kwargs`.
# Skips embedders / norms / modulation projectors that are never adapted.
_DEFAULT_EXCLUDE = (
    r".*(_modulation|_norm|_embedder|final_layer|adaln_fused_down|adaln_up_|"
    r"pooled_text_proj).*"
)

_DEFAULT_SIGMA_ROUTER_LAYERS = r".*(cross_attn\.q_proj|self_attn\.qkv_proj)$"


@dataclass(frozen=True)
class LoRANetworkCfg:
    """Run-fixed configuration for a ``LoRANetwork``.

    Field groupings mirror the comment blocks in ``factory.create_network``:
    core / targeting / dropouts / regex overrides / T-LoRA / ReFT / Hydra /
    σ-router / channel scaling / logging.
    """

    # core LoRA
    lora_dim: int = 4
    alpha: float = 1.0
    module_class: Type = LoRAModule
    # warm-start path supplies these from the checkpoint; fresh path leaves None
    modules_dim: Optional[Dict[str, int]] = None
    modules_alpha: Optional[Dict[str, float]] = None

    # targeting
    train_llm_adapter: bool = False
    exclude_patterns: List[str] = field(default_factory=list)
    include_patterns: Optional[List[str]] = None
    layer_start: Optional[int] = None
    layer_end: Optional[int] = None

    # dropouts
    dropout: Optional[float] = None
    rank_dropout: Optional[float] = None
    module_dropout: Optional[float] = None

    # per-module rank / lr regex overrides
    reg_dims: Optional[Dict[str, int]] = None
    reg_lrs: Optional[Dict[str, float]] = None

    # T-LoRA
    use_timestep_mask: bool = False
    min_rank: int = 1
    alpha_rank_scale: float = 1.0

    # ReFT
    add_reft: bool = False
    reft_dim: int = 4
    reft_alpha: Optional[float] = None
    reft_layers: object = "all"

    # Hydra (MoE) / expert warmup
    num_experts: int = 4
    # Gaussian perturb std applied to fused per-expert `lora_up_weight` at
    # init in plain HydraLoRA only (NOT OrthoHydra disjoint or fallback) —
    # paper baseline knob; production training should leave at 0.0 and use
    # `expert_warmup_ratio` instead.
    expert_init_std: float = 0.0
    expert_warmup_ratio: float = 0.0
    expert_warmup_k: int = 1
    expert_best_warmup_ratio: float = 0.0
    router_lr_scale: float = 1.0
    hydra_router_layers: Optional[str] = None
    hydra_router_names: Optional[List[str]] = None
    per_bucket_balance_weight: float = 0.3
    num_sigma_buckets: int = 3
    # Hard expert/timestep partition: when on, the E experts are split into
    # ``num_sigma_buckets`` bands of ``E // num_sigma_buckets`` experts each
    # using interleaved layout (expert e → band ``e mod num_sigma_buckets``);
    # for a sample at σ in band b, only the experts in that band can win the
    # gate (out-of-band logits masked to -inf before softmax). Soft routing
    # still operates *within* a band. Independent of (and composes with) the
    # σ-feature router. Requires ``num_experts % num_sigma_buckets == 0``.
    specialize_experts_by_sigma_buckets: bool = False
    # Optional custom σ-bucket boundaries. Length must equal
    # ``num_sigma_buckets + 1``, strictly increasing, starting at 0.0 and
    # ending at 1.0. Defaults (None) to uniform ``linspace(0, 1, B+1)``.
    # Lets you spend more capacity on a chosen σ regime — e.g.
    # ``[0.0, 0.5, 0.8, 1.0]`` gives a wide low-σ band and progressively
    # narrower mid/high-σ bands while expert count per band stays equal.
    sigma_bucket_boundaries: Optional[List[float]] = None

    # σ-conditional router (hydra add-on)
    use_sigma_router: bool = False
    sigma_feature_dim: int = 16
    sigma_hidden_dim: int = 128
    sigma_router_layers: Optional[str] = None
    sigma_router_names: Optional[List[str]] = None

    # SmoothQuant-style per-channel input pre-scaling
    channel_scales_dict: Optional[Dict[str, torch.Tensor]] = None

    # logging
    verbose: bool = False

    @classmethod
    def from_kwargs(
        cls,
        kwargs: Mapping[str, Any],
        *,
        network_dim: Optional[int],
        network_alpha: Optional[float],
        neuron_dropout: Optional[float],
        module_class: Type,
        channel_scales_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> "LoRANetworkCfg":
        """Build cfg from train.py's stringified ``net_kwargs`` dict."""
        if network_dim is None:
            network_dim = 4
        if network_alpha is None:
            network_alpha = 1.0

        train_llm_adapter = _as_bool(kwargs.get("train_llm_adapter"))

        exclude_patterns = _as_str_list(kwargs.get("exclude_patterns")) or []
        exclude_patterns.append(_DEFAULT_EXCLUDE)
        include_patterns = _as_str_list(kwargs.get("include_patterns"))

        layer_start = kwargs.get("layer_start")
        layer_start = int(layer_start) if layer_start is not None else None
        layer_end = kwargs.get("layer_end")
        layer_end = int(layer_end) if layer_end is not None else None

        rank_dropout = kwargs.get("rank_dropout")
        rank_dropout = float(rank_dropout) if rank_dropout is not None else None
        module_dropout = kwargs.get("module_dropout")
        module_dropout = float(module_dropout) if module_dropout is not None else None

        use_timestep_mask = _as_bool(kwargs.get("use_timestep_mask"))
        min_rank = kwargs.get("min_rank")
        min_rank = int(min_rank) if min_rank is not None else 1
        alpha_rank_scale = kwargs.get("alpha_rank_scale")
        alpha_rank_scale = (
            float(alpha_rank_scale) if alpha_rank_scale is not None else 1.0
        )

        add_reft = _as_bool(kwargs.get("add_reft"))
        reft_dim = kwargs.get("reft_dim")
        reft_dim = int(reft_dim) if reft_dim is not None else network_dim
        reft_alpha = kwargs.get("reft_alpha")
        reft_alpha = float(reft_alpha) if reft_alpha is not None else None
        reft_layers = kwargs.get("reft_layers", "all")

        num_experts = kwargs.get("num_experts")
        num_experts = int(num_experts) if num_experts is not None else 4
        expert_init_std = float(kwargs.get("expert_init_std", 0.0))
        expert_warmup_ratio = float(kwargs.get("expert_warmup_ratio", 0.0))
        expert_warmup_k = int(kwargs.get("expert_warmup_k", 1))
        expert_best_warmup_ratio = float(kwargs.get("expert_best_warmup_ratio", 0.0))
        if expert_warmup_ratio > 0.0 and expert_best_warmup_ratio > 0.0:
            logger.warning(
                "Both expert_warmup_ratio (%.3f) and expert_best_warmup_ratio "
                "(%.3f) are non-zero. The random path's pre-forward mask zeros "
                "non-selected experts' grads, which makes the post-backward "
                "top-k selection redundant. Set exactly one to >0.",
                expert_warmup_ratio,
                expert_best_warmup_ratio,
            )

        router_lr_scale = kwargs.get("network_router_lr_scale")
        router_lr_scale = float(router_lr_scale) if router_lr_scale is not None else 1.0

        hydra_router_layers = kwargs.get("hydra_router_layers", None)
        per_bucket_balance_weight = kwargs.get("per_bucket_balance_weight")
        per_bucket_balance_weight = (
            float(per_bucket_balance_weight)
            if per_bucket_balance_weight is not None
            else 0.3
        )
        num_sigma_buckets = int(kwargs.get("num_sigma_buckets", 3))
        specialize_experts_by_sigma_buckets = _as_bool(
            kwargs.get("specialize_experts_by_sigma_buckets")
        )
        sigma_bucket_boundaries = _as_float_list(
            kwargs.get("sigma_bucket_boundaries")
        )
        if specialize_experts_by_sigma_buckets:
            if num_sigma_buckets <= 1:
                raise ValueError(
                    "specialize_experts_by_sigma_buckets requires num_sigma_buckets > 1, "
                    f"got num_sigma_buckets={num_sigma_buckets}."
                )
            if num_experts % num_sigma_buckets != 0:
                raise ValueError(
                    "specialize_experts_by_sigma_buckets requires num_experts to be "
                    f"divisible by num_sigma_buckets, got num_experts={num_experts}, "
                    f"num_sigma_buckets={num_sigma_buckets}."
                )
            if sigma_bucket_boundaries is not None:
                _validate_sigma_bucket_boundaries(
                    sigma_bucket_boundaries, num_sigma_buckets
                )
        elif sigma_bucket_boundaries is not None:
            logger.warning(
                "sigma_bucket_boundaries set but "
                "specialize_experts_by_sigma_buckets is off — boundaries ignored."
            )
            sigma_bucket_boundaries = None

        use_sigma_router = _as_bool(kwargs.get("use_sigma_router"))
        sigma_feature_dim = int(kwargs.get("sigma_feature_dim", 16))
        sigma_hidden_dim = int(kwargs.get("sigma_hidden_dim", 128))
        sigma_router_layers = kwargs.get(
            "sigma_router_layers", _DEFAULT_SIGMA_ROUTER_LAYERS
        )

        reg_dims_str = kwargs.get("network_reg_dims")
        reg_dims = _parse_kv_pairs(reg_dims_str, is_int=True) if reg_dims_str else None
        reg_lrs_str = kwargs.get("network_reg_lrs")
        reg_lrs = _parse_kv_pairs(reg_lrs_str, is_int=False) if reg_lrs_str else None

        verbose = _as_bool(kwargs.get("verbose"))

        return cls(
            lora_dim=network_dim,
            alpha=network_alpha,
            module_class=module_class,
            train_llm_adapter=train_llm_adapter,
            exclude_patterns=exclude_patterns,
            include_patterns=include_patterns,
            layer_start=layer_start,
            layer_end=layer_end,
            dropout=neuron_dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            reg_dims=reg_dims,
            reg_lrs=reg_lrs,
            use_timestep_mask=use_timestep_mask,
            min_rank=min_rank,
            alpha_rank_scale=alpha_rank_scale,
            add_reft=add_reft,
            reft_dim=reft_dim,
            reft_alpha=reft_alpha,
            reft_layers=reft_layers,
            num_experts=num_experts,
            expert_init_std=expert_init_std,
            expert_warmup_ratio=expert_warmup_ratio,
            expert_warmup_k=expert_warmup_k,
            expert_best_warmup_ratio=expert_best_warmup_ratio,
            router_lr_scale=router_lr_scale,
            hydra_router_layers=hydra_router_layers,
            per_bucket_balance_weight=per_bucket_balance_weight,
            num_sigma_buckets=num_sigma_buckets,
            specialize_experts_by_sigma_buckets=specialize_experts_by_sigma_buckets,
            sigma_bucket_boundaries=sigma_bucket_boundaries,
            use_sigma_router=use_sigma_router,
            sigma_feature_dim=sigma_feature_dim,
            sigma_hidden_dim=sigma_hidden_dim,
            sigma_router_layers=sigma_router_layers,
            channel_scales_dict=channel_scales_dict,
            verbose=verbose,
        )

    @classmethod
    def from_weights(
        cls,
        *,
        modules_dim: Dict[str, int],
        modules_alpha: Dict[str, float],
        module_class: Type,
        train_llm_adapter: bool,
        has_reft: bool,
        reft_dim: Optional[int],
        reft_block_indices,
        is_hydra_or_ortho_hydra: bool,
        hydra_num_experts: int,
        sigma_feature_dim_detected: Optional[int],
        sigma_router_names: Optional[List[str]],
        hydra_router_names: Optional[List[str]],
        channel_scales_dict: Optional[Dict[str, torch.Tensor]],
        specialize_experts_by_sigma_buckets: bool = False,
        num_sigma_buckets: Optional[int] = None,
        sigma_bucket_boundaries: Optional[List[float]] = None,
    ) -> "LoRANetworkCfg":
        """Build cfg from a checkpoint key-sniff (warm-start / inference path).

        Mirrors the ``LoRANetwork(...)`` call previously embedded in
        ``create_network_from_weights``. Per-module dims / alphas come from
        ``modules_dim`` / ``modules_alpha``, so ``lora_dim`` / ``alpha`` here
        are placeholders. Training-time schedules (warmup, T-LoRA) stay off
        in the warm-start path.

        ``specialize_experts_by_sigma_buckets`` / ``num_sigma_buckets`` /
        ``sigma_bucket_boundaries`` come from safetensors metadata stamped by
        ``save_weights`` — the partition leaves no tensor footprint
        (``_expert_band`` / ``_sigma_edges`` are non-persistent) so it has to
        be reconstructed from those scalars at load time.
        """
        return cls(
            lora_dim=4,
            alpha=1.0,
            module_class=module_class,
            modules_dim=modules_dim,
            modules_alpha=modules_alpha,
            train_llm_adapter=train_llm_adapter,
            add_reft=has_reft,
            reft_dim=reft_dim if reft_dim is not None else 4,
            reft_layers=sorted(reft_block_indices) if has_reft else "all",
            num_experts=hydra_num_experts if is_hydra_or_ortho_hydra else 4,
            channel_scales_dict=channel_scales_dict,
            use_sigma_router=bool(sigma_router_names),
            sigma_feature_dim=sigma_feature_dim_detected or 128,
            sigma_hidden_dim=128,
            sigma_router_names=sigma_router_names,
            hydra_router_names=hydra_router_names,
            specialize_experts_by_sigma_buckets=specialize_experts_by_sigma_buckets,
            num_sigma_buckets=(
                int(num_sigma_buckets) if num_sigma_buckets else 3
            ),
            sigma_bucket_boundaries=sigma_bucket_boundaries,
        )
