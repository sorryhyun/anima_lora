"""Frozen configuration object for ``LoRANetwork``.

Two construction sites: ``from_kwargs`` (fresh training, from train.py's
stringified ``net_kwargs``) and ``from_weights`` (warm-start / inference,
from checkpoint key sniffing). Mutable runtime state (multiplier, LoRA+
ratios, hit counters, σ caches) stays as plain attributes on the network,
not here.
"""

from __future__ import annotations

import ast
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Type, Union

import torch

from networks.lora_modules import LoRAModule

# Three-axis routing config (see the `lora-routing` skill).
MoEStyle = Union[Literal[False], Literal["shared_A"]]
RouterSource = Literal["input", "sigma", "fei", "crossattn_emb", "none"]

logger = logging.getLogger(__name__)


def _as_bool(value: Any, *, default: bool = False) -> bool:
    """Parse a kwarg that may arrive as ``"true"`` / ``"false"`` / bool / None."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _as_moe_style(value: Any) -> MoEStyle:
    """Parse the ``use_moe_style`` kwarg.

    Accepts: ``False`` / ``None`` / ``"false"`` / ``""`` → ``False``;
    the literal string ``"shared_A"`` passes through.
    """
    if value is None or value is False:
        return False
    if isinstance(value, str):
        v = value.strip()
        if v.lower() in ("false", "none", ""):
            return False
        if v == "shared_A":
            return v
    raise ValueError(f"use_moe_style={value!r}: expected False or 'shared_A'.")


def _as_router_source(value: Any) -> RouterSource:
    """Parse the ``router_source`` kwarg. Empty / None → ``"none"``.

    ``"crossattn_emb"`` requires ``route_per_layer=False`` — no per-Linear
    crossattn signal exists.
    """
    if value is None:
        return "none"
    if isinstance(value, str):
        v = value.strip()
        if v == "":
            return "none"
        if v in ("input", "sigma", "fei", "crossattn_emb", "none"):
            return v  # type: ignore[return-value]
    raise ValueError(
        f"router_source={value!r}: expected 'input', 'sigma', 'fei', "
        "'crossattn_emb', or 'none'."
    )


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
    """Parse a kwarg that's a TOML list, python-literal list string, or None.

    Raises on malformed input rather than silently dropping it — a bad
    σ-bucket boundary list would otherwise change band assignments silently.
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
    """Raise ValueError on wrong length, non-0 start, non-1 end, or
    non-strictly-increasing edges."""
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


# Default exclude regex appended to user excludes in `from_kwargs` — skips
# embedders / norms / modulation projectors. `adaln_up_` is excluded here but
# rescued back via `train_adaln`'s include_patterns (on by default).
_DEFAULT_EXCLUDE = (
    r".*(_modulation|_norm|_embedder|final_layer|adaln_fused_down|adaln_up_|"
    r"pooled_text_proj).*"
)


@dataclass(frozen=True)
class LoRANetworkCfg:
    """Run-fixed configuration for a ``LoRANetwork``."""

    lora_dim: int = 4
    alpha: float = 1.0
    module_class: Type = LoRAModule
    # warm-start path supplies these from the checkpoint; fresh path leaves None
    modules_dim: Optional[Dict[str, int]] = None
    modules_alpha: Optional[Dict[str, float]] = None

    train_llm_adapter: bool = False
    exclude_patterns: List[str] = field(default_factory=list)
    include_patterns: Optional[List[str]] = None
    layer_start: Optional[int] = None
    layer_end: Optional[int] = None

    dropout: Optional[float] = None
    rank_dropout: Optional[float] = None
    module_dropout: Optional[float] = None

    reg_dims: Optional[Dict[str, int]] = None
    reg_lrs: Optional[Dict[str, float]] = None
    # Per-pattern alpha override (regex fullmatch on module path); independent
    # of reg_dims — a matched module keeps its resolved dim, takes this alpha.
    reg_alphas: Optional[Dict[str, float]] = None

    use_timestep_mask: bool = False
    min_rank: int = 1
    alpha_rank_scale: float = 1.0

    num_experts: int = 4
    # Gaussian perturb std for fused per-expert `lora_up_weight` init (plain
    # HydraLoRA only). Production leaves at 0.0.
    expert_init_std: float = 0.0
    router_lr_scale: float = 1.0
    # Single regex scoping which Linears are routed (Hydra leaves + σ/FEI
    # share it). None = apply MoE everywhere.
    router_targets: Optional[str] = None
    hydra_router_names: Optional[List[str]] = None
    per_bucket_balance_weight: float = 0.3
    num_sigma_buckets: int = 3
    # Hard expert/timestep partition: split E into num_sigma_buckets interleaved
    # bands (expert e → band ``e mod num_sigma_buckets``); out-of-band logits are
    # masked -inf before softmax. Requires num_experts % num_sigma_buckets == 0.
    specialize_experts_by_sigma_buckets: bool = False
    # Custom σ-bucket boundaries: length num_sigma_buckets+1, strictly
    # increasing, 0.0→1.0; None = uniform linspace.
    sigma_bucket_boundaries: Optional[List[float]] = None

    # Three-axis routing config — the `lora-routing` skill has the full
    # matrix. use_moe_style: expert layout. route_per_layer:
    # router location. router_source: gate input signal.
    use_moe_style: MoEStyle = False
    route_per_layer: bool = False
    router_source: RouterSource = "none"

    # SVD-Down: ``lora_down`` init for plain LoRA — "kaiming", "weight_svd"
    # (seed from W0's top-r right singular vectors), or the gradient-seeded
    # "grad_svd" / "basis_file" (top-r row space of the task gradient; the basis
    # arrives in ``grad_basis_dict``). Plain LoRAModule only.
    # See docs/methods/svd-down-lora.md, docs/proposal/grad_basis_init.md.
    down_init: str = "kaiming"

    # weight_svd window: slice k seeds ``lora_down`` from W0's right singular
    # vectors [k·r, (k+1)·r) instead of the top-r. Slices of one orthonormal
    # basis are mutually orthogonal, so adapters trained with different slices
    # never share an input subspace at merge (a per-artist address). 0 = top-r.
    svd_slice: int = 0

    # Gradient-SVD basis, {lora_name: V (in, r_store)} — required by
    # down_init="grad_svd"/"basis_file", built by networks/grad_basis.py.
    grad_basis_dict: Optional[Dict[str, torch.Tensor]] = None

    # σ-conditional router parameters (router_source="sigma"). Layer scope
    # shared with Hydra/FEI via router_targets above.
    sigma_feature_dim: int = 16
    sigma_router_names: Optional[List[str]] = None

    # FEI-conditional router parameters (router_source="fei"). fei_feature_dim
    # defaults to 2 = the (e_low, e_high) simplex from
    # library.runtime.fei.compute_fei_2band; fei_sigma_low_div=4.0 chosen by
    # dataset sweep for σ_low scaling.
    fei_feature_dim: int = 2
    fei_sigma_low_div: float = 4.0
    fei_router_names: Optional[List[str]] = None

    # GlobalRouter parameters (route_per_layer=False). Two-layer MLP feeding
    # softmax/τ. Final layer is zero-init so step-0 gates are uniform,
    # guaranteeing ΔW=0 at the first optimizer step.
    router_hidden_dim: int = 64
    router_tau: float = 0.7

    # Step-expert (turbo per-step head split). >1 → StepExpertLoRAModule:
    # shared lora_down + K up-heads selected by diffusion step. 0/1 = inactive.
    step_expert_K: int = 0

    # SmoothQuant-style per-channel input pre-scaling
    channel_scales_dict: Optional[Dict[str, torch.Tensor]] = None

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
        grad_basis_dict: Optional[Dict[str, torch.Tensor]] = None,
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

        # adaln convenience knobs: train_adaln adds adaln_up_{branch} Linears to
        # the target set (they sit in _DEFAULT_EXCLUDE, rescued via
        # include_patterns). adaln_rank/adaln_alpha give them their own
        # rank/alpha (0/absent = derived below); injected into reg_dims/
        # reg_alphas after those strings are parsed.
        train_adaln = _as_bool(kwargs.get("train_adaln"))
        adaln_rank_raw = kwargs.get("adaln_rank")
        adaln_rank = int(adaln_rank_raw) if adaln_rank_raw is not None else 0
        adaln_alpha_raw = kwargs.get("adaln_alpha")
        adaln_alpha = float(adaln_alpha_raw) if adaln_alpha_raw is not None else 0.0
        if adaln_rank > 0 and not train_adaln:
            raise ValueError("adaln_rank > 0 requires train_adaln = true")
        if adaln_alpha > 0 and not train_adaln:
            raise ValueError("adaln_alpha > 0 requires train_adaln = true")

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

        num_experts = kwargs.get("num_experts")
        num_experts = int(num_experts) if num_experts is not None else 4
        expert_init_std = float(kwargs.get("expert_init_std", 0.0))

        router_lr_scale = kwargs.get("network_router_lr_scale")
        router_lr_scale = float(router_lr_scale) if router_lr_scale is not None else 1.0

        down_init = str(kwargs.get("down_init", "kaiming"))
        if down_init not in ("kaiming", "weight_svd", "grad_svd", "basis_file"):
            raise ValueError(
                f"down_init={down_init!r}: expected 'kaiming', 'weight_svd', "
                f"'grad_svd' or 'basis_file'."
            )
        if down_init in ("grad_svd", "basis_file") and not grad_basis_dict:
            raise ValueError(
                f"down_init={down_init!r} needs a gradient basis. "
                "basis_file: pass network_args grad_basis_file=<path> (build one "
                "with bench/grad_init/build_universal_basis.py). grad_svd: run "
                "through train.py, which sketches the run's own cached dataset "
                "before the network is built."
            )
        svd_slice = int(kwargs.get("svd_slice", 0) or 0)
        if svd_slice < 0:
            raise ValueError(f"svd_slice={svd_slice}: must be a non-negative integer.")
        if svd_slice and down_init != "weight_svd":
            raise ValueError(
                f"svd_slice={svd_slice} only applies to down_init='weight_svd' "
                f"(got {down_init!r})."
            )

        _legacy_router_keys = [
            k
            for k in ("hydra_router_layers", "sigma_router_layers", "fei_router_layers")
            if k in kwargs
        ]
        if _legacy_router_keys:
            raise ValueError(
                f"{_legacy_router_keys} are no longer supported — the three "
                "router layer filters were consolidated into a single "
                "`router_targets` regex. Replace them with one `router_targets = "
                "...` entry in your method TOML."
            )
        router_targets = kwargs.get("router_targets", None)
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
        sigma_bucket_boundaries = _as_float_list(kwargs.get("sigma_bucket_boundaries"))
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

        sigma_feature_dim = int(kwargs.get("sigma_feature_dim", 16))

        fei_feature_dim = int(kwargs.get("fei_feature_dim", 2))
        fei_sigma_low_div = float(kwargs.get("fei_sigma_low_div", 4.0))

        # GlobalRouter knobs (only consumed when ``route_per_layer=False``).
        router_hidden_dim = int(
            kwargs.get("router_hidden_dim", kwargs.get("router_hidden", 64))
        )
        router_tau = float(kwargs.get("router_tau", 0.7))

        # Three-axis routing resolution. The retired ``use_hydra`` /
        # ``use_sigma_router`` / ``use_fei_router`` kwargs raise below.
        raw_moe_style = kwargs.get("use_moe_style")
        raw_route_per_layer = kwargs.get("route_per_layer")
        raw_router_source = kwargs.get("router_source")

        for legacy_key in ("use_hydra", "use_sigma_router", "use_fei_router"):
            if kwargs.get(legacy_key) is not None:
                raise ValueError(
                    f"Legacy router kwarg {legacy_key!r} is no longer "
                    "supported. Use the three-axis keys instead: "
                    "`use_moe_style` (False / 'shared_A'), "
                    "`route_per_layer` (true / false), and `router_source` "
                    "('none' / 'input' / 'sigma' / 'fei' / 'crossattn_emb'). "
                    "See plan2.md §three-axis-config."
                )

        use_moe_style: MoEStyle = (
            _as_moe_style(raw_moe_style) if raw_moe_style is not None else False
        )

        if raw_router_source is not None:
            router_source: RouterSource = _as_router_source(raw_router_source)
        elif use_moe_style is not False:
            # Hydra's default router input is the per-Linear input vector.
            router_source = "input"
        else:
            router_source = "none"

        if raw_route_per_layer is not None:
            route_per_layer = _as_bool(raw_route_per_layer)
        else:
            # no-MoE = no router; Hydra defaults to per-layer
            route_per_layer = use_moe_style is not False

        # SVD-Down / gradient-SVD target plain LoRAModule only — a
        # non-plain variant would silently ignore it, so fail loudly instead.
        if down_init != "kaiming" and use_moe_style is not False:
            raise ValueError(
                f"down_init={down_init!r} only applies to plain LoRA, but "
                f"use_moe_style={use_moe_style!r} is selected. Disable it to use "
                "SVD-Down, or keep down_init='kaiming'."
            )

        # Validate impossible combos.
        if use_moe_style is False and (route_per_layer or router_source != "none"):
            raise ValueError(
                "Routing config requires use_moe_style != False; got "
                f"use_moe_style={use_moe_style!r}, route_per_layer={route_per_layer}, "
                f"router_source={router_source!r}."
            )
        if not route_per_layer and router_source == "input":
            raise ValueError(
                "router_source='input' requires route_per_layer=True — no "
                "network-level 'input' signal exists per DiT forward."
            )
        if route_per_layer and router_source == "crossattn_emb":
            raise ValueError(
                "router_source='crossattn_emb' requires route_per_layer=False — "
                "the pooled cross-attention text feature is a single per-sample "
                "vector routed by one network-level GlobalRouter, with no "
                "per-Linear variant."
            )

        step_expert_K_raw = kwargs.get("step_expert_K")
        step_expert_K = int(step_expert_K_raw) if step_expert_K_raw is not None else 0

        reg_dims_str = kwargs.get("network_reg_dims")
        reg_dims = _parse_kv_pairs(reg_dims_str, is_int=True) if reg_dims_str else None
        reg_lrs_str = kwargs.get("network_reg_lrs")
        reg_lrs = _parse_kv_pairs(reg_lrs_str, is_int=False) if reg_lrs_str else None
        reg_alphas_str = kwargs.get("network_reg_alphas")
        reg_alphas = (
            _parse_kv_pairs(reg_alphas_str, is_int=False) if reg_alphas_str else None
        )

        if train_adaln:
            _adaln_pat = ".*adaln_up_.*"
            include_patterns = (include_patterns or []) + [_adaln_pat]
            if adaln_rank > 0:
                reg_dims = {**(reg_dims or {}), _adaln_pat: adaln_rank}
            if adaln_alpha <= 0:
                # √r law (alpha ∝ √r — docs/methods/adaln.md) instead of
                # inheriting network_alpha at a smaller rank.
                _r = adaln_rank if adaln_rank > 0 else network_dim
                adaln_alpha = network_alpha * math.sqrt(_r / max(network_dim, 1))
            reg_alphas = {**(reg_alphas or {}), _adaln_pat: adaln_alpha}

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
            reg_alphas=reg_alphas,
            use_timestep_mask=use_timestep_mask,
            min_rank=min_rank,
            alpha_rank_scale=alpha_rank_scale,
            num_experts=num_experts,
            expert_init_std=expert_init_std,
            router_lr_scale=router_lr_scale,
            router_targets=router_targets,
            per_bucket_balance_weight=per_bucket_balance_weight,
            num_sigma_buckets=num_sigma_buckets,
            specialize_experts_by_sigma_buckets=specialize_experts_by_sigma_buckets,
            sigma_bucket_boundaries=sigma_bucket_boundaries,
            use_moe_style=use_moe_style,
            route_per_layer=route_per_layer,
            router_source=router_source,
            sigma_feature_dim=sigma_feature_dim,
            fei_feature_dim=fei_feature_dim,
            fei_sigma_low_div=fei_sigma_low_div,
            router_hidden_dim=router_hidden_dim,
            router_tau=router_tau,
            down_init=down_init,
            svd_slice=svd_slice,
            step_expert_K=step_expert_K,
            channel_scales_dict=channel_scales_dict,
            grad_basis_dict=grad_basis_dict,
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
        is_hydra: bool,
        hydra_num_experts: int,
        sigma_feature_dim_detected: Optional[int],
        sigma_router_names: Optional[List[str]],
        hydra_router_names: Optional[List[str]],
        channel_scales_dict: Optional[Dict[str, torch.Tensor]],
        specialize_experts_by_sigma_buckets: bool = False,
        num_sigma_buckets: Optional[int] = None,
        sigma_bucket_boundaries: Optional[List[float]] = None,
        fei_feature_dim: int = 0,
        fei_sigma_low_div: Optional[float] = None,
        fei_router_names: Optional[List[str]] = None,
        # Three-axis stamps from save metadata. All three must be present
        # for MoE checkpoints; unstamped MoE artifacts don't load.
        new_use_moe_style: Optional[str] = None,
        new_route_per_layer: Optional[bool] = None,
        new_router_source: Optional[str] = None,
        step_expert_K: int = 0,
    ) -> "LoRANetworkCfg":
        """Build cfg from a checkpoint key-sniff (warm-start / inference path).

        Per-module dims/alphas come from modules_dim/modules_alpha, so
        lora_dim/alpha here are placeholders; training-time schedules stay off.

        specialize_experts_by_sigma_buckets/num_sigma_buckets/
        sigma_bucket_boundaries are reconstructed from safetensors metadata —
        the partition leaves no tensor footprint.

        Non-MoE checkpoints have no three-axis stamps; absence = (False,
        False, "none"). Hydra checkpoints must carry all three.
        """
        if (
            new_use_moe_style is not None
            and new_route_per_layer is not None
            and new_router_source is not None
        ):
            use_moe_style: MoEStyle = _as_moe_style(new_use_moe_style)
            route_per_layer = bool(new_route_per_layer)
            router_source: RouterSource = _as_router_source(new_router_source)
        elif is_hydra:
            raise RuntimeError(
                "MoE checkpoint is missing the three-axis routing stamps "
                "(ss_use_moe_style / ss_route_per_layer / ss_router_source). "
                "Two common causes: (1) it is a pre-plan2 checkpoint, which "
                "stops loading by design — retrain the adapter to produce the "
                "new metadata; or (2) you passed a pre-loaded weights_sd= to "
                "create_network_from_weights without file= or metadata=. "
                "load_file() drops safetensors __metadata__, so the stamps "
                "vanish — pass file=<path> or metadata=<dict> so they survive."
            )
        else:
            use_moe_style = False
            route_per_layer = False
            router_source = "none"

        return cls(
            lora_dim=4,
            alpha=1.0,
            module_class=module_class,
            modules_dim=modules_dim,
            modules_alpha=modules_alpha,
            train_llm_adapter=train_llm_adapter,
            num_experts=hydra_num_experts if is_hydra else 4,
            channel_scales_dict=channel_scales_dict,
            use_moe_style=use_moe_style,
            route_per_layer=route_per_layer,
            router_source=router_source,
            sigma_feature_dim=(
                sigma_feature_dim_detected
                if sigma_feature_dim_detected is not None
                else 128
            ),
            sigma_router_names=sigma_router_names,
            hydra_router_names=hydra_router_names,
            specialize_experts_by_sigma_buckets=specialize_experts_by_sigma_buckets,
            num_sigma_buckets=(int(num_sigma_buckets) if num_sigma_buckets else 3),
            sigma_bucket_boundaries=sigma_bucket_boundaries,
            fei_feature_dim=int(fei_feature_dim),
            fei_sigma_low_div=(
                float(fei_sigma_low_div) if fei_sigma_low_div is not None else 4.0
            ),
            fei_router_names=fei_router_names,
            step_expert_K=int(step_expert_K),
        )
