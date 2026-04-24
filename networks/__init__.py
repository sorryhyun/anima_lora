"""NetworkSpec registry for LoRA adapter-method dispatch.

Replaces the flag-cascade in ``networks.lora_anima.create_network`` with a
declarative map. Each entry pairs an adapter variant name with the module
class it instantiates and a ``save_variant`` label consumed by
``networks.lora_save``.

Flag precedence (evaluated top to bottom, first match wins):

    use_hydra + use_ortho → ortho_hydra
    use_hydra             → hydra
    use_dora              → dora
    use_ortho             → ortho
    (none)                → lora

Ambiguous combinations (``use_dora`` + ``use_ortho``, …) raise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Type

from networks.lora_deprecated import DoRAModule
from networks.lora_modules import (
    HydraLoRAModule,
    LoRAModule,
    OrthoHydraLoRAExpModule,
    OrthoLoRAExpModule,
)


@dataclass(frozen=True)
class NetworkSpec:
    """Descriptor for one adapter variant.

    Attributes:
        name: Stable identifier stamped on the network and written to
            metadata as ``ss_network_spec``. Also the key into
            ``NETWORK_REGISTRY``.
        module_class: Concrete ``LoRAModule`` subclass the network will
            instantiate per target module.
        save_variant: Label keyed into ``networks.lora_save.SAVE_HANDLERS``
            — selects the serialization pipeline for this variant.
        kwarg_flags: Tuple of kwargs this variant consumes beyond
            ``SHARED_KWARG_FLAGS``. Combined with the shared set by
            ``all_network_kwargs()`` to populate argparse schema and
            forward TOML-level args into ``create_network``. Single
            source of truth for what keys train.py recognizes.
        post_init: Optional hook run after the network is built; receives
            ``(network, kwargs)``. Used for variant-specific attribute
            attachment (e.g. hydra balance loss weight).
    """

    name: str
    module_class: Type
    save_variant: str = "standard"
    kwarg_flags: Tuple[str, ...] = ()
    post_init: Optional[Callable[[Any, Mapping[str, Any]], None]] = None


# Kwargs every LoRA-family variant consumes in ``create_network``: core
# targeting knobs + cross-cutting add-ons (ReFT, APEX, channel scaling,
# LoRA+, T-LoRA). Cross-cutting because these compose on top of any
# variant rather than belonging to a single one.
SHARED_KWARG_FLAGS: Tuple[str, ...] = (
    # Core network targeting / knobs
    "train_llm_adapter",
    "exclude_patterns",
    "include_patterns",
    "layer_start",
    "layer_end",
    "rank_dropout",
    "module_dropout",
    "verbose",
    # Regex-driven per-module rank / lr overrides
    "network_reg_dims",
    "network_reg_lrs",
    # HydraLoRA router (+ σ-router MLP) LR multiplier on top of unet_lr / reg_lr
    "network_router_lr_scale",
    # LoRA+
    "loraplus_lr_ratio",
    "loraplus_unet_lr_ratio",
    "loraplus_text_encoder_lr_ratio",
    # T-LoRA (timestep-dependent rank masking)
    "use_timestep_mask",
    "min_rank",
    "alpha_rank_scale",
    # Per-channel input pre-scaling (SmoothQuant-style)
    "per_channel_scaling",
    "channel_stats_path",
    "channel_scaling_alpha",
    # Memory-saving down-projection autograd (classic LoRA only; bitwise-equal grads)
    "use_custom_down_autograd",
    # Variant selectors (read by resolve_network_spec)
    "use_dora",
    "use_hydra",
    "use_ortho",
    # ReFT add-on (composes with any variant)
    "add_reft",
    "reft_dim",
    "reft_alpha",
    "reft_layers",
    # APEX self-adversarial condition-space shifting (composes with any variant)
    "apex_condition_shift_mode",
    "apex_condition_shift_init_a",
    "apex_condition_shift_init_b",
    "apex_shift_lr_scale",
    "apex_condition_shift_dim",
)


def _post_init_hydra(network: Any, kwargs: Mapping[str, Any]) -> None:
    blw = kwargs.get("balance_loss_weight")
    target = float(blw) if blw is not None else 0.01
    warmup = kwargs.get("balance_loss_warmup_ratio")
    warmup_ratio = float(warmup) if warmup is not None else 0.0
    network._balance_loss_target_weight = target
    network._balance_loss_warmup_ratio = warmup_ratio
    # Hold the balance penalty at 0 during the warmup window so the router can
    # specialize before load-balancing kicks in; flipped to `target` by
    # LoRANetwork.step_balance_loss_warmup once global_step crosses the ratio.
    network._balance_loss_weight = 0.0 if warmup_ratio > 0.0 else target
    network._use_hydra = True


_HYDRA_KWARG_FLAGS: Tuple[str, ...] = (
    "num_experts",
    "balance_loss_weight",
    "balance_loss_warmup_ratio",
    "expert_warmup_ratio",
    # Layer filter — concentrates MoE routers on cross-attn + MLP
    "hydra_router_layers",
    # σ-conditional router add-on (hydra-only)
    "use_sigma_router",
    "sigma_router_layers",
    "sigma_feature_dim",
    "sigma_hidden_dim",
    "per_bucket_balance_weight",
    "num_sigma_buckets",
)


NETWORK_REGISTRY: Dict[str, NetworkSpec] = {
    "lora": NetworkSpec(
        name="lora",
        module_class=LoRAModule,
        save_variant="standard",
    ),
    "ortho": NetworkSpec(
        name="ortho",
        module_class=OrthoLoRAExpModule,
        save_variant="ortho_to_lora",
    ),
    "hydra": NetworkSpec(
        name="hydra",
        module_class=HydraLoRAModule,
        save_variant="hydra_moe",
        kwarg_flags=_HYDRA_KWARG_FLAGS,
        post_init=_post_init_hydra,
    ),
    "ortho_hydra": NetworkSpec(
        name="ortho_hydra",
        module_class=OrthoHydraLoRAExpModule,
        save_variant="ortho_hydra_to_hydra",
        kwarg_flags=_HYDRA_KWARG_FLAGS,
        post_init=_post_init_hydra,
    ),
    "dora": NetworkSpec(
        name="dora",
        module_class=DoRAModule,
        save_variant="standard",
    ),
}


def all_network_kwargs() -> Tuple[str, ...]:
    """Return the union of shared + per-variant kwargs, sorted.

    Single source of truth for train.py — populates the argparse schema
    and the TOML → ``net_kwargs`` forwarding list, so adding a new kwarg
    to a ``NetworkSpec`` (or to ``SHARED_KWARG_FLAGS``) automatically
    makes it visible to training without touching train.py.
    """
    merged: set[str] = set(SHARED_KWARG_FLAGS)
    for spec in NETWORK_REGISTRY.values():
        merged.update(spec.kwarg_flags)
    return tuple(sorted(merged))


def _parse_bool_flag(kwargs: Mapping[str, Any], key: str) -> bool:
    v = kwargs.get(key, False)
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).lower() == "true"


def resolve_network_spec(kwargs: Mapping[str, Any]) -> NetworkSpec:
    """Resolve which NetworkSpec to instantiate from create_network kwargs.

    Precedence is deterministic and documented in the module docstring.
    Raises on mutually-exclusive combinations.
    """
    use_hydra = _parse_bool_flag(kwargs, "use_hydra")
    use_dora = _parse_bool_flag(kwargs, "use_dora")
    use_ortho = _parse_bool_flag(kwargs, "use_ortho")

    # Reject ambiguous combinations early so the user gets a clear message
    # instead of silently-picked winner.
    if use_dora and (use_hydra or use_ortho):
        raise ValueError(
            "use_dora is mutually exclusive with use_hydra / use_ortho"
        )

    if use_hydra and use_ortho:
        return NETWORK_REGISTRY["ortho_hydra"]
    if use_hydra:
        return NETWORK_REGISTRY["hydra"]
    if use_dora:
        return NETWORK_REGISTRY["dora"]
    if use_ortho:
        return NETWORK_REGISTRY["ortho"]
    return NETWORK_REGISTRY["lora"]


__all__ = [
    "NetworkSpec",
    "NETWORK_REGISTRY",
    "SHARED_KWARG_FLAGS",
    "all_network_kwargs",
    "resolve_network_spec",
]
