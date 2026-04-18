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
        kwarg_flags: Tuple of kwargs this variant consumes beyond the
            shared set. Documentation only — enforcement lives in
            ``train.NETWORK_KWARG_ALLOWLIST``.
        post_init: Optional hook run after the network is built; receives
            ``(network, kwargs)``. Used for variant-specific attribute
            attachment (e.g. hydra balance loss weight).
    """

    name: str
    module_class: Type
    save_variant: str = "standard"
    kwarg_flags: Tuple[str, ...] = ()
    post_init: Optional[Callable[[Any, Mapping[str, Any]], None]] = None


def _post_init_hydra(network: Any, kwargs: Mapping[str, Any]) -> None:
    blw = kwargs.get("balance_loss_weight")
    network._balance_loss_weight = float(blw) if blw is not None else 0.01
    network._use_hydra = True


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
        kwarg_flags=("num_experts", "balance_loss_weight"),
        post_init=_post_init_hydra,
    ),
    "ortho_hydra": NetworkSpec(
        name="ortho_hydra",
        module_class=OrthoHydraLoRAExpModule,
        save_variant="ortho_hydra_to_hydra",
        kwarg_flags=("num_experts", "balance_loss_weight"),
        post_init=_post_init_hydra,
    ),
    "dora": NetworkSpec(
        name="dora",
        module_class=DoRAModule,
        save_variant="standard",
    ),
}


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
    "resolve_network_spec",
]
