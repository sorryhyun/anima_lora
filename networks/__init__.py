"""NetworkSpec registry for LoRA adapter-method dispatch.

Each entry pairs an adapter variant name with the module class it
instantiates and a ``save_variant`` label consumed by ``networks.lora_save``.
``resolve_network_spec`` maps the three-axis routing kwargs to an entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Type

from networks.lora_modules import (
    HydraLoRAModule,
    LoRAModule,
    StepExpertLoRAModule,
)


@dataclass(frozen=True)
class NetworkSpec:
    """Descriptor for one adapter variant.

    Attributes:
        name: Stable identifier stamped on the network as ``ss_network_spec``
            and the key into ``NETWORK_REGISTRY``.
        module_class: Concrete ``LoRAModule`` subclass to instantiate per
            target module.
        save_variant: Key into ``networks.lora_save.SAVE_HANDLERS``.
        post_init: Optional hook run after the network is built, receiving
            ``(network, kwargs)`` — for variant-specific attribute attachment.
    """

    name: str
    module_class: Type
    save_variant: str = "standard"
    post_init: Optional[Callable[[Any, Mapping[str, Any]], None]] = None


# Modules scanned by ``_derive_network_kwargs`` for the TOML allowlist.
_KWARG_CONSUMER_MODULES = (
    "lora_anima/config.py",  # LoRANetworkCfg.from_kwargs
    "lora_anima/factory.py",  # REPA / loraplus / channel_scaling / custom_down
    "__init__.py",  # _post_init_hydra
)

# Read positionally as the *default* of a canonical key
# (``kwargs.get("router_hidden_dim", kwargs.get("router_hidden", 64))``). The
# canonical name is forwarded; this alias is not.
_KWARG_ALIAS_FALLBACKS = frozenset({"router_hidden"})


def _derive_network_kwargs() -> frozenset[str]:
    """Every literal key the LoRA-family consumers read via ``kwargs.get(...)``.

    AST-scans the consumer modules. Recognizes the ``kwargs.get("literal"[, default])`` form
    only — a consumer reading a forwarded knob another way (``kwargs["k"]``
    indexing, a helper wrapper) won't be picked up.
    """
    import ast
    from pathlib import Path

    pkg = Path(__file__).resolve().parent
    keys: set[str] = set()
    for rel in _KWARG_CONSUMER_MODULES:
        tree = ast.parse((pkg / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "kwargs"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                keys.add(node.args[0].value)
    return frozenset(keys - _KWARG_ALIAS_FALLBACKS)


NETWORK_KWARGS: frozenset[str] = _derive_network_kwargs()


def _post_init_hydra(network: Any, kwargs: Mapping[str, Any]) -> None:
    blw = kwargs.get("balance_loss_weight")
    target = float(blw) if blw is not None else 0.01
    warmup = kwargs.get("balance_loss_warmup_ratio")
    warmup_ratio = float(warmup) if warmup is not None else 0.0
    network._balance_loss_target_weight = target
    network._balance_loss_warmup_ratio = warmup_ratio
    # Hold the balance penalty at 0 during warmup so the router can specialize
    # first; flipped to `target` by LoRANetwork.step_balance_loss_warmup.
    network._balance_loss_weight = 0.0 if warmup_ratio > 0.0 else target
    network._use_hydra = True


NETWORK_REGISTRY: Dict[str, NetworkSpec] = {
    "lora": NetworkSpec(
        name="lora",
        module_class=LoRAModule,
        save_variant="standard",
    ),
    "hydra": NetworkSpec(
        name="hydra",
        module_class=HydraLoRAModule,
        save_variant="hydra_moe",
        post_init=_post_init_hydra,
    ),
    # Step-expert: shared down-proj + K step-indexed up-heads, hard-selected by
    # diffusion step (no router). Turbo DP-DMD student only; kept-live at
    # inference (K heads can't fold into one DiT weight), so save is bespoke.
    "step_expert": NetworkSpec(
        name="step_expert",
        module_class=StepExpertLoRAModule,
        save_variant="step_expert",
    ),
}


def all_network_kwargs() -> Tuple[str, ...]:
    """Return the LoRA-family TOML allowlist (``NETWORK_KWARGS``), sorted.

    train.py uses it to populate the argparse schema and the TOML ->
    net_kwargs forwarding list.
    """
    return tuple(sorted(NETWORK_KWARGS))


def resolve_network_spec(kwargs: Mapping[str, Any]) -> NetworkSpec:
    """Resolve which NetworkSpec to instantiate from create_network kwargs.

    Precedence (first match wins): step_expert_K > 1 -> step_expert;
    use_moe_style="shared_A" -> hydra; else lora.
    """
    # Step-expert short-circuits when step_expert_K > 1; K==1 collapses to plain LoRA
    raw_step_K = kwargs.get("step_expert_K")
    if raw_step_K is not None and int(raw_step_K) > 1:
        return NETWORK_REGISTRY["step_expert"]

    raw_moe = kwargs.get("use_moe_style")
    if isinstance(raw_moe, str):
        moe_style = raw_moe.strip()
        if moe_style.lower() in ("false", "none", ""):
            moe_style = ""
    elif raw_moe is False or raw_moe is None:
        moe_style = ""
    else:
        raise ValueError(f"use_moe_style={raw_moe!r}: expected False or 'shared_A'.")
    if moe_style not in ("", "shared_A"):
        raise ValueError(f"use_moe_style={raw_moe!r}: expected False or 'shared_A'.")

    if moe_style == "shared_A":
        return NETWORK_REGISTRY["hydra"]
    return NETWORK_REGISTRY["lora"]


__all__ = [
    "NetworkSpec",
    "NETWORK_REGISTRY",
    "NETWORK_KWARGS",
    "all_network_kwargs",
    "resolve_network_spec",
]
