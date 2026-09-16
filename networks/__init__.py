"""NetworkSpec registry for LoRA adapter-method dispatch.

Each entry pairs an adapter variant name with the module class it
instantiates and a ``save_variant`` label consumed by ``networks.lora_save``.
``resolve_network_spec`` maps the three-axis routing kwargs to an entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Type

from networks.lora_modules import (
    ChimeraHydraLoRAModule,
    HydraLoRAModule,
    LoRAModule,
    OrthoHydraLoRAModule,
    OrthoInitLoRAModule,
    OrthoLoRAModule,
    StackedExpertsLoRAModule,
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
# (``kwargs.get("router_hidden_dim", kwargs.get("router_hidden", 64))`` /
# ``kwargs.get("fera_num_bands", kwargs.get("num_bands", 3))``). The canonical
# names are forwarded; these aliases are not.
_KWARG_ALIAS_FALLBACKS = frozenset({"router_hidden", "num_bands"})


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
    # Mirror cfg.fera_fecl_weight to network.fecl_weight; fall back to the
    # kwarg when no cfg is present (unit tests).
    cfg_weight = getattr(getattr(network, "cfg", None), "fera_fecl_weight", None)
    if cfg_weight is not None:
        network.fecl_weight = float(cfg_weight)
    else:
        network.fecl_weight = float(kwargs.get("fera_fecl_weight", 0.0) or 0.0)

    # ChimeraHydra: stamp the chimera flag + per-pool balance weights, falling
    # back to the shared balance_loss_weight when per-pool weights are unset.
    cfg = getattr(network, "cfg", None)
    if cfg is not None and getattr(cfg, "use_chimera_hydra", False):
        network._use_chimera_hydra = True
        w_c = cfg.balance_w_content if cfg.balance_w_content is not None else target
        w_f = cfg.balance_w_freq if cfg.balance_w_freq is not None else target
        network._balance_w_content = float(w_c)
        # Hardwired-FEI freq gate has no router params (fixed function of z_t),
        # so a balance penalty on it is constant w.r.t. trained params (zero
        # gradient) — force w_f=0 to keep the loss scalar honest.
        if str(getattr(cfg, "freq_router_mode", "learned")).lower() == "fei":
            network._balance_w_freq = 0.0
        else:
            network._balance_w_freq = float(w_f)
    else:
        network._use_chimera_hydra = False


NETWORK_REGISTRY: Dict[str, NetworkSpec] = {
    "lora": NetworkSpec(
        name="lora",
        module_class=LoRAModule,
        save_variant="standard",
    ),
    "ortho": NetworkSpec(
        name="ortho",
        module_class=OrthoLoRAModule,
        save_variant="ortho_to_lora",
    ),
    # OrthoInit: trainable top-r SVD init of W0 (no Cayley/frozen subspace).
    # Distills to standard LoRA at save (sqrt-split λ → down/up), so the on-disk
    # form matches a distilled OrthoLoRA.
    "ortho_init": NetworkSpec(
        name="ortho_init",
        module_class=OrthoInitLoRAModule,
        save_variant="ortho_to_lora",
    ),
    "hydra": NetworkSpec(
        name="hydra",
        module_class=HydraLoRAModule,
        save_variant="hydra_moe",
        post_init=_post_init_hydra,
    ),
    "ortho_hydra": NetworkSpec(
        name="ortho_hydra",
        module_class=OrthoHydraLoRAModule,
        save_variant="ortho_hydra_to_hydra",
        post_init=_post_init_hydra,
    ),
    # ChimeraHydra: dual-pool additive routing on the OrthoHydra Cayley
    # parameterization. Save distills the Cayley params to the Hydra-MoE layout
    # in a *_chimera.safetensors sibling; the Cayley class is training-only
    # (resume loses the orthogonal parameterization, matching OrthoHydra).
    "chimera_hydra": NetworkSpec(
        name="chimera_hydra",
        module_class=ChimeraHydraLoRAModule,
        save_variant="chimera_hydra_moe",
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
    # FeRA paper-faithful: independent-A stacked experts, single network-level
    # router fed by FEI(z_t). Selected via use_moe_style="independent_A".
    "stacked_experts_global_fei": NetworkSpec(
        name="stacked_experts_global_fei",
        module_class=StackedExpertsLoRAModule,
        save_variant="stacked_experts_global_fei",
        post_init=_post_init_hydra,
    ),
}


def all_network_kwargs() -> Tuple[str, ...]:
    """Return the LoRA-family TOML allowlist (``NETWORK_KWARGS``), sorted.

    train.py uses it to populate the argparse schema and the TOML ->
    net_kwargs forwarding list.
    """
    return tuple(sorted(NETWORK_KWARGS))


def _parse_bool_flag(kwargs: Mapping[str, Any], key: str) -> bool:
    v = kwargs.get(key, False)
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).lower() == "true"


def resolve_network_spec(kwargs: Mapping[str, Any]) -> NetworkSpec:
    """Resolve which NetworkSpec to instantiate from create_network kwargs.

    Precedence (first match wins): use_chimera_hydra -> chimera_hydra;
    use_moe_style="independent_A" -> stacked_experts_global_fei (FeRA);
    use_moe_style="shared_A" (+use_ortho) -> ortho_hydra, else hydra;
    use_ortho_init -> ortho_init; use_ortho -> ortho; else lora. Raises on
    mutually-exclusive combinations.
    """
    use_ortho = _parse_bool_flag(kwargs, "use_ortho")
    use_ortho_init = _parse_bool_flag(kwargs, "use_ortho_init")
    use_chimera = _parse_bool_flag(kwargs, "use_chimera_hydra")
    if use_ortho and use_ortho_init:
        raise ValueError(
            "use_ortho and use_ortho_init are mutually exclusive: ortho freezes "
            "the SVD basis (Cayley-rotates within it); ortho_init trains the SVD "
            "basis (no cap). Pick one."
        )
    if use_chimera:
        # OrthoInit composes with chimera (swaps each pool's frozen Cayley
        # basis for trainable SVD-seeded bases); same spec either way.
        return NETWORK_REGISTRY["chimera_hydra"]

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
        raise ValueError(
            f"use_moe_style={raw_moe!r}: expected False, 'shared_A', or 'independent_A'."
        )
    if moe_style not in ("", "shared_A", "independent_A"):
        raise ValueError(
            f"use_moe_style={raw_moe!r}: expected False, 'shared_A', or 'independent_A'."
        )

    if use_ortho_init and moe_style:
        raise NotImplementedError(
            "use_ortho_init does not yet compose with use_moe_style — the "
            "orthoinit MoE pool is a separate family member (not implemented)."
        )
    if moe_style == "independent_A":
        return NETWORK_REGISTRY["stacked_experts_global_fei"]
    if moe_style == "shared_A":
        return (
            NETWORK_REGISTRY["ortho_hydra"] if use_ortho else NETWORK_REGISTRY["hydra"]
        )
    if use_ortho_init:
        return NETWORK_REGISTRY["ortho_init"]
    if use_ortho:
        return NETWORK_REGISTRY["ortho"]
    return NETWORK_REGISTRY["lora"]


__all__ = [
    "NetworkSpec",
    "NETWORK_REGISTRY",
    "NETWORK_KWARGS",
    "all_network_kwargs",
    "resolve_network_spec",
]
