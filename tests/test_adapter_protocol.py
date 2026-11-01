"""Contract tests for the adapter-network Protocol (proposal Part B3).

Two guards, both cheap and instantiation-free:

1. The three shipped adapter networks structurally satisfy
   ``networks.protocol.AdapterNetwork``; the LoRA family additionally satisfies
   the optional ``RouterConditionableNetwork`` per-step routing surface. This is
   the payoff test for B3 — if someone renames a trainer-facing method on one
   network without the others, it trips here.

2. ``library/inference/`` and ``anima_lora/`` import nothing from
   ``library/training/`` or the top-level ``train`` module. Today this holds by
   accident (zero direct imports either way); the test makes it an invariant so
   the inference/training split stays extractable for free.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from networks.protocol import AdapterNetwork, RouterConditionableNetwork

REPO = Path(__file__).resolve().parents[1]


def _networks():
    from networks.lora_anima.network import LoRANetwork
    from networks.methods.easycontrol import EasyControlNetwork
    from networks.methods.soft_tokens import SoftTokensNetwork

    return LoRANetwork, EasyControlNetwork, SoftTokensNetwork


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_networks_satisfy_core_protocol(idx):
    cls = _networks()[idx]
    # Non-data, @runtime_checkable protocol → issubclass checks method presence
    # without constructing the network (which would need a live DiT).
    assert issubclass(cls, AdapterNetwork), (
        f"{cls.__name__} is missing part of the AdapterNetwork surface: "
        + ", ".join(m for m in AdapterNetwork.__protocol_attrs__ if not hasattr(cls, m))
    )


def test_lora_family_satisfies_router_conditioning():
    LoRANetwork = _networks()[0]
    assert issubclass(LoRANetwork, RouterConditionableNetwork), (
        "LoRANetwork must expose the per-step routing setters "
        "(set_timestep_mask / set_sigma / set_fei / set_crossattn_routing)"
    )


# ---------------------------------------------------------------------------
# Import-boundary invariant
# ---------------------------------------------------------------------------

_FORBIDDEN_ROOTS = ("library.training", "train")


def _imports(path: Path):
    """Yield every fully-qualified module name imported by ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import — never crosses into training/
                continue
            mod = node.module or ""
            yield mod
            # ``from library import training`` — the imported *name* is the
            # offending submodule, not the module string.
            for alias in node.names:
                yield f"{mod}.{alias.name}" if mod else alias.name


def _is_forbidden(name: str) -> bool:
    return any(name == root or name.startswith(root + ".") for root in _FORBIDDEN_ROOTS)


@pytest.mark.core
@pytest.mark.parametrize("subtree", ["library/inference", "anima_lora"])
def test_inference_surface_does_not_import_training(subtree):
    root = REPO / subtree
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        for mod in _imports(path):
            if _is_forbidden(mod):
                offenders.append(f"{path.relative_to(REPO)} → {mod}")
    assert not offenders, (
        f"{subtree}/ must not import from library.training/ or train.py "
        "(keeps the inference/training split extractable). Offending imports:\n"
        + "\n".join(offenders)
    )
