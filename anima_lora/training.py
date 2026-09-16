"""In-process training — ``anima_lora.training``.

The exports ``examples/02_config_and_train.py`` uses to reproduce
``train.py``'s ``__main__`` block from Python:

| export | canonical home |
|--------|----------------|
| ``AnimaTrainer`` / ``setup_parser`` / ``build_network_extras`` / ``verify_command_line_training_args`` | repo-root ``train.py`` |
| ``create_network`` | ``networks.lora_anima`` |
| ``resolve_network_spec`` | ``networks`` |

``train.py`` is a repo-root script, not an installed module, so it is loaded by
path (against :data:`anima_lora.ROOT`) and works from any CWD.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from anima_lora import ROOT
from anima_lora._lazy import attach

_TRAIN_PY = ROOT / "train.py"


def _train_module():
    """Load repo-root ``train.py``, sharing one instance with ``import train``.

    Registers under the plain ``train`` name when that slot is free, so example
    scripts doing ``from train import AnimaTrainer`` (repo root on ``sys.path``)
    and the façade see the same module; if an unrelated ``train`` module is
    already imported, ours lives under a private key instead.
    """
    existing = sys.modules.get("train")
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file is not None and Path(existing_file).resolve() == _TRAIN_PY:
            return existing
    cached = sys.modules.get("anima_lora._train_py")
    if cached is not None:
        return cached
    name = "train" if existing is None else "anima_lora._train_py"
    spec = importlib.util.spec_from_file_location(name, _TRAIN_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


attach(
    globals(),
    {
        "AnimaTrainer": _train_module,
        "setup_parser": _train_module,
        "build_network_extras": _train_module,
        "verify_command_line_training_args": _train_module,
        "create_network": "networks.lora_anima",
        "resolve_network_spec": "networks",
    },
)
