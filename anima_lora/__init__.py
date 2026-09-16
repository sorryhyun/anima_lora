"""Anima — programmatic front door.

Lazy re-exports of the entry points an embedder needs::

    import anima_lora

    settings = anima_lora.inference.get_generation_settings(args)
    latent = anima_lora.inference.generate(args, settings)
    image = anima_lora.inference.decode_to_pil(vae, latent, device)

Exports live on the namespaced submodules ``anima_lora.{models, inference,
config, training, captioning}``; each submodule's docstring lists its exports
and their canonical homes. The pre-namespace flat names (``anima_lora.generate``,
``anima_lora.load_vae``, …) remain as aliases; ``training`` is namespaced-only.
Names resolve lazily (PEP 562), so ``import anima_lora`` stays cheap and avoids
the circular-import chains of the underlying packages.

``ROOT`` is the repo root (the directory holding ``configs/``, ``output/`` …)
as a ``pathlib.Path``.

This package is the **stable API**. ``library.*`` / ``networks.*`` /
``scripts.*`` are importable for advanced use but may change without a
deprecation cycle; pin a tag if you depend on them directly.

Repo-relative model/config paths resolve against the repo home, not the CWD
(``library.env.resolve_under_home`` / ``anima_home``; set ``ANIMA_HOME`` for a
relocated checkout).
"""

from __future__ import annotations

from pathlib import Path as _Path

from anima_lora._lazy import attach as _attach

#: Repo root (``anima_lora/``), resolved from this file's location.
ROOT = _Path(__file__).resolve().parent.parent

# Pre-namespace flat aliases: export name -> dotted module that defines it.
# New exports go on the namespaced submodules, not here.
_ATTR_TO_MODULE: dict[str, str] = {
    # generation + output (anima_lora.inference)
    "generate": "library.inference",
    "get_generation_settings": "library.inference",
    "save_output": "library.inference",
    "decode_to_pil": "library.inference",
    "GenerationRequest": "library.inference",
    "prepare_text_inputs": "library.inference",
    "ensure_text_strategies": "library.inference",
    # config merge chain (anima_lora.config)
    "load_method_preset": "library.config.io",
    "read_config_from_file": "library.config.io",
    # model loaders (anima_lora.models)
    "load_anima_model": "library.anima.weights",
    "load_dit_model": "library.inference.models",
    "load_vae": "library.models.qwen_vae",
    # captioning (anima_lora.captioning)
    "AnimaTagger": "anime_tools.tagger",
    # device / dtype helpers (anima_lora.models)
    "str_to_dtype": "library.runtime.device",
    # default checkpoint paths (anima_lora.models)
    "default_checkpoints": "library.env",
    "DefaultCheckpoints": "library.env",
    "VocabPack": "library.anima.vocab_pack",
    "load_vocab_pack": "library.anima.vocab_pack",
    "attach_vocab_pack": "library.anima.vocab_pack",
}

_SUBMODULES = ["captioning", "config", "inference", "models", "training"]

__all__ = sorted([*_ATTR_TO_MODULE, *_SUBMODULES, "ROOT"])

_attach(globals(), _ATTR_TO_MODULE)

# Eager: each submodule is only a lazy re-export table (no heavy imports), so
# `anima_lora.models.load_vae` works right after `import anima_lora`.
from anima_lora import captioning as captioning  # noqa: E402
from anima_lora import config as config  # noqa: E402
from anima_lora import inference as inference  # noqa: E402
from anima_lora import models as models  # noqa: E402
from anima_lora import training as training  # noqa: E402
