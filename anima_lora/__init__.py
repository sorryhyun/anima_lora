"""Anima — programmatic front door.

A thin façade that re-exports the handful of real entry points an embedder
needs, so driving the pipeline is "read these exports" instead of
"reverse-engineer ``inference.py`` / ``train.py`` ``main()``"::

    import anima_lora

    settings = anima_lora.inference.get_generation_settings(args)
    latent = anima_lora.inference.generate(args, settings)
    image = anima_lora.inference.decode_to_pil(vae, latent, device)

The façade is grouped into curated submodules — the preferred spelling:

| namespace | exports |
|-----------|---------|
| ``anima_lora.models`` | ``load_dit_model`` / ``load_anima_model`` / ``load_vae`` / ``default_checkpoints`` / ``DefaultCheckpoints`` / ``str_to_dtype`` |
| ``anima_lora.inference`` | ``generate`` / ``get_generation_settings`` / ``save_output`` / ``decode_to_pil`` / ``GenerationRequest`` / ``prepare_text_inputs`` / ``ensure_text_strategies`` |
| ``anima_lora.config`` | ``load_method_preset`` / ``read_config_from_file`` |
| ``anima_lora.training`` | ``AnimaTrainer`` / ``setup_parser`` / ``build_network_extras`` / ``verify_command_line_training_args`` / ``create_network`` / ``resolve_network_spec`` |
| ``anima_lora.captioning`` | ``AnimaTagger`` |

Every pre-namespace flat name (``anima_lora.generate``,
``anima_lora.load_vae``, …) keeps working as an alias; names added after the
namespacing (the ``training`` surface) are namespaced-only. Each name resolves
lazily (PEP 562) on first access, so ``import anima_lora`` itself stays cheap
and avoids the circular-import chains the underlying packages guard against.

The canonical homes are unchanged — this package only re-exports them (each
submodule's docstring carries its export → home map). ``anima_lora.training``
is the one that reaches outside the installed packages: repo-root ``train.py``
is loaded by path, so the trainer works from any CWD.

``ROOT`` is the repo root (the directory holding ``configs/``, ``output/`` …) as
a ``pathlib.Path`` — the single source of truth for building repo-relative paths
in tooling, instead of each script re-deriving it with its own
``Path(__file__).parents[N]`` arithmetic.

This package is the **stable API**. ``library.*`` / ``networks.*`` /
``scripts.*`` are installed and importable for advanced use, but may change
without a deprecation cycle; pin a tag if you depend on them directly.

Note: repo-relative model/config paths resolve against the repo home, not the
CWD, so ``import anima_lora`` works from anywhere (see
``library.env.resolve_under_home`` / ``anima_home``; set ``ANIMA_HOME`` for a
relocated checkout). ``bench/`` / ``scripts/`` / ``preprocess/`` still need
their ``sys.path`` bootstrap to import sibling modules — those trees aren't
installed packages.
"""

from __future__ import annotations

from pathlib import Path as _Path

from anima_lora._lazy import attach as _attach

#: Repo root (``anima_lora/``), resolved from this file's location.
ROOT = _Path(__file__).resolve().parent.parent

# Pre-namespace flat aliases: export name -> dotted module that defines it.
# Frozen for back-compat — new exports go on the namespaced submodules below.
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

# Eager: each submodule is just the lazy re-export table (no heavy imports), and
# importing them here makes `anima_lora.models.load_vae` work right after
# `import anima_lora`.
from anima_lora import captioning as captioning  # noqa: E402
from anima_lora import config as config  # noqa: E402
from anima_lora import inference as inference  # noqa: E402
from anima_lora import models as models  # noqa: E402
from anima_lora import training as training  # noqa: E402
