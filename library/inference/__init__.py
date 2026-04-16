"""Anima inference package — model loading, generation, and output.

Uses PEP 562 lazy imports to avoid circular dependency chains
(library.utils → inference.sampling triggers __init__ → inference.models → anima_models → utils).
"""

import importlib as _importlib

_ATTR_TO_MODULE = {
    # sampling
    "get_timesteps_sigmas": "sampling",
    "step": "sampling",
    "ERSDESampler": "sampling",
    "GradualLatent": "sampling",
    "EulerAncestralDiscreteSchedulerGL": "sampling",
    # output
    "check_inputs": "output",
    "decode_latent": "output",
    "get_time_flag": "output",
    "save_latent": "output",
    "save_images": "output",
    "save_output": "output",
    # models
    "load_dit_model": "models",
    "load_text_encoder": "models",
    "load_shared_models": "models",
    # text
    "process_escape": "text",
    "prepare_text_inputs": "text",
    # mod_guidance
    "build_mod_schedule": "mod_guidance",
    "setup_mod_guidance": "mod_guidance",
    # generation
    "GenerationSettings": "generation",
    "get_generation_settings": "generation",
    "compute_tile_positions": "generation",
    "create_tile_blend_weight": "generation",
    "generate_body": "generation",
    "generate_body_tiled": "generation",
    "generate": "generation",
}


def __getattr__(name: str):
    if name in _ATTR_TO_MODULE:
        module = _importlib.import_module(f".{_ATTR_TO_MODULE[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_ATTR_TO_MODULE.keys())
