"""Anima inference package — model loading, generation, and output.

Uses PEP 562 lazy imports to avoid circular dependency chains
(library.log → inference.sampling triggers __init__ → inference.models → anima.models → log).
"""

import importlib as _importlib

_ATTR_TO_MODULE = {
    "get_timesteps_sigmas": "sampling",
    "step": "sampling",
    "ERSDESampler": "sampling",
    "GradualLatent": "sampling",
    "EulerAncestralDiscreteSchedulerGL": "sampling",
    "check_inputs": "output",
    "decode_latent": "output",
    "decode_to_pil": "output",
    "pixels_to_pil": "output",
    "get_time_flag": "output",
    "save_latent": "output",
    "save_images": "output",
    "save_output": "output",
    "write_gen_manifest": "output",
    "load_dit_model": "models",
    "swap_adapters": "models",
    "attach_adapters": "models",
    "load_text_encoder": "models",
    "load_shared_models": "models",
    "process_escape": "text",
    "prepare_text_inputs": "text",
    "ensure_text_strategies": "text",
    # request object (typed front door for generate())
    "GenerationRequest": "request",
    # argument parser (single source of truth for generate()'s getattr knobs)
    "build_parser": "args",
    "build_default_args": "args",
    "build_mod_schedule": "corrections.mod_guidance",
    "setup_mod_guidance": "corrections.mod_guidance",
    "GenerationSettings": "generation",
    "get_generation_settings": "generation",
    "resolve_seed": "generation",
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
