"""Generation + output — ``anima_lora.inference``.

Every export's canonical home is ``library.inference``: the request-driven
engine (build a typed ``GenerationRequest``, call ``.to_args()``) plus the
settings/generate/save trio and the text-strategy installers.
"""

from __future__ import annotations

from anima_lora._lazy import attach

attach(
    globals(),
    {
        "generate": "library.inference",
        "get_generation_settings": "library.inference",
        "save_output": "library.inference",
        "decode_to_pil": "library.inference",
        "GenerationRequest": "library.inference",
        "prepare_text_inputs": "library.inference",
        "ensure_text_strategies": "library.inference",
        # --decoder pixel_head: attach before sampling, decode after, save
        "ensure_pixel_decoder": "library.inference",
        "pixel_decode": "library.inference",
        "save_decoded": "library.inference",
    },
)
