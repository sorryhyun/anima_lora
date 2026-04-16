"""Backward-compat re-exports — canonical location is library.inference.*."""

from library.inference.generation import (  # noqa: F401
    GenerationSettings,
    get_generation_settings,
    compute_tile_positions,
    create_tile_blend_weight,
    generate_body,
    generate_body_tiled,
    generate,
)

from library.inference.output import (  # noqa: F401
    check_inputs,
    decode_latent,
    get_time_flag,
    save_latent,
    save_images,
    save_output,
)

from library.inference.models import (  # noqa: F401
    load_dit_model,
    load_text_encoder,
    load_shared_models,
)

from library.inference.text import (  # noqa: F401
    process_escape,
    prepare_text_inputs,
)
