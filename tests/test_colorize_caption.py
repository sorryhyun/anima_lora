"""The colorize caption's series prefix through the trainer's caption variants.

The colour-tag filter itself is tested in anime_tools (``anime_tools.captions.color_tags``).
"""

from __future__ import annotations

import random

from anime_tools.captions.color_tags import (
    filter_to_colors_and_copyright,
    is_copyright_tag,
)

from library.preprocess import generate_caption_variants

_COPYRIGHT = frozenset({"genshin impact", "fate/grand order"})


def test_copyright_protected_from_dropout():
    random.seed(0)
    caption = filter_to_colors_and_copyright(
        "genshin impact, blue hair, red dress, green eyes", _COPYRIGHT
    )
    variants = generate_caption_variants(
        caption,
        num_variants=4,
        tag_dropout_rate=1.0,
        protect_fn=lambda t: is_copyright_tag(t, _COPYRIGHT),
    )
    assert variants[0] == caption
    for v in variants[1:]:
        assert "genshin impact" in v
        assert "blue hair" not in v and "red dress" not in v and "green eyes" not in v
