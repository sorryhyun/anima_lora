"""The thin position-clause format both lines cache with (torch-free leaf).

anime_tools ``PositionRequest`` fields: every tag a subject's crop carries moves
from the shared bag into that subject's clause (repeated subjects may repeat a
tag), only hair / eye color stay one per subject, and a man the caption counts
gets a clause of his own unless it is a POV picture. Each subject mask grows by
``mask_dilate`` px before the crop tagger sees it, so an object in hand lands in
its holder's clause. Settled on MOSS (caststation A/B, 2026-11-15) and adopted for
Anima the same day (ama mitsuki A/B, ``post_image_dataset/capab``).
"""

from __future__ import annotations

POSITION_FORMAT: dict[str, object] = {
    "thin_bag": True,
    "discriminative_only": False,
    "multi_view_gate": False,
    "free_groups": True,
    "bind_males": True,
}

MASK_DILATE = 16
MAX_CLAUSE_TAGS = 32
MAX_NOVEL_TAGS = 0


def position_fields() -> dict[str, object]:
    """``POSITION_FORMAT`` plus the dilate / clause-cap / novel-tag defaults."""
    return {
        **POSITION_FORMAT,
        "mask_dilate": MASK_DILATE,
        "max_clause_tags": MAX_CLAUSE_TAGS,
        "max_novel_tags": MAX_NOVEL_TAGS,
    }
