"""Caption correction helpers for preprocessing outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from library.captioning.correction import (
    CaptionCorrectionOptions,
    TagKnowledgeBase,
    correct_caption,
)
from library.preprocess._dataset import walk_images


@dataclass
class PreprocessCaptionStats:
    seen: int = 0
    written: int = 0
    unchanged: int = 0
    removed_stale: int = 0
    missing_source: int = 0


def write_corrected_preprocess_captions(
    source_dir: Path,
    resized_dir: Path,
    kb: TagKnowledgeBase,
    *,
    options: CaptionCorrectionOptions,
    recursive: bool = True,
    path_pattern: str | None = None,
) -> PreprocessCaptionStats:
    """Write corrected ``.txt`` captions next to already-resized images.

    The source captions are never modified. The resized image tree is the
    authority because it already reflects low-res filtering, curation decisions,
    path_scope, and path_pattern from the resize stage.
    """

    stats = PreprocessCaptionStats()
    images = walk_images(resized_dir, recursive=recursive, pattern=path_pattern)
    stats.seen = len(images)

    for image_path in images:
        rel_caption = image_path.relative_to(resized_dir).with_suffix(".txt")
        src_caption = source_dir / rel_caption
        dst_caption = resized_dir / rel_caption

        if not src_caption.exists():
            stats.missing_source += 1
            if dst_caption.exists():
                dst_caption.unlink()
                stats.removed_stale += 1
            continue

        raw = src_caption.read_text(encoding="utf-8").strip()
        corrected = correct_caption(raw, kb, options=options).text
        if dst_caption.exists() and dst_caption.read_text(encoding="utf-8") == corrected:
            stats.unchanged += 1
            continue
        dst_caption.parent.mkdir(parents=True, exist_ok=True)
        dst_caption.write_text(corrected, encoding="utf-8")
        stats.written += 1

    return stats
