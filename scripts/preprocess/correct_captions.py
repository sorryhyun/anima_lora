#!/usr/bin/env python3
"""Write corrected captions next to resized preprocessing images."""

from __future__ import annotations

import argparse
from pathlib import Path

from library.captioning.correction import (
    CaptionCorrectionOptions,
    find_tag_csv,
    load_tag_knowledge_base,
)
from library.captioning.preprocess import write_corrected_preprocess_captions


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Raw source image directory")
    parser.add_argument("--dst", required=True, help="Resized image directory")
    parser.add_argument(
        "--tag_csv",
        default=None,
        help="danbooru_tags_classified.csv path (default: models/ lookup)",
    )
    parser.add_argument(
        "--path_pattern",
        "--path-pattern",
        dest="path_pattern",
        default="*",
        help="Only write captions for resized images matching this relative glob",
    )
    parser.add_argument("--recursive", action="store_true", help="Walk subfolders")
    parser.add_argument(
        "--caption_insert_no_artist",
        "--caption-insert-no-artist",
        dest="caption_insert_no_artist",
        action="store_true",
        help="Insert @no-artist at the artist slot when no artist marker exists",
    )
    parser.add_argument(
        "--caption_trigger_word",
        "--caption-trigger-word",
        dest="caption_trigger_word",
        default="",
        help="Trigger tag to move into the caption order",
    )
    parser.add_argument(
        "--caption_trigger_at_front",
        "--caption-trigger-at-front",
        dest="caption_trigger_at_front",
        action="store_true",
        help="Place caption_trigger_word at the very front instead of artist slot",
    )
    args = parser.parse_args()

    csv_path = Path(args.tag_csv) if args.tag_csv else find_tag_csv(ROOT)
    if csv_path is None or not csv_path.exists():
        raise SystemExit(
            "danbooru_tags_classified.csv not found. Run "
            "`python tasks.py download-danbooru-tags` first."
        )

    stats = write_corrected_preprocess_captions(
        Path(args.src),
        Path(args.dst),
        load_tag_knowledge_base(csv_path),
        options=CaptionCorrectionOptions(
            insert_no_artist=bool(args.caption_insert_no_artist),
            trigger_word=str(args.caption_trigger_word or ""),
            trigger_at_front=bool(args.caption_trigger_at_front),
        ),
        recursive=bool(args.recursive),
        path_pattern=str(args.path_pattern or "*"),
    )
    print(
        "Corrected preprocess captions: "
        f"{stats.written} written, {stats.unchanged} unchanged, "
        f"{stats.missing_source} missing source, {stats.removed_stale} stale removed "
        f"({stats.seen} resized images)"
    )


if __name__ == "__main__":
    main()
