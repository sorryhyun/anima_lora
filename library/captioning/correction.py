"""Caption ordering helpers for Anima-style Danbooru captions.

Pure-stdlib and GUI-safe: the tag database is loaded lazily from a local CSV
only when correction is requested.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from library.captioning.taxonomy import CAPTION_RATINGS, is_artist_tag, is_count_tag


_CATEGORY_RE = re.compile(r"^\s*\[([^\]]+)\]")
_SPACE_RE = re.compile(r"\s+")

# Fallback path works for this source tree without importing gui. GUI passes
# ROOT explicitly so installed layouts can keep the CSV under models/.
_REPO_ROOT = Path(__file__).resolve().parents[2]

_DANBOORU_NUMERIC_CATEGORIES = {
    "0": "general",
    "1": "artist",
    "3": "copyright",
    "4": "character",
    "5": "meta",
}

_ANIMA_PERSON_TAGS = frozenset(
    {
        "solo",
        "no humans",
    }
)


@dataclass(frozen=True)
class CaptionCorrectionOptions:
    """User-visible caption correction switches."""

    insert_no_artist: bool = True
    validate_artist_tags: bool = False
    trigger_word: str = ""
    trigger_at_front: bool = False


@dataclass(frozen=True)
class CaptionCorrectionResult:
    text: str
    changed: bool
    inserted_no_artist: bool
    unknown_tags: tuple[str, ...]


@dataclass(frozen=True)
class TagInfo:
    name: str
    kind: str
    category_path: str


class TagKnowledgeBase:
    """Minimal Danbooru tag classifier backed by ``danbooru_tags_classified.csv``."""

    def __init__(self, tags: dict[str, TagInfo], source: Path):
        self.tags = tags
        self.source = source

    def classify(self, tag: str) -> str | None:
        info = self.tags.get(tag_key(tag))
        return info.kind if info else None

    def has_artist(self, tag: str) -> bool:
        bare = tag[1:] if tag.startswith("@") else tag
        return self.classify(bare) == "artist"


def default_tag_csv_candidates(root: Path | None = None) -> list[Path]:
    """Likely local CSV locations, in priority order."""

    paths: list[Path] = []
    if root is not None:
        paths.append(root / "models" / "danbooru_tags_classified.csv")
    paths.append(_REPO_ROOT / "models" / "danbooru_tags_classified.csv")
    env = os.environ.get("ANIMA_DANBOORU_TAGS_CSV")
    if env:
        paths.append(Path(env))

    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        try:
            key = path.resolve()
        except OSError:
            key = path
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def find_tag_csv(root: Path | None = None) -> Path | None:
    for path in default_tag_csv_candidates(root):
        if path.exists():
            return path
    return None


def load_tag_knowledge_base(path: str | Path) -> TagKnowledgeBase:
    csv_path = Path(path)
    tags: dict[str, TagInfo] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_name = str(row.get("name") or "").strip()
            if not raw_name:
                continue
            category_path = _description_category_path(str(row.get("description") or ""))
            kind = _kind_from_category(category_path, str(row.get("category") or ""))
            tags[tag_key(raw_name)] = TagInfo(
                name=normalize_tag(raw_name),
                kind=kind,
                category_path=category_path,
            )
    return TagKnowledgeBase(tags=tags, source=csv_path)


def normalize_tag(tag: str) -> str:
    tag = tag.strip().replace("_", " ").lower()
    return _SPACE_RE.sub(" ", tag)


def tag_key(tag: str) -> str:
    tag = normalize_tag(tag)
    if tag.startswith("@"):
        tag = tag[1:]
    return tag.replace(" ", "_")


def correct_caption(
    text: str,
    kb: TagKnowledgeBase,
    *,
    options: CaptionCorrectionOptions | None = None,
) -> CaptionCorrectionResult:
    options = options or CaptionCorrectionOptions()
    tags = _parse_tags(text)
    if not tags:
        return CaptionCorrectionResult(
            text="",
            changed=bool(text.strip()),
            inserted_no_artist=False,
            unknown_tags=(),
        )

    buckets: dict[str, list[str]] = {
        "front": [],
        "meta": [],
        "count": [],
        "character": [],
        "copyright": [],
        "artist": [],
        "general": [],
    }
    unknown: list[str] = []
    seen: set[str] = set()
    trigger = normalize_tag(options.trigger_word) if options.trigger_word.strip() else ""
    trigger_key = tag_key(trigger) if trigger else ""

    for tag in tags:
        key = tag_key(tag)
        if trigger_key and key == trigger_key:
            continue
        if key in seen:
            continue
        seen.add(key)
        kind = _classify_tag(tag, kb, options)
        if kind is None:
            unknown.append(tag)
            kind = "general"
        buckets[kind].append(tag)

    if trigger:
        if options.trigger_at_front:
            buckets["front"].append(trigger)
        else:
            buckets["artist"].insert(0, trigger)

    inserted_no_artist = False
    if options.insert_no_artist and not buckets["artist"]:
        buckets["artist"].append("@no-artist")
        inserted_no_artist = True

    ordered = [
        *buckets["front"],
        *buckets["meta"],
        *buckets["count"],
        *buckets["character"],
        *buckets["copyright"],
        *buckets["artist"],
        *buckets["general"],
    ]
    corrected = ", ".join(ordered)
    return CaptionCorrectionResult(
        text=corrected,
        changed=corrected != text.strip(),
        inserted_no_artist=inserted_no_artist,
        unknown_tags=tuple(unknown),
    )


def _parse_tags(text: str) -> list[str]:
    raw = text.replace("\n", ",").split(",")
    return [normalize_tag(tag) for tag in raw if tag.strip()]


def _classify_tag(
    tag: str, kb: TagKnowledgeBase, options: CaptionCorrectionOptions
) -> str | None:
    if tag == "@no-artist":
        return "artist"
    if is_artist_tag(tag):
        if not options.validate_artist_tags or kb.has_artist(tag):
            return "artist"
        return "general"
    if tag in CAPTION_RATINGS or _is_year_tag(tag):
        return "meta"
    if is_count_tag(tag) or tag in _ANIMA_PERSON_TAGS:
        return "count"
    return kb.classify(tag)


def _description_category_path(description: str) -> str:
    match = _CATEGORY_RE.match(description)
    if not match:
        return ""
    return normalize_tag(match.group(1).replace(">", " > "))


def _kind_from_category(category_path: str, numeric_category: str) -> str:
    numeric_kind = _DANBOORU_NUMERIC_CATEGORIES.get(numeric_category.strip())
    if numeric_kind is not None:
        return numeric_kind
    # Localsmile's fallback category path is Korean; numeric Danbooru category is
    # preferred above so this only helps non-standard rows.
    if category_path:
        root = category_path.split(">")[0].strip()
        if root == "작가":
            return "artist"
        if root == "캐릭터":
            return "character"
        if root == "작품":
            return "copyright"
        if root == "메타":
            return "meta"
        if category_path.startswith("인물 > 인원수"):
            return "count"
    return "general"


def _is_year_tag(tag: str) -> bool:
    return len(tag) == 4 and tag.isdigit() and 1900 <= int(tag) <= 2100


def correct_many(
    captions: Iterable[tuple[Path, str]],
    kb: TagKnowledgeBase,
    *,
    options: CaptionCorrectionOptions | None = None,
) -> list[tuple[Path, CaptionCorrectionResult]]:
    return [(path, correct_caption(text, kb, options=options)) for path, text in captions]
