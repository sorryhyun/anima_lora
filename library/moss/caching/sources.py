"""What gets cached: image / caption pairs and the caption the TE encodes."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from library.moss.scan import (
    caption_sha,
    layout_sidecar,
    original_tree,
    source_pairs,
)


@dataclass
class Pair:
    stem: str
    image: Path
    caption: str  # what the TE encodes
    source_sha: str  # of the .txt it came from
    plain: str = ""  # the .txt itself, encoded beside a layout-JSON caption
    variants: tuple[str, ...] = ()  # encoded as embeds_v1… (caption_variants)


def read_variants(caption_txt: Path) -> list[str]:
    """The ``v1``, ``v2``, … lines of the Anima ``{stem}.variants.txt`` beside a
    caption (``v0`` is the caption itself), or ``[]`` without one."""
    path = caption_txt.with_name(caption_txt.name[: -len(".txt")] + ".variants.txt")
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        label, sep, text = line.partition("\t")
        if sep and not line.startswith("#") and label != "v0" and text.strip():
            out.append(text.strip())
    return out


def find_pairs(
    src: Path, caption_format: str = "text", variants: bool = False
) -> list[Pair]:
    """Images with a ``.txt`` caption beside them, subfolders included. Stems that
    collide across subfolders get their relative folder as a prefix.

    Under ``layout_json`` a ``{stem}.layout.txt`` that anime_tools' Export
    published beside the caption is used as is, unless the ``.txt`` is newer
    (edited after the export); every other caption is converted here. Either
    way the ``.txt`` rides along as ``plain``, unless it already is layout JSON
    (which has no plain form). With ``variants`` each pair carries its
    ``read_variants`` lines, mapped the same way."""
    originals, skip = original_tree(src)
    images, found = source_pairs(src, originals, skip)
    if originals is not None:
        n = sum(1 for _, image, _ in found if not image.is_relative_to(src))
        if n:
            print(
                f"{n} captions without a resized image: "
                f"using the originals under {originals}",
                flush=True,
            )
    captioned = {image for _, image, _ in found}
    for p in images:
        if p not in captioned:
            print(f"skip {p.name}: no caption sidecar", flush=True)
    convert = None
    pairs = []
    from_sidecar = 0
    for stem, image, path in found:
        raw = path.read_text(encoding="utf-8").strip()
        caption = None
        if caption_format == "layout_json":
            sidecar = layout_sidecar(path)
            if sidecar.exists() and sidecar.stat().st_mtime >= path.stat().st_mtime:
                caption = drop_rating(sidecar.read_text(encoding="utf-8").strip())
                from_sidecar += 1
        if caption is None:
            convert = convert or caption_converter(caption_format)
            caption = convert(raw)
        unrated = drop_rating(raw)
        plain = (
            unrated if caption_format == "layout_json" and caption != unrated else ""
        )
        var = ()
        if variants:
            convert = convert or caption_converter(caption_format)
            var = tuple(convert(v) for v in read_variants(path))
        pairs.append(Pair(stem, image, caption, caption_sha(raw), plain, var))
    if variants:
        n = sum(1 for p in pairs if p.variants)
        print(f"caption variants: {n}/{len(pairs)} captions have them", flush=True)
    if caption_format == "layout_json":
        print(
            f"layout_json: {from_sidecar} from Export's sidecars, "
            f"{len(pairs) - from_sidecar} converted",
            flush=True,
        )
    return pairs


def manifest_pairs(
    path: Path, caption_format: str = "text"
) -> tuple[list[Pair], dict[str, Path]]:
    """The targets of an i2i pair list as ``Pair``s, and each stem's condition
    image. ``path`` is a ``pairs.jsonl`` (``i2i_tasks`` staging), one
    ``{stem, target, cond, caption}`` object per line, paths relative to it."""
    root = path.parent
    pairs, refs = [], {}
    convert = caption_converter(caption_format) if caption_format != "text" else None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        stem = row["stem"]
        if stem in refs:
            raise SystemExit(f"{path}: stem {stem!r} appears twice")
        raw = (root / row["caption"]).read_text(encoding="utf-8").strip()
        unrated = drop_rating(raw)
        caption = convert(raw) if convert else unrated
        plain = (
            unrated if caption_format == "layout_json" and caption != unrated else ""
        )
        pairs.append(Pair(stem, root / row["target"], caption, caption_sha(raw), plain))
        refs[stem] = root / row["cond"]
    return pairs, refs


def is_layout_json(raw: str) -> bool:
    """``raw`` is already a layout-JSON prompt (an object with ``summary``). The MOSS
    Toolkit's ``moss_layout.py`` carries a copy — change both together."""
    if not raw.startswith("{"):
        return False
    try:
        obj = json.loads(raw)
    except ValueError:
        return False
    return isinstance(obj, dict) and "summary" in obj


def caption_converter(caption_format: str):
    """``raw caption -> what the TE encodes`` for one ``caption_format``.

    ``layout_json`` passes a caption that already is layout JSON through as
    written (converting it again would nest the whole object in
    ``summary.caption``), and loads the tag CSV only for the first caption that
    needs converting. ``tags`` keeps the tag form in the reference tag order
    (``order_tags``). Every format drops the rating (``drop_rating``)."""
    if caption_format == "text":
        return drop_rating
    if caption_format not in ("layout_json", "tags"):
        raise SystemExit(f"unknown caption_format {caption_format!r}")
    kb = None

    def convert(raw: str) -> str:
        nonlocal kb
        raw = drop_rating(raw)
        if is_layout_json(raw):
            return raw
        if caption_format == "tags":
            if kb is None:
                kb = tag_knowledge_base()
            return order_tags(raw, kb)
        from anime_tools.captions.layout_json import layout_json

        if kb is None:
            kb = tag_knowledge_base()
        return layout_json(raw, kb)

    return convert


# layout_json's binary rating in ``summary.meta``.
_LAYOUT_RATINGS = frozenset({"sfw", "nsfw"})


def drop_rating(raw: str) -> str:
    """``raw`` without its rating: MOSS captions carry none (the reference prompts
    have no rating field; decided 2026-11-14). A tag caption loses the flat tags
    ``canonical_rating`` recognises (safe / general / sensitive / nsfw /
    questionable / explicit), position and text clauses untouched; a layout-JSON
    prompt loses ``sfw`` / ``nsfw`` from ``summary.meta``, and an emptied ``meta``
    goes. Anything else, prose included, comes back as is."""
    if is_layout_json(raw):
        obj = json.loads(raw)
        summary = obj["summary"]
        meta = summary.get("meta") if isinstance(summary, dict) else None
        if not isinstance(meta, list) or not _LAYOUT_RATINGS.intersection(meta):
            return raw
        kept = [m for m in meta if m not in _LAYOUT_RATINGS]
        if kept:
            summary["meta"] = kept
        else:
            del summary["meta"]
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    from anime_tools.captions.position_clauses import compose_caption, parse_caption
    from anime_tools.captions.taxonomy import canonical_rating

    parsed = parse_caption(raw)
    rated = [t for t in dict.fromkeys(parsed.flat_tags) if canonical_rating(t)]
    if not rated:
        return raw
    # Cut the tags out of the flat head only, so a quoted text clause keeps its words.
    end = 0
    for tag in parsed.flat_tags:
        end = raw.index(tag, end) + len(tag)
    head, tail = raw[:end], raw[end:]
    alt = "|".join(re.escape(t) for t in rated)
    lead = re.compile(rf"^(?:{alt})(?:, |$)")
    while (cut := lead.sub("", head, count=1)) != head:
        head = cut
    head = re.sub(rf", (?:{alt})(?=,|$)", "", head)
    out = (head + tail).lstrip(". ") if not head else head + tail
    kept = tuple(t for t in parsed.flat_tags if t not in rated)
    again = parse_caption(out)
    if again.flat_tags != kept or again.clauses != parsed.clauses:
        return compose_caption(kept, parsed.clauses)
    return out


def _flat_head(raw: str, flat_tags) -> tuple[str, str]:
    """``raw`` split after its last flat tag: the flat head and the clause tail."""
    end = 0
    for tag in flat_tags:
        end = raw.index(tag, end) + len(tag)
    return raw[:end], raw[end:]


# Slot of each ``tag_kind`` in the USERGUIDE tag order: Quality -> Score -> Meta ->
# Year -> Character -> Series -> Artist -> General, the count tag opening General.
_ORDER_SLOTS = {
    "quality": 0,
    "score": 1,
    "meta": 2,
    "year": 3,
    "safety": 3,
    "character": 4,
    "copyright": 5,
    "artist": 6,
    "count": 7,
    "general": 8,
}


def order_tags(raw: str, kb) -> str:
    """A tag caption's flat head in the USERGUIDE order (Quality, Score, Meta, Year,
    Character, Series, Artist, then General with the count tag first), each slot in
    its original order. Danbooru meta tags that describe the picture (``cropped``,
    ``lowres``, …, layout_json's ``_VISUAL_META``) sort as General, as layout_json
    puts them in ``summary.caption``. Position and text clauses are kept verbatim."""
    from anime_tools.captions.correction import tag_kind
    from anime_tools.captions.layout_json import _VISUAL_META
    from anime_tools.captions.position_clauses import compose_caption, parse_caption
    from anime_tools.captions.taxonomy import normalize_tag

    parsed = parse_caption(raw)
    if not parsed.flat_tags:
        return raw

    def slot(tag: str) -> int:
        kind = tag_kind(tag, kb)
        norm = normalize_tag(tag)
        if kind == "quality" and norm.startswith("score "):
            kind = "score"
        elif kind == "meta" and norm in _VISUAL_META:
            kind = "general"
        return _ORDER_SLOTS.get(kind, _ORDER_SLOTS["general"])

    ordered = tuple(sorted(parsed.flat_tags, key=slot))
    if ordered == parsed.flat_tags:
        return raw
    _, tail = _flat_head(raw, parsed.flat_tags)
    out = ", ".join(ordered) + tail
    again = parse_caption(out)
    if again.flat_tags != ordered or again.clauses != parsed.clauses:
        return compose_caption(ordered, parsed.clauses)
    return out


def tag_knowledge_base():
    from anime_tools.captions.correction import find_tag_csv, load_tag_knowledge_base

    from library.env import anima_home

    csv = find_tag_csv(anima_home())
    if csv is None:
        raise SystemExit(
            "layout_json needs the danbooru tag CSV: make download-model ARGS=danbooru_tags"
        )
    return load_tag_knowledge_base(csv)
