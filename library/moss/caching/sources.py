"""What gets cached: image / caption pairs and the caption the TE encodes."""

from __future__ import annotations

import json
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


def find_pairs(src: Path, caption_format: str = "text") -> list[Pair]:
    """Images with a ``.txt`` caption beside them, subfolders included. Stems that
    collide across subfolders get their relative folder as a prefix.

    Under ``layout_json`` a ``{stem}.layout.txt`` that anime_tools' Export
    published beside the caption is used as is, unless the ``.txt`` is newer
    (edited after the export); every other caption is converted here. Either
    way the ``.txt`` rides along as ``plain``, unless it already is layout JSON
    (which has no plain form)."""
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
                caption = sidecar.read_text(encoding="utf-8").strip()
                from_sidecar += 1
        if caption is None:
            convert = convert or caption_converter(caption_format)
            caption = convert(raw)
        plain = raw if caption_format == "layout_json" and caption != raw else ""
        pairs.append(Pair(stem, image, caption, caption_sha(raw), plain))
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
        caption = convert(raw) if convert else raw
        plain = raw if caption_format == "layout_json" and caption != raw else ""
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
    needs converting."""
    if caption_format == "text":
        return lambda raw: raw
    if caption_format != "layout_json":
        raise SystemExit(f"unknown caption_format {caption_format!r}")
    kb = None

    def convert(raw: str) -> str:
        nonlocal kb
        if is_layout_json(raw):
            return raw
        from anime_tools.captions.layout_json import layout_json

        if kb is None:
            kb = tag_knowledge_base()
        return layout_json(raw, kb)

    return convert


def tag_knowledge_base():
    from anime_tools.captions.correction import find_tag_csv, load_tag_knowledge_base

    from library.env import anima_home

    csv = find_tag_csv(anima_home())
    if csv is None:
        raise SystemExit(
            "layout_json needs the danbooru tag CSV: make download-model ARGS=danbooru_tags"
        )
    return load_tag_knowledge_base(csv)
