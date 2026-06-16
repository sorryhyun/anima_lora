#!/usr/bin/env python3
"""Build a method-agnostic typed-tag index from caption sidecars.

Walks caption ``.txt`` sidecars under a source dir, classifies each
comma-separated tag into character / copyright / artist / count via the Anima
Tagger vocab (artist additionally by the ``@`` prefix, which is exact and not
limited by the vocab's frequency cutoff), and writes a single JSON index to
``post_image_dataset/captions/caption_index.json``::

    {
      "meta":  {... provenance: vocab path+mtime, src, n_images, generated ...},
      "image_meta": {
        "<stem>": {"path": "<rel>", "character": [...], "copyright": [...],
                   "artist": [...], "count": [...]},
        ...
      },
      "groups": {
        "character": {"<tag>": ["<stem>", ...], ...},
        "copyright": {...},
        "artist":    {...}
      }
    }

This is a *dataset artifact* — it lives beside the VAE/PE caches under
``post_image_dataset/`` (not in any checkpoint) and is regenerated when the
dataset or vocab changes. It encodes **no sampling policy**: how a method backs
off across the character → copyright → artist tiers is the method's own concern
(e.g. the IP-Adapter distinct-pair sampler). Consumers: the IP-Adapter
identity-pair sampler, artist balancing, dataset analytics.
"""

import argparse
import json
import os
import re
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

# Shared tag-shape primitives (torch-free) — single source of truth for the
# artist ``@``-prefix rule, count-tag detection, and the raw-caption rating set,
# kept in sync with the Anima Tagger vocab build (scripts/anima_tagger/vocab.py).
from library.captioning.taxonomy import CAPTION_RATINGS, is_artist_tag, is_count_tag
from library.datasets.image_utils import IMAGE_EXTENSIONS
from library.datasets.subsets import filter_paths_by_glob
from library.io.walk import safe_walk


DEFAULT_VOCAB = "models/captioners/anima-tagger-v2/vocab.json"
DEFAULT_OUT = "post_image_dataset/captions/caption_index.json"
# Axes we group on. Artist is detected by the `@` prefix (superset of the
# vocab artist list); character/copyright/count are classified by vocab
# membership, matching the Anima Tagger's category labels.
VOCAB_AXES = ("character", "copyright", "count")

# Danbooru disambiguator form: ``character_name (copyright_name)``. The tagger
# vocab only carries the ~135 character names frozen at its training cutoff, so
# newer characters (``endministrator (arknights)``, ``mualani (genshin
# impact)``, …) are emitted in captions but miss the exact-membership classifier
# and the image reads as character-less. ``_recover_paren_character`` rescues
# them: a ``name (series)`` tag is a character when the parenthetical series is a
# real franchise — a known vocab copyright, or present as a standalone tag in
# the *same* caption (danbooru always co-tags the bare franchise) — and not one
# of these generic disambiguators (``X (cosplay)`` does not make X a character).
_PAREN_RE = re.compile(r"^(.+?)\s*\(([^)]+)\)$")
_GENERIC_PAREN_QUALIFIERS = frozenset(
    {"cosplay", "costume", "alternate costume", "meme", "food", "fruit", "maid", "animal", "object"}
)

# Positional recovery: many characters are tagged as a *bare name* with no
# `(series)` disambiguator (`nakiri ayame`, `yuigahama yui`, hololive/idolmaster
# members), so the paren pass can't see them either. Danbooru caption order is
# rigid — ``[rating] [count] [character…] [copyright…] @artist [general…]`` —
# so the character band is the run of pre-`@artist` tags that are not rating /
# count / copyright. The risk is *franchise sub-titles* (``pokemon scarlet and
# violet``, ``gakuen idolmaster``, ``arknights: endfield``) that sit in the same
# pre-artist span but are copyright, not character: they are excluded because
# they share a (≥4-char, non-generic) word with a known copyright in the same
# caption. The shared ``CAPTION_RATINGS`` / ``is_count_tag`` (from
# ``library.captioning.taxonomy``) strip the leading rating/count band;
# ``_COPYRIGHT_STOPWORDS`` are generic franchise-title words too weak to anchor
# the sub-title test. Residual false positives (event/brand meta tags like
# ``comiket 104``) are rare (<0.3% of images) and left as noise.
_COPYRIGHT_STOPWORDS = frozenset(
    {
        "club", "high", "school", "idol", "story", "world", "project", "series",
        "love", "live", "girl", "girls", "boy", "boys", "the", "and", "no",
    }
)


def _norm_words(tag: str) -> set[str]:
    """≥4-char alphanumeric words of a tag, minus generic franchise-title
    stopwords — the unit the positional pass uses to test whether a pre-artist
    tag is a franchise sub-title of a known copyright."""
    return {
        w
        for w in re.split(r"[^a-z0-9]+", tag)
        if len(w) >= 4 and w not in _COPYRIGHT_STOPWORDS
    }


def _load_vocab_sets(vocab_path: str) -> dict[str, set[str]]:
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    sets: dict[str, set[str]] = {axis: set() for axis in VOCAB_AXES}
    for entry in vocab["tags"]:
        cat = entry.get("category")
        if cat in sets:
            sets[cat].add(entry["name"].strip().lower())
    return sets


def _iter_captions(src: Path, path_pattern: str | None = None):
    """Yield ``(stem, rel_path, text)`` for every ``.txt`` under ``src``.

    ``image_dataset`` is a symlink to a tree of (possibly symlinked) artist
    dirs, so resolve the root and walk with ``followlinks=True`` — a plain walk
    descends into neither. Stems are assumed unique across the tree (the same
    invariant the stem-keyed VAE/PE caches rely on)."""
    root = Path(os.path.realpath(src))
    for dirpath, _dirnames, filenames in safe_walk(root, followlinks=True):
        for name in filenames:
            if not name.endswith(".txt"):
                continue
            abs_path = Path(dirpath) / name
            if path_pattern and path_pattern != "*":
                keep = filter_paths_by_glob(
                    [str(abs_path)],
                    str(root),
                    path_pattern,
                )[0]
                if not keep:
                    image_sidecars = [
                        str(abs_path.with_suffix(ext)) for ext in IMAGE_EXTENSIONS
                    ]
                    keep = any(
                        filter_paths_by_glob(image_sidecars, str(root), path_pattern)
                    )
                if not keep:
                    continue
            try:
                text = abs_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            rel = os.path.relpath(abs_path, root)
            yield name[:-4], rel, text


def _classify(
    text: str,
    vsets: dict[str, set[str]],
    *,
    recover_paren: bool = True,
    recover_positional: bool = True,
) -> dict[str, list[str]]:
    tags = [t.strip().lower() for t in text.split(",")]
    tags = [t for t in tags if t]
    bare = set(tags)
    out: dict[str, list[str]] = {axis: [] for axis in (*VOCAB_AXES, "artist")}
    seen = {axis: set() for axis in out}

    def _add(axis: str, tag: str):
        if tag not in seen[axis]:
            seen[axis].add(tag)
            out[axis].append(tag)

    for tag in tags:
        if is_artist_tag(tag):
            _add("artist", tag)
            continue
        matched = False
        for axis in VOCAB_AXES:
            if tag in vsets[axis]:
                _add(axis, tag)
                matched = True
        if matched or not recover_paren:
            continue
        # Vocab missed it — try the danbooru `name (series)` character recovery.
        m = _PAREN_RE.match(tag)
        if m:
            series = m.group(2).strip()
            if series not in _GENERIC_PAREN_QUALIFIERS and (
                series in vsets["copyright"] or series in bare
            ):
                _add("character", tag)
                _add("copyright", series)

    # Positional recovery of bare-name characters (see _RATINGS/_norm_words note):
    # everything in the pre-`@artist` band that isn't rating / count / copyright
    # and isn't a franchise sub-title (shares a word with a known copyright here).
    if recover_positional:
        artist_at = next(
            (i for i, t in enumerate(tags) if is_artist_tag(t)), None
        )
        if artist_at is not None:
            copy_words: set[str] = set()
            for cp in out["copyright"]:
                copy_words |= _norm_words(cp)
            for tag in tags[:artist_at]:
                if (
                    tag in seen["character"]
                    or tag in seen["copyright"]
                    or tag in seen["count"]
                ):
                    continue
                if tag in CAPTION_RATINGS or is_count_tag(tag) or _PAREN_RE.match(tag):
                    continue
                if _norm_words(tag) & copy_words:
                    continue  # franchise sub-title of a known copyright
                _add("character", tag)

    # Danbooru `original` copyright = original character — no *named* franchise
    # character. When it is the SOLE copyright, drop any character tags (vocab
    # OCs, `oc_name (artist)` circle tags like `ichigo (mignon)`) so OC images
    # read as character-less, which also routes them to the contrastive
    # `hard_original` negative tier. Crossover images (`original` + a real
    # franchise, e.g. `liko (pokemon)`) keep their characters.
    if set(out["copyright"]) == {"original"}:
        out["character"] = []
        seen["character"] = set()
    return out


def build_index(
    src: str,
    vocab_path: str,
    *,
    recover_paren: bool = True,
    recover_positional: bool = True,
    path_pattern: str | None = None,
) -> dict:
    vsets = _load_vocab_sets(vocab_path)
    image_meta: dict[str, dict] = OrderedDict()
    groups: dict[str, dict[str, list[str]]] = {
        axis: {} for axis in ("character", "copyright", "artist")
    }

    n_seen = 0
    for stem, rel, text in sorted(_iter_captions(Path(src), path_pattern)):
        n_seen += 1
        typed = _classify(
            text,
            vsets,
            recover_paren=recover_paren,
            recover_positional=recover_positional,
        )
        if stem in image_meta:
            # Stems must be unique across the tree; surface a collision rather
            # than silently dropping one image's tags.
            raise SystemExit(
                f"duplicate caption stem {stem!r} (paths: "
                f"{image_meta[stem]['path']} vs {rel}); stems must be unique"
            )
        image_meta[stem] = {
            "path": rel,
            "character": typed["character"],
            "copyright": typed["copyright"],
            "artist": typed["artist"],
            "count": typed["count"],
        }
        for axis in ("character", "copyright", "artist"):
            for tag in typed[axis]:
                groups[axis].setdefault(tag, []).append(stem)

    # Stable ordering: stems sorted within each group, groups sorted by key.
    for axis in groups:
        groups[axis] = {
            tag: sorted(stems) for tag, stems in sorted(groups[axis].items())
        }

    vstat = os.stat(vocab_path)
    return {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "src": str(src),
            "vocab_path": vocab_path,
            "vocab_mtime": datetime.fromtimestamp(
                vstat.st_mtime, timezone.utc
            ).isoformat(timespec="seconds"),
            "n_images": n_seen,
            "axes": ["character", "copyright", "artist", "count"],
            "paren_recover": recover_paren,
            "positional_recover": recover_positional,
            "note": "method-agnostic typed-tag parse; sampling policy lives in method config",
        },
        "image_meta": image_meta,
        "groups": groups,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--src",
        default="image_dataset",
        help="Caption sidecar root (default: image_dataset)",
    )
    ap.add_argument(
        "--vocab",
        default=DEFAULT_VOCAB,
        help=f"Tagger vocab (default: {DEFAULT_VOCAB})",
    )
    ap.add_argument(
        "--out", default=DEFAULT_OUT, help=f"Output JSON (default: {DEFAULT_OUT})"
    )
    ap.add_argument(
        "--path_pattern",
        "--path-pattern",
        dest="path_pattern",
        default="*",
        help=(
            "Only index captions whose path relative to --src matches this "
            "fnmatch glob. Use | to separate alternatives. Default: *"
        ),
    )
    ap.add_argument(
        "--no-paren-recover",
        action="store_true",
        help="Disable the danbooru `name (series)` character-recovery heuristic "
        "(exact vocab membership only).",
    )
    ap.add_argument(
        "--no-positional-recover",
        action="store_true",
        help="Disable the positional bare-name character recovery (pre-`@artist` "
        "band minus rating/count/copyright/franchise-sub-titles).",
    )
    args = ap.parse_args()

    index = build_index(
        args.src,
        args.vocab,
        recover_paren=not args.no_paren_recover,
        recover_positional=not args.no_positional_recover,
        path_pattern=args.path_pattern,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)

    m = index["meta"]
    cov = {
        axis: sum(1 for v in index["image_meta"].values() if v[axis])
        for axis in ("character", "copyright", "artist")
    }
    n = m["n_images"] or 1
    print(f"caption index → {out}")
    print(f"  images: {m['n_images']}")
    for axis in ("character", "copyright", "artist"):
        print(
            f"  {axis:9s}: {cov[axis]:5d} imgs ({100 * cov[axis] / n:4.1f}%), "
            f"{len(index['groups'][axis]):4d} groups"
        )


if __name__ == "__main__":
    main()
