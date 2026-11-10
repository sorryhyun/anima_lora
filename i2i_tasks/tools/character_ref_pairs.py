"""Same-artist character-reference pair miner (MOSS i2i ``character_ref``).

A pair is two images of one character by one artist from the raw crawl pool
(``$CAPTION_CORPUS_DIR/retrieved``): the condition is the reference, the target
is what the model draws. The target's caption loses its character, copyright,
artist and meta (``highres``, ``absurdres``, ``translated``…) tags and every tag
the reference's caption also carries, so it says only what the target changes. Crawl captions carry HTML entities
(``shin&#039;ei``, ``:&lt;``); they are unescaped first.

Tag types (:class:`TagTypes`): the danbooru KB, then gelbooru's type from the
pool's ``.tag_cache.json`` (1 artist, 3 copyright, 4 character, 5 meta), then
the crawl caption's slot order (``meta…, rating, count, names…, @artist,
general…``): a tag neither source knows, between the last count tag and the
artist, is a name and counts as a character.

Members: a rating tag in ``ratings`` (empty: any), a cached phash (``.hash_cache.json``),
one character tag and none of ``$CHARACTER_REF_EXCLUDE`` (comma-separated, from
``.env``; unset stops the run). A group is ``(artist dir, character)``. Its references
are the solo members (exactly one person: ``1girl`` or ``1boy``); its targets
are those plus the members where an unnamed ``1boy`` shares the picture
(``1girl, 1boy``). A target with a person its reference lacks gets ``add 1boy``
after its count tags. Each group gives up to ``max_per_group`` targets (0: all);
each target's reference is a random reference other than itself whose phash is
more than ``dedupe_phash`` bits away (re-uploads, crops, recolours and text
overlays of one picture mostly sit at ≤ 60 of 256). ``n_pairs`` targets are
drawn from that pool. ``holdout`` characters are left out of every pair and
written to ``holdout.jsonl`` (one pair per character) for the eval.

Output, under ``post_image_dataset/i2i/<name>/``::

    staging/{artist}__{id}.tgt.<ext>   symlink to the target
    staging/{artist}__{id}.ref.<ext>   symlink to the condition
    staging/{artist}__{id}.txt         the stripped target caption
    staging/pairs.jsonl                {stem, target, cond, caption}, MOSS's manifest
    staging/holdout.jsonl              the same rows for the held-out characters
"""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from i2i_tasks.tools.phash_edit_pairs import (
    Member,
    _default_image_dirs,
    apply_staging_config,
    expand_path,
    gather,
    load_dotenv,
)
from library.env import resolve_under_home

DEFAULT_CONFIG = "i2i_tasks/configs/character_ref.toml"
PEOPLE = ({"1girl"}, {"1boy"})
TARGET_PEOPLE = (*PEOPLE, {"1girl", "1boy"})
ADDABLE = {"1boy"}
DROP = {"character", "copyright", "artist", "meta"}
GELBOORU = {1: "artist", 3: "copyright", 4: "character", 5: "meta"}
TAG_CACHE = ".tag_cache.json"


def exclude_tags() -> frozenset[str]:
    """``$CHARACTER_REF_EXCLUDE`` as a tag set; unset or empty stops the run."""
    tags = frozenset(
        t.strip()
        for t in os.environ.get("CHARACTER_REF_EXCLUDE", "").split(",")
        if t.strip()
    )
    if not tags:
        raise SystemExit("[character_ref] CHARACTER_REF_EXCLUDE is not set (.env)")
    return tags


class TagTypes:
    def __init__(self, kb, image_dirs: list[Path]):
        self.kb = kb
        self.gel: dict[str, int] = {}
        for root in image_dirs:
            path = root / TAG_CACHE
            if path.is_file():
                raw = json.loads(path.read_text(encoding="utf-8"))
                self.gel.update(
                    {html.unescape(k).replace("_", " "): v for k, v in raw.items()}
                )

    def kind(self, tag: str) -> str:
        from anime_tools.captions.correction import tag_kind

        kind = tag_kind(tag, self.kb)
        if kind != "general":
            return kind
        return GELBOORU.get(self.gel.get(tag), kind)

    def kinds(self, tags: list[str]) -> list[str]:
        """Per tag; unknown tags in the names slot become ``character``."""
        kinds = [self.kind(t) for t in tags]
        artist = next((i for i, k in enumerate(kinds) if k == "artist"), None)
        if artist is not None:
            counts = [i for i in range(artist) if kinds[i] == "count"]
            for i in range(counts[-1] + 1 if counts else 0, artist):
                if kinds[i] == "general" and self.kb.describe(tags[i]) is None:
                    kinds[i] = "character"
        return kinds


def _bits(m: Member) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bytes.fromhex(m.phash), dtype=np.uint8))


def caption_tags(raw: str) -> set[str]:
    """Every tag of ``raw``, the flat bag and the position clauses'."""
    from anime_tools.captions.position_clauses import parse_caption

    parsed = parse_caption(html.unescape(raw).strip())
    clauses = (t for c in parsed.clauses if not c.is_text for t in c.tags)
    return {*parsed.flat_tags, *clauses}


def strip_caption(
    raw: str, types: TagTypes, add: list[str] = (), shared: set[str] = frozenset()
) -> str:
    """``raw`` without its DROP kinds and the ``shared`` tags; ``add <tag>`` for
    each of ``add`` after the count tags."""
    from anime_tools.captions.position_clauses import compose_caption, parse_caption

    parsed = parse_caption(html.unescape(raw).strip())
    flat = list(parsed.flat_tags)
    kinds = types.kinds(flat)
    kept = [t for t, k in zip(flat, kinds) if k not in DROP]
    if add:
        counts = [i for i, t in enumerate(kept) if types.kind(t) == "count"]
        at = counts[-1] + 1 if counts else 0
        kept[at:at] = [f"add {t}" for t in add]
    kept = [t for t in kept if t not in shared]
    clauses = []
    for c in parsed.clauses:
        if c.is_text:
            clauses.append(c)
            continue
        tags = tuple(t for t in c.tags if types.kind(t) not in DROP and t not in shared)
        if tags:
            clauses.append(replace(c, tags=tags))
    return compose_caption(kept, clauses)


@dataclass
class Group:
    targets: list[Member] = field(default_factory=list)
    refs: list[Member] = field(default_factory=list)


def people_of(m: Member, types: TagTypes) -> frozenset[str]:
    kinds = types.kinds(m.tags)
    return frozenset(t for t, k in zip(m.tags, kinds) if k == "count") - {
        "solo",
        "solo focus",
    }


def group_members(
    members: list[Member], types: TagTypes, ratings: set[str], exclude: frozenset[str]
) -> tuple[dict[tuple[str, str], Group], dict[str, frozenset[str]]]:
    """``({(artist, character): Group}, {member key: its count tags})``, keeping
    groups with a reference and a second member."""
    groups: dict[tuple[str, str], Group] = defaultdict(Group)
    people: dict[str, frozenset[str]] = {}
    for m in members:
        tags = set(m.tags)
        if (ratings and not ratings & tags) or exclude & tags:
            continue
        kinds = types.kinds(m.tags)
        chars = [t for t, k in zip(m.tags, kinds) if k == "character"]
        if len(chars) != 1:
            continue
        people[m.key] = p = people_of(m, types)
        group = groups[(m.artist, chars[0])]
        if p in TARGET_PEOPLE:
            group.targets.append(m)
        if p in PEOPLE:
            group.refs.append(m)
    keep = {
        k: g
        for k, g in groups.items()
        if g.refs and len({m.key for m in g.targets + g.refs}) >= 2
    }
    return keep, people


def extra_people(target: frozenset[str], ref: frozenset[str]) -> list[str] | None:
    """Count tags the target adds to its reference, or ``None`` when the reference
    is not in the target or the target adds anything outside ``ADDABLE``."""
    if not ref <= target or not target - ref <= ADDABLE:
        return None
    return sorted(target - ref)


def pick_cond(
    target: Member, refs: list[Member], people: dict, dedupe: int, rng
) -> Member | None:
    tb = _bits(target)
    far = [
        m
        for m in refs
        if m.key != target.key
        and extra_people(people[target.key], people[m.key]) is not None
        and int((_bits(m) != tb).sum()) > dedupe
    ]
    return rng.choice(far) if far else None


def mine(args, types: TagTypes, image_dirs: list[Path]):
    groups, people = group_members(
        gather(image_dirs),
        types,
        {r for r in args.ratings.split(",") if r},
        exclude_tags(),
    )
    rng = random.Random(args.seed)

    def pair(target, group):
        cond = pick_cond(target, group.refs, people, args.dedupe_phash, rng)
        if cond is None:
            return None
        return target, cond, extra_people(people[target.key], people[cond.key])

    characters = sorted({c for _, c in groups})
    held = set(rng.sample(characters, min(args.holdout, len(characters))))
    holdout = []
    for c in sorted(held):
        group = groups[rng.choice(sorted(k for k in groups if k[1] == c))]
        for target in rng.sample(group.targets, len(group.targets)):
            if row := pair(target, group):
                holdout.append(row)
                break

    pool = []
    for key in sorted(groups):
        if key[1] in held:
            continue
        group = groups[key]
        cap = args.max_per_group or len(group.targets)
        for target in rng.sample(group.targets, min(cap, len(group.targets))):
            if row := pair(target, group):
                pool.append(row)
    rng.shuffle(pool)
    pairs = pool[: args.n_pairs] if args.n_pairs else pool
    added = sum(bool(add) for _, _, add in pairs)
    print(
        f"[character_ref] {len(groups)} groups, {len(characters)} characters; "
        f"pool {len(pool)} pairs, staging {len(pairs)} ({added} with an added "
        f"person), holdout {len(holdout)}"
    )
    return pairs, holdout


def write_rows(pairs, staging: Path, types: TagTypes) -> list[dict]:
    rows = []
    for target, cond, add in pairs:
        stem = f"{target.artist}__{target.stem}"
        tgt = staging / f"{stem}.tgt{target.image.suffix}"
        ref = staging / f"{stem}.ref{cond.image.suffix}"
        tgt.symlink_to(target.image.resolve())
        ref.symlink_to(cond.image.resolve())
        raw = target.image.with_suffix(".txt").read_text(encoding="utf-8")
        shared = caption_tags(
            cond.image.with_suffix(".txt").read_text(encoding="utf-8")
        )
        (staging / f"{stem}.txt").write_text(
            strip_caption(raw, types, add, shared), encoding="utf-8"
        )
        rows.append(
            {
                "stem": stem,
                "target": tgt.name,
                "cond": ref.name,
                "caption": f"{stem}.txt",
                "source_target": target.key,
                "source_cond": cond.key,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    load_dotenv(resolve_under_home(".env"))
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--name", default="character_ref")
    p.add_argument("--image_dirs", default=_default_image_dirs())
    p.add_argument("--n_pairs", type=int, default=1000, help="0 = the whole pool")
    p.add_argument("--max_per_group", type=int, default=2, help="0 = every member")
    p.add_argument(
        "--ratings",
        default="general,sensitive",
        help="comma-separated rating tags a member must carry",
    )
    p.add_argument(
        "--dedupe_phash",
        type=int,
        default=60,
        help="a condition must be more than this many of 256 phash bits from its target",
    )
    p.add_argument("--holdout", type=int, default=8, help="characters held out")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    args.config = str(resolve_under_home(args.config))
    apply_staging_config(args, argv)

    from library.moss.caching.sources import tag_knowledge_base

    image_dirs = [Path(expand_path(d)) for d in args.image_dirs.split(",") if d]
    types = TagTypes(tag_knowledge_base(), image_dirs)
    pairs, holdout = mine(args, types, image_dirs)
    if not pairs:
        raise SystemExit("[character_ref] 0 pairs")
    base = resolve_under_home(f"post_image_dataset/i2i/{args.name}")
    staging = base / "staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for name, chosen in (("pairs.jsonl", pairs), ("holdout.jsonl", holdout)):
        rows = write_rows(chosen, staging, types)
        (staging / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8",
        )
    print(f"[character_ref] → {staging}")


if __name__ == "__main__":
    main()
