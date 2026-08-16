"""Danbooru EN→JA tag pairs (p1atdev, CC0) — a *fill* source for the glossary tail.

``tag_glossary.py`` resolves a tag from the Danbooru wiki, the Wikidata lexicon
and an MT pass whose wording is chosen by back-translation. Whatever it cannot
resolve composes as **latin passthrough** (``via: unmapped``, span weight 0):
the tag contributes no ext-row supervision at all. On the widened D1 corpus that
is 7,239 of 14,753 tags — 3.1% of occurrences.

``p1atdev/danbooru-ja-tag-pair-20241015`` is the danbooru wiki's own
``other_names`` (an Oct-2024 snapshot) with ``calm3-22b-chat`` filling the tags
that had no Japanese name, 151,431 rows, CC0. Measured against our corpus it
carries a Japanese name for **74.8%** of that unresolved tail (71.3% of its
occurrence mass); the residue is almost entirely artist handles, which are latin
by design.

**This module only fills.** It never re-opens a tag the glossary already
resolved, because every one of those is pinned (override / rating / wikidata /
passthrough) or chosen by back-translation, and a CPU pass cannot re-verify what
it overwrites — the failure `datasets/README.md` records. Filling an *absent*
tag regresses nothing: the alternative is passthrough at weight 0.

The dataset is unfiltered by its own admission, so the fill re-applies our
Chinese guards rather than trusting the field: ``is_japanese`` (kana, or Han
that Shift-JIS can encode) **and**, for a Han-only name, every kanji must appear
in the JA inventory learned from kana-bearing wiki entries — Shift-JIS accepts
traditional Chinese, the inventory does not. Names carrying latin letters are
dropped (``ウィンクX東方``).

Resolving *which* of several Japanese names is the best wording — the
katakana-loanword-vs-native-kanji question — is arbitration, needs the
back-translation GPU pass, and is a separate work item (see `plan.md`, D1-pairs).
Here the ranking rule is the existing one: kana first, as evidence of
Japaneseness.

    python tag_pairs.py --dry-run      # what would be filled
    python tag_pairs.py                # fill in place
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import build_pairs  # noqa: E402  (sibling — caption roots + loader)
import tag_glossary  # noqa: E402  (sibling — script guards, tag counts)

PAIR_REPO = "p1atdev/danbooru-ja-tag-pair-20241015"
PAIR_FILE = "data/train-00000-of-00001.parquet"

LATIN = re.compile(r"[A-Za-z]")

# Span trust for a filled wording. The evidence is the community's own
# ``other_names`` — the same field `via: wiki` (0.7) comes from — but from an
# older snapshot, partly LLM-filled, and with no back-translation behind it.
# It sits above `mt_unverified` (0.3) and below `wiki`; the rows it feeds have
# nothing today, and G4b measured that noisy supervision beats none.
TAGPAIR_VIA = "tagpair"


def load_pairs(path: Path | None = None) -> dict[str, list[str]]:
    """``tag title (spaced, lowercase) → Japanese names``, best first.

    Downloads the parquet through ``huggingface_hub`` (cached) unless a local
    copy is given. Deleted rows are skipped.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    if path is None:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        path = Path(hf_hub_download(PAIR_REPO, PAIR_FILE, repo_type="dataset"))
    out: dict[str, list[str]] = {}
    for row in pq.read_table(path).to_pylist():
        if row.get("is_deleted"):
            continue
        title = (row.get("title") or "").replace("_", " ").strip().lower()
        names = row.get("other_names") or []
        if title and names:
            out.setdefault(title, list(names))
    return out


def japanese_names(names: list[str], inventory: set[str]) -> list[str]:
    """The names that survive both Chinese guards, kana-bearing first."""
    keep = []
    for name in names:
        name = (name or "").strip()
        if not name or LATIN.search(name):
            continue
        if not tag_glossary.is_japanese(name):
            continue
        if not tag_glossary.KANA.search(name) and not all(
            c in inventory for c in name if tag_glossary.HAN.match(c)
        ):
            continue  # Han-only and built from kanji Japanese text never uses
        keep.append(name)
    return tag_glossary.rank_names(keep)


def fill(
    tags: dict[str, dict],
    counts: collections.Counter,
    pairs: dict[str, list[str]],
    inventory: set[str],
    axis_of: dict[str, str],
    *,
    max_alts: int = 4,
) -> dict:
    """Add an entry for every corpus tag the glossary leaves unresolved.

    Mutates ``tags``. A tag already carrying a ``ja`` is never touched, and
    neither is an artist handle (latin identity by construction).
    """
    filled = skipped_no_name = 0
    occ_filled = 0
    by_axis: collections.Counter = collections.Counter()
    examples: list[tuple[int, str, str]] = []
    for tag, count in counts.most_common():
        entry = tags.get(tag)
        if entry and entry.get("ja"):
            continue
        axis = axis_of.get(tag, "artist" if tag.startswith("@") else "general")
        if axis == "artist":
            continue
        names = japanese_names(pairs.get(tag, []), inventory)
        if not names:
            skipped_no_name += 1
            continue
        tags[tag] = {
            "count": count,
            "axis": axis,
            "alts": names[1 : 1 + max_alts],
            "ja": names[0],
            "via": TAGPAIR_VIA,
            "source": PAIR_REPO,
        }
        filled += 1
        occ_filled += count
        by_axis[axis] += 1
        if len(examples) < 20:
            examples.append((count, tag, names[0]))
    return {
        "filled": filled,
        "occurrences_filled": occ_filled,
        "unresolved_without_candidate": skipped_no_name,
        "by_axis": dict(by_axis),
        "examples": examples,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--glossary", type=Path, default=tag_glossary.DEFAULT_OUT)
    ap.add_argument("--out", type=Path, default=None, help="default: in place")
    ap.add_argument("--caption-index", type=Path, default=tag_glossary.DEFAULT_INDEX)
    ap.add_argument(
        "--captions", type=Path, nargs="+", default=[tag_glossary.DEFAULT_CAPTIONS]
    )
    ap.add_argument(
        "--raw-captions",
        type=Path,
        nargs="*",
        default=[build_pairs.GELCRAWL_RETRIEVED],
    )
    ap.add_argument("--tag-rules", type=Path, default=build_pairs.GELCRAWL_RULES)
    ap.add_argument(
        "--wiki",
        type=Path,
        default=tag_glossary.ASSETS / ".wiki" / tag_glossary.WIKI_FILE,
        help="kana-bearing entries here are the JA kanji inventory (Chinese guard)",
    )
    ap.add_argument("--pairs", type=Path, default=None, help="local parquet")
    ap.add_argument("--max-alts", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    payload = json.loads(args.glossary.read_text(encoding="utf-8"))
    tags = payload["tags"]
    before = sum(1 for e in tags.values() if e.get("ja"))

    index = json.loads(args.caption_index.read_text())
    axis_of = {
        tag: axis
        for axis in ("character", "copyright", "artist")
        for tag in index["groups"][axis]
    }
    roots = [(Path(p), False) for p in args.captions]
    roots += [(Path(p), True) for p in (args.raw_captions or [])]
    counts = tag_glossary.tag_counts(roots, Path(args.tag_rules))

    print(f"  [tagpair] fetching {PAIR_REPO} …", flush=True)
    pairs = load_pairs(args.pairs)
    inventory = tag_glossary.ja_kanji_inventory(args.wiki)
    print(f"  [tagpair] {len(pairs)} tag pairs, {len(inventory)} JA kanji in inventory")

    stats = fill(tags, counts, pairs, inventory, axis_of, max_alts=args.max_alts)

    total_occ = sum(counts.values())
    print(f"\n  resolved wordings {before} → {before + stats['filled']}")
    print(
        f"  filled {stats['filled']} tags / {stats['occurrences_filled']} occurrences "
        f"({100 * stats['occurrences_filled'] / total_occ:.1f}% of the corpus)"
    )
    print(f"  by axis: {stats['by_axis']}")
    print(
        f"  still unresolved (no JA candidate): {stats['unresolved_without_candidate']}"
    )
    print("  examples:")
    for count, tag, ja in stats["examples"]:
        print(f"    {count:6d}  {tag:34s} → {ja}")

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return

    meta = payload.setdefault("meta", {})
    meta["tag_pairs"] = {
        "source": PAIR_REPO,
        "filled": stats["filled"],
        "occurrences_filled": stats["occurrences_filled"],
        "via": TAGPAIR_VIA,
    }
    meta["n_tags"] = len(tags)
    out = args.out or args.glossary
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
