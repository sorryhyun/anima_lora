#!/usr/bin/env python3
"""Build an EN->JA proper-noun lexicon for the CJK distill corpus (Phase 2a, D5).

Entity list comes from our own data: ``caption_index.json`` already parses every
caption into typed axes (``character`` / ``copyright`` / ``artist``), so we only
ever ask Wikidata about names that actually occur in the training set.

Two routes, because danbooru names come in two shapes:

**Franchise-constrained**, for names qualified by a copyright -- an unguarded
label lookup on a single-token given name is useless (``aru (blue archive)``
matches a railway station, ``ann (pokemon)`` matches Anime News Network):

  1. resolve each ``copyright`` tag to candidate QIDs by label/altLabel
     (romaji alt-labels are why ``ane naru mono`` finds 姉なるもの),
  2. pull each candidate's character roster (inbound P1441/P179/P361 with a ja
     label).  **A candidate with no roster is discarded** -- that is the
     precision filter, and it is what kills ``chimera`` -> キマイラ,
  3. join ``character`` tags to their franchise's roster by token-subset match
     on the paren-stripped base name (``aru`` <= {rikuhachima, aru}).

**Global full-name**, for unqualified names (``aisaka taiga``), which the
franchise route cannot reach at all.  Flat label matching is safe *here*
because of two guards: >= 2 tokens, and the item must be a fictional character
(``P31/P279* Q95074``).  See ``match_full_names``.

Everything is CC0, so unlike the parallel-corpus sources there is no licence
question.  ko/zh labels come from the same rows for free.

Run from the repo root (the default caption-index path is repo-relative):

    python -m scripts.distill_cjk.corpus.wikidata_lexicon
    python -m scripts.distill_cjk.corpus.wikidata_lexicon --langs ja ko zh

Stdlib only -- no repo imports, no GPU. WDQS responses are cached under
``assets/.wdcache/`` so re-runs are free and a dropped batch can be re-fetched
without redoing the rest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ENDPOINT = "https://query.wikidata.org/sparql"
UA = "anima-lora-research/0.1 (research use; https://github.com/sorryhyun/anima_lora)"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ASSETS = REPO / "post_image_dataset" / "cjk_distill" / "assets"  # builder outputs (gitignored)
DEFAULT_INDEX = Path("post_image_dataset/captions/caption_index.json")
DEFAULT_OUT = ASSETS / "wikidata_lexicon.json"
CACHE_DIR = ASSETS / ".wdcache"


def sparql(query: str, *, cache: bool = True, tries: int = 4) -> list[dict]:
    """Run a SPARQL query, with on-disk caching and backoff.

    WDQS returns 502/truncated bodies under load often enough that a bare
    request loses a whole batch; cache hits also make re-runs free.
    """
    key = hashlib.sha256(query.encode()).hexdigest()[:32]
    cached = CACHE_DIR / f"{key}.json"
    if cache and cached.exists():
        return json.loads(cached.read_text())["results"]["bindings"]
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": query})
    req = urllib.request.Request(
        url, headers={"Accept": "application/sparql-results+json", "User-Agent": UA}
    )
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                body = resp.read().decode()
            data = json.loads(body)  # truncated bodies fail here, not silently
            if cache:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cached.write_text(body)
            return data["results"]["bindings"]
        except Exception as exc:  # noqa: BLE001 - any transport/parse failure retries
            print(f"    wdqs retry {attempt + 1}/{tries}: {exc}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    print("    wdqs batch FAILED (skipped)", file=sys.stderr)
    return []


def _cap(s: str) -> str:
    return " ".join(w[:1].upper() + w[1:] for w in s.split(" "))


def _toks(s: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if t}


def _strip_parens(tag: str) -> tuple[str, list[str]]:
    """``akane (bunny) (blue archive)`` -> ``("akane", ["bunny", "blue archive"])``."""
    quals = [q.strip().lower() for q in re.findall(r"\(([^()]*)\)", tag)]
    return re.sub(r"\s*\([^()]*\)", "", tag).strip(), quals


def resolve_franchises(tags: list[str], batch: int = 50) -> dict[str, set[str]]:
    """Danbooru copyright tag -> candidate QIDs (label or altLabel, case variants)."""
    variant_to_tag: dict[str, str] = {}
    for tag in tags:
        base, _ = _strip_parens(tag)
        for v in {base, _cap(base), base.upper()}:
            if v:
                variant_to_tag.setdefault(v, tag)
    variants = sorted(variant_to_tag)
    out: dict[str, set[str]] = defaultdict(set)
    for i in range(0, len(variants), batch):
        chunk = variants[i : i + batch]
        values = " ".join(f'"{v}"@en' for v in chunk)
        rows = sparql(
            f"SELECT ?en ?item WHERE {{ VALUES ?en {{ {values} }} "
            f"?item rdfs:label|skos:altLabel ?en . }} LIMIT 5000"
        )
        for r in rows:
            out[variant_to_tag[r["en"]["value"]]].add(
                r["item"]["value"].rsplit("/", 1)[-1]
            )
        print(
            f"  franchises {i + len(chunk)}/{len(variants)} variants "
            f"-> {len(out)} tags with candidates",
            file=sys.stderr,
        )
    return out


def pull_rosters(
    qids: list[str], langs: list[str], batch: int = 12
) -> dict[str, list[dict]]:
    """QID -> [{en, <lang>...}] for everything linked into it as a character/part."""
    lang_filter = ", ".join(f'"{lang}"' for lang in langs)
    rosters: dict[str, list[dict]] = defaultdict(list)
    for i in range(0, len(qids), batch):
        chunk = qids[i : i + batch]
        values = " ".join(f"wd:{q}" for q in chunk)
        rows = sparql(
            f"""SELECT ?series ?c ?en ?lab (LANG(?lab) AS ?lang) WHERE {{
  VALUES ?series {{ {values} }}
  ?c (wdt:P1441|wdt:P179|wdt:P361) ?series .
  ?c rdfs:label ?en . FILTER(LANG(?en) = "en")
  ?c rdfs:label ?lab . FILTER(LANG(?lab) IN ({lang_filter}))
}} LIMIT 20000"""
        )
        merged: dict[tuple[str, str], dict] = {}
        for r in rows:
            series = r["series"]["value"].rsplit("/", 1)[-1]
            entry = merged.setdefault(
                (series, r["c"]["value"]),
                {"en": r["en"]["value"], "qid": r["c"]["value"].rsplit("/", 1)[-1]},
            )
            entry[r["lang"]["value"]] = r["lab"]["value"]
        for (series, _), entry in merged.items():
            rosters[series].append(entry)
        print(
            f"  rosters {i + len(chunk)}/{len(qids)} qids "
            f"-> {sum(len(v) for v in rosters.values())} entries",
            file=sys.stderr,
        )
    return rosters


def pull_labels(qids: list[str], langs: list[str], batch: int = 60) -> dict[str, dict]:
    """QID -> {lang: label} for the resolved franchises themselves."""
    lang_filter = ", ".join(f'"{lang}"' for lang in langs)
    out: dict[str, dict] = defaultdict(dict)
    for i in range(0, len(qids), batch):
        chunk = qids[i : i + batch]
        values = " ".join(f"wd:{q}" for q in chunk)
        rows = sparql(
            f"""SELECT ?item ?lab (LANG(?lab) AS ?lang) WHERE {{
  VALUES ?item {{ {values} }}
  ?item rdfs:label ?lab . FILTER(LANG(?lab) IN ({lang_filter}))
}} LIMIT 5000"""
        )
        for r in rows:
            out[r["item"]["value"].rsplit("/", 1)[-1]][r["lang"]["value"]] = r["lab"][
                "value"
            ]
    return out


def _pick(hits: list[dict], lang: str) -> tuple[dict | None, list[dict]]:
    """Choose one entry, or report a genuine ambiguity.

    Dedupe by QID, prefer the plainest EN label (``Aikiyo Fuuka`` over
    ``Aikiyo Fuuka (New Year ver.)``).  Ambiguity is judged on the **target
    string**, not the tiebreak: candidates that all denote the same JA name are
    not ambiguous no matter how many QIDs they span, and a shorter label does
    not get to silently win over a genuinely different name.
    """
    seen, uniq = set(), []
    for e in sorted(hits, key=lambda e: (len(_toks(e["en"])), e["en"])):
        if e["qid"] not in seen:
            seen.add(e["qid"])
            uniq.append(e)
    if not uniq:
        return None, []
    labels = {e[lang] for e in uniq if lang in e}
    if len(labels) <= 1:
        return (uniq[0] if lang in uniq[0] else None), []
    # distinct targets: only accept if one candidate is strictly plainer
    if len(_toks(uniq[0]["en"])) < len(_toks(uniq[1]["en"])) and lang in uniq[0]:
        return uniq[0], []
    return None, uniq[:4]


def join_characters(
    tags: list[str],
    franchise_qids: dict[str, set[str]],
    rosters: dict[str, list[dict]],
    lang: str,
) -> tuple[dict, dict, list, list]:
    """Match each character tag against its own franchise's roster.

    The pool unions **every** candidate QID's roster for that copyright tag --
    a franchise routinely splits across a series item, an anime item and a game
    item, and taking only the largest roster drops the rest on the floor.
    """
    pools = {
        tag: [(_toks(e["en"]), e) for qid in qids for e in rosters.get(qid, [])]
        for tag, qids in franchise_qids.items()
    }
    matched: dict[str, dict] = {}
    ambiguous: dict[str, list] = {}
    unmatched: list[str] = []
    no_franchise: list[str] = []
    for tag in tags:
        base, quals = _strip_parens(tag)
        pool = next((pools[q] for q in reversed(quals) if q in pools), None)
        if pool is None:
            no_franchise.append(tag)
            continue
        base_toks = _toks(base)
        if not base_toks:
            unmatched.append(tag)
            continue
        pick, amb = _pick([e for toks, e in pool if base_toks <= toks], lang)
        if pick:
            matched[tag] = pick
        elif amb:
            ambiguous[tag] = amb
        else:
            unmatched.append(tag)
    return matched, ambiguous, unmatched, no_franchise


def match_full_names(
    tags: list[str], langs: list[str], batch: int = 50
) -> tuple[dict, dict, list]:
    """Global fallback for **multi-token** names with no resolvable franchise.

    A flat label lookup is unusable for single-token given names (``aru`` hits
    a railway station), but most unresolved tags are full names
    (``aisaka taiga``), where collision risk is low.  Two guards keep it
    honest: the name must have >= 2 tokens, and the item must be a fictional
    character (``P31/P279* Q95074``) -- that constraint alone is what stops
    ``Anime News Network``-class matches.
    """
    lang_filter = ", ".join(f'"{lang}"' for lang in langs)
    variant_to_tag: dict[str, str] = {}
    eligible = []
    for tag in tags:
        base, _ = _strip_parens(tag)
        if len(_toks(base)) < 2:
            continue
        eligible.append(tag)
        for v in {base, _cap(base)}:
            variant_to_tag.setdefault(v, tag)
    variants = sorted(variant_to_tag)
    hits: dict[str, dict[str, dict]] = defaultdict(dict)
    for i in range(0, len(variants), batch):
        chunk = variants[i : i + batch]
        values = " ".join(f'"{v}"@en' for v in chunk)
        rows = sparql(
            f"""SELECT ?en ?item ?lab (LANG(?lab) AS ?lang) WHERE {{
  VALUES ?en {{ {values} }}
  ?item rdfs:label|skos:altLabel ?en .
  ?item wdt:P31/wdt:P279* wd:Q95074 .
  ?item rdfs:label ?lab . FILTER(LANG(?lab) IN ({lang_filter}))
}} LIMIT 10000"""
        )
        for r in rows:
            tag = variant_to_tag[r["en"]["value"]]
            qid = r["item"]["value"].rsplit("/", 1)[-1]
            entry = hits[tag].setdefault(qid, {"en": r["en"]["value"], "qid": qid})
            entry[r["lang"]["value"]] = r["lab"]["value"]
        print(
            f"  full names {i + len(chunk)}/{len(variants)} variants "
            f"-> {len(hits)} tags hit",
            file=sys.stderr,
        )
    matched, ambiguous = {}, {}
    for tag, by_qid in hits.items():
        pick, amb = _pick(list(by_qid.values()), langs[0])
        if pick:
            matched[tag] = pick
        elif amb:
            ambiguous[tag] = amb
    still = [t for t in eligible if t not in matched and t not in ambiguous]
    return matched, ambiguous, still


def cjk_codepoints(strings) -> set[str]:
    return {ch for s in strings for ch in s if "぀" <= ch <= "ヿ" or "一" <= ch <= "鿿"}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--caption-index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--langs", nargs="+", default=["ja"], help="label languages to keep"
    )
    ap.add_argument(
        "--no-cache", action="store_true", help="ignore the on-disk WDQS cache"
    )
    args = ap.parse_args()

    index = json.loads(args.caption_index.read_text())
    groups = index["groups"]
    copyrights = sorted(groups["copyright"])
    characters = sorted(groups["character"])
    print(
        f"caption index: {len(characters)} characters, {len(copyrights)} copyrights, "
        f"{len(groups.get('artist', ()))} artists",
        file=sys.stderr,
    )
    if args.no_cache and CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()

    candidates = resolve_franchises(copyrights)
    qids = sorted({q for s in candidates.values() for q in s})
    rosters = pull_rosters(qids, args.langs)

    # a candidate with no roster at all is not a franchise -- drop it, but keep
    # every candidate that does have one (franchises split across series/anime/game items)
    franchise_qids = {
        tag: {q for q in qs if rosters.get(q)}
        for tag, qs in candidates.items()
        if any(rosters.get(q) for q in qs)
    }

    primary = args.langs[0]
    matched, ambiguous, unmatched, no_franchise = join_characters(
        characters, franchise_qids, rosters, primary
    )
    in_resolved = len(matched) + len(ambiguous) + len(unmatched)

    # global fallback for multi-token names the franchise route could not reach
    gl_matched, gl_ambiguous, gl_still = match_full_names(
        unmatched + no_franchise, args.langs
    )
    # per-row provenance: the two routes have different precision profiles, and
    # the proposal ships trained/zero-shot provenance downstream anyway
    for e in matched.values():
        e["via"] = "franchise_roster"
    for e in gl_matched.values():
        e["via"] = "full_name"
    matched.update(gl_matched)
    ambiguous.update(gl_ambiguous)

    entries = {tag: e for tag, e in matched.items() if primary in e}
    franchise_best = {
        tag: max(qs, key=lambda q: len(rosters.get(q, ())))
        for tag, qs in franchise_qids.items()
    }
    franchise_labels = pull_labels(sorted(set(franchise_best.values())), args.langs)
    franchises = {
        tag: {"qid": qid, **franchise_labels.get(qid, {})}
        for tag, qid in franchise_best.items()
    }
    covered = cjk_codepoints(
        [e[primary] for e in entries.values()]
        + [f[primary] for f in franchises.values() if primary in f]
    )
    stats = {
        "caption_index": str(args.caption_index),
        "langs": args.langs,
        "n_copyright": len(copyrights),
        "copyright_with_candidates": len(candidates),
        "copyright_resolved": len(franchise_qids),
        "n_character": len(characters),
        "character_in_resolved_franchise": in_resolved,
        "matched_via_franchise": len(matched) - len(gl_matched),
        "matched_via_full_name": len(gl_matched),
        "character_matched": len(matched),
        "character_ambiguous": len(ambiguous),
        "character_unmatched": len(gl_still),
        "recall_within_resolved": round(
            (len(matched) - len(gl_matched)) / max(1, in_resolved), 3
        ),
        "recall_overall": round(len(matched) / max(1, len(characters)), 3),
        "franchise_labelled": sum(1 for f in franchises.values() if primary in f),
        "unique_cjk_codepoints": len(covered),
    }
    payload = {
        "stats": stats,
        "characters": entries,
        "franchises": franchises,
        "ambiguous": ambiguous,
        "unmatched": gl_still,
        "cjk_codepoints": "".join(sorted(covered)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    for k, v in stats.items():
        print(f"{k}: {v}", file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
