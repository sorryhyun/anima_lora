"""EN→JA glossary for every tag our captions actually use (Phase 2a, D1 backbone).

Our captions are tag strings, so the JA side of the corpus is only as good as
the JA wording of the tags — and the *wording* is the whole point: ext rows are
only trained where the corpus visits them, so a JA caption saying 二本の髪 where
users type ツインテール trains the wrong rows. That is the same
distribution argument the proper-noun lexicon makes (``../done.md``), applied to
the general vocabulary.

Machine translation is the wrong primary tool for this: it renders meaning, not
community register. Measured on Hy-MT2-1.8B, a held-out idiom probe scored 8/28
bare and 9/28 with few-shot exemplars (``mt.py --probe``) — the misses are
knowledge (ツインテール, カメラ目線, 割座), not instruction following.

So the glossary is sourced in priority order:

1. **Danbooru wiki ``other_names``** — the tagging community's own Japanese for
   its own tags, which is exactly the register users type. Covers **85% of our
   tag occurrences** and 1,130/1,314 character tags.
2. **Wikidata lexicon** (``wikidata_lexicon.json``) — preferred for proper
   nouns where it has an entry: CC0, spot-checked, and it carries ko/zh for the
   deferred phases.
3. **Hy-MT2** (``--mt``, needs a GPU) — the residue, which is mostly
   compositional tags (``black skirt``, ``open jacket``) where literal
   translation is the correct answer anyway.
4. A small built-in map for the rating band, which is ours and not Danbooru's.

Script detection matters: ``other_names`` mixes ja/zh/ko. Entries containing
kana are unambiguously Japanese; Han-only entries are accepted only if they
encode in Shift-JIS, which rejects simplified Chinese (小鸟游星野 → out,
火宮チナツ → in) at zero dependency cost.

    python project/cjk_aware_anima/datasets/tag_glossary.py            # CPU, sources 1/2/4
    make daemon-run ARGS="project/cjk_aware_anima/datasets/tag_glossary.py --mt"
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT))

import build_pairs  # noqa: E402  (sibling module — caption roots + loader)

WIKI_REPO = "kierarkia/danbooru-wiki-2026"
WIKI_FILE = "danbooru_wiki_dataset_2026-04-28.jsonl"
DEFAULT_INDEX = REPO / "post_image_dataset/captions/caption_index.json"
DEFAULT_CAPTIONS = REPO / "image_dataset"
DEFAULT_OUT = ASSETS / "tag_glossary_ja.json"

KANA = re.compile(r"[぀-ゟ゠-ヿー]")
HAN = re.compile(r"[一-鿿]")

# The rating band is ours (safe/sensitive/nsfw/explicit), not a Danbooru tag
# whose wiki would carry a Japanese name.
# Sources the arbitration may re-open. Everything else (override / rating /
# wikidata / artist passthrough) is pinned: letting MT touch those overwrote the
# rating band with デリケート instead of センシティブ.
ARBITRATED = {"wiki", "wiki_han", "unresolved"}

# Candidate provenance, best first at equal evidence: the 2026 wiki dump is the
# freshest reading of the community field; the tag-pair set is the same field at
# Oct-2024 partly LLM-filled; MT renders the *string*, not the booru sense
# (`bow` → お辞儀 "bowing" where the tag means 蝶結び the ribbon), so at equal
# back-translation F1 a community-attested name outranks it.
SRC_RANK = {"wiki": 0, "tagpair": 1, "mt": 2}
TAGPAIR_VERIFIED_VIA = "tagpair_verified"

RATING_JA = {
    "safe": "全年齢",
    "sensitive": "センシティブ",
    "nsfw": "成人向け",
    "explicit": "露骨な性描写",
}


def is_japanese(s: str) -> bool:
    """Kana ⇒ Japanese. Han-only ⇒ Japanese iff Shift-JIS can encode it."""
    if KANA.search(s):
        return True
    if not HAN.search(s):
        return False
    try:
        s.encode("shift_jis")
    except UnicodeEncodeError:
        return False
    return True


def rank_names(names: list[str]) -> list[str]:
    """Japanese candidates, kana-bearing first (the least ambiguous evidence)."""
    ja = [n for n in names if is_japanese(n)]
    ja.sort(key=lambda n: (0 if KANA.search(n) else 1, len(n)))
    return ja


def ja_kanji_inventory(path: Path, min_count: int = 3) -> set[str]:
    """Kanji that really occur in Japanese text, learned from the dump itself.

    Shift-JIS rejects simplified Chinese but happily encodes *traditional*
    Chinese (棕毛, 藍眼睛 — both scored a perfect back-translation F1 and are
    still not Japanese). Every ``other_names`` entry containing kana is
    Japanese by construction, so the kanji inside those entries are a free
    inventory of the JA repertoire; a Han-only candidate built from characters
    that never show up there is almost certainly Chinese.
    """
    freq: collections.Counter = collections.Counter()
    for line in path.open(encoding="utf-8"):
        for name in json.loads(line).get("other_names") or []:
            if KANA.search(name):
                freq.update(c for c in name if HAN.match(c))
    return {c for c, n in freq.items() if n >= min_count}


def load_wiki(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        print(f"  [glossary] fetching {WIKI_REPO} …", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "hf",
                "download",
                WIKI_REPO,
                WIKI_FILE,
                "--repo-type",
                "dataset",
                "--local-dir",
                str(path.parent),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    out: dict[str, list[str]] = {}
    for line in path.open(encoding="utf-8"):
        d = json.loads(line)
        title = (d.get("title") or "").replace("_", " ").strip().lower()
        names = rank_names(d.get("other_names") or [])
        if title and names:
            out[title] = names
    return out


def tag_counts(
    caption_roots: list[tuple[Path, bool]], rules: Path
) -> collections.Counter:
    """Tag occurrences over the same caption roots ``build_pairs.py`` composes.

    The glossary must span whatever D1 spans, or the widened captions compose
    with latin passthrough on every tag the narrow build never saw — so this
    shares ``build_pairs.load_captions`` (multi-root, raw roots normalized
    through gelcrawl's rules) rather than re-globbing one directory.
    """
    from build_pairs import load_captions  # noqa: PLC0415

    counts: collections.Counter = collections.Counter()
    for _, text in load_captions(caption_roots, rules):
        for seg in text.split(","):
            seg = seg.strip()
            if seg:
                counts[seg] += 1
    return counts


def build(args: argparse.Namespace) -> dict:
    index = json.loads(Path(args.caption_index).read_text())
    overrides = (
        json.loads(Path(args.overrides).read_text())
        if Path(args.overrides).exists()
        else {}
    )
    lexicon = json.loads(Path(args.lexicon).read_text())["characters"]
    wiki = load_wiki(Path(args.wiki))
    roots = [(Path(p), False) for p in args.captions]
    roots += [(Path(p), True) for p in (args.raw_captions or [])]
    counts = tag_counts(roots, Path(args.tag_rules))

    axis_of: dict[str, str] = {}
    for axis in ("character", "copyright", "artist"):
        for tag in index["groups"][axis]:
            axis_of[tag] = axis

    tags: dict[str, dict] = {}
    for tag, count in counts.most_common():
        axis = axis_of.get(tag, "artist" if tag.startswith("@") else "general")
        entry = {"count": count, "axis": axis, "alts": []}

        wiki_names = wiki.get(tag.lower(), [])
        lex = lexicon.get(tag)

        if tag in overrides:
            entry |= {"ja": overrides[tag], "via": "override"}
            entry["alts"] = [n for n in wiki_names if n != overrides[tag]][
                : args.max_alts
            ]
        elif axis == "artist":
            # pixiv/danbooru handles are latin identity — users type them latin,
            # and translating them would train rows nobody prompts with.
            entry |= {"ja": tag, "via": "passthrough"}
        elif tag in RATING_JA:
            entry |= {"ja": RATING_JA[tag], "via": "rating"}
        elif lex and lex.get("ja"):
            entry |= {"ja": lex["ja"], "via": "wikidata", "qid": lex.get("qid")}
            entry["alts"] = [n for n in wiki_names if n != lex["ja"]][: args.max_alts]
        elif wiki_names:
            entry |= {
                "ja": wiki_names[0],
                "via": "wiki" if KANA.search(wiki_names[0]) else "wiki_han",
            }
            entry["alts"] = wiki_names[1 : 1 + args.max_alts]
        else:
            entry |= {"ja": None, "via": "unresolved"}
        tags[tag] = entry

    if args.reselect:
        n = reselect(tags, Path(args.reselect), args.accept_f1)
        print(f"  [glossary] re-selected {n} tags from {args.reselect} (no GPU)")
    elif args.mt:
        _mt_pass(tags, args)

    total = sum(counts.values())
    by_via: collections.Counter = collections.Counter()
    occ_via: collections.Counter = collections.Counter()
    for tag, e in tags.items():
        by_via[e["via"]] += 1
        occ_via[e["via"]] += e["count"]
    codepoints = {
        c
        for e in tags.values()
        if e["ja"]
        for c in e["ja"]
        if is_japanese(c) or KANA.search(c)
    }

    return {
        "meta": {
            "caption_index": str(args.caption_index),
            "wiki": f"{WIKI_REPO}/{WIKI_FILE}",
            "lexicon": str(args.lexicon),
            "mt_model": args.model if args.mt else None,
            "n_tags": len(tags),
            "n_occurrences": total,
            "types_by_via": dict(by_via),
            "occurrences_by_via": dict(occ_via),
            "occurrence_coverage": round(
                100 * (total - occ_via.get("unresolved", 0)) / max(total, 1), 2
            ),
            "unique_cjk_codepoints": len(codepoints),
        },
        "tags": tags,
    }


_STOP = {
    "a",
    "an",
    "the",
    "of",
    "with",
    "in",
    "on",
    "at",
    "is",
    "are",
    "to",
    "and",
    "her",
    "his",
    "their",
    "its",
    "one",
    "s",
}


def _norm_tokens(s: str) -> set[str]:
    """Lowercase content tokens, lightly stemmed — enough to compare short tags."""
    out = set()
    for w in re.split(r"[^a-z0-9]+", s.lower()):
        if not w or w in _STOP:
            continue
        for suf in ("ing", "ed", "es", "s"):
            if len(w) > 4 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.add(w)
    return out


def _f1(a: str, b: str) -> float:
    ta, tb = _norm_tokens(a), _norm_tokens(b)
    if not ta or not tb:
        return 0.0
    hit = len(ta & tb)
    if not hit:
        return 0.0
    p, r = hit / len(tb), hit / len(ta)
    return 2 * p * r / (p + r)


def choose(entry: dict, cands: list[dict], mt: str, accept_f1: float) -> None:
    """Pick the wording for one tag from its scored candidates.

    Pure post-processing over stored data — no GPU. Ranking is: recovered
    meaning first (back-translation F1), then *demonstrably* Japanese, then
    source, then brevity. Kana is the only proof of Japaneseness; the
    per-character inventory merely rules out characters Japanese never uses, so
    a Chinese word built from JA-common characters (珠宝, 拉下衣服) still gets
    through and brevity tie-breaking would hand it the win over ジュエリー at
    the same F1. Hence kana > the JA-targeting model's own rendering >
    everything else, and inside a kana tie the community field (which knows the
    booru *sense*) beats the MT rendering (which only knows the string).
    """
    keep = [c for c in cands if c.get("ja_ok", True)]
    keep.sort(
        key=lambda c: (
            -c["f1"],
            0 if c.get("kana") else 1 if c.get("mt") else 2,
            SRC_RANK.get(c.get("src"), 1),
            len(c["ja"]),
        )
    )
    entry["mt_ja"] = mt or None
    entry["candidates"] = [
        {k: c[k] for k in ("ja", "back", "f1", "kana", "src") if k in c} for c in keep
    ]
    entry["rejected_non_ja"] = [c["ja"] for c in cands if not c.get("ja_ok", True)]

    best = keep[0] if keep else None
    if best and best["f1"] >= accept_f1:
        if best["ja"] == mt:
            via = "mt_verified"
        elif best.get("src") == "tagpair":
            via = TAGPAIR_VERIFIED_VIA
        else:
            via = "wiki_verified"
        entry |= {
            "ja": best["ja"],
            "via": via,
            "f1": best["f1"],
        }
    elif mt:
        # No candidate recovers the tag — `closed mouth` came back as 目を閉じる
        # (closed *eyes*). Keep it, but mark it so the review file surfaces the
        # class instead of shipping it silently.
        entry |= {"ja": mt, "via": "mt_unverified"}
    elif best:
        entry |= {"ja": best["ja"], "via": "wiki_unverified", "f1": best["f1"]}
    else:
        entry |= {"ja": None, "via": "unresolved"}


def reselect(tags: dict[str, dict], prior_path: Path, accept_f1: float) -> int:
    """Re-derive every choice from a previous build's stored candidates.

    Selection is pure post-processing, so a ranking or threshold fix must never
    re-buy the translations: a prior ``tag_glossary_ja.json`` already holds each
    candidate's back-translation and F1 plus the MT rendering. Changing
    ``--accept-f1`` or the ranking is then seconds of CPU, not an hour of GPU.
    """
    prior = json.loads(prior_path.read_text(encoding="utf-8"))["tags"]
    n = 0
    for tag, e in tags.items():
        if e["axis"] != "general" or e["via"] not in ARBITRATED:
            continue  # never re-open a pinned source (override / rating / wikidata)
        p = prior.get(tag)
        if not p or not (p.get("candidates") or p.get("mt_ja")):
            continue
        mt = (p.get("mt_ja") or "").strip()
        cands = [
            {
                **c,
                "ja_ok": True,  # prior build already dropped the non-JA ones
                "kana": KANA.search(c["ja"]) is not None,
                "mt": c["ja"] == mt,
                # pre-src builds only banked wiki candidates + the MT rendering
                "src": c.get("src") or ("mt" if c["ja"] == mt else "wiki"),
            }
            for c in (p.get("candidates") or [])
        ]
        choose(e, cands, mt, accept_f1)
        n += 1
    return n


def _mt_pass(tags: dict[str, dict], args: argparse.Namespace) -> None:
    """Render every general tag with MT, then arbitrate MT vs wiki idiom (GPU).

    Neither source is trustworthy alone. MT is literal and safe on compositional
    tags (``black skirt``) but does not know community idiom — measured 46%/54%
    on the held-out idiom probe even at 7B. The wiki knows the idiom but its
    ``other_names`` is a free-text field that also carries narrower compounds
    (``underwear`` → 下着コート) and outright different concepts
    (``multiple girls`` → 群像).

    The arbiter is back-translation: a wiki candidate is accepted only if
    translating it *back* to English recovers the tag (token F1 ≥
    ``--accept-f1``). Idiom that verifiably means the tag wins; the rest falls
    back to the literal MT rendering. Whatever the two sources disagree on lands
    in the review file for the user sign-off the phase gate asks for.

    The tag-pair set (``tag_pairs.py``'s source) competes on the same terms
    (D1-pairs item 2): its names enter the pool guarded exactly like the fill
    (latin drop, ``is_japanese``, kanji inventory), get back-translated, and win
    only on evidence — ``via: tagpair_verified``. It knows the booru sense MT
    cannot (`bow` → 蝶結び the ribbon, not お辞儀 *bowing*), which is why the
    tie-break in ``choose`` puts community sources ahead of the MT rendering.
    """
    sys.path.insert(0, str(ROOT))
    from mt import TAG_BACKGROUND, TAG_FEWSHOT, MTEngine, Request  # noqa: PLC0415

    general = [
        t for t, e in tags.items() if e["axis"] == "general" and e["via"] in ARBITRATED
    ]
    if args.limit:
        general = sorted(general, key=lambda t: -tags[t]["count"])[: args.limit]
    todo = set(general)

    # Exemplars must be hand-checked: drawing them from the unverified wiki head
    # fed junk back into the prompts (`underwear` → 下着コート became a shot and
    # MT then reproduced it).
    shots = TAG_FEWSHOT[: args.n_shots]
    inventory = ja_kanji_inventory(Path(args.wiki))
    print(f"  [glossary] JA kanji inventory: {len(inventory)} chars", flush=True)

    pair_names: dict[str, list[str]] = {}
    if not args.no_tag_pairs:
        import tag_pairs  # noqa: PLC0415  (sibling — the shared source + guards)

        print(f"  [glossary] fetching {tag_pairs.PAIR_REPO} …", flush=True)
        raw_pairs = tag_pairs.load_pairs(args.tag_pairs_file)
        pair_names = {
            t: tag_pairs.japanese_names(raw_pairs.get(t, []), inventory)[
                : args.n_pair_candidates
            ]
            for t in todo
        }
        n_with = sum(1 for v in pair_names.values() if v)
        print(f"  [glossary] tag-pair candidates for {n_with}/{len(todo)} tags")

    engine = MTEngine(args.model, greedy=True, gpu_budget=args.gpu_budget)

    print(f"  [glossary] EN→JA over {len(todo)} general tags", flush=True)
    order = sorted(todo)
    mt_ja = dict(
        zip(
            order,
            engine.translate(
                [
                    Request(text=t, background=TAG_BACKGROUND, terms=shots)
                    for t in order
                ],
                batch_size=args.batch_size,
                progress_every=20,
                max_new_tokens=args.max_new_tokens,
                cache=args.mt_cache / "glossary_en2ja.jsonl",
            ),
        )
    )

    # Back-translate the candidates: wiki (primary + alts), tag-pair names, MT.
    pairs: list[tuple[str, str]] = []
    src_of: dict[tuple[str, str], str] = {}
    for tag in order:
        e = tags[tag]
        wiki_c = ([e["ja"]] if e["via"].startswith("wiki") and e["ja"] else []) + e[
            "alts"
        ]
        sourced = [(c, "wiki") for c in wiki_c[: args.n_candidates]]
        sourced += [(c, "tagpair") for c in pair_names.get(tag, [])]
        if mt_ja.get(tag):  # the literal rendering competes on the same terms
            sourced.append((mt_ja[tag].strip().splitlines()[0].strip(), "mt"))
        seen: dict[str, str] = {}
        for cand, src in sourced:
            if cand and cand not in seen:
                seen[cand] = src
        pairs += [(tag, c) for c in seen]
        src_of |= {(tag, c): s for c, s in seen.items()}
    print(
        f"  [glossary] JA→EN back-translation over {len(pairs)} candidates", flush=True
    )
    back = engine.translate(
        [Request(text=c) for _, c in pairs],
        target_lang="en",
        batch_size=args.batch_size,
        progress_every=20,
        max_new_tokens=args.max_new_tokens,
        cache=args.mt_cache / "glossary_ja2en.jsonl",
    )

    scored: dict[str, list[tuple[float, str, str, bool]]] = collections.defaultdict(
        list
    )
    for (tag, cand), en in zip(pairs, back):
        ja_ok = KANA.search(cand) is not None or all(
            c in inventory for c in cand if HAN.match(c)
        )
        scored[tag].append(
            {
                "f1": round(_f1(tag, en), 3),
                "ja": cand,
                "back": en.strip(),
                "ja_ok": ja_ok,
                "kana": KANA.search(cand) is not None,
                "src": src_of[(tag, cand)],
                "mt": cand == (mt_ja.get(tag) or "").strip().splitlines()[:1][0]
                if mt_ja.get(tag)
                else False,
            }
        )

    for tag in order:
        mt = (mt_ja.get(tag) or "").strip().splitlines()
        choose(
            tags[tag], scored.get(tag, []), mt[0].strip() if mt else "", args.accept_f1
        )


def kanji_review_rows(payload: dict, min_f1: float = 0.75) -> list[tuple]:
    """The D1-words axis: a katakana-only choice with a Han-only rival at
    tie-or-better F1.

    Back-translation cannot arbitrate these and the kana-first ranking must not
    be relaxed for them: the rival is either native Japanese wrongly rejected
    (鎧, 俯瞰, 接写) or Chinese that survived the character guards — and
    `bed` → 床 (*floor* in Japanese, *bed* in Chinese) shows only Japanese
    knowledge separates the two. Human review axis; fixes go to
    ``tag_overrides.json``.
    """
    rows = []
    for tag, e in payload["tags"].items():
        ja = e.get("ja")
        if e["axis"] != "general" or not ja or not KANA.search(ja) or HAN.search(ja):
            continue
        floor = max(float(e.get("f1") or 0.0), min_f1)
        rivals = [
            c
            for c in e.get("candidates") or []
            if HAN.search(c["ja"])
            and not KANA.search(c["ja"])
            and c.get("f1", 0) >= floor
        ]
        if rivals:
            rows.append((e["count"], tag, e, rivals[0]))
    rows.sort(reverse=True, key=lambda r: r[0])
    return rows


def write_review(payload: dict, path: Path, top: int) -> int:
    """Dump the highest-traffic MT-vs-wiki disagreements for human sign-off.

    Ordered by occurrence count, so reading down the list buys coverage as fast
    as possible: the top 200 rows here decide the wording of most of the corpus.
    Corrections go into ``tag_overrides.json`` (committed) and win over every
    automatic source on the next build.
    """
    rows = [
        (e["count"], t, e)
        for t, e in payload["tags"].items()
        if e["axis"] == "general"
        and e.get("mt_ja")
        and (
            e["via"] == "mt_unverified"
            or (e.get("candidates") and e["candidates"][0]["ja"] != e.get("mt_ja"))
            # MT round-tripped its own sense, but the community field disagrees
            # with real recovered meaning — the polysemy class the arbiter is
            # structurally blind to (`bow`: お辞儀 "bowing" verifies at 1.0
            # while 蝶結び the ribbon comes back "bow tie knot" at 0.5). Only a
            # human can pick the booru sense; surface it.
            or (
                e["via"] == "mt_verified"
                and any(
                    c.get("src") != "mt" and c["ja"] != e["ja"] and c["f1"] >= 0.3
                    for c in e.get("candidates") or []
                )
            )
        )
    ]
    rows.sort(reverse=True, key=lambda r: r[0])
    rows = rows[:top]

    lines = [
        "# Tag glossary — review the disagreements",
        "",
        "Rows where the Danbooru-wiki idiom and the MT rendering differ, most",
        "frequent first. **chosen** is what the corpus will use; `wiki_verified`",
        "means back-translation recovered the tag, `mt` means it did not.",
        "",
        'Fix anything wrong by adding `"<en tag>": "<ja>"` to',
        "`tag_overrides.json`, then rebuild — overrides beat every source.",
        "",
        "| n | tag | chosen | via | MT | wiki candidates (back-translation, F1) |",
        "|--:|---|---|---|---|---|",
    ]
    for count, tag, e in rows:
        cands = " · ".join(
            f"{c['ja']} ({c['back']}, {c['f1']}"
            + (f", {c['src']}" if c.get("src") and c["src"] != "wiki" else "")
            + ")"
            for c in e["candidates"][:3]
        )
        lines.append(
            f"| {count} | {tag} | **{e['ja']}** | {e['via']} | {e.get('mt_ja')} | {cands} |"
        )

    kanji = kanji_review_rows(payload)[:top]
    if kanji:
        lines += [
            "",
            "## Katakana primary vs native-kanji rival (D1-words)",
            "",
            "The chosen wording is katakana-only and a Han-only candidate ties",
            "or beats it on back-translation. The kana-first ranking is kept on",
            "purpose (`bed` → 床 is *floor* in Japanese, *bed* in Chinese — no",
            "guard separates them), so a kanji wording here only ships through",
            "an override. Judge each rival: native Japanese → override it in;",
            "Chinese → leave it.",
            "",
            "| n | tag | chosen | kanji rival (back-translation, F1, src) |",
            "|--:|---|---|---|",
        ]
        for count, tag, e, c in kanji:
            lines.append(
                f"| {count} | {tag} | **{e['ja']}** | {c['ja']} "
                f"({c['back']}, {c['f1']}, {c.get('src', 'wiki')}) |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(rows) + len(kanji)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--caption-index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--captions", type=Path, nargs="+", default=[DEFAULT_CAPTIONS])
    ap.add_argument(
        "--raw-captions",
        type=Path,
        nargs="*",
        default=[build_pairs.GELCRAWL_RETRIEVED],
        help="raw crawler roots, normalized before counting (pass empty to disable)",
    )
    ap.add_argument("--tag-rules", type=Path, default=build_pairs.GELCRAWL_RULES)
    ap.add_argument("--lexicon", type=Path, default=ASSETS / "wikidata_lexicon.json")
    ap.add_argument("--wiki", type=Path, default=ASSETS / ".wiki" / WIKI_FILE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-alts", type=int, default=4)
    ap.add_argument("--overrides", type=Path, default=ROOT / "tag_overrides.json")
    ap.add_argument("--review", type=Path, default=ASSETS / "tag_glossary_review.md")
    ap.add_argument("--review-top", type=int, default=200)
    ap.add_argument("--accept-f1", type=float, default=0.75)
    ap.add_argument("--n-candidates", type=int, default=3)
    ap.add_argument(
        "--no-tag-pairs",
        action="store_true",
        help="drop the tag-pair set from the --mt candidate pool (D1-pairs item 2)",
    )
    ap.add_argument(
        "--tag-pairs-file", type=Path, default=None, help="local parquet override"
    )
    ap.add_argument("--n-pair-candidates", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="MT only the top-N tags")
    ap.add_argument("--mt", action="store_true", help="translate the residue (GPU)")
    ap.add_argument(
        "--reselect",
        type=Path,
        default=None,
        help="re-derive choices from a previous build's stored candidates (no GPU)",
    )
    ap.add_argument("--model", default="tencent/Hy-MT2-7B")
    ap.add_argument("--gpu-budget", default="13GiB", help="offload past this")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-shots", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=32, help="tags are short")
    ap.add_argument(
        "--mt-cache",
        type=Path,
        default=ASSETS / ".mtcache",
        help="per-batch translation cache — a killed run resumes from here",
    )
    args = ap.parse_args()

    payload = build(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    n_review = write_review(payload, args.review, args.review_top)

    m = payload["meta"]
    print(f"\nwrote {args.out}")
    print(f"wrote {args.review} ({n_review} rows to review)")
    print(f"  tags {m['n_tags']} / occurrences {m['n_occurrences']}")
    print(f"  occurrence coverage: {m['occurrence_coverage']}%")
    print(f"  unique CJK codepoints: {m['unique_cjk_codepoints']}")
    for via, n in sorted(m["types_by_via"].items(), key=lambda kv: -kv[1]):
        occ = m["occurrences_by_via"][via]
        print(
            f"    {via:12s} {n:5d} types  {occ:7d} occ ({100 * occ / m['n_occurrences']:.1f}%)"
        )


if __name__ == "__main__":
    main()
