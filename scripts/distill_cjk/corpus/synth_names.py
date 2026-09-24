"""Synthetic name register: mint span visits for JA character names (2c item b).

The measured block (2026-08-27, `cjk_vocab_pack_names`): the `names` register
pins only names that already occur in our captions, so a kanji name absent from
the pool never accumulates visits — `博麗霊夢` appears 3× in 60k pairs and
`博`:19 / `麗`:44 sit far under the ~300-visit render floor. Katakana names
(初音ミク, アスカ) render because their captions are plentiful; nothing about
the script differs. So the lever is visits, and visits are text.

This composes them without images or crawling: take a real caption from the
retrieved pool that carries a character segment, substitute the target
character (EN side: its danbooru tag; JA side: one of its Japanese names) and
its copyright, keep every other segment EN exactly as `build_names` does — the
teacher context is identical outside the name span, so the supervision on the
name rows is exact. Both sides mean the same thing, which is all a pair needs.

Name source: the danbooru wiki snapshot `tag_glossary.py` already reads
(`other_names`, `post_count`, `category_name`), which carries full names the
tag-pair set lacks (`博麗霊夢` vs `霊夢`). Guards: the glossary's Chinese guards
(`tag_pairs.japanese_names`), plus a family filter — a wording that shares no
character with the longest surviving name is a pairing/meme alias
(`レイマリ` on hakurei_reimu) and is dropped.

    python -m scripts.distill_cjk.corpus.synth_names --dry-run
    python -m scripts.distill_cjk.corpus.synth_names            # writes
        post_image_dataset/cjk_distill/names_synth.jsonl and the merged
        post_image_dataset/cjk_distill/pairs_synth.jsonl (pairs.jsonl + synth)
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from scripts.distill_cjk.corpus import build_pairs, tag_glossary, tag_pairs  # noqa: E402

DEFAULT_WIKI = tag_glossary.ASSETS / ".wiki" / tag_glossary.WIKI_FILE
DEFAULT_CAPTIONS = Path.home() / "gelcrawl" / "retrieved"
DEFAULT_PAIRS = REPO / "post_image_dataset" / "cjk_distill" / "pairs.jsonl"
DEFAULT_OUT = REPO / "post_image_dataset" / "cjk_distill"
REGISTER = "names_synth"
PAREN = re.compile(r"^(.*?) \(([^()]+)\)$")
LINK = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")


def load_wiki(path: Path) -> dict[str, dict]:
    """tag title (spaced) → {names, count, cat} for character/copyright tags."""
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("is_deleted"):
                continue
            cat = (r.get("category_name") or "").lower()
            if cat not in ("character", "copyright"):
                continue
            title = (r.get("title") or "").replace("_", " ").strip().lower()
            if title:
                out[title] = {
                    "names": list(r.get("other_names") or []),
                    "count": int(r.get("post_count") or 0),
                    "cat": cat,
                    # wiki-link candidates for the copyright of a character
                    # outside our pool (``main character of the [[Touhou]]``)
                    "links": [
                        m.lower().replace("_", " ").split("|")[0].strip()
                        for m in LINK.findall(r.get("body") or "")[:8]
                    ],
                }
    return out


DIGIT = re.compile(r"\d")


def _guard(names: list[str], inventory: set[str]) -> list[str]:
    return [
        n for n in tag_pairs.japanese_names(names, inventory) if not DIGIT.search(n)
    ]


def name_family(
    wiki_names: list[str],
    tp_names: list[str],
    inventory: set[str],
    *,
    expand: bool = True,
) -> list[str]:
    """JA wordings for one tag, primary first.

    The wiki ``other_names`` field mixes real names with event/meme tags
    (``初音ミク誕生祭2025``, ``るしあ大好きだよ``, ``レイマリ``). Measured rule
    (2026-08-27 probe over the top pool tags): a wording present in **both**
    the 2026 wiki and the 2024 tag-pair snapshot is canonical almost without
    exception. Around that core, ``expand`` admits wiki-only wordings that are
    a kanji-only superset of a canonical one (``博麗霊夢`` ⊃ ``霊夢`` — the
    full name) or a substring of one (``魔理沙`` ⊂ ``霧雨魔理沙`` — the short
    form); kana-adding supersets (``初音ミクイラスト``) stay out. Primary =
    longest full name, so the student sees the form the eval prompts use.
    """
    w = _guard(wiki_names, inventory)
    t = _guard(tp_names, inventory)
    core = [n for n in w if n in t]
    if not core:
        core = t[:1] or ([min(w, key=len)] if w else [])
    if not core:
        return []
    anchor = min(core, key=len)
    core = [n for n in core if set(n) & set(anchor)]
    if not expand:
        return core
    full = [
        n
        for n in w
        if n not in core
        and any(
            c in n and all(tag_glossary.HAN.match(x) for x in n.replace(c, ""))
            for c in core
        )
    ]
    short = [
        n for n in w if n not in core and len(n) >= 2 and any(n in c for c in core)
    ]
    fam = sorted(full, key=len, reverse=True) + core + short
    out: list[str] = []
    for n in fam:
        if n not in out:
            out.append(n)
    return out


def ko_family(names: list[str]) -> list[str]:
    """KO name family: hangul wordings, longest full name primary.

    Sources are ordered canonical-first by the caller (lexicon ko label →
    KR-KB keywords → wiki ``other_names``); the overlap filter drops
    franchise/meme aliases that share no syllable with the full name
    (``동방`` on hakurei_reimu), the KO analog of the JA anchor rule.
    """
    cands: list[str] = []
    for n in names:
        if tag_glossary.is_korean(n) and not DIGIT.search(n) and n not in cands:
            cands.append(n)
    if not cands:
        return []
    # Primary = the first candidate (sources are canonical-first), stretched to
    # a same-name superset if one exists (미쿠 → 하츠네 미쿠). max-by-length is
    # wrong here: on souryuu_asuka_langley it promotes the KB alias
    # 시키나미 아스카 랑그레이 — a different continuity's identity.
    primary = max(
        (n for n in cands if cands[0] in n.replace(" ", "") or cands[0] in n),
        key=len,
        default=cands[0],
    )
    return [primary] + [n for n in cands if n != primary and set(n) & set(primary)]


def load_captions(root: Path) -> list[tuple[str, list[str]]]:
    out = []
    for p in sorted(root.rglob("*.txt")):
        text = html.unescape(p.read_text(encoding="utf-8", errors="replace"))
        segs = [s.strip() for s in text.split(",") if s.strip()]
        if segs:
            out.append((str(p.relative_to(root).with_suffix("")), segs))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--lang",
        default="ja",
        choices=["ja", "ko"],
        help="student-side language: ko mints names_synth_ko from the KO name "
        "sources (lexicon ko labels → KR KB keywords → wiki other_names), "
        "allocates by hangul-syllable visits, joins rng-free with ', '",
    )
    ap.add_argument("--wiki", type=Path, default=DEFAULT_WIKI)
    ap.add_argument("--captions", type=Path, default=DEFAULT_CAPTIONS)
    ap.add_argument(
        "--pairs", type=Path, default=DEFAULT_PAIRS, help="base corpus to merge with"
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--min-posts",
        type=int,
        default=3000,
        help="wiki post_count floor for names outside our pool",
    )
    ap.add_argument("--max-names", type=int, default=3000)
    ap.add_argument(
        "--per-name",
        type=int,
        default=24,
        help="minimum synthetic captions per character",
    )
    ap.add_argument(
        "--max-per-name",
        type=int,
        default=200,
        help="cap for names carrying rare kanji",
    )
    ap.add_argument(
        "--floor",
        type=int,
        default=300,
        help="target visits per kanji in a primary name (the measured render floor)",
    )
    ap.add_argument(
        "--p-primary", type=float, default=0.7, help="probability of the full name"
    )
    ap.add_argument(
        "--extra",
        nargs="*",
        default=["hakurei reimu", "souryuu asuka langley", "hatsune miku"],
        help="always include",
    )
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="restrict targets to these tags (plan_ko3 M1 densification: "
        "top up one minted name to a pair floor without regenerating the "
        "whole register — point --out at a scratch dir so the standing "
        "names_synth/pairs_synth files are not clobbered)",
    )
    ap.add_argument(
        "--context",
        choices=["en", "ja", "both"],
        default="ja",
        help="language of the non-name segments: en = build_names-style exact EN "
        "context (register names_synth); ja = composed through the glossary like "
        "the tags register (register names_synth_ja) — measured 2026-08-28: rows "
        "trained only between EN neighbours do not carry the name in a full-JA "
        "prompt (r1 partial, n1/r3 fail), so ja is the default",
    )
    ap.add_argument("--glossary", type=Path, default=tag_glossary.DEFAULT_OUT)
    ap.add_argument(
        "--en-per-name",
        type=int,
        default=8,
        help="with --context both: fixed EN-context captions per name on top of "
        "the rarity-weighted JA allocation",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.lang != "ja":
        if args.pairs == DEFAULT_PAIRS:
            args.pairs = DEFAULT_OUT / f"pairs_{args.lang}.jsonl"
        if args.glossary == tag_glossary.DEFAULT_OUT:
            args.glossary = tag_glossary.ASSETS / f"tag_glossary_{args.lang}.json"
    suffix = "" if args.lang == "ja" else f"_{args.lang}"
    rng = random.Random(args.seed)

    wiki = load_wiki(args.wiki)
    tp = tag_pairs.load_pairs() if args.lang == "ja" else {}
    inventory = (
        tag_glossary.ja_kanji_inventory(args.wiki) if args.lang == "ja" else set()
    )
    kr_kb: dict[str, list[str]] = {}
    lex_chars: dict = {}
    lex_frs: dict = {}
    if args.lang == "ko":
        kr_kb = tag_glossary.load_kr_kb(REPO / "models" / "danbooru_tags_classified.csv")
        lex = json.loads(
            (tag_glossary.ASSETS / "wikidata_lexicon.json").read_text(encoding="utf-8")
        )
        lex_chars, lex_frs = lex["characters"], lex["franchises"]

    def is_row_char(c: str) -> bool:
        """chars whose ext-row visits the allocation tracks (kanji / hangul)."""
        if args.lang == "ja":
            return bool(tag_glossary.HAN.match(c))
        return "가" <= c <= "힣"

    def family(tag: str, names: list[str], *, expand: bool = True) -> list[str]:
        if args.lang == "ja":
            return name_family(names, tp.get(tag, []), inventory, expand=expand)
        lex_e = lex_chars.get(tag) or lex_frs.get(tag) or {}
        pool = ([lex_e["ko"]] if lex_e.get("ko") else []) + kr_kb.get(tag, []) + names
        return ko_family(pool)

    print(
        f"wiki: {len(wiki)} character/copyright tags; inventory {len(inventory)} kanji"
    )

    captions = load_captions(args.captions)
    char_occ: collections.Counter = collections.Counter()
    co_cp: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    templates = []
    for _, segs in captions:
        chars = [s for s in segs if wiki.get(s, {}).get("cat") == "character"]
        cps = [s for s in segs if wiki.get(s, {}).get("cat") == "copyright"]
        for c in chars:
            char_occ[c] += 1
            for cp in cps:
                co_cp[c][cp] += 1
        if chars:
            templates.append(segs)
    print(
        f"captions {len(captions)}, templates with a character {len(templates)}, chars in pool {len(char_occ)}"
    )

    # Target set: pool characters ∪ popular wiki characters ∪ --extra, JA-resolvable only.
    cand = set(char_occ) | set(args.extra)
    cand |= {
        t
        for t, r in wiki.items()
        if r["cat"] == "character" and r["count"] >= args.min_posts
    }
    if args.only:
        missing = set(args.only) - cand
        if missing:
            raise SystemExit(f"--only tags not in the candidate set: {missing}")
        cand &= set(args.only)
    targets = []
    for t in sorted(cand, key=lambda t: -wiki.get(t, {}).get("count", 0)):
        r = wiki.get(t)
        if not r or r["cat"] != "character":
            continue
        fam = family(t, r["names"])
        if not fam:
            continue
        cp = None
        if co_cp.get(t):
            cp = co_cp[t].most_common(1)[0][0]
        else:
            m = PAREN.match(t)
            if m and wiki.get(m.group(2), {}).get("cat") == "copyright":
                cp = m.group(2)
            else:
                cp = next(
                    (
                        link
                        for link in r["links"]
                        if wiki.get(link, {}).get("cat") == "copyright"
                    ),
                    None,
                )
        cp_ja = family(cp, wiki[cp]["names"], expand=False) if cp else []
        targets.append((t, fam, cp, cp_ja))
        if len(targets) >= args.max_names:
            break
    n_native = sum(
        1 for _, fam, _, _ in targets if tag_glossary.NATIVE_RE[args.lang].search(fam[0])
    )
    print(f"targets {len(targets)} ({n_native} with native script in the primary name)")
    for t, fam, cp, cp_ja in targets[:5] + [x for x in targets if x[0] in args.extra]:
        print(f"   {t:32s} {fam} | {cp} → {cp_ja[:1]}")

    # Rarity-weighted allocation: a kanji that occurs in one name only needs
    # that name to carry it to the floor; a shared one is bought by many. Seed
    # the running counts with the base corpus's span text (the tags/names
    # registers are what already visit these rows), then greedily size each
    # name so its rarest primary-name kanji reaches --floor.
    visits: collections.Counter = collections.Counter()
    with args.pairs.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("register") in (
                "tags" + suffix,
                "tags_alt" + suffix,
                "names" + suffix,
            ):
                for sp in rec.get("spans") or []:
                    if sp.get("via") != "en_pinned":
                        visits.update(c for c in sp["ja"] if is_row_char(c))
    alloc: dict[str, int] = {}
    order = sorted(
        targets,
        key=lambda x: min(
            (visits[c] for c in x[1][0] if is_row_char(c)), default=10**9
        ),
    )
    for t, fam, _, _ in order:
        kanji = [c for c in fam[0] if is_row_char(c)]
        need = max((args.floor - visits[c] for c in kanji), default=0)
        n = max(args.per_name, min(args.max_per_name, int(need / args.p_primary) + 1))
        alloc[t] = n
        for c in kanji:
            visits[c] += int(n * args.p_primary)
    n_total = sum(alloc.values())
    print(
        f"allocation: {n_total} pairs over {len(alloc)} names "
        f"(min {min(alloc.values())}, max {max(alloc.values())}, "
        f"at cap {sum(1 for v in alloc.values() if v >= args.max_per_name)})"
    )

    glossary = json.loads(args.glossary.read_text(encoding="utf-8"))["tags"]

    def mint(t, fam, cp, cp_ja, k, context):
        segs = rng.choice(templates)
        en_out, ja_out, spans, placed = [], [], [], False
        pinned: dict[int, tuple[str, str, str, float]] = {}
        for s in segs:
            cat = wiki.get(s, {}).get("cat")
            if cat == "character":
                if placed:
                    continue
                placed = True
                ja = (
                    fam[0]
                    if (len(fam) == 1 or rng.random() < args.p_primary)
                    else rng.choice(fam[1:])
                )
                pinned[len(en_out)] = (t, ja, "wiki", 0.0)
                en_out.append(t)
            elif cat == "copyright":
                if cp is None or cp in en_out:
                    continue
                if cp_ja:
                    pinned[len(en_out)] = (cp, rng.choice(cp_ja), "wiki", 0.0)
                else:
                    pinned[len(en_out)] = (cp, cp, "en_pinned", 1.0)
                en_out.append(cp)
            else:
                en_out.append(s)
        if context == "ja":
            ja_out, _missing, spans = build_pairs.compose(
                en_out, glossary, alt=False, rng=rng, min_f1=0.5
            )
            for i, (en, ja, via, f1) in pinned.items():
                ja_out[i] = ja
                spans[i] = {"en": en, "ja": ja, "via": via, "f1": f1}
        else:
            for i, en in enumerate(en_out):
                if i in pinned:
                    en, ja, via, f1 = pinned[i]
                else:
                    ja, via, f1 = en, "en_pinned", 1.0
                ja_out.append(ja)
                spans.append({"en": en, "ja": ja, "via": via, "f1": f1})
        if args.lang == "ja":
            reg = REGISTER + ("_ja" if context == "ja" else "")
        else:
            reg = REGISTER + suffix + ("" if context == "ja" else "_en")
        joiner = build_pairs.pick_joiner(rng if args.lang == "ja" else None)
        return {
            "id": f"SYN/{t}/{k}/{reg}",
            "source": "SYN",
            "register": reg,
            "lang": args.lang,
            "en": ", ".join(en_out),
            "ja": joiner.join(ja_out),
            "joiner": joiner,
            "n_missing": 0,
            "spans": spans,
        }

    pairs = []
    for t, fam, cp, cp_ja in targets:
        main_ctx = "en" if args.context == "en" else "ja"
        for k in range(alloc[t]):
            pairs.append(mint(t, fam, cp, cp_ja, k, main_ctx))
        if args.context == "both":
            for k in range(args.en_per_name):
                pairs.append(mint(t, fam, cp, cp_ja, k, "en"))
    print(f"synthetic pairs: {len(pairs)}")
    if args.dry_run:
        for p in rng.sample(pairs, 3):
            print("  EN:", p["en"][:140])
            print("  JA:", p["ja"][:140])
        return

    args.out.mkdir(parents=True, exist_ok=True)
    synth = args.out / f"names_synth{suffix}.jsonl"
    with synth.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    merged = args.out / f"pairs_synth{suffix}.jsonl"
    with merged.open("w", encoding="utf-8") as f:
        f.write(args.pairs.read_text(encoding="utf-8"))
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {synth} and {merged}")


if __name__ == "__main__":
    main()
