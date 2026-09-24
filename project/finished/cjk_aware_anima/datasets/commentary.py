#!/usr/bin/env python3
"""D2 — native Japanese artist commentary from Danbooru.

``../plan.md`` lists D2 (artist commentary, native JA on anime images) as
**BLOCKED**: ``danbooru.donmai.us`` is unreachable from this machine at the
network level. That note is stale. ``../../../../gelcrawl`` already solves it —
``curl_cffi`` with Chrome impersonation, through **SpoofDPI** on
``127.0.0.1:8080``, authenticated from gelcrawl's ``.env``
(``DAN_LOGIN`` / ``DAN_API_KEY``). This script reuses that route.

D2 matters because it is the only source that is native JA *at volume*: D1 is
MT output, D3/D4 are other people's domains, and D7 (LoveHina, 3,705 lines) is
held out as the register instrument and must stay that way. Risk 3 wants D2 at
>= ~20% of the mix.

**Access was never the hard part; register is.** Measured on a 2,400-record
sample, commentary splits three ways:

  1. descriptive — ``落とし物を届ける澪ちゃん``, and body text that is genuinely
     tag-adjacent (``クール黒髪おかっぱ…爆乳に弱い`` puts 黒髪 / おかっぱ / 爆乳
     into natural Japanese, which is exactly what D2 was supposed to supply);
  2. studio chatter — ``昔に描いたにこ探したらでてきた``;
  3. promotion and event boilerplate — FANBOX/patreon/pixiv links,
     ``コミケ108表紙 二日目西け-27b``, commission thanks. 16% of bulk records,
     45% among our own curated (i.e. popular) artists.

(3) is not a *loss-correctness* problem — teacher and student share the Qwen
side, so an MT'd EN of boilerplate is still a self-consistent pair — but it is a
**distribution** problem: it trains ext rows for コミケ / 西け / FANBOX that no
user ever prompts with. So promo is stripped **per line** rather than per
record, which keeps records whose description merely *ends* in a link block.

Yield measured before writing this: ja is 61% of records, >=20 cleaned chars is
29.5% of records (~59 usable lines per 200-record request), 78% unique
(multi-image posts repeat one commentary), and 5% already carry a human EN
translation — those are free non-MT pairs and are flagged ``via: human``.

zh/ko records are kept in the raw cache but not in the JA output: Phase 2 is
ja-only, while the ext table's zh/ko rows are physically present and inert, so
this is free material for whenever that scoping lifts.

Runs under **gelcrawl's** interpreter (``curl_cffi`` lives there, and this is
network-only work, no GPU) — the mirror of gelcrawl running its classifier under
this repo's venv:

    /home/sorryhyun/gelcrawl/.venv/bin/python \
        project/finished/cjk_aware_anima/datasets/commentary.py --target 100000

Every page is appended to ``assets/.commentary/raw.jsonl`` as it lands, so a
kill costs one page and a re-run resumes from the oldest id already seen.

The second half of the build is the **JA→EN pass** (``--mt``), which fills the
teacher side for the ~69.7k records that carry no human translation. That half
is GPU work under *this* repo's interpreter, so it goes through the daemon:

    make daemon-run ARGS="project/finished/cjk_aware_anima/datasets/commentary.py --mt"

Two things it does that a plain ``translate every row`` would not:

  * **Names are pinned from the Wikidata lexicon, JA→EN.** This is D5's
    argument run backwards. The pair stays self-consistent either way, but if
    黄泉 comes back "Yomi" the teacher embedding lands on an entity the model
    does not know, so the ext rows for those kanji train toward nothing.
  * **Batches are length-bucketed.** A batch runs until *every* sequence
    finishes, so mixing a 40-char record with a 900-char one pays the long one's
    budget 32 times over. 68% of records are <= 64 chars; bucketing keeps
    ``max_new_tokens`` proportional to the bucket instead of the tail.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
CACHE_DIR = ASSETS / ".commentary"
DEFAULT_OUT = ASSETS / "commentary_ja.jsonl"
DEFAULT_MT_OUT = ASSETS / "commentary_pairs.jsonl"
MT_CACHE = ASSETS / ".mtcache"
GELCRAWL = Path("/home/sorryhyun/gelcrawl")
API = "https://danbooru.donmai.us/artist_commentaries.json"

KANA = re.compile(r"[぀-ゟ゠-ヿ]")
HAN = re.compile(r"[一-鿿]")
HANGUL = re.compile(r"[가-힯]")

# DText markup, bare urls, hashtags, translator notes, ascii rules.
NOISE = re.compile(
    r'"[^"]*":\[[^\]]*\]'  # "label":[url]
    r"|\[tn\].*?\[/tn\]"  # translator note
    r"|<https?://[^>]*>"
    r"|https?://\S+"
    r"|[#＃][^\s#＃]+"
    r"|[-–—=＝*・]{3,}"
)
# Lines whose whole job is promotion / logistics. Dropped line-wise, so a
# description that merely *ends* in a link block keeps its real content.
PROMO = re.compile(
    r"FANBOX|FANTIA|fantia|fanbox|[Pp]atreon|pixiv|PIXIV|[Ss]keb|BOOTH|booth"
    r"|ファンボ|ファンティア|支援|通販|同人誌|新刊|お品書き|頒布|委託"
    r"|コミケ|コミックマーケット|コミティア|例大祭|サンクリ|エアコミケ"
    r"|お仕事|ご依頼|依頼|コミッション|リクエスト受付|募集"
    r"|フォロー|拡散|いいね|RT|リポスト|チャンネル登録"
    r"|X\s*:|Twitter|ツイッタ|ブログ|マシュマロ|ふせったー"
    r"|差分|高解像度|フルセット|続きは|限定|R18版|購入"
    # Event-booth announcements: the densest source of vocabulary no user will
    # ever prompt with (hall codes, circle names, page counts, prices).
    r"|サークル|ホール|スペース|会場|当日は|新刊|よろしくお願い|宜しくお願い"
    r"|\d+円|\d+P/|[東西南北][ぁ-んァ-ンA-Za-z\d]{0,3}[-‐−]?\d{1,3}[ab]"
)
# Leftover "◈patreon→" / "@handle |" stubs once NOISE has taken the url out.
# Must be CJK-free: `\w` matches kana and kanji under Unicode, so an unguarded
# version of this deleted ordinary short Japanese sentences (落とし物を届ける澪ちゃん
# is all word-characters) and cost ~30 points of yield before it was caught.
HANDLE_ONLY = re.compile(r"^[\s@＠/／|｜\[\]()（）:：\-–—>＞◈◆●■□▼▽↓→\w.]{0,40}$")
# Commentary is frequently one unbroken blob, so promo has to be dropped per
# *sentence* -- per line, a single 「FANBOXで公開中」 clause takes the whole
# description with it.
SEGMENT = re.compile(r"(?<=[。！？!?♪])\s*|\n+")


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------


def load_env() -> dict:
    for line in (GELCRAWL / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    login, key = os.environ.get("DAN_LOGIN"), os.environ.get("DAN_API_KEY")
    if not (login and key):
        sys.exit("DAN_LOGIN / DAN_API_KEY missing from gelcrawl/.env")
    return {"login": login, "api_key": key}


def make_session(proxy: str):
    from curl_cffi import requests as cr  # noqa: PLC0415

    s = cr.Session(impersonate="chrome")
    if proxy:
        s.proxies = {"http": proxy, "https": proxy}
    return s


def fetch_page(session, auth: dict, cursor: int | None, *, sleep: float, retries: int):
    """One 200-record page, oldest-ward. Returns ``None`` only after ``retries``.

    The SpoofDPI hop throws intermittent ``SSLError: TLS connect error`` under
    sustained use -- roughly one page in twenty. It is transient, so it is worth
    retrying rather than treating as the end of the stream, which would silently
    truncate the crawl.
    """
    params = {**auth, "limit": 200}
    if cursor is not None:
        params["page"] = f"b{cursor}"
    for attempt in range(retries):
        try:
            time.sleep(sleep)
            r = session.get(API, params=params, timeout=45)
            if r.status_code == 200:
                return r.json()
            print(f"  http {r.status_code}", file=sys.stderr, flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt}: {type(e).__name__}", file=sys.stderr, flush=True)
        time.sleep(1.0 + attempt * 2)
    return None


# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------


def lang_of(text: str) -> str:
    """Kana first: Japanese freely mixes han, but only Japanese uses kana."""
    if KANA.search(text):
        return "ja"
    if HANGUL.search(text):
        return "ko"
    if HAN.search(text):
        return "zh"
    return "other" if text.strip() else "empty"


def clean_text(raw: str) -> str:
    kept = []
    for seg in SEGMENT.split(raw.replace("\r\n", "\n")):
        if not seg:
            continue
        seg = NOISE.sub(" ", seg).strip()
        if not seg or PROMO.search(seg):
            continue
        if not (KANA.search(seg) or HAN.search(seg)) and HANDLE_ONLY.fullmatch(seg):
            continue
        kept.append(re.sub(r"\s+", " ", seg))
    return " ".join(kept).strip()


def build_record(rec: dict, min_chars: int) -> dict | None:
    title = clean_text(rec.get("original_title") or "")
    desc = clean_text(rec.get("original_description") or "")
    text = " ".join(x for x in (title, desc) if x).strip()
    if lang_of(text) != "ja" or len(text) < min_chars:
        return None
    human_en = " ".join(
        x
        for x in (
            clean_text(rec.get("translated_title") or ""),
            clean_text(rec.get("translated_description") or ""),
        )
        if x
    ).strip()
    # A "translation" that is still Japanese is a passthrough of the original
    # (hashtag/url-only commentary gets copied verbatim by translators).
    if human_en and (KANA.search(human_en) or len(human_en) < min_chars):
        human_en = ""
    return {
        "post_id": rec["post_id"],
        "title": title,
        "description": desc,
        "text": text,
        "en": human_en or None,
        "via": "human" if human_en else "needs_mt",
    }


# --------------------------------------------------------------------------
# crawl
# --------------------------------------------------------------------------


def write_records(path: Path, stats: dict, records: list[dict]) -> None:
    """JSONL: a ``{"stats": …}`` header line, then one record per line.

    These files run to 70k records; as a single JSON blob they are one
    unopenable line, and nothing downstream can stream them.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"stats": stats}, ensure_ascii=False) + "\n")
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_records(path: Path) -> tuple[dict, list[dict]]:
    """Read the JSONL layout, or a legacy one-line ``{stats, records}`` blob."""
    with path.open(encoding="utf-8") as f:
        head = json.loads(f.readline())
        if "records" in head:  # legacy single-blob asset
            return head.get("stats", {}), head["records"]
        return head.get("stats", {}), [json.loads(line) for line in f if line.strip()]


def load_cache(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:  # torn final line from a kill
                continue
    return out


def crawl(args, auth: dict) -> list[dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = CACHE_DIR / "raw.jsonl"
    raw = load_cache(raw_path)
    seen_ids = {r["id"] for r in raw}
    cursor = min(seen_ids) if seen_ids else None
    if raw:
        print(f"[resume] {len(raw)} cached records, cursor b{cursor}", file=sys.stderr)

    session = make_session(args.proxy)
    kept_text: set[str] = set()
    usable = 0
    for r in raw:
        rec = build_record(r, args.min_chars)
        if rec and rec["text"] not in kept_text:
            kept_text.add(rec["text"])
            usable += 1

    requests_made = 0
    handle = raw_path.open("a", encoding="utf-8")
    try:
        while usable < args.target and requests_made < args.max_requests:
            page = fetch_page(
                session, auth, cursor, sleep=args.sleep, retries=args.retries
            )
            requests_made += 1
            if page is None:
                print("[stop] page failed after retries", file=sys.stderr)
                break
            if not page:
                print("[stop] stream exhausted", file=sys.stderr)
                break
            fresh = 0
            for r in page:
                if r["id"] in seen_ids:
                    continue
                seen_ids.add(r["id"])
                raw.append(r)
                handle.write(json.dumps(r, ensure_ascii=False) + "\n")
                fresh += 1
                rec = build_record(r, args.min_chars)
                if rec and rec["text"] not in kept_text:
                    kept_text.add(rec["text"])
                    usable += 1
            handle.flush()
            cursor = min(r["id"] for r in page)
            if not fresh:
                print("[stop] page held nothing new", file=sys.stderr)
                break
            if requests_made % 10 == 0 or usable >= args.target:
                print(
                    f"[crawl] req {requests_made} | raw {len(raw)} | "
                    f"usable ja {usable}/{args.target} | cursor b{cursor}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        handle.close()
    print(f"[crawl] done: {requests_made} requests, {len(raw)} raw", file=sys.stderr)
    return raw


# --------------------------------------------------------------------------
# JA -> EN (the teacher side)
# --------------------------------------------------------------------------

MT_BACKGROUND = (
    "This is an artist's own comment posted with their anime/illustration "
    "artwork on pixiv or Danbooru. It may describe the picture, name the "
    "character drawn, or be casual studio chatter about drawing it. Translate "
    "it as natural English, keeping the informal tone."
)
# Records longer than this are truncated at a sentence boundary rather than
# dropped: the pair only has to be self-consistent, and the JA that is stored is
# the JA that was translated. 0.4% of records are affected.
MAX_SOURCE_CHARS = 800
# (char ceiling, max_new_tokens, batch divisor). A batch runs until every
# sequence in it finishes, so the point of bucketing is that the 68% of records
# under 64 chars never pay the tail's budget. The divisor keeps
# batch x sequence roughly constant: the 7B leaves only ~2.4 GB for the KV cache
# after its 13 GiB of weights, and batch 32 in the <=128 bucket OOMs there.
LENGTH_BUCKETS = [
    (64, 128, 1),
    (128, 224, 2),
    (256, 416, 4),
    (512, 800, 8),
    (10**9, 1536, 16),
]


def load_name_terms() -> list[tuple[str, str]]:
    """JA surface form -> EN name, from the Wikidata lexicon (D5), reversed.

    Longest-first so 栗村アイリ wins over a shorter form nested inside it.
    Latin-only JA fields (franchise ``BLEACH``) are dropped — they need no
    pinning and would match half the corpus.
    """
    path = ASSETS / "wikidata_lexicon.json"
    if not path.exists():
        print("[mt] no wikidata_lexicon.json — running without name pinning")
        return []
    lex = json.loads(path.read_text(encoding="utf-8"))
    terms: dict[str, str] = {}
    for entry in lex.get("characters", {}).values():
        ja, en = (entry.get("ja") or "").strip(), (entry.get("en") or "").strip()
        if len(ja) >= 2 and en and (KANA.search(ja) or HAN.search(ja)):
            terms.setdefault(ja, en)
    for tag, entry in lex.get("franchises", {}).items():
        ja = (entry.get("ja") or "").strip()
        if len(ja) >= 2 and (KANA.search(ja) or HAN.search(ja)):
            terms.setdefault(ja, tag.title())
    return sorted(terms.items(), key=lambda kv: -len(kv[0]))


def truncate_source(text: str, limit: int = MAX_SOURCE_CHARS) -> str:
    if len(text) <= limit:
        return text
    kept: list[str] = []
    for seg in SEGMENT.split(text):
        if not seg:
            continue
        if sum(len(s) for s in kept) + len(seg) > limit:
            break
        kept.append(seg)
    return " ".join(kept).strip() if kept else text[:limit]


def bad_translation(src: str, out: str) -> str | None:
    """Reject the failures MT makes on this corpus. Returns a reason, or None."""
    out = out.strip()
    if not out:
        return "empty"
    if KANA.search(out) or HANGUL.search(out) or HAN.search(out):
        return "untranslated_cjk"  # passthrough of the source, or a kept name
    if len(out) > 8 * len(src) + 120:
        return "runaway"  # degenerate repetition blows past the source length
    return None


def mt_pass(args) -> None:
    """Fill ``en`` for every ``needs_mt`` record; write the D2 pair file."""
    # Set before torch is imported (via mt): the KV cache grows batch-by-batch
    # against a card that is 13/15.5 GiB weights, so fragmentation is the
    # difference between fitting and not.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.path.insert(0, str(HERE))
    from mt import MTEngine, Request, build_prompt, cache_key, load_cache  # noqa: PLC0415

    _, records = read_records(args.out)
    human = [r for r in records if r["via"] == "human"]
    todo = [r for r in records if r["via"] != "human"]
    if args.mt_limit:
        todo = todo[: args.mt_limit]
    print(f"[mt] {len(todo)} to translate, {len(human)} already human-translated")

    name_terms = load_name_terms()
    pattern = (
        re.compile("|".join(re.escape(ja) for ja, _ in name_terms))
        if name_terms
        else None
    )
    en_of = dict(name_terms)

    reqs, srcs = [], []
    for rec in todo:
        src = truncate_source(rec["text"])
        srcs.append(src)
        hits = list(dict.fromkeys(pattern.findall(src))) if pattern else []
        reqs.append(
            Request(
                text=src,
                terms=[(ja, en_of[ja]) for ja in hits[: args.max_terms]],
                background=MT_BACKGROUND,
                meta={"i": len(reqs)},
            )
        )
    pinned = sum(1 for r in reqs if r.terms)
    print(
        f"[mt] {pinned} records carry a pinned name ({100 * pinned / max(1, len(reqs)):.1f}%)"
    )

    # `--from-cache` harvests whatever a stopped pass already wrote, on CPU: the
    # cache key is a pure function of (model, decode mode, budget, prompt), so
    # the model never has to load to answer "was this row translated?".
    engine = (
        None
        if args.from_cache
        else MTEngine(args.model, greedy=True, gpu_budget=args.gpu_budget)
    )
    out_text: list[str] = [""] * len(reqs)
    order = sorted(range(len(reqs)), key=lambda i: len(srcs[i]))
    start = 0
    for ceiling, budget, divisor in LENGTH_BUCKETS:
        idx = [i for i in order[start:] if len(srcs[i]) <= ceiling]
        start += len(idx)
        if not idx:
            continue
        cache = args.mt_cache / f"commentary_ja2en_{ceiling}.jsonl"
        if engine is None:
            done = load_cache(cache)
            hits = 0
            for i in idx:
                key = cache_key(
                    args.model,
                    build_prompt(reqs[i], "en"),
                    target_lang="en",
                    max_new_tokens=budget,
                    greedy=True,
                )
                if key in done:
                    out_text[i] = done[key]
                    hits += 1
            print(f"[mt] bucket <={ceiling} chars: {hits}/{len(idx)} rows from cache")
            continue
        batch = max(1, args.batch_size // divisor)
        print(
            f"[mt] bucket <={ceiling} chars: {len(idx)} rows, "
            f"{budget} new tokens, batch {batch}"
        )
        got = engine.translate(
            [reqs[i] for i in idx],
            target_lang="en",
            batch_size=batch,
            progress_every=args.progress_every,
            max_new_tokens=budget,
            cache=cache,
        )
        for i, text in zip(idx, got):
            out_text[i] = text

    pairs = [
        {"post_id": r["post_id"], "ja": r["text"], "en": r["en"], "via": "human"}
        for r in human
    ]
    drops = Counter()
    for rec, src, got in zip(todo, srcs, out_text):
        # A stopped pass leaves untranslated rows empty — that is "not reached",
        # not "the model returned nothing", and the two must not share a counter.
        if not got.strip() and args.from_cache:
            drops["not_translated"] += 1
            continue
        reason = bad_translation(src, got)
        if reason:
            drops[reason] += 1
            continue
        pairs.append(
            {"post_id": rec["post_id"], "ja": src, "en": got.strip(), "via": "mt"}
        )

    ja_chars = "".join(p["ja"] for p in pairs)
    stats = {
        "source": "D2 danbooru artist commentary, JA->EN",
        "model": args.model,
        "decode": "greedy",
        "candidates": len(human) + len(todo),
        "pairs": len(pairs),
        "via_human": len(human),
        "via_mt": sum(1 for p in pairs if p["via"] == "mt"),
        "mt_rejected": dict(drops),
        "mt_reject_rate": round(sum(drops.values()) / max(1, len(todo)), 4),
        "names_pinned": pinned,
        "truncated_sources": sum(
            1 for r, s in zip(todo, srcs) if len(s) < len(r["text"])
        ),
        "mean_ja_chars": round(len(ja_chars) / max(1, len(pairs)), 1),
        "unique_kanji": len(set(HAN.findall(ja_chars))),
    }
    write_records(args.mt_out, stats, pairs)
    for k, v in stats.items():
        print(f"{k}: {v}")
    print("\n=== spot check ===")
    for p in pairs[len(human) : len(human) + 12]:
        print(f"  JA: {p['ja'][:110]}\n  EN: {p['en'][:160]}\n")
    print(f"wrote {args.mt_out}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=100_000, help="usable JA records")
    ap.add_argument("--max-requests", type=int, default=4000, help="runaway guard")
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--sleep", type=float, default=0.5, help="seconds between requests")
    ap.add_argument("--retries", type=int, default=8)
    ap.add_argument("--proxy", default="http://127.0.0.1:8080")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-crawl", action="store_true", help="rebuild from cache only")
    mt = ap.add_argument_group("JA->EN pass (GPU, run through the daemon)")
    mt.add_argument("--mt", action="store_true", help="translate the JA side to EN")
    mt.add_argument("--mt-out", type=Path, default=DEFAULT_MT_OUT)
    mt.add_argument("--mt-cache", type=Path, default=MT_CACHE)
    mt.add_argument("--mt-limit", type=int, default=0, help="first N records only")
    mt.add_argument(
        "--from-cache",
        action="store_true",
        help="build from translations already cached — no model, no GPU",
    )
    mt.add_argument("--model", default="tencent/Hy-MT2-1.8B")
    mt.add_argument("--batch-size", type=int, default=32)
    mt.add_argument("--max-terms", type=int, default=6, help="pinned names per record")
    mt.add_argument("--gpu-budget", default=None, help="e.g. 13GiB — fits the 7B")
    mt.add_argument("--progress-every", type=int, default=20)
    args = ap.parse_args()

    if args.mt:
        mt_pass(args)
        return

    auth = load_env()
    raw = load_cache(CACHE_DIR / "raw.jsonl") if args.no_crawl else crawl(args, auth)

    langs, drops = Counter(), Counter()
    records, seen_text = [], set()
    for r in raw:
        body = f"{r.get('original_title') or ''}\n{r.get('original_description') or ''}"
        langs[lang_of(body)] += 1
        rec = build_record(r, args.min_chars)
        if rec is None:
            drops["not_ja_or_too_short_after_cleaning"] += 1
            continue
        if rec["text"] in seen_text:
            drops["duplicate_text"] += 1
            continue
        seen_text.add(rec["text"])
        records.append(rec)

    chars = "".join(r["text"] for r in records)
    stats = {
        "source": "danbooru artist_commentaries.json (via gelcrawl route)",
        "raw_records": len(raw),
        "language_split": dict(langs.most_common()),
        "dropped": dict(drops),
        "records": len(records),
        "with_human_en": sum(1 for r in records if r["via"] == "human"),
        "total_chars": len(chars),
        "mean_chars": round(len(chars) / max(1, len(records)), 1),
        "unique_kanji": len(set(HAN.findall(chars))),
        "unique_kana": len(set(KANA.findall(chars))),
        "min_chars": args.min_chars,
    }
    write_records(args.out, stats, records)
    for k, v in stats.items():
        print(f"{k}: {v}", file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
